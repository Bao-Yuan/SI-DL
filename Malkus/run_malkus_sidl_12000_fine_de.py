from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


OUTPUT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = OUTPUT_DIR / "run_malkus_midl_sidl.py"

SAMPLE_SIZE = 12_000
DE_MAXITER = 50
DE_POPSIZE = 6
DE_TOL = 1e-5
DE_INIT_JITTER_SCALE = 0.06
DE_BOUNDS = (-2.0, 2.0)

OUTPUT_PREFIX = "malkus_sidl_12000_fine_de"
SUMMARY_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"
EXPONENTS_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_exponents.csv"
LOG_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_de_log.csv"


def load_base_module():
    spec = importlib.util.spec_from_file_location("malkus_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


malkus = load_base_module()


def score_sidl(feature: np.ndarray, y: np.ndarray) -> dict[str, float]:
    score = malkus.SI_DL.explained_variance_score(
        feature,
        y,
        bandwidth=malkus.SI_BANDWIDTH,
        boundary="mirror",
    )
    m_hat = np.asarray(score["m_hat"], dtype=float).reshape(-1)
    residual = y - m_hat
    return {
        "S_cov": float(score["S_cov"]),
        "S_cov_raw": float(score["S_cov_raw"]),
        "sidl_error_clipped": float(1.0 - score["S_cov"]),
        "sidl_error_raw": float(1.0 - score["S_cov_raw"]),
        "bandwidth": float(score["bandwidth"]),
        "n_retained": int(score["n_retained"]),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "nrmse": float(np.sqrt(np.mean(residual**2)) / np.std(y, ddof=0)),
        "corr_Y_mhat": float(np.corrcoef(y, m_hat)[0, 1]),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "median_abs_residual": float(np.median(np.abs(residual))),
    }


def run_fine_de() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_raw, y_raw, source_data = malkus.load_malkus_data()
    x, y = malkus.filter_valid_rows(x_raw, y_raw)
    idx = malkus.sample_idx(x.shape[0], SAMPLE_SIZE, 11)
    x_eval = x[idx]
    y_eval = y[idx]
    itpi_omegas = malkus.load_itpi_continuous_omegas(source_data)
    basis = malkus.itpi_malkus_basis()
    log_rows: list[dict[str, float | int | str]] = []
    start = time.time()
    evaluation_count = 0
    best_seen = -np.inf

    def objective(params: np.ndarray) -> float:
        nonlocal evaluation_count, best_seen
        evaluation_count += 1
        try:
            omegas = malkus.params_to_omegas(params, basis, canonical_sign=False)
            if np.linalg.matrix_rank(omegas, tol=1e-8) < 2:
                return 1e6
            log_feature = np.log(x_eval) @ omegas.T
            if not np.all(np.isfinite(log_feature)):
                return 1e6
            log_std = np.std(log_feature, axis=0)
            if np.min(log_std) < 0.05 or np.max(log_std) > 8.0:
                return 1e6
            if abs(float(np.corrcoef(log_feature.T)[0, 1])) > 0.995:
                return 1e6
            if np.max(np.abs(log_feature)) > 40.0:
                return 1e6
            feature = malkus.pi_from_omegas(x_eval, omegas)
            value = score_sidl(feature, y_eval)["S_cov"]
        except Exception:
            return 1e6

        if np.isfinite(value) and value > best_seen:
            best_seen = float(value)
        return -float(value)

    def callback(*args, **kwargs) -> bool:
        elapsed = time.time() - start
        intermediate_result = kwargs.get("intermediate_result")
        if intermediate_result is None and args and hasattr(args[0], "fun"):
            intermediate_result = args[0]
        best_objective = float(intermediate_result.fun) if intermediate_result is not None else np.nan
        best_score = -best_objective if np.isfinite(best_objective) else best_seen
        log_rows.append(
            {
                "iteration": len(log_rows) + 1,
                "elapsed_seconds": float(elapsed),
                "objective_evaluations": int(evaluation_count),
                "best_S_cov": float(best_score),
            }
        )
        pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False)
        print(
            f"Fine SI-DL DE iter {len(log_rows):03d}: "
            f"best S_cov={best_score:.8f}, evals={evaluation_count}, elapsed={elapsed/60:.2f} min",
            flush=True,
        )
        return False

    init = malkus.initial_population_from_omegas(
        itpi_omegas,
        basis,
        malkus.RANDOM_STATE + 11,
        jitter_scale=DE_INIT_JITTER_SCALE,
        popsize=DE_POPSIZE,
        canonical_sign=False,
    )

    result = differential_evolution(
        objective,
        bounds=[DE_BOUNDS] * (2 * basis.shape[1]),
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=DE_TOL,
        seed=malkus.RANDOM_STATE + 11,
        init=init,
        mutation=(0.45, 1.0),
        recombination=0.9,
        polish=False,
        updating="immediate",
        workers=1,
        callback=callback,
    )

    sidl_omegas = malkus.params_to_omegas(result.x, basis, canonical_sign=False)
    methods = {
        "SI-DL 2D fine DE": sidl_omegas,
        "IT-PI continuous": itpi_omegas,
    }

    rows = []
    exponent_rows = []
    for method, omegas in methods.items():
        feature = malkus.pi_from_omegas(x_eval, omegas)
        sidl = score_sidl(feature, y_eval)
        mi = malkus.information_metrics(feature, y_eval)
        rows.append(
            {
                "method": method,
                "n_single_set_samples": int(idx.size),
                "selection_set": "same_12000_points",
                "selection_metric": "max_S_cov_clipped",
                "boundary": "mirror",
                "S_cov": sidl["S_cov"],
                "S_cov_raw": sidl["S_cov_raw"],
                "sidl_error_clipped": sidl["sidl_error_clipped"],
                "sidl_error_raw": sidl["sidl_error_raw"],
                "sidl_bandwidth": sidl["bandwidth"],
                "sidl_n_retained": sidl["n_retained"],
                "rmse": sidl["rmse"],
                "nrmse": sidl["nrmse"],
                "corr_Y_mhat": sidl["corr_Y_mhat"],
                "max_abs_residual": sidl["max_abs_residual"],
                "median_abs_residual": sidl["median_abs_residual"],
                "mutual_information": mi["mutual_information"],
                "epsilon_lb_normalized": mi["epsilon_lb_normalized"],
                "optimizer_fun": float(result.fun) if method == "SI-DL 2D fine DE" else np.nan,
                "optimizer_success": bool(result.success) if method == "SI-DL 2D fine DE" else np.nan,
                "optimizer_message": str(result.message) if method == "SI-DL 2D fine DE" else "",
                "objective_evaluations": int(evaluation_count) if method == "SI-DL 2D fine DE" else np.nan,
                "elapsed_seconds": float(time.time() - start) if method == "SI-DL 2D fine DE" else np.nan,
                "de_maxiter": int(DE_MAXITER) if method == "SI-DL 2D fine DE" else np.nan,
                "de_popsize": int(DE_POPSIZE) if method == "SI-DL 2D fine DE" else np.nan,
                "de_tol": float(DE_TOL) if method == "SI-DL 2D fine DE" else np.nan,
                "formula": malkus.formula_from_exponents(omegas),
            }
        )

        for pi_idx, row in enumerate(omegas, start=1):
            for label, exponent in zip(malkus.VARIABLE_LABELS, row):
                exponent_rows.append(
                    {
                        "method": method,
                        "pi_group": f"pi{pi_idx}",
                        "variable": label,
                        "normalized_exponent": float(exponent),
                    }
                )

    summary = pd.DataFrame(rows)
    exponents = pd.DataFrame(exponent_rows)
    log = pd.DataFrame(log_rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    exponents.to_csv(EXPONENTS_CSV, index=False)
    log.to_csv(LOG_CSV, index=False)
    return summary, exponents, log


if __name__ == "__main__":
    summary, _, _ = run_fine_de()
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {EXPONENTS_CSV}")
    print(f"Wrote {LOG_CSV}")
    print()
    print(
        summary[
            [
                "method",
                "n_single_set_samples",
                "S_cov",
                "S_cov_raw",
                "nrmse",
                "corr_Y_mhat",
                "sidl_bandwidth",
                "mutual_information",
                "optimizer_success",
            ]
        ].to_string(index=False)
    )
