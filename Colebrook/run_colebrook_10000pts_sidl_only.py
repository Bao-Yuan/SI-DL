from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR.parent
SI_DIR = ROOT / "SI-DL-main"
if str(SI_DIR) not in sys.path:
    sys.path.insert(0, str(SI_DIR))

import SI_DL


FIG_DIR = OUTPUT_DIR / "figures"

N_SAMPLES = int(os.environ.get("COLEBROOK_N_SAMPLES", "10000"))
OUTPUT_PREFIX = f"colebrook_{N_SAMPLES}pts_sidl_only"
RANDOM_STATE = 42
SI_BANDWIDTH = 0.06
DE_BOUNDS = (-4.0, 4.0)
DE_MAXITER = int(os.environ.get("COLEBROOK_DE_MAXITER", "80"))
DE_POPSIZE = int(os.environ.get("COLEBROOK_DE_POPSIZE", "20"))
DE_TOL = float(os.environ.get("COLEBROOK_DE_TOL", "1e-7"))
DE_POLISH = os.environ.get("COLEBROOK_DE_POLISH", "1") != "0"
DE_MUTATION = (0.5, 1.9)
DE_RECOMBINATION = 0.95
DE_SEEDS = [0, 42, 4209]
VARIABLE_LABELS = ["u", "rho", "D", "k_s", "mu"]
D_IN = np.matrix("1 -3 1 1 -1; -1 0 0 0 -1; 0 1 0 0 1")


def normalize_exponents(exponents: np.ndarray) -> np.ndarray:
    row = np.asarray(exponents, dtype=float).reshape(-1)
    scale = float(np.max(np.abs(row)))
    if scale <= 1e-12:
        return row
    out = row / scale
    first = np.flatnonzero(np.abs(out) > 1e-10)
    if first.size and out[first[0]] < 0.0:
        out = -out
    return out


def formula_from_exponents(omegas: np.ndarray, decimals: int = 4) -> str:
    lines = []
    for idx, row in enumerate(np.asarray(omegas, dtype=float), start=1):
        terms = []
        for label, value in zip(VARIABLE_LABELS, row):
            value = float(value)
            if abs(value) < 5e-5:
                continue
            terms.append(f"{label}^{value:.{decimals}f}")
        lines.append(f"pi{idx}: " + " * ".join(terms))
    return "\n".join(lines)


def colebrook(reynolds_number: np.ndarray, relative_roughness: np.ndarray) -> np.ndarray:
    re = np.asarray(reynolds_number, dtype=float)
    rr = np.asarray(relative_roughness, dtype=float)
    f = np.full_like(re, 0.02, dtype=float)
    for _ in range(100):
        f_new = 1.0 / (-np.log10(rr / 3.7 + 5.02 / (re * np.sqrt(f))))
        if np.max(np.abs(f_new - f)) < 1e-6:
            f = f_new
            break
        f = f_new
    return f


def generate_colebrook_data(n_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 10)
    log_re = rng.uniform(3.0, 5.0, n_samples)
    log_relative_roughness = rng.uniform(-8.0, -0.7, n_samples)
    re = 10.0**log_re
    relative_roughness = 10.0**log_relative_roughness
    rho = np.full(n_samples, 200.0)
    mu = np.full(n_samples, 0.001)
    diameter = rng.uniform(1.0, 10.0, n_samples)
    roughness = relative_roughness * diameter
    velocity = re * mu / (rho * diameter)
    cf = colebrook(re, relative_roughness)
    return pd.DataFrame(
        {
            "u": velocity,
            "rho": rho,
            "D": diameter,
            "k_s": roughness,
            "mu": mu,
            "Re": re,
            "relative_roughness": relative_roughness,
            "Cf": cf,
        }
    )


def params_to_omegas(params: np.ndarray, basis: np.ndarray, num_input: int) -> np.ndarray:
    params = np.asarray(params, dtype=float).reshape(num_input, basis.shape[0])
    return np.asarray([normalize_exponents(row @ basis) for row in params])


def log_feature_from_omegas(X: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(X, dtype=float)) @ np.asarray(omegas, dtype=float).T


def evaluate_feature(feature: np.ndarray, y: np.ndarray) -> dict[str, float | int | str]:
    score = SI_DL.explained_variance_score(
        feature,
        y,
        bandwidth=SI_BANDWIDTH,
        estimator="gaussian_kernel",
        standardize=True,
        leave_one_out=True,
        boundary="mirror",
    )
    m_hat = np.asarray(score["m_hat"], dtype=float).reshape(-1)
    residual = np.asarray(y, dtype=float).reshape(-1) - m_hat
    var_y = float(np.var(y, ddof=0))
    mse = float(np.mean(residual**2))
    score.update(
        {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "nrmse_var": float(np.sqrt(mse / var_y)),
            "corr_Y_mhat": float(np.corrcoef(y, m_hat)[0, 1]),
        }
    )
    return score


def params_for_known_groups(basis: np.ndarray) -> dict[str, np.ndarray]:
    known_re = np.array([1.0, 1.0, 1.0, 0.0, -1.0])
    known_rr = np.array([0.0, 0.0, -1.0, 1.0, 0.0])
    previous_sidl_1 = np.array([1.0, 1.0, 0.23423725086294048, 0.7657627491370596, -1.0])
    previous_sidl_2 = np.array([0.6037203926408655, 0.6037203926408655, 1.0, -0.3962796073591344, -0.6037203926408655])
    rows = {
        "Known Re": known_re,
        "Known k_s/D": known_rr,
        "Previous SI-DL pi1": previous_sidl_1,
        "Previous SI-DL pi2": previous_sidl_2,
    }
    return {
        name: np.linalg.lstsq(basis.T, omega, rcond=None)[0]
        for name, omega in rows.items()
    }


def make_init_population(num_input: int, basis: np.ndarray, seed: int) -> np.ndarray:
    dim = basis.shape[0] * num_input
    pop_n = max(5, DE_POPSIZE * dim)
    rng = np.random.default_rng(seed + 991 * num_input)
    init = rng.uniform(DE_BOUNDS[0], DE_BOUNDS[1], size=(pop_n, dim))
    anchors = params_for_known_groups(basis)
    one_dim = [
        anchors["Known Re"],
        anchors["Known k_s/D"],
        anchors["Previous SI-DL pi1"],
        anchors["Previous SI-DL pi2"],
    ]
    if num_input == 1:
        rows = [row for row in one_dim]
    else:
        rows = [
            np.r_[anchors["Known Re"], anchors["Known k_s/D"]],
            np.r_[anchors["Known k_s/D"], anchors["Known Re"]],
            np.r_[anchors["Previous SI-DL pi1"], anchors["Previous SI-DL pi2"]],
            np.r_[anchors["Previous SI-DL pi2"], anchors["Previous SI-DL pi1"]],
        ]
        for base_row in list(rows):
            for scale in (0.01, 0.03, 0.06):
                rows.append(base_row + rng.normal(0.0, scale, size=dim))
    for idx, row in enumerate(rows[:pop_n]):
        init[idx] = np.clip(row, DE_BOUNDS[0], DE_BOUNDS[1])
    return init


def run_sidl_de(X: np.ndarray, y: np.ndarray, num_input: int, seed: int) -> dict:
    basis = np.asarray(SI_DL.calc_basis(D_IN, 2), dtype=float)
    bounds = [DE_BOUNDS] * (basis.shape[0] * num_input)
    init = make_init_population(num_input, basis, seed)
    eval_count = 0
    best_score = -np.inf
    best_params = None
    log_rows = []
    started = time.time()

    def objective(params: np.ndarray) -> float:
        nonlocal eval_count, best_score, best_params
        eval_count += 1
        try:
            omegas = params_to_omegas(params, basis, num_input)
            if np.linalg.matrix_rank(omegas, tol=1e-8) < num_input:
                return 1e6
            feature = log_feature_from_omegas(X, omegas)
            if not np.all(np.isfinite(feature)):
                return 1e6
            if np.any(np.std(feature, axis=0, ddof=0) <= 1e-12):
                return 1e6
            score = float(evaluate_feature(feature, y)["S_cov"])
        except Exception:
            return 1e6
        if not np.isfinite(score):
            return 1e6
        if score > best_score:
            best_score = score
            best_params = np.asarray(params, dtype=float).copy()
        return -score

    def callback(xk: np.ndarray, convergence: float) -> bool:
        elapsed = time.time() - started
        omegas = params_to_omegas(xk, basis, num_input)
        feature = log_feature_from_omegas(X, omegas)
        score = evaluate_feature(feature, y)
        row = {
            "num_input": num_input,
            "seed": seed,
            "iteration": len(log_rows) + 1,
            "evaluations": eval_count,
            "elapsed_seconds": elapsed,
            "convergence": float(convergence),
            "S_cov": float(score["S_cov"]),
            "S_cov_raw": float(score["S_cov_raw"]),
            "sidl_error": float(1.0 - score["S_cov"]),
            "nrmse_var": float(score["nrmse_var"]),
            "corr_Y_mhat": float(score["corr_Y_mhat"]),
            "formula": formula_from_exponents(omegas),
        }
        log_rows.append(row)
        print(
            f"k={num_input} seed={seed} iter={row['iteration']:03d} "
            f"eval={eval_count} S_cov={row['S_cov']:.8f} "
            f"err={row['sidl_error']:.3e} elapsed={elapsed:.1f}s",
            flush=True,
        )
        return False

    print(
        f"Starting SI-DL DE: k={num_input}, seed={seed}, n={X.shape[0]}, "
        f"maxiter={DE_MAXITER}, popsize={DE_POPSIZE}, bounds={DE_BOUNDS}, bandwidth={SI_BANDWIDTH}",
        flush=True,
    )
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=DE_TOL,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        seed=seed,
        init=init,
        polish=DE_POLISH,
        updating="immediate",
        workers=1,
        callback=callback,
    )
    params = np.asarray(result.x if result.fun <= -best_score else best_params, dtype=float)
    omegas = params_to_omegas(params, basis, num_input)
    feature = log_feature_from_omegas(X, omegas)
    score = evaluate_feature(feature, y)
    return {
        "num_input": num_input,
        "seed": seed,
        "params": params,
        "omegas": omegas,
        "feature": feature,
        "score": score,
        "optimizer_result": result,
        "log": pd.DataFrame(log_rows),
        "elapsed_seconds": time.time() - started,
        "objective_evaluations": eval_count,
    }


def evaluate_reference_rows(X: np.ndarray, y: np.ndarray, basis: np.ndarray) -> list[dict]:
    anchors = params_for_known_groups(basis)
    references = {
        "Known Re": anchors["Known Re"].reshape(1, -1),
        "Known k_s/D": anchors["Known k_s/D"].reshape(1, -1),
        "Known [Re, k_s/D]": np.vstack([anchors["Known Re"], anchors["Known k_s/D"]]),
        "Previous SI-DL 2D": np.vstack([anchors["Previous SI-DL pi1"], anchors["Previous SI-DL pi2"]]),
    }
    rows = []
    for method, params in references.items():
        omegas = params_to_omegas(params.reshape(-1), basis, params.shape[0])
        feature = log_feature_from_omegas(X, omegas)
        score = evaluate_feature(feature, y)
        rows.append(summary_row(method, params.shape[0], np.nan, omegas, score, None, 0, 0.0))
    return rows


def summary_row(
    method: str,
    num_input: int,
    seed: float,
    omegas: np.ndarray,
    score: dict,
    result,
    objective_evaluations: int,
    elapsed_seconds: float,
) -> dict:
    return {
        "method": method,
        "num_input": int(num_input),
        "seed": seed,
        "n_samples": N_SAMPLES,
        "bandwidth": SI_BANDWIDTH,
        "bounds": str(DE_BOUNDS),
        "maxiter": DE_MAXITER,
        "popsize": DE_POPSIZE,
        "polish": DE_POLISH,
        "S_cov": float(score["S_cov"]),
        "S_cov_raw": float(score["S_cov_raw"]),
        "sidl_error": float(1.0 - score["S_cov"]),
        "mse": float(score["mse"]),
        "rmse": float(score["rmse"]),
        "nrmse_var": float(score["nrmse_var"]),
        "corr_Y_mhat": float(score["corr_Y_mhat"]),
        "n_retained": int(score["n_retained"]),
        "objective_evaluations": int(objective_evaluations),
        "elapsed_seconds": float(elapsed_seconds),
        "optimizer_success": "" if result is None else bool(result.success),
        "optimizer_message": "" if result is None else str(result.message),
        "formula": formula_from_exponents(omegas),
    }


def exponents_rows(method: str, omegas: np.ndarray) -> list[dict]:
    rows = []
    for idx, row in enumerate(np.asarray(omegas, dtype=float), start=1):
        for label, value in zip(VARIABLE_LABELS, row):
            rows.append(
                {
                    "method": method,
                    "pi_group": f"pi{idx}",
                    "variable": label,
                    "normalized_exponent": float(value),
                }
            )
    return rows


def plot_summary_table(summary: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18.0, 5.6), dpi=220)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(
        0.5,
        0.96,
        f"Colebrook {N_SAMPLES} points: SI-DL only, single and two input searches",
        ha="center",
        va="top",
        fontsize=18,
        weight="bold",
    )
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                row["method"],
                str(row["num_input"]),
                f"{row['S_cov']:.6f}",
                f"{row['sidl_error']:.3e}",
                f"{row['nrmse_var']:.3e}",
                f"{row['corr_Y_mhat']:.6f}",
                row["formula"],
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["Method", "k", "S_cov", "1-S_cov", "NRMSE", "corr", "Pi groups"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.16, 0.04, 0.08, 0.09, 0.09, 0.08, 0.46],
        bbox=[0.015, 0.06, 0.97, 0.78],
    )
    table.auto_set_font_size(False)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#3f3f46")
        cell.set_linewidth(0.8)
        cell.PAD = 0.025
        text = cell.get_text()
        if r == 0:
            cell.set_facecolor("#1f2937")
            text.set_color("white")
            text.set_weight("bold")
            text.set_fontsize(9.5)
        else:
            cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
            text.set_fontsize(6.8 if c == 6 else 8.8)
            if c == 6:
                text.set_ha("left")
            if c == 0:
                text.set_weight("bold")
    output = FIG_DIR / f"{OUTPUT_PREFIX}_summary_table.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    generated = generate_colebrook_data(N_SAMPLES)
    X = generated[VARIABLE_LABELS].to_numpy(float)
    y = generated["Cf"].to_numpy(float)
    basis = np.asarray(SI_DL.calc_basis(D_IN, 2), dtype=float)

    generated_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_generated_data.csv"
    generated.to_csv(generated_csv, index=False)

    summary_rows = evaluate_reference_rows(X, y, basis)
    exponents = []
    coordinates = generated[["Cf", "Re", "relative_roughness"]].copy()
    logs = []
    for num_input in (1, 2):
        best = None
        for seed in DE_SEEDS:
            result = run_sidl_de(X, y, num_input=num_input, seed=seed)
            method = f"SI-DL {num_input}D DE seed{seed}"
            logs.append(result["log"])
            row = summary_row(
                method,
                num_input,
                seed,
                result["omegas"],
                result["score"],
                result["optimizer_result"],
                result["objective_evaluations"],
                result["elapsed_seconds"],
            )
            summary_rows.append(row)
            if best is None or row["S_cov"] > best["row"]["S_cov"]:
                best = {"row": row, "result": result}

        best_method = f"SI-DL {num_input}D DE best"
        best_row = dict(best["row"])
        best_row["method"] = best_method
        summary_rows.append(best_row)
        exponents.extend(exponents_rows(best_method, best["result"]["omegas"]))
        for j in range(num_input):
            coordinates[f"{best_method}_log_pi{j + 1}"] = best["result"]["feature"][:, j]
            coordinates[f"{best_method}_pi{j + 1}"] = np.exp(best["result"]["feature"][:, j])

    for row in summary_rows:
        if row["method"].startswith("Known") or row["method"].startswith("Previous"):
            # Reconstruct reference exponents for the exponent CSV.
            pass
    for ref in evaluate_reference_rows(X, y, basis):
        pass

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["num_input", "S_cov"], ascending=[True, False])

    reference_params = params_for_known_groups(basis)
    reference_omegas = {
        "Known Re": params_to_omegas(reference_params["Known Re"], basis, 1),
        "Known k_s/D": params_to_omegas(reference_params["Known k_s/D"], basis, 1),
        "Known [Re, k_s/D]": params_to_omegas(
            np.r_[reference_params["Known Re"], reference_params["Known k_s/D"]],
            basis,
            2,
        ),
        "Previous SI-DL 2D": params_to_omegas(
            np.r_[reference_params["Previous SI-DL pi1"], reference_params["Previous SI-DL pi2"]],
            basis,
            2,
        ),
    }
    for method, omegas in reference_omegas.items():
        exponents.extend(exponents_rows(method, omegas))

    summary_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"
    exponents_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_exponents.csv"
    coordinates_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_coordinates.csv"
    log_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_de_log.csv"
    summary.to_csv(summary_csv, index=False)
    pd.DataFrame(exponents).to_csv(exponents_csv, index=False)
    coordinates.to_csv(coordinates_csv, index=False)
    pd.concat(logs, ignore_index=True).to_csv(log_csv, index=False)
    figure = plot_summary_table(summary)

    print(
        summary[
            [
                "method",
                "num_input",
                "seed",
                "S_cov",
                "sidl_error",
                "nrmse_var",
                "corr_Y_mhat",
                "objective_evaluations",
                "elapsed_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {generated_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {exponents_csv}")
    print(f"Wrote {coordinates_csv}")
    print(f"Wrote {log_csv}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
