from __future__ import annotations

import os
import pickle
import sys
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

for module_dir in [ROOT / "Compare" / "MI-DL", ROOT / "SI-DL-main"]:
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import midl
import SI_DL


RANDOM_STATE = 42
FIG_DIR = OUTPUT_DIR / "figures"
SOURCE_PICKLE = OUTPUT_DIR / "Malkus" / "output.pkl"

VARIABLE_LABELS = ["r", "q_1", "I", "nu", "K", "g", "omega", "a_1", "b_1"]

MI_K_NEIGHBORS = 6
MI_JOINT_DE_MAXITER = 35
MI_JOINT_POPSIZE = 3
MI_RESTART_SEEDS = [42, 43, 44]

SI_BANDWIDTH = 0.1
SI_MAXITER = 40
SI_POPSIZE = 8

MI_SEARCH_SAMPLE_SIZE = 1000
SI_SEARCH_SAMPLE_SIZE = 3000
SI_VALIDATION_SAMPLE_SIZE = 6000
SI_VALIDATION_OFFSETS = [4, 5, 6]
METRIC_SAMPLE_SIZE = 6000
JOINT_SEARCH_ESTIMATOR = "ksg_joint_mi"
MI_INIT_JITTER_SCALE = 0.10
SI_INIT_JITTER_SCALE = 0.10

# The source pickle is produced by Malkus.ipynb after the notebook has already
# transformed dq to Pi0 = (d omega / dt) / K^2.
OUTPUT_LABEL = r"$\Pi_0=(d\omega/dt)/K^2$"

D_IN = np.array(
    [
        [1, 0, 2, 2, 0, 1, 0, 0, 0],
        [0, -1, 0, -1, -1, -2, -1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1],
    ],
    dtype=float,
)


def load_malkus_data() -> tuple[np.ndarray, np.ndarray, dict]:
    with SOURCE_PICKLE.open("rb") as handle:
        data = pickle.load(handle)

    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float).reshape(-1)

    return X, Y, data


def filter_valid_rows(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    valid_rows = np.all(X > 0, axis=1) & np.isfinite(Y)

    return X[valid_rows], Y[valid_rows]


def scale_exponents(exponents: np.ndarray, canonical_sign: bool = True) -> np.ndarray:
    row = np.asarray(exponents, dtype=float).reshape(-1)
    scale = float(np.max(np.abs(row)))

    if scale <= 1e-12:
        return row

    out = row / scale
    if canonical_sign:
        first = np.flatnonzero(np.abs(out) > 1e-10)

        if first.size and out[first[0]] < 0.0:
            out = -out

    return out


def normalize_exponents(exponents: np.ndarray) -> np.ndarray:
    return scale_exponents(exponents, canonical_sign=True)


def pi_from_omegas(X: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    return np.exp(
        np.log(np.asarray(X, dtype=float))
        @ np.asarray(omegas, dtype=float).T
    )


def formula_from_exponents(omegas: np.ndarray, decimals: int = 3) -> str:
    lines = []

    for idx, row in enumerate(np.asarray(omegas, dtype=float), start=1):
        terms = []

        for label, value in zip(VARIABLE_LABELS, row):
            value = float(value)

            if abs(value) < 5e-4:
                continue

            terms.append(f"{label}^{value:.{decimals}f}")

        lines.append(f"pi{idx}: " + " * ".join(terms))

    return "\n".join(lines)


def information_metrics(feature: np.ndarray, Y: np.ndarray) -> dict:
    info = midl.information_lower_bound(
        np.asarray(feature, dtype=float),
        Y,
        k_neighbors=MI_K_NEIGHBORS,
        random_state=RANDOM_STATE,
    )

    return {
        "mutual_information": float(info["mutual_information"]),
        "epsilon_lb_normalized": float(info["epsilon_lb"] / np.var(Y, ddof=0)),
    }


def true_joint_search_mi(feature: np.ndarray, Y: np.ndarray, seed: int) -> float:
    feature = np.asarray(feature, dtype=float)
    if feature.ndim == 1:
        feature = feature.reshape(-1, 1)
    if not np.all(np.isfinite(feature)):
        return -np.inf
    try:
        info = midl.information_lower_bound(
            feature,
            np.asarray(Y, dtype=float).reshape(-1),
            k_neighbors=MI_K_NEIGHBORS,
            random_state=seed,
        )
    except Exception:
        return -np.inf
    return float(info["mutual_information"])


def sample_idx(n_samples: int, size: int, seed_offset: int) -> np.ndarray:
    if n_samples <= size:
        return np.arange(n_samples)

    rng = np.random.default_rng(RANDOM_STATE + seed_offset)

    return np.sort(rng.choice(n_samples, size=size, replace=False))


def params_from_omegas(
    omegas: np.ndarray,
    basis: np.ndarray,
    canonical_sign: bool = True,
) -> np.ndarray:
    rows = []
    for row in np.asarray(omegas, dtype=float):
        coef, *_ = np.linalg.lstsq(
            basis,
            scale_exponents(row, canonical_sign=canonical_sign),
            rcond=None,
        )
        rows.append(coef)
    return np.asarray(rows, dtype=float).reshape(-1)


def initial_population_from_omegas(
    initial_omegas: np.ndarray | None,
    basis: np.ndarray,
    seed: int,
    jitter_scale: float,
    popsize: int,
    canonical_sign: bool = True,
) -> np.ndarray | str:
    n_params = 2 * basis.shape[1]
    pop_size = max(5, popsize * n_params)
    if initial_omegas is None:
        return "latinhypercube"

    center = params_from_omegas(initial_omegas, basis, canonical_sign=canonical_sign)
    rng = np.random.default_rng(seed)
    population = center + rng.normal(
        loc=0.0,
        scale=jitter_scale,
        size=(pop_size, n_params),
    )
    population[0] = center
    return np.clip(population, -2.0, 2.0)


def run_midl_search(
    X: np.ndarray,
    Y: np.ndarray,
    search_idx: np.ndarray,
    initial_omegas: np.ndarray | None = None,
) -> dict:
    basis, _ = midl.calc_basis(D_IN)
    log_x_search = np.log(X[search_idx])
    Y_search = Y[search_idx]

    rows = []
    best = None

    for seed in MI_RESTART_SEEDS:
        def objective(params: np.ndarray) -> float:
            try:
                omegas = params_to_omegas(params, basis)
                if np.linalg.matrix_rank(omegas, tol=1e-8) < 2:
                    return 1e6
                feature = np.exp(log_x_search @ omegas.T)
                if not np.all(np.isfinite(feature)):
                    return 1e6
                score = true_joint_search_mi(feature, Y_search, seed)
            except Exception:
                return 1e6
            if not np.isfinite(score):
                return 1e6
            return -float(score)

        result = differential_evolution(
            objective,
            bounds=[(-2.0, 2.0)] * (2 * basis.shape[1]),
            maxiter=MI_JOINT_DE_MAXITER,
            popsize=MI_JOINT_POPSIZE,
            seed=seed,
            init=initial_population_from_omegas(
                initial_omegas,
                basis,
                seed,
                jitter_scale=MI_INIT_JITTER_SCALE,
                popsize=MI_JOINT_POPSIZE,
            ),
            polish=False,
            updating="deferred",
            workers=1,
        )

        omegas = params_to_omegas(result.x, basis)
        feature_search = pi_from_omegas(X[search_idx], omegas)
        mi_search = information_metrics(feature_search, Y_search)

        row = {
            "seed": seed,
            "search_estimator": JOINT_SEARCH_ESTIMATOR,
            "initialized_near_itpi": bool(initial_omegas is not None),
            "init_jitter_scale": float(MI_INIT_JITTER_SCALE) if initial_omegas is not None else np.nan,
            "joint_mi_search": mi_search["mutual_information"],
            "epsilon_lb_search_normalized": mi_search["epsilon_lb_normalized"],
            "optimizer_fun": float(result.fun),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "formula": formula_from_exponents(omegas),
            "omegas": omegas,
        }

        rows.append(row)
        print(
            f"MI-DL joint seed={seed}: "
            f"search MI={row['joint_mi_search']:.6f}, "
            f"epsilon/Var={row['epsilon_lb_search_normalized']:.6g}",
            flush=True,
        )

        if best is None or row["joint_mi_search"] > best["joint_mi_search"]:
            best = row

    audit = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "omegas"} for row in rows]
    )

    return {
        "omegas": best["omegas"],
        "best_seed": int(best["seed"]),
        "restart_audit": audit,
    }


def params_to_omegas(
    params: np.ndarray,
    basis: np.ndarray,
    canonical_sign: bool = True,
) -> np.ndarray:
    coef_matrix = np.asarray(params, dtype=float).reshape(2, basis.shape[1])
    raw = coef_matrix @ basis.T

    return np.asarray(
        [scale_exponents(row, canonical_sign=canonical_sign) for row in raw]
    )


def itpi_malkus_basis() -> np.ndarray:
    basis = np.asarray(
        SI_DL.calc_basis(
            D_IN,
            D_IN.shape[1] - np.linalg.matrix_rank(D_IN),
        ),
        dtype=float,
    )
    basis[3:6, :] = -basis[3:6, :]
    return basis.T


def load_itpi_continuous_omegas(data: dict) -> np.ndarray | None:
    try:
        rows = data["results"]["input_coef"]
    except KeyError:
        return None
    return np.asarray([np.asarray(row, dtype=float).reshape(-1) for row in rows], dtype=float)


def run_sidl_search(
    X: np.ndarray,
    Y: np.ndarray,
    search_idx: np.ndarray,
    validation_indices: list[np.ndarray],
    initial_omegas: np.ndarray | None = None,
) -> dict:
    basis = itpi_malkus_basis()

    X_search = X[search_idx]
    Y_search = Y[search_idx]
    validation_sets = [(X[idx], Y[idx]) for idx in validation_indices]

    def score_omegas(omegas: np.ndarray, X_eval: np.ndarray, Y_eval: np.ndarray) -> float:
        feature = pi_from_omegas(X_eval, omegas)
        return float(
            SI_DL.explained_variance_score(
                feature,
                Y_eval,
                bandwidth=SI_BANDWIDTH,
            )["S_cov"]
        )

    def objective(params: np.ndarray) -> float:
        try:
            omegas = params_to_omegas(params, basis, canonical_sign=False)

            if np.linalg.matrix_rank(omegas, tol=1e-8) < 2:
                return 1e6

            feature = pi_from_omegas(X_search, omegas)

            score = SI_DL.explained_variance_score(
                feature,
                Y_search,
                bandwidth=SI_BANDWIDTH,
            )["S_cov"]

        except Exception:
            return 1e6

        if not np.isfinite(score):
            return 1e6

        return -float(score)

    result = differential_evolution(
        objective,
        bounds=[(-2.0, 2.0)] * (2 * basis.shape[1]),
        maxiter=SI_MAXITER,
        popsize=SI_POPSIZE,
        seed=RANDOM_STATE,
        init=initial_population_from_omegas(
            initial_omegas,
            basis,
            RANDOM_STATE,
            jitter_scale=SI_INIT_JITTER_SCALE,
            popsize=SI_POPSIZE,
            canonical_sign=False,
        ),
        polish=False,
        updating="immediate",
        workers=1,
    )

    candidate_rows = []

    def add_candidate(label: str, omegas: np.ndarray) -> None:
        try:
            if np.linalg.matrix_rank(omegas, tol=1e-8) < 2:
                return
            train_score = score_omegas(omegas, X_search, Y_search)
            validation_scores = [
                score_omegas(omegas, X_validation, Y_validation)
                for X_validation, Y_validation in validation_sets
            ]
        except Exception:
            return
        candidate_rows.append(
            {
                "candidate": label,
                "train_S_cov": float(train_score),
                "validation_mean_S_cov": float(np.mean(validation_scores)),
                "validation_min_S_cov": float(np.min(validation_scores)),
                "validation_std_S_cov": float(np.std(validation_scores, ddof=0)),
                **{
                    f"validation_{idx + 1}_S_cov": float(score)
                    for idx, score in enumerate(validation_scores)
                },
                "omegas": np.asarray(omegas, dtype=float),
            }
        )

    add_candidate(
        "de_best",
        params_to_omegas(result.x, basis, canonical_sign=False),
    )

    if getattr(result, "population", None) is not None:
        for idx, params in enumerate(np.asarray(result.population, dtype=float)):
            add_candidate(
                f"de_population_{idx:03d}",
                params_to_omegas(params, basis, canonical_sign=False),
            )

    if initial_omegas is not None:
        add_candidate("itpi_initial_raw", np.asarray(initial_omegas, dtype=float))
        init_params = params_from_omegas(
            initial_omegas,
            basis,
            canonical_sign=False,
        )
        add_candidate(
            "itpi_initial_projected_to_sidl_basis",
            params_to_omegas(init_params, basis, canonical_sign=False),
        )

    if not candidate_rows:
        raise RuntimeError("SI-DL validation selection produced no valid candidates.")

    candidates = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "omegas"} for row in candidate_rows]
    ).sort_values(
        ["validation_min_S_cov", "validation_mean_S_cov", "train_S_cov"],
        ascending=[False, False, False],
    )
    best_candidate = max(
        candidate_rows,
        key=lambda row: (
            row["validation_min_S_cov"],
            row["validation_mean_S_cov"],
            row["train_S_cov"],
        ),
    )

    return {
        "omegas": best_candidate["omegas"],
        "optimizer_result": result,
        "initialized_near_itpi": bool(initial_omegas is not None),
        "validation_audit": candidates,
        "selected_candidate": str(best_candidate["candidate"]),
        "selected_train_S_cov": float(best_candidate["train_S_cov"]),
        "selected_validation_mean_S_cov": float(best_candidate["validation_mean_S_cov"]),
        "selected_validation_min_S_cov": float(best_candidate["validation_min_S_cov"]),
    }


def score_metrics(feature: np.ndarray, Y: np.ndarray) -> dict:
    sidl_score = SI_DL.explained_variance_score(
        feature,
        Y,
        bandwidth=SI_BANDWIDTH,
    )

    mi = information_metrics(feature, Y)

    return {
        **mi,
        "S_cov": float(sidl_score["S_cov"]),
        "sidl_error": float(1.0 - sidl_score["S_cov"]),
        "sidl_bandwidth": float(sidl_score["bandwidth"]),
        "sidl_n_retained": int(sidl_score["n_retained"]),
    }


def plot_summary_table(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 5.4), dpi=220)

    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.96,
        "Malkus comparison, k=6",
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
                row["formula"],
                f"{row['mutual_information']:.4f}",
                f"{row['epsilon_lb_normalized']:.5f}",
                f"{row['S_cov']:.4f}",
                f"{row['sidl_error']:.5f}",
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Method",
            "Found pi groups",
            "MI k=6",
            "epsilon_LB/Var",
            "S_cov",
            "1 - S_cov",
        ],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.12, 0.52, 0.09, 0.11, 0.08, 0.08],
        bbox=[0.02, 0.08, 0.96, 0.78],
    )

    table.auto_set_font_size(False)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#3f3f46")
        cell.set_linewidth(0.85)
        cell.PAD = 0.03

        text = cell.get_text()

        if r == 0:
            cell.set_facecolor("#1f2937")
            text.set_color("white")
            text.set_weight("bold")
            text.set_fontsize(10.0)
        else:
            cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
            text.set_fontsize(6.8 if c == 1 else 9.8)

            if c == 1:
                text.set_ha("left")

            if c == 0:
                text.set_weight("bold")

    fig.savefig(
        FIG_DIR / "malkus_k6_summary_table.png",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def plot_3d_points(coordinates: pd.DataFrame, output_label: str) -> None:
    methods = [
        method.removesuffix("_pi1")
        for method in coordinates.columns
        if method.endswith("_pi1")
    ]

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(6.2 * len(methods), 5.3),
        subplot_kw={"projection": "3d"},
        dpi=220,
    )
    axes = np.atleast_1d(axes)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(
        len(coordinates),
        size=min(5000, len(coordinates)),
        replace=False,
    )

    for ax, method in zip(axes, methods):
        ax.scatter(
            coordinates.loc[idx, f"{method}_pi1"],
            coordinates.loc[idx, f"{method}_pi2"],
            coordinates.loc[idx, "Y"],
            s=5,
            alpha=0.42,
            linewidths=0,
        )

        ax.set_title(method)
        ax.set_xlabel(r"$\pi_1$")
        ax.set_ylabel(r"$\pi_2$")
        ax.set_zlabel(output_label)

    fig.suptitle("Malkus data in raw pi coordinates", y=1.02)
    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "malkus_raw_pi_data_points.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def run_comparison() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_raw, Y_raw, source_data = load_malkus_data()
    X, Y = filter_valid_rows(X_raw, Y_raw)

    print(f"Raw samples: {X_raw.shape[0]}")
    print(f"Valid samples: {X.shape[0]}")
    print(f"Output used from IT-PI pickle: {OUTPUT_LABEL}")

    mi_search_idx = sample_idx(X.shape[0], MI_SEARCH_SAMPLE_SIZE, 1)
    si_search_idx = sample_idx(X.shape[0], SI_SEARCH_SAMPLE_SIZE, 1)
    si_validation_indices = [
        sample_idx(X.shape[0], SI_VALIDATION_SAMPLE_SIZE, offset)
        for offset in SI_VALIDATION_OFFSETS
    ]
    metric_idx = sample_idx(X.shape[0], METRIC_SAMPLE_SIZE, 2)
    itpi_continuous = load_itpi_continuous_omegas(source_data)

    midl = run_midl_search(X, Y, mi_search_idx, initial_omegas=itpi_continuous)
    sidl = run_sidl_search(
        X,
        Y,
        si_search_idx,
        si_validation_indices,
        initial_omegas=itpi_continuous,
    )

    methods = {
        "MI-DL joint": {"omegas": midl["omegas"], "n_search_samples": int(mi_search_idx.size)},
        "SI-DL": {"omegas": sidl["omegas"], "n_search_samples": int(si_search_idx.size)},
    }
    if itpi_continuous is not None:
        methods["IT-PI continuous"] = {"omegas": itpi_continuous, "n_search_samples": int(mi_search_idx.size)}

    coordinates = pd.DataFrame(
        {
            "Y": Y,
            "Pi0_domega_dt_over_K2": Y,
        }
    )

    rows = []
    exponent_rows = []

    for method, values in methods.items():
        feature = pi_from_omegas(X, values["omegas"])

        coordinates[f"{method}_pi1"] = feature[:, 0]
        coordinates[f"{method}_pi2"] = feature[:, 1]

        common = score_metrics(
            feature[metric_idx],
            Y[metric_idx],
        )

        rows.append(
            {
                "method": method,
                "formula": formula_from_exponents(values["omegas"]),
                "n_samples": int(Y.size),
                "n_search_samples": int(values["n_search_samples"]),
                "n_sidl_validation_samples": int(SI_VALIDATION_SAMPLE_SIZE) if method == "SI-DL" else np.nan,
                "n_sidl_validation_splits": len(si_validation_indices) if method == "SI-DL" else np.nan,
                "n_metric_samples": int(metric_idx.size),
                "feature_space": "raw_pi",
                "output_from_pickle": "Pi0 = (domega_dt) / K^2",
                **common,
                "best_seed": midl["best_seed"] if method == "MI-DL joint" else np.nan,
                "sidl_selected_candidate": sidl["selected_candidate"] if method == "SI-DL" else "",
                "sidl_selected_train_S_cov": sidl["selected_train_S_cov"] if method == "SI-DL" else np.nan,
                "sidl_selected_validation_mean_S_cov": sidl["selected_validation_mean_S_cov"] if method == "SI-DL" else np.nan,
                "sidl_selected_validation_min_S_cov": sidl["selected_validation_min_S_cov"] if method == "SI-DL" else np.nan,
            }
        )

        for pi_idx, row in enumerate(values["omegas"], start=1):
            for label, exponent in zip(VARIABLE_LABELS, row):
                exponent_rows.append(
                    {
                        "method": method,
                        "pi_group": f"pi{pi_idx}",
                        "variable": label,
                        "normalized_exponent": float(exponent),
                    }
                )

    summary = pd.DataFrame(rows)

    summary["rank_by_MI"] = summary["mutual_information"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    summary["rank_by_S_cov"] = summary["S_cov"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    summary.to_csv(
        OUTPUT_DIR / "malkus_k6_summary.csv",
        index=False,
    )

    coordinates.to_csv(
        OUTPUT_DIR / "malkus_k6_coordinates.csv",
        index=False,
    )

    pd.DataFrame(exponent_rows).to_csv(
        OUTPUT_DIR / "malkus_k6_exponents.csv",
        index=False,
    )

    midl["restart_audit"].to_csv(
        OUTPUT_DIR / "malkus_midl_k6_restart_audit.csv",
        index=False,
    )

    sidl["validation_audit"].to_csv(
        OUTPUT_DIR / "malkus_sidl_validation_audit.csv",
        index=False,
    )

    plot_summary_table(summary)
    plot_3d_points(coordinates, OUTPUT_LABEL)

    return {"summary": summary}


if __name__ == "__main__":
    outputs = run_comparison()

    print("\nMalkus k=6 summary using IT-PI pickle output Pi0 = (domega_dt) / K^2:")

    print(
        outputs["summary"][
            [
                "method",
                "output_from_pickle",
                "mutual_information",
                "epsilon_lb_normalized",
                "S_cov",
                "sidl_error",
                "sidl_bandwidth",
                "sidl_n_retained",
                "rank_by_MI",
                "rank_by_S_cov",
            ]
        ].to_string(index=False)
    )
