from __future__ import annotations

import os
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR.parent
for module_dir in [ROOT / "Compare" / "MI-DL", ROOT / "SI-DL-main"]:
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import midl
import SI_DL


SOURCE_DIR = OUTPUT_DIR
FIG_DIR = OUTPUT_DIR / "figures"

N_SAMPLES = 1000
OUTPUT_PREFIX = f"colebrook_{N_SAMPLES}pts"
RANDOM_STATE = 42
K_NEIGHBORS = 6
SI_BANDWIDTH = 0.06
TEST_SIZE = 0.25
FEATURE_SPACE = "log_pi"
JOINT_SEARCH_SIZE = 350
JOINT_DE_MAXITER = 10
JOINT_DE_POPSIZE = 5
JOINT_RESTART_SEEDS = [0, 42]

VARIABLE_LABELS = ["u", "rho", "D", "k_s", "mu"]
D_IN = np.matrix("1 -3 1 1 -1; -1 0 0 0 -1; 0 1 0 0 1")
POINT_FIG_NAMES = {
    "MI-DL": f"{OUTPUT_PREFIX}_midl_3d_logpi_data_points.png",
    "SI-DL": f"{OUTPUT_PREFIX}_sidl_3d_logpi_data_points.png",
    "Known [Re, k_s/D]": f"{OUTPUT_PREFIX}_known_3d_logpi_data_points.png",
    "Joint MI-DL": f"{OUTPUT_PREFIX}_joint_midl_3d_logpi_data_points.png",
}

plt.rcParams.update(
    {
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)

METHODS = [
    {
        "method": "MI-DL",
        "log_pi_columns": ["MI-DL_log_pi1", "MI-DL_log_pi2"],
    },
    {
        "method": "SI-DL",
        "log_pi_columns": ["SI-DL_log_pi1", "SI-DL_log_pi2"],
    },
    {
        "method": "Known [Re, k_s/D]",
        "log_pi_columns": ["Known_Re_k_s/D_log_pi1", "Known_Re_k_s/D_log_pi2"],
    },
]


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


def log_feature_from_omegas(X: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(X, dtype=float)) @ np.asarray(omegas, dtype=float).T


def params_to_joint_omegas(params: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coef_matrix = np.asarray(params, dtype=float).reshape(2, 2)
    raw = coef_matrix @ basis.T
    return np.asarray([normalize_exponents(row) for row in raw])


def run_joint_midl_search(X: np.ndarray, y: np.ndarray) -> dict:
    basis, _ = midl.calc_basis(D_IN)
    rng = np.random.default_rng(RANDOM_STATE)
    search_idx = rng.choice(X.shape[0], size=min(JOINT_SEARCH_SIZE, X.shape[0]), replace=False)
    X_search = X[search_idx]
    y_search = y[search_idx]
    bounds = [(-2.0, 2.0)] * 4
    audit_rows = []
    best = None

    def objective(params: np.ndarray) -> float:
        try:
            coef_matrix = np.asarray(params, dtype=float).reshape(2, 2)
            if abs(float(np.linalg.det(coef_matrix))) < 0.05:
                return 1e6
            omegas = params_to_joint_omegas(params, basis)
            if np.linalg.matrix_rank(omegas, tol=1e-8) < 2:
                return 1e6
            feature = log_feature_from_omegas(X_search, omegas)
            if not np.all(np.isfinite(feature)) or np.any(np.std(feature, axis=0) <= 1e-12):
                return 1e6
            score = midl.information_lower_bound(
                feature,
                y_search,
                k_neighbors=K_NEIGHBORS,
                random_state=RANDOM_STATE,
            )["mutual_information"]
        except Exception:
            return 1e6
        if not np.isfinite(score):
            return 1e6
        return -float(score)

    for seed in JOINT_RESTART_SEEDS:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=JOINT_DE_MAXITER,
            popsize=JOINT_DE_POPSIZE,
            seed=seed,
            polish=False,
            updating="immediate",
            workers=1,
        )
        omegas = params_to_joint_omegas(result.x, basis)
        feature = log_feature_from_omegas(X, omegas)
        info = information_metrics(feature, y)
        row = {
            "seed": seed,
            "search_objective": float(result.fun),
            "full_mutual_information": info["mutual_information"],
            "formula": formula_from_exponents(omegas),
            "omegas": omegas,
            "feature": feature,
        }
        audit_rows.append({k: v for k, v in row.items() if k not in {"omegas", "feature"}})
        if best is None or row["full_mutual_information"] > best["full_mutual_information"]:
            best = row

    return {
        "formula": best["formula"],
        "feature": best["feature"],
        "omegas": best["omegas"],
        "audit": pd.DataFrame(audit_rows).sort_values("full_mutual_information", ascending=False),
    }


def load_formulas() -> dict[str, str]:
    summary = pd.read_csv(SOURCE_DIR / "colebrook_midl_sidl_summary.csv")
    return dict(zip(summary["method"], summary["formula"]))


def information_metrics(feature: np.ndarray, y: np.ndarray) -> dict[str, float]:
    info = midl.information_lower_bound(
        feature,
        y,
        k_neighbors=K_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    return {
        "mutual_information": float(info["mutual_information"]),
        "epsilon_lb": float(info["epsilon_lb"]),
        "epsilon_lb_normalized": float(info["epsilon_lb"] / np.var(y, ddof=0)),
    }


def gpr_fit(feature: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, float]:
    kernel = ConstantKernel(1.0) * RBF([1.0, 1.0]) + WhiteKernel(1e-6)
    model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            optimizer=None,
            n_restarts_optimizer=0,
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(feature[train_idx], y[train_idx])
    pred = model.predict(feature[test_idx])
    mse_raw = float(mean_squared_error(y[test_idx], pred))
    scale = float(np.var(y[train_idx], ddof=0))
    return {
        "gpr_mse_raw": mse_raw,
        "gpr_mse_normalized": mse_raw / scale,
        "gpr_rmse_normalized": float(np.sqrt(mse_raw / scale)),
        "error_scale_var_y_train": scale,
    }


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


def method_column_prefix(method: str) -> str:
    if method == "Known [Re, k_s/D]":
        return "Known_Re_k_s/D"
    return method


def exponent_matrix(method: str) -> np.ndarray:
    rows = pd.read_csv(SOURCE_DIR / "colebrook_found_exponents.csv")
    method_rows = rows[rows["method"] == method]
    omegas = []
    for pi_group in ["pi1", "pi2"]:
        group = method_rows[method_rows["pi_group"] == pi_group]
        values = []
        for label in VARIABLE_LABELS:
            value = group.loc[group["variable"] == label, "normalized_exponent"]
            if value.empty:
                raise ValueError(f"Missing exponent for {method} {pi_group} {label}.")
            values.append(float(value.iloc[0]))
        omegas.append(values)
    return np.asarray(omegas, dtype=float)


def add_method_features(coordinates: pd.DataFrame, generated: pd.DataFrame, method: str) -> None:
    prefix = method_column_prefix(method)
    log_pi = log_feature_from_omegas(generated[VARIABLE_LABELS].to_numpy(float), exponent_matrix(method))
    coordinates[f"{prefix}_log_pi1"] = log_pi[:, 0]
    coordinates[f"{prefix}_log_pi2"] = log_pi[:, 1]
    coordinates[f"{prefix}_pi1"] = np.exp(log_pi[:, 0])
    coordinates[f"{prefix}_pi2"] = np.exp(log_pi[:, 1])


def generate_colebrook_data(n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    generated = pd.DataFrame(
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
    coordinates = generated[["Cf", "Re", "relative_roughness"]].copy()
    for method in ["MI-DL", "SI-DL", "Known [Re, k_s/D]"]:
        add_method_features(coordinates, generated, method)
    return coordinates, generated


def load_or_generate_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    coordinates = pd.read_csv(SOURCE_DIR / "colebrook_coordinates.csv")
    generated = pd.read_csv(SOURCE_DIR / "colebrook_generated_data.csv")
    if min(len(coordinates), len(generated)) >= N_SAMPLES:
        return coordinates.head(N_SAMPLES).copy(), generated.head(N_SAMPLES).copy()
    return generate_colebrook_data(N_SAMPLES)


def build_summary() -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coordinates, generated = load_or_generate_data()
    formulas = load_formulas()
    y = coordinates["Cf"].to_numpy(float)
    X = generated[VARIABLE_LABELS].to_numpy(float)
    train_idx, test_idx = train_test_split(
        np.arange(y.size),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    rows = []
    feature_rows = pd.DataFrame(
        {
            "Cf": y,
            "Re": coordinates["Re"].to_numpy(float),
            "relative_roughness": coordinates["relative_roughness"].to_numpy(float),
        }
    )
    methods = [dict(method) for method in METHODS]
    joint = run_joint_midl_search(X, y)
    formulas["Joint MI-DL"] = joint["formula"]
    feature_rows["Joint MI-DL_log_pi1"] = joint["feature"][:, 0]
    feature_rows["Joint MI-DL_log_pi2"] = joint["feature"][:, 1]
    feature_rows["Joint MI-DL_pi1"] = np.exp(joint["feature"][:, 0])
    feature_rows["Joint MI-DL_pi2"] = np.exp(joint["feature"][:, 1])
    methods.append(
        {
            "method": "Joint MI-DL",
            "log_pi_columns": ["Joint MI-DL_log_pi1", "Joint MI-DL_log_pi2"],
            "omegas": joint["omegas"],
        }
    )

    for method in methods:
        name = method["method"]
        if name == "Joint MI-DL":
            feature = feature_rows[method["log_pi_columns"]].to_numpy(float)
        else:
            feature = coordinates[method["log_pi_columns"]].to_numpy(float)
            feature_rows[f"{name}_log_pi1"] = feature[:, 0]
            feature_rows[f"{name}_log_pi2"] = feature[:, 1]
            feature_rows[f"{name}_pi1"] = np.exp(feature[:, 0])
            feature_rows[f"{name}_pi2"] = np.exp(feature[:, 1])

        info = information_metrics(feature, y)
        sidl = SI_DL.explained_variance_score(feature, y, bandwidth=SI_BANDWIDTH)
        gpr = gpr_fit(feature, y, train_idx, test_idx)

        rows.append(
            {
                "method": name,
                "formula": formulas[name],
                "n_samples": N_SAMPLES,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "feature_space": FEATURE_SPACE,
                "k_neighbors": K_NEIGHBORS,
                **info,
                "S_cov": float(sidl["S_cov"]),
                "S_cov_raw": float(sidl["S_cov_raw"]),
                "sidl_error": float(1.0 - sidl["S_cov"]),
                "sidl_bandwidth": float(sidl["bandwidth"]),
                "sidl_n_retained": int(sidl["n_retained"]),
                **gpr,
            }
        )

    summary = pd.DataFrame(rows)
    summary["rank_by_MI"] = summary["mutual_information"].rank(ascending=False, method="min").astype(int)
    summary["rank_by_S_cov"] = summary["S_cov"].rank(ascending=False, method="min").astype(int)
    summary["rank_by_gpr"] = summary["gpr_mse_normalized"].rank(method="min").astype(int)
    exponents = []
    existing = pd.read_csv(SOURCE_DIR / "colebrook_found_exponents.csv")
    exponents.extend(existing.to_dict("records"))
    for idx, row in enumerate(joint["omegas"], start=1):
        for label, value in zip(VARIABLE_LABELS, row):
            exponents.append(
                {
                    "method": "Joint MI-DL",
                    "pi_group": f"pi{idx}",
                    "variable": label,
                    "normalized_exponent": float(value),
                }
            )
    return summary, feature_rows, methods, pd.DataFrame(exponents), joint["audit"], generated


def plot_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), dpi=220)
    metrics = [
        ("mutual_information", "Mutual information", False),
        ("sidl_error", "SI-DL error", True),
        ("gpr_mse_normalized", "GPR normalized MSE", True),
    ]
    colors = ["#3268a8", "#d26b4f", "#667085"]

    for ax, (metric, title, lower_is_better) in zip(axes, metrics):
        values = summary[metric].to_numpy(float)
        ax.bar(summary["method"], values, color=colors, width=0.62)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=18)
        for idx, value in enumerate(values):
            label = f"{value:.3e}" if abs(value) < 1e-3 else f"{value:.5f}"
            ax.text(idx, value, label, ha="center", va="bottom", fontsize=10.5)
        if lower_is_better:
            ax.set_xlabel("lower is better")
        else:
            ax.set_xlabel("higher is better")

    fig.suptitle(f"Colebrook {N_SAMPLES} points, unified log-pi feature space", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_logpi_metric_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_summary_table(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(18.8, 6.2), dpi=220)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(
        0.5,
        0.96,
        f"Colebrook {N_SAMPLES} points, log-pi comparison",
        ha="center",
        va="top",
        fontsize=22,
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
                f"{row['gpr_mse_normalized']:.3e}",
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["Method", "Found pi groups", "MI k=6", "epsilon_LB/Var", "S_cov", "GPR norm. MSE"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.14, 0.48, 0.09, 0.11, 0.08, 0.10],
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
            text.set_fontsize(12.5)
        else:
            cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
            text.set_fontsize(8.8 if c == 1 else 11.5)
            if c == 1:
                text.set_ha("left")
            if c == 0:
                text.set_weight("bold")
    fig.savefig(FIG_DIR / f"{OUTPUT_PREFIX}_summary_table.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_logpi_data_points_3d(feature_rows: pd.DataFrame, methods: list[dict]) -> list[Path]:
    cf = feature_rows["Cf"].to_numpy(float)
    written = []
    old_combined = FIG_DIR / f"{OUTPUT_PREFIX}_logpi_data_points.png"
    if old_combined.exists():
        old_combined.unlink()
    for old_pi_fig in FIG_DIR.glob(f"{OUTPUT_PREFIX}_*_3d_pi_data_points.png"):
        old_pi_fig.unlink()
    for method in methods:
        name = method["method"]
        x = feature_rows[f"{name}_log_pi1"].to_numpy(float)
        y = feature_rows[f"{name}_log_pi2"].to_numpy(float)
        fig = plt.figure(figsize=(8.0, 6.4), dpi=240)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            x,
            y,
            cf,
            s=9,
            color="#3268a8",
            alpha=0.62,
            linewidths=0,
            depthshade=True,
        )
        ax.set_title(f"Colebrook {N_SAMPLES} data points - {name}", fontsize=18, pad=18)
        ax.set_xlabel(r"$\log(\pi_1)$", fontsize=16, labelpad=10)
        ax.set_ylabel(r"$\log(\pi_2)$", fontsize=16, labelpad=10)
        ax.set_zlabel("Cf", fontsize=16, labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.24)
        ax.view_init(elev=24, azim=-58)
        output = FIG_DIR / POINT_FIG_NAMES[name]
        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        written.append(output)
    return written


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    summary, feature_rows, methods, exponents, joint_audit, generated = build_summary()
    summary_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_midl_sidl_logpi_summary.csv"
    features_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_logpi_coordinates.csv"
    exponents_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_logpi_exponents.csv"
    audit_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_joint_midl_audit.csv"
    generated_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_generated_data.csv"
    summary.to_csv(summary_csv, index=False)
    feature_rows.to_csv(features_csv, index=False)
    exponents.to_csv(exponents_csv, index=False)
    joint_audit.to_csv(audit_csv, index=False)
    generated.to_csv(generated_csv, index=False)
    plot_summary(summary)
    plot_summary_table(summary)
    point_figs = plot_logpi_data_points_3d(feature_rows, methods)

    print(
        summary[
            [
                "method",
                "n_samples",
                "feature_space",
                "mutual_information",
                "epsilon_lb_normalized",
                "S_cov",
                "sidl_error",
                "gpr_mse_normalized",
                "rank_by_MI",
                "rank_by_S_cov",
                "rank_by_gpr",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {features_csv}")
    print(f"Wrote {exponents_csv}")
    print(f"Wrote {audit_csv}")
    print(f"Wrote {generated_csv}")
    print(f"Wrote {OUTPUT_DIR / f'{OUTPUT_PREFIX}_logpi_metric_summary.png'}")
    print(f"Wrote {FIG_DIR / f'{OUTPUT_PREFIX}_summary_table.png'}")
    for point_fig in point_figs:
        print(f"Wrote {point_fig}")


if __name__ == "__main__":
    main()
