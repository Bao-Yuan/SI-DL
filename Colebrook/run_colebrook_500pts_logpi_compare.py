from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.spatial import Delaunay, cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent
if str(OUTPUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUTPUT_DIR))

import run_colebrook_1000pts_logpi_compare as base


N_SAMPLES = 500
OUTPUT_PREFIX = f"colebrook_{N_SAMPLES}pts"
SELECTED_METHODS = ["Joint MI-DL", "SI-DL", "Known [Re, k_s/D]"]
DISPLAY_METHOD_NAMES = {
    "Joint MI-DL": "Joint MI-DL",
    "SI-DL": "SI-DL",
    "Known [Re, k_s/D]": r"Known [$Re$, $k_s/D$]",
}


def configure_base() -> None:
    base.N_SAMPLES = N_SAMPLES
    base.OUTPUT_PREFIX = OUTPUT_PREFIX
    base.POINT_FIG_NAMES = {
        "MI-DL": f"{OUTPUT_PREFIX}_midl_3d_logpi_data_points.png",
        "SI-DL": f"{OUTPUT_PREFIX}_sidl_3d_logpi_data_points.png",
        "Known [Re, k_s/D]": f"{OUTPUT_PREFIX}_known_3d_logpi_data_points.png",
        "Joint MI-DL": f"{OUTPUT_PREFIX}_joint_midl_3d_logpi_data_points.png",
    }


def gpr_model_fit(feature: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    kernel = ConstantKernel(1.0) * RBF([1.0, 1.0]) + WhiteKernel(1e-6)
    model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            optimizer=None,
            n_restarts_optimizer=0,
            random_state=base.RANDOM_STATE,
        ),
    )
    model.fit(feature[train_idx], y[train_idx])
    pred = model.predict(feature[test_idx])
    mse_raw = float(mean_squared_error(y[test_idx], pred))
    mse_normalized = mse_raw / float(np.var(y[train_idx], ddof=0))
    return {"model": model, "gpr_mse_normalized": mse_normalized}


def plot_scatter_and_gpr_surfaces(feature_rows: pd.DataFrame, summary: pd.DataFrame) -> Path:
    y = feature_rows["Cf"].to_numpy(float)
    train_idx, test_idx = train_test_split(
        np.arange(y.size),
        test_size=base.TEST_SIZE,
        random_state=base.RANDOM_STATE,
    )
    z_min = float(np.min(y))
    z_max = float(np.max(y))
    z_floor = z_min - 0.06 * (z_max - z_min)
    surface_cmap = LinearSegmentedColormap.from_list(
        "gpr_surface_red",
        ["#fde2dd", "#f05a48", "#9f1239"],
    )
    surface_norm = Normalize(vmin=z_min, vmax=z_max)

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 22,
            "axes.labelsize": 16,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig = plt.figure(figsize=(19.2, 6.6), dpi=260)
    for index, method in enumerate(SELECTED_METHODS, start=1):
        x1 = feature_rows[f"{method}_log_pi1"].to_numpy(float)
        x2 = feature_rows[f"{method}_log_pi2"].to_numpy(float)
        feature = np.column_stack([x1, x2])
        fit = gpr_model_fit(feature, y, train_idx, test_idx)

        x_bounds = np.quantile(x1, [0.01, 0.99])
        y_bounds = np.quantile(x2, [0.01, 0.99])
        x_grid, y_grid = np.meshgrid(
            np.linspace(float(x_bounds[0]), float(x_bounds[1]), 120),
            np.linspace(float(y_bounds[0]), float(y_bounds[1]), 120),
        )
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        scaled_feature = (feature - feature.mean(axis=0)) / feature.std(axis=0, ddof=0)
        scaled_grid = (grid_points - feature.mean(axis=0)) / feature.std(axis=0, ddof=0)
        hull = Delaunay(scaled_feature)
        tree = cKDTree(scaled_feature)
        data_nn_dist = tree.query(scaled_feature, k=2)[0][:, 1]
        grid_nn_dist = tree.query(scaled_grid, k=1)[0]
        support_limit = 2.8 * float(np.quantile(data_nn_dist, 0.85))
        support_mask = (hull.find_simplex(scaled_grid) >= 0) & (grid_nn_dist <= support_limit)
        z_grid_flat = np.full(grid_points.shape[0], np.nan, dtype=float)
        z_grid_flat[support_mask] = fit["model"].predict(grid_points[support_mask])
        z_grid = z_grid_flat.reshape(x_grid.shape)
        z_grid = np.clip(z_grid, z_floor, z_max + 0.03 * (z_max - z_min))

        ax = fig.add_subplot(1, 3, index, projection="3d")
        surface = ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            rstride=1,
            cstride=1,
            cmap=surface_cmap,
            norm=surface_norm,
            alpha=0.52,
            linewidth=0,
            antialiased=True,
            shade=True,
            zorder=1,
        )
        surface.set_edgecolor("none")
        ax.scatter(
            x1,
            x2,
            y,
            s=15,
            color="#1f77b4",
            alpha=0.82,
            edgecolors="white",
            linewidths=0.18,
            depthshade=True,
            zorder=3,
        )
        row = summary[summary["method"].eq(method)].iloc[0]
        ax.text2D(
            0.04,
            0.86,
            f"GPR norm. MSE = {row['gpr_mse_normalized']:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "alpha": 0.94,
            },
        )
        ax.set_title(DISPLAY_METHOD_NAMES.get(method, method))
        ax.set_xlabel(r"$\log(\pi_1)$", labelpad=9)
        ax.set_ylabel(r"$\log(\pi_2)$", labelpad=9)
        ax.set_zlabel(r"$C_f$", labelpad=7)
        ax.set_zlim(z_floor, z_max + 0.02 * (z_max - z_min))
        ax.view_init(elev=27, azim=-52)
        ax.set_box_aspect((1.25, 1.0, 0.74))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.zaxis.set_major_locator(MaxNLocator(4))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor("#e5e7eb")
        ax.grid(True, color="#e5e7eb", linewidth=0.55, alpha=0.38)

    fig.suptitle(f"Colebrook {N_SAMPLES} points: data and GPR fitted surfaces", fontsize=24, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.07, top=0.88, wspace=0.03)
    output = base.FIG_DIR / f"{OUTPUT_PREFIX}_selected_gpr_surface_comparison.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_2d_gpr_colormaps(feature_rows: pd.DataFrame, summary: pd.DataFrame) -> Path:
    y = feature_rows["Cf"].to_numpy(float)
    train_idx, test_idx = train_test_split(
        np.arange(y.size),
        test_size=base.TEST_SIZE,
        random_state=base.RANDOM_STATE,
    )
    vmin = float(np.min(y))
    vmax = float(np.max(y))
    levels = np.linspace(vmin, vmax, 28)

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 19,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), dpi=240, constrained_layout=True)
    scatter = None
    for ax, method in zip(axes, SELECTED_METHODS):
        x1 = feature_rows[f"{method}_log_pi1"].to_numpy(float)
        x2 = feature_rows[f"{method}_log_pi2"].to_numpy(float)
        feature = np.column_stack([x1, x2])
        fit = gpr_model_fit(feature, y, train_idx, test_idx)
        fitted_at_points = fit["model"].predict(feature)
        triangulation = mtri.Triangulation(x1, x2)

        ax.tricontourf(
            triangulation,
            fitted_at_points,
            levels=levels,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=0.88,
        )
        scatter = ax.scatter(
            x1,
            x2,
            c=y,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=14,
            edgecolors="white",
            linewidths=0.25,
            alpha=0.92,
        )
        row = summary[summary["method"].eq(method)].iloc[0]
        ax.text(
            0.03,
            0.04,
            f"GPR norm. MSE = {row['gpr_mse_normalized']:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "alpha": 0.92,
            },
        )
        ax.set_title(method)
        ax.set_xlabel(r"$\log(\pi_1)$")
        ax.set_ylabel(r"$\log(\pi_2)$")
        ax.grid(False)

    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.92, pad=0.015)
    colorbar.set_label("Cf", fontsize=15)
    colorbar.ax.tick_params(labelsize=11)
    fig.suptitle(f"Colebrook {N_SAMPLES} points: 2D GPR fitted color maps", fontsize=22)
    output = base.FIG_DIR / f"{OUTPUT_PREFIX}_selected_gpr_colormap_comparison.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    configure_base()
    base.OUTPUT_DIR.mkdir(exist_ok=True)
    base.FIG_DIR.mkdir(exist_ok=True)

    summary, feature_rows, methods, exponents, joint_audit, generated = base.build_summary()
    selected_summary = (
        summary[summary["method"].isin(SELECTED_METHODS)]
        .set_index("method")
        .loc[SELECTED_METHODS]
        .reset_index()
        .copy()
    )
    selected_summary["rank_by_MI"] = selected_summary["mutual_information"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    selected_summary["rank_by_S_cov"] = selected_summary["S_cov"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    selected_summary["rank_by_gpr"] = selected_summary["gpr_mse_normalized"].rank(method="min").astype(int)
    selected_methods = [method for method in methods if method["method"] in SELECTED_METHODS]

    summary_csv = base.OUTPUT_DIR / f"{OUTPUT_PREFIX}_joint_sidl_known_logpi_summary.csv"
    all_summary_csv = base.OUTPUT_DIR / f"{OUTPUT_PREFIX}_midl_sidl_logpi_summary.csv"
    features_csv = base.OUTPUT_DIR / f"{OUTPUT_PREFIX}_logpi_coordinates.csv"
    exponents_csv = base.OUTPUT_DIR / f"{OUTPUT_PREFIX}_logpi_exponents.csv"
    audit_csv = base.OUTPUT_DIR / f"{OUTPUT_PREFIX}_joint_midl_audit.csv"
    generated_csv = base.OUTPUT_DIR / f"{OUTPUT_PREFIX}_generated_data.csv"

    selected_summary.to_csv(summary_csv, index=False)
    summary.to_csv(all_summary_csv, index=False)
    feature_rows.to_csv(features_csv, index=False)
    exponents.to_csv(exponents_csv, index=False)
    joint_audit.to_csv(audit_csv, index=False)
    generated.to_csv(generated_csv, index=False)

    base.plot_summary(selected_summary)
    base.plot_summary_table(selected_summary)
    point_figs = base.plot_logpi_data_points_3d(feature_rows, selected_methods)
    surface_fig = plot_scatter_and_gpr_surfaces(feature_rows, selected_summary)
    colormap_fig = plot_2d_gpr_colormaps(feature_rows, selected_summary)

    print(
        selected_summary[
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
    print(f"Wrote {all_summary_csv}")
    print(f"Wrote {features_csv}")
    print(f"Wrote {exponents_csv}")
    print(f"Wrote {audit_csv}")
    print(f"Wrote {generated_csv}")
    print(f"Wrote {base.OUTPUT_DIR / f'{OUTPUT_PREFIX}_logpi_metric_summary.png'}")
    print(f"Wrote {base.FIG_DIR / f'{OUTPUT_PREFIX}_summary_table.png'}")
    for point_fig in point_figs:
        print(f"Wrote {point_fig}")
    print(f"Wrote {surface_fig}")
    print(f"Wrote {colormap_fig}")


if __name__ == "__main__":
    main()
