from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.interpolate import griddata


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR.parent
SI_DIR = ROOT / "SI-DL-main"
if str(SI_DIR) not in sys.path:
    sys.path.insert(0, str(SI_DIR))

FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_PREFIX = "colebrook_10000pts_sidl_only"
VARIABLE_LABELS = ["u", "rho", "D", "k_s", "mu"]


def load_best_omegas() -> np.ndarray:
    exponents = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_exponents.csv")
    rows = []
    for pi_group in ["pi1", "pi2"]:
        subset = exponents[
            exponents["method"].eq("SI-DL 2D DE best")
            & exponents["pi_group"].eq(pi_group)
        ]
        omega = []
        for label in VARIABLE_LABELS:
            value = subset.loc[subset["variable"].eq(label), "normalized_exponent"]
            if value.empty:
                raise ValueError(f"Missing SI-DL 2D DE best {pi_group} {label}")
            omega.append(float(value.iloc[0]))
        rows.append(omega)
    return np.asarray(rows, dtype=float)


def best_scatter_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generated = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_generated_data.csv")
    omegas = load_best_omegas()
    eta = np.log(generated[VARIABLE_LABELS].to_numpy(float)) @ omegas.T
    return eta[:, 0], eta[:, 1], generated["Cf"].to_numpy(float)


def theta_from_omega(omega: np.ndarray) -> float:
    c = float(omega[0])
    s = float(-omega[3])
    theta = float(np.arctan2(s, c))
    if theta < 0.0:
        theta += np.pi
    if theta >= np.pi:
        theta -= np.pi
    return theta


def de_best_thetas() -> tuple[float, float]:
    omegas = load_best_omegas()
    return theta_from_omega(omegas[0]), theta_from_omega(omegas[1])


def theta_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_2d_theta_scov_surface.csv")
    diagonal = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_2d_theta_diagonal_band.csv")
    surface = pd.concat(
        [
            coarse[["theta1_over_pi", "theta2_over_pi", "S_cov"]],
            diagonal[["theta1_over_pi", "theta2_over_pi", "S_cov"]],
        ],
        ignore_index=True,
    )
    valid = surface[
        surface["S_cov"].notna()
        & surface["theta1_over_pi"].between(0.0, 1.0)
        & surface["theta2_over_pi"].between(0.0, 1.0)
    ].copy()
    valid = valid.drop_duplicates(["theta1_over_pi", "theta2_over_pi"], keep="last")
    return (
        valid["theta1_over_pi"].to_numpy(float),
        valid["theta2_over_pi"].to_numpy(float),
        valid["S_cov"].to_numpy(float),
    )


def theta_markers() -> dict[str, tuple[float, float]]:
    de_theta1, de_theta2 = de_best_thetas()
    return {
        "SI-DL 2D best": (de_theta1 / np.pi, de_theta2 / np.pi),
        "Known [Re, D/k_s]": (0.0, 0.5),
    }


def plot_combined() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    eta1, eta2, cf = best_scatter_coordinates()
    theta1, theta2, scov = theta_surface()

    plt.rcParams.update(
        {
            "font.size": 15,
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.titlesize": 21,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
        }
    )

    fig = plt.figure(figsize=(14.8, 6.2), dpi=480)
    ax2d = fig.add_axes([0.050, 0.105, 0.405, 0.865])
    norm = Normalize(vmin=float(np.nanmin(scov)), vmax=float(np.nanmax(scov)))
    grid_n = 520
    theta1_grid, theta2_grid = np.meshgrid(
        np.linspace(0.0, 1.0, grid_n),
        np.linspace(0.0, 1.0, grid_n),
    )
    scov_smooth = griddata(
        np.column_stack([theta1, theta2]),
        scov,
        (theta1_grid, theta2_grid),
        method="linear",
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    mesh = ax2d.imshow(
        np.ma.masked_invalid(scov_smooth),
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        cmap=cmap,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
    )
    ax2d.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.6, linestyle="--", alpha=0.95, label=r"$\theta_1=\theta_2$")
    marker_styles = {
        "SI-DL 2D best": ("*", "#d62728", 180),
        "Known [Re, D/k_s]": ("s", "#ffbf00", 80),
    }
    for name, (x, y) in theta_markers().items():
        marker, color, size = marker_styles.get(name, ("o", "white", 70))
        ax2d.scatter(
            x,
            y,
            marker=marker,
            s=size,
            c=color,
            edgecolors="black",
            linewidths=0.8,
            label=name,
            zorder=4,
        )
    ax2d.set_xlabel(r"$\theta_1/\pi$")
    ax2d.set_ylabel(r"$\theta_2/\pi$")
    ax2d.set_xlim(-0.03, 1.03)
    ax2d.set_ylim(-0.03, 1.03)
    ax2d.legend(frameon=True, loc="lower right")

    ax3d = fig.add_axes([0.535, 0.030, 0.455, 0.950], projection="3d")
    scatter3d = ax3d.scatter(
        eta1,
        eta2,
        cf,
        color="#123f6d",
        s=10,
        alpha=0.72,
        linewidths=0,
        depthshade=True,
    )
    ax3d.set_xlabel(r"$\log(\Pi_1)$", labelpad=9)
    ax3d.set_ylabel(r"$\log(\Pi_2)$", labelpad=9)
    ax3d.set_zlabel(r"$C_f$", labelpad=9)
    ax3d.view_init(elev=26, azim=-52)
    ax3d.set_box_aspect((1.15, 1.0, 0.68))
    ax3d.grid(True, alpha=0.25)
    ax3d.tick_params(axis="both", labelsize=13)

    cbar_ax = fig.add_axes([0.468, 0.105, 0.022, 0.865])
    cbar_scov = fig.colorbar(mesh, cax=cbar_ax)
    cbar_scov.set_label("Sobol index", fontsize=18)
    cbar_scov.ax.tick_params(labelsize=14)
    output = FIG_DIR / f"{OUTPUT_PREFIX}_combined_best_scatter_theta_topview.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    figure = plot_combined()
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
