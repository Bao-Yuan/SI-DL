from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR.parent
SI_DIR = ROOT / "SI-DL-main"
if str(SI_DIR) not in sys.path:
    sys.path.insert(0, str(SI_DIR))

import SI_DL


FIG_DIR = OUTPUT_DIR / "figures"
N_SAMPLES = 10000
OUTPUT_PREFIX = "colebrook_10000pts_sidl_only"
GRID_N = int(os.environ.get("COLEBROOK_THETA_GRID_N", "51"))
SI_BANDWIDTH = 0.06
VARIABLE_LABELS = ["u", "rho", "D", "k_s", "mu"]


def scov_for_thetas(theta1: float, theta2: float, z1: np.ndarray, z2: np.ndarray, y: np.ndarray) -> float:
    eta1 = np.cos(theta1) * z1 + np.sin(theta1) * z2
    eta2 = np.cos(theta2) * z1 + np.sin(theta2) * z2
    feature = np.column_stack([eta1, eta2])
    score = SI_DL.explained_variance_score(
        feature,
        y,
        bandwidth=SI_BANDWIDTH,
        estimator="gaussian_kernel",
        standardize=True,
        leave_one_out=True,
        boundary="mirror",
    )
    return float(score["S_cov"])


def theta_from_omega(omega: np.ndarray) -> float:
    # eta = c log(Re) + s log(D/k_s)
    # Re contributes u,rho,D,mu exponents (c,c,c,-c);
    # D/k_s contributes D,k_s exponents (s,-s).
    c = float(omega[0])
    s = float(-omega[3])
    theta = float(np.arctan2(s, c))
    if theta < 0.0:
        theta += np.pi
    if theta >= np.pi:
        theta -= np.pi
    return theta


def de_best_thetas(exponents: pd.DataFrame) -> tuple[float, float]:
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
        rows.append(theta_from_omega(np.asarray(omega, dtype=float)))
    return float(rows[0]), float(rows[1])


def nearest_grid_score(theta1: float, theta2: float, theta_grid: np.ndarray, scov: np.ndarray) -> float:
    ix = int(np.argmin(np.abs(theta_grid - theta1)))
    iy = int(np.argmin(np.abs(theta_grid - theta2)))
    return float(scov[iy, ix])


def plot_surface(theta_grid: np.ndarray, scov: np.ndarray, markers: dict[str, tuple[float, float]]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = theta_grid / np.pi
    y = theta_grid / np.pi
    Xg, Yg = np.meshgrid(x, y)
    finite = np.isfinite(scov)
    norm = Normalize(vmin=float(np.nanmin(scov)), vmax=float(np.nanmax(scov)))

    fig = plt.figure(figsize=(14.8, 6.2), dpi=240)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(
        Xg,
        Yg,
        scov,
        cmap="viridis",
        norm=norm,
        linewidth=0,
        antialiased=True,
        alpha=0.94,
    )
    for name, (theta1, theta2) in markers.items():
        z = nearest_grid_score(theta1, theta2, theta_grid, scov)
        ax3d.scatter([theta1 / np.pi], [theta2 / np.pi], [z], s=58, depthshade=False)
    ax3d.set_xlabel(r"$\theta_1/\pi$")
    ax3d.set_ylabel(r"$\theta_2/\pi$")
    ax3d.set_zlabel(r"$S_{cov}$")
    ax3d.set_title("Two-variable SI-DL theta surface")
    ax3d.view_init(elev=28, azim=-54)
    ax3d.set_box_aspect((1.1, 1.0, 0.62))
    ax3d.set_xlim(-0.03, 1.03)
    ax3d.set_ylim(-0.03, 1.03)

    ax2d = fig.add_subplot(1, 2, 2)
    mesh = ax2d.pcolormesh(Xg, Yg, scov, cmap="viridis", norm=norm, shading="auto")
    levels = np.linspace(float(np.nanmin(scov[finite])), float(np.nanmax(scov[finite])), 14)
    ax2d.contour(Xg, Yg, scov, levels=levels, colors="white", linewidths=0.55, alpha=0.65)
    marker_styles = {
        "SI-DL 2D DE best": ("*", "#d62728", 180),
        "Known [Re, D/k_s]": ("s", "#ffbf00", 80),
        "Known swapped": ("D", "#f7f7f7", 70),
    }
    for name, (theta1, theta2) in markers.items():
        marker, color, size = marker_styles.get(name, ("o", "white", 70))
        ax2d.scatter(
            theta1 / np.pi,
            theta2 / np.pi,
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
    ax2d.set_title("Top view")
    ax2d.set_xlim(-0.03, 1.03)
    ax2d.set_ylim(-0.03, 1.03)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.legend(frameon=True, fontsize=8, loc="lower right")
    cbar = fig.colorbar(mesh, ax=ax2d, shrink=0.82, pad=0.03)
    cbar.set_label(r"$S_{cov}$")

    fig.suptitle(
        f"Colebrook {N_SAMPLES} points: 2D SI-DL S_cov over theta directions",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.035, right=0.96, bottom=0.09, top=0.90, wspace=0.20)
    output = FIG_DIR / f"{OUTPUT_PREFIX}_2d_theta_scov_surface.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    generated = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_generated_data.csv")
    exponents = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_exponents.csv")
    y = generated["Cf"].to_numpy(float)
    z1 = np.log(generated["Re"].to_numpy(float))
    z2 = np.log(1.0 / generated["relative_roughness"].to_numpy(float))

    theta_grid = np.linspace(0.0, np.pi, GRID_N)
    scov = np.empty((GRID_N, GRID_N), dtype=float)
    started = time.time()
    total = GRID_N * GRID_N
    done = 0
    for iy, theta2 in enumerate(theta_grid):
        for ix, theta1 in enumerate(theta_grid):
            scov[iy, ix] = scov_for_thetas(theta1, theta2, z1, z2, y)
            done += 1
        print(
            f"row {iy + 1:03d}/{GRID_N}, evals={done}/{total}, "
            f"elapsed={time.time() - started:.1f}s, current max={np.nanmax(scov[: iy + 1]):.8f}",
            flush=True,
        )

    de_theta1, de_theta2 = de_best_thetas(exponents)
    markers = {
        "SI-DL 2D DE best": (de_theta1, de_theta2),
        "Known [Re, D/k_s]": (0.0, 0.5 * np.pi),
        "Known swapped": (0.5 * np.pi, 0.0),
    }

    rows = []
    for iy, theta2 in enumerate(theta_grid):
        for ix, theta1 in enumerate(theta_grid):
            rows.append(
                {
                    "theta1": float(theta1),
                    "theta2": float(theta2),
                    "theta1_over_pi": float(theta1 / np.pi),
                    "theta2_over_pi": float(theta2 / np.pi),
                    "S_cov": float(scov[iy, ix]),
                }
            )
    surface_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_2d_theta_scov_surface.csv"
    pd.DataFrame(rows).to_csv(surface_csv, index=False)
    figure = plot_surface(theta_grid, scov, markers)

    flat_idx = int(np.nanargmax(scov))
    iy, ix = np.unravel_index(flat_idx, scov.shape)
    print(
        f"Grid max S_cov={scov[iy, ix]:.8f} at "
        f"theta1/pi={theta_grid[ix] / np.pi:.4f}, theta2/pi={theta_grid[iy] / np.pi:.4f}"
    )
    for name, (theta1, theta2) in markers.items():
        marker_score = scov_for_thetas(theta1, theta2, z1, z2, y)
        print(
            f"{name}: theta1/pi={theta1 / np.pi:.6f}, "
            f"theta2/pi={theta2 / np.pi:.6f}, S_cov={marker_score:.8f}"
        )
    print(f"Wrote {surface_csv}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
