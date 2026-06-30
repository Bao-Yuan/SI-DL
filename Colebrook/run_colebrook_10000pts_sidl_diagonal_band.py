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
OUTPUT_PREFIX = "colebrook_10000pts_sidl_only"
CENTER_N = int(os.environ.get("COLEBROOK_DIAGONAL_CENTER_N", "121"))
OFFSET_N = int(os.environ.get("COLEBROOK_DIAGONAL_OFFSET_N", "31"))
OFFSET_LIMIT = float(os.environ.get("COLEBROOK_DIAGONAL_OFFSET_LIMIT", "0.075"))
SI_BANDWIDTH = 0.06


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


def plot_diagonal_band(results: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pivot = results.pivot(index="delta_over_pi", columns="theta1_over_pi", values="S_cov")
    y_vals = pivot.index.to_numpy(float)
    x_vals = pivot.columns.to_numpy(float)
    z = pivot.to_numpy(float)
    norm = Normalize(vmin=float(np.nanmin(z)), vmax=float(np.nanmax(z)))

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=480)
    mesh = ax.imshow(
        z,
        origin="lower",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        cmap="viridis",
        norm=norm,
        aspect="auto",
        interpolation="bicubic",
    )
    ax.axhline(0.0, color="white", linewidth=1.5, linestyle="--", alpha=0.95, label=r"$\theta_1=\theta_2$")
    ax.set_xlabel(r"$\theta_1/\pi$")
    ax.set_ylabel(r"$(\theta_2-\theta_1)/\pi$")
    ax.set_xlim(float(x_vals.min()), float(x_vals.max()))
    ax.set_ylim(float(y_vals.min()), float(y_vals.max()))
    ax.legend(frameon=True, loc="lower right")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    cbar.set_label("Sobol index")
    fig.subplots_adjust(left=0.13, right=0.93, bottom=0.15, top=0.97)
    output = FIG_DIR / f"{OUTPUT_PREFIX}_2d_theta_diagonal_band.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    generated = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_generated_data.csv")
    y = generated["Cf"].to_numpy(float)
    z1 = np.log(generated["Re"].to_numpy(float))
    z2 = np.log(1.0 / generated["relative_roughness"].to_numpy(float))

    theta1_grid = np.linspace(0.0, 1.0, CENTER_N)
    delta_grid = np.linspace(-OFFSET_LIMIT, OFFSET_LIMIT, OFFSET_N)
    rows = []
    started = time.time()
    total = CENTER_N * OFFSET_N
    done = 0
    for j, delta_over_pi in enumerate(delta_grid):
        for theta1_over_pi in theta1_grid:
            theta2_over_pi = theta1_over_pi + delta_over_pi
            if theta2_over_pi < 0.0 or theta2_over_pi > 1.0:
                score = np.nan
            else:
                score = scov_for_thetas(
                    theta1_over_pi * np.pi,
                    theta2_over_pi * np.pi,
                    z1,
                    z2,
                    y,
                )
            rows.append(
                {
                    "theta1_over_pi": float(theta1_over_pi),
                    "theta2_over_pi": float(theta2_over_pi),
                    "delta_over_pi": float(delta_over_pi),
                    "S_cov": float(score) if np.isfinite(score) else np.nan,
                }
            )
            done += 1
        partial = pd.DataFrame(rows)
        print(
            f"offset row {j + 1:03d}/{OFFSET_N}, evals={done}/{total}, "
            f"elapsed={time.time() - started:.1f}s, current max={partial['S_cov'].max(skipna=True):.8f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    output_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_2d_theta_diagonal_band.csv"
    results.to_csv(output_csv, index=False)
    figure = plot_diagonal_band(results)

    best = results.loc[results["S_cov"].idxmax()]
    diagonal = results[np.isclose(results["delta_over_pi"], 0.0)]
    print(
        f"Band max S_cov={best['S_cov']:.8f} at "
        f"theta1/pi={best['theta1_over_pi']:.6f}, theta2/pi={best['theta2_over_pi']:.6f}, "
        f"delta/pi={best['delta_over_pi']:.6f}"
    )
    print(
        f"Diagonal max S_cov={diagonal['S_cov'].max():.8f}, "
        f"diagonal min S_cov={diagonal['S_cov'].min():.8f}"
    )
    print(f"Wrote {output_csv}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
