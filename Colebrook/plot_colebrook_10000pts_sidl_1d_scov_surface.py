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
GRID_N = int(os.environ.get("COLEBROOK_SURFACE_GRID_N", "61"))
GRID_LIMIT = float(os.environ.get("COLEBROOK_SURFACE_GRID_LIMIT", "1.15"))
SI_BANDWIDTH = 0.06
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


def params_to_omega(params: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return normalize_exponents(np.asarray(params, dtype=float).reshape(2) @ basis)


def log_feature_from_omega(X: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(X, dtype=float)) @ np.asarray(omega, dtype=float).reshape(-1, 1)


def scov_for_params(params: np.ndarray, basis: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    if np.linalg.norm(params) <= 1e-12:
        return np.nan
    omega = params_to_omega(params, basis)
    feature = log_feature_from_omega(X, omega)
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


def params_from_exponent_row(exponents: pd.DataFrame, method: str, basis: np.ndarray) -> np.ndarray:
    row = []
    subset = exponents[(exponents["method"].eq(method)) & (exponents["pi_group"].eq("pi1"))]
    for label in VARIABLE_LABELS:
        value = subset.loc[subset["variable"].eq(label), "normalized_exponent"]
        if value.empty:
            raise ValueError(f"Missing exponent for {method} pi1 {label}")
        row.append(float(value.iloc[0]))
    params, *_ = np.linalg.lstsq(basis.T, np.asarray(row, dtype=float), rcond=None)
    return np.asarray(params, dtype=float)


def plot_surface(grid_x: np.ndarray, grid_y: np.ndarray, scov: np.ndarray, markers: dict[str, np.ndarray]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Xg, Yg = np.meshgrid(grid_x, grid_y)
    finite = np.isfinite(scov)
    norm = Normalize(vmin=float(np.nanmin(scov)), vmax=float(np.nanmax(scov)))

    fig = plt.figure(figsize=(14.8, 6.2), dpi=240)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(
        Xg,
        Yg,
        scov,
        cmap="viridis",
        norm=norm,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    for name, xy in markers.items():
        z = scov_for_marker(xy, grid_x, grid_y, scov)
        ax3d.scatter([xy[0]], [xy[1]], [z], s=58, label=name, depthshade=False)
        ax3d.text(xy[0], xy[1], z, f" {name}", fontsize=8)
    ax3d.set_xlabel("basis direction 1 coefficient")
    ax3d.set_ylabel("basis direction 2 coefficient")
    ax3d.set_zlabel(r"$S_{cov}$")
    ax3d.set_title("SI-DL 1D search surface")
    ax3d.view_init(elev=28, azim=-54)
    ax3d.set_box_aspect((1.1, 1.0, 0.62))

    ax2d = fig.add_subplot(1, 2, 2)
    mesh = ax2d.pcolormesh(Xg, Yg, scov, cmap="viridis", norm=norm, shading="auto")
    contour_levels = np.linspace(float(np.nanmin(scov[finite])), float(np.nanmax(scov[finite])), 12)
    ax2d.contour(Xg, Yg, scov, levels=contour_levels, colors="white", linewidths=0.55, alpha=0.65)
    marker_styles = {
        "SI-DL 1D best": ("*", "#d62728", 180),
        "Known D/k_s": ("s", "#ffbf00", 70),
        "Known Re": ("o", "#f7f7f7", 70),
    }
    for name, xy in markers.items():
        marker, color, size = marker_styles.get(name, ("o", "white", 70))
        ax2d.scatter(
            xy[0],
            xy[1],
            marker=marker,
            s=size,
            c=color,
            edgecolors="black",
            linewidths=0.8,
            label=name,
            zorder=4,
        )
    ax2d.set_xlabel("basis direction 1 coefficient")
    ax2d.set_ylabel("basis direction 2 coefficient")
    ax2d.set_title("Top view")
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.legend(frameon=True, fontsize=8, loc="upper right")
    cbar = fig.colorbar(mesh, ax=ax2d, shrink=0.82, pad=0.03)
    cbar.set_label(r"$S_{cov}$")

    fig.suptitle(
        f"Colebrook {N_SAMPLES} points: 1D SI-DL S_cov over the two search directions",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.035, right=0.96, bottom=0.09, top=0.90, wspace=0.20)
    output = FIG_DIR / f"{OUTPUT_PREFIX}_1d_scov_search_surface.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def scov_for_marker(xy: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, scov: np.ndarray) -> float:
    ix = int(np.argmin(np.abs(grid_x - xy[0])))
    iy = int(np.argmin(np.abs(grid_y - xy[1])))
    return float(scov[iy, ix])


def main() -> None:
    generated = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_generated_data.csv")
    exponents = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_exponents.csv")
    X = generated[VARIABLE_LABELS].to_numpy(float)
    y = generated["Cf"].to_numpy(float)
    basis = np.asarray(SI_DL.calc_basis(D_IN, 2), dtype=float)

    grid = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_N)
    scov = np.empty((GRID_N, GRID_N), dtype=float)
    started = time.time()
    total = GRID_N * GRID_N
    done = 0
    for iy, ycoef in enumerate(grid):
        for ix, xcoef in enumerate(grid):
            scov[iy, ix] = scov_for_params(np.array([xcoef, ycoef]), basis, X, y)
            done += 1
        print(
            f"row {iy + 1:03d}/{GRID_N}, evals={done}/{total}, "
            f"elapsed={time.time() - started:.1f}s, current max={np.nanmax(scov[: iy + 1]):.8f}",
            flush=True,
        )

    best_params = params_from_exponent_row(exponents, "SI-DL 1D DE best", basis)
    markers = {
        "SI-DL 1D best": best_params,
        "Known D/k_s": np.array([1.0, 0.0]),
        "Known Re": np.array([0.0, 1.0]),
    }

    surface_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_1d_scov_search_surface.csv"
    rows = []
    for iy, ycoef in enumerate(grid):
        for ix, xcoef in enumerate(grid):
            rows.append(
                {
                    "basis_direction_1_coef": float(xcoef),
                    "basis_direction_2_coef": float(ycoef),
                    "S_cov": float(scov[iy, ix]),
                }
            )
    pd.DataFrame(rows).to_csv(surface_csv, index=False)
    figure = plot_surface(grid, grid, scov, markers)

    flat_idx = int(np.nanargmax(scov))
    iy, ix = np.unravel_index(flat_idx, scov.shape)
    print(
        f"Grid max S_cov={scov[iy, ix]:.8f} at "
        f"basis1={grid[ix]:.4f}, basis2={grid[iy]:.4f}"
    )
    for name, xy in markers.items():
        marker_score = scov_for_params(xy, basis, X, y)
        print(f"{name}: basis1={xy[0]:.6f}, basis2={xy[1]:.6f}, S_cov={marker_score:.8f}")
    print(f"Wrote {surface_csv}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
