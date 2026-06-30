from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
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


def make_best_coordinates(generated: pd.DataFrame, omegas: np.ndarray) -> pd.DataFrame:
    X = generated[VARIABLE_LABELS].to_numpy(float)
    eta = np.log(X) @ omegas.T
    return pd.DataFrame(
        {
            "eta1": eta[:, 0],
            "eta2": eta[:, 1],
            "Cf": generated["Cf"].to_numpy(float),
            "Re": generated["Re"].to_numpy(float),
            "relative_roughness": generated["relative_roughness"].to_numpy(float),
        }
    )


def plot_best_scatter(coords: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    eta1 = coords["eta1"].to_numpy(float)
    eta2 = coords["eta2"].to_numpy(float)
    cf = coords["Cf"].to_numpy(float)

    fig = plt.figure(figsize=(14.8, 5.8), dpi=240)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    scatter3d = ax3d.scatter(
        eta1,
        eta2,
        cf,
        c=cf,
        cmap="viridis",
        s=8,
        alpha=0.74,
        linewidths=0,
        depthshade=True,
    )
    ax3d.set_title("3D scatter at SI-DL 2D best")
    ax3d.set_xlabel(r"$\eta_1=\log(\Pi_1)$", labelpad=8)
    ax3d.set_ylabel(r"$\eta_2=\log(\Pi_2)$", labelpad=8)
    ax3d.set_zlabel(r"$C_f$", labelpad=8)
    ax3d.view_init(elev=26, azim=-52)
    ax3d.set_box_aspect((1.15, 1.0, 0.68))
    ax3d.grid(True, alpha=0.25)

    ax2d = fig.add_subplot(1, 2, 2)
    scatter2d = ax2d.scatter(
        eta1,
        eta2,
        c=cf,
        cmap="viridis",
        s=7,
        alpha=0.78,
        linewidths=0,
    )
    ax2d.set_title("Top view colored by $C_f$")
    ax2d.set_xlabel(r"$\eta_1=\log(\Pi_1)$")
    ax2d.set_ylabel(r"$\eta_2=\log(\Pi_2)$")
    ax2d.grid(True, color="#e5e7eb", linewidth=0.55, alpha=0.55)
    cbar = fig.colorbar(scatter2d, ax=ax2d, shrink=0.82, pad=0.025)
    cbar.set_label(r"$C_f$")

    fig.suptitle("Colebrook 10000 points: data in the SI-DL 2D best coordinates", fontsize=15, weight="bold")
    fig.subplots_adjust(left=0.035, right=0.95, bottom=0.10, top=0.88, wspace=0.18)
    output = FIG_DIR / f"{OUTPUT_PREFIX}_2d_best_scatter.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    generated = pd.read_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_generated_data.csv")
    omegas = load_best_omegas()
    coords = make_best_coordinates(generated, omegas)
    coords_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_2d_best_scatter_coordinates.csv"
    coords.to_csv(coords_csv, index=False)
    figure = plot_best_scatter(coords)
    print(f"Wrote {coords_csv}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
