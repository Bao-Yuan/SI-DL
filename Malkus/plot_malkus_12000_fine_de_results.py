from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_PREFIX = "malkus_sidl_12000_fine_de"
SUMMARY_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"
LOG_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_de_log.csv"
FIGURE = FIG_DIR / f"{OUTPUT_PREFIX}_summary_convergence.png"


def wrap_formula(formula: str) -> str:
    wrapped_lines = []
    for line in str(formula).splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=78, subsequent_indent="  "))
    return "\n".join(wrapped_lines)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_CSV)
    log = pd.read_csv(LOG_CSV)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig = plt.figure(figsize=(15.5, 8.2), dpi=300)
    table_ax = fig.add_axes([0.025, 0.49, 0.95, 0.46])
    table_ax.axis("off")
    curve_ax = fig.add_axes([0.08, 0.09, 0.86, 0.31])

    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                row["method"],
                f"{row['S_cov']:.6f}",
                f"{row['S_cov_raw']:.6f}",
                f"{row['nrmse']:.6f}",
                f"{row['corr_Y_mhat']:.6f}",
                f"{row['mutual_information']:.6f}",
                wrap_formula(row["formula"]),
            ]
        )

    table = table_ax.table(
        cellText=rows,
        colLabels=[
            "Method",
            "S_cov",
            "Raw S_cov",
            "NRMSE",
            "corr",
            "MI",
            "Pi groups",
        ],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.13, 0.07, 0.08, 0.075, 0.07, 0.075, 0.50],
        bbox=[0.0, 0.0, 1.0, 0.88],
    )
    table.auto_set_font_size(False)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#404040")
        cell.set_linewidth(0.75)
        cell.PAD = 0.018
        text = cell.get_text()
        if r == 0:
            cell.set_facecolor("#1f2937")
            text.set_color("white")
            text.set_weight("bold")
            text.set_fontsize(9.5)
        else:
            cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
            text.set_fontsize(7.0 if c == 6 else 9.0)
            if c == 6:
                text.set_ha("left")
            if c == 0:
                text.set_weight("bold")

    table_ax.text(
        0.5,
        0.965,
        "Malkus waterwheel SI-DL fine DE, 12,000 points, bandwidth=0.1, mirror boundary",
        ha="center",
        va="top",
        fontsize=17,
        weight="bold",
    )

    curve_ax.plot(log["iteration"], log["best_S_cov"], color="#123f6d", linewidth=2.2)
    curve_ax.scatter(log["iteration"], log["best_S_cov"], color="#123f6d", s=16, zorder=3)
    curve_ax.set_xlabel("DE iteration")
    curve_ax.set_ylabel("Best Sobol index")
    curve_ax.set_xlim(float(log["iteration"].min()), float(log["iteration"].max()))
    curve_ax.grid(True, alpha=0.25)
    curve_ax.spines["top"].set_visible(False)
    curve_ax.spines["right"].set_visible(False)

    fig.savefig(FIGURE, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
