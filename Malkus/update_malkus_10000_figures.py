from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUTPUT_DIR / "figures"
SUMMARY_CSV = OUTPUT_DIR / "malkus_sidl_10000_single_set_mirror_summary.csv"
EXPONENTS_CSV = OUTPUT_DIR / "malkus_sidl_10000_single_set_mirror_exponents.csv"
MAIN_SUMMARY_CSV = OUTPUT_DIR / "malkus_k6_summary.csv"
MAIN_EXPONENTS_CSV = OUTPUT_DIR / "malkus_k6_exponents.csv"
SUMMARY_FIG = FIG_DIR / "malkus_k6_summary_table.png"


def wrap_formula(formula: str) -> str:
    wrapped_lines = []
    for line in formula.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=74, subsequent_indent="  "))
    return "\n".join(wrapped_lines)


def plot_summary_table(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18.5, 5.0), dpi=240)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.95,
        "Malkus SI-DL, 10,000 points, boundary=mirror, bandwidth=0.1",
        ha="center",
        va="top",
        fontsize=17,
        weight="bold",
    )

    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                row["method"],
                f"{row['S_cov']:.6f}",
                f"{row['S_cov_raw']:.6f}",
                f"{row['sidl_error_clipped']:.6f}",
                f"{row['nrmse']:.6f}",
                f"{row['corr_Y_mhat']:.6f}",
                f"{row['mutual_information']:.6f}",
                wrap_formula(row["formula"]),
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Method",
            "S_cov",
            "S_cov_raw",
            "1-S_cov",
            "NRMSE",
            "corr",
            "MI",
            "Pi groups",
        ],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.12, 0.07, 0.08, 0.075, 0.075, 0.07, 0.07, 0.44],
        bbox=[0.015, 0.04, 0.97, 0.80],
    )
    table.auto_set_font_size(False)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#404040")
        cell.set_linewidth(0.8)
        cell.PAD = 0.02
        text = cell.get_text()
        if r == 0:
            cell.set_facecolor("#1f2937")
            text.set_color("white")
            text.set_weight("bold")
            text.set_fontsize(9.0)
        else:
            cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
            text.set_fontsize(7.1 if c == 7 else 8.6)
            if c == 7:
                text.set_ha("left")
            if c == 0:
                text.set_weight("bold")

    fig.savefig(SUMMARY_FIG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    exponents = pd.read_csv(EXPONENTS_CSV)
    summary.to_csv(MAIN_SUMMARY_CSV, index=False)
    exponents.to_csv(MAIN_EXPONENTS_CSV, index=False)
    plot_summary_table(summary)
    print(f"Wrote {MAIN_SUMMARY_CSV}")
    print(f"Wrote {MAIN_EXPONENTS_CSV}")
    print(f"Wrote {SUMMARY_FIG}")


if __name__ == "__main__":
    main()
