#!/usr/bin/env python3
"""
Reformat H-NS overlap bar chart (publication style).

Reads counts from the existing no_tolerance/bar_chart summary TSV.
Does not modify the original bar_chart scripts or outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUMMARY_TSV = HERE.parent / "bar_chart" / "hns_overlap_summary.tsv"
OUT_DIR = HERE

# Colors sampled from the attached reference stacked-bar figure
COLOR_HNS = "#0D5097"  # dark blue — H-NS overlap
COLOR_NON_HNS = "#F0F4FF"  # pale blue — no overlap

FIGSIZE = (7.5, 8.0)
DPI = 300
DPI_HIGH = 600
BAR_WIDTH = 0.78
# Closer category centers → less gap between bars
X_POSITIONS = np.array([0.0, 0.92, 1.84])
AXIS_LABEL_SIZE = 30
X_TICK_LABEL_SIZE = 28
Y_TICK_LABEL_SIZE = 22
PCT_FONTSIZE = 20
LEGEND_FONTSIZE = 18


def _setup_font() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot(summary: pd.DataFrame, stem: Path) -> None:
    categories = ["TP", "FP", "FN"]
    # Keep TP / FP / FN order
    summary = summary.set_index("category").loc[categories].reset_index()
    counts = summary["count"].to_numpy(dtype=float)
    hns_counts = summary["hns_overlap_count"].to_numpy(dtype=float)
    non_hns_counts = counts - hns_counts
    hns_pct = hns_counts / counts * 100.0
    non_hns_pct = non_hns_counts / counts * 100.0

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.set_facecolor("white")
    x = X_POSITIONS

    ax.bar(
        x,
        hns_counts,
        BAR_WIDTH,
        label="H-NS+ (overlap)",
        color=COLOR_HNS,
        edgecolor="black",
        linewidth=1.0,
        zorder=2,
    )
    ax.bar(
        x,
        non_hns_counts,
        BAR_WIDTH,
        bottom=hns_counts,
        label="H-NS- (no overlap)",
        color=COLOR_NON_HNS,
        edgecolor="black",
        linewidth=1.0,
        zorder=2,
    )

    # Same % font size on all labels (H-NS+ and H-NS-)
    for i in range(len(categories)):
        ax.text(
            x[i],
            hns_counts[i] / 2.0,
            f"{hns_pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=PCT_FONTSIZE,
            fontweight="bold",
            color="white",
            zorder=3,
        )
        ax.text(
            x[i],
            hns_counts[i] + non_hns_counts[i] / 2.0,
            f"{non_hns_pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=PCT_FONTSIZE,
            fontweight="bold",
            color="#1a1a1a",
            zorder=3,
        )

    ax.set_ylabel("Number of predictions", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("")
    ax.set_title("")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=X_TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE, length=6, width=1.2)
    ax.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE, length=6, width=1.2, pad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format(int(v), ",")))

    ymax = float(counts.max()) if counts.max() > 0 else 1.0
    ax.set_ylim(0, ymax * 1.06)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("black")

    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.8,
        borderaxespad=0.0,
    )
    for text in leg.get_texts():
        text.set_fontsize(LEGEND_FONTSIZE)

    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.16, right=0.98)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    highres = stem.with_name(stem.name + "_highres.png")
    svg = stem.with_suffix(".svg")
    save_kw = dict(bbox_inches="tight", pad_inches=0.15, facecolor="white")
    fig.savefig(png, dpi=DPI, **save_kw)
    fig.savefig(highres, dpi=DPI_HIGH, **save_kw)
    fig.savefig(svg, **save_kw)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {highres}")
    print(f"Saved: {svg}")


def main() -> None:
    _setup_font()
    if not SUMMARY_TSV.is_file():
        raise SystemExit(f"Missing summary TSV: {SUMMARY_TSV}")
    summary = pd.read_csv(SUMMARY_TSV, sep="\t")
    plot(summary, OUT_DIR / "hns_overlap_bar_chart")


if __name__ == "__main__":
    main()
