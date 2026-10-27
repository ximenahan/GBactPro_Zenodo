#!/usr/bin/env python3
"""
Combine four TSS-class Precision@k (log-x) curves into one figure (100 bp).

Reads existing per-class precision_at_k_full.tsv under tss_class_subsets_pr_roc.
Does not modify original scripts or outputs.
Legend: four TSS classes only (no overall-precision baseline).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CLASSES = ("primary", "secondary", "internal", "antisense")
CLASS_LABELS = {
    "primary": "Primary",
    "secondary": "Secondary",
    "internal": "Internal",
    "antisense": "Antisense",
}
# Okabe–Ito (CVD / red–green colorblind safe)
CLASS_COLORS = {
    "primary": "#0072B2",
    "secondary": "#E69F00",
    "internal": "#009E73",
    "antisense": "#CC79A7",
}

FIGSIZE = (10, 7.5)
DPI = 300
DPI_HIGH = 600
LINEWIDTH = 4.5
AXIS_LABEL_SIZE = 28
TICK_SIZE = 20
LEGEND_FONTSIZE = 20
PLOT_KW = dict(
    linewidth=LINEWIDTH,
    solid_capstyle="round",
    solid_joinstyle="round",
    # Keep ylim at 0–1, but do not clip the stroke at the box edge
    # (otherwise segments at 0.0 / 1.0 look thinner).
    clip_on=False,
    zorder=3,
)


def _setup_font() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def class_tsv(class_name: str) -> Path:
    return (
        ROOT
        / class_name
        / "precision_at_k_best_f1_one_to_one"
        / "output_100bp_tolerance"
        / "precision_at_k_full.tsv"
    )


def main() -> None:
    _setup_font()
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.set_facecolor("white")

    k_max = 1
    for class_name in CLASSES:
        tsv = class_tsv(class_name)
        if not tsv.is_file():
            raise SystemExit(f"Missing: {tsv}")
        df = pd.read_csv(tsv, sep="\t")
        k_vals = df["rank_cutoff_k"].to_numpy()
        k_max = max(k_max, int(k_vals.max()))
        ax.plot(
            k_vals,
            df["precision_at_k"].to_numpy(),
            color=CLASS_COLORS[class_name],
            label=CLASS_LABELS[class_name],
            **PLOT_KW,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Rank cutoff (k)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Precision at k", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("")
    ax.set_ylim(0.0, 1.0)
    # Start at first rank (k=1 = 10^0); no blank margin before it
    ax.set_xlim(1, k_max)
    ax.margins(x=0)
    # Decade labels only; no minor tick marks between decades
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, length=6, width=1.2)
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="minor", length=4, width=1.0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        sp = ax.spines[spine_name]
        sp.set_linewidth(1.2)
        # Keep spines under the curves so they don't bisect the thick stroke
        sp.set_zorder(1)

    for line in ax.lines:
        line.set_clip_on(False)
        line.set_clip_path(None)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.4,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.14, right=0.98)
    stem = HERE / "precision_at_k_vs_rank_logx_four_classes"
    # pad so overhanging stroke at y=0/1 is not cropped by tight bbox
    save_kw = dict(bbox_inches="tight", pad_inches=0.28, facecolor="white")
    png = stem.with_suffix(".png")
    highres = stem.with_name(stem.name + "_highres.png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=DPI, **save_kw)
    fig.savefig(highres, dpi=DPI_HIGH, **save_kw)
    fig.savefig(svg, **save_kw)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {highres}")
    print(f"Saved: {svg}")


if __name__ == "__main__":
    main()
