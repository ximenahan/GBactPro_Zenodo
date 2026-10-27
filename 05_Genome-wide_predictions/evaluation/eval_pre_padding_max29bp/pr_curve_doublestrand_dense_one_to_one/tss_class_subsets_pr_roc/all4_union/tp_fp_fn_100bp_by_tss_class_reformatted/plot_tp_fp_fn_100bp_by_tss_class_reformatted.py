#!/usr/bin/env python3
"""
Reformat TP/FP/FN (±100 bp) four-class chart, grouped by TSS class.

Reads counts from best_f1_tp_fp_fn_all_classes.tsv.
Does not modify original all4_union scripts or outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLE_PATH = HERE.parent.parent / "best_f1_tp_fp_fn_all_classes.tsv"
OUT_DIR = HERE

CLASS_ORDER = ["primary", "secondary", "internal", "antisense"]
CLASS_LABELS = {
    "primary": "Primary",
    "secondary": "Secondary",
    "internal": "Internal",
    "antisense": "Antisense",
}
METRICS = ["TP", "FP", "FN"]
# CVD / red–green colorblind safe (teal, coral-orange, periwinkle)
METRIC_COLORS = {
    "TP": "#66C2A5",
    "FP": "#FC8D62",
    "FN": "#8DA0CB",
}

FIGSIZE = (9.5, 7.5)
DPI = 300
DPI_HIGH = 600
# Closer class centers → less gap between groups
X_POSITIONS = np.array([0.0, 0.82, 1.64, 2.46])
BAR_WIDTH = 0.24
Y_LABEL_SIZE = 28
X_TICK_SIZE = 24
Y_TICK_SIZE = 20
LEGEND_FONTSIZE = 24


def _setup_font() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_counts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    subset = df[df["pr_condition"] == "100bp_tolerance"].copy()
    subset = subset[subset["tss_class"].isin(CLASS_ORDER)].copy()
    if subset.empty:
        raise ValueError("No rows found for pr_condition=100bp_tolerance.")
    subset["tss_class"] = pd.Categorical(subset["tss_class"], categories=CLASS_ORDER, ordered=True)
    return subset.sort_values("tss_class")


def plot(df: pd.DataFrame, stem: Path) -> None:
    x = X_POSITIONS
    metric_offsets = np.linspace(-BAR_WIDTH, BAR_WIDTH, len(METRICS))

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.set_facecolor("white")

    for metric, offset in zip(METRICS, metric_offsets):
        values = np.array(
            [df[df["tss_class"] == class_name].iloc[0][metric] for class_name in CLASS_ORDER],
            dtype=float,
        )
        ax.bar(
            x + offset,
            values,
            width=BAR_WIDTH,
            color=METRIC_COLORS[metric],
            label=metric,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER], fontsize=X_TICK_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("Number of predictions", fontsize=Y_LABEL_SIZE)
    ax.set_title("")
    ax.tick_params(axis="x", labelsize=X_TICK_SIZE, length=6, width=1.2, pad=8)
    ax.tick_params(axis="y", labelsize=Y_TICK_SIZE, length=6, width=1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format(int(v), ",")))

    ax.set_ylim(0, 12000)
    ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
    ax.set_xlim(x[0] - 0.48, x[-1] + 0.48)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("black")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.4,
        handleheight=1.2,
        handletextpad=0.55,
        columnspacing=1.8,
        borderaxespad=0.0,
        markerscale=1.3,
    )

    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.14, right=0.98)
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
    if not TABLE_PATH.is_file():
        raise SystemExit(f"Missing table: {TABLE_PATH}")
    df = load_counts(TABLE_PATH)
    plot(df, OUT_DIR / "tp_fp_fn_100bp_tolerance_four_classes_by_tss_class")


if __name__ == "__main__":
    main()
