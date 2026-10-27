#!/usr/bin/env python3
"""
Replot 1:1 ROC/PR curves from saved TSV data.

Based on one_to_one_curves_reformatted_bigger_fonts style, with:
- no figure title
- no x/y axis titles
- much larger tick (scale) labels
- no legend (no box, no legend text)
- no grid

Does not modify existing scripts or figure folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
OUT_DIR = HERE

ROC_ROOT = EVAL_ROOT / "roc_curve_doublestrand_dense_one_to_one"
PR_ROOT = EVAL_ROOT / "pr_curve_doublestrand_dense_one_to_one"

CURVE_LINEWIDTH = 8.0
TICK_FONTSIZE = 36
FIGSIZE = (10, 8)
DPI = 300
DPI_HIGH = 600
# Keep axis edges at 0–1, but do not clip the thick stroke against the box
# (otherwise the line looks thinner near the border).
PLOT_KW = dict(
    linewidth=CURVE_LINEWIDTH,
    solid_capstyle="round",
    solid_joinstyle="round",
    clip_on=False,
    zorder=4,
)


def _unclip_artists(ax: plt.Axes) -> None:
    """Allow thick strokes/markers to render past the 0–1 box without changing limits."""
    ax.set_clip_on(False)
    ax.patch.set_clip_on(False)
    for artist in list(ax.lines) + list(ax.collections) + list(ax.patches):
        artist.set_clip_on(False)
        artist.set_clip_path(None)
    for spine in ax.spines.values():
        # Spines stay at 0–1, but sit under the curve so they don't bisect the stroke.
        spine.set_clip_on(False)
        spine.set_zorder(1)


def _style_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, length=8, width=1.4, pad=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    _unclip_artists(ax)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    highres = stem.with_name(stem.name + "_highres.png")
    svg = stem.with_suffix(".svg")
    # pad_inches keeps overhanging stroke from being cropped by tight bbox
    save_kw = dict(bbox_inches="tight", pad_inches=0.25, facecolor="white")
    fig.savefig(png, dpi=DPI, **save_kw)
    fig.savefig(highres, dpi=DPI_HIGH, **save_kw)
    fig.savefig(svg, **save_kw)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {highres}")
    print(f"Saved: {svg}")


def plot_roc(tsv: Path, stem: Path) -> None:
    df = pd.read_csv(tsv, sep="\t")
    fpr = df["fpr"].to_numpy(dtype=float)
    tpr = df["tpr"].to_numpy(dtype=float)
    order = np.argsort(fpr)
    sfpr, stpr = fpr[order], tpr[order]
    try:
        auroc = float(np.trapezoid(stpr, sfpr))
    except AttributeError:
        auroc = float(np.trapz(stpr, sfpr))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sfpr, stpr, "b-", **PLOT_KW)
    ax.scatter(
        fpr, tpr, c="blue", s=20, alpha=0.3, edgecolors="none", zorder=3, clip_on=False
    )
    _style_axes(ax)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.98)
    _unclip_artists(ax)
    _save(fig, stem)
    print(f"  AUROC={auroc:.4f}")


def plot_pr(tsv: Path, stem: Path) -> None:
    df = pd.read_csv(tsv, sep="\t")
    precision = df["precision"].to_numpy(dtype=float)
    recall = df["recall"].to_numpy(dtype=float)
    threshold = df["threshold"].to_numpy(dtype=float)
    order = np.argsort(recall)
    srec, spre = recall[order], precision[order]
    try:
        auprc = float(np.trapezoid(spre, srec))
    except AttributeError:
        auprc = float(np.trapz(spre, srec))

    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    best_idx = int(np.argmax(f1))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(srec, spre, "b-", **PLOT_KW)
    ax.scatter(
        recall,
        precision,
        c="blue",
        s=20,
        alpha=0.3,
        edgecolors="none",
        zorder=3,
        clip_on=False,
    )
    ax.plot(
        recall[best_idx],
        precision[best_idx],
        "ro",
        markersize=14,
        zorder=5,
        clip_on=False,
    )
    _style_axes(ax)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.98)
    _unclip_artists(ax)
    _save(fig, stem)
    print(
        f"  AUPRC={auprc:.3f}  Max F1={f1[best_idx]:.3f} (T={threshold[best_idx]:.4f})"
    )


def main() -> None:
    jobs = [
        (
            "roc",
            ROC_ROOT / "output_no_tolerance" / "roc_curve_data.tsv",
            OUT_DIR / "roc_curve_no_tolerance",
        ),
        (
            "pr",
            PR_ROOT / "output_no_tolerance" / "pr_curve_data.tsv",
            OUT_DIR / "pr_curve_no_tolerance",
        ),
        (
            "roc",
            ROC_ROOT / "output_100bp_tolerance" / "roc_curve_data.tsv",
            OUT_DIR / "roc_curve_100bp_tolerance",
        ),
        (
            "pr",
            PR_ROOT / "output_100bp_tolerance" / "pr_curve_data.tsv",
            OUT_DIR / "pr_curve_100bp_tolerance",
        ),
    ]
    for kind, tsv, stem in jobs:
        if not tsv.is_file():
            raise SystemExit(f"Missing data: {tsv}")
        print(f"=== {stem.name} ===")
        if kind == "roc":
            plot_roc(tsv, stem)
        else:
            plot_pr(tsv, stem)
    print(f"\nOutput directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
