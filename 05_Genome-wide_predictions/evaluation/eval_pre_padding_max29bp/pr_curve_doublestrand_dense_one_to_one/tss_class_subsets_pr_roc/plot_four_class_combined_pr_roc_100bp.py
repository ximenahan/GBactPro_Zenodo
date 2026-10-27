#!/usr/bin/env python3
"""Overlay PR and ROC curves for four TSS classes (100 bp tolerance), thick lines, large legend."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_ROOT = Path(__file__).resolve().parent
OUT_DIR = RUN_ROOT / "four_tss_class_auprc_auroc_results" / "100bp_tolerance" / "combined_four_class_curves"
PR_SUB = "pr_output_100bp_tolerance"
ROC_SUB = "roc_output_100bp_tolerance"

CLASSES = ("primary", "secondary", "internal", "antisense")
DISPLAY = ("Primary", "Secondary", "Internal", "Antisense")
COLORS = ("#0173B2", "#DE8F05", "#029E73", "#CC78BC")
LINEWIDTH = 6.0
LEGEND_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 22
TICK_FONTSIZE = 16


def parse_auroc(path: Path) -> float:
    text = path.read_text()
    m = re.search(r"^AUROC:\s*([0-9.]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else float("nan")


def auprc_from_pr(df: pd.DataFrame) -> float:
    rec = df["recall"].to_numpy(dtype=float)
    pre = df["precision"].to_numpy(dtype=float)
    order = np.argsort(rec)
    rec_s, pre_s = rec[order], pre[order]
    try:
        return float(np.trapezoid(pre_s, rec_s))
    except AttributeError:
        return float(np.trapz(pre_s, rec_s))


def sort_pr(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rec = df["recall"].to_numpy(dtype=float)
    pre = df["precision"].to_numpy(dtype=float)
    order = np.argsort(rec)
    return rec[order], pre[order]


def sort_roc(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fpr = df["fpr"].to_numpy(dtype=float)
    tpr = df["tpr"].to_numpy(dtype=float)
    order = np.argsort(fpr)
    return fpr[order], tpr[order]


def legend_kw(loc: str) -> dict:
    return {
        "loc": loc,
        "fontsize": LEGEND_FONTSIZE,
        "frameon": True,
        "fancybox": True,
        "shadow": True,
        "borderpad": 1.6,
        "labelspacing": 1.15,
        "handlelength": 5.0,
        "handletextpad": 1.1,
        "borderaxespad": 1.0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_pr, ax_pr = plt.subplots(figsize=(12, 11))
    fig_roc, ax_roc = plt.subplots(figsize=(12, 11))

    for cls, label, color in zip(CLASSES, DISPLAY, COLORS):
        pr_path = RUN_ROOT / cls / PR_SUB / "pr_curve_data.tsv"
        roc_path = RUN_ROOT / cls / ROC_SUB / "roc_curve_data.tsv"
        roc_sum = RUN_ROOT / cls / ROC_SUB / "auroc_summary.txt"
        pr_df = pd.read_csv(pr_path, sep="\t")
        roc_df = pd.read_csv(roc_path, sep="\t")
        sr, sp = sort_pr(pr_df)
        sf, st = sort_roc(roc_df)
        auprc = auprc_from_pr(pr_df)
        auroc = parse_auroc(roc_sum)

        ax_pr.plot(
            sr,
            sp,
            color=color,
            linewidth=LINEWIDTH,
            label=f"{label} (AUPRC = {auprc:.4f})",
            zorder=3,
        )
        ax_roc.plot(
            sf,
            st,
            color=color,
            linewidth=LINEWIDTH,
            label=f"{label} (AUROC = {auroc:.4f})",
            zorder=3,
        )

    ax_pr.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE)
    ax_pr.set_ylabel("Precision", fontsize=AXIS_LABEL_FONTSIZE)
    ax_pr.set_title(
        "Precision–recall (100 bp tolerance)\nFour TSS classes — dense double-strand one-to-one",
        fontsize=TITLE_FONTSIZE,
    )
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.grid(True, alpha=0.35)
    ax_pr.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_pr.legend(**legend_kw("upper right"))

    ax_roc.set_xlabel("False positive rate", fontsize=AXIS_LABEL_FONTSIZE)
    ax_roc.set_ylabel("True positive rate", fontsize=AXIS_LABEL_FONTSIZE)
    ax_roc.set_title(
        "ROC (100 bp tolerance)\nFour TSS classes — dense double-strand one-to-one",
        fontsize=TITLE_FONTSIZE,
    )
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.grid(True, alpha=0.35)
    ax_roc.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_roc.legend(**legend_kw("lower right"))

    fig_pr.tight_layout()
    fig_roc.tight_layout()

    pr_base = OUT_DIR / "pr_curve_four_classes_combined"
    roc_base = OUT_DIR / "roc_curve_four_classes_combined"
    for fig, base in ((fig_pr, pr_base), (fig_roc, roc_base)):
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_name(base.name + "_highres.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {pr_base.with_suffix('.png')}")
    print(f"Wrote {pr_base.with_name(pr_base.name + '_highres.png')}")
    print(f"Wrote {roc_base.with_suffix('.png')}")
    print(f"Wrote {roc_base.with_name(roc_base.name + '_highres.png')}")


if __name__ == "__main__":
    main()
