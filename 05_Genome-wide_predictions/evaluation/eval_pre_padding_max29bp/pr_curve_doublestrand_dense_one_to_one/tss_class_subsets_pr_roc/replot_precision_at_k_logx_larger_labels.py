#!/usr/bin/env python3
"""
Replot 100 bp-tolerance Precision@k (log-x) panels with larger axis fonts
and shortened legend text.

Does not modify existing scripts or original PNGs. Reads
  <class>/precision_at_k_best_f1_one_to_one/output_100bp_tolerance/
    precision_at_k_full.tsv
    best_f1_tp_fp_fn.tsv
and writes
    precision_at_k_vs_rank_logx_larger_labels.png
in the same output folder.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parent
CLASSES = ("primary", "secondary", "internal", "antisense")

AXIS_LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 16
TITLE_FS = 15


def plot_one(class_name: str) -> Path:
    out_dir = (
        _ROOT
        / class_name
        / "precision_at_k_best_f1_one_to_one"
        / "output_100bp_tolerance"
    )
    full = pd.read_csv(out_dir / "precision_at_k_full.tsv", sep="\t")
    meta = pd.read_csv(out_dir / "best_f1_tp_fp_fn.tsv", sep="\t").iloc[0]

    k_axis = full["rank_cutoff_k"].to_numpy()
    prec_k = full["precision_at_k"].to_numpy()
    baseline = float(meta["precision_recomputed"])
    f1 = float(meta["f1_at_best_f1"])
    thr = float(meta["threshold_at_best_f1"])
    k_op = int(meta["k_predictions_score_ge_threshold"])
    tp = int(meta["tp_one_to_one_greedy"])
    fp = int(meta["fp_one_to_one_greedy"])
    fn = int(meta["fn_one_to_one_greedy"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_axis, prec_k, color="C0", linewidth=1.5, label="Precision@k")
    ax.axhline(
        baseline,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall precision = {baseline:.4f}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Rank cutoff (k top predictions by score)", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Precision@k", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FS)
    ax.set_title(
        f"{class_name}: precision vs rank at best F1 (tol=100 bp)\n"
        f"F1={f1:.4f}, T={thr:.6f}, K={k_op:,}, TP={tp:,}, FP={fp:,}, FN={fn:,}",
        fontsize=TITLE_FS,
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=LEGEND_FS)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()

    out_path = out_dir / "precision_at_k_vs_rank_logx_larger_labels.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    for name in CLASSES:
        path = plot_one(name)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
