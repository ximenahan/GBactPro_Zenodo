#!/usr/bin/env python3
"""
Strict event-level ROC (greedy matching only — same TP/TSS logic as PR).

FPR uses cumulative unmatched peaks / total unmatched peaks after the full greedy
run; TPR = matched TSS recall. No is_positive / region-membership for ROC.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
import evaluate_event_level as ev  # noqa: E402


def compute_roc_strict(
    desc_order,
    is_tp_sorted,
    cum_matched,
    peak_scores,
    total_tss: int,
    num_thresholds: int = 100,
):
    n = len(desc_order)
    scores_sorted = peak_scores[desc_order]

    cum_tp = np.cumsum(is_tp_sorted.astype(np.float64))
    recall_at_rank = cum_matched.astype(np.float64) / total_tss if total_tss > 0 else np.zeros(n)

    total_tp_peaks = float(cum_tp[-1]) if n else 0.0
    total_unmatched = float(n - total_tp_peaks)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    cum_fp_peaks = ranks - cum_tp
    if total_unmatched > 0:
        fpr_at_rank = cum_fp_peaks / total_unmatched
    else:
        fpr_at_rank = np.zeros(n)

    percentiles = np.linspace(0, 100, num_thresholds)
    thr_pct = np.percentile(scores_sorted, percentiles)
    p99 = float(np.percentile(scores_sorted, 99))
    s_max = float(scores_sorted.max()) if n > 0 else 0.0
    thr_high = np.linspace(s_max, p99, min(80, num_thresholds))
    thresholds = np.sort(np.unique(np.concatenate([thr_pct, thr_high])))[::-1]

    roc_data = {
        "threshold": [],
        "fpr": [],
        "tpr": [],
        "num_events": [],
        "fp_peaks_cum": [],
        "tp_peaks_cum": [],
        "matched_tss_cum": [],
    }

    for thr in thresholds:
        k = int(np.searchsorted(-scores_sorted, -thr, side="right"))
        if k == 0:
            roc_data["threshold"].append(thr)
            roc_data["fpr"].append(0.0)
            roc_data["tpr"].append(0.0)
            roc_data["num_events"].append(0)
            roc_data["fp_peaks_cum"].append(0)
            roc_data["tp_peaks_cum"].append(0)
            roc_data["matched_tss_cum"].append(0)
            continue

        tpr = float(recall_at_rank[k - 1])
        fpr = float(fpr_at_rank[k - 1])
        roc_data["threshold"].append(thr)
        roc_data["fpr"].append(fpr)
        roc_data["tpr"].append(tpr)
        roc_data["num_events"].append(k)
        roc_data["fp_peaks_cum"].append(int(cum_fp_peaks[k - 1]))
        roc_data["tp_peaks_cum"].append(int(cum_tp[k - 1]))
        roc_data["matched_tss_cum"].append(int(cum_matched[k - 1]))

    meta = {
        "total_peaks": n,
        "total_tss": total_tss,
        "total_tp_peaks_full": int(total_tp_peaks),
        "total_unmatched_peaks": int(total_unmatched),
        "matched_tss_full": int(cum_matched[-1]) if n else 0,
    }
    return roc_data, meta


def plot_roc_curve(roc_data, output_file, title):
    fprs = np.array(roc_data["fpr"])
    tprs = np.array(roc_data["tpr"])
    thrs = roc_data["threshold"]
    si = np.argsort(fprs)
    sf, st = fprs[si], tprs[si]
    try:
        auroc = float(np.trapezoid(st, sf))
    except AttributeError:
        auroc = float(np.trapz(st, sf))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(sf, st, "b-", linewidth=2, label=f"ROC (strict event) AUROC={auroc:.4f}", zorder=3)
    ax.scatter(fprs, tprs, c="blue", s=20, alpha=0.3, edgecolors="none", zorder=2)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="TPR=FPR reference (not model random)")

    youden = [t - f for t, f in zip(tprs, fprs)]
    bi = int(np.argmax(youden))
    ax.plot(fprs[bi], tprs[bi], "ro", markersize=12,
            label=f"Best Youden: {youden[bi]:.3f} (T={thrs[bi]:.4f})", zorder=5)

    ax.set_xlabel(
        "FPR (cum. unmatched peaks in top-k / total unmatched peaks)",
        fontsize=11,
    )
    ax.set_ylabel("TPR = recall (matched TSS / total TSS)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.replace(".png", "_highres.png"), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")
    return auroc, youden[bi], thrs[bi], tprs[bi], fprs[bi]


def main():
    ap = argparse.ArgumentParser(description="Strict event-level ROC (greedy match only)")
    ap.add_argument("--peaks-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--tolerance", type=int, required=True)
    ap.add_argument("--num-thresholds", type=int, default=100)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tol_tag = f"{args.tolerance}bp" if args.tolerance > 0 else "no"

    band = (
        f"+ [TSS-{ev.PLUS_FAR},TSS-{ev.PLUS_NEAR}]  "
        f"- [TSS+{ev.MINUS_NEAR},TSS+{ev.MINUS_FAR}]"
    )

    print("=" * 70)
    print("  EVENT-LEVEL ROC (STRICT — GREEDY MATCH ONLY)")
    print("=" * 70)
    print(f"  Peaks:     {args.peaks_tsv}")
    print(f"  Reference: {args.reference_csv}")
    print(f"  Window:    {band} (±{args.tolerance} bp)")
    print(f"  Output:    {args.output_dir}")
    print()

    t0 = time.time()
    peaks_df = ev.load_peaks(args.peaks_tsv)
    ref_pos, ref_plus = ev.load_reference_tss(args.reference_csv)

    peak_positions = peaks_df["peak_start_1based"].values.astype(np.int64)
    peak_strands_plus = peaks_df["strand"].values == "+"
    peak_scores = peaks_df["score"].values.astype(np.float64)

    plus_tss = np.sort(ref_pos[ref_plus])
    minus_tss = np.sort(ref_pos[~ref_plus])

    print(f"\n  Greedy matching ({len(peak_positions):,} peaks, {len(ref_pos):,} TSS) ...")
    t1 = time.time()
    desc_order, is_tp_sorted, cum_matched = ev.greedy_match(
        peak_positions,
        peak_strands_plus,
        peak_scores,
        plus_tss,
        minus_tss,
        args.tolerance,
    )
    print(f"    Done in {time.time() - t1:.1f}s")

    roc_data, meta = compute_roc_strict(
        desc_order,
        is_tp_sorted,
        cum_matched,
        peak_scores,
        len(ref_pos),
        num_thresholds=args.num_thresholds,
    )

    roc_title = (
        f"ROC: Strict Event-Level (Greedy Matching)\n"
        f"({band}, tolerance={tol_tag}, FPR from unmatched peaks only)"
    )
    roc_file = os.path.join(args.output_dir, "roc_curve_strict.png")
    auroc, best_youden, best_thr, best_tpr, best_fpr = plot_roc_curve(
        roc_data, roc_file, roc_title
    )

    roc_tsv = os.path.join(args.output_dir, "roc_curve_strict_data.tsv")
    pd.DataFrame(roc_data).to_csv(roc_tsv, sep="\t", index=False)
    print(f"  Saved: {roc_tsv}")

    summary_path = os.path.join(args.output_dir, "roc_strict_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"=== Strict event-level ROC (tolerance={args.tolerance}bp) ===\n")
        f.write(f"Window: {band}\n")
        f.write(
            "FPR = (cumulative unmatched peaks in top-k) / "
            "(total unmatched peaks after full greedy run).\n"
        )
        f.write("TPR = (cumulative matched TSS in top-k) / (total reference TSS).\n\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nAUROC: {auroc:.6f}\n")
        f.write(
            f"Best Youden J: {best_youden:.6f}  "
            f"(T={best_thr:.6f}, TPR={best_tpr:.6f}, FPR={best_fpr:.6f})\n"
        )
    print(f"  Saved: {summary_path}")

    print(f"\n{'='*70}")
    print(f"  AUROC: {auroc:.6f}  |  Best Youden: {best_youden:.3f}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
