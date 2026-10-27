#!/usr/bin/env python3
"""
TP vs FP score boxplot at v5 **one-to-one** best F1 (double-strand window TSV).

Same greedy matching as `generate_pr_curve_doublestrand_one_to_one.py`; scores are
per-window from the genome scan TSV.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UPSTREAM_MIN = 6
UPSTREAM_MAX = 39


def load_predictions(tsv_path: str):
    print(f"Loading predictions: {tsv_path} ...")
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=["start_1based", "end_1based_inclusive", "strand", "score"],
        dtype={
            "start_1based": np.int64,
            "end_1based_inclusive": np.int64,
            "score": np.float64,
        },
    )
    print(f"  Loaded {len(df):,} rows")
    return (
        df["start_1based"].values,
        df["end_1based_inclusive"].values,
        df["strand"].values == "+",
        df["score"].values,
    )


def load_reference_tss(csv_path: str):
    print(f"Loading reference TSS: {csv_path} ...")
    df = pd.read_csv(csv_path)
    positions = df["Pos"].values.astype(np.int64)
    is_plus = df["Strand"].values == "+"
    print(f"  Total: {len(df):,}")
    return positions, is_plus


def find_best_f1_row(pr_data_path: str) -> tuple[float, float, float, float, int]:
    print(f"Loading PR curve data from {pr_data_path} ...")
    df = pd.read_csv(pr_data_path, sep="\t")
    best_idx = df["f1_score"].idxmax()
    row = df.loc[best_idx]
    thr = float(row["threshold"])
    f1 = float(row["f1_score"])
    prec = float(row["precision"])
    rec = float(row["recall"])
    k = int(row["num_predictions"])
    print(f"  Best F1: {f1:.6f}  thr={thr:.10f}  k={k:,}  P={prec:.4f} R={rec:.4f}")
    return thr, prec, rec, f1, k


def _build_promoter_regions(ref_pos, ref_plus, tolerance: int):
    regions = {}
    for strand_val, mask in [("+", ref_plus), ("-", ~ref_plus)]:
        pos = ref_pos[mask]
        if strand_val == "+":
            rs = pos - UPSTREAM_MAX - tolerance
            re = pos - UPSTREAM_MIN + tolerance
        else:
            rs = pos + UPSTREAM_MIN - tolerance
            re = pos + UPSTREAM_MAX + tolerance
        order = np.argsort(rs)
        rs, re = rs[order], re[order]
        prefix_max_end = np.maximum.accumulate(re)
        regions[strand_val] = (rs, re, prefix_max_end)
    return regions


def _vectorized_is_tp(w_starts, w_ends, reg_starts, reg_ends, prefix_max_end):
    n = len(w_starts)
    if n == 0 or len(reg_starts) == 0:
        return np.zeros(n, dtype=np.bool_)
    idx = np.searchsorted(reg_starts, w_starts, side="right") - 1
    valid = idx >= 0
    result = np.zeros(n, dtype=np.bool_)
    result[valid] = prefix_max_end[idx[valid]] >= w_ends[valid]
    return result


def compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, tolerance: int) -> np.ndarray:
    regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)
    out = np.zeros(len(starts), dtype=np.bool_)
    pm = is_plus
    out[pm] = _vectorized_is_tp(starts[pm], ends[pm], *regions["+"])
    out[~pm] = _vectorized_is_tp(starts[~pm], ends[~pm], *regions["-"])
    return out


def greedy_o2o_topk_tp_fp_scores(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    geom_ok: np.ndarray,
    k: int,
) -> tuple[list[float], list[float]]:
    n_ref = len(ref_pos)
    n = len(starts)
    k = min(k, n)
    desc_order = np.argsort(-scores, kind="mergesort")

    plus_ref_idx = np.where(ref_plus)[0]
    minus_ref_idx = np.where(~ref_plus)[0]
    plus_pos = ref_pos[plus_ref_idx]
    minus_pos = ref_pos[minus_ref_idx]
    o_p = np.argsort(plus_pos)
    o_m = np.argsort(minus_pos)
    plus_pos_s = plus_pos[o_p]
    minus_pos_s = minus_pos[o_m]
    plus_idx_s = plus_ref_idx[o_p]
    minus_idx_s = minus_ref_idx[o_m]

    matched_ref = np.zeros(n_ref, dtype=np.bool_)
    tp_scores: list[float] = []
    fp_scores: list[float] = []

    for step in range(k):
        idx = int(desc_order[step])
        sc = float(scores[idx])
        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2

        if not geom_ok[idx]:
            fp_scores.append(sc)
            continue

        if is_plus[idx]:
            lo_t = we + UPSTREAM_MIN - tolerance
            hi_t = ws + UPSTREAM_MAX + tolerance
            if lo_t > hi_t:
                fp_scores.append(sc)
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - UPSTREAM_MAX - tolerance
            hi_t = ws - UPSTREAM_MIN + tolerance
            if lo_t > hi_t:
                fp_scores.append(sc)
                continue
            lo = int(np.searchsorted(minus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(minus_pos_s, hi_t, side="right"))
            pos_s = minus_pos_s
            idx_s = minus_idx_s

        best_j = -1
        best_d = 10**18
        for j in range(lo, hi):
            ridx = int(idx_s[j])
            if matched_ref[ridx]:
                continue
            T = int(pos_s[j])
            if is_plus[idx]:
                reg_s = T - UPSTREAM_MAX - tolerance
                reg_e = T - UPSTREAM_MIN + tolerance
            else:
                reg_s = T + UPSTREAM_MIN - tolerance
                reg_e = T + UPSTREAM_MAX + tolerance
            if ws >= reg_s and we <= reg_e:
                d = abs(center - T)
                if d < best_d:
                    best_d = d
                    best_j = j

        if best_j >= 0:
            matched_ref[int(idx_s[best_j])] = True
            tp_scores.append(sc)
        else:
            fp_scores.append(sc)

    return tp_scores, fp_scores


def perform_statistical_test(tp_scores: list[float], fp_scores: list[float]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    try:
        from scipy import stats

        print("\n=== STATISTICAL TESTS ===")
        u_stat, u_pval = stats.mannwhitneyu(tp_scores, fp_scores, alternative="two-sided")
        print(f"Mann-Whitney U: U={u_stat:.2f}, p={u_pval:.2e}")
        results["mann_whitney_u"] = u_stat
        results["mann_whitney_p"] = u_pval
        t_stat, t_pval = stats.ttest_ind(tp_scores, fp_scores)
        print(f"T-test: t={t_stat:.2f}, p={t_pval:.2e}")
        results["ttest_t"] = t_stat
        results["ttest_p"] = t_pval
    except ImportError:
        print("Note: scipy not available, skipping tests")
    return results


def create_boxplot(
    tp_scores: list[float],
    fp_scores: list[float],
    output_file: str,
    threshold: float,
    precision: float,
    recall: float,
    f1: float,
    title: str,
) -> None:
    print("\n=== BOXPLOT ===")
    print(f"TP scores: {len(tp_scores):,}  FP scores: {len(fp_scores):,}")

    tp_mean = float(np.mean(tp_scores))
    fp_mean = float(np.mean(fp_scores))
    tp_median = float(np.median(tp_scores))
    fp_median = float(np.median(fp_scores))
    diff_mean = tp_mean - fp_mean
    diff_pct = (diff_mean / fp_mean * 100) if fp_mean != 0 else 0.0

    plt.figure(figsize=(10, 8))
    bp = plt.boxplot(
        [tp_scores, fp_scores],
        labels=["True Positives", "False Positives"],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
    )
    colors = ["#2ecc71", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "means", "medians", "caps"]:
        plt.setp(bp[element], color="black", linewidth=1.5)
    plt.setp(bp["fliers"], color="gray", marker="o", markersize=3, alpha=0.3, markeredgewidth=0.5)

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Model score (per window)", fontsize=14, fontweight="bold")
    plt.xlabel("Prediction type", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    all_scores = tp_scores + fp_scores
    if all_scores:
        p5 = float(np.percentile(all_scores, 5))
        p95 = float(np.percentile(all_scores, 95))
        y_range = p95 - p5
        y_min = max(p5 - 0.1 * y_range, min(all_scores))
        y_max = min(p95 + 0.1 * y_range, max(all_scores))
        plt.ylim([y_min, y_max])

    stats_text = (
        f"Threshold: {threshold:.4f} | F1: {f1:.4f}\n"
        f"P: {precision:.4f} | R: {recall:.4f}\n"
        f"TP mean: {tp_mean:.4f} | FP mean: {fp_mean:.4f}\n"
        f"TP med: {tp_median:.4f} | FP med: {fp_median:.4f}\n"
        f"Mean diff: {diff_mean:.4f} ({abs(diff_pct):.1f}% {'higher' if diff_mean > 0 else 'lower'})"
    )
    plt.text(
        0.98,
        0.98,
        stats_text,
        transform=plt.gcf().transFigure,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="black", linewidth=1),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    hr = output_file.replace(".png", "_highres.png")
    plt.savefig(hr, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")
    print(f"Saved: {hr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-curve-tsv", required=True)
    ap.add_argument("--predictions-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--match-tolerance", type=int, required=True)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    tol_label = "no tolerance" if args.match_tolerance == 0 else f"±{args.match_tolerance} bp"

    print("=" * 70)
    print(f"  TP vs FP boxplot — v5 one-to-one ({tol_label})")
    print("=" * 70)

    thr, prec, rec, f1, k = find_best_f1_row(args.pr_curve_tsv)
    starts, ends, is_plus, scores = load_predictions(args.predictions_tsv)
    ref_pos, ref_plus = load_reference_tss(args.reference_csv)

    geom_ok = compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, args.match_tolerance)
    tp_scores, fp_scores = greedy_o2o_topk_tp_fp_scores(
        starts, ends, is_plus, scores, ref_pos, ref_plus, args.match_tolerance, geom_ok, k
    )
    print(f"\nGreedy top-k: k={k:,}  TP={len(tp_scores):,}  FP={len(fp_scores):,}")

    stats_results = perform_statistical_test(tp_scores, fp_scores)

    title = (
        f"Score distribution: TP vs FP (v5 one-to-one)\n"
        f"({tol_label}, best F1={f1:.4f} @ thr={thr:.4f}, 29 bp windows)"
    )
    png = os.path.join(out_dir, "tp_fp_score_boxplot.png")
    create_boxplot(tp_scores, fp_scores, png, thr, prec, rec, f1, title)

    scores_df = pd.DataFrame(
        {
            "type": ["True Positive"] * len(tp_scores) + ["False Positive"] * len(fp_scores),
            "score": tp_scores + fp_scores,
        }
    )
    scores_df.to_csv(os.path.join(out_dir, "tp_fp_scores_data.csv"), index=False)

    summary = {
        "threshold": thr,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "num_predictions_k": k,
        "tp_count": len(tp_scores),
        "fp_count": len(fp_scores),
        "tp_mean": np.mean(tp_scores),
        "fp_mean": np.mean(fp_scores),
        "tp_median": np.median(tp_scores),
        "fp_median": np.median(fp_scores),
    }
    summary.update(stats_results)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "analysis_summary.tsv"), sep="\t", index=False)

    meta = os.path.join(out_dir, "run_metadata.txt")
    with open(meta, "w") as f:
        f.write(f"pr_curve_tsv\t{args.pr_curve_tsv}\n")
        f.write(f"match_tolerance_bp\t{args.match_tolerance}\n")
    print(f"Wrote: {out_dir}/tp_fp_scores_data.csv, analysis_summary.tsv, run_metadata.txt")
    print("=" * 70 + "\n  DONE\n" + "=" * 70)


if __name__ == "__main__":
    main()
