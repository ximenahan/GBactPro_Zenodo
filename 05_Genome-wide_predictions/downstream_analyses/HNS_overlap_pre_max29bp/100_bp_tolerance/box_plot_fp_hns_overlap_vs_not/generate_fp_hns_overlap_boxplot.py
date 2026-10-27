#!/usr/bin/env python3
"""
FP scores at v5 one-to-one best F1, split by H-NS overlap on window center.

Matches `generate_fp_hns_overlap_boxplot.py` (H-NS_analysis) conceptually, but
uses double-strand window TSV + greedy 1:1 top-k (same as PR one-to-one).
H-NS BED coordinates: same inclusive check as 05_HNS_exe bar_chart (raw BED cols).
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


def find_best_f1_row(pr_data_path: str) -> tuple[float, float, int]:
    print(f"Loading PR curve data from {pr_data_path} ...")
    df = pd.read_csv(pr_data_path, sep="\t")
    best_idx = df["f1_score"].idxmax()
    row = df.loc[best_idx]
    thr = float(row["threshold"])
    f1 = float(row["f1_score"])
    k = int(row["num_predictions"])
    print(f"  Best F1: {f1:.6f}  thr={thr:.10f}  k={k:,}")
    return thr, f1, k


def load_hns_binding_sites(bed_path: str) -> list[tuple[int, int]]:
    print(f"Loading H-NS BED: {bed_path} ...")
    sites: list[tuple[int, int]] = []
    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                sites.append((int(parts[1]), int(parts[2])))
    sites.sort(key=lambda x: x[0])
    print(f"  Loaded {len(sites):,} intervals (sorted by start)")
    return sites


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


def greedy_o2o_topk_fp_center_scores(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    geom_ok: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """Return list of (window_center_1based, score) for each FP among top-k preds."""
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
    fp_pairs: list[tuple[int, float]] = []

    for step in range(k):
        idx = int(desc_order[step])
        sc = float(scores[idx])
        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2

        if not geom_ok[idx]:
            fp_pairs.append((center, sc))
            continue

        if is_plus[idx]:
            lo_t = we + UPSTREAM_MIN - tolerance
            hi_t = ws + UPSTREAM_MAX + tolerance
            if lo_t > hi_t:
                fp_pairs.append((center, sc))
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - UPSTREAM_MAX - tolerance
            hi_t = ws - UPSTREAM_MIN + tolerance
            if lo_t > hi_t:
                fp_pairs.append((center, sc))
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
        else:
            fp_pairs.append((center, sc))

    return fp_pairs


def classify_fp_by_hns(
    fp_pairs: list[tuple[int, float]],
    hns_intervals: list[tuple[int, int]],
) -> tuple[list[float], list[float], list[int], list[int]]:
    overlap_scores: list[float] = []
    non_scores: list[float] = []
    overlap_pos: list[int] = []
    non_pos: list[int] = []

    for center, sc in fp_pairs:
        in_hns = False
        for start, end in hns_intervals:
            if center < start:
                break
            if start <= center <= end:
                in_hns = True
                break
        if in_hns:
            overlap_scores.append(sc)
            overlap_pos.append(center)
        else:
            non_scores.append(sc)
            non_pos.append(center)

    print(
        f"\nFP windows: {len(fp_pairs):,}  "
        f"H-NS overlap (center): {len(overlap_scores):,}  "
        f"no overlap: {len(non_scores):,}"
    )
    return overlap_scores, non_scores, overlap_pos, non_pos


def statistical_tests(overlap: list[float], non_ov: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from scipy import stats as st

        print("\n=== STATISTICAL TESTS (overlap vs non-overlap FP scores) ===")
        t_stat, p_two = st.ttest_ind(overlap, non_ov)
        print(f"Two-sample t-test (two-sided): t={t_stat:.4f}, p={p_two:.2e}")
        out["ttest_t"] = t_stat
        out["ttest_p_two_sided"] = p_two

        u2, p_mw2 = st.mannwhitneyu(overlap, non_ov, alternative="two-sided")
        print(f"Mann-Whitney U (two-sided): U={u2:.2f}, p={p_mw2:.2e}")
        out["mannwhitney_u_two_sided"] = u2
        out["mannwhitney_p_two_sided"] = p_mw2

        u_g, p_g = st.mannwhitneyu(overlap, non_ov, alternative="greater")
        print(f"Mann-Whitney U (overlap > non-overlap): U={u_g:.2f}, p={p_g:.2e}")
        out["mannwhitney_u_greater"] = u_g
        out["mannwhitney_p_overlap_greater"] = p_g

        mean_o = float(np.mean(overlap))
        mean_n = float(np.mean(non_ov))
        out["mean_overlap"] = mean_o
        out["mean_non_overlap"] = mean_n
        out["mean_diff_overlap_minus_non"] = mean_o - mean_n
    except ImportError:
        print("scipy not available; skipping tests")
    return out


def create_boxplot(
    overlap_scores: list[float],
    non_overlap_scores: list[float],
    output_file: str,
    title: str,
    f1: float,
) -> None:
    overlap_scores = [s for s in overlap_scores if not np.isnan(s)]
    non_overlap_scores = [s for s in non_overlap_scores if not np.isnan(s)]

    if not overlap_scores or not non_overlap_scores:
        print("Error: need both groups non-empty for boxplot.")
        return

    plt.figure(figsize=(10, 8))
    bp = plt.boxplot(
        [overlap_scores, non_overlap_scores],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
    )
    plt.gca().set_xticklabels(["FPs overlapping H-NS", "FPs not overlapping H-NS"])
    colors = ["#2ecc71", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "means", "medians", "caps"]:
        plt.setp(bp[element], color="black", linewidth=1.5)
    plt.setp(bp["fliers"], color="gray", marker="o", markersize=3, alpha=0.3, markeredgewidth=0.5)

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Model score (per window)", fontsize=14, fontweight="bold")
    plt.xlabel("False positive group", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    all_s = overlap_scores + non_overlap_scores
    p5, p95 = np.percentile(all_s, [5, 95])
    yr = p95 - p5
    y_min = max(np.min(all_s), p5 - 0.1 * yr)
    y_max = min(np.max(all_s), p95 + 0.1 * yr)
    plt.ylim([y_min, y_max])

    mo = np.mean(overlap_scores)
    mn = np.mean(non_overlap_scores)
    stats_txt = (
        f"Best F1 (PR 1:1): {f1:.4f}\n"
        f"Mean overlap: {mo:.4f}\n"
        f"Mean non-overlap: {mn:.4f}\n"
        f"Difference: {mo - mn:+.4f}"
    )
    plt.text(
        0.98,
        0.98,
        stats_txt,
        transform=plt.gcf().transFigure,
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85, edgecolor="black"),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    hr = output_file.replace(".png", "_highres.png") if output_file.endswith(".png") else output_file + "_highres.png"
    plt.savefig(hr, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")
    print(f"Saved: {hr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-curve-tsv", required=True)
    ap.add_argument("--predictions-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--hns-bed", required=True)
    ap.add_argument("--match-tolerance", type=int, required=True)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    tol_label = "no tolerance" if args.match_tolerance == 0 else f"±{args.match_tolerance} bp"

    print("=" * 70)
    print(f"  FP scores: H-NS overlap vs not — v5 one-to-one ({tol_label})")
    print("=" * 70)

    thr, f1, k = find_best_f1_row(args.pr_curve_tsv)
    starts, ends, is_plus, scores = load_predictions(args.predictions_tsv)
    ref_pos, ref_plus = load_reference_tss(args.reference_csv)
    hns = load_hns_binding_sites(args.hns_bed)

    geom_ok = compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, args.match_tolerance)
    fp_pairs = greedy_o2o_topk_fp_center_scores(
        starts, ends, is_plus, scores, ref_pos, ref_plus, args.match_tolerance, geom_ok, k
    )

    ov_sc, non_sc, ov_pos, non_pos = classify_fp_by_hns(fp_pairs, hns)

    pd.DataFrame({"window_center_1based": ov_pos, "score": ov_sc}).to_csv(
        os.path.join(out_dir, "fp_overlapping_hns_scores.tsv"), sep="\t", index=False
    )
    pd.DataFrame({"window_center_1based": non_pos, "score": non_sc}).to_csv(
        os.path.join(out_dir, "fp_not_overlapping_hns_scores.tsv"), sep="\t", index=False
    )

    test_out = statistical_tests(ov_sc, non_sc) if ov_sc and non_sc else {}

    png = os.path.join(out_dir, "fp_scores_hns_overlap_vs_not_overlap_boxplot.png")
    title = (
        f"FP window scores: H-NS overlap vs non-overlap\n"
        f"(v5 one-to-one, {tol_label}, best F1={f1:.4f} @ thr={thr:.4f})"
    )
    create_boxplot(ov_sc, non_sc, png, title, f1)

    summary = {
        "threshold": thr,
        "f1_score": f1,
        "num_predictions_k": k,
        "fp_total": len(fp_pairs),
        "fp_overlap_hns_count": len(ov_sc),
        "fp_non_overlap_count": len(non_sc),
    }
    summary.update(test_out)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "analysis_summary.tsv"), sep="\t", index=False)

    with open(os.path.join(out_dir, "run_metadata.txt"), "w") as f:
        f.write(f"pr_curve_tsv\t{args.pr_curve_tsv}\n")
        f.write(f"match_tolerance_bp\t{args.match_tolerance}\n")
        f.write(f"hns_bed\t{args.hns_bed}\n")

    print("\n" + "=" * 70 + "\n  DONE\n" + "=" * 70)


if __name__ == "__main__":
    main()
