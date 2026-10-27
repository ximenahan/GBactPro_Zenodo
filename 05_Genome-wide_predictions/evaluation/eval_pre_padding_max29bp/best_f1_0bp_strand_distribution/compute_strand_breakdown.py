#!/usr/bin/env python3
"""
Strand (+ / −) breakdown for dense vs one-to-one PR at 0 bp tolerance,
using the best-F1 threshold and k from pr_curve_doublestrand_dense/output_no_tolerance/pr_curve_data.tsv.

Reuses geometry from generate_pr_curve_doublestrand.py and greedy logic from
generate_pr_curve_doublestrand_one_to_one.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = EVAL_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_pr_curve_doublestrand import (  # noqa: E402
    UPSTREAM_MAX,
    UPSTREAM_MIN,
    _build_promoter_regions,
    _compute_max_score_per_ref,
    _vectorized_is_tp,
    load_predictions,
    load_reference_tss,
)

INPUT_TSV = (
    Path(__file__).resolve().parent.parent.parent
    / "output/doublestrand_intervals_all_pre_max29bp/NC_000913.tsv"
)
REFERENCE_CSV = Path(
    "/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/"
    "data/tss_class_subsets/ecoli_all4classes_tss_filtered.csv"
)
PR_DATA = (
    EVAL_DIR
    / "pr_curve_doublestrand_dense/output_no_tolerance/pr_curve_data.tsv"
)
OUT_TSV = Path(__file__).resolve().parent / "strand_distribution_0bp_best_f1.tsv"


def dense_stratified(
    starts,
    ends,
    is_plus,
    scores,
    ref_pos,
    ref_plus,
    tolerance: int,
    thr: float,
    k: int,
):
    regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)
    n = len(starts)
    is_tp = np.zeros(n, dtype=np.bool_)
    pm = is_plus
    is_tp[pm] = _vectorized_is_tp(starts[pm], ends[pm], *regions["+"])
    is_tp[~pm] = _vectorized_is_tp(starts[~pm], ends[~pm], *regions["-"])

    desc_order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[desc_order]
    k_eff = int(np.searchsorted(-sorted_scores, -thr, side="right"))
    if k_eff != k:
        print(
            f"NOTE: k from TSV={k}, recomputed from thr={thr} gives k_eff={k_eff} (using k_eff)"
        )
        k = k_eff

    sel = desc_order[:k]
    tp_p = int((is_tp[sel] & is_plus[sel]).sum())
    tp_m = int((is_tp[sel] & ~is_plus[sel]).sum())
    fp_p = int((~is_tp[sel] & is_plus[sel]).sum())
    fp_m = int((~is_tp[sel] & ~is_plus[sel]).sum())

    plus_mask = is_plus
    sp, ep, scp = starts[plus_mask], ends[plus_mask], scores[plus_mask]
    sm, em, scm = starts[~plus_mask], ends[~plus_mask], scores[~plus_mask]
    idx_p = np.argsort(sp)
    sp, ep, scp = sp[idx_p], ep[idx_p], scp[idx_p]
    idx_m = np.argsort(sm)
    sm, em, scm = sm[idx_m], em[idx_m], scm[idx_m]
    max_s = _compute_max_score_per_ref(
        ref_pos, ref_plus, sp, ep, scp, sm, em, scm, tolerance
    )
    detected = max_s >= thr
    n_plus_ref = int(ref_plus.sum())
    n_minus_ref = int((~ref_plus).sum())
    tp_ref_p = int((detected & ref_plus).sum())
    tp_ref_m = int((detected & ~ref_plus).sum())
    fn_ref_p = n_plus_ref - tp_ref_p
    fn_ref_m = n_minus_ref - tp_ref_m

    return {
        "dense_TP_windows_plus": tp_p,
        "dense_TP_windows_minus": tp_m,
        "dense_FP_windows_plus": fp_p,
        "dense_FP_windows_minus": fp_m,
        "dense_TP_refs_plus": tp_ref_p,
        "dense_TP_refs_minus": tp_ref_m,
        "dense_FN_refs_plus": fn_ref_p,
        "dense_FN_refs_minus": fn_ref_m,
    }


def one_to_one_stratified(
    starts,
    ends,
    is_plus,
    scores,
    ref_pos,
    ref_plus,
    tolerance: int,
    thr: float,
    k: int,
    geom_ok: np.ndarray,
):
    n_ref = len(ref_pos)
    n = len(starts)
    desc_order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[desc_order]
    k_eff = int(np.searchsorted(-sorted_scores, -thr, side="right"))
    if k_eff != k:
        k = k_eff

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
    fp_win_plus = fp_win_minus = 0
    tp_match_ref_plus = tp_match_ref_minus = 0

    for step in range(k):
        idx = int(desc_order[step])
        pred_plus = bool(is_plus[idx])
        if not geom_ok[idx]:
            if pred_plus:
                fp_win_plus += 1
            else:
                fp_win_minus += 1
            continue

        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2

        if pred_plus:
            lo_t = we + UPSTREAM_MIN - tolerance
            hi_t = ws + UPSTREAM_MAX + tolerance
            if lo_t > hi_t:
                fp_win_plus += 1
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - UPSTREAM_MAX - tolerance
            hi_t = ws - UPSTREAM_MIN + tolerance
            if lo_t > hi_t:
                fp_win_minus += 1
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
            if pred_plus:
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
            ridx = int(idx_s[best_j])
            matched_ref[ridx] = True
            if ref_plus[ridx]:
                tp_match_ref_plus += 1
            else:
                tp_match_ref_minus += 1
        else:
            if pred_plus:
                fp_win_plus += 1
            else:
                fp_win_minus += 1

    n_plus_ref = int(ref_plus.sum())
    n_minus_ref = int((~ref_plus).sum())
    matched_plus = int((matched_ref & ref_plus).sum())
    matched_minus = int((matched_ref & ~ref_plus).sum())
    fn_ref_p = n_plus_ref - matched_plus
    fn_ref_m = n_minus_ref - matched_minus

    return {
        "o2o_TP_matched_refs_plus": tp_match_ref_plus,
        "o2o_TP_matched_refs_minus": tp_match_ref_minus,
        "o2o_FP_windows_plus": fp_win_plus,
        "o2o_FP_windows_minus": fp_win_minus,
        "o2o_FN_refs_plus": fn_ref_p,
        "o2o_FN_refs_minus": fn_ref_m,
    }


def main():
    pr_df = pd.read_csv(PR_DATA, sep="\t")
    best = pr_df.loc[pr_df["f1_score"].idxmax()]
    thr = float(best["threshold"])
    k = int(best["num_predictions"])
    print(f"Best F1 row: thr={thr}, k={k}, f1={float(best['f1_score']):.6f}")

    if not INPUT_TSV.is_file():
        raise SystemExit(f"Missing input TSV: {INPUT_TSV}")
    starts, ends, is_plus, scores = load_predictions(str(INPUT_TSV))
    ref_pos, ref_plus = load_reference_tss(str(REFERENCE_CSV))
    tolerance = 0

    regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)
    geom_ok = np.zeros(len(starts), dtype=np.bool_)
    pm = is_plus
    geom_ok[pm] = _vectorized_is_tp(starts[pm], ends[pm], *regions["+"])
    geom_ok[~pm] = _vectorized_is_tp(starts[~pm], ends[~pm], *regions["-"])

    d = dense_stratified(
        starts, ends, is_plus, scores, ref_pos, ref_plus, tolerance, thr, k
    )
    o = one_to_one_stratified(
        starts, ends, is_plus, scores, ref_pos, ref_plus, tolerance, thr, k, geom_ok
    )

    rows = [
        ("metric", "dense_plus", "dense_minus", "dense_total"),
        (
            "TP windows (top-k)",
            d["dense_TP_windows_plus"],
            d["dense_TP_windows_minus"],
            d["dense_TP_windows_plus"] + d["dense_TP_windows_minus"],
        ),
        (
            "FP windows (top-k)",
            d["dense_FP_windows_plus"],
            d["dense_FP_windows_minus"],
            d["dense_FP_windows_plus"] + d["dense_FP_windows_minus"],
        ),
        (
            "TP refs (recall axis, max score ≥ thr)",
            d["dense_TP_refs_plus"],
            d["dense_TP_refs_minus"],
            d["dense_TP_refs_plus"] + d["dense_TP_refs_minus"],
        ),
        (
            "FN refs (recall axis)",
            d["dense_FN_refs_plus"],
            d["dense_FN_refs_minus"],
            d["dense_FN_refs_plus"] + d["dense_FN_refs_minus"],
        ),
        ("", "", "", ""),
        ("metric", "one_to_one_plus", "one_to_one_minus", "total"),
        (
            "TP matched pairs (by ref strand)",
            o["o2o_TP_matched_refs_plus"],
            o["o2o_TP_matched_refs_minus"],
            o["o2o_TP_matched_refs_plus"] + o["o2o_TP_matched_refs_minus"],
        ),
        (
            "FP preds (no match, by pred strand)",
            o["o2o_FP_windows_plus"],
            o["o2o_FP_windows_minus"],
            o["o2o_FP_windows_plus"] + o["o2o_FP_windows_minus"],
        ),
        (
            "FN refs (unmatched, by ref strand)",
            o["o2o_FN_refs_plus"],
            o["o2o_FN_refs_minus"],
            o["o2o_FN_refs_plus"] + o["o2o_FN_refs_minus"],
        ),
    ]

    out_df = pd.DataFrame(rows[1:5], columns=rows[0])
    out_df2 = pd.DataFrame(rows[7:], columns=rows[6])
    out_df.to_csv(OUT_TSV, sep="\t", index=False)
    out_df2.to_csv(
        Path(__file__).resolve().parent / "strand_distribution_0bp_one_to_one.tsv",
        sep="\t",
        index=False,
    )

    flat = {**d, **o, "threshold": thr, "k_top_predictions": k}
    pd.DataFrame([flat]).to_csv(
        Path(__file__).resolve().parent / "strand_distribution_0bp_flat.tsv",
        sep="\t",
        index=False,
    )

    print("\n=== Dense (window TP/FP in top-k; recall from max score per TSS) ===")
    print(out_df.to_string(index=False))
    print("\n=== One-to-one (same thr & k; TP by ref strand; FP by pred strand) ===")
    print(out_df2.to_string(index=False))
    print(f"\nWrote {OUT_TSV}")


if __name__ == "__main__":
    main()
