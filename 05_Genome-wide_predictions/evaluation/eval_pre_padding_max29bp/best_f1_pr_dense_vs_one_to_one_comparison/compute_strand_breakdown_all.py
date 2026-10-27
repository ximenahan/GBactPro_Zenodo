#!/usr/bin/env python3
"""
Per-strand (+ / −) breakdown at best-F1 (threshold + k) for all four PR settings:
  dense 0 bp, dense 100 bp, one-to-one 0 bp, one-to-one 100 bp.

Writes strand_distribution_best_f1_all.tsv (long) and updates wide columns
for merging into best_f1_master.tsv when run via build_best_f1_comparison.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

COMP_DIR = Path(__file__).resolve().parent
EVAL_DIR = COMP_DIR.parent
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
    EVAL_DIR.parent / "output/doublestrand_intervals_all_pre_max29bp/NC_000913.tsv"
)
REFERENCE_CSV = Path(
    "/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/"
    "data/tss_class_subsets/ecoli_all4classes_tss_filtered.csv"
)

PR_CONFIG = [
    ("dense", "no_tolerance", 0, EVAL_DIR / "pr_curve_doublestrand_dense/output_no_tolerance/pr_curve_data.tsv"),
    ("dense", "100bp_tolerance", 100, EVAL_DIR / "pr_curve_doublestrand_dense/output_100bp_tolerance/pr_curve_data.tsv"),
    ("one_to_one", "no_tolerance", 0, EVAL_DIR / "pr_curve_doublestrand_dense_one_to_one/output_no_tolerance/pr_curve_data.tsv"),
    ("one_to_one", "100bp_tolerance", 100, EVAL_DIR / "pr_curve_doublestrand_dense_one_to_one/output_100bp_tolerance/pr_curve_data.tsv"),
]


def dense_stratified(starts, ends, is_plus, scores, ref_pos, ref_plus, tolerance, thr, k):
    regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)
    n = len(starts)
    is_tp = np.zeros(n, dtype=np.bool_)
    pm = is_plus
    is_tp[pm] = _vectorized_is_tp(starts[pm], ends[pm], *regions["+"])
    is_tp[~pm] = _vectorized_is_tp(starts[~pm], ends[~pm], *regions["-"])

    desc_order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[desc_order]
    k_eff = int(np.searchsorted(-sorted_scores, -thr, side="right"))
    k = k_eff

    sel = desc_order[:k]
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

    return {
        "k_used": k,
        "TP_windows_plus": int((is_tp[sel] & is_plus[sel]).sum()),
        "TP_windows_minus": int((is_tp[sel] & ~is_plus[sel]).sum()),
        "FP_windows_plus": int((~is_tp[sel] & is_plus[sel]).sum()),
        "FP_windows_minus": int((~is_tp[sel] & ~is_plus[sel]).sum()),
        "TP_refs_plus": tp_ref_p,
        "TP_refs_minus": tp_ref_m,
        "FN_refs_plus": n_plus_ref - tp_ref_p,
        "FN_refs_minus": n_minus_ref - tp_ref_m,
    }


def one_to_one_stratified(
    starts, ends, is_plus, scores, ref_pos, ref_plus, tolerance, thr, k, geom_ok
):
    n_ref = len(ref_pos)
    desc_order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[desc_order]
    k = int(np.searchsorted(-sorted_scores, -thr, side="right"))

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

    return {
        "k_used": k,
        "TP_matched_refs_plus": tp_match_ref_plus,
        "TP_matched_refs_minus": tp_match_ref_minus,
        "FP_preds_plus": fp_win_plus,
        "FP_preds_minus": fp_win_minus,
        "FN_refs_plus": n_plus_ref - matched_plus,
        "FN_refs_minus": n_minus_ref - matched_minus,
    }


def main():
    long_rows = []

    if not INPUT_TSV.is_file():
        raise SystemExit(f"Missing scan TSV: {INPUT_TSV}")

    print(f"Loading predictions ({INPUT_TSV.name}) ...")
    starts, ends, is_plus, scores = load_predictions(str(INPUT_TSV))
    ref_pos, ref_plus = load_reference_tss(str(REFERENCE_CSV))
    n_ref_plus = int(ref_plus.sum())
    n_ref_minus = int((~ref_plus).sum())

    flat_records = []
    dense_cols = [
        "TP_win_plus",
        "TP_win_minus",
        "FP_win_plus",
        "FP_win_minus",
        "TP_ref_plus",
        "TP_ref_minus",
        "FN_ref_plus",
        "FN_ref_minus",
    ]
    o2o_cols = [
        "TP_match_ref_plus",
        "TP_match_ref_minus",
        "FP_pred_plus",
        "FP_pred_minus",
        "FN_ref_plus_o2o",
        "FN_ref_minus_o2o",
    ]

    for method, tol_label, tolerance, pr_path in PR_CONFIG:
        if not pr_path.is_file():
            raise SystemExit(f"Missing PR data: {pr_path}")
        pdf = pd.read_csv(pr_path, sep="\t")
        r = pdf.loc[pdf["f1_score"].idxmax()]
        thr = float(r["threshold"])
        prec = float(r["precision"])
        rec = float(r["recall"])
        f1 = float(r["f1_score"])
        k_tsv = int(r["num_predictions"])

        regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)
        geom_ok = np.zeros(len(starts), dtype=np.bool_)
        pm = is_plus
        geom_ok[pm] = _vectorized_is_tp(starts[pm], ends[pm], *regions["+"])
        geom_ok[~pm] = _vectorized_is_tp(starts[~pm], ends[~pm], *regions["-"])

        row = {
            "method": method,
            "tolerance_label": tol_label,
            "tolerance_bp": tolerance,
            "threshold": thr,
            "precision_at_best_f1": prec,
            "recall_at_best_f1": rec,
            "f1_at_best_f1": f1,
            "n_predictions_top_k": k_tsv,
            "n_ref_plus": n_ref_plus,
            "n_ref_minus": n_ref_minus,
        }
        for c in dense_cols + o2o_cols:
            row[c] = pd.NA

        if method == "dense":
            d = dense_stratified(
                starts, ends, is_plus, scores, ref_pos, ref_plus, tolerance, thr, k_tsv
            )
            row["k_recomputed"] = d["k_used"]
            row.update(
                {
                    "TP_win_plus": d["TP_windows_plus"],
                    "TP_win_minus": d["TP_windows_minus"],
                    "FP_win_plus": d["FP_windows_plus"],
                    "FP_win_minus": d["FP_windows_minus"],
                    "TP_ref_plus": d["TP_refs_plus"],
                    "TP_ref_minus": d["TP_refs_minus"],
                    "FN_ref_plus": d["FN_refs_plus"],
                    "FN_ref_minus": d["FN_refs_minus"],
                }
            )
            for key, strand, label in [
                ("TP_windows_plus", "+", "TP windows (pred strand)"),
                ("TP_windows_minus", "-", "TP windows (pred strand)"),
                ("FP_windows_plus", "+", "FP windows (pred strand)"),
                ("FP_windows_minus", "-", "FP windows (pred strand)"),
                ("TP_refs_plus", "+", "TP refs (recall axis)"),
                ("TP_refs_minus", "-", "TP refs (recall axis)"),
                ("FN_refs_plus", "+", "FN refs (recall axis)"),
                ("FN_refs_minus", "-", "FN refs (recall axis)"),
            ]:
                long_rows.append(
                    {
                        "method": method,
                        "tolerance_label": tol_label,
                        "metric": label,
                        "strand": strand,
                        "count": d[key],
                    }
                )
        else:
            o = one_to_one_stratified(
                starts,
                ends,
                is_plus,
                scores,
                ref_pos,
                ref_plus,
                tolerance,
                thr,
                k_tsv,
                geom_ok,
            )
            row["k_recomputed"] = o["k_used"]
            row.update(
                {
                    "TP_match_ref_plus": o["TP_matched_refs_plus"],
                    "TP_match_ref_minus": o["TP_matched_refs_minus"],
                    "FP_pred_plus": o["FP_preds_plus"],
                    "FP_pred_minus": o["FP_preds_minus"],
                    "FN_ref_plus_o2o": o["FN_refs_plus"],
                    "FN_ref_minus_o2o": o["FN_refs_minus"],
                }
            )
            for key, strand, label in [
                ("TP_matched_refs_plus", "+", "TP matched pairs (ref strand)"),
                ("TP_matched_refs_minus", "-", "TP matched pairs (ref strand)"),
                ("FP_preds_plus", "+", "FP preds (pred strand)"),
                ("FP_preds_minus", "-", "FP preds (pred strand)"),
                ("FN_refs_plus", "+", "FN refs (ref strand)"),
                ("FN_refs_minus", "-", "FN refs (ref strand)"),
            ]:
                long_rows.append(
                    {
                        "method": method,
                        "tolerance_label": tol_label,
                        "metric": label,
                        "strand": strand,
                        "count": o[key],
                    }
                )

        flat_records.append(row)

    long_df = pd.DataFrame(long_rows)
    long_path = COMP_DIR / "strand_distribution_best_f1_all.tsv"
    long_df.to_csv(long_path, sep="\t", index=False)

    flat_df = pd.DataFrame(flat_records)
    flat_path = COMP_DIR / "best_f1_master_with_strand.tsv"
    flat_df.to_csv(flat_path, sep="\t", index=False)

    print(flat_df.to_string(index=False))
    print(f"\nWrote {long_path}")
    print(f"Wrote {flat_path}")

    readme = COMP_DIR / "README.md"
    if readme.is_file():
        flat_disp = flat_df.astype(object).where(pd.notna(flat_df), "")
        snap = (
            "\n\n## Strand distribution snapshot (from `compute_strand_breakdown_all.py`)\n\n"
            + flat_disp.to_markdown(index=False)
            + "\n"
        )
        t = readme.read_text(encoding="utf-8")
        if "## Strand distribution snapshot" in t:
            t = t.split("## Strand distribution snapshot")[0].rstrip()
        readme.write_text(t + snap, encoding="utf-8")
        print(f"Appended snapshot to {readme}")


if __name__ == "__main__":
    main()
