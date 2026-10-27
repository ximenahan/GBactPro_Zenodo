#!/usr/bin/env python3
"""
Split ecoli_all4classes_tss_filtered.csv into TP vs FN reference rows using the
same v5 one-to-one greedy top-k rule as bar_chart/generate_hns_bar_chart.py
(100 bp tolerance, k and threshold from PR curve best-F1 row).

Writes subset CSVs into promoter_architecture_scan/input/ for BPM scanning.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from matching_v5_o2o import compute_geom_ok, greedy_o2o_topk_with_matched_ref

SCRIPT_DIR = Path(__file__).resolve().parent
PA_SCAN_INPUT = SCRIPT_DIR / "promoter_architecture_scan" / "input"


def load_predictions(tsv_path: Path):
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
    starts = df["start_1based"].values
    ends = df["end_1based_inclusive"].values
    is_plus = df["strand"].values == "+"
    scores = df["score"].values
    return starts, ends, is_plus, scores


def find_best_f1_row(pr_data_path: Path) -> tuple[float, float, int]:
    df = pd.read_csv(pr_data_path, sep="\t")
    best_idx = df["f1_score"].idxmax()
    row = df.loc[best_idx]
    return float(row["threshold"]), float(row["f1_score"]), int(row["num_predictions"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pr-curve-tsv",
        type=Path,
        default=None,
        help="default: v5 one-to-one PR 100bp tolerance",
    )
    ap.add_argument(
        "--predictions-tsv",
        type=Path,
        default=None,
        help="default: v5 doublestrand NC_000913.tsv",
    )
    ap.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help="default: ecoli_all4classes_tss_filtered.csv",
    )
    ap.add_argument("--match-tolerance", type=int, default=100)
    args = ap.parse_args()

    root = Path("/home/brucehan/Project_02_promoter_prediction_model")
    gb = root / "ref_code" / "GBactpro_BH"
    v5 = gb / "04_genome_scanning_v5_all_data_pool"
    pr_o2o = v5 / "eval_pre_padding_max29bp" / "pr_curve_doublestrand_dense_one_to_one"

    pr_curve = args.pr_curve_tsv or (pr_o2o / "output_100bp_tolerance" / "pr_curve_data.tsv")
    preds = args.predictions_tsv or (
        v5 / "output" / "doublestrand_intervals_all_pre_max29bp" / "NC_000913.tsv"
    )
    ref_csv = args.reference_csv or (
        root / "ref_code" / "GBactPro_new" / "data" / "tss_class_subsets" / "ecoli_all4classes_tss_filtered.csv"
    )

    for p in (pr_curve, preds, ref_csv):
        if not p.exists():
            print(f"ERROR: missing {p}")
            return 1

    PA_SCAN_INPUT.mkdir(parents=True, exist_ok=True)

    thr, best_f1, k = find_best_f1_row(pr_curve)
    print(f"Best F1 row: F1={best_f1:.6f} thr={thr:.10f} k={k:,}")

    t0 = time.time()
    starts, ends, is_plus, scores = load_predictions(preds)
    df_ref = pd.read_csv(ref_csv)
    ref_pos = df_ref["Pos"].values.astype(np.int64)
    ref_plus = df_ref["Strand"].values == "+"

    geom_ok = compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, args.match_tolerance)
    tp_c, fp_c, fn_p, matched_ref = greedy_o2o_topk_with_matched_ref(
        starts, ends, is_plus, scores, ref_pos, ref_plus, args.match_tolerance, geom_ok, k
    )
    print(
        f"Matching ({args.match_tolerance} bp tol): TP preds={len(tp_c):,} FP={len(fp_c):,} "
        f"FN TSS pos count={len(fn_p):,}  matched_ref sum={int(matched_ref.sum()):,}  "
        f"elapsed {time.time()-t0:.1f}s"
    )

    df_tp = df_ref.loc[matched_ref].copy()
    df_fn = df_ref.loc[~matched_ref].copy()

    tp_path = PA_SCAN_INPUT / "ecoli_tp_tss_subset.csv"
    fn_path = PA_SCAN_INPUT / "ecoli_fn_tss_subset.csv"
    df_tp.to_csv(tp_path, index=False)
    df_fn.to_csv(fn_path, index=False)
    print(f"Wrote TP subset ({len(df_tp):,} rows): {tp_path}")
    print(f"Wrote FN subset ({len(df_fn):,} rows): {fn_path}")

    manifest = {
        "pr_curve_tsv": str(pr_curve),
        "predictions_tsv": str(preds),
        "reference_csv": str(ref_csv),
        "match_tolerance_bp": args.match_tolerance,
        "best_f1": best_f1,
        "threshold": thr,
        "num_predictions_k": k,
        "n_reference": int(len(df_ref)),
        "n_tp_reference_rows": int(len(df_tp)),
        "n_fn_reference_rows": int(len(df_fn)),
        "n_fp_predictions": len(fp_c),
    }
    man_path = SCRIPT_DIR / "subset_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {man_path}")
    assert len(df_tp) + len(df_fn) == len(df_ref), "TP+FN must partition full reference"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
