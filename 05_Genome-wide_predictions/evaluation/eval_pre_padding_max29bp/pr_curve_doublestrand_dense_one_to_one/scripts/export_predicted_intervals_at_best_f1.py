#!/usr/bin/env python3
"""
Export BED + TSV of windows that are 1:1 greedy-matched at the best-F1 operating point.

Matches generate_pr_curve_doublestrand_one_to_one.py:
  • k = number of windows with score >= threshold (mergesort on -scores).
  • Among the first k preds in that global order, keep only windows that win a greedy
    match to an unused reference (same geometry / tolerance as the PR script).

This is NOT the same as "all windows with score >= threshold" (that set has size k but
includes FPs under the 1:1 definition).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNKSIZE = 250_000

# Same package as generate_pr_curve_doublestrand_one_to_one.py
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from generate_pr_curve_doublestrand_one_to_one import (  # noqa: E402
    compute_geom_ok,
    greedy_matched_window_indices_topk,
    load_reference_tss,
)


def bed_score_from_prob(p: float) -> int:
    return int(max(0, min(1000, round(float(p) * 1000))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pr-curve-tsv",
        required=True,
        help="pr_curve_data.tsv (best f1_score row gives threshold and k=num_predictions)",
    )
    ap.add_argument("--input-tsv", required=True, help="Full double-strand windows TSV")
    ap.add_argument("--reference-csv", required=True, help="Reference TSS CSV (same as PR run)")
    ap.add_argument("--tolerance", type=int, required=True, help="bp tolerance (same as PR run)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    pr = pd.read_csv(args.pr_curve_tsv, sep="\t")
    if "f1_score" not in pr.columns or "threshold" not in pr.columns:
        raise SystemExit(f"Missing columns in {args.pr_curve_tsv}")
    best_idx = int(pr["f1_score"].values.argmax())
    row = pr.iloc[best_idx]
    thr = float(row["threshold"])
    f1 = float(row["f1_score"])
    prec = float(row["precision"])
    rec = float(row["recall"])
    k_expected = int(row["num_predictions"]) if "num_predictions" in pr.columns else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading coords/scores from {args.input_tsv} (chunksize={CHUNKSIZE}) ...")
    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    plus_list: list[np.ndarray] = []
    score_list: list[np.ndarray] = []

    reader = pd.read_csv(
        args.input_tsv,
        sep="\t",
        usecols=["start_1based", "end_1based_inclusive", "strand", "score"],
        chunksize=CHUNKSIZE,
        dtype={
            "start_1based": np.int64,
            "end_1based_inclusive": np.int64,
            "strand": str,
            "score": np.float64,
        },
    )
    for chunk in reader:
        starts_list.append(chunk["start_1based"].to_numpy())
        ends_list.append(chunk["end_1based_inclusive"].to_numpy())
        plus_list.append((chunk["strand"].values == "+").astype(np.bool_))
        score_list.append(chunk["score"].to_numpy())

    starts = np.concatenate(starts_list)
    ends = np.concatenate(ends_list)
    is_plus = np.concatenate(plus_list)
    scores = np.concatenate(score_list)
    n = len(scores)
    print(f"  Rows: {n:,}")

    desc_order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[desc_order]
    k_topk = int(np.searchsorted(-sorted_scores, -thr, side="right"))

    ref_pos, ref_plus = load_reference_tss(args.reference_csv)
    print(f"  Greedy 1:1 (tolerance={args.tolerance} bp) for top k={k_topk:,} ...")
    geom_ok = compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, args.tolerance)
    matched_idx = greedy_matched_window_indices_topk(
        starts,
        ends,
        is_plus,
        scores,
        ref_pos,
        ref_plus,
        args.tolerance,
        geom_ok,
        k_topk,
    )
    matched_set = set(int(i) for i in matched_idx)
    n_matched = len(matched_set)

    if k_expected is not None and k_topk != k_expected:
        print(f"WARNING: k_topk {k_topk} != pr_curve num_predictions {k_expected}")
    prec_chk = n_matched / k_topk if k_topk else 0.0
    if not np.isclose(prec_chk, prec, rtol=0, atol=1e-5):
        print(
            f"WARNING: matched/k={prec_chk:.8f} vs pr_curve precision={prec:.8f} "
            f"(matched={n_matched}, k={k_topk})"
        )

    meta_path = out_dir / "best_f1_threshold_meta.tsv"
    pd.DataFrame(
        [
            {
                "threshold": thr,
                "f1_score": f1,
                "precision": prec,
                "recall": rec,
                "pr_curve_row_index": best_idx,
                "k_topk_score_ge_threshold": k_topk,
                "n_one_to_one_greedy_matched": n_matched,
                "tolerance_bp": args.tolerance,
            }
        ]
    ).to_csv(meta_path, sep="\t", index=False)

    usecols = [
        "chrom",
        "start_0based",
        "end_0based_exclusive",
        "start_1based",
        "end_1based_inclusive",
        "strand",
        "score",
        "bed_score",
        "window_index_0based",
        "sequence_genomic_plus",
        "sequence_model_input",
    ]

    tsv_path = out_dir / "_tmp_one_to_one_matched.tsv"
    bed_path = out_dir / "_tmp_one_to_one_matched.bed"
    for p in (tsv_path, bed_path):
        if p.exists():
            p.unlink()

    print(f"Streaming full rows; writing {n_matched:,} 1:1-matched windows ...")
    chrom0: str | None = None
    header_written = False
    row_i = 0

    reader2 = pd.read_csv(
        args.input_tsv,
        sep="\t",
        usecols=usecols,
        chunksize=CHUNKSIZE,
        dtype={
            "chrom": str,
            "start_0based": np.int64,
            "end_0based_exclusive": np.int64,
            "start_1based": np.int64,
            "end_1based_inclusive": np.int64,
            "strand": str,
            "score": np.float64,
            "bed_score": np.int64,
            "window_index_0based": np.int64,
            "sequence_genomic_plus": str,
            "sequence_model_input": str,
        },
    )

    for chunk in reader2:
        take = [j for j in range(len(chunk)) if (row_i + j) in matched_set]
        if take:
            sub = chunk.iloc[take].copy()
            if chrom0 is None:
                chrom0 = str(sub["chrom"].iloc[0])
            sub["bed_score"] = sub["score"].map(bed_score_from_prob)
            sub.to_csv(tsv_path, sep="\t", index=False, mode="a", header=not header_written)
            header_written = True
            with open(bed_path, "a", encoding="utf-8") as fb:
                for _, r in sub.iterrows():
                    name = f"win{int(r['window_index_0based'])}_{r['strand']}"
                    fb.write(
                        f"{r['chrom']}\t{int(r['start_0based'])}\t{int(r['end_0based_exclusive'])}\t"
                        f"{name}\t{int(r['bed_score'])}\t{r['strand']}\n"
                    )
        row_i += len(chunk)

    if chrom0 is None:
        chrom0 = "empty"
    safe = chrom0.replace(".", "_")
    final_tsv = out_dir / f"{safe}_best_f1_one_to_one_greedy_matched.tsv"
    final_bed = out_dir / f"{safe}_best_f1_one_to_one_greedy_matched.bed"
    for p in (final_tsv, final_bed):
        if p.exists():
            p.unlink()
    if tsv_path.exists():
        tsv_path.rename(final_tsv)
    else:
        pd.DataFrame(columns=usecols).to_csv(final_tsv, sep="\t", index=False)
    if bed_path.exists():
        bed_path.rename(final_bed)
    else:
        final_bed.write_text("", encoding="utf-8")

    print(f"Best F1 threshold: {thr:.12g}  F1={f1:.6f}  P={prec:.6f}  R={rec:.6f}")
    print(f"k (score>=T, top-k order): {k_topk:,}")
    print(f"1:1 greedy matched windows: {n_matched:,}  (precision*k check: {prec_chk:.6f})")
    print(f"Wrote {final_tsv}")
    print(f"Wrote {final_bed}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
