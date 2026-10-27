#!/usr/bin/env python3
"""
Precision @ rank k at the PR-curve best-F1 operating point (greedy 1:1 matching).

Reads pr_curve_data.tsv for best threshold / k, loads the same windows TSV + reference
as the one-to-one PR pipeline, then:
  • TP / FP / FN at that point (FP = k − TP under 1:1 greedy; FN = n_ref − TP)
  • precision@k = (# greedy matches in the top-k score-ranked predictions) / k, k = 1..K
  • milestone table + full TSV + plot with random (marginal) baseline = TP/(TP+FP) at K
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHUNKSIZE = 250_000

_SCRIPT_DIR = Path(__file__).resolve().parent
_PR_SCRIPTS = _SCRIPT_DIR.parent.parent / "scripts"
if str(_PR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PR_SCRIPTS))

from generate_pr_curve_doublestrand_one_to_one import (  # noqa: E402
    UPSTREAM_MIN,
    UPSTREAM_MAX,
    compute_geom_ok,
    load_reference_tss,
)


def load_scores_coords(tsv_path: Path):
    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    plus_list: list[np.ndarray] = []
    score_list: list[np.ndarray] = []

    reader = pd.read_csv(
        tsv_path,
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

    return (
        np.concatenate(starts_list),
        np.concatenate(ends_list),
        np.concatenate(plus_list),
        np.concatenate(score_list),
    )


def greedy_cum_matched_top_prefix(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    geom_ok: np.ndarray,
    k_prefix: int,
) -> np.ndarray:
    """
    cum[j] = number of 1:1 greedy matches after processing the j highest-score
    windows (global mergesort order), for j = 0 .. k_prefix.
    """
    n_ref = len(ref_pos)
    n = len(starts)
    k_prefix = int(min(max(k_prefix, 0), n))
    cum = np.zeros(k_prefix + 1, dtype=np.int64)

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
    nmatch = 0

    for step in range(k_prefix):
        idx = int(desc_order[step])
        if not geom_ok[idx]:
            cum[step + 1] = nmatch
            continue

        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2

        if is_plus[idx]:
            lo_t = we + UPSTREAM_MIN - tolerance
            hi_t = ws + UPSTREAM_MAX + tolerance
            if lo_t > hi_t:
                cum[step + 1] = nmatch
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - UPSTREAM_MAX - tolerance
            hi_t = ws - UPSTREAM_MIN + tolerance
            if lo_t > hi_t:
                cum[step + 1] = nmatch
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
            nmatch += 1
        cum[step + 1] = nmatch

    return cum


def milestone_ks(k_max: int) -> list[int]:
    raw = [
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
        200000,
        500000,
        1_000_000,
    ]
    ks = sorted({k for k in raw if k <= k_max})
    if k_max not in ks:
        ks.append(k_max)
    return sorted(set(ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pr-curve-tsv",
        type=Path,
        default=_SCRIPT_DIR.parent / "pr_curve_data.tsv",
    )
    ap.add_argument(
        "--input-tsv",
        type=Path,
        default=Path(
            "/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactpro_BH/"
            "04_genome_scanning_v5_all_data_pool/output/doublestrand_intervals_all_pre_max29bp/"
            "NC_000913.tsv"
        ),
    )
    ap.add_argument(
        "--reference-csv",
        type=Path,
        default=Path(
            "/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/scans/lstm_scan/"
            "hns_knockout_derepressed_promoter_analysis/table_s1_analysis/Table_S1_all_TSS.bed"
        ),
    )
    ap.add_argument("--tolerance", type=int, default=100)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_SCRIPT_DIR,
        help="Where to write TSV/PNG (default: this folder)",
    )
    args = ap.parse_args()

    pr = pd.read_csv(args.pr_curve_tsv, sep="\t")
    best_idx = int(pr["f1_score"].values.argmax())
    row = pr.iloc[best_idx]
    thr = float(row["threshold"])
    f1 = float(row["f1_score"])
    prec_pr = float(row["precision"])
    rec_pr = float(row["recall"])
    k_expected = int(row["num_predictions"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input_tsv} ...")
    starts, ends, is_plus, scores = load_scores_coords(args.input_tsv)
    ref_pos, ref_plus = load_reference_tss(str(args.reference_csv))

    desc_order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[desc_order]
    k_op = int(np.searchsorted(-sorted_scores, -thr, side="right"))

    if k_op != k_expected:
        print(
            f"WARNING: recomputed k_op={k_op} vs pr_curve num_predictions={k_expected}"
        )

    print(f"Greedy 1:1 prefix up to k={k_op:,} (tolerance={args.tolerance}) ...")
    geom_ok = compute_geom_ok(
        starts, ends, is_plus, ref_pos, ref_plus, args.tolerance
    )
    cum = greedy_cum_matched_top_prefix(
        starts,
        ends,
        is_plus,
        scores,
        ref_pos,
        ref_plus,
        args.tolerance,
        geom_ok,
        k_op,
    )

    tp = int(cum[k_op])
    fp = k_op - tp
    fn = int(len(ref_pos) - tp)
    prec_chk = tp / k_op if k_op else 0.0
    rec_chk = tp / len(ref_pos) if len(ref_pos) else 0.0

    if not np.isclose(prec_chk, prec_pr, rtol=0, atol=1e-4):
        print(f"WARNING: precision check {prec_chk} vs pr_curve {prec_pr}")
    if not np.isclose(rec_chk, rec_pr, rtol=0, atol=1e-4):
        print(f"WARNING: recall check {rec_chk} vs pr_curve {rec_pr}")

    pd.DataFrame(
        [
            {
                "tolerance_bp": args.tolerance,
                "pr_curve_row_index_best_f1": best_idx,
                "threshold_at_best_f1": thr,
                "f1_at_best_f1": f1,
                "precision_pr_curve": prec_pr,
                "recall_pr_curve": rec_pr,
                "k_predictions_score_ge_threshold": k_op,
                "tp_one_to_one_greedy": tp,
                "fp_one_to_one_greedy": fp,
                "fn_one_to_one_greedy": fn,
                "n_reference_tss": len(ref_pos),
                "precision_recomputed": prec_chk,
                "recall_recomputed": rec_chk,
            }
        ]
    ).to_csv(out_dir / "best_f1_tp_fp_fn.tsv", sep="\t", index=False)

    k_axis = np.arange(1, k_op + 1, dtype=np.int64)
    prec_k = cum[1 : k_op + 1].astype(np.float64) / k_axis.astype(np.float64)

    full_df = pd.DataFrame({"rank_cutoff_k": k_axis, "precision_at_k": prec_k})
    full_df.to_csv(out_dir / "precision_at_k_full.tsv", sep="\t", index=False)

    ms = milestone_ks(k_op)
    mile_df = full_df[full_df["rank_cutoff_k"].isin(ms)].copy()
    mile_df.to_csv(out_dir / "precision_at_k_milestones.tsv", sep="\t", index=False)

    baseline = tp / k_op if k_op else float("nan")

    def _save_precision_plot(path: Path, log_x: bool) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            k_axis,
            prec_k,
            color="C0",
            linewidth=1.5,
            label="Precision@k (1:1 greedy)",
        )
        ax.axhline(
            baseline,
            color="C3",
            linestyle="--",
            linewidth=1.5,
            label=f"Random / marginal baseline = TP/K = {baseline:.4f}",
        )
        ax.set_xlabel(
            "Rank cutoff k (top k predictions by score, score ≥ best-F1 threshold)"
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_ylabel("Precision@k")
        ax.set_title(
            f"Precision vs rank cutoff at best F1 (tol={args.tolerance} bp)\n"
            f"F1={f1:.4f}, T={thr:.6f}, K={k_op:,}, TP={tp:,}, FP={fp:,}, FN={fn:,}"
        )
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_ylim(0, min(1.05, float(np.max(prec_k)) * 1.05 + 0.02))
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    _save_precision_plot(out_dir / "precision_at_k_vs_rank.png", log_x=False)
    _save_precision_plot(out_dir / "precision_at_k_vs_rank_logx.png", log_x=True)
    print(f"Wrote {out_dir / 'best_f1_tp_fp_fn.tsv'}")
    print(f"Wrote {out_dir / 'precision_at_k_full.tsv'} ({len(full_df):,} rows)")
    print(f"Wrote {out_dir / 'precision_at_k_milestones.tsv'}")
    print(f"Wrote {out_dir / 'precision_at_k_vs_rank.png'}")
    print(f"Wrote {out_dir / 'precision_at_k_vs_rank_logx.png'}")


if __name__ == "__main__":
    main()
