#!/usr/bin/env python3
"""
Cumulative TP and FP along prediction rank at best-F1 (greedy 1:1 matching).

Ranked list: windows with score ≥ best-F1 threshold, ordered by descending score
(same as pr_curve_doublestrand_dense_one_to_one).

Outputs:
  • TSV: rank k, cumulative_tp, cumulative_fp (FP = k − TP at each rank)
  • Option A: two lines (cumulative TP vs k, cumulative FP vs k)
  • Option B: stacked bars of incremental TP/FP per rank bin
  • Stacked area: cumulative TP (lower) + cumulative FP (upper); total height = k
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


def binned_incremental_tp_fp(
    cum: np.ndarray, k_max: int, n_bins: int
) -> pd.DataFrame:
    """Per rank bin: incremental TP and FP (counts in that rank range)."""
    edges = np.linspace(0, k_max, n_bins + 1, dtype=np.int64)
    rows = []
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if lo >= hi:
            continue
        dtp = int(cum[hi] - cum[lo])
        width = hi - lo
        dfp = width - dtp
        rows.append(
            {
                "bin_index": b,
                "rank_lo_exclusive": lo,
                "rank_hi_inclusive": hi,
                "rank_span": width,
                "incremental_tp": dtp,
                "incremental_fp": dfp,
            }
        )
    return pd.DataFrame(rows)


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
            "/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/data/"
            "tss_class_subsets/ecoli_all4classes_tss_filtered.csv"
        ),
    )
    ap.add_argument("--tolerance", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=_SCRIPT_DIR)
    ap.add_argument(
        "--n-rank-bins",
        type=int,
        default=40,
        help="Equal-width rank bins for stacked bar chart (Option B)",
    )
    args = ap.parse_args()

    pr = pd.read_csv(args.pr_curve_tsv, sep="\t")
    best_idx = int(pr["f1_score"].values.argmax())
    row = pr.iloc[best_idx]
    thr = float(row["threshold"])
    f1 = float(row["f1_score"])
    k_expected = int(row["num_predictions"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    starts, ends, is_plus, scores = load_scores_coords(args.input_tsv)
    ref_pos, ref_plus = load_reference_tss(str(args.reference_csv))

    desc_order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[desc_order]
    k_op = int(np.searchsorted(-sorted_scores, -thr, side="right"))

    if k_op != k_expected:
        print(
            f"WARNING: recomputed k_op={k_op} vs pr_curve num_predictions={k_expected}"
        )

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

    k_axis = np.arange(1, k_op + 1, dtype=np.int64)
    cum_tp = cum[1 : k_op + 1].astype(np.int64)
    cum_fp = k_axis - cum_tp

    pd.DataFrame(
        {"rank_cutoff_k": k_axis, "cumulative_tp": cum_tp, "cumulative_fp": cum_fp}
    ).to_csv(out_dir / "cumulative_tp_fp_full.tsv", sep="\t", index=False)

    bin_df = binned_incremental_tp_fp(cum, k_op, args.n_rank_bins)
    bin_df.to_csv(out_dir / "cumulative_tp_fp_rank_bins.tsv", sep="\t", index=False)

    title_sub = (
        f"best F1 (tol={args.tolerance} bp): F1={f1:.4f}, T={thr:.6f}, K={k_op:,}"
    )

    def plot_lines(log_x: bool, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            k_axis,
            cum_tp,
            color="C2",
            linewidth=2.0,
            label="Cumulative TP",
        )
        ax.plot(
            k_axis,
            cum_fp,
            color="C1",
            linewidth=2.0,
            label="Cumulative FP",
        )
        ax.set_xlabel("Rank k (top k by score, score ≥ best-F1 threshold)")
        if log_x:
            ax.set_xscale("log")
        ax.set_ylabel("Count")
        ax.set_title(f"Cumulative TP and FP along rank\n{title_sub}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    plot_lines(False, out_dir / "cumulative_tp_fp_lines.png")
    plot_lines(True, out_dir / "cumulative_tp_fp_lines_logx.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        k_axis,
        0,
        cum_tp,
        alpha=0.75,
        color="C2",
        label="Cumulative TP",
    )
    ax.fill_between(
        k_axis,
        cum_tp,
        cum_tp + cum_fp,
        alpha=0.75,
        color="C1",
        label="Cumulative FP",
    )
    ax.plot(k_axis, k_axis, color="0.4", linewidth=0.8, linestyle=":", label="k (TP+FP)")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Cumulative count (stacked)")
    ax.set_title(f"Stacked cumulative TP + FP (height = k)\n{title_sub}")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_tp_fp_stacked_area.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    x_centers = (
        0.5 * (bin_df["rank_lo_exclusive"].values + bin_df["rank_hi_inclusive"].values)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x_centers,
        bin_df["incremental_tp"].values,
        width=bin_df["rank_span"].values * 0.92,
        align="center",
        color="C2",
        label="TP (in bin)",
        edgecolor="none",
    )
    ax.bar(
        x_centers,
        bin_df["incremental_fp"].values,
        width=bin_df["rank_span"].values * 0.92,
        align="center",
        bottom=bin_df["incremental_tp"].values,
        color="C1",
        label="FP (in bin)",
        edgecolor="none",
    )
    ax.set_xlabel("Rank (bin center; equal-width bins along 1…K)")
    ax.set_ylabel("Predictions in bin")
    ax.set_title(
        f"Stacked TP/FP by rank bin ({args.n_rank_bins} equal-width bins)\n{title_sub}"
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        out_dir / "cumulative_tp_fp_stacked_bars_bins.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Wrote {out_dir / 'cumulative_tp_fp_full.tsv'} ({k_op:,} rows)")
    print(f"Wrote {out_dir / 'cumulative_tp_fp_rank_bins.tsv'}")
    print(f"Wrote {out_dir / 'cumulative_tp_fp_lines.png'}")
    print(f"Wrote {out_dir / 'cumulative_tp_fp_lines_logx.png'}")
    print(f"Wrote {out_dir / 'cumulative_tp_fp_stacked_area.png'}")
    print(f"Wrote {out_dir / 'cumulative_tp_fp_stacked_bars_bins.png'}")


if __name__ == "__main__":
    main()
