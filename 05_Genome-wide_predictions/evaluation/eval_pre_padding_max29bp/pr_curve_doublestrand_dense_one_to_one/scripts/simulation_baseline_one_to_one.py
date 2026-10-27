#!/usr/bin/env python3
"""
Monte Carlo random-score baseline for one-to-one PR / AUPRC.

Procedure (per simulation):
  - Keep the same scan TSV rows (start, end, strand) and reference TSS.
  - Replace scores with i.i.d. Uniform(0, 1) — random global ranking of windows.
  - Run the same geometry check, greedy one-to-one matching, and threshold sweep
    as generate_pr_curve_doublestrand_one_to_one.py (same percentile + upper-tail grid).

Outputs (written to --out-dir):
  - per_sim_metrics.tsv: AUPRC, best-F1 point, FP at best F1, FP/Mb per sim
  - summary.tsv: mean, std, 95% CI (percentile) across simulations
  - README_method.md: short description

Genome length for FP/Mb: max(end_1based_inclusive) from the scan TSV.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_pr_curve_doublestrand_one_to_one as o2o  # noqa: E402


def auprc_from_pr(precisions: list, recalls: list) -> float:
    arr_p = np.asarray(precisions, dtype=np.float64)
    arr_r = np.asarray(recalls, dtype=np.float64)
    order = np.argsort(arr_r)
    sr = arr_r[order]
    sp = arr_p[order]
    try:
        return float(np.trapezoid(sp, sr))
    except AttributeError:
        return float(np.trapz(sp, sr))


def pr_curve_from_random_scores(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores_rand: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    num_thresholds: int,
):
    geom_ok = o2o.compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, tolerance)
    cum_matched = o2o.greedy_incremental_cum_matched(
        starts,
        ends,
        is_plus,
        scores_rand,
        ref_pos,
        ref_plus,
        tolerance,
        geom_ok,
    )
    desc_order = np.argsort(-scores_rand, kind="mergesort")
    sorted_scores = scores_rand[desc_order]
    percentiles = np.linspace(0, 100, num_thresholds)
    thr_pct = np.percentile(scores_rand, percentiles)
    p99 = float(np.percentile(scores_rand, 99))
    s_max = float(scores_rand.max())
    thr_high = np.linspace(s_max, p99, min(80, num_thresholds))
    thresholds = np.sort(np.unique(np.concatenate([thr_pct, thr_high])))[::-1]

    n_total_ref = len(ref_pos)
    thresholds_out, precisions_out, recalls_out, f1_out, npred_out = [], [], [], [], []
    for thr in thresholds:
        k = int(np.searchsorted(-sorted_scores, -thr, side="right"))
        if k == 0:
            thresholds_out.append(thr)
            precisions_out.append(0.0)
            recalls_out.append(0.0)
            f1_out.append(0.0)
            npred_out.append(0)
            continue
        matched = int(cum_matched[k])
        prec = matched / k
        rec = matched / n_total_ref if n_total_ref > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        thresholds_out.append(thr)
        precisions_out.append(prec)
        recalls_out.append(rec)
        f1_out.append(f1)
        npred_out.append(k)

    return thresholds_out, precisions_out, recalls_out, f1_out, npred_out


def best_f1_point(thresholds, precisions, recalls, npreds):
    f1s = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(precisions, recalls)]
    i = int(np.argmax(f1s))
    k = int(npreds[i])
    matched = int(round(precisions[i] * k)) if k > 0 else 0
    fp = k - matched
    return {
        "best_f1": f1s[i],
        "best_precision": precisions[i],
        "best_recall": recalls[i],
        "best_threshold": thresholds[i],
        "k_at_best_f1": k,
        "tp_at_best_f1": matched,
        "fp_at_best_f1": fp,
    }


def main():
    ap = argparse.ArgumentParser(description="Random-score Monte Carlo baseline for 1:1 PR AUPRC")
    ap.add_argument("--input-tsv", required=True, help="Same all-windows TSV as PR curve")
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--tolerance", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="e.g. .../output_no_tolerance/simulation_baseline")
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--num-thresholds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("Loading data ...", flush=True)
    starts, ends, is_plus, _ = o2o.load_predictions(str(args.input_tsv))
    ref_pos, ref_plus = o2o.load_reference_tss(str(args.reference_csv))
    genome_bp = int(ends.max())

    n_win = len(starts)
    n_ref = len(ref_pos)
    mb = genome_bp / 1e6
    kb1000 = genome_bp / 1000.0

    per_rows = []
    t0 = time.time()
    for s in range(args.n_sims):
        scores_rand = rng.random(n_win, dtype=np.float64)
        thr_l, pre_l, rec_l, f1_l, np_l = pr_curve_from_random_scores(
            starts,
            ends,
            is_plus,
            scores_rand,
            ref_pos,
            ref_plus,
            args.tolerance,
            args.num_thresholds,
        )
        auprc = auprc_from_pr(pre_l, rec_l)
        bf = best_f1_point(thr_l, pre_l, rec_l, np_l)
        fp = bf["fp_at_best_f1"]
        per_rows.append(
            {
                "sim": s,
                "auprc": auprc,
                "best_f1": bf["best_f1"],
                "best_precision": bf["best_precision"],
                "best_recall": bf["best_recall"],
                "best_threshold": bf["best_threshold"],
                "k_at_best_f1": bf["k_at_best_f1"],
                "tp_at_best_f1": bf["tp_at_best_f1"],
                "fp_at_best_f1": fp,
                "fp_per_genome": fp,
                "fp_per_Mb": fp / mb,
                "fp_per_1000kb": fp / kb1000,
            }
        )
        if (s + 1) % max(1, args.n_sims // 10) == 0 or s == 0:
            elapsed = time.time() - t0
            print(f"  completed {s+1}/{args.n_sims} sims  ({elapsed:.1f}s elapsed)", flush=True)

    per_df = pd.DataFrame(per_rows)
    per_path = args.out_dir / "random_baseline_per_sim.tsv"
    per_df.to_csv(per_path, sep="\t", index=False)

    def ci95(col: str):
        x = per_df[col].values
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    a_lo, a_hi = ci95("auprc")
    summary = {
        "tolerance_bp": args.tolerance,
        "n_simulations": args.n_sims,
        "random_seed": args.seed,
        "genome_length_bp": genome_bp,
        "n_windows": n_win,
        "n_reference_TSS": n_ref,
        "auprc_mean": float(per_df["auprc"].mean()),
        "auprc_std": float(per_df["auprc"].std(ddof=1)) if args.n_sims > 1 else 0.0,
        "auprc_ci95_low": a_lo,
        "auprc_ci95_high": a_hi,
        "best_f1_mean": float(per_df["best_f1"].mean()),
        "best_f1_ci95_low": float(np.percentile(per_df["best_f1"], 2.5)),
        "best_f1_ci95_high": float(np.percentile(per_df["best_f1"], 97.5)),
        "fp_at_best_f1_mean": float(per_df["fp_at_best_f1"].mean()),
        "fp_per_Mb_at_best_f1_mean": float(per_df["fp_per_Mb"].mean()),
        "fp_per_1000kb_at_best_f1_mean": float(per_df["fp_per_1000kb"].mean()),
        "elapsed_sec": time.time() - t0,
        "input_tsv": str(args.input_tsv),
        "reference_csv": str(args.reference_csv),
    }
    sum_df = pd.DataFrame([summary])
    sum_path = args.out_dir / "random_baseline_summary.tsv"
    sum_df.to_csv(sum_path, sep="\t", index=False)

    readme = args.out_dir / "README_method.md"
    readme.write_text(
        "## Random-order baseline (one-to-one PR)\n\n"
        "- **Scores**: i.i.d. Uniform(0,1) on each scan row (same `start`, `end`, `strand` as the real model).\n"
        "- **Pipeline**: identical to `generate_pr_curve_doublestrand_one_to_one.py` "
        "(geom_ok, greedy 1:1 matching, same percentile + upper-tail thresholds).\n"
        "- **Interpretation**: expected PR / AUPRC when **ranking** of windows carries no signal; "
        "strands, tolerances, and discrete window layout are preserved.\n\n"
        "Compare **AUPRC** to the real curve: see `observed_auprc_reference.txt` in this folder "
        "(trapezoid on `../pr_curve_data.tsv`, recall-sorted).\n\n"
        f"- **Simulations**: {args.n_sims}, seed={args.seed}\n"
        f"- **Tolerance**: {args.tolerance} bp\n\n"
        f"Outputs: `{per_path.name}` (per draw), `{sum_path.name}` (mean, 95% CI, FP/Mb at best F1).\n",
        encoding="utf-8",
    )

    print(f"\nWrote {per_path}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    print(f"Wrote {readme}", flush=True)
    print(
        f"AUPRC mean={summary['auprc_mean']:.6f}  95% CI [{a_lo:.6f}, {a_hi:.6f}]",
        flush=True,
    )
    print(f"Total time {summary['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
