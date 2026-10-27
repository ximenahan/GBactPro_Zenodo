#!/usr/bin/env python3
"""
Monte Carlo random-score baseline for **dense** (hybrid) double-strand PR / AUPRC.

Per simulation:
  - Same windows (start, end, strand) and reference TSS as the real scan.
  - Replace scores with i.i.d. Uniform(0, 1).
  - **Geometry fixed**: window is_tp labels unchanged (promoter containment, strand-aware).
  - Recompute max_score_per_ref from random scores (same logic as generate_pr_curve_doublestrand).
  - Same threshold grid: percentiles of random scores + upper tail (num_thresholds).
  - Hybrid PR: precision = TP windows in top-k / k; recall = fraction of refs with max ≥ thr.

This is the fair null for “no informative ranking” while preserving layout, strand rules,
and tolerance-expanded promoter boxes.

Outputs in --out-dir:
  random_baseline_per_sim.tsv, random_baseline_summary.tsv, README_method.md
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

import generate_pr_curve_doublestrand as dense  # noqa: E402


def auprc_from_pr(precisions: list, recalls: list) -> float:
    arr_p = np.asarray(precisions, dtype=np.float64)
    arr_r = np.asarray(recalls, dtype=np.float64)
    order = np.argsort(arr_r)
    sr, sp = arr_r[order], arr_p[order]
    try:
        return float(np.trapezoid(sp, sr))
    except AttributeError:
        return float(np.trapz(sp, sr))


def pr_curve_dense_random(
    sp: np.ndarray,
    ep: np.ndarray,
    sm: np.ndarray,
    em: np.ndarray,
    scp: np.ndarray,
    scm: np.ndarray,
    all_is_tp: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    num_thresholds: int,
):
    max_score_per_ref = dense._compute_max_score_per_ref(
        ref_pos, ref_plus, sp, ep, scp, sm, em, scm, tolerance
    )
    all_scores = np.concatenate([scp, scm])

    percentiles = np.linspace(0, 100, num_thresholds)
    thr_pct = np.percentile(all_scores, percentiles)
    p99 = float(np.percentile(all_scores, 99))
    s_max = float(all_scores.max())
    thr_high = np.linspace(s_max, p99, min(80, num_thresholds))
    thresholds = np.sort(np.unique(np.concatenate([thr_pct, thr_high])))[::-1]

    desc_order = np.argsort(-all_scores, kind="mergesort")
    sorted_scores = all_scores[desc_order]
    sorted_tp = all_is_tp[desc_order]
    cum_tp = np.cumsum(sorted_tp).astype(np.float64)

    sorted_ref_scores = np.sort(max_score_per_ref[max_score_per_ref > -np.inf])[::-1]
    n_total_ref = len(ref_pos)

    thresholds_out, precisions_out, recalls_out, f1_out, npred_out = [], [], [], [], []
    for thr in thresholds:
        n_pred = int(np.searchsorted(-sorted_scores, -thr, side="right"))
        if n_pred == 0:
            thresholds_out.append(thr)
            precisions_out.append(0.0)
            recalls_out.append(0.0)
            f1_out.append(0.0)
            npred_out.append(0)
            continue
        prec = float(cum_tp[n_pred - 1] / n_pred)
        n_ref_above = int(np.searchsorted(-sorted_ref_scores, -thr, side="right"))
        rec = n_ref_above / n_total_ref if n_total_ref > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        thresholds_out.append(thr)
        precisions_out.append(prec)
        recalls_out.append(rec)
        f1_out.append(f1)
        npred_out.append(n_pred)

    return thresholds_out, precisions_out, recalls_out, f1_out, npred_out


def best_f1_fp(thresholds, precisions, recalls, npreds):
    f1s = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(precisions, recalls)]
    i = int(np.argmax(f1s))
    k = int(npreds[i])
    tp_w = int(round(precisions[i] * k)) if k else 0
    fp_w = k - tp_w
    return {
        "best_f1": f1s[i],
        "best_precision": precisions[i],
        "best_recall": recalls[i],
        "best_threshold": thresholds[i],
        "k_at_best_f1": k,
        "tp_windows_at_best_f1": tp_w,
        "fp_windows_at_best_f1": fp_w,
    }


def main():
    ap = argparse.ArgumentParser(description="Random-score baseline for dense hybrid PR / AUPRC")
    ap.add_argument("--input-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--tolerance", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--num-thresholds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("Loading ...", flush=True)
    starts, ends, is_plus, _ = dense.load_predictions(str(args.input_tsv))
    ref_pos, ref_plus = dense.load_reference_tss(str(args.reference_csv))
    genome_bp = int(ends.max())
    mb = genome_bp / 1e6
    kb1000 = genome_bp / 1000.0

    plus_mask = is_plus
    p_idx = np.where(plus_mask)[0]
    m_idx = np.where(~plus_mask)[0]
    ord_p = np.argsort(starts[p_idx])
    ord_m = np.argsort(starts[m_idx])
    p_idx_s = p_idx[ord_p]
    m_idx_s = m_idx[ord_m]
    sp, ep = starts[p_idx_s], ends[p_idx_s]
    sm, em = starts[m_idx_s], ends[m_idx_s]

    regions = dense._build_promoter_regions(ref_pos, ref_plus, args.tolerance)
    tp_plus = dense._vectorized_is_tp(sp, ep, *regions["+"])
    tp_minus = dense._vectorized_is_tp(sm, em, *regions["-"])
    all_is_tp = np.concatenate([tp_plus, tp_minus])

    n = len(starts)
    per_rows = []
    t0 = time.time()
    for s in range(args.n_sims):
        scores = rng.random(n, dtype=np.float64)
        scp = scores[p_idx_s]
        scm = scores[m_idx_s]
        thr_l, pre_l, rec_l, f1_l, np_l = pr_curve_dense_random(
            sp,
            ep,
            sm,
            em,
            scp,
            scm,
            all_is_tp,
            ref_pos,
            ref_plus,
            args.tolerance,
            args.num_thresholds,
        )
        au = auprc_from_pr(pre_l, rec_l)
        bf = best_f1_fp(thr_l, pre_l, rec_l, np_l)
        fp = bf["fp_windows_at_best_f1"]
        per_rows.append(
            {
                "sim": s,
                "auprc": au,
                "best_f1": bf["best_f1"],
                "best_precision": bf["best_precision"],
                "best_recall": bf["best_recall"],
                "best_threshold": bf["best_threshold"],
                "k_at_best_f1": bf["k_at_best_f1"],
                "tp_windows_at_best_f1": bf["tp_windows_at_best_f1"],
                "fp_windows_at_best_f1": fp,
                "fp_per_genome": fp,
                "fp_per_Mb": fp / mb,
                "fp_per_1000kb": fp / kb1000,
            }
        )
        if (s + 1) % max(1, args.n_sims // 10) == 0 or s == 0:
            print(f"  {s+1}/{args.n_sims} sims  ({time.time()-t0:.1f}s)", flush=True)

    per_df = pd.DataFrame(per_rows)
    per_path = args.out_dir / "random_baseline_per_sim.tsv"
    per_df.to_csv(per_path, sep="\t", index=False)

    x = per_df["auprc"].values
    a_lo, a_hi = float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
    summary = {
        "tolerance_bp": args.tolerance,
        "n_simulations": args.n_sims,
        "random_seed": args.seed,
        "genome_length_bp": genome_bp,
        "n_windows": n,
        "n_reference_TSS": len(ref_pos),
        "auprc_mean": float(per_df["auprc"].mean()),
        "auprc_std": float(per_df["auprc"].std(ddof=1)) if args.n_sims > 1 else 0.0,
        "auprc_ci95_low": a_lo,
        "auprc_ci95_high": a_hi,
        "best_f1_mean": float(per_df["best_f1"].mean()),
        "fp_at_best_f1_mean": float(per_df["fp_windows_at_best_f1"].mean()),
        "fp_per_Mb_at_best_f1_mean": float(per_df["fp_per_Mb"].mean()),
        "fp_per_1000kb_at_best_f1_mean": float(per_df["fp_per_1000kb"].mean()),
        "elapsed_sec": time.time() - t0,
        "input_tsv": str(args.input_tsv),
        "reference_csv": str(args.reference_csv),
    }
    sum_path = args.out_dir / "random_baseline_summary.tsv"
    pd.DataFrame([summary]).to_csv(sum_path, sep="\t", index=False)

    readme = args.out_dir / "README_method.md"
    readme.write_text(
        "## Random-score baseline (dense hybrid PR)\n\n"
        "- **Scores**: i.i.d. Uniform(0,1) per window; **geometry unchanged** "
        "(strand-aware promoter boxes, 29 bp full containment for window TP labels).\n"
        "- **Hybrid PR** (same as `generate_pr_curve_doublestrand.py`): "
        "precision from **window** TP in top-k; recall from **per-TSS max** score ≥ threshold.\n"
        "- **Null interpretation**: ranking carries no signal; expected AUPRC / best-F1 / FP rates.\n\n"
        "Compare to the real curve: `observed_auprc_reference.txt` (from `../pr_curve_data.tsv`).\n\n"
        f"Simulations: **{args.n_sims}**, seed **{args.seed}**, tolerance **{args.tolerance}** bp.\n\n"
        f"Outputs: `{per_path.name}`, `{sum_path.name}`.\n",
        encoding="utf-8",
    )

    print(f"Wrote {per_path}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    print(f"AUPRC mean={summary['auprc_mean']:.6f} CI [{a_lo:.6f}, {a_hi:.6f}]", flush=True)


if __name__ == "__main__":
    main()
