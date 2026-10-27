#!/usr/bin/env python3
"""
For each promoter-expansion tolerance (same as fp_vs_reference_promoter_tolerance),
take **remaining** strict FP (window center not in any expanded reference promoter),
then compute **% overlapping H-NS** using the same rule as bar_chart:
  inclusive start <= center <= end on BED cols 2–3.

Does not modify any other script; self-contained under this folder.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_bar_chart_module():
    here = Path(__file__).resolve().parent
    p = here.parent / "bar_chart" / "generate_hns_bar_chart.py"
    if not p.is_file():
        raise FileNotFoundError(f"Expected {p}")
    spec = importlib.util.spec_from_file_location("hns_bar_chart", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def greedy_o2o_topk_fp_with_strand(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    geom_ok: np.ndarray,
    k: int,
    up_min: int,
    up_max: int,
):
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
    tp_centers: list[int] = []
    fp_pairs: list[tuple[int, bool]] = []

    for step in range(k):
        idx = int(desc_order[step])
        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2
        pm = bool(is_plus[idx])

        if not geom_ok[idx]:
            fp_pairs.append((center, pm))
            continue

        if pm:
            lo_t = we + up_min - tolerance
            hi_t = ws + up_max + tolerance
            if lo_t > hi_t:
                fp_pairs.append((center, pm))
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - up_max - tolerance
            hi_t = ws - up_min + tolerance
            if lo_t > hi_t:
                fp_pairs.append((center, pm))
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
            if pm:
                reg_s = T - up_max - tolerance
                reg_e = T - up_min + tolerance
            else:
                reg_s = T + up_min - tolerance
                reg_e = T + up_max + tolerance
            if ws >= reg_s and we <= reg_e:
                d = abs(center - T)
                if d < best_d:
                    best_d = d
                    best_j = j

        if best_j >= 0:
            matched_ref[int(idx_s[best_j])] = True
            tp_centers.append(center)
        else:
            fp_pairs.append((center, pm))

    fn_positions = [int(ref_pos[i]) for i in range(n_ref) if not matched_ref[i]]
    return tp_centers, fp_pairs, fn_positions


def build_merged_promoter_intervals(
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    strand_is_plus: bool,
    tol: int,
    up_min: int,
    up_max: int,
) -> list[tuple[int, int]]:
    mask = ref_plus if strand_is_plus else ~ref_plus
    pos = ref_pos[mask].astype(np.int64)
    iv: list[tuple[int, int]] = []
    for T in pos:
        T = int(T)
        if strand_is_plus:
            s = T - up_max - tol
            e = T - up_min + tol
        else:
            s = T + up_min - tol
            e = T + up_max + tol
        if s <= e:
            iv.append((s, e))
    if not iv:
        return []
    iv.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int]] = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce + 1:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def point_in_merged(p: int, merged: list[tuple[int, int]]) -> bool:
    if not merged:
        return False
    starts = np.array([a for a, _ in merged], dtype=np.int64)
    ends = np.array([b for _, b in merged], dtype=np.int64)
    i = int(np.searchsorted(starts, p, side="right") - 1)
    return i >= 0 and p <= ends[i]


def check_hns_overlap_centers(centers: list[int], hns_sites: list[tuple[int, int]]) -> tuple[int, float]:
    if not centers:
        return 0, 0.0
    n_ov = 0
    for pos in centers:
        for start, end in hns_sites:
            if start <= pos <= end:
                n_ov += 1
                break
    return n_ov, n_ov / len(centers)


def plot_pct_bars(tolerances: list[int], pcts: list[float], ns_rem: list[int], n_hns: list[int], out_png: str, title: str):
    x = np.arange(len(tolerances))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.bar(x, pcts, width=0.65, color="#1565C0", edgecolor="black", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"±{t} bp\n(after promoter filter)" for t in tolerances], fontsize=11)
    ax.set_ylabel("% of remaining FP overlapping H-NS", fontsize=12, fontweight="bold")
    ax.set_xlabel("Promoter expansion used to define 'remaining FP'", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(50.0, max(pcts) * 1.15) if pcts else 50.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (b, pct, nr, nh) in enumerate(zip(bars, pcts, ns_rem, n_hns)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            pct + ax.get_ylim()[1] * 0.02,
            f"{pct:.1f}%\n({nh:,}/{nr:,})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_png.replace(".png", "_highres.png"), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-curve-tsv", required=True)
    ap.add_argument("--predictions-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--hns-bed", required=True)
    ap.add_argument("--match-tolerance", type=int, default=0)
    ap.add_argument("--promoter-extra-tolerances", default="10,25,50,75,100")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    extra_tols = [int(x.strip()) for x in args.promoter_extra_tolerances.split(",") if x.strip()]

    g = _load_bar_chart_module()
    up_min = int(g.UPSTREAM_MIN)
    up_max = int(g.UPSTREAM_MAX)

    thr, best_f1, k = g.find_best_f1_row(args.pr_curve_tsv)
    t0 = time.time()
    starts, ends, is_plus, scores = g.load_predictions(args.predictions_tsv)
    ref_pos, ref_plus = g.load_reference_tss(args.reference_csv)
    hns_sites = g.load_hns_binding_sites(args.hns_bed)

    geom_ok = g.compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, args.match_tolerance)
    _, fp_pairs, _ = greedy_o2o_topk_fp_with_strand(
        starts, ends, is_plus, scores, ref_pos, ref_plus, args.match_tolerance, geom_ok, k, up_min, up_max
    )
    n_fp_total = len(fp_pairs)
    print(f"  FP total={n_fp_total:,}  ({time.time() - t0:.1f}s)")

    rows = []
    pcts: list[float] = []
    ns_rem: list[int] = []
    n_hns_list: list[int] = []

    for tol in extra_tols:
        m_plus = build_merged_promoter_intervals(ref_pos, ref_plus, True, tol, up_min, up_max)
        m_minus = build_merged_promoter_intervals(ref_pos, ref_plus, False, tol, up_min, up_max)
        remaining: list[int] = []
        for c, pm in fp_pairs:
            m = m_plus if pm else m_minus
            if not point_in_merged(int(c), m):
                remaining.append(int(c))
        n_rem = len(remaining)
        nh, frac = check_hns_overlap_centers(remaining, hns_sites)
        pct = 100.0 * frac
        pcts.append(pct)
        ns_rem.append(n_rem)
        n_hns_list.append(nh)
        rows.append(
            {
                "promoter_extra_tolerance_bp": tol,
                "n_fp_remaining_after_promoter_filter": n_rem,
                "n_fp_remaining_overlapping_hns": nh,
                "pct_remaining_fp_overlapping_hns": pct,
                "n_fp_total_strict": n_fp_total,
            }
        )
        print(
            f"  tol ±{tol:>3} bp: remaining FP={n_rem:,}, H-NS+={nh:,} ({pct:.2f}% of remaining)"
        )

    df = pd.DataFrame(rows)
    tsv_path = os.path.join(out_dir, "fp_remaining_hns_overlap_by_promoter_tol.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Wrote: {tsv_path}")

    title = (
        f"H-NS overlap among **remaining** strict FP (after promoter proximity filter)\n"
        f"Best F1 match_tol={args.match_tolerance} bp, F1={best_f1:.4f}, thr={thr:.4f}, k={k:,}; "
        f"overlap: window center in H-NS BED (inclusive)"
    )
    png = os.path.join(out_dir, "fp_remaining_hns_overlap_pct_bar_chart.png")
    plot_pct_bars(extra_tols, pcts, ns_rem, n_hns_list, png, title)

    meta = os.path.join(out_dir, "run_metadata.txt")
    with open(meta, "w") as f:
        f.write(f"pr_curve_tsv\t{args.pr_curve_tsv}\n")
        f.write(f"predictions_tsv\t{args.predictions_tsv}\n")
        f.write(f"reference_csv\t{args.reference_csv}\n")
        f.write(f"hns_bed\t{args.hns_bed}\n")
        f.write(f"match_tolerance_bp\t{args.match_tolerance}\n")
        f.write(f"promoter_extra_tolerances_bp\t{args.promoter_extra_tolerances}\n")
    print(f"  Wrote: {meta}")


if __name__ == "__main__":
    main()
