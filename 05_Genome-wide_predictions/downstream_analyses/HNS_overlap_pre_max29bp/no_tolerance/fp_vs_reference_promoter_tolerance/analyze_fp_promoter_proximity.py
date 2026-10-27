#!/usr/bin/env python3
"""
After v5 one-to-one best F1 (no tolerance), classify strict FPs by whether the
prediction window **center** lies inside **any** reference TSS promoter region
when that region is expanded by ±tol bp (same geometry as PR/bar chart:
  + strand promoter [TSS-39, TSS-6], - strand [TSS+6, TSS+39]).

Strand: only compare each FP to reference TSS on the **same strand** as the window.

Outputs: TSV summary + stacked bar chart (each bar total height = total FP count).
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
    """Same matching as bar chart; returns (tp_centers, fp_pairs, fn_positions). fp_pairs: (center, is_plus)."""
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


def count_centers_in_merged(centers: np.ndarray, merged: list[tuple[int, int]]) -> int:
    if len(centers) == 0 or not merged:
        return 0
    starts = np.array([a for a, _ in merged], dtype=np.int64)
    ends = np.array([b for _, b in merged], dtype=np.int64)
    n_in = 0
    for p in centers.astype(np.int64):
        i = int(np.searchsorted(starts, p, side="right") - 1)
        if i >= 0 and p <= ends[i]:
            n_in += 1
    return n_in


def plot_stacked_fp_bars(
    tolerances: list[int],
    n_total_fp: int,
    n_near_list: list[int],
    out_png: str,
    title: str,
):
    n_bars = len(tolerances)
    x = np.arange(n_bars)
    width = 0.72
    remain = [n_total_fp - n for n in n_near_list]
    fig, ax = plt.subplots(figsize=(11, 7.5))

    b1 = ax.bar(
        x,
        remain,
        width,
        label='Remaining "strict FP" (center not in any expanded promoter)',
        color="#8EB8C8",
        edgecolor="#4A7A8C",
        linewidth=0.9,
    )
    b2 = ax.bar(
        x,
        n_near_list,
        width,
        bottom=remain,
        label="Within expanded promoter (count / % of total FP)",
        color="#C45C3E",
        edgecolor="#8C3D2A",
        linewidth=0.9,
    )

    ymax = n_total_fp * 1.08 if n_total_fp else 1
    ax.set_ylim(0, ymax)
    ax.axhline(n_total_fp, color="#333", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"±{t} bp\npromoter expand" for t in tolerances], fontsize=11)
    ax.set_ylabel("Number of FP windows (same total height per bar)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Extra expansion on promoter interval (same definition as v5 PR ±tolerance)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: format(int(v), ",")))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for i, (near, rem) in enumerate(zip(n_near_list, remain)):
        pct = 100.0 * near / n_total_fp if n_total_fp else 0.0
        if near > 0:
            y_mid = rem + near / 2
            ax.text(i, y_mid, f"{pct:.1f}%\n(n={near:,})", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    hr = out_png.replace(".png", "_highres.png")
    plt.savefig(hr, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_png}")
    print(f"  Saved: {hr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-curve-tsv", required=True)
    ap.add_argument("--predictions-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--match-tolerance", type=int, default=0, help="PR / 1:1 matching tolerance (bp); use 0 with no_tolerance bar chart.")
    ap.add_argument(
        "--promoter-extra-tolerances",
        default="10,25,50,75,100",
        help="Comma-separated extra ±bp on promoter interval (same as PR curve tolerance on geometry).",
    )
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

    geom_ok = g.compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, args.match_tolerance)
    tp_c, fp_pairs, fn_p = greedy_o2o_topk_fp_with_strand(
        starts, ends, is_plus, scores, ref_pos, ref_plus, args.match_tolerance, geom_ok, k, up_min, up_max
    )
    n_fp = len(fp_pairs)
    print(f"  TP={len(tp_c):,} FP={n_fp:,} FN={len(fn_p):,}  ({time.time() - t0:.1f}s)")

    fp_plus_centers = np.array([c for c, pm in fp_pairs if pm], dtype=np.int64)
    fp_minus_centers = np.array([c for c, pm in fp_pairs if not pm], dtype=np.int64)

    rows = []
    for tol in extra_tols:
        m_plus = build_merged_promoter_intervals(ref_pos, ref_plus, True, tol, up_min, up_max)
        m_minus = build_merged_promoter_intervals(ref_pos, ref_plus, False, tol, up_min, up_max)
        n_plus = count_centers_in_merged(fp_plus_centers, m_plus)
        n_minus = count_centers_in_merged(fp_minus_centers, m_minus)
        n_near = n_plus + n_minus
        prop = n_near / n_fp if n_fp else 0.0
        rows.append(
            {
                "promoter_extra_tolerance_bp": tol,
                "n_fp_near_any_reference_promoter": n_near,
                "n_fp_total": n_fp,
                "prop_fp_near_promoter": prop,
                "prop_fp_strict_remaining": 1.0 - prop,
                "pct_reduction_of_fp_count": 100.0 * prop,
            }
        )
        print(
            f"  tol ±{tol:>3} bp on promoter: near={n_near:,} / {n_fp:,}  "
            f"({100*prop:.2f}% of FP would be reclassified)"
        )

    df = pd.DataFrame(rows)
    tsv_path = os.path.join(out_dir, "fp_promoter_proximity_by_tolerance.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Wrote: {tsv_path}")

    # Table figure (same numbers as TSV)
    fig_t, ax_t = plt.subplots(figsize=(11, 2.2))
    ax_t.axis("off")
    tbl = ax_t.table(
        cellText=df.round(6).astype(str).values,
        colLabels=list(df.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 1.8)
    fig_t.suptitle("FP vs expanded reference promoter (summary table)", fontsize=12, fontweight="bold", y=0.98)
    tbl_path = os.path.join(out_dir, "fp_promoter_proximity_table.png")
    fig_t.savefig(tbl_path, dpi=300, bbox_inches="tight")
    fig_t.savefig(tbl_path.replace(".png", "_highres.png"), dpi=600, bbox_inches="tight")
    plt.close(fig_t)
    print(f"  Wrote: {tbl_path}")

    title = (
        f"Strict FP vs reference promoter proximity (expanded ±tol)\n"
        f"v5 one-to-one best F1, match_tol={args.match_tolerance} bp, F1={best_f1:.4f}, thr={thr:.4f}, k={k:,}; "
        f"n_ref={len(ref_pos):,}; promoter [TSS-{up_max},TSS-{up_min}] (+) / mirror (−)"
    )
    png = os.path.join(out_dir, "fp_promoter_proximity_stacked_bars.png")
    plot_stacked_fp_bars(
        extra_tols,
        n_fp,
        [int(r["n_fp_near_any_reference_promoter"]) for r in rows],
        png,
        title,
    )

    meta = os.path.join(out_dir, "run_metadata.txt")
    with open(meta, "w") as f:
        f.write(f"pr_curve_tsv\t{args.pr_curve_tsv}\n")
        f.write(f"predictions_tsv\t{args.predictions_tsv}\n")
        f.write(f"reference_csv\t{args.reference_csv}\n")
        f.write(f"match_tolerance_bp\t{args.match_tolerance}\n")
        f.write(f"promoter_extra_tolerances_bp\t{args.promoter_extra_tolerances}\n")
        f.write(f"best_f1\t{best_f1}\n")
        f.write(f"threshold\t{thr}\n")
        f.write(f"k\t{k}\n")
        f.write(f"n_fp\t{n_fp}\n")
    print(f"  Wrote: {meta}")


if __name__ == "__main__":
    main()
