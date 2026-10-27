#!/usr/bin/env python3
"""
Load pr_curve_data.tsv from dense vs one-to-one PR runs (0 and 100 bp tolerance),
take the best-F1 row per file, derive TP/FP/FN-style counts, write TSV/MD and a chart.

Dense PR is hybrid: precision uses window TP/FP among top-k; recall uses per-TSS max score.
One-to-one: TP = greedy matched pairs; FP = predictions without a match; FN = unmatched TSS.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_REF = 11834
EVAL_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

CONFIG = [
    ("dense", "no_tolerance", EVAL_ROOT / "pr_curve_doublestrand_dense/output_no_tolerance/pr_curve_data.tsv"),
    ("dense", "100bp_tolerance", EVAL_ROOT / "pr_curve_doublestrand_dense/output_100bp_tolerance/pr_curve_data.tsv"),
    ("one_to_one", "no_tolerance", EVAL_ROOT / "pr_curve_doublestrand_dense_one_to_one/output_no_tolerance/pr_curve_data.tsv"),
    ("one_to_one", "100bp_tolerance", EVAL_ROOT / "pr_curve_doublestrand_dense_one_to_one/output_100bp_tolerance/pr_curve_data.tsv"),
]


def best_f1_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["f1_score"].idxmax()]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []

    for method, tol, path in CONFIG:
        df = pd.read_csv(path, sep="\t")
        r = best_f1_row(df)
        thr = float(r["threshold"])
        prec = float(r["precision"])
        rec = float(r["recall"])
        f1 = float(r["f1_score"])
        n_pred = int(r["num_predictions"])

        rec_out = {
            "method": method,
            "tolerance_label": tol,
            "source_tsv": str(path.relative_to(EVAL_ROOT)),
            "threshold": thr,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "n_predictions_top_k": n_pred,
        }

        if method == "dense":
            tp_win = prec * n_pred
            fp_win = n_pred - tp_win
            tp_ref = rec * N_REF
            fn_ref = N_REF - tp_ref
            rec_out.update(
                {
                    "TP_windows_in_top_k": int(round(tp_win)),
                    "FP_windows_in_top_k": int(round(fp_win)),
                    "TP_references_detected_recall_axis": int(round(tp_ref)),
                    "FN_references_missed_recall_axis": int(round(fn_ref)),
                }
            )
        else:
            tp = prec * n_pred
            if abs(tp - rec * N_REF) >= 1.5:
                raise RuntimeError(f"Inconsistent 1:1 PR: prec*k={tp} vs rec*N={rec * N_REF}")
            tp_i = int(round(tp))
            fp_i = n_pred - tp_i
            fn_i = N_REF - tp_i
            rec_out.update(
                {
                    "TP_matched_pairs": tp_i,
                    "FP_predictions_no_match": fp_i,
                    "FN_references_unmatched": fn_i,
                }
            )

        records.append(rec_out)

    summary = pd.DataFrame(records)
    int_cols = [
        "n_predictions_top_k",
        "TP_windows_in_top_k",
        "FP_windows_in_top_k",
        "TP_references_detected_recall_axis",
        "FN_references_missed_recall_axis",
        "TP_matched_pairs",
        "FP_predictions_no_match",
        "FN_references_unmatched",
    ]
    for c in int_cols:
        if c in summary.columns:
            summary[c] = pd.array(
                [int(x) if pd.notna(x) else pd.NA for x in summary[c]],
                dtype="Int64",
            )
    tsv_path = OUT_DIR / "best_f1_metrics.tsv"
    summary.to_csv(tsv_path, sep="\t", index=False)

    dense_df = summary[summary["method"] == "dense"].dropna(axis=1, how="all")
    o2o_df = summary[summary["method"] == "one_to_one"].dropna(axis=1, how="all")

    # Wide comparison table for MD (key metrics only)
    lines = [
        "# Best F1 comparison: dense vs one-to-one PR (pre-padding max29bp)",
        "",
        f"Reference TSS count: **{N_REF}** (`ecoli_all4classes_tss_filtered.csv`: **5,923** `+` TSS, **5,911** `−` TSS).",
        "",
        "## Precision / recall / F1 at best-F1 operating point (all four runs)",
        "",
        "Values come from the row with maximum `f1_score` in each `pr_curve_data.tsv`.",
        "",
        "| method | tolerance | threshold | precision | recall | F1 | k (preds ≥ thr) |",
        "|--------|-----------|-----------|-----------|--------|-----|-----------------|",
    ]
    for rec in records:
        lines.append(
            f"| {rec['method']} | {rec['tolerance_label']} | {rec['threshold']:.6f} | "
            f"{rec['precision']:.6f} | {rec['recall']:.6f} | {rec['f1']:.6f} | {rec['n_predictions_top_k']} |"
        )
    lines.extend(
        [
            "",
            "## Strand breakdown (+ / −)",
            "",
            "Run **`python3 compute_strand_breakdown_all.py`** (loads ~9.3M scan rows once) to regenerate:",
            "",
            "- `best_f1_master_with_strand.tsv` — one row per setting: precision/recall/F1, k, and TP/FP/FN split by strand.",
            "- `strand_distribution_best_f1_all.tsv` — long format (`method`, `tolerance_label`, `metric`, `strand`, `count`).",
            "",
            "**Dense:** `TP_win_*` / `FP_win_*` = strand of the **predicted window**; `TP_ref_*` / `FN_ref_*` = strand of the **reference TSS** (recall axis).",
            "",
            "**One-to-one:** `TP_match_ref_*` = strand of **matched reference**; `FP_pred_*` = strand of **prediction** without match; `FN_ref_*_o2o` = **unmatched** reference strand.",
            "",
            "## Definitions",
            "",
            "### Dense (`pr_curve_doublestrand_dense`)",
        "- **TP_windows / FP_windows**: Among the **top-k** predictions by score (k = `n_predictions_top_k`), windows that lie fully inside a true promoter (strand-aware) vs not.",
        "- **TP_references / FN_references**: **Recall axis** — TSS with at least one window in their promoter region with score ≥ threshold (max per TSS). This is **not** the same count as TP_windows.",
        "",
        "### One-to-one (`pr_curve_doublestrand_dense_one_to_one`)",
        "- **TP_matched_pairs**: Greedy matching in score order; each TSS matched at most once.",
        "- **FP**: Top-k predictions that did not receive a match.",
        "- **FN**: TSS never matched.",
        "",
        "## Dense — best F1 per tolerance (aggregated counts)",
        "",
        dense_df.to_markdown(index=False),
        "",
        "## One-to-one — best F1 per tolerance (aggregated counts)",
        "",
        o2o_df.to_markdown(index=False),
        "",
        "## Files",
        "",
        "- `best_f1_metrics.tsv` — combined table (same as dense/o2o sections above).",
        "- `best_f1_comparison_chart.png` — bar chart from `build_best_f1_comparison.py`.",
        "- `best_f1_master_with_strand.tsv`, `strand_distribution_best_f1_all.tsv` — from `compute_strand_breakdown_all.py`.",
        "",
        ]
    )
    md_path = OUT_DIR / "README.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # --- Chart: reference-level TP vs FN (comparable) + prediction FP ---
    labels = [
        "dense\n0 bp",
        "dense\n100 bp",
        "1:1\n0 bp",
        "1:1\n100 bp",
    ]
    dense0 = records[0]
    dense100 = records[1]
    o20 = records[2]
    o2100 = records[3]

    tp_ref = [
        dense0["TP_references_detected_recall_axis"],
        dense100["TP_references_detected_recall_axis"],
        o20["TP_matched_pairs"],
        o2100["TP_matched_pairs"],
    ]
    fn_ref = [
        dense0["FN_references_missed_recall_axis"],
        dense100["FN_references_missed_recall_axis"],
        o20["FN_references_unmatched"],
        o2100["FN_references_unmatched"],
    ]
    fp_pred = [
        dense0["FP_windows_in_top_k"],
        dense100["FP_windows_in_top_k"],
        o20["FP_predictions_no_match"],
        o2100["FP_predictions_no_match"],
    ]
    n_pred = [r["n_predictions_top_k"] for r in records]

    x = np.arange(len(labels))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(x - w, tp_ref, w, label="TP (refs detected / matched)", color="#2ca02c")
    ax.bar(x, fn_ref, w, label="FN (refs missed)", color="#d62728")
    ax.bar(x + w, fp_pred, w, label="FP (preds: dense=window FP, 1:1=no match)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Best-F1 confusion-style counts (reference + prediction FP)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.bar(x - w / 2, n_pred, w, label="k = predictions at threshold (top-k)", color="#1f77b4")
    ax2.bar(x + w / 2, tp_ref, w, label="TP on ref axis (detected or matched)", color="#2ca02c")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Count")
    ax2.set_title("Top-k size vs reference TP (same best-F1 threshold)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "PR best F1: dense (hybrid window/ref) vs one-to-one greedy matching",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    png = OUT_DIR / "best_f1_comparison_chart.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Wrote {tsv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
