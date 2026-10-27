#!/usr/bin/env python3
"""
Chi-square test: Are FP vs FN associated with H-NS binding regions at different rates?

Mirrors chi_square_fp_hns_from_summary.py (FP vs TP) and chi_square_tp_fn_hns_from_summary.py,
but contingency rows are FP vs FN. Uses the same v5 one-to-one H-NS bar-chart summaries
(hns_overlap_summary.tsv must include FP and FN rows):

  output_no_tolerance      → no_tolerance/bar_chart/hns_overlap_summary.tsv
  output_100bp_tolerance → 100_bp_tolerance/bar_chart/hns_overlap_summary.tsv

Contingency table:
                        Overlap H-NS    Not overlap
    FP                      a               b
    FN                      c               d

  a = FP hns_overlap_count,  b = FP count − a
  c = FN hns_overlap_count, d = FN count − c
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

PR_NO_TOL = (
    "04_genome_scanning_v5_all_data_pool/eval_pre_padding_max29bp/"
    "pr_curve_doublestrand_dense_one_to_one/output_no_tolerance"
)
PR_100 = (
    "04_genome_scanning_v5_all_data_pool/eval_pre_padding_max29bp/"
    "pr_curve_doublestrand_dense_one_to_one/output_100bp_tolerance"
)


def load_counts_from_summary(summary_tsv: str) -> tuple[int, int, int, int]:
    df = pd.read_csv(summary_tsv, sep="\t")
    if not {"category", "count", "hns_overlap_count"}.issubset(df.columns):
        raise ValueError(
            f"Summary {summary_tsv} needs columns category, count, hns_overlap_count; "
            f"got {list(df.columns)}"
        )

    fp_row = df[df["category"] == "FP"].iloc[0]
    fn_row = df[df["category"] == "FN"].iloc[0]

    fp_count = int(fp_row["count"])
    fp_hns = int(fp_row["hns_overlap_count"])
    fn_count = int(fn_row["count"])
    fn_hns = int(fn_row["hns_overlap_count"])

    a = fp_hns
    b = fp_count - fp_hns
    c = fn_hns
    d = fn_count - fn_hns
    return a, b, c, d


def run_chi2(a: int, b: int, c: int, d: int):
    observed = [[a, b], [c, d]]

    if HAS_SCIPY:
        chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=False)
        return chi2, p_value, dof, expected

    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0, 1, [[0.0, 0.0], [0.0, 0.0]]

    row0, row1 = a + b, c + d
    col0, col1 = a + c, b + d
    e_a = row0 * col0 / n
    e_b = row0 * col1 / n
    e_c = row1 * col0 / n
    e_d = row1 * col1 / n
    expected = [[e_a, e_b], [e_c, e_d]]

    chi2 = (a - e_a) ** 2 / e_a + (b - e_b) ** 2 / e_b + (c - e_c) ** 2 / e_c + (d - e_d) ** 2 / e_d
    dof = 1
    if chi2 >= 10.83:
        p_value = 0.001
    elif chi2 >= 6.63:
        p_value = 0.01
    elif chi2 >= 3.84:
        p_value = 0.05
    else:
        p_value = 0.10
    return chi2, p_value, dof, expected


def analyze_condition(
    name: str,
    summary_tsv: str,
    output_dir: str,
    pr_curve_note: str,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  CHI-SQUARE: FP vs FN / H-NS (2×2) — {name}")
    print("=" * 70)
    print(f"PR source: {pr_curve_note}")
    print(f"Summary TSV: {summary_tsv}\n")

    a, b, c, d = load_counts_from_summary(summary_tsv)

    total_fp = a + b
    total_fn = c + d

    print("Counts (from H-NS summary):")
    print(f"  a (FP ∩ H-NS):     {a:,}")
    print(f"  b (FP, no H-NS):   {b:,}")
    print(f"  c (FN ∩ H-NS):     {c:,}")
    print(f"  d (FN, no H-NS):   {d:,}")
    print(f"  Total FP:          {total_fp:,}  (a+b)")
    print(f"  Total FN:          {total_fn:,}  (c+d)\n")

    chi2, p_value, dof, expected = run_chi2(a, b, c, d)

    print("Chi-square test (no Yates correction):")
    print(f"  Chi-square: {chi2:.4f}, df: {dof}, p-value: {p_value:.2e}")
    if p_value < 0.001:
        print("  Result: FP vs FN significantly differ in H-NS overlap (p < 0.001).")
    elif p_value < 0.05:
        print("  Result: FP vs FN significantly differ in H-NS overlap (p < 0.05).")
    else:
        print("  Result: No significant difference (p >= 0.05).")
    print(
        f"\nExpected under independence: "
        f"E(a)={expected[0][0]:.2f} E(b)={expected[0][1]:.2f} "
        f"E(c)={expected[1][0]:.2f} E(d)={expected[1][1]:.2f}"
    )

    p_fp = a / total_fp if total_fp > 0 else 0.0
    p_fn = c / total_fn if total_fn > 0 else 0.0
    print(f"\nProportions overlapping H-NS:  FP {p_fp*100:.2f}%   FN {p_fn*100:.2f}%")

    safe_name = name.replace(" ", "_").replace("±", "pm")

    report_path = os.path.join(output_dir, f"chi_square_fp_fn_hns_report_{safe_name}.txt")
    with open(report_path, "w") as f:
        f.write(f"Chi-square: FP vs FN / H-NS (2×2) — {name}\n")
        f.write(f"PR curve folder (v5 one-to-one): {pr_curve_note}\n")
        f.write(f"Summary TSV: {summary_tsv}\n")
        f.write("=" * 60 + "\n\n")
        f.write("                    Overlap H-NS    Not overlap\n")
        f.write(f"    FP                  {a}               {b}\n")
        f.write(f"    FN                  {c}               {d}\n\n")
        f.write(f"Chi-square: {chi2:.4f}, df: {dof}, p-value: {p_value:.2e}\n")
        f.write(f"Proportion FP overlapping H-NS: {p_fp*100:.2f}%\n")
        f.write(f"Proportion FN overlapping H-NS: {p_fn*100:.2f}%\n")
    print(f"\nReport saved: {report_path}")

    table_path = os.path.join(output_dir, f"contingency_table_2x2_{safe_name}.tsv")
    pd.DataFrame(
        {"Overlap_HNS": [a, c], "Not_overlap": [b, d]}, index=["FP", "FN"]
    ).to_csv(table_path, sep="\t")
    print(f"Contingency table saved: {table_path}\nDone.\n")

    return {
        "name": name,
        "pr_curve_folder": pr_curve_note,
        "summary_tsv": summary_tsv,
        "a_fp_hns": a,
        "b_fp_no_hns": b,
        "c_fn_hns": c,
        "d_fn_no_hns": d,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "chi2": chi2,
        "p_value": p_value,
        "p_fp_hns": p_fp,
        "p_fn_hns": p_fn,
    }


def main():
    exe_root = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(exe_root)

    parser = argparse.ArgumentParser(
        description="Chi-square: FP vs FN / H-NS (2×2) from 05_HNS_exe v5 one-to-one bar-chart summaries"
    )
    parser.add_argument(
        "--no-tol-summary",
        default=os.path.join(parent, "no_tolerance", "bar_chart", "hns_overlap_summary.tsv"),
        help="hns_overlap_summary.tsv for no-tolerance v5 one-to-one bar chart (must include FP, FN).",
    )
    parser.add_argument(
        "--with-tol-summary",
        default=os.path.join(parent, "100_bp_tolerance", "bar_chart", "hns_overlap_summary.tsv"),
        help="hns_overlap_summary.tsv for ±100 bp v5 one-to-one bar chart (must include FP, FN).",
    )
    parser.add_argument(
        "--output-dir",
        default=exe_root,
        help="Directory for reports and contingency tables.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    results.append(
        analyze_condition(
            "No tolerance v5 one-to-one",
            args.no_tol_summary,
            args.output_dir,
            PR_NO_TOL,
        )
    )
    results.append(
        analyze_condition(
            "100 bp tolerance v5 one-to-one",
            args.with_tol_summary,
            args.output_dir,
            PR_100,
        )
    )

    summary_path = os.path.join(args.output_dir, "chi_square_fp_fn_hns_summary.tsv")
    pd.DataFrame(results).to_csv(summary_path, sep="\t", index=False)
    print(f"Combined summary saved: {summary_path}")


if __name__ == "__main__":
    main()
