#!/usr/bin/env python3
"""
Count v5 O2O best-F1 (100 bp) FN reference TSS by TSS class.

Uses the same FN subset as the TP/FN logo workflow (subset of
ecoli_all4classes_tss_filtered.csv). Class is taken from the annotation
columns Primary / Secondary / Internal / Antisense (exactly one is 1 on each
all4classes row), not from set membership in the per-class CSV files (those
lists can differ slightly from the union flags).

Outputs:
  - fn_counts_by_tss_class.tsv
  - fn_class_bar_chart.png (+ _highres.png)
  - run_metadata.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_FN_CSV = (
    Path(__file__).resolve().parents[1]
    / "o2o_best_f1_tp_fn_promoter_logos"
    / "promoter_architecture_scan"
    / "input"
    / "ecoli_fn_tss_subset.csv"
)


def class_from_flags(row: pd.Series) -> str:
    for col, name in [
        ("Primary", "primary"),
        ("Secondary", "secondary"),
        ("Internal", "internal"),
        ("Antisense", "antisense"),
    ]:
        if int(row[col]) == 1:
            return name
    return "unclassified"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fn-csv",
        type=Path,
        default=DEFAULT_FN_CSV,
        help="FN reference subset (default: from o2o_best_f1_tp_fn_promoter_logos)",
    )
    ap.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.fn_csv.exists():
        print(f"ERROR: missing {args.fn_csv}")
        return 1

    df = pd.read_csv(args.fn_csv)
    required = {"Primary", "Secondary", "Internal", "Antisense", "Pos", "Strand"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns {missing}")
        return 1

    flag_sum = df[["Primary", "Secondary", "Internal", "Antisense"]].astype(int).sum(axis=1)
    n_bad = int((flag_sum != 1).sum())
    if n_bad:
        print(f"WARN: {n_bad} FN rows do not have exactly one class flag set to 1")

    df = df.copy()
    df["tss_class"] = df.apply(class_from_flags, axis=1)
    order = ["primary", "secondary", "internal", "antisense"]
    counts = df["tss_class"].value_counts().reindex(order).fillna(0).astype(int)
    total = int(len(df))

    rows = []
    for cls in order:
        c = int(counts[cls])
        rows.append(
            {
                "tss_class": cls,
                "fn_count": c,
                "fraction_of_fn": c / total if total else 0.0,
                "percent_of_fn": 100.0 * c / total if total else 0.0,
            }
        )
    summ = pd.DataFrame(rows)
    tsv_path = out_dir / "fn_counts_by_tss_class.tsv"
    summ.to_csv(tsv_path, sep="\t", index=False)
    print(summ.to_string(index=False))
    print(f"\nTotal FN rows: {total}")
    print(f"Wrote {tsv_path}")

    # Bar chart
    labels = [c.capitalize() for c in order]
    vals = [counts[c] for c in order]
    colors = ["#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("FN count (reference TSS)")
    ax.set_title(
        "FN reference TSS by annotated class\n"
        "(v5 O2O best F1, ±100 bp; union flags Primary/Secondary/Internal/Antisense)",
        fontsize=11,
    )
    ymax = max(vals) if vals else 1
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + ymax * 0.02,
            f"{v:,}\n({100 * v / total:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, ymax * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    fig.tight_layout()
    png = out_dir / "fn_class_bar_chart.png"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(png.with_name(png.stem + "_highres.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {png}")

    meta = {
        "fn_csv": str(args.fn_csv.resolve()),
        "total_fn": total,
        "rows_flag_sum_not_1": n_bad,
        "counts_by_class": {c: int(counts[c]) for c in order},
        "note": (
            "Class from ecoli_all4classes flag columns on each FN row. "
            "Per-class subset CSVs in tss_class_subsets/ are not used here "
            "(they can differ slightly from union flags)."
        ),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
