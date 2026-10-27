#!/usr/bin/env python3
"""Make one TP/FP/FN comparison bar chart across four TSS classes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
TABLE_PATH = RUN_DIR.parent / "best_f1_tp_fp_fn_all_classes.tsv"
OUT_BASE = RUN_DIR / "tp_fp_fn_100bp_tolerance_four_classes_comparison"

CLASS_ORDER = ["primary", "secondary", "internal", "antisense"]
CLASS_LABELS = {
    "primary": "Primary",
    "secondary": "Secondary",
    "internal": "Internal",
    "antisense": "Antisense",
}
METRICS = ["TP", "FP", "FN"]
CLASS_COLORS = {
    "primary": "#0173B2",
    "secondary": "#DE8F05",
    "internal": "#029E73",
    "antisense": "#CC78BC",
}


def load_counts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    subset = df[df["pr_condition"] == "100bp_tolerance"].copy()
    subset = subset[subset["tss_class"].isin(CLASS_ORDER)].copy()
    if subset.empty:
        raise ValueError("No rows found for pr_condition=100bp_tolerance.")

    subset["tss_class"] = pd.Categorical(subset["tss_class"], categories=CLASS_ORDER, ordered=True)
    subset = subset.sort_values("tss_class")
    return subset


def format_count(n: float) -> str:
    return f"{int(n):,}"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    df = load_counts(TABLE_PATH)

    x = np.arange(len(METRICS))
    width = 0.18
    class_offsets = np.linspace(-1.5 * width, 1.5 * width, len(CLASS_ORDER))

    fig, ax = plt.subplots(figsize=(13, 8))

    for class_name, offset in zip(CLASS_ORDER, class_offsets):
        row = df[df["tss_class"] == class_name].iloc[0]
        values = np.array([row[m] for m in METRICS], dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=CLASS_COLORS[class_name],
            label=CLASS_LABELS[class_name],
            zorder=3,
        )
        for rect, val in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                format_count(val),
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=12)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_xlabel("Metric", fontsize=13)
    ax.set_title(
        "TP / FP / FN at best F1 (±100 bp tolerance)\nFour classes in each metric group",
        fontsize=15,
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(frameon=True, fontsize=11)

    ymax = float(df[METRICS].to_numpy().max())
    ax.set_ylim(0, ymax * 1.22)

    fig.tight_layout()
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT_BASE.with_name(OUT_BASE.name + "_highres.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_BASE.with_suffix('.png')}")
    print(f"Wrote {OUT_BASE.with_name(OUT_BASE.name + '_highres.png')}")


if __name__ == "__main__":
    main()
