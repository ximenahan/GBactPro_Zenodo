#!/usr/bin/env python3
"""Aggregate AUPRC, AUROC, and max F1 for four TSS-class subsets (one-to-one dense PR/ROC)."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = Path(__file__).resolve().parent

CLASSES = ("primary", "secondary", "internal", "antisense")
CLASS_LABELS = ("Primary", "Secondary", "Internal", "Antisense")


def parse_auroc_summary(path: Path) -> dict:
    text = path.read_text()
    m_au = re.search(r"^AUROC:\s*([0-9.]+)", text, re.MULTILINE)
    m_ref = re.search(r"^Reference TSS:\s*(\d+)", text, re.MULTILINE)
    return {
        "auroc": float(m_au.group(1)) if m_au else float("nan"),
        "reference_tss": int(m_ref.group(1)) if m_ref else -1,
    }


def auprc_from_pr_tsv(path: Path) -> float:
    df = pd.read_csv(path, sep="\t")
    rec = df["recall"].to_numpy(dtype=float)
    pre = df["precision"].to_numpy(dtype=float)
    order = np.argsort(rec)
    rec_s = rec[order]
    pre_s = pre[order]
    try:
        return float(np.trapezoid(pre_s, rec_s))
    except AttributeError:
        return float(np.trapz(pre_s, rec_s))


def max_f1_from_pr_tsv(path: Path) -> float:
    df = pd.read_csv(path, sep="\t")
    if "f1_score" not in df.columns:
        raise ValueError(f"{path}: expected column f1_score")
    return float(df["f1_score"].max())


def collect_rows(tolerance_key: str, pr_subdir: str, roc_subdir: str) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        pr_tsv = RUN_ROOT / cls / pr_subdir / "pr_curve_data.tsv"
        roc_sum = RUN_ROOT / cls / roc_subdir / "auroc_summary.txt"
        if not pr_tsv.is_file():
            raise FileNotFoundError(pr_tsv)
        if not roc_sum.is_file():
            raise FileNotFoundError(roc_sum)
        roc_d = parse_auroc_summary(roc_sum)
        rows.append({
            "tss_class": cls,
            "tolerance": tolerance_key,
            "auprc": auprc_from_pr_tsv(pr_tsv),
            "auroc": roc_d["auroc"],
            "max_f1": max_f1_from_pr_tsv(pr_tsv),
            "reference_tss": roc_d["reference_tss"],
            "_pr_tsv": str(pr_tsv),
            "_roc_txt": str(roc_sum),
        })
    return pd.DataFrame(rows)


def plot_pair(df: pd.DataFrame, out_dir: Path, tol_label: str, title_suffix: str) -> None:
    x = np.arange(len(CLASS_LABELS))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    auprc = df["auprc"].values
    auroc = df["auroc"].values
    mf1 = df["max_f1"].values
    axes[0].bar(x, auprc, w * 2, color="#2c7bb6", edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_title(f"AUPRC — {title_suffix}")
    ymax_pr = float(np.nanmax(auprc)) * 1.12 if np.isfinite(np.nanmax(auprc)) else 1.0
    axes[0].set_ylim(0, min(1.02, max(0.02, ymax_pr)))
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, auroc, w * 2, color="#fdae61", edgecolor="black", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title(f"AUROC — {title_suffix}")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, mf1, w * 2, color="#7fbc41", edgecolor="black", linewidth=0.4)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
    axes[2].set_ylabel("Max F1 (PR curve grid)")
    axes[2].set_title(f"Max F1 — {title_suffix}")
    ymax_f1 = float(np.nanmax(mf1)) * 1.12 if np.isfinite(np.nanmax(mf1)) else 1.0
    axes[2].set_ylim(0, min(1.02, max(0.02, ymax_f1)))
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "TSS class subsets — dense double-strand one-to-one PR/ROC\n"
        "(eval_pre_padding_max29bp / pr_curve_doublestrand_dense_one_to_one)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    base = out_dir / f"tss_class_auprc_auroc_{tol_label}"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{base.name}_highres.png", dpi=600, bbox_inches="tight")
    plt.close()


def plot_grouped_three_metrics(df: pd.DataFrame, out_dir: Path, tol_label: str, title_suffix: str) -> None:
    """Single figure: grouped bars per TSS class (AUPRC, AUROC, Max F1). Twin y-axis so AUROC (~1) and AUPRC/F1 (~0.2) are both visible."""
    x = np.arange(len(CLASS_LABELS))
    width = 0.24
    auprc = df["auprc"].values.astype(float)
    auroc = df["auroc"].values.astype(float)
    mf1 = df["max_f1"].values.astype(float)

    fig, ax1 = plt.subplots(figsize=(11, 5.2))
    ax2 = ax1.twinx()

    c_pr, c_roc, c_f1 = "#2c7bb6", "#fdae61", "#7fbc41"
    ec = "black"
    elw = 0.4

    b1 = ax1.bar(
        x - width,
        auprc,
        width,
        color=c_pr,
        edgecolor=ec,
        linewidth=elw,
        label="AUPRC",
        zorder=2,
    )
    b3 = ax1.bar(
        x + width,
        mf1,
        width,
        color=c_f1,
        edgecolor=ec,
        linewidth=elw,
        label="Max F1 (PR curve grid)",
        zorder=2,
    )
    b2 = ax2.bar(
        x,
        auroc,
        width,
        color=c_roc,
        edgecolor=ec,
        linewidth=elw,
        label="AUROC",
        zorder=2,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
    ax1.set_xlabel("TSS class")
    ax1.set_ylabel("AUPRC / Max F1")
    ax2.set_ylabel("AUROC")
    ax1.set_title(f"AUPRC, AUROC, and Max F1 — {title_suffix}")
    ymax_small = float(np.nanmax(np.concatenate([auprc, mf1])))
    ax1.set_ylim(0, min(1.02, max(0.05, ymax_small * 1.15)))
    ax2.set_ylim(0, 1.02)
    ax1.grid(axis="y", alpha=0.3)
    ax2.grid(False)

    handles = [b1, b2, b3]
    labels = [h.get_label() for h in handles]
    ax1.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
    )

    fig.suptitle(
        "TSS class subsets — dense double-strand one-to-one PR/ROC\n"
        "(eval_pre_padding_max29bp / pr_curve_doublestrand_dense_one_to_one)",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    base = out_dir / f"tss_class_grouped_bars_{tol_label}"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{base.name}_highres.png", dpi=600, bbox_inches="tight")
    plt.close()


def copy_execution_scripts() -> None:
    dest = OUT_ROOT / "execution_scripts"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("run_one_class.sh", "launch_tmux_parallel.sh"):
        src = RUN_ROOT / name
        if src.is_file():
            shutil.copy2(src, dest / name)


def main() -> None:
    copy_execution_scripts()

    specs = [
        ("no_tolerance", "no_tolerance", "pr_output_no_tolerance", "roc_output_no_tolerance", "0 bp tolerance"),
        ("100bp_tolerance", "100bp_tolerance", "pr_output_100bp_tolerance", "roc_output_100bp_tolerance", "100 bp tolerance"),
    ]

    all_rows = []
    for folder_slug, tsv_slug, pr_sub, roc_sub, title_bit in specs:
        subdir = OUT_ROOT / folder_slug
        subdir.mkdir(parents=True, exist_ok=True)
        df = collect_rows(tsv_slug, pr_sub, roc_sub)
        tsv_path = subdir / "tss_class_auprc_auroc.tsv"
        df.drop(columns=["_pr_tsv", "_roc_txt"], errors="ignore").to_csv(
            tsv_path, sep="\t", index=False
        )
        df.rename(
            columns={"_pr_tsv": "pr_curve_data_tsv", "_roc_txt": "auroc_summary_txt"},
        ).to_csv(subdir / "tss_class_auprc_auroc_with_sources.tsv", sep="\t", index=False)
        plot_pair(df, subdir, tsv_slug, title_bit)
        plot_grouped_three_metrics(df, subdir, tsv_slug, title_bit)
        all_rows.append(df)
        print(df.drop(columns=["_pr_tsv", "_roc_txt"], errors="ignore").to_string(index=False))
        print(f"\nWrote {tsv_path} and charts in {subdir}\n")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.drop(columns=["_pr_tsv", "_roc_txt"], errors="ignore").to_csv(
        OUT_ROOT / "tss_class_auprc_auroc_all_tolerances.tsv", sep="\t", index=False
    )
    combined.rename(
        columns={"_pr_tsv": "pr_curve_data_tsv", "_roc_txt": "auroc_summary_txt"},
    ).to_csv(
        OUT_ROOT / "tss_class_auprc_auroc_all_tolerances_with_sources.tsv",
        sep="\t",
        index=False,
    )
    print(f"Combined table: {OUT_ROOT / 'tss_class_auprc_auroc_all_tolerances.tsv'}")


if __name__ == "__main__":
    main()
