#!/usr/bin/env python3
"""
Sequence logos for v5 O2O best-F1 (100 bp tol) TP vs FN reference TSS subsets.

Reads BPM scan TSVs from ../promoter_architecture_scan/output/{tp,fn}/.
Same plotting logic as v4 tss_class_subsets_pr_roc_dense_type1/sequence_logos.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import logomaker
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SCAN_DIR = BASE_DIR.parent / "promoter_architecture_scan" / "output"
OUT_ROOT = BASE_DIR / "output"

COLORS = {"A": "#CC0000", "T": "#FFAA00", "C": "#0000CC", "G": "#00AA00"}
SPACER_LENS = [16, 17, 18]
TSS_CLASSES = ["tp", "fn"]

CLASS_LABELS = {
    "tp": "TP — matched reference TSS (v5 O2O, best F1, ±100 bp)",
    "fn": "FN — unmatched reference TSS (v5 O2O, best F1, ±100 bp)",
}


def seqs_to_info_matrix(seqs):
    if not seqs:
        return pd.DataFrame()
    length = len(seqs[0])
    counts = {b: np.zeros(length) for b in "ATCG"}
    for seq in seqs:
        for i, c in enumerate(seq.upper()):
            if c in counts:
                counts[c][i] += 1
    count_df = pd.DataFrame(counts)
    info_mat = logomaker.transform_matrix(count_df, from_type="counts", to_type="information")
    return info_mat


def plot_single_logo(info_mat, title, out_path, ylim_top, annotation="", figsize=(4.5, 2.2)):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    if info_mat.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        try:
            logomaker.Logo(info_mat, ax=ax, color_scheme=COLORS)
        except Exception as e:
            print(f"  WARN logomaker error: {e}", file=sys.stderr)
            ax.text(0.5, 0.5, "Logo error", ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel("bits", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    if annotation:
        ax.annotate(
            annotation,
            xy=(0.98, 0.96),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7,
            fontstyle="italic",
        )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_spacer_assembled(info_mats, ns, pcts, title_prefix, out_path, ylim_top):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 2.5), gridspec_kw={"wspace": 0.35})
    for idx, slen in enumerate(SPACER_LENS):
        ax = axes[idx]
        mat = info_mats.get(slen, pd.DataFrame())
        n = ns.get(slen, 0)
        pct = pcts.get(slen, 0.0)
        if mat.empty or n == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        else:
            try:
                logomaker.Logo(mat, ax=ax, color_scheme=COLORS)
            except Exception as e:
                print(f"  WARN logomaker error for spacer {slen}: {e}", file=sys.stderr)
                ax.text(0.5, 0.5, "Logo error", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_ylim(0, ylim_top)
        ax.set_title(f"{slen}-bp", fontsize=10, fontweight="bold", pad=6)
        annotation = f"n={n}  ({pct:.1f}%)"
        ax.annotate(
            annotation,
            xy=(0.98, 0.96),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7,
            fontstyle="italic",
        )
        ax.set_ylabel("bits" if idx == 0 else "", fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color("black")
        ax.tick_params(axis="x", labelsize=5)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(title_prefix + " — Spacer Region", fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def process_one_class(tss_class, ylim_m35, ylim_m10, ylim_spacer):
    label = CLASS_LABELS[tss_class]
    scan_tsv = SCAN_DIR / tss_class / f"Escherichia coli MG1655_{tss_class}.tsv"
    out_dir = OUT_ROOT / tss_class
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tss_class}] Loading scan result: {scan_tsv}")
    df = pd.read_csv(scan_tsv, sep="\t")
    if "label" in df.columns:
        df = df[df["label"] == "promoter"]

    minus35_seqs = df["Minus35"].dropna().tolist()
    minus10_seqs = df["Minus10"].dropna().tolist()
    n_m35 = len(minus35_seqs)
    n_m10 = len(minus10_seqs)

    m35_mat = seqs_to_info_matrix(minus35_seqs)
    m10_mat = seqs_to_info_matrix(minus10_seqs)

    spacer_mats = {}
    spacer_ns = {}
    spacer_pcts = {}
    total_spacers = len(df["Spacer"].dropna())
    for slen in SPACER_LENS:
        seqs = [s for s in df["Spacer"].dropna() if len(s) == slen]
        spacer_ns[slen] = len(seqs)
        spacer_pcts[slen] = 100.0 * len(seqs) / total_spacers if total_spacers else 0.0
        spacer_mats[slen] = seqs_to_info_matrix(seqs)

    plot_single_logo(m35_mat, f"{label} — -35 Region", out_dir / "minus35_logo.png", ylim_m35, annotation=f"n={n_m35}")
    plot_single_logo(m10_mat, f"{label} — -10 Region", out_dir / "minus10_logo.png", ylim_m10, annotation=f"n={n_m10}")

    for slen in SPACER_LENS:
        mat = spacer_mats[slen]
        n = spacer_ns.get(slen, 0)
        pct = spacer_pcts.get(slen, 0.0)
        plot_single_logo(
            mat,
            f"{label} — Spacer {slen}-bp",
            out_dir / f"spacer_{slen}bp_logo.png",
            ylim_spacer,
            annotation=f"n={n}  ({pct:.1f}%)",
        )

    plot_spacer_assembled(spacer_mats, spacer_ns, spacer_pcts, label, out_dir / "spacer_assembled.png", ylim_spacer)

    stats = {
        "class": tss_class,
        "n_m35": n_m35,
        "n_m10": n_m10,
        "m35_max": float(m35_mat.sum(axis=1).max()) if not m35_mat.empty else 0.0,
        "m10_max": float(m10_mat.sum(axis=1).max()) if not m10_mat.empty else 0.0,
    }
    for slen in SPACER_LENS:
        mat = spacer_mats[slen]
        stats[f"spacer{slen}_max"] = float(mat.sum(axis=1).max()) if not mat.empty else 0.0
        stats[f"spacer{slen}_n"] = spacer_ns[slen]
        stats[f"spacer{slen}_pct"] = spacer_pcts[slen]

    print(f"[{tss_class}] Done: {out_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tss-class", default=None, choices=TSS_CLASSES, help="tp or fn only")
    parser.add_argument("--ylim-m35", type=float, default=None)
    parser.add_argument("--ylim-m10", type=float, default=None)
    parser.add_argument("--ylim-spacer", type=float, default=None)
    args = parser.parse_args()

    fixed = (args.ylim_m35, args.ylim_m10, args.ylim_spacer)
    if any(x is not None for x in fixed) and not all(x is not None for x in fixed):
        parser.error("Either omit all ylim args or provide all three.")

    classes = [args.tss_class] if args.tss_class else TSS_CLASSES

    print("Pass 1: scanning data for raw maxima / y-axis limits ...")
    all_maxima = {"m35": [], "m10": [], "spacer": []}
    for cls in TSS_CLASSES:
        scan_tsv = SCAN_DIR / cls / f"Escherichia coli MG1655_{cls}.tsv"
        if not scan_tsv.exists():
            if cls in classes:
                print(f"  WARN: {scan_tsv} not found, skipping for y-limit calc")
            continue
        df = pd.read_csv(scan_tsv, sep="\t")
        if "label" in df.columns:
            df = df[df["label"] == "promoter"]

        m35_seqs = df["Minus35"].dropna().tolist()
        m10_seqs = df["Minus10"].dropna().tolist()
        m35_mat = seqs_to_info_matrix(m35_seqs)
        m10_mat = seqs_to_info_matrix(m10_seqs)
        if not m35_mat.empty:
            all_maxima["m35"].append(m35_mat.sum(axis=1).max())
        if not m10_mat.empty:
            all_maxima["m10"].append(m10_mat.sum(axis=1).max())
        for slen in SPACER_LENS:
            sp_seqs = [s for s in df["Spacer"].dropna() if len(s) == slen]
            sp_mat = seqs_to_info_matrix(sp_seqs)
            if not sp_mat.empty:
                all_maxima["spacer"].append(sp_mat.sum(axis=1).max())

    def ylim_from_values(vals, cap=2.0, min_ylim=0.05, pad_frac=0.03):
        if not vals:
            return float(min_ylim)
        m = float(max(vals))
        m = min(m * (1.0 + pad_frac), cap)
        return max(min_ylim, float(np.ceil(m * 100) / 100))

    raw_m35 = max(all_maxima["m35"], default=0.0)
    raw_m10 = max(all_maxima["m10"], default=0.0)
    raw_spacer = max(all_maxima["spacer"], default=0.0)

    print(f"  raw max (bits): -35={raw_m35:.4f}, -10={raw_m10:.4f}, spacer={raw_spacer:.4f}")

    if all(x is not None for x in fixed):
        ylim_m35, ylim_m10, ylim_spacer = args.ylim_m35, args.ylim_m10, args.ylim_spacer
        ylim_source = "fixed_cli"
        print(f"  y-limits (shared, fixed): -35={ylim_m35}, -10={ylim_m10}, spacer={ylim_spacer}")
    else:
        ylim_m35 = ylim_from_values(all_maxima["m35"])
        ylim_m10 = ylim_from_values(all_maxima["m10"])
        ylim_spacer = ylim_from_values(all_maxima["spacer"])
        ylim_source = "computed"
        print(f"  y-limits (shared, computed): -35={ylim_m35}, -10={ylim_m10}, spacer={ylim_spacer}")

    print("Pass 2: generating logos ...")
    all_stats = []
    for cls in classes:
        try:
            stats = process_one_class(cls, ylim_m35, ylim_m10, ylim_spacer)
            all_stats.append(stats)
        except Exception as e:
            print(f"  FAILED {cls}: {e}", file=sys.stderr)

    meta = {
        "ylim_m35": ylim_m35,
        "ylim_m10": ylim_m10,
        "ylim_spacer": ylim_spacer,
        "ylim_source": ylim_source,
        "raw_max_bits_m35": raw_m35,
        "raw_max_bits_m10": raw_m10,
        "raw_max_bits_spacer": raw_spacer,
        "stats": all_stats,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"All done. Outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()
