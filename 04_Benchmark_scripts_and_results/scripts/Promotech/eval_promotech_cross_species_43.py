#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-species AUROC / AUPRC for Promotech (RF-HOT / RF-TETRA).

- Default test data type 6: 40 bp - .../test/{species}/40bp_upstream_tss/random/
- Type 8: 40 bp .../test/{species}/35+s+10_leftpad40/random/ (same target length and Promotech parsers)
- Negatives: negative_random.fasta (random), same order as iPro-MP / Prompt workflow.
- Model: joblib under Promotech/models/{model}.model; inference via sequences.process_sequences.parseSequences.

Compatible with promotech_env (Python 3.6+).
"""

from __future__ import print_function

import argparse
import contextlib
import csv
import io
import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import average_precision_score, roc_auc_score

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENT_DIR = os.path.dirname(SCRIPT_DIR)
# .../GBactpro_BH
GBH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))

PROMOTECH_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(GBH_ROOT),
        "model_comparision",
        "Promotech",
    )
)
DATA_ROOT = os.path.join(GBH_ROOT, "02_dataset_generation_v3_1to10", "output", "75-25")
SPECIES_TSV = os.path.join(
    GBH_ROOT,
    "02_dataset_generation_v3_1to10",
    "tss_order_tables",
    "tss_species_table_43_ref2_order.tsv",
)

DEFAULT_REGION_DIR = "40bp_upstream_tss"
NEG_NAME = "random"
DEFAULT_TARGET_LEN = 40
DEFAULT_DATASET_CAPTION = "type 6 / 40 bp"


def model_label_for(model_type, dataset_caption=DEFAULT_DATASET_CAPTION):
    return "Promotech ({0}, {1})".format(model_type, dataset_caption)


LINE_COLOR = "#ff7f0e"


def load_species_43(tsv_path):
    # type: (str) -> List[str]
    species = []
    with open(tsv_path, "r") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            name = (row.get("species") or row.get("Species") or "").strip()
            if not name and row:
                name = str(next(iter(row.values()))).strip()
            if name:
                species.append(name)
    return species


def pad_or_truncate(seq, target_len=DEFAULT_TARGET_LEN):
    # type: (str, int) -> str
    """Repeat-sequence padding if shorter (Promotech execution plan); truncate if longer."""
    s = "".join(c if c in "ATCGatcg" else "N" for c in str(seq)).upper()
    if len(s) == 0:
        return "N" * target_len
    if len(s) >= target_len:
        return s[:target_len]
    reps = (target_len // len(s)) + 1 if len(s) > 0 else 1
    padded = (s * reps)[:target_len]
    return padded


def read_fasta_seqs(path, target_len):
    # type: (str, int) -> List[str]
    out = []
    for rec in SeqIO.parse(path, "fasta"):
        out.append(pad_or_truncate(str(rec.seq), target_len))
    return out


def load_promotech_model(model_type):
    # type: (str) -> object
    import joblib

    path = os.path.join("models", "{0}.model".format(model_type))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def predict_probs_batch(seqs, model, model_type, sequence_length):
    # type: (List[str], object, str, int) -> np.ndarray
    """parseSequences + predict_proba; cwd must be PROMOTECH_ROOT."""
    from sequences.process_sequences import parseSequences

    devnull = open(os.devnull, "w")

    def silent(*args, **kwargs):
        return

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            X = parseSequences(
                seqs,
                sequence_length=sequence_length,
                data_type=model_type,
                log_file=devnull,
                print_fn=silent,
            )
        y_probs = model.predict_proba(X)
        if y_probs.shape[1] == 2:
            return y_probs[:, 1].astype(np.float64)
        return y_probs[:, 0].astype(np.float64)
    finally:
        devnull.close()


def evaluate_species(
    species,
    test_root,
    model,
    model_type,
    dir_name=None,
    region_dir=None,
    target_len=DEFAULT_TARGET_LEN,
):
    # type: (str, str, object, str, Optional[str], Optional[str], int) -> Optional[Dict]
    rdir = region_dir or DEFAULT_REGION_DIR
    sub = os.path.join(test_root, "test", dir_name or species, rdir, NEG_NAME)
    pos_f = os.path.join(sub, "positive.fasta")
    neg_f = os.path.join(sub, "negative_{0}.fasta".format(NEG_NAME))
    if not os.path.isfile(pos_f) or not os.path.isfile(neg_f):
        print("  SKIP {0}: missing positive or negative FASTA".format(species))
        return None
    pos = read_fasta_seqs(pos_f, target_len)
    neg = read_fasta_seqs(neg_f, target_len)
    if not pos or not neg:
        print("  SKIP {0}: empty FASTA".format(species))
        return None
    seqs = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    preds = predict_probs_batch(seqs, model, model_type, target_len)
    try:
        auroc = float(roc_auc_score(y, preds))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y, preds))
    except ValueError:
        auprc = float("nan")
    return {
        "species": species,
        "auroc": auroc,
        "auprc": auprc,
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def plot_lines(
    species_list,
    aurocs,
    auprcs,
    out_dir,
    model_label,
    dataset_type_caption,
):
    # type: (List[str], List[float], List[float], str, str) -> None
    idx = np.arange(len(species_list))

    fig, axes = plt.subplots(2, 1, figsize=(24, 12))
    axes[0].plot(
        idx,
        aurocs,
        marker="o",
        markersize=3,
        linewidth=2,
        color=LINE_COLOR,
        label=model_label,
    )
    axes[0].set_title(
        "Cross-Species AUROC — Promotech (43 species, {0}, random negatives)".format(
            dataset_type_caption
        )
    )
    axes[0].set_ylabel("AUROC")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(species_list, rotation=90, ha="right", fontsize=7)
    axes[0].set_ylim([0, 1.0])
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(
        idx,
        auprcs,
        marker="s",
        markersize=3,
        linewidth=2,
        color=LINE_COLOR,
        label=model_label,
    )
    axes[1].set_title(
        "Cross-Species AUPRC — Promotech (43 species, {0}, random negatives)".format(
            dataset_type_caption
        )
    )
    axes[1].set_ylabel("AUPRC")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(species_list, rotation=90, ha="right", fontsize=7)
    axes[1].set_ylim([0, 1.0])
    axes[1].legend(loc="best", fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(
        os.path.join(out_dir, "cross_species_auroc_auprc_line_charts.png"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.5,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(
        idx,
        aurocs,
        marker="o",
        markersize=4,
        linewidth=2.5,
        color=LINE_COLOR,
        label=model_label,
    )
    ax.set_title("Cross-Species AUROC — Promotech")
    ax.set_ylabel("AUROC")
    ax.set_xticks(idx)
    ax.set_xticklabels(species_list, rotation=90, ha="right", fontsize=7)
    ax.set_ylim([0, 1.0])
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(
        os.path.join(out_dir, "cross_species_auroc_line_chart.png"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.5,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(
        idx,
        auprcs,
        marker="s",
        markersize=4,
        linewidth=2.5,
        color=LINE_COLOR,
        label=model_label,
    )
    ax.set_title("Cross-Species AUPRC — Promotech")
    ax.set_ylabel("AUPRC")
    ax.set_xticks(idx)
    ax.set_xticklabels(species_list, rotation=90, ha="right", fontsize=7)
    ax.set_ylim([0, 1.0])
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(
        os.path.join(out_dir, "cross_species_auprc_line_chart.png"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.5,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotech-root", default=PROMOTECH_ROOT)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--species-tsv", default=SPECIES_TSV)
    parser.add_argument("--output-dir", default=os.path.join(SEGMENT_DIR, "output"))
    parser.add_argument(
        "--model",
        default="RF-HOT",
        choices=["RF-HOT", "RF-TETRA"],
        help="Promotech sklearn model file name (without .model)",
    )
    parser.add_argument(
        "--region-dir",
        default=DEFAULT_REGION_DIR,
        help=(
            "Subfolder under test/{{species}}/ for FASTA positives/negatives "
            "(default: 40bp_upstream_tss; type 8: 35+s+10_leftpad40)"
        ),
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=DEFAULT_TARGET_LEN,
        help="Sequence length passed to Promotech parseSequences (default 40)",
    )
    parser.add_argument(
        "--dataset-type-caption",
        default=DEFAULT_DATASET_CAPTION,
        help="Short description for plots/CSV labels (default: type 6 / 40 bp)",
    )
    args = parser.parse_args()

    promotech_root = os.path.abspath(args.promotech_root)
    if not os.path.isdir(promotech_root):
        sys.exit("Promotech root not found: {0}".format(promotech_root))

    out_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    species_list = load_species_43(os.path.abspath(args.species_tsv))
    print("Species count: {0}".format(len(species_list)))

    model_label = model_label_for(args.model, args.dataset_type_caption)
    region_dir = args.region_dir
    target_len = args.target_len
    dataset_cap = args.dataset_type_caption
    old_cwd = os.getcwd()
    sys.path.insert(0, promotech_root)
    try:
        os.chdir(promotech_root)
        model = load_promotech_model(args.model)
    except Exception as e:
        os.chdir(old_cwd)
        sys.exit("Failed to load model: {0}".format(e))

    _perf_dir = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    sys.path.insert(0, _perf_dir)
    from species_dir_resolver import build_species_dir_map
    sp_map = build_species_dir_map(species_list, os.path.join(os.path.abspath(args.data_root), "test"))
    print("Species dir map built ({0} remapped)".format(
        sum(1 for s in species_list if sp_map.get(s, s) != s)))

    rows = []
    try:
        for i, sp in enumerate(species_list, 1):
            dn = sp_map.get(sp, sp)
            print("[{0}/{1}] {2} -> {3}".format(i, len(species_list), sp, dn))
            r = evaluate_species(
                sp,
                os.path.abspath(args.data_root),
                model,
                args.model,
                dir_name=dn,
                region_dir=region_dir,
                target_len=target_len,
            )
            if r:
                rows.append(r)
                print(
                    "  AUROC={0:.4f}  AUPRC={1:.4f}  n={2}+{3}".format(
                        r["auroc"],
                        r["auprc"],
                        r["n_pos"],
                        r["n_neg"],
                    )
                )
    finally:
        os.chdir(old_cwd)

    if not rows:
        sys.exit("No species evaluated — check paths.")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "cross_species_metrics.csv"), index=False)

    ecoli_name = "Escherichia coli"
    eco_df = df[df["species"] == ecoli_name]
    if not eco_df.empty:
        t1 = eco_df[["species", "auroc", "auprc", "n_pos", "n_neg"]]
        t1.to_csv(os.path.join(out_dir, "table1_ecoli_Promotech.csv"), index=False)
        md_path = os.path.join(out_dir, "table1_ecoli_Promotech.md")
        with open(md_path, "w") as f:
            f.write("# Promotech — E. coli test ({0}, random negatives)\n\n".format(dataset_cap))
            try:
                f.write(t1.to_markdown(index=False))
            except Exception:
                f.write(t1.to_string(index=False))

    species_set = set(df["species"].values)
    aurocs_m = []
    auprcs_m = []
    for s in species_list:
        if s in species_set:
            aurocs_m.append(float(df[df["species"] == s]["auroc"].values[0]))
            auprcs_m.append(float(df[df["species"] == s]["auprc"].values[0]))
        else:
            aurocs_m.append(float("nan"))
            auprcs_m.append(float("nan"))

    valid_au = [x for x in aurocs_m if np.isfinite(x)]
    valid_ap = [x for x in auprcs_m if np.isfinite(x)]
    t2 = pd.DataFrame(
        [
            {
                "Model": model_label,
                "Avg AUROC": "{0:.4f}".format(np.mean(valid_au)) if valid_au else "N/A",
                "Avg AUPRC": "{0:.4f}".format(np.mean(valid_ap)) if valid_ap else "N/A",
                "Species count": len(valid_au),
            }
        ]
    )
    t2.to_csv(os.path.join(out_dir, "table2_average_Promotech.csv"), index=False)
    with open(os.path.join(out_dir, "table2_average_Promotech.md"), "w") as f:
        f.write("# Promotech — Average across evaluated species\n\n")
        try:
            f.write(t2.to_markdown(index=False))
        except Exception:
            f.write(t2.to_string(index=False))

    plot_lines(species_list, aurocs_m, auprcs_m, out_dir, model_label, dataset_cap)

    print("Done.")
    print("Output: {0}".format(out_dir))


if __name__ == "__main__":
    main()
