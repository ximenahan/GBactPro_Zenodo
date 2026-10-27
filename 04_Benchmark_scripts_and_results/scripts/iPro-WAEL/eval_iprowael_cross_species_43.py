#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iPro-WAEL full pipeline (fairer): ilearnplus RF features (2466-d) + word2vec CNN + weighted ensemble.

- RF: CKSNAP + Mismatch + RCKmer + PCPseTNC + TPCP (see iprowael_rf_features_ilearn.py)
- CNN: same 616-dim word2vec pooling as original repo
- Weights: scipy SLSQP log-loss on 20%% stratified validation split (same logic as Weighted_average.py)
- Default test: type 7 / 81 bp, minus60_plus21_tss / random.
- Type 2: --region-dir up+35+s+10_original, left-pad variable 77-81 bp to 81 (--fixed-seq-len 81).

Requires: conda env iprowael (TF2 + Keras 2.3 + sklearn 0.24 + torch + ilearnplus + PyQt5).
"""

from __future__ import print_function

import argparse
import csv
import json
import multiprocessing
import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENT_DIR = os.path.dirname(SCRIPT_DIR)
GBH_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
)

IPROWAEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(GBH_ROOT), "model_comparision", "iPro-WAEL")
)
DATA_ROOT = os.path.join(GBH_ROOT, "02_dataset_generation_v3_1to10", "output", "75-25")
SPECIES_TSV = os.path.join(
    GBH_ROOT,
    "02_dataset_generation_v3_1to10",
    "tss_order_tables",
    "tss_species_table_43_ref2_order.tsv",
)

DEFAULT_REGION_DIR = "minus60_plus21_tss"
NEG_NAME = "random"
CNN_POOL = 616

MODEL_LABEL_BASE = "iPro-WAEL (E.coli RF+CNN+weights, ilearn+ RF)"
DEFAULT_DATASET_CAPTION = "type 7 / 81 bp"
LINE_COLOR = "#9467bd"


def sanitize_dna(seq):
    allowed = {"A": "A", "T": "T", "C": "C", "G": "G", "a": "A", "t": "T", "c": "C", "g": "G"}
    return "".join(allowed.get(c, "N") for c in str(seq))


def normalize_to_fixed_bp(seq, fixed_len, pad_char="A"):
    s = sanitize_dna(seq)
    pc = pad_char.upper() if pad_char else "A"
    if pc not in "ATCG":
        pc = "A"
    if len(s) >= fixed_len:
        return s[-fixed_len:]
    return pc * (fixed_len - len(s)) + s

# Local RF extraction (ilearnplus)
sys.path.insert(0, SCRIPT_DIR)
from iprowael_rf_features import extract_rf_features_for_sequences  # noqa: E402


def load_species_43(tsv_path):
    species = []
    with open(tsv_path, "r") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            name = (row.get("species") or row.get("Species") or "").strip()
            if not name and row:
                name = str(next(iter(row.values()))).strip()
            if name:
                species.append(name)
    return species


def load_cnn_hdf5_safely(cnn_model_path):
    import h5py
    from keras.models import model_from_json

    with h5py.File(cnn_model_path, "r") as f:
        mc = f.attrs["model_config"]
        if isinstance(mc, bytes):
            mc = mc.decode("utf-8")
        elif not isinstance(mc, str):
            mc = json.dumps(mc)
        model_config = json.loads(mc)
        model = model_from_json(json.dumps(model_config))
        if "model_weights" in f:
            weights_group = f["model_weights"]
            for layer in model.layers:
                if layer.name in weights_group:
                    layer_group = weights_group[layer.name]
                    layer_weights = []
                    for weight_name in sorted(layer_group.keys()):
                        weight_dataset = layer_group[weight_name]
                        if isinstance(weight_dataset, h5py.Dataset):
                            layer_weights.append(np.array(weight_dataset))
                    if layer_weights:
                        layer.set_weights(layer_weights)
        else:
            raise RuntimeError("No model_weights in hdf5.")
    return model


def extract_word2vec_features(sequences, index_file, word2vec_file):
    with open(index_file, "r") as f:
        index = f.read().strip().split(" ")
    word2vec = np.loadtxt(word2vec_file)
    k = 5
    kmer_list = []
    for seq in sequences:
        seq = str(seq).upper()
        kmers = []
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if kmer in index:
                kmers.append(index.index(kmer))
            else:
                kmers.append(0)
        kmer_list.append(kmers)

    feature_word2vec = []
    with torch.no_grad():
        pool = nn.AdaptiveAvgPool1d(CNN_POOL)
        for kmers in kmer_list:
            feature_seq = []
            for kmer_index in kmers:
                for j in word2vec[kmer_index].tolist():
                    feature_seq.append(j)
            feature_seq_tensor = torch.Tensor(feature_seq)
            feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)
            feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)
            feature_seq_tensor_avg = pool(feature_seq_tensor)
            feature_seq_numpy = feature_seq_tensor_avg.numpy()
            feature_seq_numpy = np.squeeze(feature_seq_numpy)
            feature_seq_numpy = np.squeeze(feature_seq_numpy)
            feature_word2vec.append(feature_seq_numpy.tolist())

    return np.array(feature_word2vec, dtype=np.float64)


def minmax_rows(X):
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    r = mx - mn
    r[r == 0] = 1.0
    return (X - mn) / r


def ensemble_weighted_proba(
    y_val,
    rf_val,
    cnn_val,
    rf_test,
    cnn_test,
    iterations_num=50,
):
    """Same optimization as Weighted_average.weight; returns 1d test probabilities."""
    y_val = np.asarray(y_val, dtype=np.float64)
    rf_val = np.asarray(rf_val, dtype=np.float64).flatten()
    cnn_val = np.asarray(cnn_val, dtype=np.float64).flatten()
    rf_test = np.asarray(rf_test, dtype=np.float64).flatten()
    cnn_test = np.asarray(cnn_test, dtype=np.float64).flatten()

    test_proba = np.vstack([rf_test, cnn_test])

    def make_loss(yv, rfv, cnv):
        def loss(w):
            p = w[0] * rfv + w[1] * cnv
            p = np.clip(p, 1e-15, 1.0 - 1e-15)
            return log_loss(yv, p)

        return loss

    bounds = [(0, 1), (0, 1)]
    constraints = ({"type": "eq", "fun": lambda w: 1.0 - float(np.sum(w))},)
    best_loss = float("inf")
    best_w = None
    for _ in range(iterations_num):
        pw = np.random.rand(2)
        pw = pw / pw.sum()
        result = minimize(
            make_loss(y_val, rf_val, cnn_val),
            pw,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result["fun"] < best_loss:
            best_loss = result["fun"]
            best_w = result["x"]
    w = best_w
    final = w[0] * test_proba[0] + w[1] * test_proba[1]
    return final, w


def read_fasta_seqs(path, fixed_seq_len=None, pad_char="A"):
    raw = [str(rec.seq).upper() for rec in SeqIO.parse(path, "fasta")]
    if fixed_seq_len is None:
        return raw
    return [
        normalize_to_fixed_bp(s, fixed_seq_len, pad_char).replace("N", "A")
        for s in raw
    ]


def evaluate_species(
    species,
    test_root,
    rf_model,
    cnn_model,
    index_file,
    word2vec_file,
    dir_name=None,
    region_dir=None,
    fixed_seq_len=None,
    pad_char="A",
):
    rdir = region_dir or DEFAULT_REGION_DIR
    sub = os.path.join(test_root, "test", dir_name or species, rdir, NEG_NAME)
    pos_f = os.path.join(sub, "positive.fasta")
    neg_f = os.path.join(sub, "negative_{0}.fasta".format(NEG_NAME))
    if not os.path.isfile(pos_f) or not os.path.isfile(neg_f):
        print("  SKIP {0}: missing FASTA".format(species))
        return None
    pos = read_fasta_seqs(pos_f, fixed_seq_len, pad_char)
    neg = read_fasta_seqs(neg_f, fixed_seq_len, pad_char)
    if not pos or not neg:
        print("  SKIP {0}: empty FASTA".format(species))
        return None
    seqs = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)

    print("  RF features (ilearnplus)...", flush=True)
    x_rf = extract_rf_features_for_sequences(seqs)
    x_cnn = extract_word2vec_features(seqs, index_file, word2vec_file)

    x_rf_n = minmax_rows(x_rf)
    x_cnn_n = minmax_rows(x_cnn)

    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=10
    )

    rf_val = rf_model.predict_proba(x_rf_n[va_idx])[:, 1]
    cnn_val = cnn_model.predict(
        np.expand_dims(x_cnn_n[va_idx], 2), verbose=0
    ).flatten()
    rf_full = rf_model.predict_proba(x_rf_n)[:, 1]
    cnn_full = cnn_model.predict(np.expand_dims(x_cnn_n, 2), verbose=0).flatten()

    preds, w = ensemble_weighted_proba(
        y[va_idx], rf_val, cnn_val, rf_full, cnn_full
    )
    print("  ensemble weights (RF, CNN): {0}".format(w), flush=True)

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


def split_into_n_chunks(lst, n):
    # type: (List[str], int) -> List[List[str]]
    """Split lst into up to n non-empty chunks (as even as possible)."""
    lst = list(lst)
    if not lst:
        return []
    n = max(1, min(int(n), len(lst)))
    k, m = divmod(len(lst), n)
    out = []
    i = 0
    for j in range(n):
        size = k + (1 if j < m else 0)
        if size:
            out.append(lst[i : i + size])
            i += size
    return out


def _configure_worker_process():
    """Avoid GPU contention and BLAS oversubscription when many workers run."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _worker_eval_chunk(payload):
    # type: (Tuple[List[str], str, str, str, Dict, Dict]) -> List[Optional[Dict]]
    """
    Run evaluate_species for each species in chunk. Loads RF+CNN once per chunk
    (one process per chunk; use --workers to set number of parallel chunks).
    """
    species_chunk, test_root, iprowael_root, model_species, sp_map, eval_opts = payload
    _configure_worker_process()
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import joblib
    import torch

    torch.set_num_threads(1)

    index_file = os.path.join(iprowael_root, "index_promoters.txt")
    word2vec_file = os.path.join(iprowael_root, "word2vec_promoters.txt")
    cnn_path = os.path.join(iprowael_root, "model", model_species, "model_cnn.hdf5")
    rf_path = os.path.join(iprowael_root, "model", model_species, "RF.model")

    print(
        "Worker PID {0}: loading models for {1} species...".format(
            os.getpid(), len(species_chunk)
        ),
        flush=True,
    )
    rf_model = joblib.load(rf_path)
    try:
        from keras.models import load_model

        cnn_model = load_model(cnn_path, compile=False)
    except Exception:
        cnn_model = load_cnn_hdf5_safely(cnn_path)
    cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    results = []
    for sp in species_chunk:
        dn = sp_map.get(sp, sp) if sp_map else sp
        print(
            "Worker PID {0}: [{1}] -> {2}".format(os.getpid(), sp, dn),
            flush=True,
        )
        try:
            r = evaluate_species(
                sp,
                test_root,
                rf_model,
                cnn_model,
                index_file,
                word2vec_file,
                dir_name=dn,
                region_dir=eval_opts.get("region_dir", DEFAULT_REGION_DIR),
                fixed_seq_len=eval_opts.get("fixed_seq_len"),
                pad_char=eval_opts.get("pad_char", "A"),
            )
        except Exception as e:
            print(
                "Worker PID {0}: ERROR {1}: {2}".format(os.getpid(), sp, e),
                flush=True,
            )
            import traceback

            traceback.print_exc()
            r = None
        if r:
            print(
                "  AUROC={0:.4f}  AUPRC={1:.4f}  n={2}+{3}".format(
                    r["auroc"],
                    r["auprc"],
                    r["n_pos"],
                    r["n_neg"],
                ),
                flush=True,
            )
        results.append(r)
    return results


def plot_lines(species_list, aurocs, auprcs, out_dir, model_label, dataset_type_caption):
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
        "Cross-Species AUROC — iPro-WAEL full (43 species, {0}, random negatives)".format(
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
        "Cross-Species AUPRC — iPro-WAEL full (43 species, {0}, random negatives)".format(
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
    ax.set_title("Cross-Species AUROC — iPro-WAEL (full)")
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
    ax.set_title("Cross-Species AUPRC — iPro-WAEL (full)")
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
    parser.add_argument("--iprowael-root", default=IPROWAEL_DIR)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--species-tsv", default=SPECIES_TSV)
    parser.add_argument("--output-dir", default=os.path.join(SEGMENT_DIR, "output"))
    parser.add_argument("--model-species", default="E.coli")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (species split into chunks; RF+CNN loaded once per chunk). "
        "Use 8–16 on a many-core CPU; each worker is memory-heavy (~TF+sklearn). Default 1 (sequential).",
    )
    parser.add_argument(
        "--region-dir",
        default=DEFAULT_REGION_DIR,
        help="FASTA subdirectory under test/{species}/ (type 7: minus60_plus21_tss; type 2: up+35+s+10_original)",
    )
    parser.add_argument(
        "--fixed-seq-len",
        type=int,
        default=0,
        help="If >0, left-pad sequences to this length. If 0, use raw FASTA (legacy type 7). Default 0.",
    )
    parser.add_argument(
        "--pad-char",
        default="A",
        help="Left padding nucleotide when --fixed-seq-len > 0. Default A.",
    )
    parser.add_argument(
        "--dataset-type-caption",
        default=DEFAULT_DATASET_CAPTION,
        help="Short label for figures/tables (default: type 7 / 81 bp).",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.iprowael_root)
    if not os.path.isdir(root):
        sys.exit("iPro-WAEL root not found: {0}".format(root))

    out_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    species_list = load_species_43(os.path.abspath(args.species_tsv))
    print("Species count: {0}".format(len(species_list)))

    fixed_len = int(args.fixed_seq_len) if int(args.fixed_seq_len) > 0 else None
    eval_opts = {
        "region_dir": args.region_dir,
        "fixed_seq_len": fixed_len,
        "pad_char": args.pad_char or "A",
    }
    model_label_full = "{0}, {1}".format(MODEL_LABEL_BASE, args.dataset_type_caption)

    index_file = os.path.join(root, "index_promoters.txt")
    word2vec_file = os.path.join(root, "word2vec_promoters.txt")
    cnn_path = os.path.join(root, "model", args.model_species, "model_cnn.hdf5")
    rf_path = os.path.join(root, "model", args.model_species, "RF.model")
    if not os.path.isfile(cnn_path) or not os.path.isfile(rf_path):
        sys.exit("Missing RF.model or model_cnn.hdf5 under model/{0}".format(args.model_species))

    test_root = os.path.abspath(args.data_root)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))))
    from species_dir_resolver import build_species_dir_map
    sp_map = build_species_dir_map(species_list, os.path.join(test_root, "test"))
    print("Species dir map built ({0} remapped)".format(
        sum(1 for s in species_list if sp_map.get(s, s) != s)))

    rows = []

    if args.workers is not None and int(args.workers) > 1:
        n_workers = int(args.workers)
        chunks = split_into_n_chunks(species_list, n_workers)
        print(
            "Parallel mode: {0} workers, {1} chunks (species per chunk: {2})".format(
                len(chunks),
                len(chunks),
                [len(c) for c in chunks],
            ),
            flush=True,
        )
        payloads = [
            (chunk, test_root, root, args.model_species, sp_map, eval_opts)
            for chunk in chunks
        ]
        # spawn avoids fork+TF/torch issues; Pool works on Python 3.6 (no mp_context on Executor)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(payloads)) as pool:
            chunk_results = pool.map(_worker_eval_chunk, payloads)
        for cr in chunk_results:
            for r in cr:
                if r:
                    rows.append(r)
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        import tensorflow as tf

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        import joblib

        print("Loading RF: {0}".format(rf_path))
        rf_model = joblib.load(rf_path)
        print("Loading CNN: {0}".format(cnn_path))
        try:
            from keras.models import load_model

            cnn_model = load_model(cnn_path, compile=False)
        except Exception:
            cnn_model = load_cnn_hdf5_safely(cnn_path)
        cnn_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        for i, sp in enumerate(species_list, 1):
            dn = sp_map.get(sp, sp)
            print("[{0}/{1}] {2} -> {3}".format(i, len(species_list), sp, dn), flush=True)
            r = evaluate_species(
                sp,
                test_root,
                rf_model,
                cnn_model,
                index_file,
                word2vec_file,
                dir_name=dn,
                region_dir=eval_opts["region_dir"],
                fixed_seq_len=eval_opts["fixed_seq_len"],
                pad_char=eval_opts["pad_char"],
            )
            if r:
                rows.append(r)
                print(
                    "  AUROC={0:.4f}  AUPRC={1:.4f}  n={2}+{3}".format(
                        r["auroc"],
                        r["auprc"],
                        r["n_pos"],
                        r["n_neg"],
                    ),
                    flush=True,
                )

    if not rows:
        sys.exit("No species evaluated — check paths.")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "cross_species_metrics.csv"), index=False)

    ecoli_name = "Escherichia coli"
    eco_df = df[df["species"] == ecoli_name]
    if not eco_df.empty:
        t1 = eco_df[["species", "auroc", "auprc", "n_pos", "n_neg"]]
        t1.to_csv(os.path.join(out_dir, "table1_ecoli_iPro-WAEL.csv"), index=False)
        md_path = os.path.join(out_dir, "table1_ecoli_iPro-WAEL.md")
        with open(md_path, "w") as f:
            f.write(
                "# iPro-WAEL — E. coli test ({0}, random negatives, full pipeline)\n\n".format(
                    args.dataset_type_caption
                )
            )
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
                "Model": model_label_full,
                "Avg AUROC": "{0:.4f}".format(np.mean(valid_au)) if valid_au else "N/A",
                "Avg AUPRC": "{0:.4f}".format(np.mean(valid_ap)) if valid_ap else "N/A",
                "Species count": len(valid_au),
            }
        ]
    )
    t2.to_csv(os.path.join(out_dir, "table2_average_iPro-WAEL.csv"), index=False)
    with open(os.path.join(out_dir, "table2_average_iPro-WAEL.md"), "w") as f:
        f.write("# iPro-WAEL — Average (full RF+CNN+weights)\n\n")
        try:
            f.write(t2.to_markdown(index=False))
        except Exception:
            f.write(t2.to_string(index=False))

    plot_lines(
        species_list,
        aurocs_m,
        auprcs_m,
        out_dir,
        model_label_full,
        args.dataset_type_caption,
    )

    print("Done.")
    print("Output: {0}".format(out_dir))


if __name__ == "__main__":
    main()
