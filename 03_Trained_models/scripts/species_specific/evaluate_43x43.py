#!/usr/bin/env python3
"""
43x43 cross-species LSTM evaluation (Type 1: 35+s+10, random negatives).
For every trained model (43 species), predict on every species' test split.
Metrics: AUROC, AUPRC.  Outputs: CSV matrices + v2 heatmaps.

Speed strategy:
 1. Pre-load + cache all 43 test datasets (one-hot encoded, padded).
 2. Load each model once, predict on all 43 test sets, then discard model.
"""

import os, sys, time, json, csv, logging, traceback, importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import pad_sequences
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # GBactpro_BH

# ── species list: ref-2 order, full strain names (matches train/test folder names) ──
_cfg = PROJECT_ROOT / "03_model_training_v3" / "config" / "type_configs.py"
_spec = importlib.util.spec_from_file_location("type_configs_v3_43", _cfg)
_tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tc)
TARGET_SPECIES = list(_tc.SPECIES_43_REF2_ORDER)
N_SPECIES = len(TARGET_SPECIES)

# ── paths / constants ─────────────────────────────────────────────────
BASE_DATA_DIR = PROJECT_ROOT / "02_dataset_generation_v3_1to10" / "output" / "75-25"
REGION = "35+s+10"
NEG_TYPE = "random"
MAXLEN = 31
MODEL_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "cross_species_evaluation"
BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "256"))

# ── GPU setup ─────────────────────────────────────────────────────────
try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# ── logging ───────────────────────────────────────────────────────────
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────
def readfasta_seqs(path):
    seqs = []
    for rec in SeqIO.parse(str(path), "fasta"):
        seqs.append(str(rec.seq))
    return seqs

def encode_seqs(seqs):
    m = {"A": [1,0,0,0], "T": [0,1,0,0], "C": [0,0,1,0], "G": [0,0,0,1]}
    return [[m.get(c, [0,0,0,0]) for c in s] for s in seqs]

# ── pre-load all test datasets ────────────────────────────────────────
def load_all_test_data():
    """Return dict: species -> (X_test, y_test)"""
    cache = {}
    for sp in TARGET_SPECIES:
        d = BASE_DATA_DIR / "test" / sp / REGION / NEG_TYPE
        pos_f = d / "positive.fasta"
        neg_f = d / f"negative_{NEG_TYPE}.fasta"
        if not pos_f.exists() or not neg_f.exists():
            logger.warning(f"Missing test data for {sp}, skipping")
            continue
        p = readfasta_seqs(pos_f)
        n = readfasta_seqs(neg_f)
        X = pad_sequences(encode_seqs(p + n), padding="post", maxlen=MAXLEN)
        y = np.array([1]*len(p) + [0]*len(n), dtype=np.float32)
        cache[sp] = (X, y)
    logger.info(f"Cached test data for {len(cache)}/{len(TARGET_SPECIES)} species")
    return cache

# ── evaluation core ───────────────────────────────────────────────────
def evaluate_all():
    """Load each model once; predict on all cached test sets."""
    test_cache = load_all_test_data()

    auroc_mat = pd.DataFrame(np.nan, index=TARGET_SPECIES, columns=TARGET_SPECIES)
    auprc_mat = pd.DataFrame(np.nan, index=TARGET_SPECIES, columns=TARGET_SPECIES)
    all_results = []

    for mi, model_sp in enumerate(TARGET_SPECIES, 1):
        mpath = MODEL_DIR / model_sp
        if not mpath.exists():
            logger.warning(f"[{mi}/{N_SPECIES}] No model for {model_sp}")
            continue
        logger.info(f"[{mi}/{N_SPECIES}] Loading model: {model_sp}")
        model = keras.models.load_model(str(mpath))

        for test_sp in TARGET_SPECIES:
            if test_sp not in test_cache:
                continue
            X, y = test_cache[test_sp]
            preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0).flatten()
            try:
                auroc = float(roc_auc_score(y, preds))
            except ValueError:
                auroc = float("nan")
            try:
                auprc = float(average_precision_score(y, preds))
            except ValueError:
                auprc = float("nan")
            auroc_mat.loc[model_sp, test_sp] = auroc
            auprc_mat.loc[model_sp, test_sp] = auprc
            all_results.append({
                "model_species": model_sp, "test_species": test_sp,
                "auroc": auroc, "auprc": auprc,
            })

        del model
        tf.keras.backend.clear_session()
        logger.info(f"  done — {mi}/{N_SPECIES} models evaluated")

    return auroc_mat, auprc_mat, all_results

# ── heatmap (v2 style) ───────────────────────────────────────────────
def create_heatmap_v2(matrix: pd.DataFrame, title: str, output_path: str, metric_name: str):
    data_min = float(np.nanmin(matrix.values))
    data_max = float(np.nanmax(matrix.values))
    n = len(matrix)

    matrix_flipped = matrix.iloc[::-1]
    row_avgs = matrix.mean(axis=1).iloc[::-1]
    col_avgs = matrix.mean(axis=0)

    ext = np.full((n + 2, n + 2), np.nan)
    ext[2:, :n] = matrix_flipped.values
    ext[2:, n + 1] = row_avgs.values
    ext[0, :n] = col_avgs.values

    row_labels = ["Col Avg", ""] + list(matrix_flipped.index)
    col_labels = list(matrix.columns) + ["", "Row Avg"]

    mask = np.zeros_like(ext, dtype=bool)
    mask[:, n] = True
    mask[1, :] = True

    fig, ax = plt.subplots(figsize=(22, 20))
    norm = Normalize(vmin=data_min, vmax=data_max)

    sns.heatmap(ext, annot=False, cmap="Greens", vmin=data_min, vmax=data_max,
                cbar=False, square=False, linewidths=0.1, linecolor="gray",
                xticklabels=col_labels, yticklabels=row_labels, mask=mask, ax=ax)

    sm = ScalarMappable(cmap="Greens", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label(metric_name, fontsize=12, fontweight="bold")
    cbar.set_ticks([data_min, data_max])
    cbar.set_ticklabels([f"{data_min:.4f}", f"{data_max:.4f}"])
    cbar.ax.tick_params(labelsize=10)

    fs = 7
    for i in range(n):
        v = ext[i + 2, n + 1]
        if not np.isnan(v):
            ax.text(n + 1.5, i + 2.5, f"{v:.3f}", ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="white")
    for j in range(n):
        v = ext[0, j]
        if not np.isnan(v):
            ax.text(j + 0.5, 0.5, f"{v:.3f}", ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="white", rotation=90)

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Test Species (Columns)", fontsize=12, fontweight="bold")
    plt.ylabel("Model Species (Rows) — Flipped", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved: {output_path}")

# ── main ──────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 70)
    logger.info(f"{N_SPECIES}×{N_SPECIES} Cross-Species Evaluation (Type 1: 35+s+10, random)")
    logger.info(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    logger.info("=" * 70)

    t0 = time.time()
    auroc_mat, auprc_mat, results = evaluate_all()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    auroc_mat.to_csv(OUTPUT_DIR / "cross_species_auroc_matrix.csv")
    auprc_mat.to_csv(OUTPUT_DIR / "cross_species_auprc_matrix.csv")
    logger.info("Saved CSV matrices")

    with open(OUTPUT_DIR / "cross_species_results.json", "w") as f:
        json.dump({"results": results, "species_order": TARGET_SPECIES,
                    "timestamp": datetime.now().isoformat()}, f, indent=2)

    create_heatmap_v2(auroc_mat,
                      f"Cross-Species AUROC ({N_SPECIES}×{N_SPECIES}) — Type 1: 35+s+10",
                      str(OUTPUT_DIR / "cross_species_auroc_heatmap_v2.png"), "AUROC")
    create_heatmap_v2(auprc_mat,
                      f"Cross-Species AUPRC ({N_SPECIES}×{N_SPECIES}) — Type 1: 35+s+10",
                      str(OUTPUT_DIR / "cross_species_auprc_heatmap_v2.png"), "AUPRC")

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"DONE in {elapsed:.0f}s")
    ok = [r for r in results if not np.isnan(r["auroc"])]
    logger.info(f"Evaluations: {len(ok)}/{N_SPECIES ** 2}")
    if ok:
        logger.info(f"Mean AUROC={np.mean([r['auroc'] for r in ok]):.4f}  "
                     f"Mean AUPRC={np.mean([r['auprc'] for r in ok]):.4f}")
    logger.info("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal: {e}\n{traceback.format_exc()}")
        sys.exit(1)
