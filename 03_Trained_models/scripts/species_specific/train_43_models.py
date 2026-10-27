#!/usr/bin/env python3
"""
Train 43 species-specific LSTM models (Type 1: 35+s+10, random negatives only).
Uses v3 architecture from 03_model_training_v3 (binary_crossentropy, sigmoid output).
Data: 02_dataset_generation_v3_1to10/output/75-25/train/{species}/35+s+10/random/

All sequences in the species train split are used for gradient updates (no validation_split
inside model.fit). Hyperparameters: EPOCHS, BATCH_SIZE via env.

Progress file allows resume after crash.
"""

import os, sys, time, json, logging, traceback, importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import optimizers
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # GBactpro_BH

# ── species list: ref-2 order, full strain names (matches train/test folder names) ──
_cfg = PROJECT_ROOT / "03_model_training_v3" / "config" / "type_configs.py"
_spec = importlib.util.spec_from_file_location("type_configs_v3_43", _cfg)
_tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tc)
TARGET_SPECIES = list(_tc.SPECIES_43_REF2_ORDER)
N_SPECIES = len(TARGET_SPECIES)

# ── data root ─────────────────────────────────────────────────────────
BASE_DATA_DIR = PROJECT_ROOT / "02_dataset_generation_v3_1to10" / "output" / "75-25"
REGION = "35+s+10"
NEG_TYPE = "random"
MAXLEN = 31

# ── training hyperparameters (GPU-fast overrides via env) ─────────────
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LR = 0.001

MODEL_DIR = SCRIPT_DIR / "models"
PROGRESS_FILE = SCRIPT_DIR / "training_progress.json"

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
        logging.FileHandler(LOG_DIR / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────
def readfasta(path):
    ids, seqs = [], []
    for rec in SeqIO.parse(str(path), "fasta"):
        ids.append(rec.id)
        seqs.append(str(rec.seq))
    return ids, seqs

def encode(seq):
    m = {"A": [1,0,0,0], "T": [0,1,0,0], "C": [0,0,1,0], "G": [0,0,0,1]}
    return [m.get(c, [0,0,0,0]) for c in seq]

def build_model(seq_size):
    inp = Input(batch_shape=(None, seq_size, 4), name="input")
    x = Masking(mask_value=0.0, input_shape=(seq_size, 4))(inp)
    x = Conv1D(32, 3, padding="same", activation="relu", kernel_regularizer="L1L2")(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, 5, padding="same", activation="relu", kernel_regularizer="L1L2")(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer="L1L2")(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 5, padding="same", activation="relu", kernel_regularizer="L1L2")(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(1024, activation="relu", return_sequences=False))(x)
    x = Dense(1024, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer=optimizers.Adam(LR), loss="binary_crossentropy", metrics=["binary_accuracy"])
    return model

def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "failed": []}

def save_progress(p):
    PROGRESS_FILE.write_text(json.dumps(p, indent=2))

# ── train one species ─────────────────────────────────────────────────
def train_species(species):
    d = BASE_DATA_DIR / "train" / species / REGION / NEG_TYPE
    pos_path = d / "positive.fasta"
    neg_path = d / f"negative_{NEG_TYPE}.fasta"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(f"Missing FASTA: {d}")

    _, p_seqs = readfasta(pos_path)
    _, n_seqs = readfasta(neg_path)
    data = [encode(s) for s in p_seqs] + [encode(s) for s in n_seqs]
    labels = np.array([1]*len(p_seqs) + [0]*len(n_seqs), dtype=np.float32)
    data = pad_sequences(data, padding="post", maxlen=MAXLEN)

    model = build_model(MAXLEN)
    t0 = time.time()
    model.fit(data, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)
    elapsed = time.time() - t0

    out = MODEL_DIR / species
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    return elapsed, len(p_seqs), len(n_seqs)

# ── main ──────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 70)
    logger.info(f"Training {N_SPECIES} species-specific LSTM models (Type 1: 35+s+10, random)")
    logger.info(f"EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  (full train split, no validation_split)")
    logger.info(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    logger.info("=" * 70)

    progress = load_progress()
    done = set(progress["completed"])

    for i, sp in enumerate(TARGET_SPECIES, 1):
        if sp in done:
            logger.info(f"[{i}/{N_SPECIES}] SKIP {sp}")
            continue
        logger.info(f"[{i}/{N_SPECIES}] Training: {sp}")
        try:
            elapsed, np_, nn_ = train_species(sp)
            progress["completed"].append(sp)
            save_progress(progress)
            logger.info(f"  done {elapsed:.1f}s  pos={np_} neg={nn_}")
        except Exception as e:
            logger.error(f"  FAILED: {e}\n{traceback.format_exc()}")
            progress["failed"].append({"species": sp, "error": str(e), "ts": datetime.now().isoformat()})
            save_progress(progress)

    logger.info("=" * 70)
    logger.info(f"DONE — completed={len(progress['completed'])} failed={len(progress['failed'])}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
