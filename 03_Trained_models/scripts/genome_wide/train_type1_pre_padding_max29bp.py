#!/usr/bin/env python3
"""
Type1 (35+s+10) with PRE (beginning) padding, maxlen 29 bp, training on ≤29 bp examples only.

Pool: same species list as genome_use type1 training. From each species FASTA, keep only
sequences with length ≤29 bp (spacer 15–17 bp; drop 30–31 bp = spacer 18–19).

Pre padding: one-hot sequence is right-aligned in the window; zeros pad the left to length 29.
(Masking in the model treats pad positions as masked.)
"""
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers

import importlib.util

SCRIPT_DIR = Path(__file__).resolve().parent
GBH = SCRIPT_DIR.parent.parent
TRAIN_V3_DIR = SCRIPT_DIR.parent

config_path = TRAIN_V3_DIR / "config" / "type_configs.py"
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")
spec = importlib.util.spec_from_file_location("type_configs", config_path)
type_configs_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(type_configs_mod)

TRAINING_SPECIES_LIST_49 = type_configs_mod.TRAINING_SPECIES_LIST_49
MODEL_CHECKPOINT_SUBDIR = type_configs_mod.MODEL_CHECKPOINT_SUBDIR
TRAINING_CONFIG = type_configs_mod.TRAINING_CONFIG
NEGATIVE_TYPES = type_configs_mod.NEGATIVE_TYPES

MAX_RAW_LEN = 29
MAXLEN = 29
REGION_35_LEN = 6
REGION_10_LEN = 6
MIDDLE_LEN = MAXLEN - REGION_35_LEN - REGION_10_LEN  # 17
REGION_NAME = "35+s+10"
SAVE_REGION = "35+s+10_pre_max29bp"

EPOCHS = int(os.environ.get("EPOCHS", TRAINING_CONFIG["epochs"]))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", TRAINING_CONFIG["batch_size"]))
VALIDATION_SPLIT = float(os.environ.get("VALIDATION_SPLIT", TRAINING_CONFIG["validation_split"]))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "1"))
ENABLE_EARLY_STOPPING = os.environ.get("ENABLE_EARLY_STOPPING", "1") == "1"

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

_OH_BYTE = np.zeros((256, 4), dtype=np.float32)
for _b, _v in (("A", (1, 0, 0, 0)), ("T", (0, 1, 0, 0)), ("C", (0, 0, 1, 0)), ("G", (0, 0, 0, 1))):
    _OH_BYTE[ord(_b)] = _v
    _OH_BYTE[ord(_b.lower())] = _v


def encode_sequence(seq: str) -> np.ndarray:
    if not seq:
        return np.zeros((0, 4), dtype=np.float32)
    b = np.frombuffer(seq.encode("latin-1"), dtype=np.uint8)
    return _OH_BYTE[b].copy()


def pre_pad(onehot: np.ndarray) -> np.ndarray:
    """Right-align sequence; pad zeros on the left to length MAXLEN."""
    seq_len = onehot.shape[0]
    if seq_len >= MAXLEN:
        return onehot[-MAXLEN:]
    out = np.zeros((MAXLEN, 4), dtype=np.float32)
    out[MAXLEN - seq_len :] = onehot
    return out


def setup_logging(log_file):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(str(log_path)), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def readfasta(file_path):
    ids, seqs = [], []
    for rec in SeqIO.parse(file_path, "fasta"):
        ids.append(rec.id)
        seqs.append(str(rec.seq))
    return pd.DataFrame(ids), pd.DataFrame(seqs)


def get_compiled_model(seq_size):
    input_layer = Input(batch_shape=(None, seq_size, 4), name="input")
    mask_layer = Masking(mask_value=0.0, input_shape=(seq_size, 4))(input_layer)
    x = Conv1D(32, 3, padding="same", activation="relu", strides=1, kernel_regularizer="L1L2")(mask_layer)
    x = BatchNormalization()(x)
    x = Conv1D(32, 5, padding="same", activation="relu", strides=1, kernel_regularizer="L1L2")(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(64, 3, padding="same", activation="relu", strides=1, kernel_regularizer="L1L2")(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 5, padding="same", activation="relu", strides=1, kernel_regularizer="L1L2")(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(units=1024, activation="relu", return_sequences=False))(x)
    x = Dense(units=1024, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(TRAINING_CONFIG["learning_rate"]),
        loss=TRAINING_CONFIG["loss"],
        metrics=TRAINING_CONFIG["metrics"],
    )
    return model


def get_model_save_dir(negative_type):
    scripts_dir = TRAIN_V3_DIR / "scripts"
    return scripts_dir / "models" / "lstm" / MODEL_CHECKPOINT_SUBDIR / SAVE_REGION / negative_type


def main():
    negative_type = sys.argv[1] if len(sys.argv) > 1 else "random"
    if negative_type not in NEGATIVE_TYPES:
        print(f"Error: Unknown negative type '{negative_type}'")
        sys.exit(1)

    log_file = SCRIPT_DIR / "logs" / f"train_type1_pre_max29bp_{negative_type}.log"
    logger = setup_logging(str(log_file))

    logger.info("=" * 60)
    logger.info("  type1_35s10 — PRE PADDING, maxlen 29, raw len ≤29 only")
    logger.info("=" * 60)
    logger.info(f"  Max raw len: {MAX_RAW_LEN} bp (drop 30–31 bp examples)")
    logger.info(f"  Model input: {MAXLEN} bp (zeros padded on the left; sequence right-aligned)")
    logger.info(f"  Raw 5'→3' occupies the right side of the window; shorter raw → more left pad")
    logger.info(f"  Negative:  {negative_type}")
    logger.info(f"  Save to:   {get_model_save_dir(negative_type)}")
    logger.info(f"  Pool:      {len(TRAINING_SPECIES_LIST_49)} species")
    logger.info(f"  Params:    epochs={EPOCHS} batch={BATCH_SIZE} val_split={VALIDATION_SPLIT}")
    logger.info(f"  GPUs:      {tf.config.list_physical_devices('GPU')}")

    base_data_dir = GBH / "02_dataset_generation_v3_genome_use" / "output" / "train_all_1to10"
    logger.info(f"  Data dir:  {base_data_dir}")

    train_data = []
    train_labels = []
    total_pos = 0
    total_neg = 0
    total_pos_skip = 0
    total_neg_skip = 0

    for species in TRAINING_SPECIES_LIST_49:
        train_data_dir = base_data_dir / species / REGION_NAME / negative_type
        pos_path = train_data_dir / "positive.fasta"
        neg_file = train_data_dir / (
            "negative_shuffle-mix.fasta" if negative_type == "shuffle-mix" else f"negative_{negative_type}.fasta"
        )
        if not pos_path.exists():
            raise FileNotFoundError(f"Missing: {pos_path}")
        if not neg_file.exists():
            raise FileNotFoundError(f"Missing: {neg_file}")

        _, p_df = readfasta(str(pos_path))
        _, n_df = readfasta(str(neg_file))
        np_i, nn_i = len(p_df[0]), len(n_df[0])
        total_pos += np_i
        total_neg += nn_i

        pos_kept = neg_kept = 0
        pos_skip = neg_skip = 0

        for i in range(np_i):
            raw = p_df[0][i]
            if len(raw) > MAX_RAW_LEN:
                pos_skip += 1
                continue
            oh = encode_sequence(raw)
            train_data.append(pre_pad(oh))
            train_labels.append(1)
            pos_kept += 1

        for i in range(nn_i):
            raw = n_df[0][i]
            if len(raw) > MAX_RAW_LEN:
                neg_skip += 1
                continue
            oh = encode_sequence(raw)
            train_data.append(pre_pad(oh))
            train_labels.append(0)
            neg_kept += 1

        total_pos_skip += pos_skip
        total_neg_skip += neg_skip
        logger.info(f"  {species}: pos {pos_kept}/{np_i} (skip {pos_skip})  neg {neg_kept}/{nn_i} (skip {neg_skip})")

    logger.info(
        f"Pooled raw counts — positive: {total_pos}  negative: {total_neg}  "
        f"skipped (>{MAX_RAW_LEN}bp): pos {total_pos_skip} neg {total_neg_skip}"
    )

    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)
    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Training labels shape: {train_labels.shape}")

    logger.info("Creating model...")
    model = get_compiled_model(seq_size=MAXLEN)
    model.summary(print_fn=logger.info)

    logger.info("Training model...")
    t0 = time.time()

    callbacks = []
    if ENABLE_EARLY_STOPPING:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
            )
        )

    model.fit(
        train_data,
        train_labels,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(f"Training completed in {(time.time() - t0) / 60:.2f} minutes")

    model_dir = get_model_save_dir(negative_type)
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to: {model_dir}")
    model.save(str(model_dir))
    logger.info("Model saved successfully")

    print(f"\nModel saved: {model_dir}")


if __name__ == "__main__":
    main()
