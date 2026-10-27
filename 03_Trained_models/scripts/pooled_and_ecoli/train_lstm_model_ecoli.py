#!/usr/bin/env python3
"""
LSTM Model Training Script with maxlen + Masking support.
V3: Uses 02_dataset_generation_v3_1to10 outputs (75/25 train/test split)

Trains LSTM model on E.coli data, then evaluates on all species.
Supports variable-length sequences through padding and masking.
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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import config
import importlib.util
config_path = parent_dir / "config" / "type_configs.py"
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")
spec = importlib.util.spec_from_file_location("type_configs", config_path)
type_configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(type_configs)
TYPE_CONFIGS = type_configs.TYPE_CONFIGS
TRAINING_SPECIES = type_configs.TRAINING_SPECIES
ALL_SPECIES = type_configs.ALL_SPECIES
NEGATIVE_TYPES = type_configs.NEGATIVE_TYPES
TRAINING_CONFIG = type_configs.TRAINING_CONFIG

# Runtime overrides (do not change model architecture)
EPOCHS = int(os.environ.get("EPOCHS", TRAINING_CONFIG['epochs']))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", TRAINING_CONFIG['batch_size']))
VALIDATION_SPLIT = float(os.environ.get("VALIDATION_SPLIT", TRAINING_CONFIG['validation_split']))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "1"))
ENABLE_EARLY_STOPPING = os.environ.get("ENABLE_EARLY_STOPPING", "1") == "1"

# Enable predictable GPU memory behavior when GPUs are present
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# Setup logging
def setup_logging(log_file):
    """Setup logging to file and console."""
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def readfasta(file_path):
    """Read FASTA file and return IDs and sequences."""
    ids = []
    seqs = []
    for seq_record in SeqIO.parse(file_path, "fasta"):
        ids.append(seq_record.id)
        seqs.append(str(seq_record.seq))
    return pd.DataFrame(ids), pd.DataFrame(seqs)

def encode_sequence(seq):
    """Convert DNA sequence to one-hot encoding."""
    encoding = []
    for c in seq:
        if c == 'A':
            encoding.append([1, 0, 0, 0])
        elif c == 'T':
            encoding.append([0, 1, 0, 0])
        elif c == 'C':
            encoding.append([0, 0, 1, 0])
        elif c == 'G':
            encoding.append([0, 0, 0, 1])
        else:
            # Handle unknown nucleotides (N, etc.) - use random
            encoding.append([0, 0, 0, 0])  # Will be masked
    return encoding

def get_compiled_model(seq_size):
    """Create compiled LSTM model with specified sequence size."""
    input_layer = Input(batch_shape=(None, seq_size, 4), name='input')
    mask_layer = Masking(mask_value=0., input_shape=(seq_size, 4))(input_layer)
    
    # CNN Block 1
    conv_layer_1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', 
                          strides=1, kernel_regularizer='L1L2')(mask_layer)
    bn_1 = BatchNormalization()(conv_layer_1)
    conv_layer_2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', 
                          strides=1, kernel_regularizer='L1L2')(bn_1)
    bn_2 = BatchNormalization()(conv_layer_2)
    avg_pool_1 = MaxPool1D(pool_size=2)(bn_2)
    dp_1 = Dropout(0.5)(avg_pool_1)
    
    # CNN Block 2
    conv_layer_3 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', 
                          strides=1, kernel_regularizer='L1L2')(dp_1)
    bn_3 = BatchNormalization()(conv_layer_3)
    conv_layer_4 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu', 
                          strides=1, kernel_regularizer='L1L2')(bn_3)
    bn_4 = BatchNormalization()(conv_layer_4)
    avg_pool_2 = MaxPool1D(pool_size=2)(bn_4)
    dp_2 = Dropout(0.5)(avg_pool_2)
    
    # LSTM Block (preserve temporal dimension from CNN)
    lstm_1 = Bidirectional(LSTM(units=1024, activation='relu', return_sequences=False))(dp_2)

    # Dense Layers
    dense_1 = Dense(units=1024, activation='relu')(lstm_1)
    dense_2 = Dense(1, activation='sigmoid')(dense_1)
    
    model = Model(inputs=input_layer, outputs=dense_2)
    model.compile(
        optimizer=optimizers.Adam(TRAINING_CONFIG['learning_rate']),
        loss=TRAINING_CONFIG['loss'],
        metrics=TRAINING_CONFIG['metrics']
    )
    
    return model

def train_model(type_name, negative_type='random'):
    """Train LSTM model for a specific type."""
    config = TYPE_CONFIGS[type_name]
    script_dir = Path(__file__).parent
    log_file = script_dir.parent / "logs" / f"train_{type_name}_{negative_type}.log"
    logger = setup_logging(str(log_file))
    
    logger.info(f"Starting training for {type_name} with {negative_type} negative samples")
    logger.info(f"Sequence maxlen: {config['maxlen']}")
    logger.info(f"Region: {config['region']}")
    logger.info(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}")
    logger.info(f"Training params: epochs={EPOCHS}, batch_size={BATCH_SIZE}, val_split={VALIDATION_SPLIT}, early_stopping={ENABLE_EARLY_STOPPING}")
    
    script_dir = Path(__file__).resolve().parent
    dataset_subdir = config.get("dataset_dir", "02_dataset_generation_v3_1to10")
    base_data_dir = script_dir.parent.parent / dataset_subdir / "output" / "75-25"
    train_data_dir = base_data_dir / "train" / TRAINING_SPECIES / config['region'] / negative_type
    
    # Load training data
    logger.info(f"Loading training data from: {train_data_dir}")
    logger.info(f"Absolute path: {train_data_dir.resolve()}")
    p_id_df, p_seq_df = readfasta(str(train_data_dir / "positive.fasta"))
    
    # Handle different negative file naming conventions
    if negative_type == 'shuffle-mix':
        neg_file = train_data_dir / "negative_shuffle-mix.fasta"
    else:
        neg_file = train_data_dir / f"negative_{negative_type}.fasta"
    
    if not neg_file.exists():
        logger.error(f"Negative file not found: {neg_file}")
        raise FileNotFoundError(f"Negative file not found: {neg_file}")
    
    n_id_df, n_seq_df = readfasta(str(neg_file))
    
    logger.info(f"Positive samples: {len(p_seq_df)}")
    logger.info(f"Negative samples: {len(n_seq_df)}")
    
    # Encode sequences
    logger.info("Encoding sequences...")
    train_data = []
    train_labels = []
    
    for i in range(len(p_seq_df[0])):
        seq = p_seq_df[0][i]
        encoded = encode_sequence(seq)
        train_data.append(encoded)
        train_labels.append(1)
    
    for i in range(len(n_seq_df[0])):
        seq = n_seq_df[0][i]
        encoded = encode_sequence(seq)
        train_data.append(encoded)
        train_labels.append(0)
    
    # Pad sequences to maxlen
    logger.info(f"Padding sequences to maxlen={config['maxlen']}...")
    train_data = pad_sequences(train_data, padding=config['padding'], maxlen=config['maxlen'])
    train_labels = np.array(train_labels, dtype=np.float32)
    
    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Training labels shape: {train_labels.shape}")
    
    # Create model
    logger.info("Creating model...")
    model = get_compiled_model(seq_size=config['maxlen'])
    model.summary(print_fn=logger.info)
    
    # Train model
    logger.info("Training model...")
    start_time = time.time()
    
    callbacks = []
    if ENABLE_EARLY_STOPPING:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
        )

    history = model.fit(
        train_data, 
        train_labels, 
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time/60:.2f} minutes")
    
    # Save model - save in scripts/models/ to match where evaluation looks
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir / "models" / "lstm" / "1-1" / config['region'] / negative_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to: {model_dir}")
    model.save(str(model_dir))
    logger.info("Model saved successfully")
    
    return model, model_dir

def evaluate_model(type_name, negative_type='random'):
    """Evaluate trained model on all species."""
    config = TYPE_CONFIGS[type_name]
    script_dir = Path(__file__).resolve().parent
    log_file = script_dir.parent / "logs" / f"eval_{type_name}_{negative_type}.log"
    logger = setup_logging(str(log_file))
    
    logger.info(f"Starting evaluation for {type_name} with {negative_type} negative samples")
    
    # Load model - models are saved in scripts/models/ when run from scripts directory
    # Try both possible locations
    model_dir = script_dir / "models" / "lstm" / "1-1" / config['region'] / negative_type
    if not model_dir.exists():
        model_dir = script_dir.parent / "models" / "lstm" / "1-1" / config['region'] / negative_type
    logger.info(f"Loading model from: {model_dir}")
    logger.info(f"Model directory exists: {model_dir.exists()}")
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    model = load_model(str(model_dir))
    
    dataset_subdir = config.get("dataset_dir", "02_dataset_generation_v3_1to10")
    base_data_dir = script_dir.parent.parent / dataset_subdir / "output" / "75-25"
    logger.info(f"Base data directory: {base_data_dir}")
    logger.info(f"Base data directory exists: {base_data_dir.exists()}")
    
    # Results storage
    arr_accuracy = []
    arr_auc = []
    
    # Evaluate on all species
    for species in ALL_SPECIES:
        logger.info(f"Evaluating on: {species}")
        
        try:
            test_data_dir = base_data_dir / "test" / species / config['region'] / negative_type
            
            # Check if data exists
            pos_file = test_data_dir / "positive.fasta"
            logger.debug(f"Checking for data at: {pos_file}")
            if not pos_file.exists():
                logger.warning(f"Data not found for {species} at {test_data_dir}, skipping...")
                logger.debug(f"Directory exists: {test_data_dir.exists()}")
                if test_data_dir.exists():
                    logger.debug(f"Files in directory: {list(test_data_dir.iterdir())}")
                continue
            
            # Load test data
            p_id_df, p_seq_df = readfasta(str(test_data_dir / "positive.fasta"))
            
            # Handle different negative file naming conventions
            if negative_type == 'shuffle-mix':
                neg_file = test_data_dir / "negative_shuffle-mix.fasta"
            else:
                neg_file = test_data_dir / f"negative_{negative_type}.fasta"
            
            if not neg_file.exists():
                logger.warning(f"Negative file not found for {species}: {neg_file}, skipping...")
                continue
            
            n_id_df, n_seq_df = readfasta(str(neg_file))
            
            # Encode sequences
            test_data = []
            test_labels = []
            
            for i in range(len(p_seq_df[0])):
                seq = p_seq_df[0][i]
                encoded = encode_sequence(seq)
                test_data.append(encoded)
                test_labels.append(1)
            
            for i in range(len(n_seq_df[0])):
                seq = n_seq_df[0][i]
                encoded = encode_sequence(seq)
                test_data.append(encoded)
                test_labels.append(0)
            
            # Pad sequences (same as training)
            test_data = pad_sequences(test_data, padding=config['padding'], maxlen=config['maxlen'])
            
            # Predict
            pred_test = model.predict(test_data, verbose=0)
            pred_test_classes = (pred_test.flatten() >= 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, pred_test_classes)
            
            # AUC (need probabilities for class 1)
            pred_probs = pred_test.flatten()
            fpr, tpr, threshold = roc_curve(test_labels, pred_probs, pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            logger.info(f"  Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")
            
            arr_accuracy.append([species, accuracy])
            arr_auc.append([species, roc_auc])
            
            # Save per-species results
            result_dir = model_dir / "result" / species
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            pd.DataFrame(pred_test).to_csv(result_dir / "sequences_predictions.csv", index=False, header=False)
            
            # Save performance metrics
            with open(result_dir / "performance.txt", "w") as f:
                f.write(f"Testing Accuracy: {accuracy}\n")
                f.write(f"AUROC: {roc_auc}\n")
                f.write(f"Confusion Matrix:\n")
                cm = tf.math.confusion_matrix(test_labels, pred_test_classes, num_classes=2)
                f.write(str(cm.numpy()) + "\n")
            
        except Exception as e:
            logger.error(f"Error evaluating {species}: {str(e)}")
            continue
    
    # Save summary results
    df_accuracy = pd.DataFrame(arr_accuracy, columns=['name', 'accuracy'])
    df_auc = pd.DataFrame(arr_auc, columns=['name', 'auc'])
    
    summary_dir = model_dir / "result"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    df_accuracy.to_csv(summary_dir / "accuracy.csv", index=False)
    df_auc.to_csv(summary_dir / "auc.csv", index=False)
    
    logger.info(f"Evaluation complete. Results saved to: {summary_dir}")
    logger.info(f"Average accuracy: {df_accuracy['accuracy'].mean():.4f}")
    logger.info(f"Average AUC: {df_auc['auc'].mean():.4f}")

def main():
    """Main function."""
    evaluate_only = '--evaluate-only' in sys.argv
    if evaluate_only:
        sys.argv.remove('--evaluate-only')
    
    if len(sys.argv) < 2:
        print("Usage: python train_lstm_model.py <type_name> [negative_type] [--evaluate-only]")
        print("Available types:")
        for type_name in TYPE_CONFIGS.keys():
            print(f"  - {type_name}")
        sys.exit(1)
    
    type_name = sys.argv[1]
    negative_type = sys.argv[2] if len(sys.argv) > 2 else 'random'
    
    if type_name not in TYPE_CONFIGS:
        print(f"Error: Unknown type '{type_name}'")
        sys.exit(1)
    
    if negative_type not in NEGATIVE_TYPES:
        print(f"Error: Unknown negative type '{negative_type}'")
        sys.exit(1)
    
    if evaluate_only:
        # Only evaluate (skip training)
        print(f"Re-evaluating model: {type_name} ({negative_type})")
        evaluate_model(type_name, negative_type)
        print(f"\n✅ Evaluation complete for {type_name} ({negative_type})")
    else:
        # Train and evaluate
        model, model_dir = train_model(type_name, negative_type)
        evaluate_model(type_name, negative_type)
        print(f"\n✅ Training and evaluation complete for {type_name} ({negative_type})")
        print(f"Model saved to: {model_dir}")

if __name__ == "__main__":
    main()
