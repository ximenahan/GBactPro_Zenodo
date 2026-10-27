# Deep learning model hyperparameters (03_model_training_v3)

Model family: CNN + BiLSTM (binary classification)

| Hyper-parameter | Value |
|---|---|
| Input encoding | One-hot (A/T/C/G -> 4 channels) |
| Input tensor shape | `(None, seq_len, 4)` |
| Sequence padding | `post` |
| Masking | `Masking(mask_value=0.0)` |
| Conv block 1 | Conv1D(32, k=3, same, relu, stride=1, L1L2) -> BN -> Conv1D(32, k=5, same, relu, stride=1, L1L2) -> BN -> MaxPool1D(2) -> Dropout(0.5) |
| Conv block 2 | Conv1D(64, k=3, same, relu, stride=1, L1L2) -> BN -> Conv1D(64, k=5, same, relu, stride=1, L1L2) -> BN -> MaxPool1D(2) -> Dropout(0.5) |
| Recurrent layer | Bidirectional LSTM, units=1024, activation=`relu`, `return_sequences=False` |
| Dense hidden layer | Dense(1024, relu) |
| Output layer | Dense(1, sigmoid) |
| Loss function | `binary_crossentropy` |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Metric (training/eval) | `binary_accuracy` (train/eval script also reports AUROC during testing) |
| Epochs (default) | `5` |
| Batch size (default) | `32` |
| Validation split (default) | `0.20` |
| Early stopping | Enabled by env flag (`ENABLE_EARLY_STOPPING`, default `1`) |
| Early stopping patience | `1` (env `EARLY_STOPPING_PATIENCE`) |
| Data split source | Train from `02_dataset_generation_v3_1to10/output/75-25/train/...`; evaluate on `.../test/...` |
| Negative types supported | `random`, `shuffle-mix` |

## Type-specific sequence length (maxlen)

| Type | Region | Maxlen |
|---|---|---|
| type1_35s10 | `35+s+10` | 31 |
| type2_up35s10_original | `up+35+s+10_original` | 81 |
| type3_up35s10_new | `up+35+s+10_new` | 51 |
| type4_total50 | `up+35+s+10+down/total_50/up_8_12` | 50 |
| type5_up35_40 | `up+35+s+10+down/up_35_40` | 100 |
| type6_40bp_upstream | `40bp_upstream_tss` | 40 |
| type7_minus60_plus21 | `minus60_plus21_tss` | 81 |
| type8_35s10_leftpad40 | `35+s+10_leftpad40` | 40 |

## Runtime-overridable knobs (env vars)

- `EPOCHS`
- `BATCH_SIZE`
- `VALIDATION_SPLIT`
- `ENABLE_EARLY_STOPPING`
- `EARLY_STOPPING_PATIENCE`
