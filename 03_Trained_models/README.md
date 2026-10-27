# 03 — Trained models

All checkpoints are TensorFlow SavedModels. Each `variables.data-00000-of-00001` file is about 127 MB. GitHub rejects blobs over 100 MB, so those weight files are stored with Git LFS. The files in this folder are the real weights; `git lfs pull` after cloning restores them. The architecture is a CNN–BiLSTM: one-hot DNA, masking of zero padding, Conv1D blocks of 32 and 64 filters, bidirectional LSTM of 1024 units, a 1024-unit dense layer, and a sigmoid output. Loss is binary cross-entropy. Optimizer is Adam at learning rate 0.001. The recorded stack is Python 3.8 and TensorFlow/Keras 2.13 (`environment_gbactpro.yml`).

Intermediate epoch checkpoints were not stored by the training scripts. Each directory is the final `model.save()` output. Per-species score tables were moved to section 04 instead of being duplicated inside the checkpoints.

## Species-specific Type 1 models

`species_specific/Type1_35_spacer_10_27-31bp/` has one directory per species in `species_43.tsv` (43 models). Folder names use underscores (`Escherichia_coli_MG1655`); the table still uses spaces. Training is `scripts/species_specific/train_43_models.py`.

- Data: that species only, `Training/Type1.../positive.fasta` and `negative_random.fasta`.
- Encoding: one-hot, post-padding to 31 bp.
- Fit: 5 epochs, batch size 64, shuffle, no validation split.
- `training_progress.json` lists 43 completed species and no failures.

No species-specific checkpoints were found for Types 2–8. `03_model_training_v3/scripts/models/lstm/1-1/` looks per-type, but `train_lstm_model.py` trains on *E. coli* and then evaluates other species. Those weights are under `ecoli_trained/`.

## Pooled 43-species models

`pooled_43_species/` has Types 1–8, random negatives only. Training script: `scripts/pooled_and_ecoli/train_lstm_model_pooled43.py`. Each model uses the pooled 75% training splits and is evaluated on each species’ test split. Padding is post-padding to the type maximum in `type_configs_v3.py` (31, 81, 51, 44, 50, 100, 40, and 81 bp).

The Type 1 variable file is byte-identical to `models/type1_35s10_random` in the public GBactPro repository.

## *E. coli*-trained models

`ecoli_trained/` is an additional set required by the tool-comparison table. Each network was trained on the *E. coli* MG1655 training split and then scored on all 43 test species. Default fit settings are 5 epochs, batch 32, validation split 0.20, and early stopping with patience 1.

These weights differ from both the pooled Type 1 model and the species-specific *E. coli* model.

## Genome-wide model

`genome_wide/type1_pre_padding_max29bp_random/` is the scan checkpoint.

- Input: Type 1, but only sequences of at most 29 bp (spacer 15–17 bp). Longer examples were dropped.
- Padding: zeros on the left, sequence right-aligned (pre-padding) to 29 bp.
- Data: `training_sequences_100pct_pooled_type1/`. This is a separate 1:10 genomic-random sample of all annotated promoters, not the concatenation of the 75/25 FASTA files. The training log records 60,771 positives and 43 species. There is no held-out test split.
- `model.fit` still used `validation_split=0.2`, batch 32, and early stopping. That internal validation split is a difference from a literal “100% of rows used for gradient updates.”
- The variable file is byte-identical to public `models/type1_pre_max29bp_random`, and it is different from the pooled Type 1 checkpoint.
- The source directory is named `genome_use_49`; the log’s species list is the published 43.

Training code and the short log are in `scripts/genome_wide/` and `genome_wide/logs/`.
