# GBactPro research reproducibility archive

This archive contains the datasets, trained models, benchmark results, and genome-wide prediction outputs used for the GBactPro study. It is a data and results archive, not a software release.

GBactPro asks whether short, element-resolved promoter windows are enough to recognize bacterial promoters, and whether a model trained on many species is more accurate than a model trained on *Escherichia coli* alone. Promoters from 43 bacteria were represented in eight sequence windows (Types 1–8). Convolutional–bidirectional-LSTM classifiers were trained on genomic-random negatives at a 1:10 positive-to-negative ratio. The published tool comparison uses a 1:1 test subset. A separate Type 1 model, trained on the pooled promoters with no held-out species split, was slid across the *E. coli* K-12 chromosome.

Nothing in this folder was regenerated. Files were copied from the local analysis directories after the experimental configuration was matched to the sequence definitions and the published metric table. Abandoned runs (shuffle-mix negatives, the local 40 bp left-pad window, other padding trials, and six species outside the published 43) were left out.

## Layout

| Directory | Contents |
|---|---|
| `01_Element-resolved_promoter_annotations/` | Published PAS tables for 43 species. Identical to [changlabtw/GBactPro `data/pas`](https://github.com/changlabtw/GBactPro/tree/master/data/pas) at commit `927a08f`. |
| `02_Training_and_testing_datasets/` | Type 1–8 FASTA files, the scan tables the generators actually read, and the generation scripts. |
| `03_Trained_models/` | 43 species-specific Type 1 models, pooled models for Types 1–8, the separate *E. coli*-trained models, and the genome-wide Type 1 checkpoint. |
| `04_Benchmark_scripts_and_results/` | The 1:1 tool-comparison table, per-species metrics, GBactPro scores, and evaluation scripts. |
| `05_Genome-wide_predictions/` | All-window scores for *E. coli* NC_000913.2 from `type1_pre_max29bp_random`, plus TSS-matching evaluation and the H-NS overlap analysis that uses those scores. |
| `MANIFEST.tsv` | Archive path, original path, role, configuration, size, and SHA-256. |

## File names

Paths in this archive use only letters, digits, underscore, hyphen, and period. That is the character set Zenodo accepts in file and folder names. Scientific names in `01_Element-resolved_promoter_annotations/species_43.tsv` still contain spaces (`Escherichia coli MG1655`). Each space becomes an underscore in the path (`Escherichia_coli_MG1655`). Periods that belong to a strain identifier stay in the name (`Rhodobacter_sphaeroides_2.4.1`, `Synechocystis_sp._PCC_6803`).

`MANIFEST.tsv` keeps both forms. `archive_path` is the name in this folder. `source_path` is the original analysis path, spaces included. Copied scripts were not rewritten, so they still refer to those original paths.

## Large files

GitHub rejects a single file over 100 MB. Sixty SavedModel variable files (about 127 MB each) and `05_Genome-wide_predictions/predictions/NC_000913.2_type1_pre_max29bp_random_allwindows.tsv` (about 1.2 GB) are tracked with Git LFS. Clone with Git LFS installed, then run `git lfs pull`, or the working tree will contain pointer text instead of the weights and the chromosome table.

## Which files belong to which experiment

**Sequence types.** Local script numbers are not the manuscript numbers. This archive uses the manuscript definitions:

| Archive folder | Composition | Length | Local script name |
|---|---|---|---|
| Type 1 | −35 + spacer + −10 | 27–31 bp | `data_type1_35s10.py` (`35+s+10`) |
| Type 2 | 50 bp UP + −35 + spacer + −10 | 77–81 bp | `data_type2_up35s10_original.py` |
| Type 3 | 20 bp UP + −35 + spacer + −10 | 47–51 bp | `data_type3_up35s10_new.py` |
| Type 4 | −35 + spacer + −10 + discriminator | 27–44 bp | `data_type3_5_35s10_dis.py` (redo) |
| Type 5 | Type 1 extended on both sides | 50 bp | `data_type4_total50_up8_12.py` |
| Type 6 | Type 1 extended on both sides | 100 bp | `data_type5_up35_40.py` |
| Type 7 | 40 bp upstream of the TSS | 40 bp | `data_type6_40bp_upstream_tss.py` |
| Type 8 | TSS −60 to +21 | 81 bp | `data_type7_minus60_plus21.py` |

A local “type 8” that left-pads the core promoter to 40 bp (`35+s+10_leftpad40`) is not one of these eight windows and is not included.

**Models.**

- Species-specific: one Type 1 network per species, trained only on that species’ 75% split. These weights are the 43 SavedModels under `03_Trained_models/species_specific/`. `03_model_training_v3` does not store 43 separate checkpoints; it trains one *E. coli* model and then scores every species.
- Pooled: `03_Trained_models/pooled_43_species/`, trained on the pooled 75% splits. The Type 1 checkpoint is byte-identical to `models/type1_35s10_random` in the public GBactPro repository.
- *E. coli*-trained: `03_Trained_models/ecoli_trained/`. These are the “Training data: *E. coli*” rows of the tool table. They are not the species-specific *E. coli* model.
- Genome-wide: `03_Trained_models/genome_wide/type1_pre_padding_max29bp_random`. Byte-identical to public `models/type1_pre_max29bp_random`. It is not the pooled Type 1 evaluation model.

**Benchmarks.** `04_Benchmark_scripts_and_results/results/formal_1to1_performance_table_with_ecoli.csv` is the source table. Its rounded AUROC and AUPRC values match the manuscript reference. The Promotech training description in that file is “7 cross species”; a later summary that says 9 species was not used to edit the file.

**Genome-wide scan.** Predictions are `05_Genome-wide_predictions/predictions/NC_000913.2_type1_pre_max29bp_random_allwindows.tsv`, produced with the genome-wide checkpoint above.

## How to reproduce the key results

Software recorded for the public models is in `03_Trained_models/environment_gbactpro.yml` (Python 3.8, TensorFlow 2.13.0, Keras 2.13.1, Biopython, scikit-learn). The training shells activated a conda environment named `gbactpro_env` or `bioinfo`. Copied scripts still contain absolute paths under `/home/brucehan/Project_02_promoter_prediction_model` and will not run unchanged on another machine.

1. **Datasets.** Read `02_Training_and_testing_datasets/scripts/`. The 75/25 promoter split uses random seed 42 (`split_train_test_input.py`). Negatives are genomic intervals drawn outside a ±200 bp promoter exclusion zone, 10 per positive, and the sampler is not seeded. The FASTA files are the reproducible objects; rerunning the scripts will not recreate the same negative sequences. Genomic GenBank inputs were not copied. Accessions are in `01_Element-resolved_promoter_annotations/species_43.tsv`. The local GenBank directory used at generation time was `GBactPro_Pipeline/01_promoter_extraction/input/genome/`.
2. **Species-specific Type 1 models.** `03_Trained_models/scripts/species_specific/train_43_models.py`. Post-padding to 31 bp, 5 epochs, batch size 64, Adam learning rate 0.001, no validation split. All 43 species completed (`training_progress.json`).
3. **Pooled and *E. coli* models.** `03_Trained_models/scripts/pooled_and_ecoli/`. Same CNN–BiLSTM (Conv 32/32/64/64, BiLSTM 1024, dense 1024, sigmoid, binary cross-entropy). Default fit settings in `type_configs_v3.py` are 5 epochs, batch 32, validation split 0.20, early stopping patience 1. Padding is post-padding to each type’s maximum length.
4. **Tool comparison.** GBactPro scores and labels are under `04_Benchmark_scripts_and_results/prediction_scores/`. Each `sequences_predictions.csv` is one score per line, with every positive first and then an equal number of negatives. AUROC/AUPRC can be recomputed with scikit-learn from those scores. For Types 1 and 8 the 1:1 negative FASTA that matches those scores was not stored separately; the archived test FASTA for those types is the full 1:10 set. For Type 7, the 1:1 FASTA is `02_Training_and_testing_datasets/Testing/Type7_40bp_upstream_of_TSS_1to1_benchmark/`. Third-party weights (iPro-MP, Prompt, iPro-WAEL, Promotech) were not redistributed. Their per-species metrics are included; per-sequence scores were not saved.
5. **Genome-wide scan and TSS matching.** `05_Genome-wide_predictions/scripts/` and `evaluation/eval_pre_padding_max29bp/`. The scan reads the model input length from the checkpoint (29 bp) and scores every non-N window on both strands of NC_000913.2.

## Limits

- The directory suggested as a local GitHub checkout, `ref_code/GBactpro`, is `ximenahan/GBactPro_v1` and has no `data/pas`. The matching checkout is `ref_code/GBactPro_released` (`changlabtw/GBactPro`, commit `927a08f`). GitHub and that checkout are the same 43 files.
- Published PAS tables and the internal scan tables have the same promoter counts and different columns. Generators read the scan tables in `02_Training_and_testing_datasets/generation_inputs/`.
- The genome-wide model was fit on 100% of the 43-species Type 1 promoters (no held-out test split) but `model.fit` used `validation_split=0.2`. Sequences longer than 29 bp were dropped (9,221 of 60,771 positives). The checkpoint directory is still named `genome_use_49` even though the training log lists 43 species.
- Promotech is “7 cross species” in the source table and “9 species” in the requested reference. The source file was kept.
- `HNS_binding_sites.bed` has no recorded publication source inside the file. Confirm that it can be redistributed before a Zenodo upload.
- An older H-NS analysis that refers to a 34 bp window (`GBactpro_BH/H-NS_analysis`) was not copied. It is not this scan.

Each section has its own README with the configuration details.
