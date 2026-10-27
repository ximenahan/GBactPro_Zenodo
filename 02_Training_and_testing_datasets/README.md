# 02 — Training and testing datasets

FASTA files for the eight manuscript input windows. Only the genomic-random negative set is included. Shuffle-mix negatives were not copied.

## Split and sampling

- Promoter rows were split 75% train / 25% test with `pandas.DataFrame.sample(..., random_state=42)` in `split_train_test_input.py`. The same split is shared across sequence types.
- Negatives are real genomic windows outside a ±200 bp exclusion zone around known promoter cores (`genomic_negatives.py`). The code calls this the random set. They are not uniformly random synthetic DNA, and the sampler does not set a random seed. The archived FASTA files are the sequences that were used.
- Types 1–3 and 5–8: 10 negatives per positive in both `Training/` and `Testing/`.
- Type 4 (discriminator redo): 10 negatives per positive in training and 1 per positive in testing, as set in `scripts/type4_discriminator_redo/run_all.sh`.
- *E. coli* Type 1 counts: train 1,398 / 13,980 and test 467 / 4,670. These files are byte-identical to `data/type1_35s10` in `changlabtw/GBactPro` for all 43 species.
- The published tool table evaluates a 1:1 negative subset (467/467 in *E. coli*). That subset is stored for Type 7 as `Testing/Type7_40bp_upstream_of_TSS_1to1_benchmark/`. Separate 1:1 FASTA files for Types 1 and 8 were not found. Their 1:1 scores are in section 04.

## Folder names

See the type table in the archive README. Each species directory contains `positive.fasta` and `negative_random.fasta`. The directory name is the strain name with spaces replaced by underscores (`Escherichia_coli_MG1655`). Headers keep genomic coordinates. The redundant `all_random.fasta` concatenation was not copied.

Length checks on the *E. coli* training positives: Type 1 is 27–31 bp, Type 2 is 77–81 bp, Type 3 is 47–51 bp, Type 4 is 28–43 bp (the generator allows 27–44 bp), Type 5 is 50 bp, Type 6 is 100 bp, Type 7 is 40 bp, and Type 8 is 81 bp.

## What the scripts need

`scripts/v3_types_except_discriminator/` builds Types 1–3 and 5–8 from `generation_inputs/scan_result_100_types_1_2_3_5_6_7_8/`.

`scripts/type4_discriminator_redo/` builds Type 4 from `generation_inputs/scan_result_100_type4_with_Dis/`, which adds the `Dis` column. `pas_tables/` in the redo directory matches the published PAS files and was not duplicated.

`split_summary_all_species_in_source_run.tsv` lists every species in the original run, including six that are not in the published set. Only the 43 published species were copied.

Genome sequences used for negative sampling are GenBank files named `{accession}.gb`. They were not copied. Accessions are in section 01.
