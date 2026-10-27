# 05 — Genome-wide predictions

This section is the final *E. coli* chromosome scan for configuration `type1_pre_max29bp_random`. Other padding modes, exact-29-only trials, and internal-padding scans in `04_genome_scanning_v5_all_data_pool` were not copied.

The checkpoint is `03_Trained_models/genome_wide/type1_pre_padding_max29bp_random`. `scripts/scan_doublestrand_pre_max29bp/run_scan_all_windows.sh` points at that directory and at `scan_double_strand_intervals.py`.

## Predictions

`predictions/NC_000913.2_type1_pre_max29bp_random_allwindows.tsv` is every non-N 29 bp window on both strands of NC_000913.2 (4,639,647 windows per strand, 9,279,294 data rows). The table is about 1.2 GB and is stored with Git LFS. Columns include 0-based and 1-based coordinates, strand, sigmoid score, an integer BED score (`round(score * 1000)` clipped to 0–1000), the genomic plus-strand sequence, and the sequence given to the model (reverse complement on the minus strand).

The source filename is `output/doublestrand_intervals_all_pre_max29bp/NC_000913.tsv`. The scan log names that path. The current copy of the scanner would write `{prefix}_allwindows.tsv` instead; the script was changed after this run. The archived table is the file the log wrote.

The companion BED was not copied. It repeats the same coordinates with the integer score only.

Windows are exactly 29 bp because the scanner sets the window to `model.input_shape[1]`. The log prints “Double-strand 31 bp scan” and then `Warning: model input length 29 != 31`. For a window that is already 29 bp, the post-padding call in the scanner adds no bases, so it does not differ from the pre-padding used in training. The genome FASTA was `GBactPro_new/data/raw/raw_data/genome/NC_000913.2.fasta` and was not copied; the scored sequences are already in the TSV.

## Evaluation

`evaluation/eval_pre_padding_max29bp/` is the PR/ROC and one-to-one TSS matching for this scan, including 0 bp and 100 bp matching tolerances. TSS reference tables used by those scripts are inside that directory (`table_s1_all_tss_reference`).

## Downstream H-NS analysis

`downstream_analyses/HNS_overlap_pre_max29bp/` is `GBactpro_BH/05_HNS_exe` without launch logs. Its shell scripts pass this scan TSV and `eval_pre_padding_max29bp` PR curves as inputs, at both 0 bp and 100 bp TSS tolerance. That includes the overlap bar charts, box plots, chi-square tests, false-positive density, and the H-NS-derepressed TSS analysis.

`downstream_analyses/reference_data/` holds the two external files those scripts open:

- `ecoli_all4classes_tss_filtered.csv` — processed *E. coli* TSS class table.
- `HNS_binding_sites.bed` — H-NS intervals. The file does not name the source paper or database. Confirm redistribution rights before upload.

`GBactpro_BH/H-NS_analysis` is an earlier analysis labeled as a 34 bp window. It does not point at this scan and was not copied.
