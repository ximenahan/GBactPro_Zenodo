# 01 — Element-resolved promoter annotations

These are the processed promoter-architecture (PAS) tables for the 43 species in the GBactPro study. Each TSV has one promoter per row: TSS coordinate, strand, gene fields, −35/−10 coordinates, the sequence elements `UP`, `Minus35`, `Spacer`, `Minus10`, `DOWN`, and the discriminator `Dis`.

## Source

Copied from the local clone

`ref_code/GBactPro_released/data/pas`

That clone tracks `https://github.com/changlabtw/GBactPro` and is at commit `927a08f` (“Add 43-species PAS and Type 1 datasets for public reuse”), even with `origin/master`.

The GitHub directory `data/pas` was compared through the GitHub contents API. All 43 filenames, git blob SHAs, and byte sizes match the local files. The two copies were not merged or rewritten.

`ref_code/GBactpro` (remote `ximenahan/GBactPro_v1`) does not contain `data/pas`. It was not used for this section.

`species_43.tsv` is the published species index (order, accession, PMID, promoter counts, Type 1 train/test counts). `released_data_README.md` and `LICENSE_changlabtw_GBactPro.txt` are the README and MIT license shipped with that repository. The license applies to these released tables.

Filenames in this archive replace spaces with underscores (`pas/Escherichia_coli_MG1655.tsv`). `species_43.tsv` still spells the strain names with spaces, and that column is the crosswalk back to the paper and to `changlabtw/GBactPro`, whose `data/pas` files still contain spaces. `released_data_README.md` is the public data README with that filename difference noted at the top.

## Role

PAS tables are the element-resolved annotation the study starts from. Dataset scripts do not read this schema directly. They read the internal scan tables in `02_Training_and_testing_datasets/generation_inputs/`, which have the same promoter counts and a different column layout (`name`, `genome`, `loc`, `UP`, `Minus35`, `Spacer`, `Minus10`, `DOWN`). For *E. coli* both files have 1,865 data rows.

Six additional genomes were scanned during dataset generation and are absent here, matching the public data README: *Shigella flexneri* M90T, *Pseudomonas putida* KT2440, *Lachnoclostridium phytofermentans* ISDg, *Nanosynbacter lyticus* TM7x, *Mycoplasma pneumoniae* M129, and *Phytoplasma* sp. OY-M.
