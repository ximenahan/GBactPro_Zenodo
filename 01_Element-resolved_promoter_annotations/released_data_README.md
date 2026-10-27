# Released datasets (43 species)

In this archive, spaces in species filenames are underscores (`Escherichia_coli_MG1655.tsv`). The public `changlabtw/GBactPro` repository still uses spaces (`Escherichia coli MG1655.tsv`). The command later in this file is the public-repository invocation, so that path is quoted and still contains spaces.

This folder ships the **promoter architecture tables** and the **Type 1 training/test FASTA** used with the v1 GBactpro model (`models/type1_35s10_random`).

Anyone may reuse these files under the repository [MIT license](../LICENSE).

```
data/
├── species_43.tsv          # index: 43 species, accessions, counts
├── pas/                    # PAS architecture tables (one TSV per species)
│   └── <species>.tsv
└── type1_35s10/            # Minus35 + Spacer + Minus10 windows
    ├── train/<species>/positive.fasta
    ├── train/<species>/negative_random.fasta
    ├── test/<species>/positive.fasta
    └── test/<species>/negative_random.fasta
```

Filenames keep the full strain names used in the paper. In the public repository those names include spaces, e.g. `Escherichia coli MG1655.tsv`. In this archive the same file is `Escherichia_coli_MG1655.tsv`.

---

## 43-species set

`species_43.tsv` lists the 43 bacteria used to train the integrated Type 1 model (same order as the paper’s ref-2 table). Totals:

| Set | Sequences |
|-----|-----------|
| PAS promoter rows | 60,767 |
| Type 1 train positives | 45,564 |
| Type 1 train random negatives | 455,640 (1:10) |
| Type 1 test positives | 15,207 |
| Type 1 test random negatives | 152,070 (1:10) |

PAS was also run on 6 additional genomes that are **not** part of the 43-species training pool and are **not** included here: *Shigella flexneri* M90T, *Pseudomonas putida* KT2440, *Lachnoclostridium phytofermentans* ISDg, *Nanosynbacter lyticus* TM7x, *Mycoplasma pneumoniae* M129, *Phytoplasma* sp. OY-M.

---

## 1. PAS architecture tables (`pas/`)

One TSV per species. These are the Promoter Architecture Scanner (PAS) outputs that were used as **input to dataset construction**. Each row is one TSS-anchored promoter with motif coordinates and sequence elements.

Key columns (tab-separated):

| Column | Meaning |
|--------|---------|
| `Type` | TSS/gene annotation class from the source TSS table |
| `TSS_coordinate` | TSS genomic coordinate |
| `Direction` | `+` or `-` |
| `Gene_start`, `Gene_end`, `Gene_name`, `Locus_tag` | Associated gene |
| `Minus10_3end_coordinate` | 3′ end of the −10 motif |
| `loc_m35_start` | genomic start of the −35 motif |
| `Genome_accession` | RefSeq / INSDC accession |
| `UP` | sequence upstream of −35 |
| `Minus35`, `Spacer`, `Minus10` | core-promoter parts |
| `DOWN` | sequence downstream of −10 |
| `Dis` | discriminator bases between −10 and TSS (PAS `tss_m10_range=[0, 14]`) |

`Dis` is the field used for Type 3.5-style windows (`Minus35 + Spacer + Minus10 + Dis`). Type 1 windows use only `Minus35 + Spacer + Minus10` and ignore `Dis` / `DOWN`.

PAS scan settings: original PAS default `tss_m10_range=[0, 14]`.

---

## 2. Type 1 constructed data (`type1_35s10/`)

**Type 1 / setting 1** = core promoter window **Minus35 + Spacer + Minus10** (27–31 bp). This is the same window scored by `scripts/gbactpro_predict.py`.

Split: **75% train / 25% test** of TSS rows, per species.

Negatives: **genomic random** windows of the same length as each positive, sampled outside ±200 bp of all TSS in that species (train+test combined), **10 negatives per positive**. Shuffle-mix negatives from the research pipeline are omitted here; the released model was trained on **random** negatives.

| File | Contents |
|------|----------|
| `positive.fasta` | Type 1 promoter windows |
| `negative_random.fasta` | genomic random non-promoter windows |

FASTA headers for positives are `>accession:start-end` using −35 start and −10 3′ end (strand order follows the source coordinate convention).

To score Type 1 positives with the v1 CLI:

```bash
python scripts/gbactpro_predict.py \
  -i "data/type1_35s10/test/Escherichia coli MG1655/positive.fasta" \
  -o results/ecoli_type1_test_pos.tsv \
  --model models/type1_35s10_random
```

---

## Notes

- Type 1 FASTA coordinates come from the Type 1 dataset-construction scan. PAS redo tables in `pas/` include a later `loc_m35_start` computation; a 1 bp offset versus Type 1 headers can appear and does not change the Type 1 **sequence** (`Minus35+Spacer+Minus10`).
- Folder spelling `Streptococcus pyrogenes S119` matches the internal dataset name (paper table: *Streptococcus pyogenes*).
- These files are data, not a runtime dependency of the predict CLI or Docker image.
