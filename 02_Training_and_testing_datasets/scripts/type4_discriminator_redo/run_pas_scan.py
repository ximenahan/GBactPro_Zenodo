#!/usr/bin/env python3
"""
Run original PAS (tss_m10_range=[0, 14]) for all 49 species in TSS_List.tsv.

Writes:
  pas_tables/{species}.tsv          PAS-style tables with Dis (and DOWN kept)
  input/scan_result_100/{name}.tsv  dataset-generation input (includes Dis)

Does not run PAS random/shuffle controls; those are unused by type 3.5 dataset gen.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from Bio import SeqIO

REDO_ROOT = Path(__file__).resolve().parent.parent
PAS_ROOT = Path("/home/brucehan/Project_02_promoter_prediction_model/ref_code/Promoter_Architecture_Scanner")
PAS_SCRIPTS = PAS_ROOT / "scripts"
TSS_LIST = PAS_ROOT / "TSS_List.tsv"
GENOME_DIR = PAS_ROOT / "genomes"

PAS_TABLE_DIR = REDO_ROOT / "pas_tables"
SCAN_DIR = REDO_ROOT / "input" / "scan_result_100"

# Dataset-generation folder names (PAS uses A3(2) in TSS_List)
SPECIES_NAME_MAP = {
    "Streptomyces coelicolor A3(2)": "Streptomyces coelicolor A3",
}

TSS_M10_RANGE = [0, 14]  # original PAS default
N_WORKERS = int(os.environ.get("PAS_WORKERS", "6"))

os.chdir(PAS_SCRIPTS)
if str(PAS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PAS_SCRIPTS))

from common import Genome  # noqa: E402
from scanner import get_element_DOWN, get_element_UP, scan_promoter  # noqa: E402


def pas_get_dis(m10_end: int, tss_loc: int, down: str) -> str:
    dist = abs(int(m10_end) - int(tss_loc))
    if dist == 1:
        return "*"
    return str(down)[: (dist - 2)]


def loc_m35_start(m10_end: int, direction: str, core_len: int) -> int:
    if direction == "+":
        return int(m10_end) - core_len + 1
    return int(m10_end) + core_len - 1


def dataset_species_name(pas_species: str) -> str:
    return SPECIES_NAME_MAP.get(pas_species, pas_species)


def process_species(args):
    species, df_tss = args
    g = str(df_tss["Genome_accession"].iloc[0])
    gb_file = GENOME_DIR / f"{g}.gb"
    if not gb_file.exists():
        raise FileNotFoundError(f"Missing genome: {gb_file}")

    genome = Genome(next(SeqIO.parse(open(gb_file, "r"), "genbank")))
    pas_rows = []
    scan_rows = []

    for _, row in df_tss.iterrows():
        direction = row["Direction"]
        tss_loc = int(row["TSS_coordinate"])
        promoter, _ = scan_promoter(tss_loc, direction, genome, TSS_M10_RANGE)
        seq = str(promoter.getSequence())
        minus35, spacer, minus10 = seq[:6], seq[6:-6], seq[-6:]
        core_len = len(seq)
        m10_end = promoter.getLoc_m10_end()
        m35_start = loc_m35_start(m10_end, direction, core_len)
        up = str(get_element_UP(promoter))
        down = str(get_element_DOWN(promoter))
        dis = pas_get_dis(m10_end, tss_loc, down)

        pas_rows.append(
            {
                "Type": row.get("Type", ""),
                "TSS_coordinate": tss_loc,
                "Direction": direction,
                "Gene_start": row.get("Gene_start", ""),
                "Gene_end": row.get("Gene_end", ""),
                "Gene_name": row.get("Gene_name", ""),
                "Locus_tag": row.get("Locus_tag", ""),
                "Locus_tag (old)": row.get("Locus_tag (old)", ""),
                "Minus10_3end_coordinate": m10_end,
                "loc_m35_start": m35_start,
                "Genome_accession": g,
                "UP": up,
                "Minus35": minus35,
                "Spacer": spacer,
                "Minus10": minus10,
                "DOWN": down,
                "Dis": dis,
            }
        )
        scan_rows.append(
            {
                "name": dataset_species_name(species),
                "genome": g,
                "gene": row.get("Locus_tag") if pd.notna(row.get("Locus_tag")) else row.get("Gene_name", ""),
                "loc": tss_loc,
                "dir": direction,
                "loc_m10_end": m10_end,
                "loc_m35_start": m35_start,
                "UP": up,
                "Minus35": minus35,
                "Spacer": spacer,
                "Minus10": minus10,
                "DOWN": down,
                "Dis": dis,
                "label": "promoter",
            }
        )

    ds_name = dataset_species_name(species)
    PAS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    pas_df = pd.DataFrame(pas_rows)
    scan_df = pd.DataFrame(scan_rows)
    pas_path = PAS_TABLE_DIR / f"{ds_name}.tsv"
    scan_path = SCAN_DIR / f"{ds_name}.tsv"
    pas_df.to_csv(pas_path, sep="\t", index=False)
    scan_df.to_csv(scan_path, sep="\t", index=False)

    dis_lens = []
    for d in pas_df["Dis"]:
        if d in ("*", "") or pd.isna(d):
            dis_lens.append(0)
        else:
            dis_lens.append(len(str(d)))
    return species, ds_name, len(pas_df), min(dis_lens), max(dis_lens)


def main() -> None:
    PAS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)

    tss = pd.read_csv(TSS_LIST, sep="\t")
    chunks = [(sp, grp.copy()) for sp, grp in tss.groupby("Species")]
    print(f"PAS scan: {len(chunks)} species, tss_m10_range={TSS_M10_RANGE}, workers={N_WORKERS}")
    print(f"TSS rows: {len(tss)}")

    with Pool(N_WORKERS) as pool:
        results = pool.map(process_species, chunks)

    dis_minmax = Counter()
    print("\nPer-species PAS output:")
    for pas_name, ds_name, n, dmin, dmax in sorted(results, key=lambda x: x[1]):
        print(f"  {ds_name}: n={n} Dis_len={dmin}-{dmax} (PAS species={pas_name})")
        dis_minmax[(dmin, dmax)] += 1

    # Global Dis length from written scan files
    all_dis = []
    all_tot = []
    for fp in sorted(SCAN_DIR.glob("*.tsv")):
        df = pd.read_csv(fp, sep="\t")
        for _, row in df.iterrows():
            core = len(str(row["Minus35"])) + len(str(row["Spacer"])) + len(str(row["Minus10"]))
            dis = row["Dis"]
            dlen = 0 if pd.isna(dis) or str(dis) in ("*", "") else len(str(dis))
            all_dis.append(dlen)
            all_tot.append(core + dlen)
    print(f"\nWrote {len(list(SCAN_DIR.glob('*.tsv')))} species TSVs")
    print(f"Dis length (PAS design): {min(all_dis)}-{max(all_dis)} bp")
    print(f"Type 3.5 sequence length (-35+spacer+-10+Dis): {min(all_tot)}-{max(all_tot)} bp")
    print(f"PAS tables: {PAS_TABLE_DIR}")
    print(f"Dataset input: {SCAN_DIR}")


if __name__ == "__main__":
    main()
