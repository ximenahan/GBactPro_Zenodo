#!/usr/bin/env python3
"""
Shared helper module for generating genomic negative sequences.
V2_1to10: Generates 10 negatives per positive (1:10 ratio)

Negatives are sampled from real genome sequences outside promoter regions
(±200 bp exclusion zone around known TSS/promoter cores).

Exclusion intervals for train/test runs use train+test TSS rows combined per species
(build_excluded_regions_train_test_combined) so negatives never sit on promoters from
either split.

Based on: GBactPro_Pipeline/03_model_training/cross_species_total50_up8_12/scripts/generate_random_negatives.py
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


# Path to GenBank genomes
GENOME_DIR = Path("/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_Pipeline/01_promoter_extraction/input/genome")

# Exclusion zone around promoters (bp)
MIN_DISTANCE_FROM_PROMOTER = 200

# Number of negatives per positive (override with env var)
NEGATIVES_PER_POSITIVE = int(os.environ.get("NEGATIVES_PER_POSITIVE", "10"))


def load_genome_sequence(genome_accession: str) -> Optional[str]:
    """Load genome sequence from GenBank file."""
    gb_file = GENOME_DIR / f"{genome_accession}.gb"
    if not gb_file.exists():
        return None
    
    try:
        record = next(SeqIO.parse(str(gb_file), "genbank"))
        return str(record.seq).upper()
    except Exception as e:
        print(f"  Error reading {gb_file}: {e}")
        return None


# Maximum reasonable promoter core span (bp). 
# Any span larger than this is likely a circular genome wrap-around artifact.
MAX_REASONABLE_CORE_SPAN = 500


def build_excluded_regions(df_tss: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Build list of excluded genomic intervals from TSS data.
    Uses loc_m35_start / loc_m10_end and expands by ±MIN_DISTANCE_FROM_PROMOTER.

    Special case - circular genome wrap-around:
        When a promoter is located near the origin of a circular chromosome,
        loc_m35_start and loc_m10_end may appear on opposite ends of the
        linearized sequence (e.g., 13 and 816378). The naive approach
        min/max would create an interval spanning the entire genome.
        
        We detect this by checking if |loc_m35_start - loc_m10_end| > MAX_REASONABLE_CORE_SPAN.
        If so, we treat each coordinate separately and exclude ±200 bp around each,
        rather than one giant interval.
    """
    regions: List[Tuple[int, int]] = []
    
    if "loc_m35_start" not in df_tss.columns or "loc_m10_end" not in df_tss.columns:
        # Fallback: use 'loc' (TSS position)
        if "loc" in df_tss.columns:
            for _, row in df_tss.iterrows():
                tss = int(row["loc"])
                regions.append(
                    (max(0, tss - MIN_DISTANCE_FROM_PROMOTER), tss + MIN_DISTANCE_FROM_PROMOTER)
                )
        return regions
    
    for _, row in df_tss.iterrows():
        start = int(row["loc_m35_start"])
        end = int(row["loc_m10_end"])
        
        span = abs(start - end)
        
        if span > MAX_REASONABLE_CORE_SPAN:
            # Circular wrap-around case: exclude regions around each coordinate separately
            # This avoids creating one giant interval spanning the whole genome
            regions.append((max(0, start - MIN_DISTANCE_FROM_PROMOTER), start + MIN_DISTANCE_FROM_PROMOTER))
            regions.append((max(0, end - MIN_DISTANCE_FROM_PROMOTER), end + MIN_DISTANCE_FROM_PROMOTER))
        else:
            # Normal case: single contiguous interval
            core_start = min(start, end)
            core_end = max(start, end)
            regions.append(
                (
                    max(0, core_start - MIN_DISTANCE_FROM_PROMOTER),
                    core_end + MIN_DISTANCE_FROM_PROMOTER,
                )
            )
    
    return regions


def build_excluded_regions_train_test_combined(species_tsv_stem: str, input_scan_dir: str) -> List[Tuple[int, int]]:
    """
    Build exclusion intervals from all TSS rows for this species in both train and test
    (full species table), so genomic negatives for either split avoid promoter regions
    from the entire labeled set.

    Expects layout .../input_split/{train,test}/scan_result_100/{species}.tsv. If only one
    split file exists, uses that file only. Non-split single-directory layouts fall back to
    input_scan_dir/{species}.tsv.
    """
    p = Path(input_scan_dir).resolve()
    tsv_name = f"{species_tsv_stem}.tsv"
    frames: List[pd.DataFrame] = []

    if p.name == "scan_result_100" and p.parent.name in ("train", "test"):
        sibling = p.parent.parent / ("test" if p.parent.name == "train" else "train") / "scan_result_100"
        for scan_dir in (p, sibling):
            fp = scan_dir / tsv_name
            if fp.exists():
                frames.append(pd.read_csv(fp, sep="\t"))
        if not frames:
            return []
        df_all = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    else:
        single = p / tsv_name
        if not single.exists():
            return []
        df_all = pd.read_csv(single, sep="\t")

    return build_excluded_regions(df_all)


def is_valid_position(pos: int, length: int, excluded_regions: List[Tuple[int, int]], 
                      genome_length: int) -> bool:
    """Check if [pos, pos+length) is valid and doesn't overlap excluded regions."""
    if pos < 0 or pos + length > genome_length:
        return False
    
    pos_end = pos + length - 1
    for (start, end) in excluded_regions:
        if not (pos_end < start or pos > end):
            return False
    
    return True


def sample_genomic_negative(genome_seq: str, seq_length: int, 
                           excluded_regions: List[Tuple[int, int]],
                           genome_accession: str) -> Tuple[str, str]:
    """
    Sample a single negative sequence from the genome.
    
    Returns:
        (header, sequence) where header is like "NC_000913.2:12345-12395(+)"
    """
    genome_length = len(genome_seq)
    max_attempts = seq_length * 500
    
    for _ in range(max_attempts):
        pos = random.randint(0, genome_length - seq_length - 1)
        
        if not is_valid_position(pos, seq_length, excluded_regions, genome_length):
            continue
        
        seq = genome_seq[pos:pos + seq_length]
        if "N" in seq or any(c not in "ATCG" for c in seq):
            continue
        
        strand = random.choice(["+", "-"])
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        
        header = f"{genome_accession}:{pos}-{pos + seq_length}({strand})"
        return header, seq
    
    raise RuntimeError(f"Could not sample negative of length {seq_length} after {max_attempts} attempts")


def sample_multiple_genomic_negatives(genome_seq: str, seq_length: int, 
                                      excluded_regions: List[Tuple[int, int]],
                                      genome_accession: str,
                                      num_negatives: int = NEGATIVES_PER_POSITIVE) -> List[Tuple[str, str]]:
    """
    Sample multiple negative sequences from the genome for a single positive.
    
    Args:
        num_negatives: Number of negatives to generate (default: 10 for 1:10 ratio)
    
    Returns:
        List of (header, sequence) tuples
    """
    negatives = []
    for i in range(num_negatives):
        try:
            header, seq = sample_genomic_negative(genome_seq, seq_length, 
                                                  excluded_regions, genome_accession)
            negatives.append((header, seq))
        except RuntimeError as e:
            print(f"  Warning: {e}")
            break
    
    return negatives


def generate_genomic_negatives(genome_seq: str, num_sequences: int, seq_length: int,
                               excluded_regions: List[Tuple[int, int]], 
                               genome_accession: str) -> List[Tuple[str, str]]:
    """
    Generate multiple genomic negative sequences.
    
    Returns:
        List of (header, sequence) tuples
    """
    negatives = []
    for _ in range(num_sequences):
        try:
            header, seq = sample_genomic_negative(genome_seq, seq_length, 
                                                  excluded_regions, genome_accession)
            negatives.append((header, seq))
        except RuntimeError as e:
            print(f"  Warning: {e}")
            break
    
    return negatives
