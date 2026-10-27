#!/usr/bin/env python3
"""
BPM promoter-architecture scan for an arbitrary TSS subset CSV (Pos, Strand, ...).

Same logic as v4 scan_ecoli_tss_class.py; reads genome from input/NC_000913.2.gb
and subset CSV from input/ecoli_{tp|fn}_tss_subset.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

sys.path.insert(0, str(SCRIPT_DIR))
from BPM.BPM import *  # noqa: E402,F403

GENOME_FILE = INPUT_DIR / "NC_000913.2.gb"
GENOME_ACC = "NC_000913.2"
SPECIES_NAME = "Escherichia coli MG1655"

TSS_M10_RANGE = [0, 20]
UP_LEN = 100
DOWN_LEN = 100


class Genome:
    def __init__(self, record):
        self.record = record
        self.length = len(record.seq)
        t = record.annotations["topology"]
        if t in ("circular", "linear"):
            self.type = t
        else:
            raise Exception(f"Invalid sequence topology {t}")

    def sequence(self, l, r):
        l = l % self.length
        r = r % self.length
        if r > l:
            return self.record.seq[l:r]
        elif r <= l:
            ret = self.record.seq[l:] + self.record.seq[:r]
            if r != 0 and self.type == "linear":
                raise Exception("Invalid index range for linear sequence")
            return ret


class Locus:
    def __init__(self, l, r, d, g):
        self.genome = g
        self.d = d
        self.l = l
        self.r = r
        self._stdidx()

    def _stdidx(self):
        while self.r < 0:
            self.r += self.genome.length
        while self.l < 0:
            self.l += self.genome.length
        self.l = self.l % self.genome.length
        self.r = self.r % self.genome.length
        if self.r != 0 and self.r < self.l and self.genome.type == "linear":
            raise Exception("Invalid index range for linear sequence")

    def getLoc_m10_end(self):
        return self.r if self.d == "+" else self.l

    def getLoc_m35_start(self):
        return self.l if self.d == "+" else self.r

    def getSequence(self):
        seq = self.genome.sequence(self.l, self.r)
        if self.d == "-":
            seq = seq.reverse_complement()
        return seq


def b1to0(l, r):
    return l - 1, r


def getPromoters(tss_loc, direction, genome, tss_m10_dist, promoter_lens=None):
    if promoter_lens is None:
        promoter_lens = [12 + 15, 12 + 16, 12 + 17, 12 + 18, 12 + 19]
    promoters = []
    if direction == "+":
        p = tss_loc - tss_m10_dist - 1
        for pl in promoter_lens:
            l, r = b1to0(p - pl + 1, p)
            promoters.append(Locus(l, r, direction, genome))
    else:
        p = tss_loc + tss_m10_dist + 1
        for pl in promoter_lens:
            l, r = b1to0(p, p + pl - 1)
            promoters.append(Locus(l + 1, r + 1, direction, genome))
    return promoters


def scan_promoter(tss_loc, direction, genome, tss_m10_range):
    ps = []
    for tss_m10_dist in range(tss_m10_range[0], tss_m10_range[1] + 1):
        ps += getPromoters(tss_loc, direction, genome, tss_m10_dist)
    ps = [(p, score_promoter(p.getSequence())) for p in ps]
    promoter = min(ps, key=lambda x: x[1])
    return (promoter[0], (tss_loc, direction))


def get_element_UP(locus, length=UP_LEN):
    if locus.d == "+":
        return locus.genome.sequence(locus.l - length, locus.l)
    else:
        return locus.genome.sequence(locus.r, locus.r + length).reverse_complement()


def get_element_DOWN(locus, length=DOWN_LEN):
    if locus.d == "+":
        return locus.genome.sequence(locus.r, locus.r + length)
    else:
        return locus.genome.sequence(locus.l - length, locus.l).reverse_complement()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=["tp", "fn"])
    args = ap.parse_args()

    subset = args.subset
    csv_path = INPUT_DIR / f"ecoli_{subset}_tss_subset.csv"
    out_path = OUTPUT_DIR / subset / f"Escherichia coli MG1655_{subset}.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not GENOME_FILE.exists():
        print(f"ERROR: genome missing: {GENOME_FILE}")
        return 1
    if not csv_path.exists():
        print(f"ERROR: subset CSV missing: {csv_path}")
        return 1

    print(f"[{subset}] Loading genome: {GENOME_FILE}")
    genome = Genome(next(SeqIO.parse(str(GENOME_FILE), "genbank")))

    print(f"[{subset}] Loading TSS from: {csv_path}")
    df_tss = pd.read_csv(csv_path)

    results = []
    skipped = 0
    total = len(df_tss)

    for idx, row in df_tss.iterrows():
        tss_loc = int(row["Pos"])
        direction = row["Strand"]
        if direction not in ("+", "-"):
            direction = "+" if str(direction).strip() == "+" else "-"
        gene = row.get("Locus_tag", f"tss_{tss_loc}")

        try:
            result = scan_promoter(tss_loc, direction, genome, TSS_M10_RANGE)
            prom = result[0]
            seq_str = str(prom.getSequence())

            if "N" in seq_str:
                skipped += 1
                continue

            results.append(
                {
                    "name": SPECIES_NAME,
                    "genome": GENOME_ACC,
                    "gene": gene,
                    "loc": tss_loc,
                    "dir": direction,
                    "loc_m10_end": prom.getLoc_m10_end(),
                    "loc_m35_start": prom.getLoc_m35_start(),
                    "UP": str(get_element_UP(prom)),
                    "Minus35": seq_str[:6],
                    "Spacer": seq_str[6:-6],
                    "Minus10": seq_str[-6:],
                    "DOWN": str(get_element_DOWN(prom)),
                    "label": "promoter",
                }
            )
        except Exception as e:
            print(f"  WARN: skipping TSS {tss_loc} {direction}: {e}")
            skipped += 1

        if (idx + 1) % 500 == 0:
            print(f"  [{subset}] {idx + 1}/{total} processed ...")

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, sep="\t", index=False)
    print(f"[{subset}] Done: {len(results)} promoters written, {skipped} skipped")
    print(f"[{subset}] Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
