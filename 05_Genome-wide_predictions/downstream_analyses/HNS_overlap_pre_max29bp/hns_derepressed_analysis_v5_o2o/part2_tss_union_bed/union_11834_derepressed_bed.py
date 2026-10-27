#!/usr/bin/env python3
"""
Part 2: BED file = 11834 all4classes E. coli TSS ∪ Table_S1_derepressed_TSS (371).

(Copy of H-NS_analysis/.../part2; PROJECT_ROOT depth adjusted for 05_HNS_exe layout.)
"""

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]

ALL4CLASSES_BED = PROJECT_ROOT / "ref_code" / "GBactPro_new" / "data" / "tss_class_subsets" / "ecoli_all4classes_tss_filtered.bed"
DEREPRESSED_BED = (
    PROJECT_ROOT
    / "ref_code"
    / "GBactPro_new"
    / "scans"
    / "lstm_scan"
    / "hns_knockout_derepressed_promoter_analysis"
    / "table_s1_analysis"
    / "Table_S1_derepressed_TSS.bed"
)
CHR = "NC_000913.2"


def load_bed_rows(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 6:
                end_1 = int(parts[2])
                name = parts[3]
                score = parts[4]
                strand = parts[5]
                rows.append((end_1, strand, name, score))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Union BED: 11834 all4classes ∪ Table_S1 derepressed.")
    parser.add_argument("--known", type=Path, default=ALL4CLASSES_BED, help="ecoli_all4classes_tss_filtered.bed")
    parser.add_argument("--derepressed", type=Path, default=DEREPRESSED_BED, help="Table_S1_derepressed_TSS.bed")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR, help="Output directory")
    parser.add_argument("--out-name", default="all4classes_11834_union_derepressed_371.bed", help="Output BED filename")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.known.exists():
        raise FileNotFoundError(f"All4classes TSS BED not found: {args.known}")
    if not args.derepressed.exists():
        raise FileNotFoundError(f"Derepressed BED not found: {args.derepressed}")

    known_rows = load_bed_rows(args.known)
    derep_rows = load_bed_rows(args.derepressed)

    seen = set()
    union = []
    for (pos, strand, name, score) in known_rows:
        key = (pos, strand)
        if key not in seen:
            seen.add(key)
            union.append((pos, strand, name, score))
    for (pos, strand, name, score) in derep_rows:
        key = (pos, strand)
        if key not in seen:
            seen.add(key)
            union.append((pos, strand, name, score))

    union.sort(key=lambda x: (x[0], x[1]))

    out_path = args.out_dir / args.out_name
    with open(out_path, "w") as f:
        for (pos, strand, name, score) in union:
            start_0 = pos - 1
            end_1 = pos
            f.write(f"{CHR}\t{start_0}\t{end_1}\t{name}\t{score}\t{strand}\n")

    n_known = len(known_rows)
    n_derep = len(derep_rows)
    n_union = len(union)
    n_overlap = n_known + n_derep - n_union
    report_lines = [
        "Part 2: Union BED — 11834 all4classes ∪ Table_S1 derepressed (371)",
        "=" * 72,
        f"All4classes TSS:              {n_known}",
        f"Derepressed TSS:              {n_derep}",
        f"Overlap (in both):            {n_overlap}",
        f"Union (unique TSS):           {n_union}",
        "",
        f"Output: {out_path}",
        f"Format: {CHR}, start (0-based), end (1-based), name, score, strand",
    ]
    report_path = args.out_dir / "union_bed_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    exit(main())
