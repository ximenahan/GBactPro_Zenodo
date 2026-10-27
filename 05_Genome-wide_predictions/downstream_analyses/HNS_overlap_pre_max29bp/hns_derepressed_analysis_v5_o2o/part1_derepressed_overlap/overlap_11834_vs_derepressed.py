#!/usr/bin/env python3
"""
Part 1: Overlap between E. coli all4classes TSS (11834) and Table_S1_derepressed_TSS (371).

(Copy of H-NS_analysis/.../part1; PROJECT_ROOT depth adjusted for 05_HNS_exe layout.)
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


def load_bed_positions(path: Path) -> set:
    out = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 6:
                end_1 = int(parts[2])
                strand = parts[5]
                out.add((end_1, strand))
    return out


def main():
    parser = argparse.ArgumentParser(description="Overlap: 11834 all4classes TSS vs Table_S1 derepressed BED.")
    parser.add_argument("--known", type=Path, default=ALL4CLASSES_BED, help="ecoli_all4classes_tss_filtered.bed (11834)")
    parser.add_argument("--derepressed", type=Path, default=DEREPRESSED_BED, help="Table_S1_derepressed_TSS.bed")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR, help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.known.exists():
        raise FileNotFoundError(f"All4classes TSS BED not found: {args.known}")
    if not args.derepressed.exists():
        raise FileNotFoundError(f"Derepressed BED not found: {args.derepressed}")

    known = load_bed_positions(args.known)
    derep = load_bed_positions(args.derepressed)

    n_known = len(known)
    n_derep = len(derep)
    overlap = known & derep
    only_known = known - derep
    only_derep = derep - known

    report_lines = [
        "Part 1: Overlap — 11834 all4classes E. coli TSS vs Table_S1_derepressed_TSS (371)",
        "=" * 72,
        f"All4classes TSS (ecoli_all4classes_tss_filtered): {n_known}",
        f"Table_S1 derepressed TSS:                         {n_derep}",
        "",
        "Overlap (same position + strand, ignoring chromosome):",
        f"  In both:                             {len(overlap)}",
        f"  All4classes only (not in derepressed):{len(only_known)}",
        f"  Derepressed only (not in 11834):     {len(only_derep)}",
        "",
        "Fractions:",
        f"  Of derepressed that are in all4classes: {100 * len(overlap) / n_derep:.1f}%" if n_derep else "  N/A",
        f"  Of all4classes that are in derepressed: {100 * len(overlap) / n_known:.1f}%" if n_known else "  N/A",
    ]
    report_path = args.out_dir / "overlap_11834_vs_derepressed_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    exit(main())
