#!/usr/bin/env python3
"""
Compare 11834 all4classes TSS with Table_S1 (668). Copy for v5_o2o pipeline; PROJECT_ROOT = parents[5].
"""

import argparse
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]

ALL4CLASSES_BED = PROJECT_ROOT / "ref_code" / "GBactPro_new" / "data" / "tss_class_subsets" / "ecoli_all4classes_tss_filtered.bed"
TABLE_S1_TSV = PROJECT_ROOT / "ref_code" / "GBactPro_new" / "data" / "Table_S1.tsv"


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


def load_table_s1(path: Path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            try:
                pos = int(float(row["TSS Coordinate"]))
            except (ValueError, KeyError):
                continue
            strand = str(row.get("Strand", "")).strip() or "+"
            rows.append((pos, strand, row))
    return set((pos, strand) for pos, strand, _ in rows), rows


def main():
    parser = argparse.ArgumentParser(description="Compare 11834 all4classes TSS vs Table_S1 (668) TSS.")
    parser.add_argument("--known-bed", type=Path, default=ALL4CLASSES_BED, help="ecoli_all4classes_tss_filtered.bed (11834)")
    parser.add_argument("--table-s1", type=Path, default=TABLE_S1_TSV, help="Path to Table_S1.tsv")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR, help="Output directory")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.known_bed.exists():
        raise FileNotFoundError(f"All4classes TSS BED not found: {args.known_bed}")
    if not args.table_s1.exists():
        raise FileNotFoundError(f"Table_S1 not found: {args.table_s1}")

    known = load_bed_positions(args.known_bed)
    table_s1_set, table_s1_rows = load_table_s1(args.table_s1)

    n_known = len(known)
    n_table_s1 = len(table_s1_set)

    overlap_keys = table_s1_set & known
    only_table_s1 = table_s1_set - known
    only_known = known - table_s1_set

    n_overlap = len(overlap_keys)
    n_only_table_s1 = len(only_table_s1)
    n_only_known = len(only_known)

    report_lines = [
        "TSS comparison: 11834 all4classes vs Table_S1 (668 σ70-paired)",
        "=" * 72,
        f"All4classes TSS (ecoli_all4classes_tss_filtered): {n_known}",
        f"Table_S1 (σ70-paired TSS):                        {n_table_s1}",
        "",
        "Overlap (same position + strand, ignoring chromosome):",
        f"  In both:                                         {n_overlap}",
        f"  Table_S1 only (in Table_S1, not in 11834):       {n_only_table_s1}",
        f"  All4classes only (in 11834, not in Table_S1):     {n_only_known}",
        "",
        "Fractions:",
        f"  Of Table_S1 that are in all4classes set:  {100 * n_overlap / n_table_s1:.1f}%" if n_table_s1 else "  N/A",
        f"  Of all4classes set that are in Table_S1:  {100 * n_overlap / n_known:.1f}%" if n_known else "  N/A",
    ]
    report_path = out_dir / "tss_11834_vs_table_s1_overlap_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport written to {report_path}")

    overlap_pos_strand = set(overlap_keys)
    overlap_rows = [
        r
        for _, _, r in table_s1_rows
        if (int(float(r["TSS Coordinate"])), str(r.get("Strand", "+")).strip() or "+") in overlap_pos_strand
    ]
    only_table_s1_rows = [
        r
        for _, _, r in table_s1_rows
        if (int(float(r["TSS Coordinate"])), str(r.get("Strand", "+")).strip() or "+") in only_table_s1
    ]

    def write_tsv(rows, out_path, fieldnames=None):
        if not rows:
            fieldnames = fieldnames or list(table_s1_rows[0][2].keys()) if table_s1_rows else []
        else:
            fieldnames = fieldnames or list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    overlap_tsv = out_dir / "Table_S1_overlap_with_11834_all4classes_TSS.tsv"
    write_tsv(overlap_rows, overlap_tsv)
    print(f"  {overlap_tsv.name}: {len(overlap_rows)} rows")

    only_table_s1_tsv = out_dir / "Table_S1_only_not_in_11834_all4classes_TSS.tsv"
    write_tsv(only_table_s1_rows, only_table_s1_tsv)
    print(f"  {only_table_s1_tsv.name}: {len(only_table_s1_rows)} rows")

    overlap_list_path = out_dir / "all4classes_11834_TSS_positions_in_Table_S1.txt"
    with open(overlap_list_path, "w") as f:
        f.write("position\tstrand\n")
        for (pos, strand) in sorted(overlap_keys):
            f.write(f"{pos}\t{strand}\n")
    print(f"  {overlap_list_path.name}: {len(overlap_keys)} positions")

    return 0


if __name__ == "__main__":
    exit(main())
