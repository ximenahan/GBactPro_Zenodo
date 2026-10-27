"""
Type 3.5 redo: 35+s+10+dis from original PAS Dis (tss_m10_range=[0,14]).
Length: 27-31 bp core + Dis 0-13 bp → 27-44 bp
Position: Core promoter (Minus35 + Spacer + Minus10) + PAS discriminator (Dis)
UP: 0 bp | DOWN (beyond Dis): 0 bp

Uses the PAS Dis column written by run_pas_scan.py.
"""
import os
import random

import pandas as pd

from genomic_negatives import (
    NEGATIVES_PER_POSITIVE,
    build_excluded_regions_train_test_combined,
    load_genome_sequence,
    sample_multiple_genomic_negatives,
)
from pas_dis import dis_for_sequence

INPUT_SCAN_DIR = os.environ.get("INPUT_SCAN_DIR", "../input/scan_result_100")
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "../output/1-1")
OUTPUT_REGION = "35+s+10+dis"

bacteria = {
    "Synechocystis sp. PCC 6803": 170,
    "Pseudomonas aeruginosa UCBPP-PA14": 2117,
    "Nanosynbacter lyticus TM7x": 188,
    "Mycolicibacterium smegmatis MC2 155": 3043,
    "Chlamydia pneumoniae CWL029": 382,
    "Chlamydia trachomatis L2b": 349,
    "Thermotoga maritima MSB8": 550,
    "Burkholderia cenocepacia J2315": 1086,
    "Clostridioides difficile 630": 1288,
    "Mycoplasma pneumoniae M129": 718,
    "Neisseria gonorrhoeae MS11": 919,
    "Bordetella pertussis Tohama I": 593,
    "Caulobacter crescentus NA1000": 1443,
    "Leptospira interrogans L495": 2864,
    "Lachnoclostridium phytofermentans ISDg": 2057,
    "Streptomyces coelicolor A3": 1016,
    "Zymomonas mobilis ZM4": 3205,
    "Anabaena sp. PCC7120": 3955,
    "Corynebacterium glutamicum ATCC 13032": 2147,
    "Corynebacterium diphtheriae NCTC 13129": 1202,
    "Mycobacterium tuberculosis H37Rv": 1778,
    "Escherichia coli MG1655": 1865,
    "Bacillus subtilis 168": 600,
    "Bacteroides thetaiotaomicron VPI-5482": 1616,
    "Salmonella typhimurium SL1344": 1120,
    "Synechococcus elongatus UTEX 2973": 2429,
    "Xanthomonas campestris B100": 1545,
    "Helicobacter pylori 26695": 716,
    "Staphylococcus aureus MW2": 1800,
    "Acinetobacter baumannii ATCC 17978": 864,
    "Staphylococcus epidermidis ATCC 12228": 1206,
    "Klebsiella aerogenes KCTC 2190": 468,
    "Paenibacillus riograndensis SBR5": 1269,
    "Methylorubrum extorquens DM4": 2030,
    "Shewanella oneidensis MR-1": 1802,
    "Agrobacterium tumefaciens C58": 468,
    "Campylobacter jejuni NCTC11168": 675,
    "Bradyrhizobium japonicum USDA 110": 4451,
    "Listeria monocytogenes": 1079,
    "Listeria innocua": 911,
    "Streptococcus pyrogenes S119": 683,
    "Streptococcus agalactiae NEM316": 885,
    "Vibrio cholerae N16961": 1283,
    "Shigella flexneri M90T": 2051,
    "Pseudomonas putida KT2440": 1142,
    "Phytoplasma sp. OY-M": 82,
    "Geobacter sulfurreducens PCA": 875,
    "Novosphingobium aromaticivorans DSM 12444": 944,
    "Rhodobacter sphaeroides 2.4.1": 1087,
}


def build_positive_sequence(row) -> str:
    core = str(row["Minus35"]) + str(row["Spacer"]) + str(row["Minus10"])
    return core + dis_for_sequence(row)


def location_header(row) -> str:
    if row["dir"] == "+":
        return ">" + row["genome"] + ":" + str(row["loc_m35_start"]) + "-" + str(row["loc_m10_end"])
    return ">" + row["genome"] + ":" + str(row["loc_m10_end"]) + "-" + str(row["loc_m35_start"])


for b in bacteria.keys():
    df_tss = pd.read_csv(INPUT_SCAN_DIR + "/" + b + ".tsv", sep="\t")

    genome_accession = str(df_tss["genome"].iloc[0])
    genome_seq = load_genome_sequence(genome_accession)
    excluded_regions = build_excluded_regions_train_test_combined(b, INPUT_SCAN_DIR) if genome_seq else []

    out_dir_r = OUTPUT_BASE + "/" + b + f"/{OUTPUT_REGION}/random/"
    os.makedirs(out_dir_r, exist_ok=True)
    out_dir_s = OUTPUT_BASE + "/" + b + f"/{OUTPUT_REGION}/shuffle-mix/"
    os.makedirs(out_dir_s, exist_ok=True)

    p_random_file = open(out_dir_r + "positive.fasta", "w")
    p_shuffle_file = open(out_dir_s + "positive.fasta", "w")
    n_random_file = open(out_dir_r + "negative_random.fasta", "w")
    n_shuffle_file = open(out_dir_s + "negative_shuffle-mix.fasta", "w")

    random_all_file = open(out_dir_r + "all_random.fasta", "w")
    shuffle_all_file = open(out_dir_s + "all_shuffle-mix.fasta", "w")

    pos_count = 0
    neg_count = 0

    for i in range(len(df_tss)):
        seq = build_positive_sequence(df_tss.iloc[i])
        if "N" not in seq:
            loc = location_header(df_tss.iloc[i])
            print(loc + "\n" + seq, file=p_random_file)
            print(loc + "\n" + seq, file=p_shuffle_file)
            print(loc + "\n" + seq, file=random_all_file)
            print(loc + "\n" + seq, file=shuffle_all_file)
            pos_count += 1

            if genome_seq and excluded_regions:
                negatives = sample_multiple_genomic_negatives(
                    genome_seq,
                    len(seq),
                    excluded_regions,
                    genome_accession,
                    num_negatives=NEGATIVES_PER_POSITIVE,
                )
                for neg_header, neg_seq in negatives:
                    print(">" + neg_header + "\n" + neg_seq, file=n_random_file)
                    print(">" + neg_header + "\n" + neg_seq, file=random_all_file)
                    neg_count += 1

    p_random_file.close()
    p_shuffle_file.close()

    for i in range(len(df_tss)):
        seq = build_positive_sequence(df_tss.iloc[i])
        if "N" not in seq:
            loc = location_header(df_tss.iloc[i])
            for _ in range(NEGATIVES_PER_POSITIVE):
                promoter = list(seq)
                random.shuffle(promoter)
                print(loc + "_shuffle\n" + "".join(promoter), file=n_shuffle_file)
                print(loc + "_shuffle\n" + "".join(promoter), file=shuffle_all_file)

    n_shuffle_file.close()
    shuffle_all_file.close()
    n_random_file.close()
    random_all_file.close()

    print(f"{b} {OUTPUT_REGION} (1:{NEGATIVES_PER_POSITIVE} ratio) completed. pos={pos_count}, neg={neg_count}")
