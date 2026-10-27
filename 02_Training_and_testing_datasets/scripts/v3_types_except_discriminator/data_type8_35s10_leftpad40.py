"""
Type 8: 35+s+10 with left upstream padding to fixed 40 bp
Core: same as type 1 (Minus35 + Spacer + Minus10), 27-31 bp variable.
Total: 40 bp fixed — take (40 - len(core)) bp from the UP region immediately
      upstream of -35 (last n bases of UP column, same idea as type 2/4).

UP: (40 - core_len) bp, variable 9-13 | DOWN: 0 bp

V2_1to10: Genomic negatives with 1:10 pos:neg ratio (train); 1:1 (test via run_all)
"""
import pandas as pd
import os
import random
from genomic_negatives import (load_genome_sequence, build_excluded_regions_train_test_combined,
                                sample_multiple_genomic_negatives, NEGATIVES_PER_POSITIVE)

INPUT_SCAN_DIR = os.environ.get("INPUT_SCAN_DIR", "../input/scan_result_100")
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "../output/1-1")

TARGET_LEN = 40
REGION_SUBDIR = "35+s+10_leftpad40"

bacteria = {
            'Synechocystis sp. PCC 6803': 170,
            'Pseudomonas aeruginosa UCBPP-PA14': 2117,
            'Nanosynbacter lyticus TM7x': 188,
            'Mycolicibacterium smegmatis MC2 155': 3043,
            'Chlamydia pneumoniae CWL029': 382,
            'Chlamydia trachomatis L2b': 349,
            'Thermotoga maritima MSB8': 550,
            'Burkholderia cenocepacia J2315': 1086,
            'Clostridioides difficile 630': 1288,
            'Mycoplasma pneumoniae M129': 718,
            'Neisseria gonorrhoeae MS11': 919,
            'Bordetella pertussis Tohama I': 593,
            'Caulobacter crescentus NA1000': 1443,
            'Leptospira interrogans L495': 2864,
            'Lachnoclostridium phytofermentans ISDg': 2057,
            'Streptomyces coelicolor A3': 1016,
            'Zymomonas mobilis ZM4': 3205,
            'Anabaena sp. PCC7120': 3955,
            'Corynebacterium glutamicum ATCC 13032': 2147,
            'Corynebacterium diphtheriae NCTC 13129': 1202,
            'Mycobacterium tuberculosis H37Rv': 1778,
            'Escherichia coli MG1655': 1865,
            'Bacillus subtilis 168': 600,
            'Bacteroides thetaiotaomicron VPI-5482': 1616,
            'Salmonella typhimurium SL1344': 1120,
            'Synechococcus elongatus UTEX 2973': 2429,
            'Xanthomonas campestris B100': 1545,
            'Helicobacter pylori 26695': 716,
            'Staphylococcus aureus MW2': 1800,
            'Acinetobacter baumannii ATCC 17978': 864,
            'Staphylococcus epidermidis ATCC 12228': 1206,
            'Klebsiella aerogenes KCTC 2190': 468,
            'Paenibacillus riograndensis SBR5': 1269,
            'Methylorubrum extorquens DM4': 2030,
            'Shewanella oneidensis MR-1': 1802,
            'Agrobacterium tumefaciens C58': 468,
            'Campylobacter jejuni NCTC11168': 675,
            'Bradyrhizobium japonicum USDA 110': 4451,
            'Listeria monocytogenes': 1079,
            'Listeria innocua': 911,
            'Streptococcus pyrogenes S119': 683,
            'Streptococcus agalactiae NEM316': 885,
            'Vibrio cholerae N16961': 1283,
            'Shigella flexneri M90T': 2051,
            'Pseudomonas putida KT2440': 1142,
            'Phytoplasma sp. OY-M': 82,
            'Geobacter sulfurreducens PCA': 875,
            'Novosphingobium aromaticivorans DSM 12444': 944,
            'Rhodobacter sphaeroides 2.4.1': 1087
            }


def clean_ambiguous(seq: str) -> str:
    tmp = ""
    for s in seq:
        if s not in ['A', 'T', "C", "G"]:
            tmp += random.choice(["A", "T", "C", "G"])
        else:
            tmp += s
    return tmp


def left_pad_from_up(core: str, up_full: str) -> str:
    """Prefix UP (closest to -35) so total length == TARGET_LEN."""
    need = TARGET_LEN - len(core)
    if need < 0:
        raise ValueError(f"core longer than {TARGET_LEN}: len={len(core)}")
    if need == 0:
        return core
    up_str = str(up_full)
    if len(up_str) >= need:
        prefix = up_str[-need:]
    else:
        # Rare: pad farther upstream with random bases
        pad = need - len(up_str)
        prefix = "".join(random.choice("ATCG") for _ in range(pad)) + up_str
    return prefix + core


for b in bacteria.keys():
    df_tss = pd.read_csv(INPUT_SCAN_DIR + "/" + b + ".tsv", sep="\t")

    genome_accession = str(df_tss["genome"].iloc[0])
    genome_seq = load_genome_sequence(genome_accession)
    excluded_regions = build_excluded_regions_train_test_combined(b, INPUT_SCAN_DIR) if genome_seq else []

    out_dir_r = OUTPUT_BASE + "/" + b + "/" + REGION_SUBDIR + "/random/"
    os.makedirs(out_dir_r, exist_ok=True)
    out_dir_s = OUTPUT_BASE + "/" + b + "/" + REGION_SUBDIR + "/shuffle-mix/"
    os.makedirs(out_dir_s, exist_ok=True)

    p_random_file = open(out_dir_r + 'positive.fasta', 'w')
    p_shuffle_file = open(out_dir_s + 'positive.fasta', 'w')
    n_random_file = open(out_dir_r + 'negative_random.fasta', 'w')
    n_shuffle_file = open(out_dir_s + 'negative_shuffle-mix.fasta', 'w')

    random_all_file = open(out_dir_r + 'all_random.fasta', 'w')
    shuffle_all_file = open(out_dir_s + 'all_shuffle-mix.fasta', 'w')

    pos_count = 0
    neg_count = 0

    for i in range(len(df_tss)):
        seq_35_s_10 = df_tss["Minus35"][i] + df_tss["Spacer"][i] + df_tss["Minus10"][i]
        if 'N' in seq_35_s_10:
            continue

        seq_fixed = clean_ambiguous(left_pad_from_up(seq_35_s_10, df_tss["UP"][i]))
        assert len(seq_fixed) == TARGET_LEN, (len(seq_fixed), b, i)

        if df_tss["dir"][i] == '+':
            loc = ">" + df_tss["genome"][i] + ":" + str(df_tss["loc_m35_start"][i]) + "-" + str(df_tss["loc_m10_end"][i])
        else:
            loc = ">" + df_tss["genome"][i] + ":" + str(df_tss["loc_m10_end"][i]) + "-" + str(df_tss["loc_m35_start"][i])

        print(loc + "\n" + seq_fixed, file=p_random_file)
        print(loc + "\n" + seq_fixed, file=p_shuffle_file)
        print(loc + "\n" + seq_fixed, file=random_all_file)
        print(loc + "\n" + seq_fixed, file=shuffle_all_file)
        pos_count += 1

        if genome_seq and excluded_regions:
            negatives = sample_multiple_genomic_negatives(
                genome_seq, TARGET_LEN, excluded_regions, genome_accession,
                num_negatives=NEGATIVES_PER_POSITIVE)
            for neg_header, neg_seq in negatives:
                print(">" + neg_header + "\n" + neg_seq, file=n_random_file)
                print(">" + neg_header + "\n" + neg_seq, file=random_all_file)
                neg_count += 1

    p_random_file.close()
    p_shuffle_file.close()

    for i in range(len(df_tss)):
        seq_35_s_10 = df_tss["Minus35"][i] + df_tss["Spacer"][i] + df_tss["Minus10"][i]
        if 'N' in seq_35_s_10:
            continue
        seq_fixed = clean_ambiguous(left_pad_from_up(seq_35_s_10, df_tss["UP"][i]))

        if df_tss["dir"][i] == '+':
            loc = ">" + df_tss["genome"][i] + ":" + str(df_tss["loc_m35_start"][i]) + "-" + str(df_tss["loc_m10_end"][i])
        else:
            loc = ">" + df_tss["genome"][i] + ":" + str(df_tss["loc_m10_end"][i]) + "-" + str(df_tss["loc_m35_start"][i])

        for _ in range(NEGATIVES_PER_POSITIVE):
            promoter = list(seq_fixed)
            random.shuffle(promoter)
            print(loc + "_shuffle\n" + ''.join(promoter), file=n_shuffle_file)
            print(loc + "_shuffle\n" + ''.join(promoter), file=shuffle_all_file)

    n_shuffle_file.close()
    shuffle_all_file.close()
    n_random_file.close()
    random_all_file.close()

    print(f"{b} {REGION_SUBDIR} (1:{NEGATIVES_PER_POSITIVE} ratio) completed. pos={pos_count}, neg={neg_count}")
