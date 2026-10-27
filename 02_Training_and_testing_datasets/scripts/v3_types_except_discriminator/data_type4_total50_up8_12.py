"""
Type 4: total_50/up_8_12
Total length: 50 bp (fixed)
UP: 8-12 bp | DOWN: 19-23 bp

V2_1to10: Genomic negatives with 1:10 pos:neg ratio (10 negatives per positive)
"""
import pandas as pd
import os
import random
from Bio import SeqIO
from genomic_negatives import (load_genome_sequence, build_excluded_regions_train_test_combined,
                                sample_multiple_genomic_negatives, NEGATIVES_PER_POSITIVE)

INPUT_SCAN_DIR = os.environ.get("INPUT_SCAN_DIR", "../input/scan_result_100")
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "../output/1-1")

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
            'Acinetobacter baumannii ATCC 17978':864,
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

SEQ_LENGTH = 50  # Fixed length for genomic negatives

for b in bacteria.keys():
    df_tss = pd.read_csv(INPUT_SCAN_DIR + "/" + b + ".tsv", sep="\t")

    # Load genome for genomic negatives
    genome_accession = str(df_tss["genome"].iloc[0])
    genome_seq = load_genome_sequence(genome_accession)
    excluded_regions = build_excluded_regions_train_test_combined(b, INPUT_SCAN_DIR) if genome_seq else []

    # total_50/up_8_12
    out_dir_r = OUTPUT_BASE + "/" + b + "/up+35+s+10+down/total_50/up_8_12/random/"
    os.makedirs(out_dir_r, exist_ok=True)
    out_dir_s = OUTPUT_BASE + "/" + b + "/up+35+s+10+down/total_50/up_8_12/shuffle-mix/"
    os.makedirs(out_dir_s, exist_ok=True)

    p_random_file = open(out_dir_r+'positive.fasta', 'w')
    p_shuffle_file = open(out_dir_s+'positive.fasta', 'w')
    n_random_file = open(out_dir_r+'negative_random.fasta', 'w')
    n_shuffle_file = open(out_dir_s+'negative_shuffle-mix.fasta', 'w')

    random_all_file = open(out_dir_r+'all_random.fasta', 'w')
    shuffle_all_file = open(out_dir_s+'all_shuffle-mix.fasta', 'w')

    pos_count = 0
    neg_count = 0

    # positive + genomic random negatives (1:10 ratio)
    seq = []
    for i in range(len(df_tss)):
        n = random.randint(8, 12)
        m = 50 - n - 12 - len(df_tss["Spacer"][i])
        tmp = ""
        seq_up_35_s_10_down = df_tss["UP"][i][100-n:] + df_tss["Minus35"][i] + df_tss["Spacer"][i] + df_tss["Minus10"][i] + df_tss["DOWN"][i][:m]
        seq.append([df_tss["name"][i], df_tss["genome"][i], df_tss["gene"][i], df_tss["loc"][i], df_tss["dir"][i], df_tss["loc_m10_end"][i], df_tss["loc_m35_start"][i], 
                    df_tss["UP"][i][100-n:], df_tss["Minus35"][i], df_tss["Spacer"][i], df_tss["Minus10"][i], df_tss["DOWN"][i][:m]])

        for s in seq_up_35_s_10_down:
            if s not in ['A', 'T', "C", "G"]:
                tmp += random.choice(["A", "T", "C", "G"])
            else:
                tmp += s
        seq_up_35_s_10_down = tmp
        if df_tss["dir"][i] == '+':
            loc = f'>{df_tss["genome"][i]}:{str(df_tss["loc_m35_start"][i])}-{str(df_tss["loc_m10_end"][i])}(+)'
        else:
            loc = f'>{df_tss["genome"][i]}:{str(df_tss["loc_m10_end"][i])}-{str(df_tss["loc_m35_start"][i])}(-)'
        # positive
        print(loc + "\n" + seq_up_35_s_10_down, file=p_random_file)
        print(loc + "\n" + seq_up_35_s_10_down, file=p_shuffle_file)
        print(loc + "\n" + seq_up_35_s_10_down, file=random_all_file)
        print(loc + "\n" + seq_up_35_s_10_down, file=shuffle_all_file)
        pos_count += 1

        # negative - genomic random (10 per positive, fixed 50bp)
        if genome_seq and excluded_regions:
            negatives = sample_multiple_genomic_negatives(
                genome_seq, SEQ_LENGTH, excluded_regions, genome_accession,
                num_negatives=NEGATIVES_PER_POSITIVE)
            for neg_header, neg_seq in negatives:
                print(">" + neg_header + "\n" + neg_seq, file=n_random_file)
                print(">" + neg_header + "\n" + neg_seq, file=random_all_file)
                neg_count += 1

    p_random_file.close()
    p_shuffle_file.close()

    df = pd.DataFrame(seq, columns=['name', 'genome', 'gene', 'loc', 'dir', 'loc_m10_end', 'loc_m35_start', 'UP', 'Minus35', 'Spacer', 'Minus10', 'DOWN'])
    df.to_csv(f'{OUTPUT_BASE}/{b}/up+35+s+10+down/total_50/up_8_12/positive_seq.tsv', sep='\t', index=False)

    # negative - shuffle-mix (10 per positive)
    input_file = out_dir_s+'positive.fasta'
    fasta_sequences = SeqIO.parse(open(input_file),'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        tmp = ""
        for s in sequence:
            if s not in ['A', 'T', "C", "G"]:
                tmp += random.choice(["A", "T", "C", "G"])
            else:
                tmp += s
        sequence = tmp
        loc = ">" + fasta.id
        
        for _ in range(NEGATIVES_PER_POSITIVE):
            promoter = list(sequence)
            random.shuffle(promoter)
            print(loc + "_shuffle\n" + ''.join(promoter), file=n_shuffle_file)
            print(loc + "_shuffle\n" + ''.join(promoter), file=shuffle_all_file)

    n_shuffle_file.close()
    shuffle_all_file.close()
    n_random_file.close()
    random_all_file.close()

    print(f"{b} total_50/up_8_12 (1:{NEGATIVES_PER_POSITIVE} ratio) completed. pos={pos_count}, neg={neg_count}")
