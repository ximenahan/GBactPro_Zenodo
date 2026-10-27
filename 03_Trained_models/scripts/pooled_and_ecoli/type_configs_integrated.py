"""
Configuration for all dataset types for LSTM model training.
V3 integrated: pools train split from 43 species (ref-2 order TSV); models saved under integrated_43/.
Evaluation still uses per-species test split (full ALL_SPECIES order).
Original V3: E. coli-only training, checkpoint subdir 1-1.

Each type has:
- maxlen: Maximum sequence length (SEQ_SIZE for model)
- region: Path to dataset in output directory
- description: Human-readable description
- padding: Padding strategy ('pre' or 'post')
"""

from pathlib import Path
import importlib.util

# 43-species ref-2 list (same as random43 analysis / per-species train folders under 02_dataset…/75-25).
_GBH_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_species_43_ref2_order() -> list:
    """Match folder names under 02_dataset_generation_v3_1to10/output/75-25/train/<species>/…"""
    v3_cfg = _GBH_ROOT / "03_model_training_v3" / "config" / "type_configs.py"
    if not v3_cfg.exists():
        raise FileNotFoundError(f"Need v3 type_configs for SPECIES_43_REF2_ORDER: {v3_cfg}")
    spec = importlib.util.spec_from_file_location("v3_type_configs", v3_cfg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.SPECIES_43_REF2_ORDER)


# Kept for scripts that still point at the TSV path (documentation / split helpers)
TSS_ORDER_TABLE_43 = _GBH_ROOT / "02_dataset_generation_v3_1to10" / "tss_order_tables" / "tss_species_table_43_ref2_order.tsv"


TRAINING_SPECIES_LIST_43 = _load_species_43_ref2_order()
# Saved models and result trees live here (separate from E.coli-only v3 "1-1")
MODEL_CHECKPOINT_SUBDIR = "integrated_43"

TYPE_CONFIGS = {
    'type1_35s10': {
        'maxlen': 31,  # 27-31 bp variable, use max
        'region': '35+s+10',
        'description': 'Core promoter only (Minus35 + Spacer + Minus10)',
        'padding': 'post',
        'expected_range': (27, 31)
    },
    'type2_up35s10_original': {
        'maxlen': 81,  # 77-81 bp variable (spacer 15-19), use max
        'region': 'up+35+s+10_original',
        'description': 'Original method: Core + 50bp upstream (77-81 bp)',
        'padding': 'post',
        'expected_range': (77, 81)  # UP(50) + Minus35(6) + Spacer(15-19) + Minus10(6)
    },
    'type3_up35s10_new': {
        'maxlen': 51,  # 47-51 bp variable, use max
        'region': 'up+35+s+10_new',
        'description': 'New method: Core + 20bp upstream (47-51 bp)',
        'padding': 'post',
        'expected_range': (47, 51)
    },
    'type3_5_35s10_dis': {
        'maxlen': 50,  # 27-50 bp variable (core 27-31 + Dis 0-19), use max
        'region': '35+s+10+dis',
        'description': 'Core promoter + PAS discriminator (Minus35 + Spacer + Minus10 + Dis)',
        'padding': 'post',
        'expected_range': (27, 50),
        'dataset_dir': '02_dataset_generation_v3_1to10_type_3.5',
    },
    'type4_total50': {
        'maxlen': 50,  # Fixed 50 bp
        'region': 'up+35+s+10+down/total_50/up_8_12',
        'description': '50 bp fixed (UP: 8-12 bp, DOWN: 19-23 bp)',
        'padding': 'post',
        'expected_range': (50, 50)  # Fixed
    },
    'type5_up35_40': {
        'maxlen': 100,  # Fixed 100 bp
        'region': 'up+35+s+10+down/up_35_40',
        'description': '100 bp fixed (UP: 35-40 bp, DOWN: 24-34 bp)',
        'padding': 'post',
        'expected_range': (100, 100)  # Fixed
    },
    'type6_40bp_upstream': {
        'maxlen': 40,  # Fixed 40 bp
        'region': '40bp_upstream_tss',
        'description': '40 bp upstream of TSS',
        'padding': 'post',
        'expected_range': (40, 40)  # Fixed
    },
    'type7_minus60_plus21': {
        'maxlen': 81,  # Fixed 81 bp
        'region': 'minus60_plus21_tss',
        'description': '-60 to +21 relative to TSS (81 bp window)',
        'padding': 'post',
        'expected_range': (81, 81)  # Fixed
    },
    'type8_35s10_leftpad40': {
        'maxlen': 40,
        'region': '35+s+10_leftpad40',
        'description': 'Minus35+spacer+Minus10 padded upstream to 40 bp (dataset type 8)',
        'padding': 'post',
        'expected_range': (40, 40)
    }
}

# Training species (E.coli for training)
TRAINING_SPECIES = 'Escherichia coli MG1655'

# All species for evaluation - ORDERED by phylogenetic/taxonomic grouping (ref 1)
ALL_SPECIES = [
    # Thermotogae
    'Thermotoga maritima MSB8',
    # Cyanobacteria
    'Anabaena sp. PCC7120',
    'Synechococcus elongatus UTEX 2973',
    'Synechocystis sp. PCC 6803',
    # CPR (Candidate Phyla Radiation)
    'Nanosynbacter lyticus TM7x',
    # Actinobacteria
    'Streptomyces coelicolor A3',
    'Corynebacterium diphtheriae NCTC 13129',
    'Corynebacterium glutamicum ATCC 13032',
    'Mycolicibacterium smegmatis MC2 155',
    'Mycobacterium tuberculosis H37Rv',
    # Tenericutes (Mollicutes)
    'Phytoplasma sp. OY-M',
    'Mycoplasma pneumoniae M129',
    # Firmicutes - Clostridia
    'Clostridioides difficile 630',
    'Lachnoclostridium phytofermentans ISDg',
    # Firmicutes - Bacilli
    'Paenibacillus riograndensis SBR5',
    'Bacillus subtilis 168',
    'Staphylococcus epidermidis ATCC 12228',
    'Staphylococcus aureus MW2',
    'Listeria innocua',
    'Listeria monocytogenes',
    'Streptococcus pyrogenes S119',
    'Streptococcus agalactiae NEM316',
    # Spirochaetes
    'Leptospira interrogans L495',
    # Bacteroidetes
    'Bacteroides thetaiotaomicron VPI-5482',
    # Chlamydiae
    'Chlamydia trachomatis L2b',
    'Chlamydia pneumoniae CWL029',
    # Epsilonproteobacteria
    'Campylobacter jejuni NCTC11168',
    'Helicobacter pylori 26695',
    # Deltaproteobacteria
    'Geobacter sulfurreducens PCA',
    # Alphaproteobacteria
    'Zymomonas mobilis ZM4',
    'Novosphingobium aromaticivorans DSM 12444',
    'Rhodobacter sphaeroides 2.4.1',
    'Caulobacter crescentus NA1000',
    'Agrobacterium tumefaciens C58',
    'Methylorubrum extorquens DM4',
    'Bradyrhizobium japonicum USDA 110',
    # Betaproteobacteria
    'Neisseria gonorrhoeae MS11',
    'Bordetella pertussis Tohama I',
    'Burkholderia cenocepacia J2315',
    # Gammaproteobacteria
    'Xanthomonas campestris B100',
    'Acinetobacter baumannii ATCC 17978',
    'Pseudomonas aeruginosa UCBPP-PA14',
    'Pseudomonas putida KT2440',
    'Shewanella oneidensis MR-1',
    'Vibrio cholerae N16961',
    'Klebsiella aerogenes KCTC 2190',
    'Salmonella typhimurium SL1344',
    'Shigella flexneri M90T',
    'Escherichia coli MG1655',
]

# Negative sample types
NEGATIVE_TYPES = ['random', 'shuffle-mix']

# Training hyperparameters
TRAINING_CONFIG = {
    'epochs': 5,
    'batch_size': 32,
    'validation_split': 0.20,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'metrics': ['binary_accuracy']
}
