"""
Resolve short species names (from tss_species_table_43_ref2_order.tsv) to full directory names
used in 02_dataset_generation_v3_1to10/output/75-25/test/ and 03_model_training_v3 result dirs.

TSV has short names like "Thermotoga maritima"; directories use "Thermotoga maritima MSB8".
Two special cases with naming mismatches:
  "Streptococcus pyogenes"          -> "Streptococcus pyrogenes S119"  (historical typo in dir)
  "Salmonella enterica Typhimurium" -> "Salmonella typhimurium SL1344"
"""

import os

SPECIAL_CASES = {
    "Streptococcus pyogenes": "Streptococcus pyrogenes S119",
    "Salmonella enterica Typhimurium": "Salmonella typhimurium SL1344",
}


def build_species_dir_map(tsv_species, data_dir):
    """
    Build mapping {tsv_short_name: actual_directory_name}.
    data_dir should contain subdirectories named with full strain names.
    Compatible with Python 3.6+.
    """
    data_dir = str(data_dir)
    if not os.path.isdir(data_dir):
        return {s: s for s in tsv_species}

    actual_dirs = set(os.listdir(data_dir))
    mapping = {}
    for short in tsv_species:
        if short in SPECIAL_CASES and SPECIAL_CASES[short] in actual_dirs:
            mapping[short] = SPECIAL_CASES[short]
        elif short in actual_dirs:
            mapping[short] = short
        else:
            matches = [d for d in actual_dirs if d.startswith(short)]
            if len(matches) == 1:
                mapping[short] = matches[0]
            elif len(matches) > 1:
                mapping[short] = sorted(matches)[0]
            else:
                mapping[short] = short
    return mapping
