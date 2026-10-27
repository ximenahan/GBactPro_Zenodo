# -*- coding: utf-8 -*-
"""
Extract iPro-WAEL RF feature vector (2466 dims) using ilearnplus Descriptor.

Decomposition (sum = 2466):
  CKSNAP kspace=5 (96) + Mismatch kmer=5 m=1 (1024) + RCKmer kmer=4 (136)
  + PCPseTNC lambda=378 weight=0.05 (442) + TPCP all trinucleotide props (768)

Order matches iPro-WAEL/main.py: cksnap, mismatch, rckmer, psetnc, tpcp.

Sequences are repeat-padded to RF_PAD_LEN (default 1200) for stable PCPseTNC.
"""

from __future__ import print_function

import importlib.util
import os
import pickle
import sys
import sysconfig
import tempfile
from typing import List

import numpy as np

RF_PAD_LEN = 1200


def _fileprocessing_path():
    purelib = sysconfig.get_paths()["purelib"]
    p = os.path.join(purelib, "ilearnplus", "util", "FileProcessing.py")
    if os.path.isfile(p):
        return p
    for root in sys.path:
        if not root:
            continue
        cand = os.path.join(root, "ilearnplus", "util", "FileProcessing.py")
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError("ilearnplus/util/FileProcessing.py not found (pip install ilearnplus).")


def _load_descriptor_class():
    path = _fileprocessing_path()
    spec = importlib.util.spec_from_file_location("ilearn_fp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Descriptor


def _pad_seq(seq, target_len=RF_PAD_LEN):
    s = "".join(c if c in "ACGTacgt" else "" for c in str(seq)).upper()
    if len(s) == 0:
        s = "A"
    reps = (target_len // len(s)) + 1
    return (s * reps)[:target_len]


def _encoding_to_float(enc):
    return np.asarray(enc[1:, 2:], dtype=np.float64)


def tridna_phyche_path():
    purelib = sysconfig.get_paths()["purelib"]
    p = os.path.join(purelib, "ilearnplus", "util", "data", "tridnaPhyche.data")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError("tridnaPhyche.data not found inside ilearnplus.")


def extract_rf_features_for_sequences(sequences):
    # type: (List[str]) -> np.ndarray
    Descriptor = _load_descriptor_class()
    tridna_pickle_path = tridna_phyche_path()
    with open(tridna_pickle_path, "rb") as h:
        prop_dict = pickle.load(h)
    all_props = ";".join(sorted(prop_dict.keys()))
    pse_names = ["Dnase I", "Bendability (DNAse)"]
    pse_val = {k: prop_dict[k] for k in pse_names}

    fd, fpath = tempfile.mkstemp(suffix=".fasta")
    os.close(fd)
    try:
        with open(fpath, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(">s{0}|1|training\n".format(i))
                f.write(_pad_seq(seq) + "\n")

        blocks = []

        d = Descriptor(fpath, {"kspace": 5})
        if not d.CKSNAP():
            raise RuntimeError("CKSNAP: " + str(d.error_msg))
        blocks.append(_encoding_to_float(d.encoding_array))

        d = Descriptor(fpath, {"kmer": 5, "mismatch": 1})
        if not d.Mismatch():
            raise RuntimeError("Mismatch: " + str(d.error_msg))
        blocks.append(_encoding_to_float(d.encoding_array))

        d = Descriptor(fpath, {"kmer": 4})
        if not d.RCKmer():
            raise RuntimeError("RCKmer: " + str(d.error_msg))
        blocks.append(_encoding_to_float(d.encoding_array))

        d = Descriptor(fpath, {"lambdaValue": 378, "weight": 0.05})
        if not d.PCPseTNC(pse_names, pse_val):
            raise RuntimeError("PCPseTNC: " + str(d.error_msg))
        blocks.append(_encoding_to_float(d.encoding_array))

        d = Descriptor(fpath, {"Tri-DNA-Phychem": all_props})
        if not d.TPCP():
            raise RuntimeError("TPCP: " + str(d.error_msg))
        blocks.append(_encoding_to_float(d.encoding_array))

        X = np.concatenate(blocks, axis=1)
        if X.shape[1] != 2466:
            raise RuntimeError("Expected 2466 RF features, got {0}".format(X.shape[1]))
        return X
    finally:
        try:
            os.unlink(fpath)
        except OSError:
            pass
