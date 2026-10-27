#!/usr/bin/env python3
"""
Build Type 1–8 (renumbered) AUROC/AUPRC table: E. coli–trained vs 43-species-trained.

Does not modify any existing scripts or result trees. Writes only into this folder.

E. coli–trained checkpoints: 03_model_training_v3/.../lstm/1-1
  (numbers match the published Type 1–8 E. coli cross-species table).

43-species-trained checkpoints: 03_model_training_v3_integrated_data/.../lstm/integrated_43

Type 7 (40 bp upstream TSS) uses the both-strand 1:1 eval (E. coli 467:467),
not the older plus-strand-only 224:224 stored predictions.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
GBH = HERE.parents[2]
V3_CFG = GBH / "03_model_training_v3" / "config" / "type_configs.py"

V3_LSTM = GBH / "03_model_training_v3" / "scripts" / "models" / "lstm" / "1-1"
INT_LSTM = (
    GBH
    / "03_model_training_v3_integrated_data"
    / "scripts"
    / "models"
    / "lstm"
    / "integrated_43"
)
T7_ECOLI = (
    GBH
    / "performance_comparison"
    / "formal_1to1_type7_40bp_467"
    / "gbactpro_ecoli_trained_type7"
    / "cross_species_metrics.csv"
)
T7_POOLED = (
    GBH
    / "performance_comparison"
    / "formal_1to1_type7_40bp_467"
    / "gbactpro_43species_trained_type7"
    / "cross_species_metrics.csv"
)

# Renumbered Type 1–8 (old leftpad type 8 excluded; old 3.5 → Type 4).
TYPES = [
    {
        "type": 1,
        "model": "35+s+10",
        "input_sequence": "Element-resolved",
        "region": "35+s+10",
        "use_t7_467": False,
    },
    {
        "type": 2,
        "model": "up (50 bp)+35+s+10",
        "input_sequence": "Element-resolved",
        "region": "up+35+s+10_original",
        "use_t7_467": False,
    },
    {
        "type": 3,
        "model": "up (20 bp)+35+s+10",
        "input_sequence": "Element-resolved",
        "region": "up+35+s+10_new",
        "use_t7_467": False,
    },
    {
        "type": 4,
        "model": "35+s+10+dis",
        "input_sequence": "Element-resolved",
        "region": "35+s+10+dis",
        "use_t7_467": False,
    },
    {
        "type": 5,
        "model": "total_50 / up_8_12",
        "input_sequence": "Fixed-length",
        "region": "up+35+s+10+down/total_50/up_8_12",
        "use_t7_467": False,
    },
    {
        "type": 6,
        "model": "up_35_40",
        "input_sequence": "Fixed-length",
        "region": "up+35+s+10+down/up_35_40",
        "use_t7_467": False,
    },
    {
        "type": 7,
        "model": "40 bp upstream TSS",
        "input_sequence": "Fixed-length",
        "region": "40bp_upstream_tss",
        "use_t7_467": True,
    },
    {
        "type": 8,
        "model": "−60 to +21 TSS",
        "input_sequence": "Fixed-length",
        "region": "minus60_plus21_tss",
        "use_t7_467": False,
    },
]


def r4(x: float) -> float:
    return float(Decimal(str(float(x))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def load_species_43() -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("v3_type_configs", V3_CFG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.SPECIES_43_REF2_ORDER)


def metrics_from_pred_dir(result_root: Path, species: list[str]) -> tuple[float, float, int, int]:
    aurocs, auprcs = [], []
    ecoli_npos = ecoli_nneg = None
    missing = []
    for sp in species:
        f = result_root / sp / "sequences_predictions.csv"
        if not f.is_file():
            missing.append(sp)
            continue
        preds = pd.read_csv(f, header=None)[0].to_numpy(dtype=np.float64)
        n_pos = len(preds) // 2
        n_neg = len(preds) - n_pos
        y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int32)
        aurocs.append(float(roc_auc_score(y, preds)))
        auprcs.append(float(average_precision_score(y, preds)))
        if sp == "Escherichia coli MG1655":
            ecoli_npos, ecoli_nneg = n_pos, n_neg
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} species under {result_root}: {missing[:3]}")
    if len(aurocs) != 43:
        raise RuntimeError(f"Expected 43 species, got {len(aurocs)} from {result_root}")
    return float(np.mean(aurocs)), float(np.mean(auprcs)), int(ecoli_npos), int(ecoli_nneg)


def metrics_from_csv(path: Path, species: list[str]) -> tuple[float, float, int, int]:
    df = pd.read_csv(path)
    by = {str(r["species"]).strip(): r for _, r in df.iterrows()}
    aurocs, auprcs = [], []
    ecoli_npos = ecoli_nneg = None
    for sp in species:
        if sp not in by:
            raise KeyError(f"{sp} not in {path}")
        r = by[sp]
        aurocs.append(float(r["auroc"]))
        auprcs.append(float(r["auprc"]))
        if sp == "Escherichia coli MG1655":
            ecoli_npos, ecoli_nneg = int(r["n_pos"]), int(r["n_neg"])
    return float(np.mean(aurocs)), float(np.mean(auprcs)), ecoli_npos, ecoli_nneg


def fmt4(x: float) -> str:
    return f"{r4(x):.4f}"


def main() -> None:
    species = load_species_43()
    rows = []
    notes = []

    for spec in TYPES:
        t = spec["type"]
        if spec["use_t7_467"]:
            au_e, ap_e, np_e, nn_e = metrics_from_csv(T7_ECOLI, species)
            au_p, ap_p, np_p, nn_p = metrics_from_csv(T7_POOLED, species)
            src_e = str(T7_ECOLI)
            src_p = str(T7_POOLED)
            if (np_e, nn_e) != (467, 467) or (np_p, nn_p) != (467, 467):
                raise RuntimeError(
                    f"Type 7 E. coli test is not 467:467 "
                    f"(E.coli-train {np_e}:{nn_e}, pooled {np_p}:{nn_p})"
                )
            notes.append(
                f"Type 7: both-strand 1:1 (E. coli {np_e}:{nn_e}); "
                "not the older plus-strand 224:224 stored 40 bp eval."
            )
        else:
            au_e, ap_e, np_e, nn_e = metrics_from_pred_dir(
                V3_LSTM / spec["region"] / "random" / "result", species
            )
            au_p, ap_p, np_p, nn_p = metrics_from_pred_dir(
                INT_LSTM / spec["region"] / "random" / "result", species
            )
            src_e = str(V3_LSTM / spec["region"] / "random" / "result")
            src_p = str(INT_LSTM / spec["region"] / "random" / "result")

        common = {
            "Type": t,
            "Model": spec["model"],
            "Input sequence": spec["input_sequence"],
        }
        rows.append(
            {
                **common,
                "Species data": "E.coli",
                "Avg AUROC": r4(au_e),
                "Avg AUPRC": r4(ap_e),
                "E.coli n_pos": np_e,
                "E.coli n_neg": nn_e,
                "source": src_e,
            }
        )
        rows.append(
            {
                **common,
                "Species data": "43 cross species",
                "Avg AUROC": r4(au_p),
                "Avg AUPRC": r4(ap_p),
                "E.coli n_pos": np_p,
                "E.coli n_neg": nn_p,
                "source": src_p,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = HERE / "table_type1_8_ecoli_vs_43species.csv"
    out_md = HERE / "table_type1_8_ecoli_vs_43species.md"
    df.to_csv(out_csv, index=False)

    show = df[
        ["Type", "Model", "Input sequence", "Species data", "Avg AUROC", "Avg AUPRC"]
    ]
    lines = [
        "# Type 1–8: E. coli–trained vs 43-species-trained (test 1:1)",
        "",
        "Averages are unweighted means over the 43-species ref-2 list.",
        "Type 7 (40 bp upstream TSS) uses both-strand 1:1 tests (E. coli **467:467**).",
        "",
        "| Type | Model | Input sequence | Species data | Avg AUROC | Avg AUPRC |",
        "|---|---|---|---|---:|---:|",
    ]
    for _, r in show.iterrows():
        lines.append(
            f"| {int(r['Type'])} | {r['Model']} | {r['Input sequence']} | "
            f"{r['Species data']} | {fmt4(r['Avg AUROC'])} | {fmt4(r['Avg AUPRC'])} |"
        )
    lines += ["", "## Notes", ""]
    lines += [f"- {n}" for n in notes]
    lines += [
        "- E. coli–trained models: `03_model_training_v3/scripts/models/lstm/1-1`.",
        "- 43-species-trained models: `03_model_training_v3_integrated_data/.../integrated_43`.",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text())
    print("Wrote:", out_csv)
    print("Wrote:", out_md)


if __name__ == "__main__":
    main()
