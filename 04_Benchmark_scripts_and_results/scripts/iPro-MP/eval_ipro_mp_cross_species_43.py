#!/usr/bin/env python3
"""
Cross-species AUROC / AUPRC for iPro-MP (E. coli fine-tuned checkpoints).

- Default test (type 7): .../test/{species}/minus60_plus21_tss/random/
- Type 2 (--region-dir up+35+s+10_original): variable 77-81 bp; pad left with A to fixed 81 for eval (--fixed-seq-len 81).
- Checkpoints: iPro-MP/models/10_fold_*.pth (ensemble mean of softmax positive class)
- DNABERT weights: iPro-MP/DNABERT-6
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENT_DIR = SCRIPT_DIR.parent
# .../GBactpro_BH/performance_comparison/first_comparison.../iPro-MP/scripts
GBH_ROOT = SCRIPT_DIR.parent.parent.parent.parent

IPRO_MP_ROOT = Path(
    "/home/brucehan/Project_02_promoter_prediction_model/ref_code/model_comparision/iPro-MP"
)
DATA_ROOT = GBH_ROOT / "02_dataset_generation_v3_1to10" / "output" / "75-25"
SPECIES_TSV = (
    GBH_ROOT
    / "02_dataset_generation_v3_1to10"
    / "tss_order_tables"
    / "tss_species_table_43_ref2_order.tsv"
)

DEFAULT_REGION_DIR = "minus60_plus21_tss"
NEG_NAME = "random"
DEFAULT_DATASET_CAPTION = "type 7 / 81 bp"
MODEL_LABEL_BASE = "iPro-MP (E. coli, 81 bp, 10-fold ensemble)"
LINE_COLOR = "#e377c2"


def sanitize_dna(seq: str) -> str:
    allowed = {"A": "A", "T": "T", "C": "C", "G": "G", "a": "A", "t": "T", "c": "C", "g": "G"}
    return "".join(allowed.get(c, "N") for c in seq)


def normalize_to_fixed_bp(seq: str, fixed_len: int, pad_char: str = "A") -> str:
    """Left-pad to fixed_len; truncate from the left if longer (keep rightmost fixed_len bp)."""
    s = sanitize_dna(seq)
    pc = pad_char.upper() if pad_char else "A"
    if pc not in "ATCG":
        pc = "A"
    if len(s) >= fixed_len:
        return s[-fixed_len:]
    return pc * (fixed_len - len(s)) + s


class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, kmer_size=6, max_len=128):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.kmer_size = kmer_size
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        kmers = [sequence[i : i + self.kmer_size] for i in range(len(sequence) - self.kmer_size + 1)]
        encoded = self.tokenizer(
            kmers,
            is_split_into_words=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
            truncation=True,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class DNABERTPromoterClassifier(nn.Module):
    def __init__(self, bert_path: Path):
        super().__init__()
        self.bert = BertModel.from_pretrained(str(bert_path))
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.layer_norm1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.layer_norm2 = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, 2)
        self.activation = nn.GELU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.layer_norm1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.layer_norm2(x)
        x = self.dropout(x)
        return self.classifier(x)


def load_species_43(tsv_path: Path) -> list[str]:
    species = []
    with open(tsv_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            name = (row.get("species") or row.get("Species") or "").strip()
            if not name and row:
                name = str(next(iter(row.values()))).strip()
            if name:
                species.append(name)
    return species


def read_fasta_seqs(
    path: Path,
    fixed_seq_len: int | None,
    pad_char: str,
) -> list[str]:
    if fixed_seq_len is None:
        return [str(rec.seq).upper().replace("N", "A") for rec in SeqIO.parse(str(path), "fasta")]
    raw = [sanitize_dna(str(rec.seq)) for rec in SeqIO.parse(str(path), "fasta")]
    return [
        normalize_to_fixed_bp(s, fixed_seq_len, pad_char).replace("N", "A") for s in raw
    ]


def load_fold_checkpoints(models_dir: Path) -> list[Path]:
    paths = sorted(models_dir.glob("10_fold_*.pth"), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No 10_fold_*.pth under {models_dir}")
    return paths


def predict_ensemble(
    sequences: list[str],
    tokenizer,
    models: list[DNABERTPromoterClassifier],
    device: torch.device,
    batch_size: int,
    num_loader_workers: int = 4,
) -> np.ndarray:
    labels = [0] * len(sequences)
    ds = DNADataset(sequences, labels, tokenizer, kmer_size=6, max_len=128)
    use_cuda = device.type == "cuda"
    nw = max(0, num_loader_workers)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=nw > 0,
    )
    n = len(sequences)
    agg = np.zeros(n, dtype=np.float64)
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_cuda else nullcontext()
    with torch.no_grad():
        for model in models:
            model.eval()
            probs = []
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=use_cuda)
                attention_mask = batch["attention_mask"].to(device, non_blocking=use_cuda)
                with amp_ctx:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                probs.append(p)
            agg += np.concatenate(probs, axis=0)
    agg /= len(models)
    return agg


def evaluate_species(
    species: str,
    test_root: Path,
    tokenizer,
    models: list[DNABERTPromoterClassifier],
    device: torch.device,
    batch_size: int,
    num_loader_workers: int = 4,
    dir_name: str | None = None,
    region_dir: str | None = None,
    fixed_seq_len: int | None = None,
    pad_char: str = "A",
) -> dict | None:
    sub = test_root / "test" / (dir_name or species) / (region_dir or DEFAULT_REGION_DIR) / NEG_NAME
    pos_f = sub / "positive.fasta"
    neg_f = sub / f"negative_{NEG_NAME}.fasta"
    if not pos_f.is_file() or not neg_f.is_file():
        print(f"  SKIP {species}: missing {pos_f.name} or {neg_f.name}")
        return None
    pos = read_fasta_seqs(pos_f, fixed_seq_len, pad_char)
    neg = read_fasta_seqs(neg_f, fixed_seq_len, pad_char)
    if not pos or not neg:
        print(f"  SKIP {species}: empty FASTA")
        return None
    seqs = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    preds = predict_ensemble(
        seqs, tokenizer, models, device, batch_size, num_loader_workers=num_loader_workers
    )
    try:
        auroc = float(roc_auc_score(y, preds))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y, preds))
    except ValueError:
        auprc = float("nan")
    return {
        "species": species,
        "auroc": auroc,
        "auprc": auprc,
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def plot_lines(
    species_list: list[str],
    aurocs: list[float],
    auprcs: list[float],
    out_dir: Path,
    dataset_caption: str,
    model_label: str,
):
    idx = np.arange(len(species_list))

    fig, axes = plt.subplots(2, 1, figsize=(24, 12))
    axes[0].plot(idx, aurocs, marker="o", markersize=3, linewidth=2, color=LINE_COLOR, label=model_label)
    axes[0].set_title(f"Cross-Species AUROC — iPro-MP (43 species, {dataset_caption}, random negatives)")
    axes[0].set_ylabel("AUROC")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(species_list, rotation=90, ha="right", fontsize=7)
    axes[0].set_ylim([0, 1.0])
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(idx, auprcs, marker="s", markersize=3, linewidth=2, color=LINE_COLOR, label=model_label)
    axes[1].set_title(f"Cross-Species AUPRC — iPro-MP (43 species, {dataset_caption}, random negatives)")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(species_list, rotation=90, ha="right", fontsize=7)
    axes[1].set_ylim([0, 1.0])
    axes[1].legend(loc="best", fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(out_dir / "cross_species_auroc_auprc_line_charts.png", bbox_inches="tight", dpi=300, pad_inches=0.5)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(idx, aurocs, marker="o", markersize=4, linewidth=2.5, color=LINE_COLOR, label=model_label)
    ax.set_title("Cross-Species AUROC — iPro-MP")
    ax.set_ylabel("AUROC")
    ax.set_xticks(idx)
    ax.set_xticklabels(species_list, rotation=90, ha="right", fontsize=7)
    ax.set_ylim([0, 1.0])
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(out_dir / "cross_species_auroc_line_chart.png", bbox_inches="tight", dpi=300, pad_inches=0.5)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(idx, auprcs, marker="s", markersize=4, linewidth=2.5, color=LINE_COLOR, label=model_label)
    ax.set_title("Cross-Species AUPRC — iPro-MP")
    ax.set_ylabel("AUPRC")
    ax.set_xticks(idx)
    ax.set_xticklabels(species_list, rotation=90, ha="right", fontsize=7)
    ax.set_ylim([0, 1.0])
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(out_dir / "cross_species_auprc_line_chart.png", bbox_inches="tight", dpi=300, pad_inches=0.5)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipro-root", type=Path, default=IPRO_MP_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--species-tsv", type=Path, default=SPECIES_TSV)
    parser.add_argument("--output-dir", type=Path, default=SEGMENT_DIR / "output")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EVAL_BATCH_SIZE", "256")),
        help="Inference batch size (default 256; reduce if OOM)",
    )
    parser.add_argument(
        "--num-loader-workers",
        type=int,
        default=int(os.environ.get("EVAL_NUM_LOADER_WORKERS", "4")),
        help="DataLoader workers for sequence batching (default 4)",
    )
    parser.add_argument(
        "--region-dir",
        type=str,
        default=DEFAULT_REGION_DIR,
        help="FASTA subdirectory under test/{species}/ (type 7: minus60_plus21_tss; type 2: up+35+s+10_original)",
    )
    parser.add_argument(
        "--fixed-seq-len",
        type=int,
        default=0,
        help="If >0, left-pad with --pad-char after sanitizing ATCG/N (type 2 -> 81). "
        "If 0, read sequences as uppercase N->A only (matches historical type 7 behaviour). Default 0.",
    )
    parser.add_argument(
        "--pad-char",
        type=str,
        default="A",
        help="Left padding nucleotide ('A' encodes as 0-like for compatibility with fixed-width eval). Default A.",
    )
    parser.add_argument(
        "--dataset-type-caption",
        type=str,
        default=DEFAULT_DATASET_CAPTION,
        help='Plot/table caption fragment (default: type 7 / 81 bp; use "type 2 / left-padded 81 bp" for UP+35+S+10)',
    )
    args = parser.parse_args()

    ipro = args.ipro_root.resolve()
    bert_path = ipro / "DNABERT-6"
    models_dir = ipro / "models"
    if not bert_path.is_dir():
        sys.exit(f"DNABERT-6 not found: {bert_path}")
    ckpts = load_fold_checkpoints(models_dir)
    print(f"Ensemble folds: {[p.name for p in ckpts]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    species_list = load_species_43(args.species_tsv)
    print(f"Species count: {len(species_list)}")

    fixed_len = args.fixed_seq_len if args.fixed_seq_len and args.fixed_seq_len > 0 else None
    pad_c = args.pad_char.upper()[:1] if args.pad_char else "A"
    model_label_full = "{0}, {1}".format(MODEL_LABEL_BASE, args.dataset_type_caption)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    print(
        f"Speed: batch_size={args.batch_size}  "
        f"loader_workers={args.num_loader_workers}  AMP={'on' if device.type == 'cuda' else 'off'}"
    )

    tokenizer = BertTokenizer.from_pretrained(str(bert_path))
    models = []
    for ck in ckpts:
        m = DNABERTPromoterClassifier(bert_path)
        try:
            state = torch.load(ck, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(ck, map_location=device)
        m.load_state_dict(state)
        m.to(device)
        m.eval()
        models.append(m)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from species_dir_resolver import build_species_dir_map
    sp_map = build_species_dir_map(species_list, args.data_root / "test")
    print(f"Species dir map built ({sum(1 for s in species_list if sp_map.get(s, s) != s)} remapped)")

    rows = []
    for i, sp in enumerate(species_list, 1):
        dn = sp_map.get(sp, sp)
        print(f"[{i}/{len(species_list)}] {sp} -> {dn}")
        r = evaluate_species(
            sp,
            args.data_root,
            tokenizer,
            models,
            device,
            args.batch_size,
            num_loader_workers=args.num_loader_workers,
            dir_name=dn,
            region_dir=args.region_dir,
            fixed_seq_len=fixed_len,
            pad_char=pad_c,
        )
        if r:
            rows.append(r)
            print(f"  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  n={r['n_pos']}+{r['n_neg']}")

    if not rows:
        sys.exit("No species evaluated — check paths.")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "cross_species_metrics.csv", index=False)

    ecoli_name = "Escherichia coli"
    eco_df = df[df["species"] == ecoli_name]
    if not eco_df.empty:
        t1 = eco_df[["species", "auroc", "auprc", "n_pos", "n_neg"]]
        t1.to_csv(args.output_dir / "table1_ecoli_iPro-MP.csv", index=False)
        with open(args.output_dir / "table1_ecoli_iPro-MP.md", "w") as f:
            f.write(f"# iPro-MP — E. coli test ({args.dataset_type_caption}, random negatives)\n\n")
            try:
                f.write(t1.to_markdown(index=False))
            except ImportError:
                f.write(t1.to_string(index=False))

    aurocs_m = [float(df[df["species"] == s]["auroc"].values[0]) if s in df["species"].values else np.nan for s in species_list]
    auprcs_m = [float(df[df["species"] == s]["auprc"].values[0]) if s in df["species"].values else np.nan for s in species_list]

    valid_au = [x for x in aurocs_m if np.isfinite(x)]
    valid_ap = [x for x in auprcs_m if np.isfinite(x)]
    t2 = pd.DataFrame(
        [
            {
                "Model": model_label_full,
                "Avg AUROC": f"{np.mean(valid_au):.4f}" if valid_au else "N/A",
                "Avg AUPRC": f"{np.mean(valid_ap):.4f}" if valid_ap else "N/A",
                "Species count": len(valid_au),
            }
        ]
    )
    t2.to_csv(args.output_dir / "table2_average_iPro-MP.csv", index=False)
    with open(args.output_dir / "table2_average_iPro-MP.md", "w") as f:
        f.write("# iPro-MP — Average across evaluated species\n\n")
        try:
            f.write(t2.to_markdown(index=False))
        except ImportError:
            f.write(t2.to_string(index=False))

    plot_lines(species_list, aurocs_m, auprcs_m, args.output_dir, args.dataset_type_caption, model_label_full)

    print("Done.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
