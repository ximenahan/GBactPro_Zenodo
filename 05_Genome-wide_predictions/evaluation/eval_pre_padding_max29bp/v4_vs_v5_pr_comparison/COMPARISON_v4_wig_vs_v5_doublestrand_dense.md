# v4 `pr_curve_all4classes_dense` vs v5 `pr_curve_doublestrand_dense` (100 bp tolerance)

This note compares **how** each pipeline evaluates precision and recall, **why** “FN” (missed reference TSS on the recall axis) can differ a lot, and **what** is *not* comparable between the two outputs.

---

## 1. Which scripts run

### v4 — `type1_dense` → `pr_curve_highres.png`

- **Shell:**  
  `04_genome_scanning_v4_all_data_pool/pr_curve_all4classes_dense/run_pr_dense_parallel.sh`
- **Python:**  
  `ref_code/GBactPro_new/scans/lstm_scan/pr_curve_classic_recall_dense_upper/generate_pr_curve_classic_recall_dense.py`
- **Inputs:**
  - **Predictions:** WIG variable-step scores, one value per **genomic position** (single strand along the chromosome). Example: `NC_000913.2_type1_scan.wig` (~4,639,649 positions).
  - **Reference:** `pr_curve_all4classes_dense/ecoli_all4classes_tss_filtered.tsv` with columns **`genome`**, **`loc`** only (strand is **not** used). Positions come from BED via `convert_bed_to_tss_tsv.py` (`loc = bed_start + 1`).

### v5 — `eval_pre_padding_max29bp/pr_curve_doublestrand_dense/output_100bp_tolerance`

- **Shell:**  
  `04_genome_scanning_v5_all_data_pool/eval_pre_padding_max29bp/run_all_evaluations.sh` (parallel PR jobs).
- **Python:**  
  `eval_pre_padding_max29bp/scripts/generate_pr_curve_doublestrand.py`
- **Inputs:**
  - **Predictions:** All-windows **double-strand** TSV (~9.28M rows): each **29 bp window** on `+` and `−` with a score.
  - **Reference:** `ecoli_all4classes_tss_filtered.csv` with **`Pos`** and **`Strand`** (strand-aware).

---

## 2. Evaluation mechanism (side-by-side)

| Aspect | v4 (classic WIG dense) | v5 (double-strand interval dense) |
|--------|-------------------------|-----------------------------------|
| **Prediction unit** | A **single base position** `P` with score ≥ threshold (from WIG). | A **29 bp interval** `[start, end]` (1-based inclusive) on **`+` or `−`**, score ≥ threshold. |
| **Strand** | **Ignored.** All TSS are merged into one sorted `loc` array. | **Required.** `+` windows evaluated only against `+` genes; `−` windows only against `−` genes. |
| **“Hit” for recall (per TSS)** | **Classic detection:** TSS `T` is “recovered” if **any** prediction position `P` with score ≥ *T* falls in the **accepted interval** (see below). | For each TSS, take **max score** among **strand-matching** windows whose **entire** 29 bp interval lies **inside** the (tolerance-expanded) promoter region; recall counts TSS with that max ≥ threshold. |
| **Promoter + tolerance (v4 defaults)** | One interval for **every** TSS: `[T − upstream_max − tol, T − upstream_min + tol]` with defaults `upstream_max=40`, `upstream_min=6`, `tol=100` → **`[T−140, T+94]`** (same geometry for all `loc`). | **Strand-specific:** `+`: `[T−39−100, T−6+100]` → **`[T−139, T+94]`**; `−`: `[T+6−100, T+39+100]` → **`[T−94, T+139]`**. |
| **Precision** | Among positions above threshold: fraction that are **TP**, where TP = ∃ TSS such that `P` lies in that TSS’s **same** accepted interval (binary search over sorted TSS). | Among **windows** above threshold (global sort by score): fraction labeled **TP** = window fully inside **some** true promoter on the **same** strand. |
| **Recall in PR script** | `matched_refs / N` with classic per-TSS detection (any point in window). | **Hybrid PR:** recall = fraction of TSS with **max** (contained, strand-correct window score) ≥ threshold — *not* the same numerator as “TP windows”. |

v4 docstring stresses **“CLASSIC DETECTION RECALL”** (one prediction can in principle help multiple TSS if intervals overlap; each TSS checked independently).

---

## 3. Reported results at **best F1** (from existing `pr_curve_data.tsv`)

These are **not** at the same threshold and use **different models** (type-1 LSTM WIG vs v5 double-strand scan), so numbers illustrate **method + model** combined.

| Pipeline | Best F1 | Recall | Implied FN (`11834 × (1 − recall)`) | `num_predictions` at best F1 |
|----------|---------|--------|--------------------------------------|------------------------------|
| **v4** `type1_dense` | **0.646** | **0.923** | **~917** | 46,866 |
| **v5** `doublestrand_dense` 100 bp | **0.467** | **0.804** | **~2,321** | 42,669 |

So on these saved curves, **v5 shows many more missed TSS (higher FN)** on the **recall axis** than v4.

---

## 4. Why FN can differ so much (main reasons)

### A. **Point vs 29 bp window (largest conceptual gap)**

- **v4:** It is enough that **one base** in the wide accepted interval `[T−140, T+94]` exceeds the threshold.
- **v5:** A window counts toward a TSS only if **all 29 bp** lie **inside** the strand-specific promoter (with tolerance). A **high score at a single base** does **not** guarantee that **any** full 29 bp window **fully contained** in the promoter has a high **max** score — especially near **edges** of the region.

So the **same biological locus** can pass v4’s “any point” rule but fail v5’s “fully contained window” rule → **more FN** for v5 for recall defined from **window maxima**.

### B. **Strand handling**

- **v4** does not use strand; every TSS uses the **same** scalar interval shape `[T−140, T+94]`.
- **v5** uses **different** intervals for `+` vs `−` (see table above). That changes which windows are **eligible** and which **strand’s** scores count. This can **increase or decrease** FN relative to v4 depending on the model; it is **not** a pure “correction” of v4.

### C. **Different predictors (not apples-to-apples)**

- v4 uses **type 1** LSTM **WIG** (single-track scores).
- v5 uses the **v5 all-data-pool** double-strand **interval scan** TSV (different architecture and **two** strands).

So part of the recall/FN gap is **model performance**, not only the metric.

### D. **Hybrid PR recall in v5**

In `generate_pr_curve_doublestrand.py`, **precision** is window-based and **recall** is reference-based (max score per TSS). That is **standard for this script** but means “FN” on the recall axis is **strictly** “TSS whose best in-region window score is still below threshold,” which couples tightly to **window** geometry and **double-strand** scoring.

---

## 5. What to compare when you need a fair metric comparison

- Same **reference** file and **strand** semantics (CSV with `Strand` vs loc-only TSV).
- Same **prediction source** (or accept that model change dominates).
- Align **promoter definition** (v4 uses 40/6; v5 uses 39/6 — 1 bp shift on upstream extent) and **tolerance**.
- Decide whether the scientific question needs **point-wise** (WIG-style) or **interval / window** (29 bp) detection.

---

## 6. Files in this folder

| File | Purpose |
|------|---------|
| `COMPARISON_v4_wig_vs_v5_doublestrand_dense.md` | This document |
| `best_f1_summary.tsv` | Best-F1 rows and implied FN for v4 type1_dense vs v5 100 bp |
| `build_summary.py` | Regenerates `best_f1_summary.tsv` from the two `pr_curve_data.tsv` paths |

Run: `python3 build_summary.py`
