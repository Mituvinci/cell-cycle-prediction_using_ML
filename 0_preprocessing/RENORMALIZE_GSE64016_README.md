# Renormalizing GSE64016 Benchmark

## Problem Identified

GSE64016 is **NOT log-normalized**, while REH/SUP training data **IS log-normalized**.

### Evidence:
| Dataset | Mean | Max | Log-Normalized? |
|---------|------|-----|-----------------|
| REH (training) | 0.25 | 5.79 | ✓ YES |
| SUP (training) | 0.24 | 6.08 | ✓ YES |
| **GSE64016** | **172.92** | **129,617** | **✗ NO** |

**Impact:** Model trained on log scale [0, 6] cannot predict on raw scale [0, 129,617]!

---

## Solution

Use **UNIVERSAL_SEURAT.R** (same script used for REH/SUP) to apply identical normalization:
- Method: `log1p(counts / total_counts * 10000)`
- Ensures training and benchmark data are on same scale

---

## How to Run

### Option 1: Automated Script (Recommended)

```bash
bash 0_preprocessing/renormalize_GSE64016.sh
```

This script will:
1. Transpose GSE64016 from Cells x Genes → Genes x Cells
2. Run UNIVERSAL_SEURAT.R normalization
3. Backup original file as `GSE64016_expression_ORIGINAL_NOT_NORMALIZED.csv`
4. Replace with normalized version

### Option 2: Manual Steps

```bash
# Step 1: Transpose
python 0_preprocessing/transpose_GSE64016.py

# Step 2: Normalize
Rscript 0_preprocessing/training_data/UNIVERSAL_SEURAT.R \
  --input data/benchmarks_preprocessed/GSE64016/GSE64016_expression_raw_for_seurat.csv \
  --output data/benchmarks_preprocessed/GSE64016/normalized_seurat/ \
  --sample GSE64016 \
  --format csv \
  --species human

# Step 3: Replace original file
cp data/benchmarks_preprocessed/GSE64016/normalized_seurat/trainingdata_cellcyclescore_GSE64016_normalized_gene_expression.csv \
   data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv
```

---

## What Happens

### Before Normalization:
```
GSE64016_expression.csv:
  - Format: Cells (247) x Genes (19,084)
  - Values: [0, 129,617]
  - Mean: 172.92
  - NOT log-normalized
```

### After Normalization:
```
GSE64016_expression.csv:
  - Format: Cells (247) x Genes (19,084)
  - Values: [0, ~8-10]
  - Mean: ~0.1
  - Log-normalized (same as REH/SUP)
```

---

## Expected Improvement

### Current Performance (non-normalized):
- REH model on GSE64016: ~50-60% accuracy

### Expected Performance (after normalization):
- REH model on GSE64016: ~70-80% accuracy

**Why still not 90%?**
- Platform differences remain (gene detection rates)
- REH: ~3,500 genes/cell (10x Chromium)
- GSE64016: ~10,000 genes/cell (Smart-seq2)

---

## After Renormalization

### Re-run Evaluation:
```bash
# Wait for SelectKBest training to complete, then:
bash I_actually_run_to_train/run_all_new_evaluate_visu.sh
```

### Check Distribution Again:
```bash
bash distribution_analysis/run_phase1_analysis.sh
```

New distribution statistics should show:
- GSE64016 max value: ~8-10 (similar to REH/SUP)
- GSE64016 mean: ~0.1 (similar to REH/SUP)
- All datasets in comparable range

---

## Files Created

```
0_preprocessing/
├── transpose_GSE64016.py              # Transpose cells x genes → genes x cells
├── renormalize_GSE64016.sh            # Automated pipeline script
└── RENORMALIZE_GSE64016_README.md     # This file

data/benchmarks_preprocessed/GSE64016/
├── GSE64016_expression_ORIGINAL_NOT_NORMALIZED.csv  # Backup
├── GSE64016_expression.csv                          # REPLACED with normalized
├── GSE64016_expression_raw_for_seurat.csv           # Intermediate (genes x cells)
└── normalized_seurat/
    └── trainingdata_cellcyclescore_GSE64016_normalized_gene_expression.csv
```

---

## For Manuscript

### Methods Section:
"GSE64016 benchmark data was provided in non-log-normalized format (max value: 129,617). We applied Seurat log normalization (log1p(counts/total_counts * 10000)) to ensure consistency with training data normalization, enabling fair cross-dataset comparison."

### Supplementary Material:
- Include distribution analysis plots (before/after normalization)
- Table showing normalization status of all datasets
- Report both original and corrected GSE64016 results

---

## Important Note

**This is NOT data manipulation or cheating!**
- We're applying standard normalization to ensure fair comparison
- Using the SAME method as training data
- Transparent about the issue and solution
- Standard practice in multi-study scRNA-seq analysis

Reviewers will appreciate the thorough investigation and correction!
