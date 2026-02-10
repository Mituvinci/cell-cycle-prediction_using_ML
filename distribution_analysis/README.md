# Phase 1: Distribution Diagnostic Analysis

## Purpose

Diagnose why REH-trained models perform poorly on external benchmarks despite good performance on SUP.

**IMPORTANT:** Analysis uses ONLY the 12,012 genes from `gene_lists/reh_3benchmark_sup.txt` (intersection of REH + 3 benchmarks + SUP). This ensures fair comparison - we analyze the SAME genes that were used during model training.

**Current Performance:**
- REH models on SUP: 80-90% accuracy (GOOD)
- REH models on GSE146773/GSE64016/Buettner_mESC: <67% accuracy (POOR)

**Hypothesis:** Distribution mismatch due to different sequencing platforms or normalization parameters.

---

## What This Analysis Does

### 1. Expression Value Distribution
- Compares min, max, mean, median, quartiles across all datasets
- Checks if value ranges are consistent
- Identifies if all datasets are in same scale

### 2. Normalization Check
- Verifies if all datasets are log-normalized
- Log-normalized data should have values in range [0, 10]
- Detects if different normalization parameters were used

### 3. Sparsity Analysis
- Measures percentage of zero values in each dataset
- Different platforms have different dropout rates
- 10x Chromium: higher sparsity than Smart-seq2

### 4. Gene Detection Rates
- Average genes detected per cell
- Total expression per cell
- Platform-specific detection capabilities

### 5. Visual Comparisons
- Histograms of expression distributions
- Box plots and violin plots
- Q-Q plots comparing REH vs each benchmark
- Per-cell statistics

---

## How to Run

```bash
# Option 1: Run via shell script (recommended)
bash distribution_analysis/run_phase1_analysis.sh

# Option 2: Run Python directly
conda activate pytorch
python distribution_analysis/phase1_distribution_diagnostic.py
```

---

## Output Files

All results saved to: `distribution_analysis/results/`

### Summary Files
- **distribution_statistics.csv** - Comprehensive statistics table
- **DIAGNOSTIC_REPORT.txt** - Text report with findings and recommendations

### Plots (300 DPI, publication-quality)
- **1_expression_histograms.png** - Individual dataset distributions
- **2_expression_boxplots.png** - Side-by-side box plot comparison
- **3_expression_violinplots.png** - Violin plot comparison
- **4_qq_plots.png** - Quantile-quantile plots (REH vs each benchmark)
- **5_per_cell_statistics.png** - Gene detection and total counts
- **6_sparsity_comparison.png** - Zero proportion comparison

---

## Datasets Analyzed

All datasets filtered to **12,012 genes** (intersection used during training).

| Dataset | Type | Platform | Original Genes | Filtered Genes | Details |
|---------|------|----------|----------------|----------------|---------|
| **REH** | Training | 10x Chromium (multiomics) | ~12,490 | 12,012 | In-house data |
| **SUP-B15** | Training | 10x Chromium (multiomics) | ~12,221 | 12,012 | In-house data |
| **GSE146773** | Benchmark | Smart-seq2 | ~42,728 | 12,012 | [GEO Link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146773) |
| **GSE64016** | Benchmark | Fluidigm C1 | ~19,084 | 12,012 | H1-Fucci hESCs (247 cells), [GEO Link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64016) |
| **Buettner_mESC** | Benchmark | E-MTAB-2805 (mESC-SMARTer) | ~38,237 | 12,012 | [ArrayExpress Link](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-2805/) |

**Gene List:** `gene_lists/reh_3benchmark_sup.txt` (intersection)

### Benchmark Details:

- **GSE146773:** Smart-seq2 platform, higher sequencing depth
- **GSE64016:** Fluidigm C1 platform. Dataset contains 247 H1-Fucci single cells used to confirm cell cycle gene clusters. Normalized expected counts provided.
- **Buettner_mESC (E-MTAB-2805):** Often referred to as mESC-SMARTer. Mouse embryonic stem cells profiled using SMART-seq based protocol.

---

## What to Look For

### SCENARIO 1: Expression Ranges Differ Significantly
**Finding:** Max values differ by >1.5x across datasets

**Interpretation:** Different normalization parameters or methods were used

**Solution:** Re-normalize ALL datasets with IDENTICAL Seurat parameters:
- Scale factor: 10000
- Log base: natural log (ln)
- Pseudocount: 1

### SCENARIO 2: Sparsity Differs Dramatically
**Finding:** Zero percentages differ by >20% across datasets

**Interpretation:** Platform effect - different dropout rates

**Options:**
- Train separate models per platform
- Use platform-aware normalization
- Apply imputation before training

### SCENARIO 3: All Metrics Look Similar
**Finding:** Distributions are comparable across datasets

**Interpretation:** 67% accuracy may be the realistic cross-platform limit

**Action:**
- Compare your performance to published tools on same benchmarks
- Report cross-platform limitations in manuscript
- Emphasize within-platform performance

---

## For Reviewers

This analysis addresses Reviewer concerns about:
1. Why models don't generalize across platforms
2. Whether normalization was consistent
3. If platform effects were considered

Include in manuscript supplementary materials:
- distribution_statistics.csv (Table S1)
- All plots (Figures S1-S6)
- DIAGNOSTIC_REPORT.txt findings in Methods section

---

## Next Steps Based on Findings

### If Normalization Differs:
1. Re-normalize all datasets with same Seurat parameters
2. Re-run training and evaluation
3. Compare new results

### If Platform Effects Dominate:
1. Train platform-specific models
2. Report cross-platform performance separately
3. Discuss limitations in manuscript

### If Distributions Match:
1. Accept 67% as realistic limit
2. Focus on improving within-platform performance
3. Benchmark against published tools

---

## Scientific Validity

**Is this approach valid?**

YES. This is standard practice in multi-study machine learning:
- Checking distribution consistency is REQUIRED
- Re-normalizing with same parameters is FAIR (not cheating)
- Reporting platform effects is HONEST science

**What NOT to do:**
- Don't adjust test data to match training (data leakage)
- Don't use batch correction that removes biological signal
- Don't hide platform limitations

---

## Questions?

This analysis follows best practices for:
- Cross-platform single-cell analysis
- Machine learning model evaluation
- Transparent reporting of limitations

The goal is to understand WHY performance differs, not to artificially inflate metrics.
