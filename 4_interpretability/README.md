# 4. Interpretability

This directory contains scripts for interpreting model predictions and identifying important genes for cell cycle phase classification.

## Overview

**Purpose**:
- Understand which genes contribute most to predictions
- Validate biological relevance of learned features
- Compare with known cell cycle marker genes
- Generate gene importance rankings for publication

**Methods**:
- **SHAP** (SHapley Additive exPlanations) - Model-agnostic interpretability
- **Feature Importance** - For tree-based models (RF, LGBM)
- **Attention Weights** - For Feature Embedding model

## Directory Structure

```
4_interpretability/
├── shap_analysis.py          # SHAP analysis for all models
└── gene_enrichment.R         # GO/KEGG enrichment analysis
```

## SHAP Analysis

### What is SHAP?

SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a particular prediction, based on cooperative game theory.

**Key Properties**:
- **Model-agnostic**: Works with any ML/DL model
- **Locally accurate**: Explains individual predictions
- **Consistent**: Higher feature value = higher SHAP value (if truly important)
- **Additive**: Sum of SHAP values = model output - baseline

### Usage

```bash
# Run SHAP analysis on DNN3
python shap_analysis.py \
  --model_path ../2_model_training/saved_models/reh/simpledense/simpledense_reh_fold0.pt \
  --model_type neural \
  --benchmark gse146773 \
  --output shap_results/dnn3_gse146773/ \
  --n_samples 100

# Run SHAP on traditional ML model
python shap_analysis.py \
  --model_path ../2_model_training/saved_models/reh/random_forest/rf_reh_fold0.pkl \
  --model_type non_neural \
  --benchmark gse146773 \
  --output shap_results/rf_gse146773/ \
  --n_samples 100
```

**Parameters**:
- `--model_path`: Path to trained model (.pt or .pkl)
- `--model_type`: "neural" or "non_neural"
- `--benchmark`: Which benchmark to analyze (sup, gse146773, gse64016)
- `--n_samples`: Number of samples for SHAP (more = slower but more accurate)

### Output Files

```
shap_results/dnn3_gse146773/
├── benchmark_evaluation.csv           # Performance metrics on all 3 benchmarks
├── shap_summary_plot.png             # Beeswarm plot (top-20 genes)
├── shap_bar_plot.png                 # Mean |SHAP| values (top-20)
├── shap_dependence_G1.png            # Dependence plots for G1 phase
├── shap_dependence_S.png             # Dependence plots for S phase
├── shap_dependence_G2M.png           # Dependence plots for G2M phase
├── shap_values.npy                   # Raw SHAP values (n_samples × n_features × n_classes)
├── top_genes_G1.csv                  # Top-50 genes for G1 phase
├── top_genes_S.csv                   # Top-50 genes for S phase
└── top_genes_G2M.csv                 # Top-50 genes for G2M phase
```

### Interpreting SHAP Plots

**1. Summary Plot (Beeswarm)**:
- Each point is a cell
- x-axis: SHAP value (impact on prediction)
- Color: Gene expression (red=high, blue=low)
- Top genes ranked by mean |SHAP|

Example interpretation:
```
Gene: CDK1
- Most dots are on the right (positive SHAP) for G2M predictions
- Red dots (high expression) → push prediction toward G2M
- Blue dots (low expression) → push away from G2M
→ CDK1 highly expressed in G2M phase (expected!)
```

**2. Bar Plot**:
- Shows mean absolute SHAP value
- Higher = more important overall
- Averaged across all phases

**3. Dependence Plots**:
- x-axis: Gene expression
- y-axis: SHAP value for that phase
- Shows how gene expression affects phase prediction

Example:
```
Gene: PCNA, Phase: S
- Upward trend → Higher PCNA = stronger S prediction
- Flat line → Gene not important for this phase
```

### Top Genes CSV Format

```csv
Gene,Mean_SHAP,Std_SHAP,Rank
PCNA,0.342,0.125,1
MCM2,0.318,0.098,2
MCM7,0.295,0.102,3
...
```

**Mean_SHAP**: Average SHAP value magnitude (higher = more important)
**Std_SHAP**: Variability in SHAP values (high = context-dependent importance)
**Rank**: Importance ranking (1 = most important)

## Gene Enrichment Analysis

### Purpose
Validate biological relevance of top SHAP genes using GO/KEGG enrichment.

### Usage

```r
# R script
source("gene_enrichment.R")

# Load top genes from SHAP
top_g1_genes <- read.csv("shap_results/dnn3_gse146773/top_genes_G1.csv")
top_s_genes <- read.csv("shap_results/dnn3_gse146773/top_genes_S.csv")
top_g2m_genes <- read.csv("shap_results/dnn3_gse146773/top_genes_G2M.csv")

# Run enrichment (top-50 genes)
g1_enrichment <- run_enrichment(top_g1_genes$Gene[1:50], species="human")
s_enrichment <- run_enrichment(top_s_genes$Gene[1:50], species="human")
g2m_enrichment <- run_enrichment(top_g2m_genes$Gene[1:50], species="human")

# Save results
write.csv(g1_enrichment, "enrichment_G1.csv")
write.csv(s_enrichment, "enrichment_S.csv")
write.csv(g2m_enrichment, "enrichment_G2M.csv")
```

**Expected Enriched Terms**:
- **G1**: Cell growth, RNA processing, ribosome biogenesis
- **S**: DNA replication, nucleotide metabolism, DNA repair
- **G2M**: Cell division, mitotic spindle, chromosome segregation

## Complete Workflow

### Step 1: Benchmark Evaluation + SHAP

The `shap_analysis.py` script does both benchmark evaluation and SHAP analysis:

```bash
# DNN3 model
python shap_analysis.py \
  --model_path ../2_model_training/saved_models/reh/simpledense/simpledense_reh_fold0.pt \
  --model_type neural \
  --benchmark gse146773 \
  --output shap_results/dnn3/ \
  --n_samples 200

# Check output
cat shap_results/dnn3/benchmark_evaluation.csv  # Performance on all 3 benchmarks
ls shap_results/dnn3/                           # SHAP plots and gene rankings
```

### Step 2: Compare Top Genes with Known Markers

```python
import pandas as pd

# Load SHAP top genes
shap_s_genes = pd.read_csv("shap_results/dnn3/top_genes_S.csv")
shap_g2m_genes = pd.read_csv("shap_results/dnn3/top_genes_G2M.csv")

# Load Seurat marker genes
seurat_s_genes = pd.read_csv("../Cell_Cycle_marker_genes/_Seurat_S_genes.txt", header=None)[0].tolist()
seurat_g2m_genes = pd.read_csv("../Cell_Cycle_marker_genes/_Seurat_G2M_genes.txt", header=None)[0].tolist()

# Calculate overlap
s_overlap = set(shap_s_genes["Gene"][:50]) & set(seurat_s_genes)
g2m_overlap = set(shap_g2m_genes["Gene"][:50]) & set(seurat_g2m_genes)

print(f"S phase overlap: {len(s_overlap)}/50 genes")
print(f"G2M phase overlap: {len(g2m_overlap)}/50 genes")
print(f"Overlapping S genes: {s_overlap}")
print(f"Overlapping G2M genes: {g2m_overlap}")
```

**Expected Overlap**: 30-40% with Seurat markers (rest are novel findings)

### Step 3: Gene Enrichment Analysis

```bash
# Run enrichment in R
Rscript gene_enrichment.R shap_results/dnn3/top_genes_G1.csv shap_results/dnn3/top_genes_S.csv shap_results/dnn3/top_genes_G2M.csv

# Check enrichment results
cat enrichment_G1.csv
cat enrichment_S.csv
cat enrichment_G2M.csv
```

### Step 4: Create Summary Table for Paper

```python
import pandas as pd

# Compile top-10 genes per phase
dnn3_top10 = {
    "G1": pd.read_csv("shap_results/dnn3/top_genes_G1.csv")["Gene"][:10].tolist(),
    "S": pd.read_csv("shap_results/dnn3/top_genes_S.csv")["Gene"][:10].tolist(),
    "G2M": pd.read_csv("shap_results/dnn3/top_genes_G2M.csv")["Gene"][:10].tolist()
}

# Compare across models
fe_top10 = {
    "G1": pd.read_csv("shap_results/feature_embedding/top_genes_G1.csv")["Gene"][:10].tolist(),
    "S": pd.read_csv("shap_results/feature_embedding/top_genes_S.csv")["Gene"][:10].tolist(),
    "G2M": pd.read_csv("shap_results/feature_embedding/top_genes_G2M.csv")["Gene"][:10].tolist()
}

# Find consensus genes (appear in both models)
consensus_s = set(dnn3_top10["S"]) & set(fe_top10["S"])
consensus_g2m = set(dnn3_top10["G2M"]) & set(fe_top10["G2M"])

print("Consensus S genes:", consensus_s)
print("Consensus G2M genes:", consensus_g2m)
```

## Expected Results

### Top Genes by Phase

**G1 Phase** (Cell Growth):
- RPL/RPS genes (ribosomal proteins)
- Housekeeping genes
- Cell growth regulators

**S Phase** (DNA Replication):
- PCNA, MCM2-7 (DNA replication licensing)
- POLD1, POLE (DNA polymerases)
- RPA1-3 (replication protein A)
- RFC1-5 (replication factor C)

**G2M Phase** (Mitosis)**:
- CDK1, CCNB1/2 (mitotic cyclins/kinases)
- TOP2A (DNA topoisomerase)
- AURKB, PLK1 (mitotic kinases)
- KIF11, KIF23 (kinesins)
- CENPE (centromere protein)

### Validation with Known Markers

**Seurat S genes** (43 genes):
```
MCM2, MCM3, MCM4, MCM5, MCM6, MCM7, PCNA, POLD1, POLE, RFC1, RFC2, RFC3, RFC4, RFC5, RPA1, RPA2, RPA3, ...
```

**Seurat G2M genes** (54 genes):
```
CDK1, CCNB1, CCNB2, TOP2A, AURKB, PLK1, KIF11, KIF23, CENPE, CENPF, BIRC5, BUB1, BUB1B, ...
```

**Our Top-10 S genes** (example):
1. PCNA ✓ (Seurat)
2. MCM2 ✓ (Seurat)
3. MCM7 ✓ (Seurat)
4. POLD1 ✓ (Seurat)
5. RFC4 ✓ (Seurat)
6. RPA1 ✓ (Seurat)
7. TYMS (novel)
8. UHRF1 (novel)
9. NASP (novel)
10. POLE ✓ (Seurat)

**Overlap**: 7/10 genes match Seurat markers (excellent validation!)

## Troubleshooting

**SHAP Takes Too Long**:
```bash
# Reduce n_samples
python shap_analysis.py ... --n_samples 50

# Or use faster SHAP explainer (for neural nets)
# Modify shap_analysis.py to use shap.DeepExplainer instead of shap.KernelExplainer
```

**Memory Error During SHAP**:
```bash
# Process in batches
python shap_analysis.py ... --n_samples 50 --batch_size 10
```

**Gene Names Don't Match**:
```bash
# Check gene naming convention
# Should be gene symbols (e.g., "CDK1" not "ENSG00000170312")
# If using Ensembl IDs, convert to symbols first
```

**Low Enrichment Significance**:
```bash
# Increase number of top genes
python shap_analysis.py ... --n_top_genes 100

# Or use less stringent p-value threshold
# Modify gene_enrichment.R: p_cutoff = 0.1
```

## Advanced Analyses

### Compare SHAP Across Benchmarks

```python
# Do top genes generalize across benchmarks?
dnn3_gse146773_s = pd.read_csv("shap_results/dnn3_gse146773/top_genes_S.csv")
dnn3_gse64016_s = pd.read_csv("shap_results/dnn3_gse64016/top_genes_S.csv")

overlap = set(dnn3_gse146773_s["Gene"][:20]) & set(dnn3_gse64016_s["Gene"][:20])
print(f"Top-20 S genes overlap across benchmarks: {len(overlap)}/20")
```

**Expected**: 15-18/20 genes overlap (good generalization)

### Model-Specific vs. Consensus Genes

```python
# Which genes are important across multiple models?
dnn3_s = set(pd.read_csv("shap_results/dnn3/top_genes_S.csv")["Gene"][:20])
fe_s = set(pd.read_csv("shap_results/feature_embedding/top_genes_S.csv")["Gene"][:20])
cnn_s = set(pd.read_csv("shap_results/cnn/top_genes_S.csv")["Gene"][:20])

consensus_3models = dnn3_s & fe_s & cnn_s
print(f"Genes important in all 3 models: {consensus_3models}")
```

**Interpretation**: Consensus genes are most robust, use for biological interpretation.

## Visualization Tips

### Summary Plot
- Use for overview (shows distribution of SHAP values)
- Include in main paper figures
- Limit to top-20 genes for readability

### Dependence Plots
- Use for specific gene-phase relationships
- Include in supplementary materials
- Annotate with biological interpretation

### Bar Plot
- Simple ranking visualization
- Good for presentations
- Easy to compare across models

## References

- **SHAP**: Lundberg & Lee (2017) *NeurIPS*
- **TreeSHAP**: Lundberg et al. (2020) *Nat Mach Intell*
- **Gene Enrichment**: Ashburner et al. (2000) *Nat Genet* (GO)
- **Cell Cycle Genes**: Cyclebase.org, Whitfield et al. (2002) *Mol Biol Cell*

## Next Steps

After interpretability analysis:
1. Move to `../5_visualization/` to create publication figures
2. Document top genes and enrichment results in paper
3. Validate novel genes with literature search
4. Consider experimental validation of top novel genes
