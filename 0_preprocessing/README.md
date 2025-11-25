# 0. Preprocessing & Tool-Based Labeling

This directory contains scripts for generating cell cycle phase labels using 4 existing computational tools. Since our training datasets (REH and SUP-B15 cell lines) lack ground truth labels, we use consensus labeling from multiple established tools.

## Overview

**Purpose**: Generate cell cycle phase predictions from 4 computational tools to create consensus training labels.

**Tools Used**:
1. **Seurat CellCycleScore** - Based on marker gene expression
2. **Tricycle** - Projects cells onto a circular cell cycle manifold
3. **ccAFv2** - Deep learning classifier trained on reference datasets
4. **Revelio** - Cyclic gene expression patterns

## Directory Structure

```
0_preprocessing/
├── training_data/          # Label REH & SUP-B15 for model training
│   ├── 0_seurat_label_reh_sup.R
│   ├── 0_tricycle_label_reh_sup.R
│   ├── 0_revelio_label_reh_sup.R
│   ├── 0_ccaf_label_reh_sup.ipynb
│   └── 0_ccafv2_label_reh_sup.ipynb
│
├── benchmark_data/         # Label benchmark datasets for validation
│   ├── GSE146773/         # SmartSeq2, FUCCI ground truth
│   │   ├── 0_seurat_predict_gse146773.R
│   │   ├── 0_tricycle_predict_gse146773.R
│   │   ├── 0_revelio_predict_gse146773.R
│   │   ├── 0_ccafv2_predict_gse146773.ipynb
│   │   └── 0_add_fucci_ground_truth.ipynb
│   │
│   └── GSE64016/          # Fluidigm C1, FUCCI ground truth
│       ├── 0_seurat_predict_gse64016.R
│       ├── 0_tricycle_predict_gse64016.R
│       ├── 0_revelio_predict_gse64016.R
│       └── 0_ccafv2_predict_gse64016.ipynb
│
└── format_conversion/     # Convert SmartSeq2 to 10x format
    └── GSE146773_from_smart_seq2_to_10x_chromium/
```

## Workflow

### Step 1: Label Training Data (REH & SUP-B15)
**Location**: `training_data/`

Run all 4 tools on REH and SUP-B15 cell lines (10x Multiome):
```bash
# Run Seurat
Rscript training_data/0_seurat_label_reh_sup.R

# Run Tricycle
Rscript training_data/0_tricycle_label_reh_sup.R

# Run Revelio
Rscript training_data/0_revelio_label_reh_sup.R

# Run ccAF v1 (Jupyter notebook)
jupyter notebook training_data/0_ccaf_label_reh_sup.ipynb

# Run ccAF v2 (Jupyter notebook)
jupyter notebook training_data/0_ccafv2_label_reh_sup.ipynb
```

**Output**: CSV files with predicted cell cycle phases for each tool.

### Step 2: Label Benchmark Data
**Location**: `benchmark_data/GSE146773/` and `benchmark_data/GSE64016/`

Run the same 4 tools on benchmark datasets (with ground truth FUCCI labels):

**GSE146773** (SmartSeq2, 230 cells):
```bash
# Format conversion required first (SmartSeq2 → 10x)
cd format_conversion/GSE146773_from_smart_seq2_to_10x_chromium/
bash step.sh

# Then run predictions
cd ../../benchmark_data/GSE146773/
Rscript 0_seurat_predict_gse146773.R
Rscript 0_tricycle_predict_gse146773.R
Rscript 0_revelio_predict_gse146773.R
jupyter notebook 0_ccafv2_predict_gse146773.ipynb

# Add FUCCI ground truth labels
jupyter notebook 0_add_fucci_ground_truth.ipynb
```

**GSE64016** (Fluidigm C1, 288 cells):
```bash
cd benchmark_data/GSE64016/
Rscript 0_seurat_predict_gse64016.R
Rscript 0_tricycle_predict_gse64016.R
Rscript 0_revelio_predict_gse64016.R
jupyter notebook 0_ccafv2_predict_gse64016.ipynb
```

**Output**: CSV files with predictions from all 4 tools on benchmark data.

### Step 3: Create Consensus Labels
**Location**: `../1_consensus_labeling/`

After obtaining predictions from all 4 tools, proceed to the consensus labeling pipeline:
1. Create contingency tables comparing tool predictions
2. Calculate observed vs. expected ratios
3. Generate heatmaps for visual phase mapping
4. Manually map sub-phases to 3 main phases (G1, S, G2M)
5. Merge predictions where ≥3 tools agree

See `../1_consensus_labeling/README.md` for details.

## Key Notes

### Phase Labels Produced
Different tools predict different numbers of phases:
- **Seurat**: G1, S, G2M
- **Tricycle**: G1, S, G2, M (continuous position mapped to stages)
- **ccAFv2**: G1, Late G1, S, G2, G2/M, M, M/G1, Unknown
- **Revelio**: G0, G1, S, G2, M

**Next step**: Consensus labeling harmonizes these into 3 main phases (G1, S, G2M).

### Data Requirements
- **Training data**: 10x Multiome (Gene Expression matrix)
- **Benchmark data**: Various platforms (10x, SmartSeq2, Fluidigm C1)
- **Format conversion**: Some benchmarks require conversion to 10x format

### R Dependencies
```r
library(Seurat)
library(Signac)
library(tricycle)
library(Revelio)
library(SingleCellExperiment)
library(EnsDb.Hsapiens.v86)
```

### Python Dependencies
```python
import scanpy as sc
import ccAF
import ccAFv2
import pandas as pd
import numpy as np
```

## Output Files

Each tool generates a CSV file with the following structure:
```
CellID,Predicted
AAACAGCCAATATGGA-1,G1
AAACAGCCACATTGCA-1,S
AAACAGCCAGAATGAC-1,G2M
...
```

These files are then used as input for the consensus labeling pipeline in `../1_consensus_labeling/`.

## References

- **Seurat**: Satija et al. (2015) *Nat Biotechnol*
- **Tricycle**: Zheng et al. (2022) *Genome Biol*
- **ccAF**: Liu et al. (2022) *Nucleic Acids Res*
- **Revelio**: Schwabe et al. (2020) *Nat Commun*

## Next Steps

After running all preprocessing scripts:
1. Move to `../1_consensus_labeling/` to create consensus labels
2. Then to `../2_model_training/` to train ML/DL models
3. Finally to `../3_evaluation/` to evaluate on benchmark data
