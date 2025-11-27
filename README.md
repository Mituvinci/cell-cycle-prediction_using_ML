# Cell Cycle Phase Prediction Pipeline
## Deep Learning Models for Single-Cell RNA-Seq Data

> **Status:** Phase A (Foundation) - Modular Structure Created
> **Last Updated:** 2025-11-24

---

## Overview

This repository contains a modular, reproducible pipeline for predicting cell cycle phases (G1, S, G2M) from single-cell RNA-seq data using deep learning and traditional machine learning models.

### Key Features
- Consensus Labeling: Training labels generated from 4 existing tools (Seurat, Tricycle, Revelio, ccAFv2)
- Multiple Models: DNN3 (top performer), DNN5, CNN, Hybrid CNN, Feature Embedding, and traditional ML
- Class Balancing: SMOTE/ADASYN oversampling + undersampling
- Focal Loss: Addresses class imbalance during training
- Model Interpretability: SHAP analysis for biological validation
- Benchmark Validation: Evaluated on FUCCI-labeled datasets (GSE146773, GSE64016)

---

## Datasets

### Training Data
- **REH and SUP-B15**: Human leukemia cell lines (10x Chromium Multiome)
- **Download**: [GSE293316](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293316)

### Benchmark Data (Ground Truth)
- **GSE146773**: Human U-2 OS cells with FUCCI reporter
  - Download: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146773](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146773)
- **GSE64016**: Human ESCs with FUCCI reporter
  - Download: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64016](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64016)
- **Buettner mESC**: Mouse embryonic stem cells with ground-truth labels
  - 182 single-cell RNA-seq profiles from mouse ESCs
  - Cell cycle phases (G1, S, G2M) determined by Hoechst 33342 staining and FACS sorting
  - ~9,570 expressed genes (raw counts) with Ensembl IDs and gene symbols
  - Access via Bioconductor:
    ```R
    library(scRNAseq)
    sce <- BuettnerESCData()
    ```
  - **Citation**: Buettner, F., Natarajan, K. N., Casale, F. P., et al. *Computational analysis of cell-to-cell heterogeneity in single-cell RNA-sequencing data reveals hidden subpopulations of cells.* Nature Biotechnology 33, 155–160 (2015).

---

## Directory Structure

```
cell_cycle_prediction/
 0_preprocessing/              # Data preprocessing scripts (TBD)
 1_consensus_labeling/         # Consensus label generation (TBD)
   predict/                  # Run 4 prediction tools
   analyze/                  # Contingency tables, heatmaps
   assign/                   # Phase reassignment
   merge/                    # Merge consensus labels
 2_model_training/             # Model training modules
   models/                   # Model architectures
     __init__.py
     dense_models.py       # DNN3, DNN5
     cnn_models.py         # CNNModel
     hybrid_models.py      # Hybrid, Feature Embedding
   utils/                    # Training utilities
       __init__.py
       training_utils.py     # Focal loss, training, evaluation
       data_utils.py         # Data loading, preprocessing
 3_evaluation/                 # Evaluation scripts (TBD)
 4_interpretability/           # SHAP analysis (TBD)
 5_visualization/              # Figure generation (TBD)
 configs/                      # Configuration files
   datasets.yaml             # Dataset paths and parameters
   models/
     dnn3.yaml             # DNN3 configuration
   phase_mappings/
       training_data.yaml    # Phase mapping rationale
 data/                         # Data directories
   marker_genes/             # Reference marker genes
   raw/                      # Original data
   processed/                # Preprocessed data
   predictions/              # Tool predictions
 models/saved_models/          # Trained model weights
 results/                      # Results outputs
   metrics/
   figures/
   tables/
   phase_assignment_heatmaps/
 scripts/                      # Pipeline scripts (TBD)
 docs/                         # Documentation (TBD)
 README.md                     # This file
 requirements.txt              # Python dependencies
 environment.yml               # Conda environment
```

---

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
cd cell_cycle_prediction

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate cell_cycle_prediction
```

### Using pip

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Train Deep Learning Models

```bash
# Train DNN3 (SimpleDense - top performer) on REH data
python 2_model_training/train_deep_learning.py \
  --model simpledense \
  --dataset reh \
  --output models/saved_models/dnn3/ \
  --trials 50 \
  --cv 5

# Train DNN5 (DeepDense) on SUP data
python 2_model_training/train_deep_learning.py \
  --model deepdense \
  --dataset sup \
  --output models/saved_models/dnn5/ \
  --trials 30 \
  --cv 5

# Train CNN model (dataset defaults to reh)
python 2_model_training/train_deep_learning.py \
  --model cnn \
  --output models/saved_models/cnn/ \
  --trials 50 \
  --cv 5
```

### 2. Train Traditional ML Models

```bash
# Train Random Forest on REH data
python 2_model_training/train_traditional_ml.py \
  --model random_forest \
  --dataset reh \
  --output models/saved_models/random_forest/ \
  --trials 50 \
  --cv 5

# Train LightGBM on SUP data
python 2_model_training/train_traditional_ml.py \
  --model lgbm \
  --dataset sup \
  --output models/saved_models/lgbm/ \
  --trials 50 \
  --cv 5

# Train Ensemble (VotingClassifier: AdaBoost + RF + LGBM)
python 2_model_training/train_traditional_ml.py \
  --model ensemble \
  --dataset reh \
  --output models/saved_models/ensemble_tml/ \
  --trials 50 \
  --cv 5
```

### 3. Evaluate Models on Benchmark Data

```bash
# Evaluate Deep Learning model on all benchmarks
python 3_evaluation/evaluate_models.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --benchmarks SUP GSE146773 GSE64016 \
  --output results/dnn3_all_benchmarks.csv

# Evaluate Traditional ML model on all benchmarks
python 3_evaluation/evaluate_models.py \
  --model_path models/saved_models/random_forest/rf_NFT_reh_fld_1.joblib \
  --benchmarks SUP GSE146773 GSE64016 \
  --output results/rf_all_benchmarks.csv

# Evaluate on specific benchmarks only
python 3_evaluation/evaluate_models.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --benchmarks GSE146773 \
  --output results/dnn3_gse146773_only.csv
```

### 4. Ensemble Methods

Ensemble fusion is implemented in `3_evaluation/ensemble_fusion.py` with two methods:
- **Score Fusion**: Averages predicted probabilities across models
- **Decision Fusion**: Majority voting across model predictions

Example usage (call the functions directly in Python):

```python
from ensemble_fusion import score_level_fusion, decision_level_fusion

# Top-3 models
model_paths = [
    "models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt",
    "models/saved_models/dnn5/dnn5_NFT_reh_fld_2.pt",
    "models/saved_models/cnn/cnn_NFT_reh_fld_3.pt"
]

# Score fusion on GSE146773
result_score = score_level_fusion(model_paths, "GSE146773")
result_score.to_csv("results/top3_score_gse146773.csv", index=False)

# Decision fusion on GSE64016
result_decision = decision_level_fusion(model_paths, "GSE64016")
result_decision.to_csv("results/top3_decision_gse64016.csv", index=False)
```

### 5. SHAP Interpretability Analysis

```bash
# SHAP analysis for DNN3
python 4_interpretability/shap_analysis.py \
  --model_dir models/saved_models/dnn3/fold_0/ \
  --data data/processed/consensus_labels.csv \
  --model_type deep_learning \
  --output_dir results/shap/dnn3/

# SHAP analysis for Random Forest
python 4_interpretability/shap_analysis.py \
  --model_dir models/saved_models/random_forest/fold_0/ \
  --data data/processed/consensus_labels.csv \
  --model_type traditional_ml \
  --output_dir results/shap/random_forest/
```

---

## Using Custom Data

The pipeline supports training and evaluation with your own datasets while keeping REH/SUP as defaults.

### Custom Training Data

**CSV Format Required:**
```
cell_id,phase_label,gene1,gene2,gene3,...
CELL_001,G1,2.5,3.1,0.8,...
CELL_002,S,1.2,4.5,2.1,...
CELL_003,G2M,3.4,1.9,5.2,...
```

**Important:**
- **First column**: Cell ID (any name)
- **Second column**: Phase label (must be `G1`, `S`, or `G2M`)
- **Remaining columns**: Gene expression values (normalized)

**Train with Custom Data:**

```bash
# Deep Learning with custom data
python 2_model_training/train_deep_learning.py \
  --model simpledense \
  --data /path/to/your_training_data.csv \
  --output models/custom_model/ \
  --trials 50 \
  --cv 5

# Traditional ML with custom data
python 2_model_training/train_traditional_ml.py \
  --model random_forest \
  --data /path/to/your_training_data.csv \
  --output models/custom_rf/ \
  --trials 50
```

**Note:** When `--data` is provided, the `--dataset` argument (reh/sup) is ignored.

### Custom Benchmark Data

**Same CSV format as training data** (cell_id, phase_label, genes...)

**Evaluate on Custom Benchmark:**

```bash
# Evaluate model on custom benchmark
python 3_evaluation/evaluate_models.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --custom_benchmark /path/to/your_benchmark.csv \
  --custom_benchmark_name "MyBenchmark" \
  --output results/custom_evaluation.csv

# Combine standard + custom benchmarks
python 3_evaluation/evaluate_models.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --benchmarks GSE146773 GSE64016 \
  --custom_benchmark /path/to/your_benchmark.csv \
  --custom_benchmark_name "MyBenchmark"
```

**SHAP Analysis on Custom Benchmark:**

```bash
python 4_interpretability/run_shap_analysis.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --custom_benchmark /path/to/your_benchmark.csv \
  --custom_benchmark_name "MyBenchmark" \
  --output_dir results/shap/custom/
```

### Data Preprocessing Tips

1. **Normalization**: Use Seurat normalization (LogNormalize) or similar
2. **Gene Selection**: Include only highly variable genes or marker genes
3. **Phase Labels**: Must be exactly `G1`, `S`, or `G2M` (case-sensitive)
4. **Missing Features**: Pipeline automatically handles gene mismatches by:
   - Adding missing genes as zeros
   - Dropping extra genes
   - Reordering to match training features

---

## Models

### Deep Learning Models

| Model | Description | Architecture | Performance |
|-------|-------------|--------------|-------------|
| DNN3 | 3-layer dense network | input→128→64→3 | Top performer (75% accuracy on benchmarks) |
| DNN5 | 5-layer dense network | input→256→128→64→3 | Deep architecture for complex patterns |
| CNN | 1D Convolutional | Conv(32)→Conv(64)→Dense | Captures local gene patterns |
| Hybrid | CNN + Dense | Conv→Dense layers | Feature extraction + classification |
| Feature Embedding | Learned embedding | Embedding→Dense | Dimensionality reduction |

### Traditional ML Models
- AdaBoost
- Random Forest
- LightGBM
- Ensemble (Embedding3TML)

### Ensemble Models
- Top-3 Decision Fusion
- Top-3 Score Fusion

---

## Configuration

All parameters are configured via YAML files in `configs/`:

### Dataset Configuration (`configs/datasets.yaml`)
- Training data paths (REH, SUP-B15)
- Benchmark data paths (GSE146773, GSE64016)
- Preprocessing parameters
- Class labels

### Model Configuration (`configs/models/dnn3.yaml`)
- Model architecture
- Training hyperparameters
- Optimizer settings
- Loss function parameters

### Phase Mapping (`configs/phase_mappings/training_data.yaml`)
- Sub-phase to main phase mappings
- Rationale for each mapping based on heatmap analysis

---

## Phase Mapping Methodology

### Critical Manual Step

Tools predict different numbers of phases (3-8 phases). We map them to 3 main phases:

1. Create Contingency Tables: Compare predictions from different tools
2. Calculate Obs/Expected Ratios: Identify phase co-occurrences
3. Generate Heatmaps: Visualize ratios with color gradients
4. Manual Phase Assignment: Inspect heatmap colors to determine mappings

Example mappings:
- `G1.S → S` (high correlation with S phase)
- `G2, G2.M → G2M` (high correlation with G2M)
- `M.G1 → G1` (transition to G1)

See `configs/phase_mappings/training_data.yaml` for complete rationale.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{anonymous2025cellcycle,
  title={Deep Learning Models for Cell Cycle Phase Prediction from Single-Cell RNA Sequencing Data},
  author={Anonymous},
  booktitle={Under Review},
  year={2025}
}
```

---

## Acknowledgments

- Existing Tools: Seurat, Tricycle, Revelio, ccAFv2
