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
- **GSE64016**: Mouse ESCs with FUCCI reporter
  - Download: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64016](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64016)

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
# Train DNN3 (top performer) with nested CV
python 2_model_training/train_deep_learning.py \
  --model dnn3 \
  --data data/processed/consensus_labels.csv \
  --n_trials 50 \
  --outer_folds 5 \
  --inner_folds 3 \
  --feature_selection none \
  --output_dir models/saved_models/dnn3/

# Train with different architecture
python 2_model_training/train_deep_learning.py \
  --model dnn5 \
  --n_trials 30 \
  --outer_folds 5 \
  --inner_folds 3 \
  --output_dir models/saved_models/dnn5/

# Train CNN model
python 2_model_training/train_deep_learning.py \
  --model cnn \
  --n_trials 50 \
  --output_dir models/saved_models/cnn/
```

### 2. Train Traditional ML Models

```bash
# Train Random Forest
python 2_model_training/train_traditional_ml.py \
  --model random_forest \
  --data data/processed/consensus_labels.csv \
  --n_trials 50 \
  --outer_folds 5 \
  --inner_folds 3 \
  --output_dir models/saved_models/random_forest/

# Train LightGBM
python 2_model_training/train_traditional_ml.py \
  --model lgbm \
  --n_trials 50 \
  --output_dir models/saved_models/lgbm/

# Train Ensemble (VotingClassifier)
python 2_model_training/train_traditional_ml.py \
  --model ensemble \
  --n_trials 50 \
  --output_dir models/saved_models/ensemble_tml/
```

### 3. Evaluate Models on Benchmark Data

```bash
# Evaluate single model
python 3_evaluation/evaluate_models.py \
  --model_dir models/saved_models/dnn3/fold_0/ \
  --benchmark_data data/processed/GSE146773_preprocessed.csv \
  --ground_truth data/processed/GSE146773_ground_truth.csv \
  --output results/dnn3_gse146773_evaluation.csv
```

### 4. Ensemble Methods

```bash
# Top-3 Deep Learning Score Fusion
python 3_evaluation/ensemble_fusion.py \
  --fusion_type score \
  --top_k 3 \
  --model_dirs models/saved_models/dnn3/fold_0/ models/saved_models/dnn5/fold_0/ models/saved_models/cnn/fold_0/ \
  --benchmark_data data/processed/GSE146773_preprocessed.csv \
  --ground_truth data/processed/GSE146773_ground_truth.csv \
  --output results/ensemble_top3_score.csv

# Top-5 Deep Learning Decision Fusion
python 3_evaluation/ensemble_fusion.py \
  --fusion_type decision \
  --top_k 5 \
  --model_dirs models/saved_models/dnn3/fold_0/ models/saved_models/dnn5/fold_0/ models/saved_models/cnn/fold_0/ models/saved_models/hybrid/fold_0/ models/saved_models/feature_embedding/fold_0/ \
  --benchmark_data data/processed/GSE146773_preprocessed.csv \
  --ground_truth data/processed/GSE146773_ground_truth.csv \
  --output results/ensemble_top5_decision.csv
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
