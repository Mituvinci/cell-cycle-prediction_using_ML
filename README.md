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

## Directory Structure

```
cell_cycle_prediction/
 0_preprocessing/              # Data preprocessing scripts (TBD)
 1_consensus_labeling/         # Consensus label generation (TBD)
   predict/                  # Run 4 prediction tools
   analyze/                  # Contingency tables, heatmaps
   assign/                   # Phase reassignment
   merge/                    # Merge consensus labels
 2_model_training/             # ✅ Model training modules
   models/                   # ✅ Model architectures
     __init__.py
     dense_models.py       # DNN3, DNN5
     cnn_models.py         # CNNModel
     hybrid_models.py      # Hybrid, Feature Embedding
   utils/                    # ✅ Training utilities
       __init__.py
       training_utils.py     # Focal loss, training, evaluation
       data_utils.py         # Data loading, preprocessing
 3_evaluation/                 # Evaluation scripts (TBD)
 4_interpretability/           # SHAP analysis (TBD)
 5_visualization/              # Figure generation (TBD)
 configs/                      # ✅ Configuration files
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
 README.md                     # ✅ This file
 requirements.txt              # ✅ Python dependencies
 environment.yml               # ✅ Conda environment
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

### 1. Import Models

```python
from models import SimpleDenseModel, DeepDenseModel, CNNModel
import torch

# Create DNN3 model (top performer)
model = SimpleDenseModel(input_dim=2000, num_classes=3)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### 2. Load and Preprocess Data

```python
from utils import load_and_preprocess_data

# Load training data
reh_data, sup_data = load_and_preprocess_data(
    reh_path="data/Training_data/reh/filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv",
    sup_path="data/Training_data/sup/filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv",
    scaling_method='standard'
)
```

### 3. Train Model

```python
from utils import train_model, focal_loss
import torch.optim as optim

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = focal_loss

# Train
model = train_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=100,
    log_dir="models/saved_models/dnn3/logs",
    early_stopping_patience=10,
    device=device
)
```

### 4. Evaluate Model

```python
from utils import evaluate_model

metrics = evaluate_model(
    model=model,
    data_loader=test_loader,
    criterion=criterion,
    label_encoder=label_encoder,
    save_dir="results/",
    device=device,
    dataset_name="GSE146773"
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
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

- HPC Cluster: DOLLY SODS (155 GPUs)
- Reviewers: Briefings in Bioinformatics
- Existing Tools: Seurat, Tricycle, Revelio, ccAFv2
