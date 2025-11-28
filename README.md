# Cell Cycle Phase Prediction Pipeline
## Deep Learning Models for Single-Cell RNA-Seq Data

> **Status:** Phase D - SHAP Analysis & Custom Data Support Complete
> **Last Updated:** 2025-11-28

---

## Overview

This repository contains a **modular, reproducible pipeline** for predicting cell cycle phases (G1, S, G2M) from single-cell RNA-seq data using deep learning and traditional machine learning models.

### Key Features
- **Flexible Consensus Labeling**: Create training labels from ANY combination of prediction tools
- **Multiple Models**: DNN3 (top performer), DNN5, CNN, Hybrid CNN, Feature Embedding, and traditional ML
- **Custom Data Support**: Train and evaluate on your own datasets
- **Class Balancing**: SMOTE/ADASYN oversampling + undersampling
- **Focal Loss**: Addresses class imbalance during training
- **Model Interpretability**: SHAP analysis for biological validation (both DL and TML)
- **Benchmark Validation**: Evaluated on FUCCI-labeled datasets (GSE146773, GSE64016, Buettner_mESC)
- **Species-Independent Gene Naming**: Automatic gene name capitalization for cross-species model compatibility

---

## Species-Independent Gene Naming

**NEW**: The pipeline now automatically capitalizes all gene names to enable cross-species model training and evaluation.

### How It Works
- **Automatic capitalization**: All gene names converted to first letter uppercase, rest lowercase
- **Mouse genes**: `Gnai3, Pbsn, Cdc45, H19`
- **Human genes**: `Gapdh, Actb, Tp53, Myc`
- **Benchmark data**: Expression and ground truth labels stored separately

### Benefits
✅ **Train on human, evaluate on mouse**: Models trained on REH/SUP (human) can be evaluated on Buettner_mESC (mouse)
✅ **Train on mouse, evaluate on human**: Works in reverse too!
✅ **No gene name mismatches**: Consistent feature naming across species
✅ **Automatic handling**: All data loading functions apply capitalization

### Example: Cross-Species Evaluation
```bash
# Train model on human REH data
python 2_model_training/train_deep_learning.py \
  --model simpledense \
  --dataset reh \
  --output models/human_model/

# Evaluate on mouse Buettner_mESC benchmark (works seamlessly!)
python 3_evaluation/evaluate_models.py \
  --model_path models/human_model/simpledense_NFT_reh_fld_1.pt \
  --benchmarks Buettner_mESC
```

### Data Preparation
Before using Buettner_mESC benchmark, run:
```bash
# Clean benchmark data (remove embedded Phase column, capitalize genes)
python utils/clean_buettner_benchmark.py

# Test complete standardization pipeline
bash test_data_standardization.sh
```

See `DATA_STANDARDIZATION_SUMMARY.md` for complete documentation.

---

## Datasets

### Training Data (Default)
- **REH and SUP-B15**: Human leukemia cell lines (10x Chromium Multiome)
- **Download**: [GSE293316](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293316)
-**PBMCs from a Healthy Donor: Whole Transcriptome Analysis Human scRNA 10x Chrmium **: https://www.10xgenomics.com/datasets/pbm-cs-from-a-healthy-donor-whole-transcriptome-analysis-3-1-standard-4-0-0
-**10k Brain Cells from an E18 Mouse (v3 chemistry) Mouse  scRNA 10x Chrmium **: https://www.10xgenomics.com/datasets/10-k-brain-cells-from-an-e-18-mouse-v-3-chemistry-3-standard-3-0-0


### Benchmark Data (Ground Truth)
- **GSE146773**: Human U-2 OS cells with FUCCI reporter
  - Download: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146773](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146773)
- **GSE64016**: Human ESCs with FUCCI reporter
  - Download: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64016](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64016)
- **Buettner_mESC**: Mouse embryonic stem cells with ground-truth labels
  - 182 single-cell RNA-seq profiles from mouse ESCs
  - Cell cycle phases (G1, S, G2M) determined by Hoechst 33342 staining and FACS sorting
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
├── 1_consensus_labeling/         # Create consensus training labels
│   ├── analyze/
│   │   ├── create_contingency_flexible.py   # Compare tool predictions
│   │   ├── generate_heatmap_flexible.py     # Obs/expect ratio heatmaps
│   │   └── run_analysis_mouse_human.sh      # Master analysis script
│   ├── assign/                              # Phase reassignment (TBD)
│   ├── merge/
│   │   └── merge_consensus.py               # Merge where ≥N tools agree
│   └── WORKFLOW_MOUSE_HUMAN.md              # Detailed consensus workflow
│
├── 2_model_training/             # Model training modules
│   ├── models/                   # Model architectures
│   │   ├── dense_models.py       # SimpleDenseModel (DNN3), DeepDenseModel (DNN5)
│   │   ├── cnn_models.py         # CNNModel
│   │   └── hybrid_models.py      # HybridCNNDenseModel, FeatureEmbeddingModel
│   ├── utils/                    # Training utilities
│   │   ├── training_utils.py     # Focal loss, training, evaluation
│   │   ├── data_utils.py         # Data loading, preprocessing, custom data
│   │   ├── optuna_utils.py       # Hyperparameter optimization (DL)
│   │   ├── optuna_tml.py         # Hyperparameter optimization (TML)
│   │   ├── nested_cv.py          # Nested cross-validation
│   │   └── io_utils.py           # Save/load utilities
│   ├── train_deep_learning.py    # CLI for DL training
│   └── train_traditional_ml.py   # CLI for TML training
│
├── 3_evaluation/                 # Evaluation scripts
│   ├── model_loader.py           # Load trained models with artifacts
│   ├── evaluate_models.py        # Benchmark evaluation
│   └── ensemble_fusion.py        # Score/decision fusion
│
├── 4_interpretability/           # SHAP analysis
│   └── run_shap_analysis.py      # SHAP for DL and TML models
│
├── 5_visualization/              # Visualization scripts
│   └── advanced_plots.py         # Heatmaps, calibration curves
│
├── configs/                      # Configuration files
│   ├── datasets.yaml             # Dataset paths and parameters
│   ├── models/                   # Model configurations
│   └── phase_mappings/           # Phase mapping rationales
│
├── data/                         # Data directories
│   ├── marker_genes/             # Reference marker genes
│   ├── raw/                      # Original data
│   ├── processed/                # Preprocessed data
│   └── predictions/              # Tool predictions
│
├── models/saved_models/          # Trained model weights (.pt, .joblib)
├── results/                      # Results outputs
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── environment.yml               # Conda environment
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

# Pipeline Steps

## STEP 0: Data Preprocessing

**Purpose**: Normalize scRNA-seq data and generate predictions from existing tools.

**Preprocessing Requirements:**
- Normalize gene expression using Seurat's `LogNormalize` or similar method
- Format: CSV with rows=cells, columns=genes
- Run prediction tools: Seurat CellCycleScore, Tricycle, Revelio, ccAFv2

**Output Format Required:**
```csv
CellID,Predicted
CELL_001,G1
CELL_002,S
CELL_003,G2M
```

**Note**: Revelio only supports human genes. For mouse data, use Seurat, Tricycle, and ccAFv2.

---

## STEP 1: Consensus Labeling

**Purpose**: Create consensus training labels by harmonizing predictions from multiple tools.

**Why Consensus?** No ground truth exists for most training data. Consensus from multiple established tools provides more reliable labels than any single method.

**Approach**: We use **Seurat CellCycleScore as the reference** and map other tools (Tricycle, ccAFv2, Revelio) to align with Seurat's predictions.

### Workflow Overview

```
1. ANALYZE: Compare OTHER tools vs SEURAT (reference)
   ↓
2. MANUAL INSPECTION: Inspect heatmap colors
   ↓
3. CREATE YAML: Map other tool phases to Seurat phases
   ↓
4. ASSIGN: Apply mappings to tool predictions
   ↓
5. MERGE: Create consensus where ≥N tools agree
```

### 1.1 Analyze Tool Agreements

**Master Script** (automated for mouse + human):
```bash
cd 1_consensus_labeling/analyze
bash run_analysis_mouse_human.sh
```

**What it does:**
- **Mouse:** Creates 2 heatmaps (Seurat vs Tricycle, Seurat vs ccAFv2)
- **Human:** Creates 3 heatmaps (Seurat vs Tricycle, Seurat vs ccAFv2, Seurat vs Revelio)
- **IMPORTANT:** Only compares OTHER tools against Seurat (reference), NOT all pairwise combinations

**For Custom Datasets** (manual):
```bash
# Step 1: Create contingency table (compare YOUR_TOOL vs Seurat)
python create_contingency_flexible.py \
  --tool1-file /path/to/seurat_predictions.csv \
  --tool1-name seurat \
  --tool2-file /path/to/tricycle_predictions.csv \
  --tool2-name tricycle \
  --dataset-name my_dataset \
  --output-dir results/

# Step 2: Generate heatmap
python generate_heatmap_flexible.py \
  --contingency-table results/contingency_tables/contingency_seurat_vs_tricycle_my_dataset.csv \
  --tool1-name seurat \
  --tool2-name tricycle \
  --dataset-name my_dataset \
  --output-dir results/heatmaps/
```

**Output**:
- Contingency tables: How often Seurat and other tool agree
- Heatmaps: Observed/expected ratios (GREEN = high alignment with Seurat, RED = low alignment)

### 1.2 Manual Inspection

Open the heatmap images and inspect color patterns to see which OTHER tool phases align with Seurat:
- **GREEN cells** → Strong alignment with Seurat → Map this phase to the corresponding Seurat phase
- **RED cells** → Weak alignment → These phases don't match well
- **YELLOW cells** → Random association

**Example Interpretation** (mapping Tricycle/ccAFv2/Revelio to Seurat):
- If Tricycle's "G1.S" has GREEN with Seurat's "S" → map `G1.S → S`
- If ccAFv2's "G2" has GREEN with Seurat's "G2M" → map `G2 → G2M`
- If Tricycle's "M.G1" has GREEN with Seurat's "G1" → map `M.G1 → G1`

**Goal**: Map all other tool phases to Seurat's 3 main phases (G1, S, G2M)

### 1.3 Create YAML Phase Mappings

Based on heatmap inspection, create a YAML config file:

**Example**: `configs/phase_mappings/my_dataset.yaml`

```yaml
# Phase mappings for my_dataset
# Based on heatmap analysis

dataset:
  name: "my_dataset"
  species: "human"
  tools: ["seurat", "tricycle", "ccafv2", "revelio"]

mappings:
  seurat:
    G1: G1
    S: S
    G2M: G2M

  tricycle:
    # Adjust based on YOUR heatmap inspection!
    G1: G1
    G1.S: S         # If heatmap shows G1.S aligns with S
    S: S
    G2: G2M         # If heatmap shows G2 aligns with G2M
    G2M: G2M
    M: G2M
    M.G1: G1
    NA: G1

  ccafv2:
    qG0: G1
    Early G1: G1
    Late G1: G1
    G1: G1
    G1.S: S
    S: S
    Late S: S
    G2: G2M
    G2M: G2M
    M: G2M

  revelio:
    G0: G1
    G1: G1
    G1.S: S
    S: S
    G2: G2M
    G2M: G2M
    M: G2M
    M.G1: G1

rationale:
  "Mappings determined by obs/expected ratio heatmaps."
```

### 1.4 Apply Phase Reassignment

**Coming soon**: Flexible reassignment script that reads your YAML config and applies mappings.

### 1.5 Merge Consensus Labels

```bash
python merge/merge_consensus.py \
  --input ./reassigned/ \
  --output ./consensus/ \
  --sample my_dataset \
  --dataset my_dataset
```

**Output**:
- `my_dataset_overlapped_at_least_two_regions.csv` (≥2 tools agree)
- `my_dataset_overlapped_at_least_three_regions.csv` (≥3 tools agree)
- `my_dataset_overlapped_all_four_regions.csv` (all 4 tools agree, if applicable)

**Recommendation**: Use ≥3 tools agreement for high-confidence labels.

**See `1_consensus_labeling/WORKFLOW_MOUSE_HUMAN.md` for detailed workflow!**

---

## STEP 2: Model Training

Train deep learning and traditional ML models on consensus-labeled data.

### 2.1 Deep Learning Models

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

# Train with CUSTOM training data (your own CSV)
python 2_model_training/train_deep_learning.py \
  --model simpledense \
  --data /path/to/your_training_data.csv \
  --output models/custom_model/ \
  --trials 50 \
  --cv 5
```

**Available Models**:
- `simpledense` - DNN3 (3-layer: input→128→64→3)
- `deepdense` - DNN5 (5-layer: input→256→128→64→3)
- `cnn` - 1D Convolutional (Conv(32)→Conv(64)→Dense)
- `hbdcnn` - Hybrid CNN+Dense
- `fe` - Feature Embedding

### 2.2 Traditional ML Models

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

# Train with CUSTOM training data (your own CSV)
python 2_model_training/train_traditional_ml.py \
  --model random_forest \
  --data /path/to/your_training_data.csv \
  --output models/custom_rf/ \
  --trials 50 \
  --cv 5
```

**Available Models**:
- `adaboost` - AdaBoost classifier
- `random_forest` - Random Forest
- `lgbm` - LightGBM
- `ensemble` - VotingClassifier (AdaBoost + RF + LGBM)

**Note**: When `--data` is provided, the `--dataset` argument (reh/sup) is ignored.

---

## STEP 3: Model Evaluation

Evaluate trained models on benchmark datasets with ground truth labels.

### 3.1 Evaluate on Standard Benchmarks

```bash
# Evaluate Deep Learning model on all benchmarks
python 3_evaluation/evaluate_models.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --benchmarks SUP GSE146773 GSE64016 Buettner_mESC \
  --output results/dnn3_all_benchmarks.csv

# Evaluate Traditional ML model on all benchmarks
python 3_evaluation/evaluate_models.py \
  --model_path models/saved_models/random_forest/rf_NFT_reh_fld_1.joblib \
  --benchmarks SUP GSE146773 GSE64016 Buettner_mESC \
  --output results/rf_all_benchmarks.csv

# Evaluate on specific benchmarks only
python 3_evaluation/evaluate_models.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --benchmarks GSE146773 \
  --output results/dnn3_gse146773_only.csv
```

### 3.2 Evaluate on Custom Benchmark

```bash
# Evaluate on CUSTOM benchmark data (your own CSV with ground truth)
python 3_evaluation/evaluate_models.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --custom_benchmark /path/to/your_benchmark.csv \
  --custom_benchmark_name "MyBenchmark" \
  --output results/dnn3_custom_benchmark.csv
```

**Custom Benchmark CSV Format**:
```csv
CellID,Predicted,gene1,gene2,gene3,...
CELL_001,G1,2.5,3.1,0.8,...
CELL_002,S,1.2,4.5,2.1,...
CELL_003,G2M,3.4,1.9,5.2,...
```

**Important**: First column = cell_id, second column = phase_label (G1/S/G2M), remaining = genes

### 3.3 Ensemble Methods

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

---

## STEP 4: SHAP Interpretability Analysis

Perform SHAP (SHapley Additive exPlanations) analysis to identify biologically important features.

**Works with both Deep Learning (.pt) and Traditional ML (.joblib) models!**

```bash
# SHAP analysis for DNN3 on GSE146773 benchmark (default)
python 4_interpretability/run_shap_analysis.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --benchmark GSE146773 \
  --output_dir results/shap/dnn3/

# SHAP analysis for Random Forest on GSE64016 benchmark
python 4_interpretability/run_shap_analysis.py \
  --model_path models/saved_models/random_forest/rf_NFT_reh_fld_1.joblib \
  --benchmark GSE64016 \
  --output_dir results/shap/random_forest/

# SHAP analysis on CUSTOM benchmark data
python 4_interpretability/run_shap_analysis.py \
  --model_path models/saved_models/dnn3/dnn3_NFT_reh_fld_1.pt \
  --custom_benchmark /path/to/your_benchmark.csv \
  --custom_benchmark_name "MyBenchmark" \
  --output_dir results/shap/custom/
```

**Available Benchmarks**:
- `GSE146773` (default)
- `GSE64016`
- `SUP`
- `Buettner_mESC`
- Custom benchmark via `--custom_benchmark`

**Outputs**:
- SHAP summary plots (PNG)
- Top features CSV (ranked by importance)
- SHAP values text file

**Technical Note**: Uses `KernelExplainer` for both DL and TML models. LGBM compatibility ensured via lambda wrapping.

---

## STEP 5: Visualization

Advanced visualization tools for publication-quality figures.

```python
from advanced_plots import plot_calibration_curve, plot_heatmap

# Calibration curves
plot_calibration_curve(y_true, y_pred_proba, output_path="results/calibration.png")

# Class-wise precision/recall heatmaps
plot_heatmap(confusion_matrix, output_path="results/confusion_heatmap.png")
```

---

## Using Custom Data

The pipeline supports training and evaluation with your own datasets while keeping REH/SUP as defaults.

### Custom Training Data

**CSV Format Required:**
```csv
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
  --trials 50 \
  --cv 5
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
| DNN3 (SimpleDense) | 3-layer dense network | input→128→64→3 | Top performer (75% accuracy on benchmarks) |
| DNN5 (DeepDense) | 5-layer dense network | input→256→128→64→3 | Deep architecture for complex patterns |
| CNN | 1D Convolutional | Conv(32)→Conv(64)→Dense | Captures local gene patterns |
| Hybrid | CNN + Dense | Conv→Dense layers | Feature extraction + classification |
| Feature Embedding | Learned embedding | Embedding→Dense | Dimensionality reduction |

### Traditional ML Models
- AdaBoost
- Random Forest
- LightGBM
- Ensemble (VotingClassifier: AdaBoost + RF + LGBM)

### Ensemble Models
- Top-3 Decision Fusion (majority voting)
- Top-3 Score Fusion (probability averaging)

---

## Configuration

All parameters are configured via YAML files in `configs/`:

### Dataset Configuration (`configs/datasets.yaml`)
- Training data paths (REH, SUP-B15)
- Benchmark data paths (GSE146773, GSE64016, Buettner_mESC)
- Preprocessing parameters
- Class labels

### Model Configuration (`configs/models/dnn3.yaml`)
- Model architecture
- Training hyperparameters
- Optimizer settings
- Loss function parameters

### Phase Mapping (`configs/phase_mappings/`)
- Sub-phase to main phase mappings
- Rationale for each mapping based on heatmap analysis
- Dataset-specific configurations

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
- Benchmark Datasets: GSE146773, GSE64016, Buettner mESC
