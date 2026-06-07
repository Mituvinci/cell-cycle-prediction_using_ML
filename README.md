# Cell Cycle Phase Prediction Pipeline
## Deep Learning and Traditional ML Models for Single-Cell RNA-Seq Data

**Status:** Complete
**Last Updated:** 2026-02-23

---

## Overview

This repository provides a complete, reproducible pipeline for predicting cell cycle phases (G1, S, G2M) from single-cell RNA-seq (scRNA-seq) data using deep learning (DL) and traditional machine learning (TML) models.

The pipeline covers five stages: data preprocessing, consensus phase labeling, model training, benchmark evaluation, and publication visualization. All training was performed on an HPC cluster (DOLLY SODS, 155 GPUs) using SLURM job scheduling.

### Key Features

- Consensus labeling from 4 existing tools (Seurat, Tricycle, Revelio, ccAFv2)
- 6 deep learning architectures + 4 traditional ML models
- Nested cross-validation with Optuna hyperparameter optimization
- SMOTE + undersampling for class imbalance handling
- Focal loss for deep learning models
- Ensemble fusion (decision and score-level) from top-3 models
- SHAP interpretability analysis
- Cross-species evaluation (human trained, mouse evaluated and vice versa)
- Automatic gene name format conversion across species

---

## Datasets

### Training Datasets

| Dataset | Species | Platform | Cells (after labeling) | Source |
|---------|---------|----------|------------------------|--------|
| REH leukemia cell line | Human | 10x Chromium Multiome | ~3,000 | [GSE293316](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293316) |
| SUP-B15 leukemia cell line | Human | 10x Chromium Multiome | ~3,000 | [GSE293316](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293316) |
| Human PBMCs | Human | 10x Chromium | 2,431 (1,963 G1 / 127 S / 341 G2M) | [10x Genomics PBMC](https://www.10xgenomics.com/datasets/pbm-cs-from-a-healthy-donor-whole-transcriptome-analysis-3-1-standard-4-0-0) |
| Mouse E18 Brain Cells | Mouse | 10x Chromium v3 | 5,524 (3,830 G1 / 1,116 S / 578 G2M) | [10x Genomics Mouse Brain](https://www.10xgenomics.com/datasets/10-k-brain-cells-from-an-e-18-mouse-v-3-chemistry-3-standard-3-0-0) |
| GSE75748 (hPSC/hESC) | Human | Smart-seq2 | 1,776 | [GSE75748](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748) |
| Nestorova (mouse HSC) | Mouse | Smart-seq2 | - | [GSE81682](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81682) |

### Benchmark Datasets (Ground Truth FUCCI or FACS labels)

| Dataset | Species | Platform | Cells | Source |
|---------|---------|----------|-------|--------|
| GSE146773 | Human | Smart-seq2 | ~1,100 | [GSE146773](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146773) |
| GSE64016 | Human | Fluidigm C1 | 247 | [GSE64016](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64016) |
| Buettner_mESC (E-MTAB-2805) | Mouse | SMARTer | 288 | [ArrayExpress E-MTAB-2805](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-2805/) |
| SUP-B15 (internal) | Human | 10x Chromium Multiome | - | [GSE293316](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293316) |

**Buettner_mESC access via Bioconductor:**
```R
library(scRNAseq)
sce <- BuettnerESCData()
```

**Citation for Buettner_mESC:** Buettner, F., Natarajan, K. N., Casale, F. P., et al. Computational analysis of cell-to-cell heterogeneity in single-cell RNA-sequencing data reveals hidden subpopulations of cells. Nature Biotechnology 33, 155-160 (2015).

---

## Gene Name Format Convention

All new models (trained on the 7-dataset intersection of 6,066 genes) use **capitalized gene names** (first letter uppercase, rest lowercase).

| Context | Format | Example |
|---------|--------|---------|
| Training data (new models) | Capitalized | `Actb`, `Gapdh`, `Tp53` |
| Benchmark data (human: GSE146773, GSE64016, SUP) | UPPERCASE | `ACTB`, `GAPDH`, `TP53` |
| Benchmark data (mouse: Buettner_mESC) | Capitalized | `Actb`, `Gapdh`, `Tp53` |

**Automatic conversion:** The evaluation scripts (`3_evaluation/evaluate_models.py`) automatically detect the benchmark gene format and convert to match the training format via `match_scaler_feature_format()`. No manual conversion is needed.

---

## Installation

This project uses a Conda environment with PyTorch GPU support. CUDA is required for training.

```bash
# Clone the repository
git clone <repository_url>
cd cell_cycle_prediction

# Activate the existing pytorch conda environment
conda activate pytorch

# Alternatively, install from requirements file
pip install -r requirements.txt
```

**Requirements:** PyTorch (GPU), scikit-learn, LightGBM, Optuna, imbalanced-learn, SHAP, pandas, numpy, matplotlib, seaborn

---

## Pipeline

The pipeline has 5 steps. Steps 0-1 were completed during project development and are documented here for reproducibility. A reviewer can reproduce Steps 2-5 directly.

---

### Step 0: Data Preprocessing

**Purpose:** Normalize raw scRNA-seq data and generate phase predictions from 4 existing tools.

**Note:** This step was run once during data preparation. The scripts are provided for reproducibility with new datasets.

Run all 4 tools on a dataset:
```bash
cd 0_preprocessing

# Seurat CellCycleScoring (R)
Rscript UNIVERSAL_SEURAT.R \
  --input /path/to/expression_matrix.csv \
  --species human \
  --output /path/to/seurat_predictions.csv

# Tricycle (R)
Rscript UNIVERSAL_TRICYCLE.R \
  --input /path/to/expression_matrix.csv \
  --species human \
  --output /path/to/tricycle_predictions.csv

# Revelio (R, human and mouse via ortholog gene conversion to uppercase)
Rscript UNIVERSAL_REVELIO.R \
  --input /path/to/expression_matrix.csv \
  --output /path/to/revelio_predictions.csv

# ccAFv2 (Python)
python UNIVERSAL_ccAFv2.py \
  --input /path/to/expression_matrix.csv \
  --output /path/to/ccafv2_predictions.csv
```

See `0_preprocessing/UNIVERSAL_TOOLS_README.md` for detailed usage and supported input formats (10X MTX, CSV, TXT).

**Important note for GSE64016:** This benchmark was not log-normalized in the original download (values up to 129,617 vs training data max ~6). Fix with:
```bash
bash 0_preprocessing/renormalize_GSE64016.sh /path/to/GSE64016.csv
```

---

### Step 1: Consensus Labeling

**Purpose:** Create high-confidence training labels by merging predictions from multiple tools. No ground truth exists for training datasets, so consensus from multiple tools provides reliable labels.

**Note:** This step was completed for all training datasets. The methodology and outputs are in `1_consensus_labeling/results/`.

**Workflow (5 steps):**

```
1. ANALYZE   - Generate contingency tables and obs/expected ratio heatmaps
2. INSPECT   - Manually inspect heatmap colors to guide phase mapping
3. MAP       - Create YAML mapping files (sub-phases -> G1/S/G2M)
4. ASSIGN    - Apply mappings to reassign each tool's predictions
5. MERGE     - Take cells where >=3 tools agree as consensus labels
```

**Step 1 - Analyze tool agreement:**
```bash
cd 1_consensus_labeling/analyze

# Automated for mouse and human datasets
bash run_analysis_mouse_human.sh

# Or manually for a custom dataset
python create_contingency_flexible.py \
  --tool1-file seurat_predictions.csv --tool1-name seurat \
  --tool2-file tricycle_predictions.csv --tool2-name tricycle \
  --dataset-name my_dataset --output-dir results/

python generate_heatmap_flexible.py \
  --contingency-table results/contingency_tables/contingency_seurat_vs_tricycle_my_dataset.csv \
  --tool1-name seurat --tool2-name tricycle \
  --dataset-name my_dataset --output-dir results/heatmaps/
```

**Step 2 - Manual heatmap inspection:**
- Green cells = strong agreement between tool and Seurat (reference) -> map to that Seurat phase
- Red cells = weak agreement
- Map each tool's sub-phases (G1.S, G2, M.G1, etc.) to one of G1, S, G2M

**Step 5 - Merge consensus:**
```bash
python 1_consensus_labeling/merge/merge_consensus.py \
  --input ./reassigned/ \
  --output ./consensus/ \
  --sample my_dataset \
  --dataset my_dataset
```

Outputs: cells where >=2, >=3, or all 4 tools agree (use >=3 for high-confidence labels).

See `1_consensus_labeling/WORKFLOW_MOUSE_HUMAN.md` for the complete methodology.

---

### Step 2: Model Training

**Purpose:** Train DL and TML models using nested cross-validation with Optuna hyperparameter optimization.

**On HPC (recommended):** Use the generic SLURM scripts in `I_actually_run_to_train/`:
```bash
# Train a DL model via SLURM
bash I_actually_run_to_train/train_dl_generic.sh --model simpledense --dataset reh

# Train a TML model via SLURM
bash I_actually_run_to_train/train_tml_generic.sh --model random_forest --dataset reh
```

**Direct CLI (for custom datasets):**
```bash
# Train SimpleDense (DNN3) on REH data
python 2_model_training/train_deep_learning.py \
  --model simpledense \
  --dataset reh \
  --output models/reh/simpledense/ \
  --trials 50 \
  --cv 5

# Train EnhanceDense (DNN5) on PBMC data
python 2_model_training/train_deep_learning.py \
  --model enhancedense \
  --dataset pbmc \
  --output models/pbmc/enhancedense/ \
  --trials 50 \
  --cv 5

# Train Random Forest on mouse brain data
python 2_model_training/train_traditional_ml.py \
  --model random_forest \
  --dataset mouse_brain \
  --output models/mouse_brain/rf/ \
  --trials 50 \
  --cv 5

# Train with your own CSV (--data overrides --dataset)
python 2_model_training/train_deep_learning.py \
  --model simpledense \
  --data /path/to/your_training_data.csv \
  --output models/custom/ \
  --trials 50 \
  --cv 5
```

**Training data CSV format:**
```
cell_id,phase_label,gene1,gene2,gene3,...
CELL_001,G1,2.5,3.1,0.8,...
CELL_002,S,1.2,4.5,2.1,...
CELL_003,G2M,3.4,1.9,5.2,...
```
- First column: cell ID
- Second column: phase label (G1, S, or G2M)
- Remaining columns: log-normalized gene expression values

**Available DL models:**

| Argument | Model | Architecture |
|----------|-------|--------------|
| `simpledense` | DNN3 (SimpleDense) | Dense 128->64->3 |
| `deepdense` | DNN4 (DeepDense) | Dense 256->128->64->3 |
| `enhancedense` | DNN5 (EnhanceDense) | Dense 512->256->128->64->3 |
| `cnn` | CNN | 1D Conv(32)->Conv(64)->Dense |
| `hybrid` | Hybrid CNN-Dense | Conv + Dense layers |
| `fe` | Feature Embedding | Embedding->Dense |

**Available TML models:**

| Argument | Model |
|----------|-------|
| `adaboost` | AdaBoost |
| `random_forest` | Random Forest |
| `lgbm` | LightGBM |
| `ensemble_embedding3tml` | Ensemble (AdaBoost + RF + LGBM) |

**Training features:**
- 5-fold nested cross-validation (5-fold outer, 5-fold inner for hyperparameter search)
- Optuna hyperparameter optimization (50-100 trials per model)
- SMOTE + random undersampling for class imbalance
- Focal loss for DL models
- Outputs: model weights (`.pt` for DL, `.pkl`/`.joblib` for TML), scalers, metrics CSVs

---

### Step 3: Benchmark Evaluation

**Purpose:** Evaluate trained models on benchmark datasets with ground truth labels.

```bash
# Evaluate a DL model on all standard benchmarks
python 3_evaluation/evaluate_models.py \
  --model_path models/reh/simpledense/simpledense_NFT_reh_fld_1.pt \
  --output results/simpledense_reh_all_benchmarks.csv

# Evaluate a TML model
python 3_evaluation/evaluate_models.py \
  --model_path models/reh/rf/rf_NFT_reh_fld_1.joblib \
  --output results/rf_reh_all_benchmarks.csv

# Evaluate on specific benchmarks
python 3_evaluation/evaluate_models.py \
  --model_path models/reh/simpledense/simpledense_NFT_reh_fld_1.pt \
  --benchmarks GSE146773 GSE64016 \
  --output results/simpledense_two_benchmarks.csv

# Evaluate on your own custom benchmark
python 3_evaluation/evaluate_models.py \
  --model_path models/reh/simpledense/simpledense_NFT_reh_fld_1.pt \
  --custom_benchmark /path/to/your_benchmark.csv \
  --custom_benchmark_name MyDataset \
  --output results/simpledense_custom.csv
```

**Standard benchmarks:** `GSE146773`, `GSE64016`, `Buettner_mESC`, `SUP`

**Custom benchmark CSV format** (same as training format):
```
cell_id,phase_label,gene1,gene2,...
CELL_001,G1,2.5,3.1,...
```

**Output metrics per benchmark row:**
- Overall: `accuracy`, `f1`, `precision`, `recall`, `roc_auc`, `balanced_acc`, `mcc`, `kappa`
- Per-class (G1, S, G2M): `precision_g1`, `recall_g1`, `f1_g1`, `mcc_g1`, etc.

**Ensemble fusion (top-3 models):**

Decision fusion and score fusion are implemented in `3_evaluation/ensemble_fusion.py`. The top-3 models are selected once (based on cross-validation performance) and applied consistently across all benchmarks.

---

### Step 4: SHAP Interpretability

**Purpose:** Identify biologically important genes driving model predictions.

**Note:** SHAP analysis was completed on the Buettner_mESC benchmark. Results are in `4_interpretability/SHAP_results/`.

To run SHAP on a new model or benchmark (submit via SLURM):
```bash
sbatch 4_interpretability/run_4_shap_analyses.slurm
```

Or modify and run the shell script:
```bash
bash 4_interpretability/run_4_shap_analyses.sh
```

**SHAP outputs per model/benchmark:**
- `*_shap_summary.png` - Bar plot of mean absolute SHAP values (feature importance ranking)
- `*_top_features.csv` - Top genes ranked by SHAP importance
- `*_SHAP.txt` - Text summary

---

### Step 5: Visualization

**Purpose:** Generate all publication-quality figures.

Run scripts in order:
```bash
# Figure: marker gene expression across phases (Table 1 equivalent)
python 5_visualization/1_plot_table_1.py

# Generate benchmark result CSVs and heatmaps from trained models
python 5_visualization/3_generate_all_benchmark_results.py

# Figure: 2x2 benchmark performance line plots (main manuscript figure)
python 5_visualization/4_plot_benchmark_results.py

# Figure: model vs existing tools comparison bar plot
python 5_visualization/5_plot_tool_comparison_barplot.py
```

**Output figures** (in `5_visualization/heatmap_barplot_lineplots_csv/`):
- Precision/recall heatmaps per benchmark (PDF, editable)
- 2x2 combined benchmark panel (PDF, PNG, EPS)
- Tool comparison bar plot (PDF, PNG)

**Figure settings:** Times New Roman, 600 DPI, PDF with TrueType fonts (editable in Adobe Illustrator).

---

## Models

### Deep Learning

| Model Name | Alias | Architecture | Notes |
|------------|-------|--------------|-------|
| SimpleDense | DNN3 | Dense 128->64->3 | Strong cross-dataset performer |
| DeepDense | DNN4 | Dense 256->128->64->3 | |
| EnhanceDense | DNN5 | Dense 512->256->128->64->3 | Top DL model on REH training |
| CNN | CNN | 1D Conv(32)->Conv(64)->Dense | |
| Hybrid CNN-Dense | Hybrid | Conv + Dense | |
| Feature Embedding | FE | Embedding->Dense | |

### Traditional ML

| Model | Notes |
|-------|-------|
| AdaBoost | |
| Random Forest | |
| LightGBM | |
| Ensemble (AdaBoost + RF + LGBM) | Voting classifier |

### Ensemble Fusion (Top-3 DL)

- **Score Fusion:** Average predicted probabilities from top-3 DL models
- **Decision Fusion:** Majority vote from top-3 DL model predictions
- Top-3 models are selected once from cross-validation results and applied to all benchmarks

---

## Reproducibility Notes

1. **Consensus labeling** requires manual heatmap inspection (Step 1, Step 2). The manual phase mappings used in this study are recorded in `1_consensus_labeling/assign/` YAML files.

2. **Gene intersection:** All models are trained on the intersection of genes present in all 7 datasets and all benchmarks (6,066 genes). The pre-computed gene list is in `gene_lists/`.

3. **Gene name format:** Benchmark gene names (UPPERCASE) are auto-converted to capitalized format at evaluation time. No manual preprocessing needed.

4. **GSE64016 normalization:** The raw download is not log-normalized. Use `0_preprocessing/renormalize_GSE64016.sh` before evaluation.

5. **Class imbalance:** REH and SUP-B15 have weak cell cycle activity (~80% G1 cells). Models handle this via SMOTE + undersampling + focal loss, but cross-dataset generalization on highly proliferative datasets is the main evaluation target.

---

## Citation

If you use this code or pipeline, please cite:

```bibtex
@article{akhter2026cellcycle, author = {Halima Akhter and Debra Piktel and Laura F. Gibson and Gangqing Hu and Donald A. Adjeroh}, title = {Deep Learning Models for Cell Cycle Phase Prediction from Single-Cell RNA Sequencing Data}, journal = {Briefings in Bioinformatics}, year = {2026}, doi = {10.1093/bib/bbag342}, url = {https://doi.org/10.1093/bib/bbag342} }
```

---

## Acknowledgments

- Existing tools used for consensus labeling: Seurat, Tricycle, Revelio, ccAFv2
- Benchmark datasets: GSE146773, GSE64016, Buettner mESC (E-MTAB-2805)
- Training datasets: GSE293316 (REH, SUP-B15), GSE75748 (hPSC), GSE81682 (Nestorova mouse HSC), 10x Genomics PBMC, 10x Genomics Mouse Brain
