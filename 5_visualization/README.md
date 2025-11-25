# 5. Visualization

This directory contains scripts for creating publication-quality figures and plots for model comparison and results visualization.

## Overview

**Purpose**:
- Create figures for manuscript and presentations
- Visualize model performance comparisons
- Generate calibration curves and confidence analyses
- Produce heatmaps for multi-model comparison

**Output**: Publication-ready figures (high-resolution PNG/PDF).

## Directory Structure

```
5_visualization/
└── advanced_plots.py       # Calibration, heatmaps, ROC curves
```

## Plotting Functions

### 1. Calibration Curves

**Purpose**: Assess how well predicted probabilities match actual outcomes.

**Usage**:
```python
from advanced_plots import plot_calibration_curve
import pandas as pd
import numpy as np

# Load predictions
results = pd.read_csv("../3_evaluation/results/dnn3_gse146773_predictions.csv")
y_true = results["True_Label"]
y_pred_proba = results[["G1_Prob", "S_Prob", "G2M_Prob"]].values

# Plot calibration curve
plot_calibration_curve(
    y_true=y_true,
    y_pred_proba=y_pred_proba,
    label_encoder=label_encoder,
    n_bins=10,
    save_path="figures/dnn3_calibration.png"
)
```

**Output**: Calibration curve showing predicted vs. actual probabilities for each class.

**Interpretation**:
- **Diagonal line**: Perfect calibration
- **Above diagonal**: Over-confident (predicted probabilities too high)
- **Below diagonal**: Under-confident (predicted probabilities too low)

**Good Model**: Points close to diagonal across all probability ranges.

### 2. Multi-Model Heatmaps

**Purpose**: Compare precision/recall across models and benchmarks.

**Usage**:
```python
from advanced_plots import build_heatmap_data, plot_heatmap
import pandas as pd

# Load evaluation results from all models
all_results = pd.read_csv("../3_evaluation/results/all_models_evaluation.csv")

# Filter for GSE146773 benchmark
gse146773_results = all_results[all_results["Benchmark"] == "GSE146773"]

# Build heatmap data (precision)
heatmap_df = build_heatmap_data(gse146773_results, metric="Precision", dataset_prefix="GSE146773")

# Plot heatmap
plot_heatmap(
    df=heatmap_df,
    title="Precision Comparison (GSE146773)",
    save_path="figures/precision_heatmap_gse146773.png",
    cmap="YlGnBu",
    vmin=70,
    vmax=100
)
```

**Output**: Heatmap showing precision (or recall) for each model × class combination.

Example:
```
              G1    S    G2M
DNN3         89.3  85.2  86.8
DNN5         87.1  83.9  85.2
CNN          88.2  84.1  85.9
Feature Emb  88.9  86.5  87.1
Hybrid       86.5  82.8  84.3
Top3 Score   90.1  87.2  88.3  ← Best
ccAFv2       83.5  79.8  81.2
Seurat       78.2  74.5  76.8
```

**Interpretation**: Darker colors = better performance.

### 3. ROC Curves

**Purpose**: Visualize true positive rate vs. false positive rate.

**Usage**:
```python
from advanced_plots import plot_roc_curves
import pandas as pd

# Load predictions from multiple models
models = ["dnn3", "feature_embedding", "cnn"]
predictions = {}

for model in models:
    results = pd.read_csv(f"../3_evaluation/results/{model}_gse146773_predictions.csv")
    predictions[model] = {
        "y_true": results["True_Label"],
        "y_pred_proba": results[["G1_Prob", "S_Prob", "G2M_Prob"]].values
    }

# Plot ROC curves (one-vs-rest for multiclass)
plot_roc_curves(
    predictions=predictions,
    label_encoder=label_encoder,
    save_path="figures/roc_curves_comparison.png"
)
```

**Output**: ROC curves for each class, overlaid for model comparison.

**Interpretation**:
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random guessing
- **Higher AUC**: Better discrimination

### 4. Confusion Matrices (Grid)

**Purpose**: Visualize prediction errors for multiple models.

**Usage**:
```python
from advanced_plots import plot_confusion_matrix_grid
import pandas as pd

# Load confusion matrices from all models
models = ["dnn3", "dnn5", "cnn", "feature_embedding", "hybrid"]
cms = []

for model in models:
    cm = pd.read_csv(f"../3_evaluation/results/{model}_gse146773_cm.csv", index_col=0)
    cms.append((model, cm.values))

# Plot grid
plot_confusion_matrix_grid(
    confusion_matrices=cms,
    class_names=["G1", "S", "G2M"],
    save_path="figures/confusion_matrices_grid.png",
    figsize=(15, 10)
)
```

**Output**: 5×1 grid of confusion matrices for easy comparison.

**Interpretation**: Diagonal = correct predictions, off-diagonal = errors.

### 5. Performance Bar Charts

**Purpose**: Compare aggregate metrics across models and benchmarks.

**Usage**:
```python
from advanced_plots import plot_performance_bars
import pandas as pd

# Load evaluation results
results = pd.read_csv("../3_evaluation/results/all_models_evaluation.csv")

# Plot F1 scores for all models on GSE146773
gse146773_results = results[results["Benchmark"] == "GSE146773"]

plot_performance_bars(
    df=gse146773_results,
    metric="F1",
    title="F1 Score Comparison (GSE146773)",
    save_path="figures/f1_comparison_gse146773.png",
    color_map={"DNN3": "#1f77b4", "Feature Embedding": "#ff7f0e", "CNN": "#2ca02c"}
)
```

**Output**: Bar chart with error bars (if multiple folds).

### 6. Benchmark-Wise Performance Comparison

**Purpose**: Show how models perform across all 3 benchmarks.

**Usage**:
```python
from advanced_plots import plot_benchmark_comparison
import pandas as pd

# Load results
results = pd.read_csv("../3_evaluation/results/all_models_evaluation.csv")

# Select top-3 models
top3_models = ["SimpleDenseModel", "FeatureEmbeddingModel", "CNNModel"]
top3_results = results[results["Model"].isin(top3_models)]

# Plot
plot_benchmark_comparison(
    df=top3_results,
    metric="F1",
    save_path="figures/benchmark_comparison.png"
)
```

**Output**: Grouped bar chart showing model performance on SUP, GSE146773, GSE64016.

**Interpretation**: Consistent performance across benchmarks = good generalization.

## Complete Visualization Workflow

### Step 1: Model Performance Heatmaps

```bash
python advanced_plots.py \
  --plot heatmap \
  --results ../3_evaluation/results/all_models_evaluation.csv \
  --benchmark gse146773 \
  --metric Precision \
  --output figures/precision_heatmap_gse146773.png

python advanced_plots.py \
  --plot heatmap \
  --results ../3_evaluation/results/all_models_evaluation.csv \
  --benchmark gse146773 \
  --metric Recall \
  --output figures/recall_heatmap_gse146773.png
```

### Step 2: Calibration Curves (Top-3 Models)

```bash
python advanced_plots.py \
  --plot calibration \
  --predictions ../3_evaluation/results/dnn3_gse146773_predictions.csv \
  --output figures/dnn3_calibration.png

python advanced_plots.py \
  --plot calibration \
  --predictions ../3_evaluation/results/feature_embedding_gse146773_predictions.csv \
  --output figures/fe_calibration.png

python advanced_plots.py \
  --plot calibration \
  --predictions ../3_evaluation/results/cnn_gse146773_predictions.csv \
  --output figures/cnn_calibration.png
```

### Step 3: ROC Curves Comparison

```bash
python advanced_plots.py \
  --plot roc \
  --predictions \
    ../3_evaluation/results/dnn3_gse146773_predictions.csv \
    ../3_evaluation/results/feature_embedding_gse146773_predictions.csv \
    ../3_evaluation/results/cnn_gse146773_predictions.csv \
  --labels "DNN3" "Feature Embedding" "CNN" \
  --output figures/roc_curves_comparison.png
```

### Step 4: Confusion Matrix Grid

```bash
python advanced_plots.py \
  --plot cm_grid \
  --confusion_matrices \
    ../3_evaluation/results/dnn3_gse146773_cm.csv \
    ../3_evaluation/results/dnn5_gse146773_cm.csv \
    ../3_evaluation/results/cnn_gse146773_cm.csv \
    ../3_evaluation/results/feature_embedding_gse146773_cm.csv \
    ../3_evaluation/results/hybrid_gse146773_cm.csv \
  --labels "DNN3" "DNN5" "CNN" "Feature Embedding" "Hybrid" \
  --output figures/confusion_matrices_grid.png
```

### Step 5: Benchmark Comparison

```bash
python advanced_plots.py \
  --plot benchmark \
  --results ../3_evaluation/results/all_models_evaluation.csv \
  --models "SimpleDenseModel" "FeatureEmbeddingModel" "CNNModel" \
  --metric F1 \
  --output figures/benchmark_comparison.png
```

### Step 6: Model vs. Tools Comparison

```bash
python advanced_plots.py \
  --plot comparison \
  --model_results ../3_evaluation/results/top3_score_gse146773.csv \
  --tool_results ../3_evaluation/results/tools_gse146773.csv \
  --output figures/model_vs_tools.png
```

## Figure Specifications for Publication

### Resolution
- **PNG**: 300 DPI minimum
- **PDF**: Vector format (preferred for journals)

### Size
- **Full page**: 7 inches wide
- **Half page**: 3.5 inches wide
- **Height**: As needed, typically 4-6 inches

### Font
- **Helvetica** or **Arial** (sans-serif)
- Minimum 8pt for labels
- 10-12pt for axis labels
- 12-14pt for titles

### Color Palettes

**Sequential** (for heatmaps):
- `YlGnBu`: Yellow → Green → Blue
- `RdYlGn`: Red → Yellow → Green (reversed for error rates)

**Categorical** (for model comparison):
- DNN3: `#1f77b4` (blue)
- Feature Embedding: `#ff7f0e` (orange)
- CNN: `#2ca02c` (green)
- DNN5: `#d62728` (red)
- Hybrid: `#9467bd` (purple)

**Tools** (for comparison):
- Our models: Blue shades
- Existing tools: Gray shades

### File Naming Convention

```
figures/
├── fig1_model_architecture.png          # Main paper figures
├── fig2_performance_comparison.png
├── fig3_shap_summary.png
├── fig4_benchmark_heatmaps.png
├── suppfig1_calibration_curves.png      # Supplementary figures
├── suppfig2_confusion_matrices.png
├── suppfig3_roc_curves.png
└── suppfig4_gene_enrichment.png
```

## Main Paper Figures (Suggested)

**Figure 1**: Study Design & Workflow
- Panel A: Consensus labeling workflow
- Panel B: Model architectures (DNN3, Feature Embedding, CNN)
- Panel C: Training/evaluation pipeline

**Figure 2**: Model Performance Comparison
- Panel A: Heatmap (precision/recall) for all models on GSE146773
- Panel B: Bar chart comparing F1 scores
- Panel C: Benchmark-wise performance (SUP, GSE146773, GSE64016)

**Figure 3**: Interpretability (SHAP)
- Panel A: SHAP summary plot (top-20 genes)
- Panel B: SHAP dependence plots (CDK1, PCNA, MCM2)
- Panel C: Gene enrichment for top-50 genes per phase

**Figure 4**: Comparison with Existing Tools
- Panel A: Performance metrics (our models vs. 4 tools)
- Panel B: Confusion matrices (best model vs. best tool)
- Panel C: Calibration curves (best model vs. best tool)

## Supplementary Figures (Suggested)

**Supp Figure 1**: Consensus Labeling
- Panel A: Contingency tables (all tool pairs)
- Panel B: Obs/expect heatmaps
- Panel C: Phase mapping flowchart

**Supp Figure 2**: Model Architectures (Detailed)
- All 5 DL architectures with layer details

**Supp Figure 3**: Training Curves
- Loss curves for all models (all folds)

**Supp Figure 4**: Extended Performance
- ROC curves (all models, all benchmarks)
- Precision-Recall curves
- Class-wise metrics tables

**Supp Figure 5**: SHAP Extended
- SHAP dependence plots for top-50 genes
- SHAP bar plots for all phases
- Model comparison (consensus genes)

## CLI Usage

```bash
# General format
python advanced_plots.py \
  --plot {heatmap|calibration|roc|cm_grid|benchmark|comparison} \
  --results RESULTS_CSV \
  [--predictions PREDICTIONS_CSV] \
  [--output OUTPUT_PATH] \
  [--metric {Precision|Recall|F1|Accuracy}] \
  [--benchmark {sup|gse146773|gse64016}] \
  [--dpi 300] \
  [--format {png|pdf}]
```

**Arguments**:
- `--plot`: Type of plot to generate
- `--results`: Path to evaluation results CSV
- `--predictions`: Path to predictions CSV (for calibration/ROC)
- `--output`: Output file path
- `--metric`: Performance metric to visualize
- `--benchmark`: Which benchmark dataset
- `--dpi`: Resolution (default: 300)
- `--format`: Output format (png or pdf)

## Troubleshooting

**Figures Too Small**:
```python
# Increase figure size in advanced_plots.py
plt.figure(figsize=(12, 8))  # Default: (8, 6)
```

**Text Overlapping**:
```python
# Adjust layout
plt.tight_layout()
# Or increase figure size
```

**Colors Not Colorblind-Friendly**:
```python
# Use colorblind-safe palettes
import seaborn as sns
sns.set_palette("colorblind")
```

**Low Resolution**:
```bash
# Increase DPI
python advanced_plots.py ... --dpi 600

# Or save as PDF (vector format)
python advanced_plots.py ... --format pdf
```

**Missing Data in Heatmap**:
```python
# Check that all models have results for all metrics
# May need to impute missing values or exclude models
```

## References

- **Calibration**: Niculescu-Mizil & Caruana (2005) *ICML*
- **Confusion Matrix**: Stehman (1997) *Remote Sens Environ*
- **ROC Curves**: Fawcett (2006) *Pattern Recogn Lett*
- **Visualization Best Practices**: Tufte (1983) *The Visual Display of Quantitative Information*

## Next Steps

After creating figures:
1. Review all figures for clarity and accuracy
2. Get feedback from co-authors
3. Incorporate into manuscript
4. Prepare figure legends
5. Upload manuscript + code to GitHub for reviewers
