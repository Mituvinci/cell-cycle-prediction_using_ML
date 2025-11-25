# 2. Model Training

This directory contains all code for training deep learning and traditional machine learning models on consensus-labeled training data (REH and SUP-B15).

## Overview

**Purpose**: Train 9 different models using nested cross-validation with hyperparameter optimization.

**Models**:
- **Deep Learning** (5): DNN3, DNN5, CNN, Hybrid CNN-Dense, Feature Embedding
- **Traditional ML** (4): AdaBoost, Random Forest, LightGBM, Ensemble (VotingClassifier)

**Training Strategy**:
- Nested cross-validation (5-fold outer, 5-fold inner)
- Optuna hyperparameter optimization (20-100 trials)
- SMOTE + RandomUnderSampler for class imbalance
- Focal loss for deep learning models
- GPU acceleration (CUDA)

## Directory Structure

```
2_model_training/
├── train_deep_learning.py          # CLI for DL training
├── train_traditional_ml.py         # CLI for traditional ML training
│
├── models/                          # Model architectures
│   ├── dense_models.py             # DNN3, DNN5
│   ├── cnn_models.py               # CNN
│   └── hybrid_models.py            # Hybrid, Feature Embedding
│
├── utils/                           # Training utilities
│   ├── data_utils.py               # Data loading, preprocessing
│   ├── training_utils.py           # Training loops, evaluation
│   ├── nested_cv.py                # Nested CV implementation
│   ├── optuna_dl.py                # Optuna for deep learning
│   ├── optuna_tml.py               # Optuna for traditional ML
│   └── io_utils.py                 # Save/load models & results
│
└── saved_models/                    # Trained model weights
    ├── reh/
    │   ├── simpledense/
    │   ├── deepdense/
    │   ├── cnn/
    │   ├── hybrid/
    │   └── feature_embedding/
    └── sup/
        └── ...
```

## Quick Start

### Training Deep Learning Models

```bash
# Activate conda environment
conda activate pytorch

# Train DNN3 on REH data
python train_deep_learning.py \
  --model simpledense \
  --dataset reh \
  --output saved_models/reh/simpledense \
  --config ../configs/models/dnn3.yaml

# Train on SUP-B15 data
python train_deep_learning.py \
  --model simpledense \
  --dataset sup \
  --output saved_models/sup/simpledense \
  --config ../configs/models/dnn3.yaml

# Train other DL models
python train_deep_learning.py --model deepdense --dataset reh --output saved_models/reh/deepdense
python train_deep_learning.py --model cnn --dataset reh --output saved_models/reh/cnn
python train_deep_learning.py --model hybrid --dataset reh --output saved_models/reh/hybrid
python train_deep_learning.py --model feature_embedding --dataset reh --output saved_models/reh/feature_embedding
```

### Training Traditional ML Models

```bash
# Train AdaBoost on REH data
python train_traditional_ml.py \
  --model adaboost \
  --dataset reh \
  --output saved_models/reh/adaboost

# Train other traditional ML models
python train_traditional_ml.py --model random_forest --dataset reh --output saved_models/reh/rf
python train_traditional_ml.py --model lgbm --dataset reh --output saved_models/reh/lgbm
python train_traditional_ml.py --model ensemble --dataset reh --output saved_models/reh/ensemble
```

### Using SLURM (HPC Cluster)

```bash
# Edit SLURM script with your parameters
nano ../scripts/train_model.slurm

# Submit job
sbatch ../scripts/train_model.slurm

# Monitor job
squeue -u $USER
tail -f slurm-<job_id>.out
```

## Model Architectures

### Deep Learning Models

**1. SimpleDenseModel (DNN3)**
- 3 hidden layers: 256 → 128 → 64
- BatchNorm + Dropout (0.3)
- Best performer in our experiments
- Fast training, good generalization

**2. DeepDenseModel (DNN5)**
- 5 hidden layers: 512 → 256 → 128 → 64 → 32
- BatchNorm + Dropout (0.4)
- More capacity, slower training

**3. CNNModel**
- 3 Conv1D layers: 64 → 128 → 256 filters
- MaxPooling + Dropout
- Captures local gene expression patterns
- Treats genes as sequential features

**4. HybridCNNDenseModel**
- CNN feature extractor + Dense classifier
- Combines pattern detection with classification
- Higher capacity, requires more data

**5. FeatureEmbeddingModel**
- Learnable gene embeddings
- 2 transformer-like attention layers
- Captures gene-gene interactions
- Most complex, best for large datasets

### Traditional ML Models

**1. AdaBoost**
- Boosting ensemble
- Good for imbalanced data
- Fast inference

**2. Random Forest**
- Bagging ensemble
- Robust to overfitting
- Feature importance available

**3. LightGBM**
- Gradient boosting
- Fast training
- Memory efficient

**4. Ensemble (VotingClassifier)**
- Soft voting of AdaBoost, RF, LGBM
- Combines strengths of all methods
- Best traditional ML performance

## Training Configuration

### Nested Cross-Validation

**Outer Loop** (5 folds):
- Split data into train/test (80%/20%)
- Evaluate final model performance
- Compute generalization metrics

**Inner Loop** (5 folds):
- Split training data for validation
- Optimize hyperparameters with Optuna
- Select best hyperparameters

**Why Nested CV?**
- Prevents hyperparameter overfitting
- Unbiased performance estimates
- Standard for small datasets (<10,000 cells)

### Hyperparameter Optimization

**Optuna Trials**: 20-100 depending on model complexity

**Optimized Hyperparameters (DNN3 example)**:
```python
{
    "learning_rate": [1e-5, 1e-2],
    "batch_size": [32, 64, 128, 256],
    "dropout": [0.2, 0.5],
    "hidden_dim1": [128, 512],
    "hidden_dim2": [64, 256],
    "hidden_dim3": [32, 128],
    "weight_decay": [1e-6, 1e-3]
}
```

### Class Imbalance Handling

**SMOTE (Synthetic Minority Oversampling)**:
- Generates synthetic samples for minority classes
- k=5 neighbors used for interpolation

**RandomUnderSampler**:
- Reduces majority class samples
- Balanced dataset for training

**Focal Loss** (Deep Learning):
- γ=2, α=1 (default)
- Focuses learning on hard examples
- Reduces impact of easy negatives

**Class Weights** (Traditional ML):
- Automatic class balancing
- Penalizes misclassification of minority classes

### Data Preprocessing

**Standard Pipeline**:
1. Load consensus labels from `../1_consensus_labeling/`
2. Load gene expression matrix (log-normalized)
3. Apply StandardScaler or RobustScaler
4. Split into train/test folds
5. Apply SMOTE + undersampling to training set only
6. Train model
7. Evaluate on held-out test set

**Scaling Methods**:
- **StandardScaler**: Mean=0, Std=1 (default)
- **RobustScaler**: Median-based, robust to outliers
- **MinMaxScaler**: [0,1] range

## Configuration Files

Models can be configured via YAML files in `../configs/models/`:

```yaml
# configs/models/dnn3.yaml
model:
  type: simpledense
  hidden_dim1: 256
  hidden_dim2: 128
  hidden_dim3: 64
  dropout: 0.3

training:
  learning_rate: 0.001
  batch_size: 128
  epochs: 100
  early_stopping_patience: 10
  focal_loss_gamma: 2
  focal_loss_alpha: 1

optimization:
  n_trials: 50
  n_jobs: 1
  outer_splits: 5
  inner_splits: 5
```

## Output Files

Each training run produces:

**Saved Models**:
```
saved_models/reh/simpledense/
├── simpledense_reh_fold0.pt        # Model weights
├── simpledense_reh_fold1.pt
├── ...
└── simpledense_reh_fold4.pt
```

**Model Files (.pt) Contain**:
```python
{
    "model_state_dict": ...,           # PyTorch state dict
    "hyperparameters": {...},          # Optuna best params
    "scaler": StandardScaler(...),     # Fitted scaler
    "label_encoder": LabelEncoder(...),# Fitted encoder
    "selected_features": [...],        # Feature names (genes)
    "training_history": {...}          # Loss curves
}
```

**Results CSV**:
```
results/simpledense_reh_nested_cv_results.csv
```
Columns: Fold, Accuracy, F1, Precision, Recall, ROC-AUC, Balanced Accuracy, MCC, Kappa

**Metrics CSV** (per fold):
```
results/simpledense_reh_fold0_metrics.csv
```
Contains class-wise precision, recall, F1-score.

## CLI Arguments

### train_deep_learning.py

```bash
python train_deep_learning.py \
  --model {simpledense|deepdense|cnn|hybrid|feature_embedding} \
  --dataset {reh|sup} \
  --output OUTPUT_DIR \
  [--config CONFIG_YAML] \
  [--n_trials N_TRIALS] \
  [--outer_splits OUTER_SPLITS] \
  [--inner_splits INNER_SPLITS] \
  [--scaling {standard|robust|minmax}] \
  [--feature_selection {kbest|elasticnet|none}] \
  [--use_smote] \
  [--seed SEED]
```

### train_traditional_ml.py

```bash
python train_traditional_ml.py \
  --model {adaboost|random_forest|lgbm|ensemble} \
  --dataset {reh|sup} \
  --output OUTPUT_DIR \
  [--n_trials N_TRIALS] \
  [--outer_splits OUTER_SPLITS] \
  [--scaling {standard|robust|minmax}] \
  [--feature_selection {kbest|elasticnet|none}] \
  [--seed SEED]
```

## Training Tips

### GPU Usage
- All DL models use CUDA if available
- Check GPU: `nvidia-smi`
- Monitor usage: `watch -n 1 nvidia-smi`

### Training Time Estimates (on single GPU)
- **DNN3**: ~30-60 min per fold
- **DNN5**: ~60-90 min per fold
- **CNN**: ~45-75 min per fold
- **Hybrid**: ~60-120 min per fold
- **Feature Embedding**: ~90-180 min per fold

### Memory Requirements
- **DNN3, DNN5**: ~4 GB GPU RAM
- **CNN, Hybrid**: ~6 GB GPU RAM
- **Feature Embedding**: ~8 GB GPU RAM

### Best Practices
1. Start with DNN3 (fastest, often best)
2. Use config files for reproducibility
3. Set random seed for reproducibility
4. Monitor training with early stopping
5. Save models after each fold (automatic)
6. Check GPU utilization (should be >80%)

## Troubleshooting

**Out of Memory Error**:
```bash
# Reduce batch size
python train_deep_learning.py --model simpledense --dataset reh --output out/ --config configs/models/dnn3_small_batch.yaml
```

**Slow Training**:
```bash
# Reduce number of trials
python train_deep_learning.py --model simpledense --dataset reh --output out/ --n_trials 20
```

**Feature Selection Issues**:
```bash
# Use full feature set (often best)
python train_deep_learning.py --model simpledense --dataset reh --output out/ --feature_selection none
```

**Class Imbalance Warning**:
```bash
# SMOTE is enabled by default for DL, automatic class weights for traditional ML
# No action needed
```

## Next Steps

After training models:
1. Move to `../3_evaluation/` to evaluate on benchmark data
2. Use `model_loader.py` to load trained models
3. Compare performance across models
4. Select top-3 models for ensemble
5. Run SHAP analysis in `../4_interpretability/`

## References

- Nested CV: Varma & Simon (2006) *BMC Bioinformatics*
- SMOTE: Chawla et al. (2002) *JAIR*
- Focal Loss: Lin et al. (2017) *ICCV*
- Optuna: Akiba et al. (2019) *KDD*
