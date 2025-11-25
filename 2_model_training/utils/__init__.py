"""
Utilities for Cell Cycle Prediction Deep Learning Models
========================================================

This module contains all utility functions for data processing, training,
evaluation, optimization, and I/O operations.

Author: Halima Akhter
Date: 2025-11-24
"""

# Data utilities
from .data_utils import (
    apply_scaling,
    elasticnet_feature_selection,
    preprocess_rna_data,
    load_and_preprocess_data
)

# Training utilities
from .training_utils import (
    focal_loss,
    train_model,
    evaluate_model,
    initialize_optimizer,
    init_weights
)

# Optuna optimization
from .optuna_utils import optimize_model_with_optuna

# Nested cross-validation (MAIN TRAINING FUNCTION)
from .nested_cv import perform_nested_cv_dn

# Visualization
from .visualization import (
    plot_training_history,
    plot_validation_accuracy
)

# I/O utilities
from .io_utils import (
    save_scaler,
    load_scaler,
    save_fold_to_csv
)

__all__ = [
    # Data
    'apply_scaling',
    'elasticnet_feature_selection',
    'preprocess_rna_data',
    'load_and_preprocess_data',
    # Training
    'focal_loss',
    'train_model',
    'evaluate_model',
    'initialize_optimizer',
    'init_weights',
    # Optimization
    'optimize_model_with_optuna',
    # Main training function
    'perform_nested_cv_dn',
    # Visualization
    'plot_training_history',
    'plot_validation_accuracy',
    # I/O
    'save_scaler',
    'load_scaler',
    'save_fold_to_csv',
]
