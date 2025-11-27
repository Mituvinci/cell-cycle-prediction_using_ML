#!/usr/bin/env python3
"""
Deep Learning Model Training Script
====================================

Simple CLI wrapper for perform_nested_cv_dn function.

Usage:
    python train_deep_learning.py --model cnn --dataset sup --output ./models/cnn_sup/

This is exactly how you called it in your original code:
    perform_nested_cv_dn(
        model_type=cnn,
        reh_or_sup=reh_or_sup,
        save_model_here=models_path,
        selection_method=None,
        scaling_method=standard,
        n_trials=n_trials,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        cv=cv
    )

Author: Halima Akhter
Date: 2025-11-24
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import THE MAIN TRAINING FUNCTION
from utils.nested_cv import perform_nested_cv_dn


def main():
    parser = argparse.ArgumentParser(
        description='Train deep learning models for cell cycle prediction using nested CV with Optuna',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train CNN on SUP data
  python train_deep_learning.py --model cnn --dataset sup --output ./models/CNN/ --trials 20 --cv 5

  # Train DNN3 (SimpleDenseModel) on REH data
  python train_deep_learning.py --model simpledense --dataset reh --output ./models/DNN3/

  # Train with feature selection
  python train_deep_learning.py --model deepdense --dataset sup --feature-selection ElasticCV --output ./models/DNN5/

Note:
  Training uses max_epochs=100 with early_stopping_patience=100 (for testing).
  Change epochs=100 to epochs=1500 in utils/optuna_utils.py for full training.
  Optuna optimizes learning_rate, optimizer, and architecture parameters.

Available models:
  - simpledense (DNN3)
  - deepdense (DNN4/DNN5)
  - cnn
  - hbdcnn (Hybrid CNN+Dense)
  - fe (Feature Embedding)
        """
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['simpledense', 'deepdense', 'cnn', 'hbdcnn', 'fe'],
        help='Model type to train'
    )

    # Data
    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='reh',
        choices=['reh', 'sup'],
        help='Dataset to use: reh or sup (default: reh). Ignored if --data is provided.'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to custom training data CSV. If provided, overrides --dataset. Format: cell_id, phase_label, gene1, gene2, ...'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for saving models and results'
    )

    # Feature selection
    parser.add_argument(
        '--feature-selection',
        type=str,
        default=None,
        choices=['ElasticCV', 'SelectKBest', None],
        help='Feature selection method (default: None)'
    )

    # Scaling
    parser.add_argument(
        '--scaling',
        type=str,
        default='standard',
        choices=['standard', 'minmax', 'robust'],
        help='Scaling method (default: standard)'
    )

    # Optuna parameters
    parser.add_argument(
        '--trials',
        type=int,
        default=20,
        help='Number of Optuna trials for hyperparameter optimization (default: 20)'
    )

    # Cross-validation
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("DEEP LEARNING MODEL TRAINING - NESTED CV WITH OPTUNA")
    print("=" * 80)
    print(f"Model: {args.model.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Output Directory: {args.output}")
    print(f"Feature Selection: {args.feature_selection if args.feature_selection else 'None'}")
    print(f"Scaling Method: {args.scaling}")
    print(f"Optuna Trials: {args.trials}")
    print(f"Max Epochs: 100 (testing mode - change to 1500 for full training)")
    print(f"Early Stopping Patience: 100")
    print(f"Cross-Validation Folds: {args.cv}")
    print("=" * 80)
    print()

    # Call THE MAIN TRAINING FUNCTION (your exact function!)
    perform_nested_cv_dn(
        model_type=args.model,
        reh_or_sup=args.dataset,
        save_model_here=args.output,
        selection_method=args.feature_selection,
        scaling_method=args.scaling,
        n_trials=args.trials,
        cv=args.cv,
        custom_data_path=args.data
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✅ All results saved to: {args.output}")


if __name__ == "__main__":
    main()
