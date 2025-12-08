#!/usr/bin/env python3
"""
Traditional ML Model Training Script
====================================

Simple CLI wrapper for perform_nested_cv_non_neural function.

Usage:
    python train_traditional_ml.py --model adaboost --dataset sup --output ./models/adaboost_sup/

This is exactly how you called it in your original code:
    perform_nested_cv_non_neural(
        model_type=adaboost,
        reh_or_sup=reh_or_sup,
        save_model_here=models_path,
        selection_method=None,
        scaling_method=standard,
        n_trials=n_trials,
        outer_splits=outer_splits
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
from utils.nested_cv import perform_nested_cv_non_neural


def main():
    parser = argparse.ArgumentParser(
        description='Train traditional ML models for cell cycle prediction using nested CV with Optuna',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train AdaBoost on SUP data
  python train_traditional_ml.py --model adaboost --dataset sup --output ./models/AdaBoost/ --trials 20 --cv 5

  # Train Random Forest on REH data
  python train_traditional_ml.py --model random_forest --dataset reh --output ./models/RandomForest/

  # Train LGBM with feature selection
  python train_traditional_ml.py --model lgbm --dataset sup --feature-selection SelectKBest --output ./models/LGBM/

  # Train Ensemble (AdaBoost + RF + LGBM)
  python train_traditional_ml.py --model ensemble --dataset sup --output ./models/Ensemble/

Available models:
  - adaboost
  - random_forest
  - lgbm
  - ensemble (VotingClassifier with AdaBoost, RF, LGBM)
        """
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['adaboost', 'random_forest', 'lgbm', 'ensemble'],
        help='Model type to train'
    )

    # Data
    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='hpsc',
        choices=['hpsc', 'pbmc', 'mouse_brain', 'reh', 'sup'],
        help='Dataset to use: hpsc, pbmc, mouse_brain, reh, sup (default: hpsc).'
    )

    parser.add_argument(
        '--gene-list',
        type=str,
        required=True,
        help='Path to gene list file (one gene per line, UPPERCASE). REQUIRED.'
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
        choices=['SelectKBest', 'ElasticCV', None],
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
    print("TRADITIONAL ML MODEL TRAINING - NESTED CV WITH OPTUNA")
    print("=" * 80)
    print(f"Model: {args.model.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Gene List: {args.gene_list if args.gene_list else 'None (will compute 7-dataset intersection)'}")
    print(f"Output Directory: {args.output}")
    print(f"Feature Selection: {args.feature_selection if args.feature_selection else 'None'}")
    print(f"Scaling Method: {args.scaling}")
    print(f"Optuna Trials: {args.trials}")
    print(f"Cross-Validation Folds: {args.cv}")
    print("=" * 80)
    print()

    # Call THE MAIN TRAINING FUNCTION (your exact function!)
    perform_nested_cv_non_neural(
        model_type=args.model,
        dataset=args.dataset,
        save_model_here=args.output,
        selection_method=args.feature_selection,
        scaling_method=args.scaling,
        n_trials=args.trials,
        outer_splits=args.cv,
        gene_list_path=args.gene_list
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"All results saved to: {args.output}")


if __name__ == "__main__":
    main()
