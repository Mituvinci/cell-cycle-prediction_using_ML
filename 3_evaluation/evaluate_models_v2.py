"""
Model Evaluation Script
========================

Evaluate trained models on benchmark datasets with ground truth labels.

Usage:
    python evaluate_models.py --model_path /path/to/model.pt --output results/evaluation.csv

Author: Halima Akhter
Date: 2025-11-26
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../2_model_training'))

from utils.training_utils import evaluate_model, evaluate_model_non_neural
from model_loader import load_model_components

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_traditional_ml(model):
    """Check if model is a Traditional ML model (sklearn) vs Deep Learning (PyTorch)."""
    import sklearn
    return isinstance(model, sklearn.base.BaseEstimator)


def extract_classwise_metrics(report_df, label_encoder, benchmark_prefix=""):
    """
    Extract class-wise metrics from classification report DataFrame.

    Args:
        report_df: DataFrame containing classification report
        label_encoder: LabelEncoder with class names
        benchmark_prefix: Prefix for column names (e.g., "sup_", "gse146773_")

    Returns:
        dict: Class-wise metrics (precision, recall, f1-score, accuracy, MCC)
    """
    classwise_metrics = {}

    # Get class names from label encoder
    class_names = label_encoder.classes_

    for cls in class_names:
        cls_lower = cls.lower()  # g1, s, g2m

        # Extract metrics from report_df
        # report_df has columns for each class and rows for precision, recall, f1-score, support
        if cls in report_df.columns:
            # Precision, Recall, F1-score from classification report
            if 'precision' in report_df.index:
                classwise_metrics[f'{benchmark_prefix}precision_{cls_lower}'] = report_df.loc['precision', cls]
            if 'recall' in report_df.index:
                classwise_metrics[f'{benchmark_prefix}recall_{cls_lower}'] = report_df.loc['recall', cls]
            if 'f1-score' in report_df.index:
                classwise_metrics[f'{benchmark_prefix}f1-score_{cls_lower}'] = report_df.loc['f1-score', cls]

            # Accuracy and MCC from the added rows
            if 'Accuracy' in report_df.index:
                classwise_metrics[f'{benchmark_prefix}accuracy_{cls_lower}'] = report_df.loc['Accuracy', cls]
            if 'MCC' in report_df.index:
                classwise_metrics[f'{benchmark_prefix}mcc_{cls_lower}'] = report_df.loc['MCC', cls]

    return classwise_metrics


def evaluate_on_benchmark(model, scaler, label_encoder, selected_features, model_dir, dataset_name, is_tml=False, scaling_method='simple'):
    """
    Evaluate model on a single benchmark dataset.

    Args:
        model: Trained model
        scaler: Fitted scaler
        label_encoder: Fitted label encoder
        selected_features: List of feature names
        model_dir: Model directory path
        dataset_name: "SUP", "GSE146773", "GSE64016", or "Buettner_mESC"
        is_tml: Whether model is traditional ML
        scaling_method: 'simple' (no double normalization) or 'double' (old method)

    Returns:
        dict: Evaluation metrics (overall + class-wise)
    """
    from utils.data_utils import (
        load_reh_or_sup_benchmark,
        load_gse146773,
        load_gse64016,
        load_buettner_mesc
    )

    # Detect if this is an old model (models/old_human/)
    is_old_model = "old_human" in model_dir or "/old_human/" in model_dir

    # Load benchmark data
    if dataset_name == "SUP":
        benchmark_features, benchmark_labels, _ = load_reh_or_sup_benchmark(scaler, reh_sup="sup", is_old_model=is_old_model, scaling_method=scaling_method)
    elif dataset_name == "GSE146773":
        benchmark_features, benchmark_labels, _ = load_gse146773(scaler, False, is_old_model=is_old_model, scaling_method=scaling_method)
    elif dataset_name == "GSE64016":
        benchmark_features, benchmark_labels, _ = load_gse64016(scaler, False, is_old_model=is_old_model, scaling_method=scaling_method)
    elif dataset_name == "Buettner_mESC":
        benchmark_features, benchmark_labels, _ = load_buettner_mesc(scaler, False, is_old_model=is_old_model, scaling_method=scaling_method)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # DEBUG: Print feature names comparison
    print(f"\n[DEBUG] Feature Names Comparison for {dataset_name}")
    print(f"{'='*80}")

    # Training features (what model expects)
    training_features = list(selected_features)
    print(f"TRAINING FEATURES (Total: {len(training_features)})")
    print(f"  First 10: {training_features[:10]}")
    print(f"  Last 10:  {training_features[-10:]}")
    print()

    # Benchmark features (what dataset provides)
    benchmark_feature_names = list(benchmark_features.columns)
    print(f"BENCHMARK FEATURES - {dataset_name} BEFORE REORDERING (Total: {len(benchmark_feature_names)})")
    print(f"  First 10: {benchmark_feature_names[:10]}")
    print(f"  Last 10:  {benchmark_feature_names[-10:]}")
    print()

    print(f"BEFORE REORDERING - First 5 rows x 5 columns:")
    print(benchmark_features.iloc[:5, :5])
    print()

    # Check for missing features
    missing_in_benchmark = set(training_features) - set(benchmark_feature_names)
    missing_in_training = set(benchmark_feature_names) - set(training_features)

    if missing_in_benchmark:
        print(f"WARNING: {len(missing_in_benchmark)} training features NOT found in {dataset_name} benchmark!")
        print(f"  First 10 missing: {list(missing_in_benchmark)[:10]}")
        raise ValueError(f"Cannot evaluate: {len(missing_in_benchmark)} features missing from benchmark!")

    if missing_in_training:
        print(f"INFO: {len(missing_in_training)} extra features in {dataset_name} benchmark (will be ignored)")

    # CRITICAL FIX: Reorder benchmark features to EXACTLY match training feature order
    print(f"\nREORDERING benchmark features to match training order...")
    benchmark_features = benchmark_features[training_features]

    # Verify reordering worked
    reordered_cols = list(benchmark_features.columns)
    print(f"AFTER REORDERING:")
    print(f"  First 10: {reordered_cols[:10]}")
    print(f"  Last 10:  {reordered_cols[-10:]}")
    print(f"  Order matches training: {reordered_cols == training_features}")
    print(f"  NaN values: {benchmark_features.isna().sum().sum()}")
    print()

    print(f"AFTER REORDERING - First 5 rows x 5 columns:")
    print(benchmark_features.iloc[:5, :5])
    print()

    if reordered_cols != training_features:
        raise ValueError("Feature reordering FAILED! Column order still doesn't match training!")
    if benchmark_features.isna().sum().sum() > 0:
        raise ValueError(f"NaN values detected after reordering! This will cause evaluation errors.")

    print(f"SUCCESS: Feature order verified and matched!")
    print(f"{'='*80}\n")

    if is_tml:
        # Traditional ML evaluation (sklearn models)
        # Features are already reordered above to match training order
        benchmark_labels_encoded = label_encoder.transform(benchmark_labels)

        accuracy, f1, precision, recall, roc_auc, balanced_acc, mcc, kappa, _, report_df = evaluate_model_non_neural(
            model, benchmark_features, benchmark_labels_encoded, label_encoder,
            model_dir, dataset_name=dataset_name
        )

        # Create benchmark prefix (lowercase)
        benchmark_prefix = dataset_name.lower().replace(" ", "_") + "_"

        # Extract class-wise metrics from report_df
        classwise_metrics = extract_classwise_metrics(report_df, label_encoder, benchmark_prefix)

        # Combine overall and class-wise metrics with benchmark prefix
        metrics = {
            f'{benchmark_prefix}accuracy': accuracy,
            f'{benchmark_prefix}f1': f1,
            f'{benchmark_prefix}precision': precision,
            f'{benchmark_prefix}recall': recall,
            f'{benchmark_prefix}roc_auc': roc_auc,
            f'{benchmark_prefix}balanced_acc': balanced_acc,
            f'{benchmark_prefix}mcc': mcc,
            f'{benchmark_prefix}kappa': kappa
        }
        metrics.update(classwise_metrics)
        return metrics
    else:
        # Deep Learning evaluation (PyTorch models)
        # Features are already reordered above to match training order
        benchmark_labels_encoded = torch.tensor(
            label_encoder.transform(benchmark_labels),
            dtype=torch.long
        ).to(device)

        benchmark_tensor = torch.tensor(
            benchmark_features.values,
            dtype=torch.float32
        ).to(device)

        # Create DataLoader
        benchmark_loader = DataLoader(
            TensorDataset(benchmark_tensor, benchmark_labels_encoded),
            batch_size=32,
            shuffle=False
        )

        # Evaluate model
        accuracy, test_loss, f1, precision, recall, roc_auc, balanced_acc, mcc, kappa, _, _, report_df = evaluate_model(
            model, benchmark_loader, nn.CrossEntropyLoss(), label_encoder,
            model_dir, dataset_name=dataset_name
        )

        # Create benchmark prefix (lowercase)
        benchmark_prefix = dataset_name.lower().replace(" ", "_") + "_"

        # Extract class-wise metrics from report_df
        classwise_metrics = extract_classwise_metrics(report_df, label_encoder, benchmark_prefix)

        # Combine overall and class-wise metrics with benchmark prefix
        metrics = {
            f'{benchmark_prefix}accuracy': accuracy,
            f'{benchmark_prefix}loss': test_loss,
            f'{benchmark_prefix}f1': f1,
            f'{benchmark_prefix}precision': precision,
            f'{benchmark_prefix}recall': recall,
            f'{benchmark_prefix}roc_auc': roc_auc,
            f'{benchmark_prefix}balanced_acc': balanced_acc,
            f'{benchmark_prefix}mcc': mcc,
            f'{benchmark_prefix}kappa': kappa
        }
        metrics.update(classwise_metrics)
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model on all 4 benchmarks'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model (.pt or .pkl file)'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='*',
        default=None,
        choices=["SUP", "GSE146773", "GSE64016", "Buettner_mESC"],
        help='Benchmarks to evaluate on (default: all 4 if no custom benchmark, else none)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: results/model_name_evaluation.csv)'
    )
    parser.add_argument(
        '--custom_benchmark',
        type=str,
        default=None,
        help='Path to custom benchmark data CSV'
    )
    parser.add_argument(
        '--custom_benchmark_name',
        type=str,
        default='CustomBenchmark',
        help='Name for custom benchmark (used in output files)'
    )
    parser.add_argument(
        '--scaling_method',
        type=str,
        choices=['simple', 'double'],
        default='simple',
        help='Scaling method for benchmarks: simple (correct - no double normalization) or double (old method with double normalization). Default: simple'
    )

    args = parser.parse_args()

    # Set default benchmarks behavior
    if args.benchmarks is None:
        # If no custom benchmark: use all 4 standard benchmarks
        # If custom benchmark provided: skip standard benchmarks (custom only)
        if args.custom_benchmark is None:
            args.benchmarks = ["SUP", "GSE146773", "GSE64016", "Buettner_mESC"]
        else:
            args.benchmarks = []  # Empty list = skip standard benchmarks

    # Validate that at least one benchmark is provided
    if len(args.benchmarks) == 0 and args.custom_benchmark is None:
        parser.error("Must provide at least one benchmark (either --benchmarks or --custom_benchmark)")

    # Load model components
    print(f"\n{'='*80}")
    print(f"LOADING MODEL")
    print(f"{'='*80}")
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Scaling method: {args.scaling_method}")

    model, scaler, label_encoder, selected_features = load_model_components(args.model_path)
    model_dir = os.path.dirname(args.model_path)
    model_name = os.path.basename(args.model_path).replace('.pt', '').replace('.pkl', '').replace('.joblib', '')

    # Check if TML or DL
    is_tml = is_traditional_ml(model)
    model_category = "Traditional ML" if is_tml else "Deep Learning"

    print(f"Model loaded successfully ({model_category})")
    print(f"{'='*80}\n")

    # Evaluate on selected benchmarks and collect results in WIDE format
    wide_results = {'prefix_name': model_name}

    for dataset_name in args.benchmarks:
        print(f"{'='*80}")
        print(f"{dataset_name}: Evaluating...")
        print(f"{'='*80}")

        metrics = evaluate_on_benchmark(
            model, scaler, label_encoder, selected_features,
            model_dir, dataset_name, is_tml=is_tml, scaling_method=args.scaling_method
        )

        # Add all metrics with benchmark prefix to wide_results
        wide_results.update(metrics)

        # Print accuracy (need to extract from prefixed key)
        benchmark_prefix = dataset_name.lower().replace(" ", "_") + "_"
        print(f"  Accuracy: {metrics[f'{benchmark_prefix}accuracy']:.2f}%")
        print(f"{'='*80}\n")

    # Evaluate on custom benchmark if provided
    if args.custom_benchmark is not None:
        print(f"{'='*80}")
        print(f"{args.custom_benchmark_name}: Evaluating custom benchmark...")
        print(f"{'='*80}")

        # Load custom benchmark
        from utils.data_utils import load_custom_benchmark
        benchmark_features, benchmark_labels, _ = load_custom_benchmark(
            args.custom_benchmark, scaler, args.custom_benchmark_name
        )

        if is_tml:
            from utils.training_utils import evaluate_model_non_neural
            benchmark_labels_encoded = label_encoder.transform(benchmark_labels)
            accuracy, f1, precision, recall, roc_auc, balanced_acc, mcc, kappa, _, report_df = evaluate_model_non_neural(
                model, benchmark_features, benchmark_labels_encoded, label_encoder,
                model_dir, dataset_name=args.custom_benchmark_name
            )
        else:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            import torch.nn as nn
            benchmark_labels_encoded = torch.tensor(
                label_encoder.transform(benchmark_labels),
                dtype=torch.long
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            benchmark_tensor = torch.tensor(
                benchmark_features.values,
                dtype=torch.float32
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            benchmark_loader = DataLoader(
                TensorDataset(benchmark_tensor, benchmark_labels_encoded),
                batch_size=32,
                shuffle=False
            )

            from utils.training_utils import evaluate_model
            accuracy, test_loss, f1, precision, recall, roc_auc, balanced_acc, mcc, kappa, _, _, report_df = evaluate_model(
                model, benchmark_loader, nn.CrossEntropyLoss(), label_encoder,
                model_dir, dataset_name=args.custom_benchmark_name
            )

        # Create benchmark prefix for custom benchmark (lowercase)
        custom_prefix = args.custom_benchmark_name.lower().replace(" ", "_") + "_"

        # Extract class-wise metrics
        classwise_metrics = extract_classwise_metrics(report_df, label_encoder, custom_prefix)

        custom_metrics = {
            f'{custom_prefix}accuracy': accuracy,
            f'{custom_prefix}f1': f1,
            f'{custom_prefix}precision': precision,
            f'{custom_prefix}recall': recall,
            f'{custom_prefix}roc_auc': roc_auc,
            f'{custom_prefix}balanced_acc': balanced_acc,
            f'{custom_prefix}mcc': mcc,
            f'{custom_prefix}kappa': kappa
        }
        custom_metrics.update(classwise_metrics)

        # Add custom metrics to wide_results
        wide_results.update(custom_metrics)

        print(f"  Accuracy: {custom_metrics[f'{custom_prefix}accuracy']:.2f}%")
        print(f"{'='*80}\n")

    # Create results DataFrame (WIDE format - one row with all benchmarks)
    results_df = pd.DataFrame([wide_results])

    # Save results as *_details_benchmark.csv
    if args.output is None:
        # Save in same directory as model file
        args.output = os.path.join(model_dir, f"{model_name}_details_benchmark.csv")
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results_df.to_csv(args.output, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY - Accuracies")
    print(f"{'='*80}")

    # Print accuracy for each benchmark evaluated
    for dataset_name in args.benchmarks:
        benchmark_prefix = dataset_name.lower().replace(" ", "_") + "_"
        accuracy_key = f'{benchmark_prefix}accuracy'
        if accuracy_key in wide_results:
            print(f"{dataset_name:15s}: {wide_results[accuracy_key]:6.2f}%")

    # Print custom benchmark accuracy if evaluated
    if args.custom_benchmark is not None:
        custom_prefix = args.custom_benchmark_name.lower().replace(" ", "_") + "_"
        accuracy_key = f'{custom_prefix}accuracy'
        if accuracy_key in wide_results:
            print(f"{args.custom_benchmark_name:15s}: {wide_results[accuracy_key]:6.2f}%")

    print(f"\n✅ Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
