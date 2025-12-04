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


def extract_classwise_metrics(report_df, label_encoder):
    """
    Extract class-wise metrics from classification report DataFrame.

    Args:
        report_df: DataFrame containing classification report
        label_encoder: LabelEncoder with class names

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
                classwise_metrics[f'precision_{cls_lower}'] = report_df.loc['precision', cls]
            if 'recall' in report_df.index:
                classwise_metrics[f'recall_{cls_lower}'] = report_df.loc['recall', cls]
            if 'f1-score' in report_df.index:
                classwise_metrics[f'f1_{cls_lower}'] = report_df.loc['f1-score', cls]

            # Accuracy and MCC from the added rows
            if 'Accuracy' in report_df.index:
                classwise_metrics[f'accuracy_{cls_lower}'] = report_df.loc['Accuracy', cls]
            if 'MCC' in report_df.index:
                classwise_metrics[f'mcc_{cls_lower}'] = report_df.loc['MCC', cls]

    return classwise_metrics


def evaluate_on_benchmark(model, scaler, label_encoder, selected_features, model_dir, dataset_name, is_tml=False):
    """
    Evaluate model on a single benchmark dataset.

    Args:
        model: Trained model
        scaler: Fitted scaler
        label_encoder: Fitted label encoder
        selected_features: List of feature names
        model_dir: Model directory path
        dataset_name: "SUP", "GSE146773", "GSE64016", or "Buettner_mESC"

    Returns:
        dict: Evaluation metrics (overall + class-wise)
    """
    from utils.data_utils import (
        load_reh_or_sup_benchmark,
        load_gse146773,
        load_gse64016,
        load_buettner_mesc
    )

    # Load benchmark data
    if dataset_name == "SUP":
        benchmark_features, benchmark_labels, _ = load_reh_or_sup_benchmark(scaler, reh_sup="sup")
    elif dataset_name == "GSE146773":
        benchmark_features, benchmark_labels, _ = load_gse146773(scaler, False)
    elif dataset_name == "GSE64016":
        benchmark_features, benchmark_labels, _ = load_gse64016(scaler, False)
    elif dataset_name == "Buettner_mESC":
        benchmark_features, benchmark_labels, _ = load_buettner_mesc(scaler, False)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    if is_tml:
        # Traditional ML evaluation (sklearn models)
        # IMPORTANT: TML models need only the selected features (after feature selection)
        # Filter benchmark_features to match the exact features used during training
        benchmark_features_filtered = benchmark_features[selected_features]

        benchmark_labels_encoded = label_encoder.transform(benchmark_labels)

        accuracy, f1, precision, recall, roc_auc, balanced_acc, mcc, kappa, _, report_df = evaluate_model_non_neural(
            model, benchmark_features_filtered, benchmark_labels_encoded, label_encoder,
            model_dir, dataset_name=dataset_name
        )

        # Extract class-wise metrics from report_df
        classwise_metrics = extract_classwise_metrics(report_df, label_encoder)

        # Combine overall and class-wise metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'balanced_acc': balanced_acc,
            'mcc': mcc,
            'kappa': kappa
        }
        metrics.update(classwise_metrics)
        return metrics
    else:
        # Deep Learning evaluation (PyTorch models)
        # IMPORTANT: DL models also need only the selected features (after feature selection)
        # Filter benchmark_features to match the exact features used during training
        benchmark_features_filtered = benchmark_features[selected_features]

        benchmark_labels_encoded = torch.tensor(
            label_encoder.transform(benchmark_labels),
            dtype=torch.long
        ).to(device)

        benchmark_tensor = torch.tensor(
            benchmark_features_filtered.values,
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

        # Extract class-wise metrics from report_df
        classwise_metrics = extract_classwise_metrics(report_df, label_encoder)

        # Combine overall and class-wise metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'balanced_acc': balanced_acc,
            'mcc': mcc,
            'kappa': kappa
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

    model, scaler, label_encoder, selected_features = load_model_components(args.model_path)
    model_dir = os.path.dirname(args.model_path)
    model_name = os.path.basename(args.model_path).replace('.pt', '').replace('.pkl', '').replace('.joblib', '')

    # Check if TML or DL
    is_tml = is_traditional_ml(model)
    model_category = "Traditional ML" if is_tml else "Deep Learning"

    print(f"✓ Model loaded successfully ({model_category})")
    print(f"{'='*80}\n")

    # Evaluate on selected benchmarks
    results = []

    for dataset_name in args.benchmarks:
        print(f"{'='*80}")
        print(f"{dataset_name}: Evaluating...")
        print(f"{'='*80}")

        metrics = evaluate_on_benchmark(
            model, scaler, label_encoder, selected_features,
            model_dir, dataset_name, is_tml=is_tml
        )
        results.append(metrics)

        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
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

        # Extract class-wise metrics
        classwise_metrics = extract_classwise_metrics(report_df, label_encoder)

        custom_metrics = {
            'dataset': args.custom_benchmark_name,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'balanced_acc': balanced_acc,
            'mcc': mcc,
            'kappa': kappa
        }
        custom_metrics.update(classwise_metrics)
        results.append(custom_metrics)

        print(f"  Accuracy: {custom_metrics['accuracy']:.2f}%")
        print(f"{'='*80}\n")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.insert(0, 'model', model_name)

    # Save results
    if args.output is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'results'
        )
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{model_name}_all_benchmarks.csv")
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results_df.to_csv(args.output, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY - Accuracies")
    print(f"{'='*80}")
    for _, row in results_df.iterrows():
        print(f"{row['dataset']:15s}: {row['accuracy']:6.2f}%")
    print(f"\n✅ Full results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
