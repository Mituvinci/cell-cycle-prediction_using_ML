"""
Ensemble Fusion Methods
========================

Contains score-level and decision-level fusion for combining multiple trained models.

EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, classification_report
)

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../2_model_training'))
from utils.training_utils import evaluate_model
from utils.data_utils import load_reh_or_sup_benchmark, load_gse146773, load_gse64016, load_buettner_mesc
from model_loader import load_model_components
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_benchmark_data_fusion(scaler, selected_features, label_encoder, dataset_name):
    """
    Load and preprocess benchmark datasets for fusion.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 1005-1021

    Args:
        scaler: Fitted sklearn scaler
        selected_features (list): List of feature names
        label_encoder: Fitted LabelEncoder
        dataset_name (str): "SUP", "GSE146773", or "GSE64016"

    Returns:
        tuple: (benchmark_loader, benchmark_labels_encoded)
    """
    if dataset_name == "SUP":
        benchmark_features, benchmark_labels, _ = load_reh_or_sup_benchmark(scaler, reh_sup="sup", is_old_model=True, scaling_method='double')
    elif dataset_name == "GSE146773":
        benchmark_features, benchmark_labels, _ = load_gse146773(scaler, check_feature=False, is_old_model=True, scaling_method='double')
    elif dataset_name == "GSE64016":
        benchmark_features, benchmark_labels, _ = load_gse64016(scaler, check_feature=False, is_old_model=True, scaling_method='double')
    elif dataset_name == "Buettner_mESC" or dataset_name == "BUETTNER":
        benchmark_features, benchmark_labels, _ = load_buettner_mesc(scaler, check_feature=False, is_old_model=True, scaling_method='double')
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    benchmark_labels_encoded = torch.tensor(label_encoder.transform(benchmark_labels), dtype=torch.long).to(device)
    benchmark_tensor = torch.tensor(benchmark_features.values, dtype=torch.float32).to(device)

    # Create DataLoader
    benchmark_loader = DataLoader(TensorDataset(benchmark_tensor, benchmark_labels_encoded), batch_size=32, shuffle=False)
    return benchmark_loader, benchmark_labels_encoded


def calculate_metrics(label_encoder, dataset_name, method_name, benchmark_labels_encoded, final_predictions, avg_probabilities=None):
    """
    Calculate comprehensive metrics for ensemble predictions.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 893-957

    Args:
        label_encoder: Fitted LabelEncoder
        dataset_name (str): Dataset name for prefix
        method_name (str): "score" or "decision"
        benchmark_labels_encoded (np.ndarray): True labels
        final_predictions (np.ndarray): Predicted labels
        avg_probabilities (np.ndarray, optional): Predicted probabilities (for score fusion)

    Returns:
        pd.DataFrame: Single-row DataFrame with all metrics
    """
    dataset_prefix = dataset_name.lower()

    # Compute general metrics
    accuracy = np.mean(benchmark_labels_encoded == final_predictions) * 100
    f1 = f1_score(benchmark_labels_encoded, final_predictions, average='weighted') * 100
    precision = precision_score(benchmark_labels_encoded, final_predictions, average='weighted') * 100
    recall = recall_score(benchmark_labels_encoded, final_predictions, average='weighted') * 100
    balanced_acc = balanced_accuracy_score(benchmark_labels_encoded, final_predictions) * 100
    mcc = matthews_corrcoef(benchmark_labels_encoded, final_predictions)
    kappa = cohen_kappa_score(benchmark_labels_encoded, final_predictions)
    auroc = roc_auc_score(benchmark_labels_encoded, avg_probabilities, multi_class="ovr") * 100 if avg_probabilities is not None else None

    # Classification report
    y_true_str = label_encoder.inverse_transform(np.array(benchmark_labels_encoded).astype(int))
    y_pred_str = label_encoder.inverse_transform(np.array(final_predictions).astype(int))
    index_realname = label_encoder.inverse_transform(np.unique(benchmark_labels_encoded))
    report = classification_report(
        y_true_str,
        y_pred_str,
        labels=index_realname,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report)

    # Class-wise Accuracy and MCC
    classwise_accuracy = {}
    classwise_mcc = {}
    for idx, cls in enumerate(label_encoder.classes_):
        mask = (np.array(benchmark_labels_encoded) == idx)
        class_acc = accuracy_score(np.array(benchmark_labels_encoded)[mask], np.array(final_predictions)[mask])
        bin_true = (np.array(benchmark_labels_encoded) == idx).astype(int)
        bin_pred = (np.array(final_predictions) == idx).astype(int)
        class_mcc = matthews_corrcoef(bin_true, bin_pred)
        classwise_accuracy[cls] = class_acc * 100
        classwise_mcc[cls] = class_mcc * 100

    # Flatten class-wise report metrics
    flat_metrics = {}
    for metric in ['precision', 'recall', 'f1-score']:
        for cls in label_encoder.classes_:
            flat_metrics[f"{dataset_prefix}_{metric}_{cls.lower()}"] = report_df.at[metric, cls] * 100

    for cls in label_encoder.classes_:
        flat_metrics[f"{dataset_prefix}_accuracy_{cls.lower()}"] = classwise_accuracy[cls]
        flat_metrics[f"{dataset_prefix}_mcc_{cls.lower()}"] = classwise_mcc[cls]

    # Add summary metrics
    summary = {
        f"{dataset_prefix}_accuracy": accuracy,
        f"{dataset_prefix}_f1": f1,
        f"{dataset_prefix}_precision": precision,
        f"{dataset_prefix}_recall": recall,
        f"{dataset_prefix}_roc_auc": auroc,
        f"{dataset_prefix}_balanced_acc": balanced_acc,
        f"{dataset_prefix}_mcc": mcc,
        f"{dataset_prefix}_kappa": kappa,
    }

    summary.update(flat_metrics)
    summary_df = pd.DataFrame([summary])
    summary_df.insert(0, "prefix_name", method_name)

    return summary_df


def score_level_fusion(models, dataset_name):
    """
    Perform score-level fusion by averaging model probabilities across multiple models.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 1023-1051

    Args:
        models (list): List of model paths (.pt files)
        dataset_name (str): "SUP", "GSE146773", or "GSE64016"

    Returns:
        pd.DataFrame: Metrics for fused predictions
    """
    avg_probabilities = None
    benchmark_labels_encoded = None
    label_encoder = None

    for model_path in models:
        # Load model components
        model, scaler, label_encoder, selected_features = load_model_components(model_path)

        # Load Benchmark Data
        benchmark_loader, benchmark_labels_encoded = preprocess_benchmark_data_fusion(
            scaler, selected_features, label_encoder, dataset_name
        )

        # Evaluate model and get predicted probabilities
        _, _, _, _, _, _, _, _, _, _, probabilities, _ = evaluate_model(
            model, benchmark_loader, nn.CrossEntropyLoss(), label_encoder,
            os.path.dirname(model_path), dataset_name=dataset_name
        )

        if avg_probabilities is None:
            avg_probabilities = probabilities
        else:
            avg_probabilities += probabilities  # Summing probabilities from all models

    avg_probabilities /= len(models)  # Average probabilities across models
    final_predictions = np.argmax(avg_probabilities, axis=1)  # Get final predicted labels

    # Compute fusion metrics
    final_df = calculate_metrics(
        label_encoder, dataset_name, "score",
        benchmark_labels_encoded.cpu().numpy(), final_predictions, avg_probabilities
    )
    return final_df


def decision_level_fusion(models, dataset_name):
    """
    Perform decision-level fusion by majority voting across models.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 1054-1080

    Args:
        models (list): List of model paths (.pt files)
        dataset_name (str): "SUP", "GSE146773", or "GSE64016"

    Returns:
        pd.DataFrame: Metrics for fused predictions
    """
    all_predictions = []
    benchmark_labels_encoded = None
    label_encoder = None

    for model_path in models:
        # Load model components
        model, scaler, label_encoder, selected_features = load_model_components(model_path)

        # Load Benchmark Data
        benchmark_loader, benchmark_labels_encoded = preprocess_benchmark_data_fusion(
            scaler, selected_features, label_encoder, dataset_name
        )

        # Evaluate model and get predicted labels
        _, _, _, _, _, _, _, _, _, _, probabilities, _ = evaluate_model(
            model, benchmark_loader, nn.CrossEntropyLoss(), label_encoder,
            os.path.dirname(model_path), dataset_name=dataset_name
        )

        predictions = np.argmax(probabilities, axis=1)  # Convert probabilities to class labels
        all_predictions.append(predictions)

    # Majority voting
    all_predictions = np.array(all_predictions).T  # Transpose to have models on the second axis
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=all_predictions)

    # Compute fusion metrics
    final_df = calculate_metrics(
        label_encoder, dataset_name, "decision",
        benchmark_labels_encoded.cpu().numpy(), final_predictions
    )
    return final_df


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='Ensemble fusion for deep learning models')
    parser.add_argument('--fusion_type', type=str, required=True, choices=['score', 'decision'],
                        help='Fusion type: score (average probabilities) or decision (majority voting)')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top models to use')
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='List of model directories containing .pt files')
    parser.add_argument('--benchmark_data', type=str, required=True,
                        help='Path to benchmark data CSV file')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth labels CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')

    args = parser.parse_args()

    # Find all .pt files in model directories
    model_paths = []
    for model_dir in args.model_dirs:
        pt_files = glob.glob(os.path.join(model_dir, "*.pt"))
        if pt_files:
            model_paths.append(pt_files[0])  # Use first .pt file in directory

    # Limit to top_k models
    model_paths = model_paths[:args.top_k]

    # Determine dataset name from benchmark_data path
    if 'GSE146773' in args.benchmark_data:
        dataset_name = 'GSE146773'
    elif 'GSE64016' in args.benchmark_data:
        dataset_name = 'GSE64016'
    elif 'SUP' in args.benchmark_data:
        dataset_name = 'SUP'
    else:
        raise ValueError(f"Cannot determine dataset name from: {args.benchmark_data}")

    print(f"\n{'='*80}")
    print(f"ENSEMBLE FUSION: {args.fusion_type.upper()}")
    print(f"{'='*80}")
    print(f"Top-{args.top_k} models:")
    for i, model_path in enumerate(model_paths, 1):
        print(f"  {i}. {os.path.basename(model_path)}")
    print(f"Benchmark: {dataset_name}")
    print(f"Benchmark Data: {args.benchmark_data}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output: {args.output}")
    print(f"{'='*80}\n")

    # Run fusion
    if args.fusion_type == 'score':
        results_df = score_level_fusion(model_paths, dataset_name)
    else:
        results_df = decision_level_fusion(model_paths, dataset_name)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)

    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
