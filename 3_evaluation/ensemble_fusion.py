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
from model_loader import load_model_components

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
    from torch.utils.data import DataLoader, TensorDataset
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../2_model_training'))
    from utils.data_utils import load_reh_or_sup_benchmark, load_gse14773, load_gse64016

    if dataset_name == "SUP":
        benchmark_features, benchmark_labels, _ = load_reh_or_sup_benchmark(scaler, reh_sup="sup")
    elif dataset_name == "GSE146773":
        benchmark_features, benchmark_labels, _ = load_gse14773(scaler, False)
    elif dataset_name == "GSE64016":
        benchmark_features, benchmark_labels, _ = load_gse64016(scaler, False)
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
