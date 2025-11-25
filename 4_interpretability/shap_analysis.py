"""
SHAP Analysis for Model Interpretability
=========================================

Performs SHAP (SHapley Additive exPlanations) analysis on trained models
to understand feature importance and model predictions.

Based on 3_0_principle_aurelien_ml_evaluate.py line 616-892

NOTE: The SHAP_analysis function in the original code prepares data and evaluates
models on benchmarks, but the actual SHAP explainer computation appears to be
incomplete or was done separately. This script provides the framework.

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../2_model_training'))
from utils.data_utils import load_reh_or_sup_benchmark, load_gse14773, load_gse64016
from utils.training_utils import evaluate_model
from utils.io_utils import save_fold_to_csv_benchmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_predict_np(model, data):
    """
    Wrapper function for SHAP to make predictions on numpy arrays.

    Args:
        model: PyTorch model
        data (np.ndarray): Input data

    Returns:
        np.ndarray: Model predictions
    """
    data_tensor = torch.FloatTensor(data).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        predictions = model(data_tensor)
    return predictions.cpu().detach().numpy()


def SHAP_analysis(model_type, model, scaler, label_encoder, selected_features, model_prefix, model_dir, classifier=None):
    """
    Perform SHAP analysis and benchmark evaluation for a trained model.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 616-892

    This function:
    1. Loads benchmark data (SUP, GSE146773, GSE64016)
    2. Evaluates model on all benchmarks
    3. Saves comprehensive results to CSV
    4. Prepares data for SHAP analysis
    5. TODO: Complete SHAP explainer implementation

    Args:
        model_type (str): Model type identifier
        model: Trained PyTorch model
        scaler: Fitted sklearn scaler
        label_encoder: Fitted LabelEncoder
        selected_features (list): Feature names
        model_prefix (str): Model prefix for saving results
        model_dir (str): Directory containing model files
        classifier (optional): Classifier model (for VAE-based models)
    """
    # (1) Load SUP benchmark data
    sup_benchmark_features, sup_benchmark_labels, sup_benchmark_cell_ids = load_reh_or_sup_benchmark(scaler, reh_sup="sup")
    sup_y_test_encoded = torch.tensor(label_encoder.transform(sup_benchmark_labels), dtype=torch.long).to(device)
    sup_X_test_tensor = torch.tensor(sup_benchmark_features.values, dtype=torch.float32).to(device)
    sup_test_loader = DataLoader(TensorDataset(sup_X_test_tensor, sup_y_test_encoded), batch_size=32, shuffle=False)

    # (2) Load GSE146773 benchmark data
    gse146773_benchmark_features, gse146773_benchmark_labels, gse146773_benchmark_cell_ids = load_gse14773(scaler, check_feature=False)
    gse146773_y_test_encoded = torch.tensor(label_encoder.transform(gse146773_benchmark_labels), dtype=torch.long).to(device)
    gse146773_X_test_tensor = torch.tensor(gse146773_benchmark_features.values, dtype=torch.float32).to(device)
    gse146773_test_loader = DataLoader(TensorDataset(gse146773_X_test_tensor, gse146773_y_test_encoded), batch_size=32, shuffle=False)

    # (3) Load GSE64016 benchmark data
    gse64016_benchmark_features, gse64016_benchmark_labels, gse64016_benchmark_cell_ids = load_gse64016(scaler, check_feature=False)
    gse64016_y_test_encoded = torch.tensor(label_encoder.transform(gse64016_benchmark_labels), dtype=torch.long).to(device)
    gse64016_X_test_tensor = torch.tensor(gse64016_benchmark_features.values, dtype=torch.float32).to(device)
    gse64016_test_loader = DataLoader(TensorDataset(gse64016_X_test_tensor, gse64016_y_test_encoded), batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # (4) Evaluate on all three benchmarks
    print(f"\n{'='*80}")
    print(f"SHAP ANALYSIS & BENCHMARK EVALUATION: {model_prefix}")
    print(f"{'='*80}\n")

    # Evaluate on GSE146773
    (gse146773_test_accuracy, gse146773_test_loss, gse146773_test_f1, gse146773_test_precision,
     gse146773_test_recall, gse146773_test_roc_auc, gse146773_balanced_acc, gse146773_mcc,
     gse146773_kappa, _, what_nparra, gse146773_report_df) = evaluate_model(
        model, gse146773_test_loader, criterion, label_encoder, model_dir,
        dataset_name=f"Benchmark GSE146773({model_prefix})"
    )

    print(f"GSE146773 - Accuracy: {gse146773_test_accuracy:.2f}%")

    # Evaluate on GSE64016
    (gse64016_test_accuracy, gse64016_test_loss, gse64016_test_f1, gse64016_test_precision,
     gse64016_test_recall, gse64016_test_roc_auc, gse64016_balanced_acc, gse64016_mcc,
     gse64016_kappa, _, _, gse64016_report_df) = evaluate_model(
        model, gse64016_test_loader, criterion, label_encoder, model_dir,
        dataset_name=f"Benchmark GSE64016({model_prefix})"
    )

    print(f"GSE64016 - Accuracy: {gse64016_test_accuracy:.2f}%")

    # Evaluate on SUP
    (test_accuracy, test_loss, test_f1, test_precision, test_recall, test_roc_auc,
     test_balanced_acc, test_mcc, test_kappa, _, _, sup_report_df) = evaluate_model(
        model, sup_test_loader, criterion, label_encoder, model_dir,
        dataset_name=f"Benchmark SUP({model_prefix})"
    )

    print(f"SUP - Accuracy: {test_accuracy:.2f}%")

    # (5) Save benchmark results to CSV
    save_fold_to_csv_benchmark(
        model_prefix,
        test_accuracy, test_loss, test_f1, test_precision, test_recall, test_roc_auc,
        test_balanced_acc, test_mcc, test_kappa, sup_report_df,
        gse146773_test_accuracy, gse146773_test_loss, gse146773_test_f1, gse146773_test_precision,
        gse146773_test_recall, gse146773_test_roc_auc, gse146773_balanced_acc, gse146773_mcc,
        gse146773_kappa, gse146773_report_df,
        gse64016_test_accuracy, gse64016_test_loss, gse64016_test_f1, gse64016_test_precision,
        gse64016_test_recall, gse64016_test_roc_auc, gse64016_balanced_acc, gse64016_mcc,
        gse64016_kappa, gse64016_report_df,
        model_dir
    )

    # (6) Prepare data for SHAP analysis
    background_data = gse146773_benchmark_features.sample(n=10, random_state=42)
    remaining_data = gse146773_benchmark_features.drop(background_data.index)
    benchmark_data = remaining_data.sample(n=10, random_state=42)

    num_features = benchmark_data.shape[1]
    nsamples = min(50, num_features + 1)
    if num_features > nsamples:
        nsamples = num_features + 1

    print(f"\nSHAP Analysis Setup:")
    print(f"  Background samples: {len(background_data)}")
    print(f"  Explanation samples: {len(benchmark_data)}")
    print(f"  Features: {num_features}")
    print(f"  nsamples: {nsamples}")

    # TODO: Complete SHAP explainer implementation
    # The original code prepares data but doesn't show the complete SHAP computation
    # You may need to add:
    # import shap
    # explainer = shap.DeepExplainer(model, background_data_tensor)
    # shap_values = explainer.shap_values(benchmark_data_tensor)
    # shap.summary_plot(shap_values, benchmark_data, feature_names=selected_features)
    # Save SHAP values to file for later analysis

    print("\n✅ Benchmark evaluation complete!")
    print(f"   Results saved to: {model_dir}")
