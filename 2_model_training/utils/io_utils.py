"""
I/O Utilities for Saving and Loading Results
=============================================

Contains functions for saving model results, scalers, and metadata to CSV and pickle files.

EXACT implementation from 1_0_principle_aurelien_ml.py

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import pandas as pd
import joblib


def save_scaler(scaler, filepath):
    """
    Save scaler to pickle file.

    Args:
        scaler: sklearn scaler object
        filepath (str): Path to save the scaler
    """
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to: {filepath}")


def load_scaler(filepath):
    """
    Load scaler from pickle file.

    Args:
        filepath (str): Path to the scaler file

    Returns:
        sklearn scaler object
    """
    scaler = joblib.load(filepath)
    print(f"Scaler loaded from: {filepath}")
    return scaler


def save_fold_to_csv(
    prefix_name,
    best_params,
    training_time,
    gpu_memory_used,
    cpu_memory_used,
    train_scores,
    train_losses,
    test_accuracy,
    test_loss,
    test_f1,
    test_precision,
    test_recall,
    test_roc_auc,
    test_balanced_acc,
    test_mcc,
    test_kappa,
    save_model_here
):
    """
    Appends this fold's results to a CSV named '{prefix_name}_details.csv'.
    If the CSV does not exist, we write the header; if it does exist, we append without header.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 2099-2170

    Args:
        prefix_name (str): Prefix for naming the CSV file
        best_params (dict): Best hyperparameters from Optuna
        training_time (float): Training time in seconds
        gpu_memory_used (float): GPU memory used in GB
        cpu_memory_used (float): CPU memory used in GB
        train_scores (list): Training accuracies per epoch
        train_losses (list): Training losses per epoch
        test_accuracy (float): Test accuracy
        test_loss (float): Test loss
        test_f1 (float): Test F1 score
        test_precision (float): Test precision
        test_recall (float): Test recall
        test_roc_auc (float): Test ROC-AUC
        test_balanced_acc (float): Test balanced accuracy
        test_mcc (float): Test Matthews correlation coefficient
        test_kappa (float): Test Cohen's kappa
        save_model_here (str): Directory to save the CSV file
    """
    # You might pick either the final or average training values. For demonstration,
    # let's use final epoch's training accuracy / loss if available:
    final_train_accuracy = train_scores[-1] if train_scores else None
    final_train_loss = train_losses[-1] if train_losses else None

    # Build dictionary for a single row
    fold_data = {
        "prefix_name": prefix_name,
        "best_params": str(best_params),
        "training_time": training_time,
        "gpu_memory_used": gpu_memory_used,
        "cpu_memory_used": cpu_memory_used,
        "final_train_accuracy": final_train_accuracy,
        "final_train_loss": final_train_loss,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_roc_auc": test_roc_auc,
        "test_balanced_acc": test_balanced_acc,
        "test_mcc": test_mcc,
        "test_kappa": test_kappa
    }

    df_fold = pd.DataFrame([fold_data])

    # Build the CSV name
    csv_filename = f"{prefix_name}_details.csv"
    csv_filepath = os.path.join(save_model_here, csv_filename)

    # Append logic
    file_exists = os.path.isfile(csv_filepath)
    if not file_exists:
        # Write with header
        df_fold.to_csv(csv_filepath, index=False, mode='w')
        print(f"Created new CSV file: {csv_filepath}")
    else:
        # Append without header
        df_fold.to_csv(csv_filepath, index=False, mode='a', header=False)
        print(f"Appended to existing CSV file: {csv_filepath}")


def save_fold_to_csv_tml(
    prefix_name,
    best_params,
    training_time,
    cpu_memory_used,
    gpu_memory_used,
    test_accuracy,
    test_f1,
    test_precision,
    test_recall,
    test_roc_auc,
    test_balanced_acc,
    test_mcc,
    test_kappa,
    save_model_here
):
    """
    Traditional ML version: Appends this fold's results to a CSV named '{prefix_name}_details.csv'.
    If the CSV does not exist, we write the header; if it does exist, we append without header.

    EXACT implementation from 2_0_principle_aurelien_ml_traditional.py line 510-567

    Args:
        prefix_name (str): Prefix for naming the CSV file
        best_params (dict): Best hyperparameters from Optuna
        training_time (float): Training time in seconds
        cpu_memory_used (float): CPU memory used in GB
        gpu_memory_used (float): GPU memory used in GB
        test_accuracy (float): Test accuracy
        test_f1 (float): Test F1 score
        test_precision (float): Test precision
        test_recall (float): Test recall
        test_roc_auc (float): Test ROC-AUC
        test_balanced_acc (float): Test balanced accuracy
        test_mcc (float): Test Matthews correlation coefficient
        test_kappa (float): Test Cohen's kappa
        save_model_here (str): Directory to save the CSV file
    """
    # Build dictionary for a single row
    fold_data = {
        "prefix_name": prefix_name,
        "best_params": str(best_params),
        "training_time": training_time,
        "cpu_memory_used": cpu_memory_used,
        "gpu_memory_used": gpu_memory_used,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_roc_auc": test_roc_auc,
        "test_balanced_acc": test_balanced_acc,
        "test_mcc": test_mcc,
        "test_kappa": test_kappa
    }

    df_fold = pd.DataFrame([fold_data])

    # Build the CSV name
    csv_filename = f"{prefix_name}_details.csv"
    csv_filepath = os.path.join(save_model_here, csv_filename)

    # Append logic
    file_exists = os.path.isfile(csv_filepath)
    if not file_exists:
        # Write with header
        df_fold.to_csv(csv_filepath, index=False)
        print(f"Created CSV with header => {csv_filepath}")
    else:
        # Append without header
        df_fold.to_csv(csv_filepath, mode='a', header=False, index=False)
        print(f"Appended row to existing CSV => {csv_filepath}")


def flatten_classification_report(report_df, prefix=""):
    """
    Flatten a classification report DataFrame into a single-row dictionary with custom prefix.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 263-275

    Args:
        report_df (pd.DataFrame): Classification report as DataFrame
        prefix (str): Prefix to add to column names (e.g., "sup_", "gse146773_")

    Returns:
        pd.DataFrame: Single-row DataFrame with flattened metrics
    """
    report_df = report_df.copy()
    flat_dict = {}

    for index, row in report_df.iterrows():
        for col in report_df.columns:
            col_name = f"{prefix}{index}_{col}".replace(" ", "_").lower()
            flat_dict[col_name] = row[col]

    return pd.DataFrame([flat_dict])


def save_fold_to_csv_benchmark(
    prefix_name,
    test_accuracy,
    test_loss,
    test_f1,
    test_precision,
    test_recall,
    test_roc_auc,
    test_balanced_acc,
    test_mcc,
    test_kappa,
    test_report_df,
    gse146773_test_accuracy,
    gse146773_test_loss,
    gse146773_test_f1,
    gse146773_test_precision,
    gse146773_test_recall,
    gse146773_test_roc_auc,
    gse146773_balanced_acc,
    gse146773_mcc,
    gse146773_kappa,
    gse146773_report_df,
    gse64016_test_accuracy,
    gse64016_test_loss,
    gse64016_test_f1,
    gse64016_test_precision,
    gse64016_test_recall,
    gse64016_test_roc_auc,
    gse64016_balanced_acc,
    gse64016_mcc,
    gse64016_kappa,
    gse64016_report_df,
    save_model_here
):
    """
    Saves benchmark evaluation results (SUP, GSE146773, GSE64016) to CSV.
    Overwrites existing file (not append mode).

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 277-383

    Args:
        prefix_name (str): Model prefix name
        test_accuracy, test_loss, etc.: SUP benchmark metrics
        gse146773_*: GSE146773 benchmark metrics
        gse64016_*: GSE64016 benchmark metrics
        test_report_df, gse146773_report_df, gse64016_report_df: Classification reports
        save_model_here (str): Directory to save CSV
    """
    # Build dictionary for a single row
    fold_data = {
        "prefix_name": prefix_name,
        "sup_accuracy": test_accuracy,
        "sup_loss": test_loss,
        "sup_f1": test_f1,
        "sup_precision": test_precision,
        "sup_recall": test_recall,
        "sup_roc_auc": test_roc_auc,
        "sup_balanced_acc": test_balanced_acc,
        "sup_mcc": test_mcc,
        "sup_kappa": test_kappa,

        "gse146773_accuracy": gse146773_test_accuracy,
        "gse146773_loss": gse146773_test_loss,
        "gse146773_f1": gse146773_test_f1,
        "gse146773_precision": gse146773_test_precision,
        "gse146773_recall": gse146773_test_recall,
        "gse146773_roc_auc": gse146773_test_roc_auc,
        "gse146773_balanced_acc": gse146773_balanced_acc,
        "gse146773_mcc": gse146773_mcc,
        "gse146773_kappa": gse146773_kappa,

        "gse64016_accuracy": gse64016_test_accuracy,
        "gse64016_loss": gse64016_test_loss,
        "gse64016_f1": gse64016_test_f1,
        "gse64016_precision": gse64016_test_precision,
        "gse64016_recall": gse64016_test_recall,
        "gse64016_roc_auc": gse64016_test_roc_auc,
        "gse64016_balanced_acc": gse64016_balanced_acc,
        "gse64016_mcc": gse64016_mcc,
        "gse64016_kappa": gse64016_kappa
    }

    flat_sup_report = flatten_classification_report(test_report_df, prefix="sup_")
    flat_gse146773_report = flatten_classification_report(gse146773_report_df, prefix="gse146773_")
    flat_gse64016_report = flatten_classification_report(gse64016_report_df, prefix="gse64016_")

    df_fold = pd.DataFrame([fold_data])
    df_fold = pd.concat([df_fold, flat_sup_report, flat_gse146773_report, flat_gse64016_report], axis=1)

    columns_to_drop = [
        'sup_accuracy_accuracy', 'sup_accuracy_macro_avg', 'sup_accuracy_weighted_avg',
        'sup_mcc_accuracy', 'sup_mcc_macro_avg', 'sup_mcc_weighted_avg',
        'gse146773_accuracy_accuracy', 'gse146773_accuracy_macro_avg', 'gse146773_accuracy_weighted_avg',
        'gse64016_accuracy_accuracy', 'gse64016_accuracy_macro_avg', 'gse64016_accuracy_weighted_avg',
        'gse146773_mcc_accuracy', 'gse146773_mcc_macro_avg', 'gse146773_mcc_weighted_avg',
        'gse64016_mcc_accuracy', 'gse64016_mcc_macro_avg', 'gse64016_mcc_weighted_avg'
    ]

    df_fold = df_fold.drop(columns=columns_to_drop, errors='ignore')

    # Build the CSV name
    csv_filename = f"{prefix_name}_details_benchmark.csv"
    csv_filepath = os.path.join(save_model_here, csv_filename)

    def delete_csv_file(csv_filepath):
        """Deletes the CSV file if it exists."""
        if os.path.exists(csv_filepath):
            os.remove(csv_filepath)
            print(f"Deleted: {csv_filepath}")

    delete_csv_file(csv_filepath)
    df_fold.to_csv(csv_filepath, index=False)
    print(f"Created CSV with header => {csv_filepath}")
