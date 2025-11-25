"""
Advanced Visualization for Publication-Quality Plots
====================================================

Contains advanced plotting functions for:
- Calibration curves
- Class-wise precision/recall heatmaps
- Model comparison visualizations

EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_true, y_pred_proba, label_encoder, n_bins=10):
    """
    Plot calibration curve for multi-class classification with actual class names.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 10-39

    Args:
        y_true (array-like): Ground truth labels.
        y_pred_proba (array-like): Predicted probabilities.
        label_encoder (LabelEncoder): Fitted label encoder to inverse transform class labels.
        n_bins (int): Number of bins for calibration curve.
    """
    # Convert y_pred_proba to NumPy array if it's a list
    if isinstance(y_pred_proba, list):
        y_pred_proba = np.array(y_pred_proba)

    # Get actual class names using the label encoder
    class_names = label_encoder.inverse_transform(np.unique(y_true))

    plt.figure(figsize=(10, 5))

    # Loop over each class and plot the calibration curve
    for i, class_name in enumerate(class_names):
        prob_true, prob_pred = calibration_curve(np.array(y_true) == i, y_pred_proba[:, i], n_bins=n_bins)
        plt.plot(prob_pred, prob_true, label=f'Class {class_name}')

    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend()
    plt.show()


def build_heatmap_data(df, dataset_prefix):
    """
    Extract precision and recall for each class and model to build heatmap data.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 1785-1821

    Args:
        df (pd.DataFrame): Results DataFrame with all model metrics
        dataset_prefix (str): "gse146773", "gse64016", or "sup"

    Returns:
        pd.DataFrame: Heatmap data with rows=[Precision/Recall per model], columns=[G1, G2M, S]
    """
    rows = []
    labels = []

    def extract_and_append(row, model_label, dataset_prefix):
        rows.append([
            row[f"{dataset_prefix}_precision_g1"],
            row[f"{dataset_prefix}_precision_g2m"],
            row[f"{dataset_prefix}_precision_s"]
        ])
        labels.append(f"Precision_{model_label}")

        rows.append([
            row[f"{dataset_prefix}_recall_g1"],
            row[f"{dataset_prefix}_recall_g2m"],
            row[f"{dataset_prefix}_recall_s"]
        ])
        labels.append(f"Recall_{model_label}")

    # Extract for each model type
    extract_and_append(df[df['prefix_name'].str.contains("deepdense")].iloc[0], "DNN3", dataset_prefix)
    extract_and_append(df[df['prefix_name'].str.contains("enhanced")].iloc[0], "DNN5", dataset_prefix)
    extract_and_append(df[df['prefix_name'].str.contains("fe")].iloc[0], "FE", dataset_prefix)
    extract_and_append(df[df['prefix_name'].str.contains("cnn")].iloc[0], "CNN", dataset_prefix)
    extract_and_append(df[df['prefix_name'].str.contains("hbdcnn")].iloc[0], "Hybrid CNN", dataset_prefix)

    # Extract for ensemble models if they exist
    for label, key in [("Top_3_D.F.", "Top_3_D.F."), ("Top_3_S.F.", "Top_3_S.F.")]:
        match = df[df['prefix_name'].str.contains(key)]
        if not match.empty:
            extract_and_append(match.iloc[0], label, dataset_prefix)

    # Extract for traditional ML models
    extract_and_append(df[df['prefix_name'].str.contains("lgbm")].iloc[0], "LGBM", dataset_prefix)
    extract_and_append(df[df['prefix_name'].str.contains("adaboost")].iloc[0], "Adaboost", dataset_prefix)
    extract_and_append(df[df['prefix_name'].str.contains("random")].iloc[0], "RF", dataset_prefix)
    extract_and_append(df[df['prefix_name'].str.contains("ensemble")].iloc[0], "Embedding3TML", dataset_prefix)

    return pd.DataFrame(rows, index=labels, columns=['G1', 'G2M', 'S'])


def plot_heatmap(df, title, save_path):
    """
    Generate and plot a heatmap for class-wise precision and recall.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 1824-1838

    Args:
        df (pd.DataFrame): Heatmap data (rows=metrics, columns=classes)
        title (str): Plot title
        save_path (str): Path to save the heatmap image
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(df.values, cmap='YlGnBu', vmin=0, vmax=100.0)

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    plt.savefig(save_path)
    plt.show()

    print(f"✅ Heatmap saved to: {save_path}")


def create_comparison_heatmaps(results_csv, output_dir):
    """
    Create heatmaps comparing all models across benchmarks.

    Args:
        results_csv (str): Path to CSV file with all model results
        output_dir (str): Directory to save heatmap images

    Usage:
        create_comparison_heatmaps("results/all_models_summary.csv", "figures/")
    """
    os.makedirs(output_dir, exist_ok=True)
    df_all = pd.read_csv(results_csv)

    # Prepare and plot for both datasets
    df_146773 = build_heatmap_data(df_all, "gse146773")
    df_64016 = build_heatmap_data(df_all, "gse64016")

    print(df_146773)

    heatmap_146773_path = os.path.join(output_dir, "gse146773_precision_recall_heatmap.png")
    heatmap_64016_path = os.path.join(output_dir, "gse64016_precision_recall_heatmap.png")

    plot_heatmap(df_146773, "GSE146773 Class-wise Precision and Recall", heatmap_146773_path)
    plot_heatmap(df_64016, "GSE64016 Class-wise Precision and Recall", heatmap_64016_path)

    return heatmap_146773_path, heatmap_64016_path
