"""
Visualization Utilities for Training Results
=============================================

Contains plotting functions for training history and validation metrics.

EXACT implementation from 1_0_principle_aurelien_ml.py

Author: Halima Akhter
Date: 2025-11-24
"""

import matplotlib.pyplot as plt
import os


def plot_training_history(train_loss, train_score, save_dir, name="training"):
    """
    Plot training loss and accuracy over epochs.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 1433-1459

    Args:
        train_loss (list): Training losses per epoch
        train_score (list): Training accuracies per epoch
        save_dir (str): Directory to save plot
        name (str): Name prefix for saved file

    Returns:
        str: Path to saved plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Loss
    ax1.plot(train_loss, label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{name.capitalize()} Loss over Epochs')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(train_score, label='Training Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{name.capitalize()} Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, f"{name}_history.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def plot_validation_accuracy(test_scores, outer_fold, save_dir, title=""):
    """
    Plot validation accuracy across folds.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 1461-1495

    Args:
        test_scores (list): Test accuracies for each fold
        outer_fold (int): Number of folds
        save_dir (str): Directory to save plot
        title (str): Plot title

    Returns:
        str: Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    folds = list(range(1, outer_fold + 1))
    ax.plot(folds, test_scores, marker='o', linestyle='-', color='purple', linewidth=2, markersize=8)
    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(title if title else 'Validation Accuracy per Fold', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(folds)

    # Add mean and std annotation
    mean_acc = sum(test_scores) / len(test_scores)
    import numpy as np
    std_acc = np.std(test_scores)
    ax.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.2f}%')
    ax.legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, "validation_accuracy_per_fold.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path
