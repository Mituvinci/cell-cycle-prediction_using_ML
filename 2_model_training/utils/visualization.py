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
    Plots training loss and accuracy over epochs.
    train_loss and train_score should each be a list of floats,
    with the same length = number of epochs.

    EXACT implementation from 1_0_principle_aurelien_ml.py

    Args:
        train_loss (list): Training losses per epoch
        train_score (list): Training accuracies per epoch
        save_dir (str): Directory to save plot
        name (str): Name prefix for saved file

    Returns:
        str: Path to saved plot
    """
    if name == "training":
        title = "Training Loss and Accuracy"
    if name == "classification":
        title = "Classification Loss and Accuracy"
    else:
        title = "Pretraining Loss and reconstruction similarity"
    epochs = range(len(train_loss))  # e.g. 0..(epochs-1)
    plot_path = os.path.join(save_dir, f"{name}_training_plot.png")
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, train_score, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f"{title}")
    plt.legend()
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    return plot_path


def plot_validation_accuracy(test_scores, outer_fold, save_dir, title=""):
    """
    Plots validation accuracy across inner folds for each outer fold.

    EXACT implementation from 1_0_principle_aurelien_ml.py

    Args:
        test_scores (list): Test accuracies for each fold
        outer_fold (int): Outer fold number
        save_dir (str): Directory to save plot
        title (str): Plot title prefix

    Returns:
        str: Path to saved plot
    """
    plot_path = os.path.join(save_dir, "_validationperfold_plot.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(test_scores)), test_scores, marker='o', label='Validation Accuracy')
    plt.xlabel('Inner Fold')
    plt.ylabel('Validation Accuracy')
    plt.title(f'{title} Validation Accuracy per Inner Fold (Outer Fold {outer_fold})')
    plt.legend()
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    return plot_path
