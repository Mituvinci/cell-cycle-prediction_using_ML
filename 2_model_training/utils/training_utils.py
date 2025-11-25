"""
Training Utilities for Cell Cycle Prediction Models
===================================================

Contains key training functions:
- Focal loss for class imbalance
- Model training with early stopping
- Model evaluation with comprehensive metrics

EXACT implementation from 1_0_principle_aurelien_ml.py

Author: Halima Akhter
Date: 2025-11-24
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torchmetrics
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, log_loss
)
import matplotlib.pyplot as plt
import optuna
import os


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Compute Focal Loss for multi-class classification.

    Focal loss helps address class imbalance by down-weighting easy examples
    and focusing on hard misclassified examples.

    Parameters:
    -----------
    logits : torch.Tensor
        Model outputs (raw scores before softmax), shape (batch_size, num_classes)
    targets : torch.Tensor
        True class labels, shape (batch_size,)
    alpha : float, default=0.25
        Balancing factor to handle class imbalance
    gamma : float, default=2.0
        Focusing parameter to penalize confident misclassifications
        Higher gamma puts more focus on hard examples

    Returns:
    --------
    loss : torch.Tensor
        Computed focal loss (scalar)

    References:
    -----------
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
    targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()

    # Compute focal loss components
    ce_loss = -targets_one_hot * torch.log(probs + 1e-9)  # Cross-entropy
    focal_weight = alpha * (1 - probs) ** gamma  # Modulation factor

    loss = focal_weight * ce_loss  # Apply focal weighting
    return loss.sum(dim=1).mean()  # Reduce loss across batch


def train_model(model, train_loader, optimizer, criterion, epochs, log_dir, early_stopping_patience, trial=None, use_lr_scheduler=False, step_size=30, gamma=0.1):
    """
    Train model with early stopping, gradient clipping, and optional learning rate scheduling.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 1498-1580

    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        epochs: Maximum number of epochs
        log_dir: Directory for TensorBoard logs
        early_stopping_patience: Patience for early stopping
        trial: Optuna trial (optional)
        use_lr_scheduler: Whether to use LR scheduler
        step_size: Step size for scheduler
        gamma: Gamma for scheduler

    Returns:
        tuple: (train_losses, train_scores, model)
    """
    device = next(model.parameters()).device  # Get device from model
    writer = SummaryWriter(log_dir=log_dir)  # Initialize TensorBoard writer
    model.train()
    train_losses = []
    train_scores = []
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None  # To store the best model weights
    scaler = None  # For mixed precision training (if needed)

    # Optional learning rate scheduler
    if use_lr_scheduler:
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    output = model(data.float())
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data.float())
                loss = criterion(output, target)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_accuracy = 100. * correct / total if total > 0 else 0
        train_scores.append(train_accuracy)

        if use_lr_scheduler:
            scheduler.step()

        # Optionally report to Optuna and check for pruning
        if trial is not None:
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                writer.close()
                raise optuna.exceptions.TrialPruned()

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        writer.add_scalar('Epoch Loss', avg_loss, epoch)
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)

        # Early stopping: Save the best model state
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    writer.close()
    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return train_losses, train_scores, model


def evaluate_model(model, data_loader, criterion, label_encoder, save_dir, dataset_name="Test set Data"):
    """
    Evaluate model with comprehensive metrics using focal loss.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 1613-1753

    Args:
        model: PyTorch model
        data_loader: Data loader
        criterion: Loss function (not used, we use focal_loss)
        label_encoder: Label encoder
        save_dir: Directory to save plots
        dataset_name: Name of dataset

    Returns:
        tuple: (accuracy, test_loss, f1, precision, recall, roc_auc, balanced_acc, mcc, kappa, cm_plot_path, y_pred_proba, report_df)
    """
    device = next(model.parameters()).device  # Get device from model

    print(f"###############{dataset_name}##############")
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    y_pred_proba = []


    # Extract benchmark labels from dataloader
    all_targets = []
    with torch.no_grad():
        for _, target in data_loader:
            all_targets.extend(target.cpu().numpy())

    # Convert to NumPy array
    benchmark_labels = np.array(all_targets)

    # ✅ Compute class distribution safely
    unique_classes, class_counts = np.unique(benchmark_labels, return_counts=True)
    class_weights = class_counts / np.sum(class_counts)

    print("🔹 Estimated Class Distribution from Benchmark:", class_weights)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())

            # ✅ Step 3: Apply Focal Loss for better class balance
            loss = focal_loss(output, target)
            test_loss += loss.item()

            # ✅ Get predictions
            probs = torch.nn.functional.softmax(output, dim=1)
            preds = probs.argmax(dim=1, keepdim=True)

            correct += preds.eq(target.view_as(preds)).sum().item()

            # Append results
            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_pred_proba.extend(probs.cpu().numpy())

    test_loss /= len(data_loader)
    accuracy = 100. * correct / len(data_loader.dataset)

    print(f"accuracy accuracy = 100. * correct / len(data_loader.dataset) , {accuracy}")

    # Convert lists to tensors
    y_true_tensor = torch.tensor(y_true).to(device)
    y_pred_tensor = torch.tensor(y_pred).squeeze().to(device)
    y_pred_proba_tensor = torch.tensor(y_pred_proba).to(device)

    # 🔍 Check how many cells were predicted for each label
    predicted_class_labels = label_encoder.inverse_transform(y_pred_tensor.cpu().numpy())
    pred_counts = Counter(predicted_class_labels)
    print(f"🔍 Predicted label counts: {pred_counts}")


    num_classes = len(torch.unique(y_true_tensor))
    # Calculate metrics using torchmetrics
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted').to(device)
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='weighted').to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='weighted').to(device)
    auroc_metric = torchmetrics.AUROC(task="multiclass", num_classes=num_classes, average='weighted').to(device)

    # Calculate metrics
    f1 = f1_metric(y_pred_tensor, y_true_tensor)
    precision = precision_metric(y_pred_tensor, y_true_tensor)
    recall = recall_metric(y_pred_tensor, y_true_tensor)

    if y_pred_proba_tensor.shape[1] == num_classes:
        roc_auc = auroc_metric(y_pred_proba_tensor, y_true_tensor)
    else:
        roc_auc = torch.tensor(float('nan')).to(device)

    # Calculate balanced accuracy, MCC, and Cohen's kappa using scikit-learn
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Generate and display confusion matrix
    cm_plot_path = os.path.join(save_dir, f"{dataset_name}_confusion_matrix.png")
    labels = label_encoder.transform(label_encoder.classes_)  # [0, 1, 2] if 3 classes
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"label_encoder.classes_: {label_encoder.classes_}")

    # Convert integer labels to string labels before generating report
    y_true_str = label_encoder.inverse_transform(np.array(y_true).astype(int))
    y_pred_str = label_encoder.inverse_transform(np.array(y_pred).astype(int))

    print(f"y_true_str (before inverse): {y_true_str[:5]}")
    print(f"y_pred_str (before inverse): {y_pred_str[:5]}")
    print(f"✅ Unique labels in y_true_str: {np.unique(y_true_str)}")
    print(f"✅ Unique labels in y_pred_str: {np.unique(y_pred_str)}")



    report = classification_report(y_true_str, y_pred_str, labels=label_encoder.classes_, output_dict=True, zero_division=0)

    report_df = pd.DataFrame(report)



    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)  # Adjust class names if needed
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.savefig(cm_plot_path)
    plt.close()


    return accuracy, test_loss, f1.cpu().item(), precision.cpu().item(), recall.cpu().item(), roc_auc.cpu().item(), balanced_acc, mcc, kappa, cm_plot_path, np.array(y_pred_proba), report_df


def initialize_optimizer(model, optimizer_name, learning_rate):
    """
    Initialize optimizer with weight decay.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 1755-1770

    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('Adam', 'RMSprop', 'SGD', 'AdamW')
        learning_rate: Learning rate

    Returns:
        torch.optim.Optimizer: Initialized optimizer
    """
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer


def init_weights(m):
    """
    Aurélien suggests Xavier (Glorot) or Kaiming (He) initialization.
    This function checks if 'm' is a Linear or Conv layer, then applies the init.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 1478-1493

    Used with model.apply(init_weights) for proper weight initialization.
    """
    if isinstance(m, nn.Linear):
        # Kaiming (He) uniform initialization for ReLU activation
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        # If you have conv layers
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ============================================================================
# TRADITIONAL ML EVALUATION (Non-Neural)
# ============================================================================

def evaluate_model_non_neural(model, X, y, label_encoder, save_dir, dataset_name="Dataset"):
    """
    Evaluate a scikit-learn model on X,y. Prints multiple metrics, returns them.
    This function does not do any data loading/scaling. We assume X,y are aligned.

    EXACT implementation from 2_0_principle_aurelien_ml_traditional.py line 428-503

    Args:
        model: A scikit-learn classifier (fit already).
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.ndarray): True labels.
        label_encoder: sklearn LabelEncoder object
        save_dir (str): Directory to save confusion matrix plot
        dataset_name (str): For printing logs.

    Returns:
        A tuple of (accuracy, f1, precision, recall, roc_auc, balanced_acc, mcc, kappa, cm_plot_path, report_df).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
        cohen_kappa_score, classification_report, confusion_matrix,
        ConfusionMatrixDisplay
    )
    import os

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average='weighted')
    prec= precision_score(y, y_pred, average='weighted')
    rec = recall_score(y, y_pred, average='weighted')

    # Convert integer labels to string labels before generating report
    y_true_str = label_encoder.inverse_transform(np.array(y).astype(int))
    y_pred_str = label_encoder.inverse_transform(np.array(y_pred).astype(int))
    report = classification_report(y_true_str, y_pred_str, labels=label_encoder.classes_, output_dict=True, zero_division=0)

    report_df = pd.DataFrame(report)

    # Class-wise Accuracy and MCC
    classwise_accuracy = {}
    classwise_mcc = {}
    for idx, cls in enumerate(label_encoder.classes_):
        mask = (np.array(y) == idx)
        class_acc = accuracy_score(np.array(y)[mask], np.array(y_pred)[mask])
        bin_true = (np.array(y) == idx).astype(int)
        bin_pred = (np.array(y_pred) == idx).astype(int)
        class_mcc = matthews_corrcoef(bin_true, bin_pred)
        classwise_accuracy[cls] = class_acc
        classwise_mcc[cls] = class_mcc

    accuracy_df = pd.DataFrame({"Accuracy": classwise_accuracy, "MCC": classwise_mcc})
    accuracy_df_T = accuracy_df.T
    report_df = pd.concat([report_df, accuracy_df_T], axis=0)
    report_df = report_df * 100.0
    print(f"report_df  {report_df}")

    cm_plot_path = os.path.join(save_dir, f"{dataset_name}_confusion_matrix.png")
    cm_test = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title(f"Confusion Matrix {dataset_name}")
    plt.savefig(cm_plot_path)
    plt.show()
    plt.close()

    # If it's multi-class, we do 'ovr'
    if len(np.unique(y)) > 2:
        roc_val = roc_auc_score(y, y_proba, multi_class='ovr')
    else:
        # binary classification
        roc_val = roc_auc_score(y, y_proba[:,1])

    bal_acc = balanced_accuracy_score(y, y_pred)
    mcc  = matthews_corrcoef(y, y_pred)
    kappa= cohen_kappa_score(y, y_pred)

    return (acc*100.0, f1*100.0, prec*100.0, rec*100.0,
            roc_val*100.0, bal_acc*100.0, mcc*100.0, kappa*100.0, cm_plot_path, report_df)
