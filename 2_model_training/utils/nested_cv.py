"""
Nested Cross-Validation for Deep Learning Models
=================================================

THE MAIN TRAINING FUNCTION - perform_nested_cv_dn

This performs nested 5-fold cross-validation with Optuna hyperparameter optimization.

EXACT implementation from 1_0_principle_aurelien_ml.py line 2171-2570

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import psutil
import joblib

from .data_utils import load_and_preprocess_data
from .optuna_utils import optimize_model_with_optuna
from .training_utils import train_model, evaluate_model
from .visualization import plot_training_history, plot_validation_accuracy
from .io_utils import save_scaler, save_fold_to_csv
from sklearn.model_selection import train_test_split


def perform_nested_cv_dn(
    model_type="simpledense",
    reh_or_sup="reh",
    save_model_here=None,
    selection_method=None,
    scaling_method="standard",
    n_trials=2,
    cv=5
):
    """
    Performs nested cross-validation for PyTorch models.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 2171-2570

    This is THE MAIN TRAINING FUNCTION that:
    1. Performs K-fold cross-validation
    2. Optimizes hyperparameters with Optuna for each fold
    3. Trains the model with best hyperparameters
    4. Evaluates on test set
    5. Saves models, parameters, scalers, and results

    Args:
        model_type (str): Type of model to use ('simpledense', 'deepdense', 'cnn', 'hbdcnn', 'fe').
                          Defaults to 'simpledense'.
        reh_or_sup (str): Specifies whether to use REH or SUP data for training. Defaults to 'reh'.
        save_model_here (str): Path to the directory where models should be saved.
        selection_method (str): Feature selection method (None, 'SelectKBest', 'ElasticCV'). Defaults to None.
        scaling_method (str): Scaling method ('standard', 'minmax', 'robust'). Defaults to 'standard'.
        n_trials (int): Number of Optuna trials for hyperparameter optimization. Defaults to 2.
        cv (int): Number of cross-validation folds. Defaults to 5.

    Note:
        Training uses fixed max_epochs=1500 with early_stopping_patience=100.
        Optuna optimizes learning_rate, optimizer, and model architecture parameters.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_model_here, exist_ok=True)

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model type constants
    Select_None_ft = None
    vae_pretrain = "vaeprt"
    DANN = "DANN"
    MSDA = "MSDA"

    # Outer CV Loop (5 folds)
    how_many_fold = cv
    model_type_string = model_type
    train_scores = []
    train_losses = []
    test_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    roc_auc_scores = []
    balanced_accuracy_scores = []
    mcc_scores = []
    kappa_scores = []
    gse146773_accuracies = []
    gse64016_accuracies = []

    for outer_fold in range(how_many_fold):
        print(f"\n{'='*80}")
        print(f"Outer Fold: {outer_fold + 1}/{how_many_fold}")
        print(f"{'='*80}\n")

        # Load and preprocess data
        X_train, X_test, y_train_encoded, y_test, cell_ids_test, scaler, label_encoder = load_and_preprocess_data(
            scaling_method=scaling_method, is_reh=(reh_or_sup == "reh"), selection_method=selection_method
        )

        input_dim = X_train.shape[1]
        # Encode the labels
        num_classes = len(set(y_train_encoded))

        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        # Lists to store scores for plotting
        best_model = None
        best_params = None

        # Apply to all inner folds
        selected_features = sorted(set(X_train.columns))

        # Apply selected features to train/test datasets
        X_train = X_train[selected_features].copy()
        X_test = X_test[selected_features].copy()

        # Apply selected features
        X_train = X_train[selected_features].copy()
        X_test = X_test[selected_features].copy()

        # Create inner train/validation split for Optuna hyperparameter optimization
        X_train_inner, X_val, y_train_inner_encoded, y_val_encoded = train_test_split(
            X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded, shuffle=True
        )

        # Convert data to tensors
        X_train_inner_tensor = torch.tensor(X_train_inner.values, dtype=torch.float32).to(device)
        y_train_inner_tensor = torch.tensor(y_train_inner_encoded, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long).to(device)

        # Create DataLoaders for inner training and validation
        train_inner_loader = DataLoader(TensorDataset(X_train_inner_tensor, y_train_inner_tensor), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

        # Encode the labels for full training set
        y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(label_encoder.transform(y_test), dtype=torch.long).to(device)

        # Convert training and test sets to tensors (full fold)
        X_train_resampled_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        # Create DataLoader for training
        train_loader = DataLoader(TensorDataset(X_train_resampled_tensor, y_train_tensor), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        # Initialize the model
        if model_type not in [vae_pretrain, DANN, MSDA]:
            print(f"\nOptimizing hyperparameters with Optuna ({n_trials} trials)...")
            best_model, optimizer, best_params = optimize_model_with_optuna(
                model_type=model_type,
                input_dim=input_dim,
                num_classes=num_classes,
                train_loader=train_inner_loader,
                val_loader=val_loader,
                device=device,
                n_trials=n_trials
            )

            best_model.to(device)
            use_scheduler = True  # If you want scheduling
            step_size = 30
            gamma = 0.1

            # ========== Track training time in the outer fold ==========
            start_time = time.time()
            # Start tracking GPU memory
            start_mem_gpu = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
            peak_mem_gpu = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
            start_mem_cpu = psutil.virtual_memory().used / 1e9  # in GB

            print(f"\nTraining final model with best hyperparameters (max {best_params['epochs']} epochs, early stopping patience=100)...")
            train_loss, train_score, best_model = train_model(
                best_model, train_loader, optimizer, criterion, epochs=best_params['epochs'], log_dir=save_model_here,
                early_stopping_patience=100, trial=None, use_lr_scheduler=use_scheduler, step_size=step_size, gamma=gamma
            )
            train_scores.append(train_score)
            train_losses.append(train_loss)
            # train_loss, train_score come from train_model
            train_plot_path = plot_training_history(train_loss, train_score, save_model_here)
            end_time = time.time()
            training_time = end_time - start_time
            # End tracking memory usage
            end_mem_gpu = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
            peak_mem_gpu = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
            end_mem_cpu = psutil.virtual_memory().used / 1e9

            # Compute memory used
            gpu_memory_used = end_mem_gpu - start_mem_gpu
            cpu_memory_used = end_mem_cpu - start_mem_cpu
            print(f"[Outer Fold {outer_fold+1}] Training time = {training_time:.2f} seconds")
            print(f"GPU Memory Used: {gpu_memory_used:.2f} GB, Peak: {peak_mem_gpu:.2f} GB")
            print(f"CPU Memory Used: {cpu_memory_used:.2f} GB")

            # Evaluate the model on the test set
            print(f"\nEvaluating on test set...")
            test_accuracy, test_loss, test_f1, test_precision, test_recall, test_roc_auc, balanced_acc, mcc, kappa, cm_plot_path, _, _ = evaluate_model(
                best_model, test_loader, criterion, label_encoder, save_model_here, dataset_name="Test Set"
            )
            test_scores.append(test_accuracy)

        else:
            # VAE, DANN, MSDA not implemented yet
            raise NotImplementedError(f"Model type {model_type} not implemented yet. Use: simpledense, deepdense, cnn, hbdcnn, fe")

        # Create a string representing the hyperparameters
        params_str = "_".join([f"{k}={v}" for k, v in best_params.items()])

        has_used_feature_sl = "NFT" if selection_method is None else "YFT"
        # Save the trained model with hyperparameters in the filename
        prefix_name = f"{model_type}_{has_used_feature_sl}_{reh_or_sup}_fld_{outer_fold + 1}"
        model_filename = f"{prefix_name}.pt"
        model_filepath = os.path.join(save_model_here, model_filename)
        torch.save(best_model.state_dict(), model_filepath)  # Save the model's state_dict
        print(f"✅ Saved model to: {model_filepath}")

        params_filename = f"{prefix_name}_params.txt"
        params_filepath = os.path.join(save_model_here, params_filename)

        indetails = f"{prefix_name}_{params_str}"

        with open(params_filepath, 'w') as f:
            f.write("".join(indetails))

        # Save the features used for training
        features_filename = f"{prefix_name}_features.txt"
        features_filepath = os.path.join(save_model_here, features_filename)

        with open(features_filepath, 'w') as f:
            f.write("\n".join(selected_features))

        # Save the features used for training
        features_sl_method_filename = f"{prefix_name}_ft_method.txt"
        features_sl_method_filename_filepath = os.path.join(save_model_here, features_sl_method_filename)

        SCALER_SAVE_PATH = f"{save_model_here}/{prefix_name}_{scaling_method}_scl.pkl"
        # Save scaler for later use
        save_scaler(scaler, SCALER_SAVE_PATH)

        label_encoder_filepath = os.path.join(save_model_here, f"{prefix_name}_le.pkl")
        joblib.dump(label_encoder, label_encoder_filepath)

        if selection_method == None:
            selection_method = None
        with open(features_sl_method_filename_filepath, 'w') as f:
            if selection_method is None:
                f.write("None!")
            else:
                f.write(selection_method)

        train_plot_filename = os.path.join(save_model_here, f"{prefix_name}_training_plot.png")
        cm_plot_filename = os.path.join(save_model_here, f"{prefix_name}_confusion_matrix.png")
        os.rename(train_plot_path, train_plot_filename)
        os.rename(cm_plot_path, cm_plot_filename)

        # Loop through all files in the directory and delete TensorBoard event files
        directory = save_model_here
        for filename in os.listdir(directory):
            # Check if the filename starts with 'events.out.'
            if filename.startswith("events.out."):
                # Construct the full file path
                file_path = os.path.join(directory, filename)

                # Delete the file
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted TensorBoard file: {file_path}")

        # Save fold results to CSV
        save_fold_to_csv(
            prefix_name=prefix_name,
            best_params=best_params,
            training_time=training_time,
            gpu_memory_used=gpu_memory_used,
            cpu_memory_used=cpu_memory_used,
            train_scores=train_scores,          # or final epoch only
            train_losses=train_losses,
            test_accuracy=test_accuracy,
            test_loss=test_loss,
            test_f1=test_f1,
            test_precision=test_precision,
            test_recall=test_recall,
            test_roc_auc=test_roc_auc,
            test_balanced_acc=balanced_acc,
            test_mcc=mcc,
            test_kappa=kappa,
            save_model_here=save_model_here
        )

    print(f"\n{'='*80}")
    print("CROSS-VALIDATION COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Test scores across folds: {test_scores}")

    # Plot validation accuracy across folds
    validation_plot_filename = plot_validation_accuracy(
        test_scores=test_scores, outer_fold=how_many_fold, save_dir=save_model_here, title=f"{model_type} accuracy per fold"
    )
    updated_plot_filename = os.path.join(save_model_here, f"{model_type}_per_fold_plot.png")
    os.rename(validation_plot_filename, updated_plot_filename)

    print(f"✅ All results saved to: {save_model_here}")
    print(f"✅ Mean test accuracy: {sum(test_scores)/len(test_scores):.2f}%")


# ============================================================================
# TRADITIONAL ML NESTED CV (Non-Neural Networks)
# ============================================================================

def perform_nested_cv_non_neural(
    model_type="adaboost",
    reh_or_sup="reh",
    save_model_here=None,
    selection_method=None,   # e.g. "SelectKBest", "ElasticNetCV", or None
    scaling_method="standard",
    n_trials=20,              # How many Optuna trials
    outer_splits=5          # Outer folds
):
    """
    Aurélien Géron's recommended nested CV approach for 4 model types:
      - 'adaboost' => Optuna optimization
      - 'random_forest' => Optuna optimization
      - 'lgbm' => Optuna optimization
      - 'ensemble' => separate searches for each (Ada, RF, LGBM), then VotingClassifier

    EXACT implementation from 2_0_principle_aurelien_ml_traditional.py line 689-846

    * Outer loop (5 folds) => train/test splits
    * Inner loop => Optuna optimization on inner train/val split
    * Print confusion matrix & classification report for each outer test set
    * Save final model & artifacts

    Parameters:
    -----------
    model_type : str
        Model type: 'adaboost', 'random_forest', 'lgbm', or 'ensemble'
    reh_or_sup : str
        Dataset: 'reh' or 'sup'
    save_model_here : str
        Directory to save models and results
    selection_method : str or None
        Feature selection method: 'SelectKBest', 'ElasticCV', or None
    scaling_method : str
        Scaling method: 'standard', 'minmax', or 'robust'
    n_trials : int
        Number of Optuna trials for hyperparameter optimization
    outer_splits : int
        Number of outer cross-validation folds
    """
    import os
    import time
    import psutil
    import torch
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from .data_utils import load_and_preprocess_data
    from .io_utils import save_scaler, save_fold_to_csv_tml
    from .training_utils import evaluate_model_non_neural
    from .visualization import plot_validation_accuracy
    from .optuna_tml import optimize_traditional_model, optimize_ensemble

    if save_model_here is None:
        save_model_here = os.getcwd()
    os.makedirs(save_model_here, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_scores = []

    for outer_fold in range(outer_splits):
        print(f"\n=== [Outer Fold {outer_fold+1}/{outer_splits}] ===")

        # (A) Load data for this outer fold
        X_train, X_test, y_train, y_test, cell_ids_test, scaler, label_encoder = load_and_preprocess_data(
            scaling_method=scaling_method,
            is_reh=(reh_or_sup == "reh"),
            selection_method=selection_method
        )

        # (B) Align features
        selected_features = sorted(set(X_train.columns))
        X_train = X_train[selected_features].copy()
        X_test  = X_test[selected_features].copy()

        # (C) Encode labels
        y_test = label_encoder.transform(y_test)

        # (D) Split into train/validation for Optuna
        X_train_inner, X_val, y_train_inner, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        start_mem_cpu = psutil.virtual_memory().used / 1e9  # Initial memory in GB
        start_mem_gpu = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
        peak_mem_gpu = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
        start_time = time.time()  # Start timing

        # (E) Optimize the selected model
        if model_type in ["adaboost", "random_forest", "lgbm"]:
            best_model, best_params = optimize_traditional_model(
                X_train, y_train, X_train_inner, y_train_inner, X_val, y_val, model_type, n_trials
            )
        elif model_type == "ensemble":
            best_model, best_params = optimize_ensemble(
                X_train, y_train, X_train_inner, y_train_inner, X_val, y_val, n_trials
            )
        else:
            raise ValueError("Invalid model type.")

        # Evaluate on outer test
        print("[Outer Fold] Evaluate on test set =>")

        end_time = time.time()  # End timing
        end_mem_cpu = psutil.virtual_memory().used / 1e9  # Final memory in GB
        memory_used = end_mem_cpu - start_mem_cpu
        training_time = end_time - start_time

        end_mem_gpu = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
        gpu_memory_used = end_mem_gpu - start_mem_gpu

        # Note: evaluate_model_non_neural returns 10 items, but we only unpack 9 (ignoring report_df)
        acc, f1, prec, rec, roc_val, bal_acc, mcc, kappa, cm_plot_path = evaluate_model_non_neural(
            best_model, X_test, y_test, label_encoder, save_model_here, dataset_name="Test Set"
        )[:9]  # Only take first 9 items

        test_scores.append(acc)

        # (G) Save best model, hyperparams, etc.
        if isinstance(best_params, dict):
            params_str = "_".join([f"{k}={v}" for (k, v) in best_params.items()])
        else:
            params_str = str(best_params)

        has_used_feature_sl = "NFT" if selection_method is None else "YFT"

        # Save the trained model with hyperparameters in the filename
        prefix_name = f"{model_type}_{has_used_feature_sl}_{reh_or_sup}_fld_{outer_fold + 1}"
        model_filename = f"{prefix_name}.joblib"
        model_filepath = os.path.join(save_model_here, model_filename)
        joblib.dump(best_model, model_filepath)

        X_train_filename = os.path.join(save_model_here, f"{prefix_name}_X_train.csv")
        X_train.to_csv(X_train_filename, index=False)

        params_filename = f"{prefix_name}_params.txt"
        params_filepath = os.path.join(save_model_here, params_filename)

        indetails = f"{prefix_name}_{params_str}"

        with open(params_filepath, 'w') as f:
            f.write("".join(indetails))

        # Save the features used for training
        features_filename = f"{prefix_name}_features.txt"
        features_filepath = os.path.join(save_model_here, features_filename)

        with open(features_filepath, 'w') as f:
            f.write("\n".join(selected_features))

        # Save the features used for training
        features_sl_method_filename = f"{prefix_name}_ft_method.txt"
        features_sl_method_filename_filepath = os.path.join(save_model_here, features_sl_method_filename)

        SCALER_SAVE_PATH = f"{save_model_here}/{prefix_name}_{scaling_method}_scl.pkl"
        # Save scaler for later use
        save_scaler(scaler, SCALER_SAVE_PATH)

        label_encoder_filepath = os.path.join(save_model_here, f"{prefix_name}_le.pkl")
        joblib.dump(label_encoder, label_encoder_filepath)

        if selection_method == None:
            selection_method = None
        with open(features_sl_method_filename_filepath, 'w') as f:
            if selection_method is None:
                f.write("None!")
            else:
                f.write(selection_method)

        cm_plot_filename = os.path.join(save_model_here, f"{prefix_name}_confusion_matrix.png")
        os.rename(cm_plot_path, cm_plot_filename)

        save_fold_to_csv_tml(
            prefix_name=prefix_name,
            best_params=best_params,
            training_time=training_time,
            cpu_memory_used=memory_used,
            gpu_memory_used=gpu_memory_used,
            test_accuracy=acc,
            test_f1=f1,
            test_precision=prec,
            test_recall=rec,
            test_roc_auc=roc_val,
            test_balanced_acc=bal_acc,
            test_mcc=mcc,
            test_kappa=kappa,
            save_model_here=save_model_here
        )

    validation_plot_filename = plot_validation_accuracy(
        test_scores=test_scores, outer_fold=outer_splits, save_dir=save_model_here, title=f"{model_type} accuracy per fold"
    )
    updated_plot_filename = os.path.join(save_model_here, f"{model_type}_per_fold_plot.png")
    os.rename(validation_plot_filename, updated_plot_filename)

    print("\nAll outer folds complete for non-neural models!")
    print(f"✅ All results saved to: {save_model_here}")
    print(f"✅ Mean test accuracy: {sum(test_scores)/len(test_scores):.2f}%")
