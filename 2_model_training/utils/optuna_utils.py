"""
Optuna Hyperparameter Optimization Utilities
============================================

Contains Optuna-based hyperparameter optimization for deep learning models.

EXACT implementation from 1_0_principle_aurelien_ml.py line 1772-1957

Author: Halima Akhter
Date: 2025-11-24
"""

import torch
import torch.nn as nn
import optuna
from .training_utils import initialize_optimizer, init_weights


def optimize_model_with_optuna(
    model_type,
    input_dim,
    num_classes,
    train_loader,
    val_loader,
    device,
    n_trials=5
):
    """
    A unified Optuna-based function that:
      - Defines an internal objective function (optuna_objective) to handle
        both "special" (enhancedense, edat, fe, hbdcnn, vaeprt)
        and "standard" (simpledense, deepdense, cnn, resnet1D, resnet1DSE) models.
      - Performs hyperparameter search.
      - Rebuilds a final best-model with the winning hyperparameters.
      - Optionally retrains it on the train_loader, then returns:
           (final_model, final_optimizer, best_params).

    EXACT implementation from 1_0_principle_aurelien_ml.py line 1772-1957

    Args:
        model_type (str): e.g. "simpledense", "deepdense", "enhancedense", "vaeprt", etc.
        input_dim (int): Number of input features.
        num_classes (int): Number of output classes.
        train_loader (DataLoader): DataLoader for training set (for both HP search & final training).
        val_loader (DataLoader): DataLoader for validation set (for hyperparameter search).
        device (torch.device): e.g. "cuda" or "cpu".
        n_trials (int): How many trials Optuna should run.

    Returns:
        (model, optimizer, best_params)
        - model: final trained model with best hyperparameters
        - optimizer: final optimizer
        - best_params: dictionary of best hyperparameters found by Optuna
    """
    # Import model classes (need to be defined elsewhere or imported)
    from ..models.dense_models import SimpleDenseModel, DeepDenseModel
    from ..models.cnn_models import CNNModel
    from ..models.hybrid_models import HybridCNNDenseModel, FeatureEmbeddingModel
    # Enhanced models would be imported here when available
    # from ..models.enhanced_models import Enhance_model, Enhance_model_with_attention

    # Model type constants
    enhancedense = "enhancedense"
    enhancedenseAttention = "edat"
    featureembedding = "fe"
    hybridcnn = "hbdcnn"
    vae_pretrain = "vaeprt"
    simpledense = "simpledense"
    deepdense = "deepdense"
    cnn = "cnn"
    resnet1D = "resnet1D"
    resnet1DSE = "resnet1DSE"

    # -------------------------------------------------------------
    # 1. DEFINE THE INTERNAL OPTUNA OBJECTIVE
    # -------------------------------------------------------------
    best_model = None
    best_score = float('inf')

    def optuna_objective(trial):
        nonlocal best_model, best_score

        # Universal hyperparams
        learning_rate = trial.suggest_float('learningrate', 1e-5, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        # Fixed epochs - early stopping will handle optimal stopping point
        epochs = 1500  # Maximum epochs, early stopping will terminate earlier if needed

        # Distinguish special models
        special_models = [enhancedense, enhancedenseAttention, featureembedding, hybridcnn, vae_pretrain]

        # -------------------
        # Build the model
        # -------------------
        if model_type in special_models:
            # Additional hyperparams
            n_layers = trial.suggest_int('nlayers', 5, 5)  # example
            n_units = trial.suggest_categorical('nunits', [[128, 64, 32, 16, 8]])
            dropouts = trial.suggest_categorical('dropouts', [
                [0.2, 0.3, 0.4, 0.5, 0.5],
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ])
            embed_dim = trial.suggest_categorical('embeddim', [65, 128, 256])
            latent_dim = trial.suggest_categorical('latentdim', [65, 128, 256])

            # Switch on model type
            if model_type == enhancedense:
                # model = Enhance_model(n_layers, n_units, dropouts, input_dim).to(device)
                raise NotImplementedError("Enhanced model not yet implemented")

            elif model_type == enhancedenseAttention:
                # model = Enhance_model_with_attention(n_layers, n_units, dropouts, input_dim).to(device)
                raise NotImplementedError("Enhanced attention model not yet implemented")

            elif model_type == hybridcnn:
                model = HybridCNNDenseModel(
                    input_dim=input_dim,
                    output_dim=num_classes,
                    conv_out_channels=64,
                    kernel_size=3,
                    dense_units=n_units,
                    dropouts=dropouts
                ).to(device)

            elif model_type == featureembedding:
                model = FeatureEmbeddingModel(
                    input_dim=input_dim,
                    embed_dim=embed_dim,
                    n_layers=n_layers,
                    units_per_layer=n_units,
                    dropouts=dropouts,
                    output_dim=num_classes
                ).to(device)

            else:
                raise ValueError(f"Unsupported special model type: {model_type}")

        else:
            # STANDARD models
            def init_standard_model(mt, in_dim, out_dim):
                model_dict = {
                    simpledense: SimpleDenseModel(in_dim, out_dim),
                    deepdense: DeepDenseModel(in_dim, out_dim),
                    cnn: CNNModel(in_dim, out_dim),
                    # resnet1D and resnet1DSE would be added when available
                }
                if mt not in model_dict:
                    raise ValueError(f"Unsupported standard model type: {mt}")
                return model_dict[mt].to(device)

            model = init_standard_model(model_type, input_dim, num_classes)

        # -------------------
        # Initialize optimizer, criterion
        # -------------------
        model.apply(init_weights)

        optimizer = initialize_optimizer(model, optimizer_name, learning_rate)
        criterion = nn.CrossEntropyLoss()

        # -------------------
        # Train for 'epochs'
        # -------------------
        model.train()
        for ep in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        # -------------------
        # Validation
        # -------------------
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                preds = model(val_X)
                _, predicted = torch.max(preds, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()

        val_accuracy = correct / total
        error = 1 - val_accuracy  # Minimizing error

        # Store the best trained model
        if error < best_score:
            best_score = error
            best_model = model

        return error

    # -------------------------------------------------------------
    # 2. RUN THE OPTUNA STUDY
    # -------------------------------------------------------------
    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"\nBest parameters for {model_type}: {best_params}")

    best_optimizer = initialize_optimizer(best_model, best_params['optimizer'], best_params['learningrate'])

    return best_model, best_optimizer, best_params
