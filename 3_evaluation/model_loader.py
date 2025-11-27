"""
Model Loading Utilities for Inference
======================================

Contains functions for loading trained models, scalers, and metadata for evaluation and inference.

EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import re
import glob
import joblib
import torch
import torch.nn as nn
import sys

# Add parent directory to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../2_model_training'))
from models.dense_models import SimpleDenseModel, DeepDenseModel
from models.cnn_models import CNNModel
from models.hybrid_models import HybridCNNDenseModel, FeatureEmbeddingModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_matching_file(directory, pattern, exclude=None):
    """
    Searches for a file in the directory that matches a given regex pattern.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 460-477

    Args:
        directory (str): The directory to search.
        pattern (str): The regex pattern to match filenames.
        exclude (str, optional): A substring to exclude from matching files.

    Returns:
        str or None: The full path of the matched file or None if not found.
    """
    for file in os.listdir(directory):
        if exclude and exclude in file:  # Exclude unwanted filenames
            continue
        if re.match(pattern, file):
            return os.path.join(directory, file)
    return None  # Return None if no match is found


def build_model(model_type, input_dim, hyperparams):
    """
    Dynamically builds the correct model based on model_type and hyperparameters.

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 569-607
    Supports 5 core deep learning models:
    - simpledense (DNN3)
    - deepdense (DNN5)
    - cnn
    - hbdcnn (Hybrid CNN+Dense)
    - fe (Feature Embedding)

    Args:
        model_type (str): Model type prefix
        input_dim (int): Number of input features
        hyperparams (dict): Hyperparameters parsed from params file

    Returns:
        torch.nn.Module: Instantiated model on device
    """
    num_classes = 3  # Assuming 3 cell cycle phases (G1, S, G2M)

    if model_type == "simpledense":
        return SimpleDenseModel(input_dim, num_classes).to(device)

    elif model_type == "deepdense":
        return DeepDenseModel(input_dim, num_classes).to(device)

    elif model_type == "cnn":
        return CNNModel(input_dim, num_classes).to(device)

    elif model_type.startswith("hbdcnn"):
        # HybridCNNDenseModel with hyperparams
        return HybridCNNDenseModel(
            input_dim,
            output_dim=num_classes,
            conv_out_channels=64,
            kernel_size=3,
            dense_units=eval(hyperparams.get("nunits", "[128, 64]")),
            dropouts=eval(hyperparams.get("dropouts", "[0.3, 0.3]"))
        ).to(device)

    elif model_type.startswith("fe"):
        # FeatureEmbeddingModel with hyperparams
        return FeatureEmbeddingModel(
            input_dim,
            embed_dim=int(hyperparams.get("embeddim", "128")),
            n_layers=int(hyperparams.get("nlayers", "3")),
            units_per_layer=eval(hyperparams.get("nunits", "[128, 64, 32]")),
            dropouts=eval(hyperparams.get("dropouts", "[0.3, 0.3, 0.3]")),
            output_dim=num_classes
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model_components(model_path):
    """
    Load a trained model, its scaler, label encoder, and selected features.

    Supports both Deep Learning (.pt) and Traditional ML (.joblib) models.

    Args:
        model_path (str): Full path to the model file (.pt or .joblib)

    Returns:
        tuple: (model, scaler, label_encoder, selected_features)
               For TML models, returns (model, scaler, label_encoder, selected_features)
               For DL models, returns (model, scaler, label_encoder, selected_features)
    """
    model_dir = os.path.dirname(model_path)

    # Determine model type from extension
    if model_path.endswith('.joblib'):
        # Traditional ML model
        model_prefix = os.path.basename(model_path).replace(".joblib", "")
        is_tml = True
    elif model_path.endswith('.pt'):
        # Deep Learning model
        model_prefix = os.path.basename(model_path).replace(".pt", "")
        is_tml = False
    else:
        raise ValueError(f"Unsupported model format: {model_path}. Use .pt or .joblib")

    # Locate required files
    scaler_path = os.path.join(model_dir, f"{model_prefix}_standard_scl.pkl")
    label_encoder_path = os.path.join(model_dir, f"{model_prefix}_le.pkl")
    features_path = os.path.join(model_dir, f"{model_prefix}_features.txt")
    params_path = os.path.join(model_dir, f"{model_prefix}_params.txt")

    # Load components
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    with open(features_path, "r") as f:
        selected_features = f.read().splitlines()

    # Parse hyperparameters
    hyperparams = {}
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            param_lines = f.read().strip().split("_")
            for param in param_lines:
                key_value = param.split("=")
                if len(key_value) == 2:
                    hyperparams[key_value[0]] = key_value[1]

    # Extract model type from prefix
    model_type = model_prefix.split("_")[0]
    print(f"Loading model type: {model_type}")
    print(f"Hyperparameters: {hyperparams}")

    if is_tml:
        # Load Traditional ML model (joblib)
        model = joblib.load(model_path)
        print(f"✓ Loaded TML model: {model_type}")
    else:
        # Build and load Deep Learning model
        model = build_model(model_type, len(selected_features), hyperparams)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"✓ Loaded DL model: {model_type}")

    return model, scaler, label_encoder, selected_features


def load_model_from_dir(model_dir):
    """
    Dynamically loads the trained model from a directory (finds .pt file automatically).

    EXACT implementation from 3_0_principle_aurelien_ml_evaluate.py line 479-565

    Args:
        model_dir (str): Path to the directory containing saved model files.

    Returns:
        tuple: (model, scaler, label_encoder, selected_features, model_prefix)
    """
    print(f"Searching for model files in {model_dir}...")

    # Find model file (exclude classifier/domain files)
    model_files = [
        file for file in glob.glob(os.path.join(model_dir, "*_fld_*.pt"))
        if not any(substring in file for substring in ["_clsf", "_label_cls", "_domain_clsf_finetune"])
    ]

    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_dir}")

    # Use first found model
    model_file = model_files[0]
    print(f"Found model: {os.path.basename(model_file)}")

    # Extract the full prefix (everything before .pt)
    model_prefix = os.path.basename(model_file).replace(".pt", "")

    # Find matching files
    params_path = find_matching_file(model_dir, rf"{model_prefix}_params\.txt")
    scaler_path = find_matching_file(model_dir, rf"{model_prefix}.*?_scl\.pkl")
    label_encoder_path = find_matching_file(model_dir, rf"{model_prefix}.*?_le\.pkl")
    features_path = find_matching_file(model_dir, rf"{model_prefix}.*?_features\.txt")

    # Load components
    scaler = joblib.load(scaler_path) if scaler_path else None
    label_encoder = joblib.load(label_encoder_path) if label_encoder_path else None

    # Load feature names
    with open(features_path, "r") as f:
        selected_features = f.read().splitlines()

    # Read hyperparameters from _params.txt
    hyperparams = {}
    if params_path:
        with open(params_path, "r") as f:
            param_lines = f.read().strip().split("_")
            for param in param_lines:
                key_value = param.split("=")
                if len(key_value) == 2:
                    hyperparams[key_value[0]] = key_value[1]

    # Extract model type from prefix
    model_type = model_prefix.split("_")[0]
    print(f"Model type: {model_type}")
    print(f"Hyperparameters: {hyperparams}")

    # Restore model architecture dynamically
    input_dim = len(selected_features)
    model = build_model(model_type, input_dim, hyperparams)

    # Load model weights
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model, scaler, label_encoder, selected_features, model_prefix
