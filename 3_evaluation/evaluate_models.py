"""
Model Evaluation Script
========================

Evaluate trained models on benchmark datasets (GSE146773, GSE64016) with
ground truth FUCCI labels.

Usage:
    python evaluate_models.py --model-dir ../models/saved_models/dnn3_experiment --benchmark gse146773
    python evaluate_models.py --model-path ../models/saved_models/dnn3_final.pt --model-type dnn3 --benchmark gse64016

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models and utilities
from models import (
    SimpleDenseModel, DeepDenseModel, CNNModel,
    HybridCNNDenseModel, FeatureEmbeddingModel
)
from utils import focal_loss, evaluate_model


def load_model_from_checkpoint(model_path, model_type, input_dim, num_classes, device):
    """
    Load trained model from checkpoint.

    Parameters:
    -----------
    model_path : str
        Path to model checkpoint (.pt file)
    model_type : str
        Model type
    input_dim : int
        Input dimension
    num_classes : int
        Number of classes
    device : torch.device
        Device to load model on

    Returns:
    --------
    nn.Module
        Loaded model
    """
    # Create model
    if model_type == 'dnn3':
        model = SimpleDenseModel(input_dim=input_dim, num_classes=num_classes)
    elif model_type == 'dnn5':
        model = DeepDenseModel(input_dim=input_dim, num_classes=num_classes)
    elif model_type == 'cnn':
        model = CNNModel(input_dim=input_dim, num_classes=num_classes)
    elif model_type == 'hybrid':
        model = HybridCNNDenseModel(input_dim=input_dim, output_dim=num_classes)
    elif model_type == 'feature_embedding':
        model = FeatureEmbeddingModel(
            input_dim=input_dim,
            embed_dim=256,
            n_layers=3,
            units_per_layer=[128, 64, 32],
            dropouts=[0.5, 0.4, 0.3],
            output_dim=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def load_benchmark_data(benchmark_name, data_dir):
    """
    Load benchmark dataset with ground truth labels.

    Parameters:
    -----------
    benchmark_name : str
        Benchmark dataset name: gse146773 or gse64016
    data_dir : str
        Data directory

    Returns:
    --------
    tuple
        (X_data, y_true, cell_ids)
    """
    print(f"\nLoading {benchmark_name.upper()} benchmark data...")

    # Define paths
    if benchmark_name.lower() == 'gse146773':
        data_file = 'GSE146773_seurat_normalized_gene_expression.csv'
        fucci_file = 'GSE146773_fucci_coords.csv'
    elif benchmark_name.lower() == 'gse64016':
        data_file = 'GSE64016_seurat_normalized_gene_expression.csv'
        fucci_file = 'sc1_GSE64016_original_CellCycleOrder.csv'
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Try to find data
    possible_dirs = [
        os.path.join(data_dir, 'Benchmark_data'),
        os.path.join('../data/Training_data/Benchmark_data'),
        os.path.join('../../data/Training_data/Benchmark_data')
    ]

    data_path = None
    for base_dir in possible_dirs:
        test_path = os.path.join(base_dir, data_file)
        if os.path.exists(test_path):
            data_path = test_path
            break

    if data_path is None:
        raise FileNotFoundError(f"Could not find {data_file}")

    print(f"  Loading from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"  Loaded: {data.shape}")

    # Load FUCCI ground truth labels
    # Note: Implementation depends on your specific file format
    # This is a placeholder - adjust based on your actual FUCCI label format

    return data, None  # Placeholder


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on benchmark datasets'
    )

    # Model loading
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--model-dir', type=str, help='Model directory (will use best_model.pt or *_final.pt)')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['dnn3', 'dnn5', 'cnn', 'hybrid', 'feature_embedding'])

    # Data
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=['gse146773', 'gse64016', 'both'])
    parser.add_argument('--data-dir', type=str, default='../data/Training_data')

    # Device
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpu', type=int, default=0)

    # Output
    parser.add_argument('--output-dir', type=str, default='../results/evaluation')

    args = parser.parse_args()

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    elif args.model_dir:
        # Look for best_model.pt or *_final.pt
        possible_paths = [
            os.path.join(args.model_dir, 'best_model.pt'),
            os.path.join(args.model_dir, f'{args.model_type}_final.pt')
        ]
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        if model_path is None:
            raise FileNotFoundError(f"No model found in {args.model_dir}")
    else:
        parser.error("Either --model-path or --model-dir must be provided")

    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Model Evaluation on Benchmarks")
    print("=" * 80)
    print(f"Model: {args.model_type.upper()}")
    print(f"Model path: {model_path}")
    print(f"Benchmark: {args.benchmark.upper()}")
    print(f"Device: {device}")
    print("=" * 80)

    # Note: Full implementation would load benchmark data,
    # preprocess it, create data loaders, and run evaluation
    # This is a template structure

    print("\n✅ Evaluation script template created!")
    print("📝 Note: Full benchmark evaluation requires FUCCI label integration")
    print("   See existing notebook: 4_ML_DL/0_0_principle_Aurelien_ML_evaluate.ipynb")


if __name__ == '__main__':
    main()
