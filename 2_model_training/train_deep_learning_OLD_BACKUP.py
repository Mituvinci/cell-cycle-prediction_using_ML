"""
Deep Learning Model Training Script with CLI
=============================================

Train deep learning models (DNN3, DNN5, CNN, Hybrid, FeatureEmbedding) for
cell cycle phase prediction with full command-line interface.

Usage:
    python train_deep_learning.py --model dnn3 --dataset reh --epochs 100
    python train_deep_learning.py --config configs/models/dnn3.yaml

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models and utilities
from models import (
    SimpleDenseModel, DeepDenseModel, CNNModel,
    HybridCNNDenseModel, FeatureEmbeddingModel
)
from utils import focal_loss, train_model, evaluate_model, preprocess_rna_data


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type, input_dim, num_classes, **kwargs):
    """
    Create model based on type.

    Parameters:
    -----------
    model_type : str
        Model type: dnn3, dnn5, cnn, hybrid, feature_embedding
    input_dim : int
        Number of input features
    num_classes : int
        Number of output classes
    **kwargs : dict
        Additional model-specific parameters

    Returns:
    --------
    nn.Module
        Created model
    """
    model_type = model_type.lower()

    if model_type == 'dnn3':
        model = SimpleDenseModel(input_dim=input_dim, num_classes=num_classes)

    elif model_type == 'dnn5':
        model = DeepDenseModel(input_dim=input_dim, num_classes=num_classes)

    elif model_type == 'cnn':
        model = CNNModel(input_dim=input_dim, num_classes=num_classes)

    elif model_type == 'hybrid':
        model = HybridCNNDenseModel(
            input_dim=input_dim,
            output_dim=num_classes,
            conv_out_channels=kwargs.get('conv_out_channels', 64),
            kernel_size=kwargs.get('kernel_size', 3),
            dense_units=kwargs.get('dense_units', [128, 64]),
            dropouts=kwargs.get('dropouts', [0.3, 0.3])
        )

    elif model_type == 'feature_embedding':
        model = FeatureEmbeddingModel(
            input_dim=input_dim,
            embed_dim=kwargs.get('embed_dim', 256),
            n_layers=kwargs.get('n_layers', 3),
            units_per_layer=kwargs.get('units_per_layer', [128, 64, 32]),
            dropouts=kwargs.get('dropouts', [0.5, 0.4, 0.3]),
            output_dim=num_classes
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def load_data(dataset_name, data_dir, scaling_method='standard', use_subset=False, subset_size=1000):
    """
    Load and preprocess training data.

    Parameters:
    -----------
    dataset_name : str
        Dataset name: reh, sup, or combined
    data_dir : str
        Data directory path
    scaling_method : str
        Scaling method
    use_subset : bool
        Whether to use subset for testing
    subset_size : int
        Size of subset

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, label_encoder)
    """
    print(f"\nLoading {dataset_name} data...")

    # Define data paths
    data_paths = {
        'reh': os.path.join(data_dir, 'filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv'),
        'sup': os.path.join(data_dir, 'filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv')
    }

    # Try to find data
    data_path = None
    if dataset_name.lower() in data_paths:
        possible_paths = [
            data_paths[dataset_name.lower()],
            os.path.join('../data', os.path.basename(data_paths[dataset_name.lower()])),
            os.path.join('../../data', os.path.basename(data_paths[dataset_name.lower()]))
        ]

        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break

    if data_path is None:
        raise FileNotFoundError(f"Could not find data for {dataset_name}")

    print(f"  Loading from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"  Loaded: {data.shape}")

    # Subset if requested
    if use_subset:
        data = data.sample(n=min(subset_size, len(data)), random_state=42)
        print(f"  Using subset: {data.shape}")

    # Preprocess
    print(f"  Preprocessing with {scaling_method} scaling...")
    X_train, X_test, y_train, y_test, _, scaler, label_encoder = preprocess_rna_data(
        data=data,
        scaling_method=scaling_method,
        selection_method=None,
        use_smote=True,
        use_undersampling=True,
        test_size=0.2,
        random_state=42
    )

    print(f"  ✓ Training samples: {X_train.shape[0]}")
    print(f"  ✓ Test samples: {X_test.shape[0]}")
    print(f"  ✓ Features: {X_train.shape[1]}")
    print(f"  ✓ Classes: {label_encoder.classes_}")

    return X_train, X_test, y_train, y_test, label_encoder, scaler


def main():
    parser = argparse.ArgumentParser(
        description='Train deep learning models for cell cycle prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train DNN3 on REH data
  python train_deep_learning.py --model dnn3 --dataset reh --epochs 100

  # Train with config file
  python train_deep_learning.py --config configs/models/dnn3.yaml

  # Quick test with subset
  python train_deep_learning.py --model dnn3 --dataset reh --epochs 10 --use-subset
        """
    )

    # Configuration
    parser.add_argument('--config', type=str, help='Path to YAML config file')

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['dnn3', 'dnn5', 'cnn', 'hybrid', 'feature_embedding'],
        help='Model type to train'
    )

    # Data
    parser.add_argument('--dataset', type=str, choices=['reh', 'sup', 'combined'], help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--scaling', type=str, default='standard', choices=['standard', 'minmax', 'robust'])
    parser.add_argument('--use-subset', action='store_true', help='Use subset for quick testing')
    parser.add_argument('--subset-size', type=int, default=1000, help='Subset size')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'cross_entropy'])

    # Loss function parameters
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')

    # Device
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    # Output
    parser.add_argument('--output-dir', type=str, default='../models/saved_models', help='Output directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load config file if provided (overrides CLI args)
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        # Update args from config
        if 'model' in config:
            args.model = config['model'].get('type', args.model)
        if 'training' in config:
            training_config = config['training']
            args.batch_size = training_config.get('batch_size', args.batch_size)
            args.epochs = training_config.get('epochs', args.epochs)
            args.lr = training_config.get('learning_rate', args.lr)
            args.early_stopping = training_config.get('early_stopping_patience', args.early_stopping)
            args.device = training_config.get('device', args.device)

            if 'loss' in training_config:
                loss_config = training_config['loss']
                args.loss = loss_config.get('type', args.loss)
                args.focal_alpha = loss_config.get('alpha', args.focal_alpha)
                args.focal_gamma = loss_config.get('gamma', args.focal_gamma)

    # Validate required arguments
    if not args.model:
        parser.error("--model is required (or provide --config)")
    if not args.dataset:
        parser.error("--dataset is required (or provide --config)")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    # Print banner
    print("=" * 80)
    print("Deep Learning Model Training")
    print("=" * 80)
    print(f"Model: {args.model.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss function: {args.loss}")
    print("=" * 80)

    # Load data
    X_train, X_test, y_train, y_test, label_encoder, scaler = load_data(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        scaling_method=args.scaling,
        use_subset=args.use_subset,
        subset_size=args.subset_size
    )

    # Create data loaders
    print("\nCreating data loaders...")
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
    y_test_tensor = torch.LongTensor(y_test.values if isinstance(y_test, pd.Series) else y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Test batches: {len(test_loader)}")

    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    model = create_model(args.model, input_dim, num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")

    # Setup training
    print("\nSetting up training...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == 'focal':
        criterion = lambda outputs, targets: focal_loss(outputs, targets, args.focal_alpha, args.focal_gamma)
        print(f"  ✓ Using Focal Loss (α={args.focal_alpha}, γ={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"  ✓ Using Cross Entropy Loss")

    # Create output directory
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{args.model}_{args.dataset}_{timestamp}"

    output_dir = os.path.join(args.output_dir, exp_name)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"  ✓ Output directory: {output_dir}")

    # Save configuration
    config_save = {
        'model': args.model,
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'loss': args.loss,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'total_params': total_params,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_save, f, indent=2)

    # Train model
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
        log_dir=log_dir,
        early_stopping_patience=args.early_stopping,
        device=device,
        trial=None,
        use_lr_scheduler=False
    )

    print("\n✓ Training complete!")

    # Evaluate model
    print("\n" + "=" * 80)
    print("Evaluating Model")
    print("=" * 80 + "\n")

    metrics = evaluate_model(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        label_encoder=label_encoder,
        save_dir=output_dir,
        device=device,
        dataset_name=f"{args.dataset}_test"
    )

    # Save final model
    model_path = os.path.join(output_dir, f'{args.model}_final.pt')
    torch.save(trained_model.state_dict(), model_path)
    print(f"\n✓ Saved final model: {model_path}")

    # Save metrics
    metrics_df = pd.DataFrame([{
        'model': args.model,
        'dataset': args.dataset,
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'mcc': metrics['mcc'],
        'kappa': metrics['kappa'],
        'epochs_trained': args.epochs,
        'timestamp': datetime.now().isoformat()
    }])
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Saved metrics: {metrics_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"   MCC: {metrics['mcc']:.4f}")
    print(f"   Cohen's Kappa: {metrics['kappa']:.4f}")
    print(f"\n📁 Output: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
