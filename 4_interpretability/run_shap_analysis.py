#!/usr/bin/env python3
"""
SHAP Analysis CLI for Model Interpretability
=============================================

Performs SHAP (SHapley Additive exPlanations) analysis on trained models.
Based on user's original implementation from 3_0_principle_aurelien_ml_evaluate.py

Usage:
    python run_shap_analysis.py --model_path /path/to/model.pt --output_dir results/shap/

Author: Halima Akhter
Date: 2025-11-27
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import VotingClassifier

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../3_evaluation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../2_model_training'))

from model_loader import load_model_components
from utils.data_utils import load_gse146773

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_traditional_ml(model):
    """Check if model is Traditional ML (sklearn) vs Deep Learning (PyTorch)."""
    import sklearn
    return isinstance(model, sklearn.base.BaseEstimator)


def run_shap_deep_learning(model, benchmark_features, feature_names, output_dir, model_name):
    """
    Run SHAP analysis for Deep Learning models using KernelExplainer.

    Based on user's original code - uses KernelExplainer, not DeepExplainer.

    Args:
        model: PyTorch model
        benchmark_features: Benchmark dataset (pandas DataFrame)
        feature_names: List of feature names
        output_dir: Directory to save SHAP outputs
        model_name: Model name for file naming
    """
    print(f"\n{'='*80}")
    print(f"SHAP ANALYSIS - DEEP LEARNING MODEL")
    print(f"{'='*80}")
    print(f"Model: {model_name}")

    # Define model prediction wrapper (from user's original code)
    def model_predict_np(data):
        data_tensor = torch.FloatTensor(data).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            predictions = model(data_tensor)
        return predictions.cpu().detach().numpy()

    # Sample background and test data (user's original: 10 background, 10 test)
    background_data = benchmark_features.sample(n=10, random_state=42)
    background_data_tensor = torch.tensor(background_data.values, dtype=torch.float32).to(device)

    remaining_data = benchmark_features.drop(background_data.index)
    benchmark_data = remaining_data.sample(n=10, random_state=42)
    benchmark_data_tensor = torch.tensor(benchmark_data.values, dtype=torch.float32).to(device)

    num_features = benchmark_data.shape[1]
    nsamples = min(50, num_features + 1)

    if num_features > nsamples:
        nsamples = num_features + 1

    print(f"Background samples: {len(background_data)}")
    print(f"Test samples: {len(benchmark_data)}")
    print(f"Features: {num_features}")
    print(f"nsamples: {nsamples}")

    # Initialize KernelExplainer (user's original approach)
    print("\nCreating SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(model_predict_np, background_data_tensor.cpu().numpy())

    # Calculate SHAP values
    print("Computing SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(benchmark_data_tensor.cpu().numpy(), nsamples=nsamples)

    # Take the mean SHAP value across classes (user's original code)
    shap_values = np.mean(shap_values, axis=2)

    print(f"Shape of shap_values after mean: {shap_values.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot SHAP summary
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, benchmark_data.values, feature_names=feature_names,
                      max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {model_name}_shap_summary.png")

    # Get the average absolute SHAP value for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_shap_values)[-20:][::-1]
    top_features = [feature_names[i] for i in top_features_idx]

    print("\nTop 20 SHAP features:", top_features)

    # Get indices sorted by SHAP value importance, in descending order
    all_features_idx = np.argsort(mean_shap_values)[::-1]
    all_features = [feature_names[i] for i in all_features_idx]

    # Save SHAP output to text file (user's original format)
    shap_file_path = os.path.join(output_dir, f"{model_name}_SHAP.txt")
    with open(shap_file_path, 'w') as f:
        f.write("\n".join(all_features))

    print(f"✅ SHAP analysis completed. Results saved to {shap_file_path}")

    # Also save as CSV for convenience
    top_features_df = pd.DataFrame({
        'feature': all_features,
        'mean_abs_shap': [mean_shap_values[all_features_idx[i]] for i in range(len(all_features))]
    })
    top_features_df.to_csv(os.path.join(output_dir, f"{model_name}_top_features.csv"), index=False)

    return shap_values, top_features_df


def run_shap_traditional_ml(model, benchmark_features, feature_names, output_dir, model_name, model_type):
    """
    Run SHAP analysis for Traditional ML models using KernelExplainer.

    Based on user's original code.

    Args:
        model: sklearn model
        benchmark_features: Benchmark dataset (pandas DataFrame)
        feature_names: List of feature names
        output_dir: Directory to save SHAP outputs
        model_name: Model name for file naming
        model_type: Model type (adaboost, lgbm, random_forest, ensemble)
    """
    print(f"\n{'='*80}")
    print(f"SHAP ANALYSIS - TRADITIONAL ML MODEL")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Model type: {model_type}")

    model_list = ["adaboost", "lgbm", "random_forest", "ensemble"]

    if model_type not in model_list or len(feature_names) > 7408:
        print("No SHAP analysis needed.")
        return None, None

    print("Generating SHAP explanations for the benchmark dataset...")

    if isinstance(benchmark_features, pd.DataFrame):
        X_benchmark_data = benchmark_features
    else:
        X_benchmark_data = pd.DataFrame(benchmark_features)

    # Sample 2 data points for SHAP analysis (user's original code)
    shap_data = X_benchmark_data.sample(n=2, random_state=42)

    # Use shap.sample to get a faster background summary
    background_data = shap.sample(X_benchmark_data, 10)

    print(f"Background samples: {len(background_data)}")
    print(f"Test samples: {len(shap_data)}")
    print(f"Features: {len(feature_names)}")

    shap_values = None

    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy arrays to avoid LGBM compatibility issues
    background_np = background_data.values if isinstance(background_data, pd.DataFrame) else background_data
    shap_data_np = shap_data.values if isinstance(shap_data, pd.DataFrame) else shap_data

    if model_type in ["adaboost", "lgbm", "random_forest"]:
        classifier = model

        # Use Kernel SHAP (user's original code)
        # Wrap predict function to avoid LGBM attribute access issues
        print("\nUsing SHAP KernelExplainer...")
        explainer = shap.KernelExplainer(lambda x: classifier.predict(x), background_np)
        shap_values = explainer.shap_values(shap_data_np)

        print("Generating SHAP summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, shap_data, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {model_name}_shap_summary.png")

    elif model_type == "ensemble":
        # Handle VotingClassifier separately (user's original code)
        print("SHAP analysis for ensemble model")

        if not isinstance(model, VotingClassifier):
            print("Error: Model is not a VotingClassifier")
            return None, None

        shap_values = np.zeros((shap_data.shape[0], shap_data.shape[1]))

        # Loop through each base estimator
        for estimator in model.estimators_:
            print(f"Computing SHAP values for: {estimator}")

            # Wrap predict function to avoid LGBM attribute access issues
            explainer = shap.KernelExplainer(lambda x: estimator.predict(x), background_np)
            model_shap_values = explainer.shap_values(shap_data_np)

            if isinstance(model_shap_values, list):
                model_shap_values = np.array(model_shap_values[0])

            shap_values += model_shap_values

        # Average SHAP values over all models
        shap_values /= len(model.estimators_)

        # Plot SHAP summary
        print("Generating SHAP summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, shap_data, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {model_name}_shap_summary.png")

    # Save SHAP-ranked features (user's original code)
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    all_features_idx = np.argsort(mean_shap_values)[::-1]
    all_features = [feature_names[i] for i in all_features_idx]

    print("All features ranked by SHAP importance:", all_features[:20])

    shap_file_path = os.path.join(output_dir, f"{model_name}_SHAP.txt")
    with open(shap_file_path, 'w') as f:
        f.write("\n".join(all_features))

    print(f"✅ SHAP analysis completed. Results saved to {shap_file_path}")

    # Also save as CSV
    top_features_df = pd.DataFrame({
        'feature': all_features,
        'mean_abs_shap': [mean_shap_values[all_features_idx[i]] for i in range(len(all_features))]
    })
    top_features_df.to_csv(os.path.join(output_dir, f"{model_name}_top_features.csv"), index=False)

    return shap_values, top_features_df


def main():
    parser = argparse.ArgumentParser(
        description='SHAP Analysis for Deep Learning and Traditional ML models'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model (.pt for DL, .joblib for TML)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='GSE146773',
        choices=['GSE146773', 'GSE64016', 'SUP', 'Buettner_mESC'],
        help='Benchmark dataset to use for SHAP analysis (default: GSE146773). Ignored if --custom_benchmark is provided.'
    )
    parser.add_argument(
        '--custom_benchmark',
        type=str,
        default=None,
        help='Path to custom benchmark data CSV'
    )
    parser.add_argument(
        '--custom_benchmark_name',
        type=str,
        default='CustomBenchmark',
        help='Name for custom benchmark (used in output files)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: results/shap/model_name/)'
    )

    args = parser.parse_args()

    # Load model components
    print(f"\n{'='*80}")
    print(f"LOADING MODEL")
    print(f"{'='*80}")
    print(f"Model: {os.path.basename(args.model_path)}")

    model, scaler, label_encoder, selected_features = load_model_components(args.model_path)
    model_name = os.path.basename(args.model_path).replace('.pt', '').replace('.pkl', '').replace('.joblib', '')

    # Extract model type from prefix
    model_type = model_name.split('_')[0].lower()

    # Check if TML or DL
    is_tml = is_traditional_ml(model)
    model_category = "Traditional ML" if is_tml else "Deep Learning"

    print(f"✓ Model loaded successfully ({model_category})")
    print(f"{'='*80}\n")

    # Load benchmark data
    from utils.data_utils import load_gse146773, load_gse64016, load_reh_or_sup_benchmark, load_buettner_mesc, load_custom_benchmark

    if args.custom_benchmark is not None:
        # Load custom benchmark
        print(f"Loading custom benchmark data: {args.custom_benchmark_name}")
        benchmark_features, benchmark_labels, _ = load_custom_benchmark(
            args.custom_benchmark, scaler, args.custom_benchmark_name, False
        )
    else:
        # Load standard benchmark
        print(f"Loading benchmark data: {args.benchmark}")
        if args.benchmark == "GSE146773":
            benchmark_features, benchmark_labels, _ = load_gse146773(scaler, False)
        elif args.benchmark == "GSE64016":
            benchmark_features, benchmark_labels, _ = load_gse64016(scaler, False)
        elif args.benchmark == "SUP":
            benchmark_features, benchmark_labels, _ = load_reh_or_sup_benchmark(scaler, reh_sup="sup")
        elif args.benchmark == "Buettner_mESC":
            benchmark_features, benchmark_labels, _ = load_buettner_mesc(scaler, False)

    # Select only the features used by the model
    benchmark_features = benchmark_features[selected_features]

    print(f"  Loaded {len(benchmark_features)} samples")
    print(f"  Features: {len(selected_features)}")

    # Set output directory
    if args.output_dir is None:
        base_output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(args.model_path))),
            'results', 'shap', model_name
        )
        args.output_dir = base_output

    os.makedirs(args.output_dir, exist_ok=True)

    # Run SHAP analysis
    if is_tml:
        shap_values, top_features = run_shap_traditional_ml(
            model, benchmark_features, selected_features,
            args.output_dir, model_name, model_type
        )
    else:
        shap_values, top_features = run_shap_deep_learning(
            model, benchmark_features, selected_features,
            args.output_dir, model_name
        )

    # Print top 10 features
    if top_features is not None:
        print(f"\n{'='*80}")
        print(f"TOP 10 MOST IMPORTANT FEATURES")
        print(f"{'='*80}")
        for idx, row in top_features.head(10).iterrows():
            print(f"{row['feature']:30s}: {row['mean_abs_shap']:.6f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
