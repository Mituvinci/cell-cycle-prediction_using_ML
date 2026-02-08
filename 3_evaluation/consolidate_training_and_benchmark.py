#!/usr/bin/env python3
"""
Consolidate Training and Benchmark Results
===========================================

Merges *_details.csv (training metrics) with *_details_benchmark.csv (benchmark metrics)
into a single consolidated CSV file.

Usage:
  python consolidate_training_and_benchmark.py --input models/ --output results/consolidated.csv
  For Mouse Brain models:
  python consolidate_training_and_benchmark.py \
      --input models/mouse_brain \
      --output results/ft_6466_mouse_brain.csv \
      --species all

  For Human PBMC models:
  python consolidate_training_and_benchmark.py \
      --input models/human_hpsc_5770_standard \
      --output results/ft_5770_std_human_hpsc.csv \
      --species all




Author: Halima Akhter
Date: 2025-12-04
"""

import os
import sys
import glob
import argparse
import pandas as pd


def find_model_directories(base_path):
    """
    Find all model directories containing trained models.

    Args:
        base_path: Root path to search (e.g., "models/")

    Returns:
        list: List of model directory paths
    """
    model_dirs = []

    # Search for directories containing .pt or .joblib files
    for root, dirs, files in os.walk(base_path):
        has_models = any(f.endswith('.pt') or f.endswith('.joblib') for f in files)
        if has_models:
            model_dirs.append(root)

    return model_dirs


def merge_training_and_benchmark(model_dir):
    """
    Merge training details and benchmark details for all models in a directory.

    Args:
        model_dir: Path to model directory

    Returns:
        pd.DataFrame: Merged results, or None if no matches found
    """
    model_name = os.path.basename(model_dir)

    # Find training details and benchmark details files
    training_files = glob.glob(os.path.join(model_dir, "*_details.csv"))
    benchmark_files = glob.glob(os.path.join(model_dir, "*_details_benchmark.csv"))

    if len(training_files) == 0:
        print(f"  WARNING: No training details files found in {model_dir}")
        return None

    if len(benchmark_files) == 0:
        print(f"  WARNING: No benchmark details files found in {model_dir}")
        return None

    print(f"  Found {len(training_files)} training files and {len(benchmark_files)} benchmark files")

    # Create dict of training files by prefix_name
    training_dict = {}
    for f in training_files:
        prefix = os.path.basename(f).replace("_details.csv", "")
        training_dict[prefix] = f

    # Merge each benchmark file with matching training file
    merged_results = []

    for benchmark_file in benchmark_files:
        prefix = os.path.basename(benchmark_file).replace("_details_benchmark.csv", "")

        if prefix in training_dict:
            training_file = training_dict[prefix]

            # Read both CSVs
            training_df = pd.read_csv(training_file)
            benchmark_df = pd.read_csv(benchmark_file)

            # Merge on prefix_name
            if "prefix_name" in training_df.columns and "prefix_name" in benchmark_df.columns:
                merged_df = pd.merge(
                    training_df, benchmark_df,
                    on="prefix_name",
                    suffixes=("_train", "_bench"),
                    how="outer"
                )
                merged_df["Model"] = model_name
                merged_results.append(merged_df)
            else:
                print(f"  WARNING: 'prefix_name' column missing in {prefix}")
        else:
            print(f"  WARNING: No matching training file for {benchmark_file}")

    if merged_results:
        return pd.concat(merged_results, ignore_index=True)
    else:
        return None


def reorder_columns(df):
    """
    Reorder columns in preferred order: Model, prefix_name, training metrics, benchmark metrics.

    Args:
        df: DataFrame to reorder

    Returns:
        pd.DataFrame: Reordered DataFrame
    """
    # Define exact column order matching user's Windows code
    desired_order = [
        "Model", "prefix_name", "best_params", "training_time", "cpu_memory_used", "gpu_memory_used",
        "test_accuracy", "test_roc_auc", "test_f1", "test_mcc", "test_kappa", "test_balanced_acc",
        "test_precision", "test_recall",
        "sup_accuracy", "sup_roc_auc", "sup_f1", "sup_mcc", "sup_kappa", "sup_balanced_acc",
        "sup_precision", "sup_recall",
        'sup_precision_g1', 'sup_precision_g2m', 'sup_precision_s',
        'sup_recall_g1', 'sup_recall_g2m', 'sup_recall_s',
        'sup_f1-score_g1', 'sup_f1-score_g2m', 'sup_f1-score_s',
        'sup_accuracy_g1', 'sup_accuracy_g2m', 'sup_accuracy_s',
        'sup_mcc_g1', 'sup_mcc_g2m', 'sup_mcc_s',
        "gse146773_accuracy", "gse146773_roc_auc", "gse146773_f1", "gse146773_mcc", "gse146773_kappa", "gse146773_balanced_acc",
        "gse146773_precision", "gse146773_recall",
        'gse146773_precision_g1', 'gse146773_precision_g2m', 'gse146773_precision_s',
        'gse146773_recall_g1', 'gse146773_recall_g2m', 'gse146773_recall_s',
        'gse146773_f1-score_g1', 'gse146773_f1-score_g2m', 'gse146773_f1-score_s',
        'gse146773_accuracy_g1', 'gse146773_accuracy_g2m', 'gse146773_accuracy_s',
        'gse146773_mcc_g1', 'gse146773_mcc_g2m', 'gse146773_mcc_s',
        "gse64016_accuracy", "gse64016_roc_auc", "gse64016_f1", "gse64016_mcc", "gse64016_kappa", "gse64016_balanced_acc",
        "gse64016_precision", "gse64016_recall",
        'gse64016_precision_g1', 'gse64016_precision_g2m', 'gse64016_precision_s',
        'gse64016_recall_g1', 'gse64016_recall_g2m', 'gse64016_recall_s',
        'gse64016_f1-score_g1','gse64016_f1-score_g2m', 'gse64016_f1-score_s',
        'gse64016_accuracy_g1', 'gse64016_accuracy_g2m', 'gse64016_accuracy_s',
        'gse64016_mcc_g1', 'gse64016_mcc_g2m', 'gse64016_mcc_s',
        "buettner_mesc_accuracy", "buettner_mesc_roc_auc", "buettner_mesc_f1", "buettner_mesc_mcc", "buettner_mesc_kappa", "buettner_mesc_balanced_acc",
        "buettner_mesc_precision", "buettner_mesc_recall",
        'buettner_mesc_precision_g1', 'buettner_mesc_precision_g2m', 'buettner_mesc_precision_s',
        'buettner_mesc_recall_g1', 'buettner_mesc_recall_g2m', 'buettner_mesc_recall_s',
        'buettner_mesc_f1-score_g1', 'buettner_mesc_f1-score_g2m', 'buettner_mesc_f1-score_s',
        'buettner_mesc_accuracy_g1', 'buettner_mesc_accuracy_g2m', 'buettner_mesc_accuracy_s',
        'buettner_mesc_mcc_g1', 'buettner_mesc_mcc_g2m', 'buettner_mesc_mcc_s',
        "leng_accuracy", "leng_roc_auc", "leng_f1", "leng_mcc", "leng_kappa", "leng_balanced_acc",
        "leng_precision", "leng_recall",
        'leng_precision_g1', 'leng_precision_g2m', 'leng_precision_s',
        'leng_recall_g1', 'leng_recall_g2m', 'leng_recall_s',
        'leng_f1-score_g1', 'leng_f1-score_g2m', 'leng_f1-score_s',
        'leng_accuracy_g1', 'leng_accuracy_g2m', 'leng_accuracy_s',
        'leng_mcc_g1', 'leng_mcc_g2m', 'leng_mcc_s'
    ]

    # Only keep columns that exist in the dataframe
    ordered_columns = [col for col in desired_order if col in df.columns]

    # Add any remaining columns not in desired_order
    for col in df.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)

    return df[ordered_columns]


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate training and benchmark results into single CSV'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='models/',
        help='Base directory containing model directories (default: models/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/consolidated_results.csv',
        help='Output CSV file path (default: results/consolidated_results.csv)'
    )
    parser.add_argument(
        '--species',
        type=str,
        choices=['human', 'mouse', 'all'],
        default='all',
        help='Which species to consolidate (default: all)'
    )

    args = parser.parse_args()

    print("="*80)
    print("CONSOLIDATING TRAINING AND BENCHMARK RESULTS")
    print("="*80)
    print(f"Input directory: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Species: {args.species}")
    print("")

    # Determine which directories to process
    if args.species == 'all':
        search_paths = [args.input]
    elif args.species == 'human':
        search_paths = [os.path.join(args.input, 'human_pbmc'),
                       os.path.join(args.input, 'old_human')]
    elif args.species == 'mouse':
        search_paths = [os.path.join(args.input, 'mouse_brain')]

    # Find all model directories
    all_model_dirs = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            all_model_dirs.extend(find_model_directories(search_path))

    if not all_model_dirs:
        print(f"ERROR: No model directories found in {args.input}")
        sys.exit(1)

    print(f"Found {len(all_model_dirs)} model directories")
    print("")

    # Merge results from all directories
    all_merged = []

    for model_dir in all_model_dirs:
        print(f"Processing: {model_dir}")
        merged_df = merge_training_and_benchmark(model_dir)

        if merged_df is not None:
            all_merged.append(merged_df)
            print(f"  SUCCESS: Merged {len(merged_df)} rows")
        print("")

    if not all_merged:
        print("ERROR: No results to consolidate")
        sys.exit(1)

    # Combine all results
    final_df = pd.concat(all_merged, ignore_index=True)

    # Move Model column to first position
    model_col = final_df.pop("Model")
    final_df.insert(0, "Model", model_col)

    # Convert training_time from seconds to minutes if exists
    if "training_time" in final_df.columns:
        final_df["training_time"] = final_df["training_time"] / 60

    # Convert memory to absolute values
    if "gpu_memory_used" in final_df.columns:
        final_df["gpu_memory_used"] = abs(final_df["gpu_memory_used"])

    if "cpu_memory_used" in final_df.columns:
        final_df["cpu_memory_used"] = abs(final_df["cpu_memory_used"])

    # Reorder columns
    final_df = reorder_columns(final_df)

    # Save consolidated results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_df.to_csv(args.output, index=False)

    # Print summary
    print("="*80)
    print("CONSOLIDATION COMPLETE")
    print("="*80)
    print(f"Total rows: {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")
    print(f"Models: {final_df['Model'].nunique()}")
    print("")
    print(f"SUCCESS: Results saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
