#!/usr/bin/env python3
"""
Generate table with best accuracy for each training dataset on each benchmark.

For each training dataset, finds the model with highest accuracy on each benchmark.

Output: CSV table with training datasets as rows, benchmarks as columns
"""

import pandas as pd
import os

# Base directory
BASE_DIR = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/5_visualization/heatmap_barplot_lineplots_csv"

# Training datasets configuration
TRAINING_DATASETS = {
    'nft_reh': {
        'display_name': 'REH',
        'type': 'Multiome RNA'
    },
    'nft_hpsc': {
        'display_name': 'GSE75748 hPSCs',
        'type': 'Pure scRNA-seq'
    },
    'nft_pbmc': {
        'display_name': 'PBMC',
        'type': 'Pure scRNA-seq'
    },
    'nft_mouse_brain': {
        'display_name': 'Mouse Brain',
        'type': 'Pure scRNA-seq'
    }
}

# Benchmarks
BENCHMARKS = ['gse146773', 'gse64016', 'buettner_mesc']
BENCHMARK_DISPLAY = {
    'gse146773': 'GSE146773',
    'gse64016': 'GSE64016',
    'buettner_mesc': 'Buettner mESC'
}


def get_best_accuracy(training_dataset, benchmark):
    """
    Get the best (highest) accuracy for a training dataset on a benchmark.

    Args:
        training_dataset: e.g., 'nft_reh'
        benchmark: e.g., 'gse146773'

    Returns:
        float: Highest accuracy value, or None if file not found
    """
    # Construct file path
    csv_file = os.path.join(BASE_DIR, training_dataset,
                            f"{benchmark}_results_{training_dataset}.csv")

    if not os.path.exists(csv_file):
        print(f"  WARNING: File not found: {csv_file}")
        return None

    # Read CSV
    df = pd.read_csv(csv_file)

    # Check if Accuracy column exists
    if 'Accuracy' not in df.columns:
        print(f"  WARNING: No 'Accuracy' column in {csv_file}")
        return None

    # Sort by Accuracy (descending) and get the highest
    df_sorted = df.sort_values('Accuracy', ascending=False)
    best_accuracy = df_sorted['Accuracy'].iloc[0]
    best_model = df_sorted['Model'].iloc[0]

    print(f"  {training_dataset} on {benchmark}: {best_accuracy:.2f}% ({best_model})")

    return best_accuracy


def generate_table():
    """Generate the accuracy table."""

    print("="*70)
    print("GENERATING BEST ACCURACY TABLE")
    print("="*70)

    # Prepare data for table
    table_data = []

    for training_dataset, config in TRAINING_DATASETS.items():
        print(f"\nProcessing: {config['display_name']} ({training_dataset})")

        row = {
            'Training Data': config['display_name'],
            'Type': config['type']
        }

        # Get best accuracy for each benchmark
        for benchmark in BENCHMARKS:
            best_acc = get_best_accuracy(training_dataset, benchmark)
            benchmark_col = BENCHMARK_DISPLAY[benchmark]

            if best_acc is not None:
                row[benchmark_col] = f"{best_acc:.2f}"
            else:
                row[benchmark_col] = "N/A"

        table_data.append(row)

    # Create DataFrame
    df_table = pd.DataFrame(table_data)

    # Set column order
    columns = ['Training Data', 'Type', 'GSE146773', 'GSE64016', 'Buettner mESC']
    df_table = df_table[columns]

    # Save to CSV
    output_file = os.path.join(BASE_DIR, 'best_accuracy_comparison_table.csv')
    df_table.to_csv(output_file, index=False)

    print("\n" + "="*70)
    print("TABLE GENERATED")
    print("="*70)
    print(f"\nSaved to: {output_file}\n")
    print(df_table.to_string(index=False))
    print("\n" + "="*70)


if __name__ == '__main__':
    generate_table()
