#!/usr/bin/env python3
"""
Gene Overlap Utilities

This module provides functions to find overlapping genes across training
and benchmark datasets to ensure no imputation is needed during evaluation.

Cross-species training requires using only genes that exist in ALL datasets:
- Training data (human or mouse)
- GSE146773 (human benchmark)
- GSE64016 (human benchmark)
- Buettner_mESC (mouse benchmark)

This allows:
- Train on human → evaluate on human benchmarks AND mouse benchmark
- Train on mouse → evaluate on mouse benchmark AND human benchmarks
"""

import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def capitalize_gene_names(df, exclude_cols=['gex_barcode', 'Predicted', 'Cell_ID', 'cell', 'Phase', 'phase', 'CellID']):
    """
    Capitalize gene names: First letter uppercase, rest lowercase.

    Args:
        df: DataFrame with gene columns
        exclude_cols: Columns to exclude from capitalization

    Returns:
        DataFrame with capitalized gene names
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if col not in exclude_cols:
            df_copy = df_copy.rename(columns={col: col.capitalize()})
    return df_copy


def get_gene_names_from_csv(csv_path, exclude_cols=['gex_barcode', 'Predicted', 'Cell_ID', 'cell', 'Phase', 'phase', 'CellID', '']):
    """
    Extract gene names from a CSV file.

    Args:
        csv_path: Path to CSV file
        exclude_cols: Columns to exclude (metadata, not genes)

    Returns:
        set: Set of gene names (capitalized)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Read only header (first row)
    df_header = pd.read_csv(csv_path, nrows=0)

    # Capitalize column names
    df_header = capitalize_gene_names(df_header, exclude_cols)

    # Extract gene names (exclude metadata columns)
    genes = set()
    for col in df_header.columns:
        if col not in exclude_cols and col.strip() != '':
            genes.add(col)

    return genes


def find_overlapping_genes_across_datasets(training_csv, benchmark_csvs):
    """
    Find genes that exist in training data AND all benchmark datasets.

    Args:
        training_csv: Path to training data CSV
        benchmark_csvs: List of paths to benchmark CSV files

    Returns:
        set: Overlapping gene names (capitalized)
    """
    print(f"\n{'='*80}")
    print(f"FINDING OVERLAPPING GENES")
    print(f"{'='*80}\n")

    # Get genes from training data
    print(f"Loading genes from training data: {os.path.basename(training_csv)}")
    training_genes = get_gene_names_from_csv(training_csv)
    print(f"  Training genes: {len(training_genes)}")

    # Start with training genes
    overlapping_genes = training_genes.copy()

    # Intersect with each benchmark
    for bench_csv in benchmark_csvs:
        print(f"\nLoading genes from benchmark: {os.path.basename(bench_csv)}")
        bench_genes = get_gene_names_from_csv(bench_csv)
        print(f"  Benchmark genes: {len(bench_genes)}")

        # Find intersection
        before = len(overlapping_genes)
        overlapping_genes = overlapping_genes.intersection(bench_genes)
        after = len(overlapping_genes)
        print(f"  After intersection: {after} genes (removed {before - after})")

    print(f"\n{'='*80}")
    print(f"OVERLAP RESULT")
    print(f"{'='*80}")
    print(f"Final overlapping genes: {len(overlapping_genes)}")
    print(f"Percentage of training genes kept: {len(overlapping_genes)/len(training_genes)*100:.1f}%")
    print(f"{'='*80}\n")

    return overlapping_genes


def filter_dataframe_to_overlapping_genes(df, overlapping_genes, metadata_cols=['gex_barcode', 'CellID', 'Predicted', 'Cell_ID', 'cell', 'Phase', 'phase']):
    """
    Filter DataFrame to only include overlapping genes.

    Args:
        df: DataFrame with all genes
        overlapping_genes: Set of gene names to keep
        metadata_cols: Metadata columns to keep

    Returns:
        DataFrame with only overlapping genes + metadata
    """
    # Identify gene columns (not metadata)
    gene_cols = [col for col in df.columns if col not in metadata_cols]

    # Capitalize gene column names
    df_capitalized = df.copy()
    rename_dict = {col: col.capitalize() for col in gene_cols}
    df_capitalized = df_capitalized.rename(columns=rename_dict)

    # Filter to overlapping genes
    genes_to_keep = [col for col in df_capitalized.columns if col.capitalize() in overlapping_genes or col in metadata_cols]

    df_filtered = df_capitalized[genes_to_keep]

    print(f"  Filtered from {df.shape[1]} to {df_filtered.shape[1]} columns")
    print(f"  Genes: {len([c for c in df_filtered.columns if c not in metadata_cols])}")
    print(f"  Metadata: {len([c for c in df_filtered.columns if c in metadata_cols])}")

    return df_filtered


def get_benchmark_paths():
    """
    Get paths to all benchmark datasets for gene overlap calculation.

    Returns:
        dict: {dataset_name: csv_path}
    """
    # Dynamically find project root
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    data_dir = os.path.join(project_root, "data")

    benchmarks = {
        'GSE146773': os.path.join(data_dir, 'GSE146773_seurat_normalized_gene_expression.csv'),
        'GSE64016': os.path.join(data_dir, 'GSE64016_seurat_normalized_gene_expression.csv'),
        'Buettner_mESC': os.path.join(data_dir, 'Buettner_mESC_benchmark_clean.csv')
    }

    # Check which files exist
    existing_benchmarks = {}
    for name, path in benchmarks.items():
        if os.path.exists(path):
            existing_benchmarks[name] = path
        else:
            print(f"Warning: Benchmark not found: {name} at {path}")

    return existing_benchmarks


if __name__ == "__main__":
    """
    Test gene overlap functionality
    """
    import argparse

    parser = argparse.ArgumentParser(description="Find overlapping genes across datasets")
    parser.add_argument('--training', required=True, help='Path to training data CSV')
    parser.add_argument('--output', help='Optional output file for gene list')

    args = parser.parse_args()

    # Get benchmark paths
    benchmarks = get_benchmark_paths()
    print(f"Found {len(benchmarks)} benchmarks: {list(benchmarks.keys())}")

    # Find overlapping genes
    overlapping = find_overlapping_genes_across_datasets(
        args.training,
        list(benchmarks.values())
    )

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            for gene in sorted(overlapping):
                f.write(f"{gene}\n")
        print(f"Saved {len(overlapping)} overlapping genes to: {args.output}")
