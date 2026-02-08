#!/usr/bin/env python3
"""
General-Purpose Dataset Concatenation Script
==============================================

Concatenates multiple scRNA-seq training datasets with optional gene filtering.

Features:
- Accepts multiple dataset CSV files
- Finds common genes (intersection) across all datasets
- Optional: Filter to specific marker genes from gene list files
- Optional: Take only top K genes per gene list
- Outputs: concatenated_training_data.csv + gene_list.txt

Usage:
    python concatenate_datasets_with_genes.py \
        --datasets dataset1.csv dataset2.csv dataset3.csv \
        --gene-lists markers1.txt markers2.txt \
        --top-k-per-list 50 \
        --output-dir output/ \
        --output-name my_dataset

Author: Halima Akhter
Date: 2026-02-07
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


def load_gene_lists(gene_list_files, top_k=None):
    """
    Load marker genes from multiple gene list files.

    Args:
        gene_list_files (list): List of paths to gene list .txt files
        top_k (int): Take only top K genes from each file (None = all)

    Returns:
        list: Combined list of unique marker genes
    """
    all_marker_genes = []

    for gene_file in gene_list_files:
        print(f"  Loading: {Path(gene_file).name}")
        with open(gene_file, 'r') as f:
            genes = [line.strip().upper() for line in f if line.strip()]

        # Take top K if specified
        if top_k is not None:
            original_count = len(genes)
            genes = genes[:top_k]
            if original_count < top_k:
                print(f"    Genes: {len(genes)} (WARNING: file has only {original_count} genes, requested top {top_k})")
            else:
                print(f"    Genes: {len(genes)} (top {top_k} from {original_count})")
        else:
            print(f"    Genes: {len(genes)}")

        all_marker_genes.extend(genes)

    # Remove duplicates and sort
    unique_markers = sorted(list(set(all_marker_genes)))
    print(f"\n  Total unique marker genes: {len(unique_markers)}")

    return unique_markers


def load_dataset(csv_path, metadata_cols=['gex_barcode', 'Predicted', 'CellID', 'Cell_ID', 'Dataset']):
    """
    Load a dataset CSV and separate genes from metadata.

    Args:
        csv_path (str): Path to CSV file
        metadata_cols (list): Column names that are NOT genes

    Returns:
        tuple: (DataFrame, gene_columns_list, dataset_name)
    """
    df = pd.read_csv(csv_path)
    dataset_name = Path(csv_path).stem

    # Rename gene columns to UPPERCASE for consistent intersection
    # (Keeps metadata columns as-is)
    rename_map = {}
    for col in df.columns:
        if col not in metadata_cols:
            rename_map[col] = col.upper()

    df.rename(columns=rename_map, inplace=True)

    # Get gene columns (now in UPPERCASE)
    gene_cols = [col for col in df.columns if col not in metadata_cols]

    print(f"  {dataset_name}:")
    print(f"    Cells: {len(df)}")
    print(f"    Genes: {len(gene_cols)}")
    print(f"    Phases: {df['Predicted'].value_counts().to_dict()}")

    return df, gene_cols, dataset_name


def load_benchmark_genes(benchmark_path):
    """
    Load gene names from a benchmark expression CSV file.

    Args:
        benchmark_path (str): Path to benchmark expression CSV

    Returns:
        tuple: (gene_list, benchmark_name)
    """
    # Read only header to get gene names (faster for large files)
    df_header = pd.read_csv(benchmark_path, nrows=0)
    benchmark_name = Path(benchmark_path).stem

    # First column is usually CellID, rest are genes
    # Convert to UPPERCASE for consistent intersection (Buettner is Capitalized, GSE are UPPERCASE)
    gene_cols = [col.upper() for col in df_header.columns if col not in ['CellID', 'Cell_ID', 'gex_barcode']]

    print(f"  {benchmark_name}:")
    print(f"    Genes: {len(gene_cols)}")

    return gene_cols, benchmark_name


def main():
    parser = argparse.ArgumentParser(
        description='Concatenate multiple datasets with optional gene filtering'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='Paths to dataset CSV files (e.g., dataset1.csv dataset2.csv)'
    )

    parser.add_argument(
        '--gene-lists',
        nargs='*',
        default=None,
        help='Paths to gene list .txt files for filtering (optional). If not provided, uses all common genes.'
    )

    parser.add_argument(
        '--top-k-per-list',
        type=int,
        default=None,
        help='Take only top K genes from each gene list (default: None = use all genes)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for concatenated dataset'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        required=True,
        help='Name for output files (e.g., "concate_2ds_marker_genes")'
    )

    parser.add_argument(
        '--benchmarks',
        nargs='*',
        default=None,
        help='Paths to benchmark expression CSV files for gene intersection (optional). If provided, only genes present in ALL benchmarks will be used.'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("DATASET CONCATENATION WITH GENE FILTERING")
    print("=" * 80)
    print(f"Datasets to concatenate: {len(args.datasets)}")
    print(f"Benchmark intersection: {'Yes (' + str(len(args.benchmarks)) + ' benchmarks)' if args.benchmarks else 'No'}")
    print(f"Gene filtering: {'Yes' if args.gene_lists else 'No (use all common genes)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output name: {args.output_name}")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load marker genes (if provided)
    # =========================================================================
    marker_genes = None
    if args.gene_lists:
        print("\nSTEP 1: Loading marker genes...")
        print("-" * 80)
        marker_genes = load_gene_lists(args.gene_lists, args.top_k_per_list)

    # =========================================================================
    # STEP 2: Load all datasets
    # =========================================================================
    print("\nSTEP 2: Loading datasets...")
    print("-" * 80)

    datasets = []
    gene_lists = []
    dataset_names = []

    for csv_path in args.datasets:
        df, genes, name = load_dataset(csv_path)
        datasets.append(df)
        gene_lists.append(genes)
        dataset_names.append(name)

    # =========================================================================
    # STEP 2.5: Load benchmark genes (if provided)
    # =========================================================================
    benchmark_gene_lists = []
    benchmark_names = []

    if args.benchmarks:
        print("\nSTEP 2.5: Loading benchmark genes...")
        print("-" * 80)

        for benchmark_path in args.benchmarks:
            genes, name = load_benchmark_genes(benchmark_path)
            benchmark_gene_lists.append(genes)
            benchmark_names.append(name)

    # =========================================================================
    # STEP 3: Find common genes (intersection across training + benchmarks)
    # =========================================================================
    print("\nSTEP 3: Finding common genes across all datasets...")
    print("-" * 80)

    # Start with first training dataset
    common_genes = set(gene_lists[0])

    # Intersect with remaining training datasets
    for genes in gene_lists[1:]:
        common_genes = common_genes & set(genes)

    print(f"Common genes (training datasets only): {len(common_genes)}")

    # Intersect with benchmarks if provided
    if benchmark_gene_lists:
        genes_before_benchmark = len(common_genes)
        for benchmark_genes in benchmark_gene_lists:
            common_genes = common_genes & set(benchmark_genes)
        print(f"Common genes (after benchmark intersection): {len(common_genes)}")
        print(f"  Genes removed by benchmark intersection: {genes_before_benchmark - len(common_genes)}")

    common_genes = sorted(list(common_genes))

    # Show unique genes per dataset
    print(f"\nGene availability:")
    for i, (name, genes) in enumerate(zip(dataset_names, gene_lists)):
        in_common = len(set(genes) & set(common_genes))
        unique = len(set(genes) - set(common_genes))
        print(f"  {name}: {in_common} common, {unique} unique")

    if benchmark_gene_lists:
        for i, (name, genes) in enumerate(zip(benchmark_names, benchmark_gene_lists)):
            in_common = len(set(genes) & set(common_genes))
            unique = len(set(genes) - set(common_genes))
            print(f"  {name} (benchmark): {in_common} common, {unique} unique")

    # =========================================================================
    # STEP 4: Filter to marker genes (if provided)
    # =========================================================================
    if marker_genes:
        print("\nSTEP 4: Filtering to marker genes...")
        print("-" * 80)

        # Find intersection of common genes and marker genes
        available_markers = sorted(list(set(common_genes) & set(marker_genes)))
        missing_markers = set(marker_genes) - set(common_genes)

        print(f"Marker genes requested: {len(marker_genes)}")
        print(f"Marker genes available: {len(available_markers)}")
        print(f"Marker genes missing: {len(missing_markers)}")

        if missing_markers and len(missing_markers) <= 20:
            print(f"  Missing markers: {sorted(list(missing_markers))}")

        final_genes = available_markers
    else:
        print("\nSTEP 4: Using all common genes (no filtering)")
        print("-" * 80)
        final_genes = common_genes

    print(f"Final gene count: {len(final_genes)}")

    # =========================================================================
    # STEP 5: Subset datasets to final genes
    # =========================================================================
    print("\nSTEP 5: Subsetting datasets to final genes...")
    print("-" * 80)

    cols_to_keep = ['gex_barcode'] + final_genes + ['Predicted']

    subsetted_datasets = []
    for df, name in zip(datasets, dataset_names):
        df_subset = df[cols_to_keep].copy()
        df_subset['Dataset'] = name
        subsetted_datasets.append(df_subset)
        print(f"  {name}: {df_subset.shape}")

    # =========================================================================
    # STEP 6: Concatenate datasets
    # =========================================================================
    print("\nSTEP 6: Concatenating datasets...")
    print("-" * 80)

    concatenated = pd.concat(subsetted_datasets, axis=0, ignore_index=True)

    print(f"Concatenated dataset:")
    print(f"  Total cells: {len(concatenated)}")
    print(f"  Total genes: {len(final_genes)}")
    print(f"  Columns: {concatenated.shape[1]}")
    print(f"  Phase distribution: {concatenated['Predicted'].value_counts().to_dict()}")
    print(f"  Dataset distribution: {concatenated['Dataset'].value_counts().to_dict()}")

    # =========================================================================
    # STEP 7: Save outputs
    # =========================================================================
    print("\nSTEP 7: Saving outputs...")
    print("-" * 80)

    # Save concatenated CSV
    output_csv = os.path.join(args.output_dir, f"{args.output_name}.csv")
    concatenated.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")

    # Save gene list
    output_genes = os.path.join(args.output_dir, f"gene_list.txt")
    with open(output_genes, 'w') as f:
        f.write('\n'.join(final_genes))
    print(f"  Saved: {output_genes}")

    # Save metadata
    output_meta = os.path.join(args.output_dir, f"dataset_metadata.txt")
    with open(output_meta, 'w') as f:
        f.write(f"Dataset: {args.output_name}\n")
        f.write(f"Created: {pd.Timestamp.now()}\n")
        f.write(f"Total cells: {len(concatenated)}\n")
        f.write(f"Total genes: {len(final_genes)}\n")
        f.write(f"Gene format: UPPERCASE (for cross-species compatibility)\n")
        f.write(f"Datasets concatenated: {len(args.datasets)}\n")
        for name in dataset_names:
            count = len(concatenated[concatenated['Dataset'] == name])
            f.write(f"  - {name}: {count} cells\n")
        f.write(f"\nPhase distribution:\n")
        for phase, count in concatenated['Predicted'].value_counts().items():
            f.write(f"  - {phase}: {count} cells\n")
        if args.benchmarks:
            f.write(f"\nBenchmark intersection:\n")
            f.write(f"  Benchmarks used: {len(args.benchmarks)}\n")
            for bf in args.benchmarks:
                f.write(f"    - {Path(bf).name}\n")
        if args.gene_lists:
            f.write(f"\nGene filtering:\n")
            f.write(f"  Marker gene lists used: {len(args.gene_lists)}\n")
            for gf in args.gene_lists:
                f.write(f"    - {Path(gf).name}\n")
            if args.top_k_per_list:
                f.write(f"  Top K per list: {args.top_k_per_list}\n")
    print(f"  Saved: {output_meta}")

    print("\n" + "=" * 80)
    print("CONCATENATION COMPLETE!")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Cells: {len(concatenated)}")
    print(f"Genes: {len(final_genes)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
