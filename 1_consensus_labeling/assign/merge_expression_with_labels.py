#!/usr/bin/env python3
"""
Merge Gene Expression Matrix with Consensus Labels

This script merges the normalized gene expression matrix with consensus cell cycle
phase labels to create final training data in the format expected by training scripts.

Usage:
    python merge_expression_with_labels.py \\
        --expression <gene_expression.csv> \\
        --labels <consensus_labels.csv> \\
        --output <output.csv>

Example:
    python merge_expression_with_labels.py \\
        --expression /data/Training_data/pbmc_healthy_human/trainingdata_cellcylescore_pbmc_healthy_human_normalized_gene_expression.csv \\
        --labels final_training_data_human/pbmc_human_consensus_full.csv \\
        --output final_training_data_human/pbmc_human_training_data.csv
"""

import argparse
import pandas as pd
import sys
import os

# Import gene overlap utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../2_model_training/utils'))
from gene_overlap import find_overlapping_genes_across_datasets, filter_dataframe_to_overlapping_genes, get_benchmark_paths


def merge_expression_labels(expression_file, labels_file, output_file, min_agreement=2, filter_to_overlap=True):
    """
    Merge gene expression matrix with consensus labels.

    Args:
        expression_file: Path to gene expression CSV
        labels_file: Path to consensus labels CSV
        output_file: Path to output merged CSV
        min_agreement: Minimum number of tools that must agree (default: 2)
    """
    print(f"\n{'='*80}")
    print(f"MERGING GENE EXPRESSION WITH CONSENSUS LABELS")
    print(f"{'='*80}\n")

    # Load gene expression matrix
    print(f"Loading gene expression matrix: {expression_file}")
    expr_df = pd.read_csv(expression_file, index_col=0)  # First column as index
    print(f"  Shape: {expr_df.shape} (cells × genes)")
    print(f"  Genes: {expr_df.shape[1]}")
    print(f"  Cells: {expr_df.shape[0]}")

    # Reset index to make cell IDs a column
    expr_df = expr_df.reset_index()
    expr_df = expr_df.rename(columns={expr_df.columns[0]: 'gex_barcode'})

    # Load consensus labels
    print(f"\nLoading consensus labels: {labels_file}")
    labels_df = pd.read_csv(labels_file)
    print(f"  Total consensus cells: {len(labels_df)}")

    # Filter by minimum agreement if needed
    if 'n_tools_agree' in labels_df.columns:
        before_filter = len(labels_df)
        labels_df = labels_df[labels_df['n_tools_agree'] >= min_agreement]
        print(f"  After filtering (≥{min_agreement} tools): {len(labels_df)} cells")
        if before_filter > len(labels_df):
            print(f"  Removed {before_filter - len(labels_df)} cells with <{min_agreement} agreement")

    # Show phase distribution
    if 'Predicted' in labels_df.columns:
        phase_counts = labels_df['Predicted'].value_counts()
        print(f"\n  Phase distribution in consensus labels:")
        for phase, count in sorted(phase_counts.items()):
            pct = (count / len(labels_df)) * 100
            print(f"    {phase}: {count} ({pct:.1f}%)")

    # Filter to overlapping GENES (features) across all benchmarks - NO imputation needed!
    if filter_to_overlap:
        print(f"\n{'='*80}")
        print(f"FILTERING TO OVERLAPPING GENES (features only, NOT labels)")
        print(f"{'='*80}")

        # Get benchmark paths
        benchmarks = get_benchmark_paths()
        print(f"\nBenchmarks to check: {list(benchmarks.keys())}")

        # Get gene names from current expression matrix (capitalize for matching)
        print(f"\nExtracting gene names from training data...")
        training_genes = set([col.capitalize() for col in expr_df.columns
                             if col not in ['gex_barcode', 'CellID', 'Predicted', 'Cell_ID', 'cell', 'Phase', 'phase']])
        print(f"  Training genes (before overlap): {len(training_genes)}")

        # Get gene names from each benchmark
        overlapping_genes = training_genes.copy()

        for bench_name, bench_path in benchmarks.items():
            print(f"\nLoading gene names from {bench_name}...")
            bench_df = pd.read_csv(bench_path, nrows=0)  # Only read header
            bench_genes = set([col for col in bench_df.columns
                              if col not in ['gex_barcode', 'CellID', 'Predicted', 'Cell_ID', 'cell', 'Phase', 'phase', '']])

            # Capitalize gene names
            bench_genes = set([g.capitalize() for g in bench_genes])
            print(f"  {bench_name} genes: {len(bench_genes)}")

            # Find intersection
            before = len(overlapping_genes)
            overlapping_genes = overlapping_genes.intersection(bench_genes)
            after = len(overlapping_genes)
            print(f"  After intersection: {after} genes (removed {before - after})")

        print(f"\n{'─'*80}")
        print(f"Final overlapping GENES: {len(overlapping_genes)}")
        print(f"Percentage kept: {len(overlapping_genes)/len(training_genes)*100:.1f}%")
        print(f"{'─'*80}")

        # Filter expression data to ONLY overlapping genes
        print(f"\nFiltering expression matrix to overlapping genes...")
        metadata_cols = ['gex_barcode']
        gene_cols_to_keep = [col for col in expr_df.columns if col.capitalize() in overlapping_genes or col in metadata_cols]

        expr_df = expr_df[gene_cols_to_keep]
        print(f"  Filtered to {len(gene_cols_to_keep)} columns ({len(gene_cols_to_keep)-1} genes + 1 metadata)")
        print(f"{'='*80}\n")

    # Merge on cell IDs
    print(f"\n Merging expression and labels...")
    merged_df = expr_df.merge(
        labels_df[['CellID', 'Predicted']],
        left_on='gex_barcode',
        right_on='CellID',
        how='inner'  # Only keep cells that have both expression and labels
    )

    print(f"  Cells after merge: {len(merged_df)}")
    print(f"  Features: {len(merged_df.columns) - 2} genes + 2 metadata (gex_barcode, CellID, Predicted)")

    # Check for any missing cells
    missing_in_expr = set(labels_df['CellID']) - set(expr_df['gex_barcode'])
    missing_in_labels = set(expr_df['gex_barcode']) - set(labels_df['CellID'])

    if missing_in_expr:
        print(f"\n  WARNING: {len(missing_in_expr)} cells in labels NOT found in expression matrix")
    if missing_in_labels:
        print(f"  Note: {len(missing_in_labels)} cells in expression matrix NOT in consensus labels (filtered out)")

    # Reorder columns: gex_barcode, genes..., CellID, Predicted
    gene_cols = [col for col in merged_df.columns if col not in ['gex_barcode', 'CellID', 'Predicted']]
    final_cols = ['gex_barcode'] + gene_cols + ['CellID', 'Predicted']
    merged_df = merged_df[final_cols]

    # Save merged data
    print(f"\nSaving merged training data: {output_file}")
    merged_df.to_csv(output_file, index=False)

    # Final statistics
    print(f"\n{'='*80}")
    print(f"MERGE COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Final training data statistics:")
    print(f"  Total cells: {len(merged_df)}")
    print(f"  Total genes: {len(gene_cols)}")
    print(f"  Format: [gex_barcode, {len(gene_cols)} genes, CellID, Predicted]")

    final_phase_counts = merged_df['Predicted'].value_counts()
    print(f"\n  Final phase distribution:")
    for phase, count in sorted(final_phase_counts.items()):
        pct = (count / len(merged_df)) * 100
        print(f"    {phase}: {count} ({pct:.1f}%)")

    print(f"\n  Output file: {output_file}")
    print(f"  Ready for training!\n")
    print(f"{'='*80}\n")

    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge gene expression matrix with consensus labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Human PBMC data
    python merge_expression_labels.py \\
        --expression /data/Training_data/pbmc_healthy_human/trainingdata_cellcylescore_pbmc_healthy_human_normalized_gene_expression.csv \\
        --labels final_training_data_human/pbmc_human_consensus_full.csv \\
        --output final_training_data_human/pbmc_human_training_data.csv

    # Mouse brain data
    python merge_expression_labels.py \\
        --expression /data/Training_data/10-k-brain-cells_healthy_mouse/trainingdata_cellcylescore_10-k-brain-cells_healthy_mouse_normalized_gene_expression.csv \\
        --labels final_training_data_mouse/mouse_brain_consensus_full.csv \\
        --output final_training_data_mouse/mouse_brain_training_data.csv
        """
    )

    parser.add_argument(
        '--expression',
        type=str,
        required=True,
        help='Path to normalized gene expression CSV file'
    )

    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to consensus labels CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output merged training data CSV'
    )

    parser.add_argument(
        '--min_agreement',
        type=int,
        default=2,
        help='Minimum number of tools that must agree (default: 2)'
    )

    args = parser.parse_args()

    # Validate inputs
    import os
    if not os.path.exists(args.expression):
        print(f"ERROR: Expression file not found: {args.expression}")
        sys.exit(1)

    if not os.path.exists(args.labels):
        print(f"ERROR: Labels file not found: {args.labels}")
        sys.exit(1)

    # Merge
    merge_expression_labels(args.expression, args.labels, args.output, args.min_agreement)


if __name__ == "__main__":
    main()
