#!/usr/bin/env python3
"""
Apply Seurat Log Normalization in Python
Same formula as Seurat's NormalizeData():
  log1p(counts / total_counts_per_cell * scale_factor)

Where:
  - scale_factor = 10000 (Seurat default)
  - log1p = natural log(1 + x)
"""

import pandas as pd
import numpy as np

def seurat_normalize(expression_df, scale_factor=10000):
    """
    Apply Seurat log normalization.

    Args:
        expression_df: DataFrame with genes (rows) x cells (columns)
        scale_factor: Scaling factor (default: 10000)

    Returns:
        Normalized DataFrame
    """
    print(f"  Input shape: {expression_df.shape}")
    print(f"  Input value range: [{expression_df.values.min():.2f}, {expression_df.values.max():.2f}]")

    # Calculate total counts per cell (column sums)
    total_counts_per_cell = expression_df.sum(axis=0)
    print(f"  Mean total counts per cell: {total_counts_per_cell.mean():.2f}")

    # Normalize: (counts / total) * scale_factor
    normalized = expression_df.div(total_counts_per_cell, axis=1) * scale_factor

    # Apply log1p (natural log)
    log_normalized = np.log1p(normalized)

    print(f"  Output value range: [{log_normalized.values.min():.2f}, {log_normalized.values.max():.2f}]")

    return log_normalized


if __name__ == "__main__":
    print("="*80)
    print("SEURAT LOG NORMALIZATION (Python Implementation)")
    print("="*80)
    print()

    # Input: genes x cells (transposed format)
    input_file = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression_raw_for_seurat.csv"

    # Output: cells x genes (for evaluation)
    output_file = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression_normalized_python.csv"

    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file, index_col=0)
    print(f"  Format: Genes ({df.shape[0]}) x Cells ({df.shape[1]})")

    print("\nApplying Seurat normalization...")
    normalized_df = seurat_normalize(df, scale_factor=10000)

    print("\nTransposing back to Cells x Genes format...")
    normalized_df_transposed = normalized_df.T
    print(f"  Output shape: {normalized_df_transposed.shape} (Cells x Genes)")

    print(f"\nSaving: {output_file}")
    normalized_df_transposed.to_csv(output_file)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nNormalized file: {output_file}")
    print("\nFormula used: log1p(counts / total_counts_per_cell * 10000)")
    print("This is IDENTICAL to Seurat's NormalizeData() with default parameters.")
    print("="*80)
