#!/usr/bin/env python3
"""
Prepare Benchmarks for Dataset Concatenation

Processes GSE64016 and Buettner_mESC benchmarks into the format expected by evaluation code.

GSE64016:
- Input: Genes (rows) x Cells (columns) - normalized expected counts
- Output:
  - GSE64016_expression.csv: Cells x Genes with CellID column
  - GSE64016_labels.csv: barcodes, Labeled columns

Buettner_mESC:
- Input: Genes (rows) x Cells (columns) - RAW COUNTS
- Processing: Log normalization (library size + log1p) + transpose
- Output:
  - Buettner_mESC_expression.csv: Cells x Genes with CellID column (SAME AS GSE64016)
  - Buettner_mESC_labels.csv: CellID, Predicted columns

Author: Halima Akhter
Date: 2026-02-07
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re

# Paths
BASE_DIR = Path("/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq")
OUTPUT_DIR = BASE_DIR / "data/benchmarks_preprocessed"

# GSE64016 paths
GSE64016_INPUT = BASE_DIR / "cell_cycle_prediction/Has_cell_cycle_effect_or_not/scRNA_data/3_GSE64016_H1andFUCCI_normalized_EC.csv/GSE64016_H1andFUCCI_normalized_EC.csv"
GSE64016_OUTPUT = OUTPUT_DIR / "GSE64016"

# Buettner paths
BUETTNER_EXPRESSION_INPUT = BASE_DIR / "cell_cycle_prediction/Has_cell_cycle_effect_or_not/scRNA_data/Buettner_mESC/Buettner_mESC_raw_counts.csv"
BUETTNER_LABELS_INPUT = BASE_DIR / "cell_cycle_prediction/Has_cell_cycle_effect_or_not/scRNA_data/Buettner_mESC/Buettner_mESC_labels.csv"
BUETTNER_OUTPUT = OUTPUT_DIR / "Buettner_mESC"


def extract_phase_from_cellname(cell_name):
    """
    Extract phase label from GSE64016 cell names based on PREFIX.

    Examples:
    - G1_Exp1.001 -> G1
    - S_Exp1.045 -> S
    - G2_Exp1.059 -> G2M
    - H1_Exp1.001 -> None (unlabeled, will be filtered out)
    """
    # Extract prefix (part before first underscore)
    prefix = cell_name.split('_')[0] if '_' in cell_name else cell_name

    # Map prefix to phase
    if prefix == 'G1':
        return 'G1'
    elif prefix == 'G2' or prefix == 'G2M':
        return 'G2M'
    elif prefix == 'S':
        return 'S'
    else:
        # Unlabeled cells (H1, etc.) - return None to filter out
        return None


def log_normalize_counts(df_counts):
    """
    Apply Seurat-style log normalization to raw counts.

    Steps:
    1. Normalize each cell by total counts (library size)
    2. Scale to 10,000 (standard)
    3. Log1p transform: log(x + 1)

    Args:
        df_counts: DataFrame with genes (rows) x cells (columns), raw counts

    Returns:
        DataFrame with same shape, log-normalized values
    """
    print("  Applying log normalization...")
    print(f"    Before: min={df_counts.min().min():.2f}, max={df_counts.max().max():.2f}, mean={df_counts.mean().mean():.2f}")

    # Normalize by library size (total counts per cell)
    library_sizes = df_counts.sum(axis=0)  # Sum each column (cell)
    df_normalized = df_counts / library_sizes * 10000

    # Log1p transform
    df_log = np.log1p(df_normalized)

    print(f"    After: min={df_log.min().min():.2f}, max={df_log.max().max():.2f}, mean={df_log.mean().mean():.2f}")

    return df_log


def process_gse64016():
    """
    Process GSE64016 benchmark.

    Steps:
    1. Load genes x cells format
    2. Transpose to cells x genes
    3. Extract phase labels from cell names
    4. Save separate expression and labels files
    """
    print("=" * 80)
    print("PROCESSING GSE64016")
    print("=" * 80)

    # Load data
    print(f"\nStep 1: Loading data from {GSE64016_INPUT}")
    df = pd.read_csv(GSE64016_INPUT, index_col=0)
    print(f"  Loaded: {df.shape[0]} genes x {df.shape[1]} cells")
    print(f"  First few genes: {list(df.index[:5])}")
    print(f"  First few cells: {list(df.columns[:5])}")

    # Transpose to cells x genes
    print(f"\nStep 2: Transposing to cells x genes...")
    df_transposed = df.T
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'CellID'}, inplace=True)
    print(f"  Transposed: {df_transposed.shape[0]} cells x {df_transposed.shape[1]-1} genes")

    # Extract phase labels and filter unlabeled cells
    print(f"\nStep 3: Extracting phase labels from cell names...")
    labels_dict = {}
    unlabeled_count = 0
    for cell_name in df_transposed['CellID']:
        phase = extract_phase_from_cellname(cell_name)
        if phase is not None:
            labels_dict[cell_name] = phase
        else:
            unlabeled_count += 1

    print(f"  Total cells: {len(df_transposed)}")
    print(f"  Labeled cells: {len(labels_dict)}")
    print(f"  Unlabeled cells (filtered out): {unlabeled_count}")

    # Filter expression data to keep only labeled cells
    labeled_cells = list(labels_dict.keys())
    df_transposed = df_transposed[df_transposed['CellID'].isin(labeled_cells)].copy()
    print(f"  Keeping {len(df_transposed)} labeled cells")

    # Count labels
    label_counts = pd.Series(labels_dict.values()).value_counts()
    print(f"  Label distribution:")
    for label, count in label_counts.items():
        print(f"    {label}: {count} cells")

    # Create labels dataframe
    labels_df = pd.DataFrame({
        'barcodes': list(labels_dict.keys()),
        'Labeled': list(labels_dict.values())
    })

    # Save outputs
    print(f"\nStep 4: Saving outputs to {GSE64016_OUTPUT}")
    GSE64016_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Save expression (cells x genes)
    expression_file = GSE64016_OUTPUT / "GSE64016_expression.csv"
    df_transposed.to_csv(expression_file, index=False)
    print(f"  Saved expression: {expression_file}")
    print(f"    Shape: {df_transposed.shape[0]} cells x {df_transposed.shape[1]-1} genes")

    # Save labels
    labels_file = GSE64016_OUTPUT / "GSE64016_labels.csv"
    labels_df.to_csv(labels_file, index=False)
    print(f"  Saved labels: {labels_file}")
    print(f"    Shape: {len(labels_df)} labels")

    # Save metadata
    metadata_file = GSE64016_OUTPUT / "GSE64016_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Benchmark: GSE64016\n")
        f.write(f"Data source: GEO (normalized expected counts)\n")
        f.write(f"Preprocessing: None (data already normalized by authors)\n")
        f.write(f"Format: Cells x Genes\n")
        f.write(f"Cells: {df_transposed.shape[0]}\n")
        f.write(f"Genes: {df_transposed.shape[1]-1}\n")
        f.write(f"Label distribution:\n")
        for label, count in label_counts.items():
            f.write(f"  {label}: {count} cells\n")
    print(f"  Saved metadata: {metadata_file}")

    print(f"\nGSE64016 processing complete!")
    print("=" * 80)

    return df_transposed.shape[0], df_transposed.shape[1]-1


def process_buettner():
    """
    Process Buettner_mESC benchmark.

    Steps:
    1. Load raw counts (genes x cells)
    2. Apply log normalization
    3. Transpose to cells x genes (SAME FORMAT AS GSE64016)
    4. Load labels (already in correct format)
    5. Save separate expression and labels files
    """
    print("\n" + "=" * 80)
    print("PROCESSING Buettner_mESC")
    print("=" * 80)

    # Load RAW COUNTS
    print(f"\nStep 1: Loading RAW COUNTS from {BUETTNER_EXPRESSION_INPUT}")
    df_raw = pd.read_csv(BUETTNER_EXPRESSION_INPUT, index_col=0)
    print(f"  Loaded: {df_raw.shape[0]} genes x {df_raw.shape[1]} cells")
    print(f"  First few genes: {list(df_raw.index[:5])}")
    print(f"  First few cells: {list(df_raw.columns[:5])}")

    # Apply log normalization
    print(f"\nStep 2: Applying log normalization (Seurat-style)...")
    df_normalized = log_normalize_counts(df_raw)

    # Transpose to cells x genes (SAME AS GSE64016)
    print(f"\nStep 3: Transposing to cells x genes...")
    df_transposed = df_normalized.T
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'CellID'}, inplace=True)
    print(f"  Transposed: {df_transposed.shape[0]} cells x {df_transposed.shape[1]-1} genes")
    print(f"  Format: SAME AS GSE64016 (Cells x Genes with CellID column)")

    # Load labels
    print(f"\nStep 4: Loading labels from {BUETTNER_LABELS_INPUT}")
    df_labels = pd.read_csv(BUETTNER_LABELS_INPUT)
    print(f"  Loaded: {len(df_labels)} labels")

    print(f"  Label distribution:")
    for phase, count in df_labels['Predicted'].value_counts().items():
        print(f"    {phase}: {count} cells")

    # Save outputs
    print(f"\nStep 5: Saving outputs to {BUETTNER_OUTPUT}")
    BUETTNER_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Save expression (cells x genes, log-normalized - SAME FORMAT AS GSE64016)
    expression_file = BUETTNER_OUTPUT / "Buettner_mESC_expression.csv"
    df_transposed.to_csv(expression_file, index=False)
    print(f"  Saved expression: {expression_file}")
    print(f"    Format: {df_transposed.shape[0]} cells x {df_transposed.shape[1]-1} genes (CONSISTENT WITH GSE64016)")

    # Save labels
    labels_file = BUETTNER_OUTPUT / "Buettner_mESC_labels.csv"
    df_labels.to_csv(labels_file, index=False)
    print(f"  Saved labels: {labels_file}")
    print(f"    Shape: {len(df_labels)} labels")

    # Save metadata
    metadata_file = BUETTNER_OUTPUT / "Buettner_mESC_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Benchmark: Buettner_mESC\n")
        f.write(f"Data source: Original publication (raw counts)\n")
        f.write(f"Preprocessing: Log normalization (library size + log1p)\n")
        f.write(f"File format: Cells x Genes (SAME AS GSE64016)\n")
        f.write(f"Cells: {df_transposed.shape[0]}\n")
        f.write(f"Genes: {df_transposed.shape[1]-1}\n")
        f.write(f"Gene format: Capitalized (mouse genes)\n")
        f.write(f"Label distribution:\n")
        for phase, count in df_labels['Predicted'].value_counts().items():
            f.write(f"  {phase}: {count} cells\n")
    print(f"  Saved metadata: {metadata_file}")

    print(f"\nBuettner_mESC processing complete!")
    print("=" * 80)

    return df_transposed.shape[0], df_transposed.shape[1]-1  # cells, genes


def main():
    print("\n" + "=" * 80)
    print("BENCHMARK PREPROCESSING FOR DATASET CONCATENATION")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    # Process GSE64016
    gse_cells, gse_genes = process_gse64016()

    # Process Buettner
    buettner_cells, buettner_genes = process_buettner()

    # Summary
    print("\n" + "=" * 80)
    print("ALL BENCHMARKS PROCESSED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nGSE64016:")
    print(f"  Cells: {gse_cells}")
    print(f"  Genes: {gse_genes}")
    print(f"  Location: {GSE64016_OUTPUT}")
    print(f"\nBuettner_mESC:")
    print(f"  Cells: {buettner_cells}")
    print(f"  Genes: {buettner_genes}")
    print(f"  Location: {BUETTNER_OUTPUT}")
    print("\n" + "=" * 80)
    print("Next step: Run create_marker_gene_datasets.sh")
    print("=" * 80)


if __name__ == "__main__":
    main()
