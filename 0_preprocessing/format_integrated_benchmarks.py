"""
Format Integrated Benchmark Files
=================================

This script formats the integrated benchmark files (from scTQuery REH alignment)
to match the exact format of existing benchmark files, so evaluation code works
without modification.

IMPORTANT RULES:
----------------
1. ALL gene names are converted to UPPERCASE (e.g., ACTB, GAPDH, TP53)
2. ALL gene columns are sorted ALPHABETICALLY
3. Labels are merged from existing benchmark files

Formatting Rules:
-----------------
1. GSE146773:
   - Column order: Unnamed:0 (index), cell, genes (UPPERCASE, alphabetical)..., paper_phase
   - Labels from: existing GSE146773 'paper_phase' column

2. GSE64016:
   - Column order: (unnamed index as cell_id), genes (UPPERCASE, alphabetical)..., gex_barcode, Labeled
   - Labels from: existing GSE64016 'Labeled' column

3. Buettner_mESC:
   - Column order: Cell_ID, genes (UPPERCASE, alphabetical)...
   - Labels from: Buettner_mESC_goundTruth.csv 'Phase' column
   - Note: Phase is in separate file, merged during evaluation

Usage:
------
python format_integrated_benchmarks.py --input_dir <path_to_integrated_files> --output_dir <path_to_output>

Example:
python format_integrated_benchmarks.py \
    --input_dir /users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/scTQuery/REH_aligned \
    --output_dir /users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/scTQuery/REH_aligned_formatted

Author: Halima Akhter
Date: 2025-12-06
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


def uppercase_gene_name(gene_name):
    """Convert gene name to UPPERCASE format."""
    if isinstance(gene_name, str):
        return gene_name.upper()
    return gene_name


def format_gse146773(input_path, output_path, existing_benchmark_path):
    """
    Format integrated GSE146773 file to match existing benchmark format.

    Output format:
    - Columns: Unnamed: 0, cell, GENE1, GENE2 (alphabetical), ..., paper_phase
    - Unnamed: 0 is numeric index (1, 2, 3, ...)
    - cell contains cell IDs (A1_355, A1_356, ...)
    - Genes are UPPERCASE and sorted ALPHABETICALLY
    - paper_phase contains labels (G1, S, G2M)
    """
    print(f"\nProcessing GSE146773...")
    print(f"  Input: {input_path}")

    # Load integrated data
    print("  Loading integrated data...")
    df_new = pd.read_csv(input_path)
    print(f"  Loaded {len(df_new)} cells, {len(df_new.columns)} columns")

    # Load existing benchmark to get labels
    print("  Loading existing benchmark for labels...")
    df_existing = pd.read_csv(existing_benchmark_path)
    print(f"  Existing benchmark: {len(df_existing)} cells")

    # Extract cell IDs and labels from existing
    # Existing has 'cell' column and 'paper_phase' column
    existing_labels = df_existing[['cell', 'paper_phase']].copy()
    existing_labels = existing_labels.dropna(subset=['paper_phase'])
    print(f"  Found {len(existing_labels)} labeled cells in existing benchmark")

    # Remove 'dataset' column from new data
    if 'dataset' in df_new.columns:
        df_new = df_new.drop(columns=['dataset'])

    # Rename Cell_ID to cell
    df_new = df_new.rename(columns={'Cell_ID': 'cell'})

    # Merge labels based on cell ID
    df_merged = df_new.merge(existing_labels, on='cell', how='inner')
    print(f"  After merging labels: {len(df_merged)} cells")

    if len(df_merged) == 0:
        print("  WARNING: No matching cell IDs found! Checking cell ID formats...")
        print(f"    New data cell IDs (first 5): {df_new['cell'].head().tolist()}")
        print(f"    Existing cell IDs (first 5): {existing_labels['cell'].head().tolist()}")
        return False

    # Get gene columns (exclude metadata)
    gene_cols = [col for col in df_merged.columns if col not in ['cell', 'paper_phase']]

    # Convert ALL genes to UPPERCASE
    print("  Converting gene names to UPPERCASE...")
    rename_dict = {}
    for col in gene_cols:
        upper_col = col.upper()
        if upper_col != col:
            rename_dict[col] = upper_col
    if rename_dict:
        df_merged = df_merged.rename(columns=rename_dict)
        gene_cols = [rename_dict.get(col, col) for col in gene_cols]

    # Sort gene columns ALPHABETICALLY
    gene_cols_sorted = sorted(gene_cols)
    print(f"  Sorted {len(gene_cols_sorted)} gene columns alphabetically")
    print(f"  First 5 genes: {gene_cols_sorted[:5]}")
    print(f"  Last 5 genes: {gene_cols_sorted[-5:]}")

    # Create final column order: cell, sorted_genes, paper_phase
    df_final = df_merged[['cell'] + gene_cols_sorted + ['paper_phase']].copy()

    # Add numeric index as first column (Unnamed: 0)
    df_final.insert(0, 'Unnamed: 0', range(1, len(df_final) + 1))

    # Save
    print(f"  Saving to: {output_path}")
    df_final.to_csv(output_path, index=False)
    print(f"  Saved {len(df_final)} cells, {len(df_final.columns)} columns")

    # Verify format
    print(f"  First 5 columns: {df_final.columns[:5].tolist()}")
    print(f"  Last 3 columns: {df_final.columns[-3:].tolist()}")
    print(f"  Label distribution: {df_final['paper_phase'].value_counts().to_dict()}")

    return True


def format_gse64016(input_path, output_path, existing_benchmark_path):
    """
    Format integrated GSE64016 file to match existing benchmark format.

    Output format:
    - First column is unnamed (cell ID as index): H1_Exp1.001, H1_Exp1.002, ...
    - Then gene columns (UPPERCASE, sorted ALPHABETICALLY)
    - gex_barcode column
    - Labeled column (G1, S, G2M)
    """
    print(f"\nProcessing GSE64016...")
    print(f"  Input: {input_path}")

    # Load integrated data
    print("  Loading integrated data...")
    df_new = pd.read_csv(input_path)
    print(f"  Loaded {len(df_new)} cells, {len(df_new.columns)} columns")

    # Load existing benchmark to get labels (first column is unnamed index)
    print("  Loading existing benchmark for labels...")
    df_existing = pd.read_csv(existing_benchmark_path)
    print(f"  Existing benchmark: {len(df_existing)} cells")

    # First column is 'Unnamed: 0' which contains cell IDs
    first_col = df_existing.columns[0]
    existing_labels = pd.DataFrame({
        'cell_id': df_existing[first_col],
        'Labeled': df_existing['Labeled'],
        'gex_barcode': df_existing['gex_barcode'] if 'gex_barcode' in df_existing.columns else df_existing[first_col]
    })
    existing_labels = existing_labels.dropna(subset=['Labeled'])
    print(f"  Found {len(existing_labels)} labeled cells in existing benchmark")

    # Remove 'dataset' column from new data
    if 'dataset' in df_new.columns:
        df_new = df_new.drop(columns=['dataset'])

    # Rename Cell_ID to match
    df_new = df_new.rename(columns={'Cell_ID': 'cell_id'})

    # Merge labels based on cell ID
    df_merged = df_new.merge(existing_labels[['cell_id', 'Labeled', 'gex_barcode']], on='cell_id', how='inner')
    print(f"  After merging labels: {len(df_merged)} cells")

    if len(df_merged) == 0:
        print("  WARNING: No matching cell IDs found! Checking cell ID formats...")
        print(f"    New data cell IDs (first 5): {df_new['cell_id'].head().tolist()}")
        print(f"    Existing cell IDs (first 5): {existing_labels['cell_id'].head().tolist()}")
        return False

    # Get gene columns (exclude metadata)
    gene_cols = [col for col in df_merged.columns if col not in ['cell_id', 'Labeled', 'gex_barcode']]

    # Convert ALL genes to UPPERCASE
    print("  Converting gene names to UPPERCASE...")
    rename_dict = {}
    for col in gene_cols:
        upper_col = col.upper()
        if upper_col != col:
            rename_dict[col] = upper_col
    if rename_dict:
        df_merged = df_merged.rename(columns=rename_dict)
        gene_cols = [rename_dict.get(col, col) for col in gene_cols]

    # Sort gene columns ALPHABETICALLY
    gene_cols_sorted = sorted(gene_cols)
    print(f"  Sorted {len(gene_cols_sorted)} gene columns alphabetically")
    print(f"  First 5 genes: {gene_cols_sorted[:5]}")
    print(f"  Last 5 genes: {gene_cols_sorted[-5:]}")

    # Create final DataFrame - first column should be unnamed (empty header)
    # Reorder: cell_id (will become unnamed), sorted_genes, gex_barcode, Labeled
    df_final = df_merged[['cell_id'] + gene_cols_sorted + ['gex_barcode', 'Labeled']].copy()

    # Rename cell_id to empty string so it becomes unnamed column in CSV
    df_final = df_final.rename(columns={'cell_id': ''})

    # Save (first column will have empty header like original)
    print(f"  Saving to: {output_path}")
    df_final.to_csv(output_path, index=False)
    print(f"  Saved {len(df_final)} cells, {len(df_final.columns)} columns")

    # Verify format
    print(f"  First 5 columns: {list(df_final.columns[:5])}")
    print(f"  Last 3 columns: {list(df_final.columns[-3:])}")
    print(f"  Label distribution: {df_final['Labeled'].value_counts().to_dict()}")

    return True


def format_buettner(input_path, output_path, existing_benchmark_path, ground_truth_path):
    """
    Format integrated Buettner_mESC file to match existing benchmark format.

    Output format:
    - Cell_ID column
    - Gene columns in UPPERCASE format (GNAI3, PBSN, CDC45) sorted ALPHABETICALLY
    - No Phase column (Phase is in separate ground truth file)
    """
    print(f"\nProcessing Buettner_mESC...")
    print(f"  Input: {input_path}")

    # Load integrated data
    print("  Loading integrated data...")
    df_new = pd.read_csv(input_path)
    print(f"  Loaded {len(df_new)} cells, {len(df_new.columns)} columns")

    # Load ground truth for validation
    print("  Loading ground truth for validation...")
    df_gt = pd.read_csv(ground_truth_path)
    print(f"  Ground truth: {len(df_gt)} cells with phases")

    # Load existing benchmark to verify cell IDs
    print("  Loading existing benchmark for reference...")
    df_existing = pd.read_csv(existing_benchmark_path)
    print(f"  Existing benchmark: {len(df_existing)} cells")

    # Remove 'dataset' column from new data
    if 'dataset' in df_new.columns:
        df_new = df_new.drop(columns=['dataset'])

    # Get gene columns (everything except Cell_ID)
    gene_cols = [col for col in df_new.columns if col != 'Cell_ID']

    # Convert ALL gene names to UPPERCASE
    print("  Converting gene names to UPPERCASE...")
    rename_dict = {}
    for col in gene_cols:
        upper_col = col.upper()
        if upper_col != col:
            rename_dict[col] = upper_col

    if rename_dict:
        df_new = df_new.rename(columns=rename_dict)
        gene_cols = [rename_dict.get(col, col) for col in gene_cols]
        print(f"  Renamed {len(rename_dict)} gene columns to UPPERCASE")

    # Verify cell IDs match between new data and ground truth
    new_cells = set(df_new['Cell_ID'])
    gt_cells = set(df_gt['Cell_ID'])
    matching_cells = new_cells & gt_cells
    print(f"  New data cells: {len(new_cells)}")
    print(f"  Ground truth cells: {len(gt_cells)}")
    print(f"  Matching cells: {len(matching_cells)}")

    if len(matching_cells) == 0:
        print("  WARNING: No matching cell IDs found! Checking formats...")
        print(f"    New data cell IDs (first 5): {list(new_cells)[:5]}")
        print(f"    Ground truth cell IDs (first 5): {list(gt_cells)[:5]}")
        return False

    # Filter to only cells with ground truth
    df_final = df_new[df_new['Cell_ID'].isin(matching_cells)].copy()
    print(f"  After filtering to labeled cells: {len(df_final)} cells")

    # Sort gene columns ALPHABETICALLY
    gene_cols_sorted = sorted([col for col in df_final.columns if col != 'Cell_ID'])
    print(f"  Sorted {len(gene_cols_sorted)} gene columns alphabetically")
    print(f"  First 5 genes: {gene_cols_sorted[:5]}")
    print(f"  Last 5 genes: {gene_cols_sorted[-5:]}")

    # Reorder columns: Cell_ID first, then sorted genes
    df_final = df_final[['Cell_ID'] + gene_cols_sorted]

    # Save
    print(f"  Saving to: {output_path}")
    df_final.to_csv(output_path, index=False)
    print(f"  Saved {len(df_final)} cells, {len(df_final.columns)} columns")

    # Verify format
    print(f"  First 5 columns: {df_final.columns[:5].tolist()}")
    print(f"  Sample gene names: {df_final.columns[1:6].tolist()}")

    return True


def find_integrated_file(input_dir, benchmark_name):
    """
    Find integrated file matching the benchmark name, regardless of prefix.

    Supports patterns like:
    - Integrated_REH_aligned_to_GSE146773.csv
    - Integrated_PBMC_aligned_to_GSE146773.csv
    - Integrated_MouseBrain_aligned_to_GSE146773.csv
    """
    import glob

    # Try to find file with pattern Integrated_*_aligned_to_{benchmark_name}.csv
    pattern = os.path.join(input_dir, f'Integrated_*_aligned_to_{benchmark_name}.csv')
    matches = glob.glob(pattern)

    if matches:
        return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description='Format integrated benchmark files to match existing format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing integrated benchmark files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as input with _formatted suffix)')
    parser.add_argument('--data_root', type=str,
                        default='/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data',
                        help='Root data directory containing existing benchmarks')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir.rstrip('/') + '_formatted'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Define paths
    existing_gse146773 = os.path.join(args.data_root, 'Training_data/Benchmark_data/GSE146773_seurat_normalized_gene_expression.csv')
    existing_gse64016 = os.path.join(args.data_root, 'Training_data/Benchmark_data/GSE64016_seurat_normalized_gene_expression.csv')
    existing_buettner = os.path.join(args.data_root, 'Buettner_mESC_benchmark_clean.csv')
    buettner_ground_truth = os.path.join(args.data_root, 'Buettner_mESC_goundTruth.csv')

    # Process each benchmark type
    results = {}

    # GSE146773
    input_gse146773 = find_integrated_file(args.input_dir, 'GSE146773')
    if input_gse146773:
        print(f"\nFound GSE146773 file: {os.path.basename(input_gse146773)}")
        output_gse146773 = os.path.join(args.output_dir, 'GSE146773_seurat_normalized_gene_expression.csv')
        results['GSE146773'] = format_gse146773(input_gse146773, output_gse146773, existing_gse146773)
    else:
        print(f"\nSkipping GSE146773: No matching file found in {args.input_dir}")

    # GSE64016
    input_gse64016 = find_integrated_file(args.input_dir, 'GSE64016')
    if input_gse64016:
        print(f"\nFound GSE64016 file: {os.path.basename(input_gse64016)}")
        output_gse64016 = os.path.join(args.output_dir, 'GSE64016_seurat_normalized_gene_expression.csv')
        results['GSE64016'] = format_gse64016(input_gse64016, output_gse64016, existing_gse64016)
    else:
        print(f"\nSkipping GSE64016: No matching file found in {args.input_dir}")

    # Buettner
    input_buettner = find_integrated_file(args.input_dir, 'Buettner_mESC')
    if input_buettner:
        print(f"\nFound Buettner file: {os.path.basename(input_buettner)}")
        output_buettner = os.path.join(args.output_dir, 'Buettner_mESC_benchmark_clean.csv')
        results['Buettner'] = format_buettner(input_buettner, output_buettner, existing_buettner, buettner_ground_truth)
    else:
        print(f"\nSkipping Buettner: No matching file found in {args.input_dir}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nOutput files saved to: {args.output_dir}")
    print("\nNote: Update data_utils.py to point to the new formatted files.")


if __name__ == '__main__':
    main()
