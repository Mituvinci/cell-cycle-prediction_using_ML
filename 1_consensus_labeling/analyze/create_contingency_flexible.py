#!/usr/bin/env python3
"""
Create Contingency Tables - Flexible Version
=============================================

This script creates contingency tables comparing predictions from different
cell cycle prediction tools. Accepts direct file paths for flexibility.

Usage:
    python create_contingency_flexible.py \
        --tool1-file path/to/seurat.csv \
        --tool1-name seurat \
        --tool2-file path/to/tricycle.csv \
        --tool2-name tricycle \
        --output-dir results/

Author: Halima Akhter
Date: 2025-11-28
"""

import os
import argparse
import pandas as pd
import numpy as np


def load_predictions(file_path, tool_name):
    """
    Load predictions from a tool output file.

    Parameters:
    -----------
    file_path : str
        Path to the prediction CSV file
    tool_name : str
        Name of the tool (for column naming)

    Returns:
    --------
    pd.DataFrame
        DataFrame with CellID and Predicted columns
    """
    print(f"Loading {tool_name} from: {file_path}")
    df = pd.read_csv(file_path)

    # Ensure required columns exist
    if 'CellID' not in df.columns or 'Predicted' not in df.columns:
        raise ValueError(f"File must have 'CellID' and 'Predicted' columns. Found: {df.columns.tolist()}")

    # Handle NA values
    df['Predicted'] = df['Predicted'].fillna('NA')

    print(f"  Loaded {len(df)} cells")
    print(f"  Phases: {df['Predicted'].unique().tolist()}")

    return df[['CellID', 'Predicted']].rename(columns={'Predicted': f'Predicted_{tool_name}'})


def create_contingency_table(df1, df2, tool1_name, tool2_name):
    """
    Create contingency table comparing two tool predictions.

    Parameters:
    -----------
    df1 : pd.DataFrame
        Predictions from tool 1
    df2 : pd.DataFrame
        Predictions from tool 2
    tool1_name : str
        Name of tool 1
    tool2_name : str
        Name of tool 2

    Returns:
    --------
    pd.DataFrame
        Contingency table (crosstab)
    """
    # Merge on cell IDs
    merged = df1.merge(df2, on='CellID', how='inner')

    print(f"\n  Merged cells: {len(merged)}")

    # Create contingency table
    contingency = pd.crosstab(
        merged[f'Predicted_{tool1_name}'],
        merged[f'Predicted_{tool2_name}'],
        margins=False
    )

    return contingency, merged


def main():
    parser = argparse.ArgumentParser(
        description='Create contingency tables comparing cell cycle prediction tools (flexible version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mouse: Compare Seurat vs Tricycle
  python create_contingency_flexible.py \\
    --tool1-file /path/to/mouse/seurat.csv \\
    --tool1-name seurat \\
    --tool2-file /path/to/mouse/tricycle.csv \\
    --tool2-name tricycle \\
    --dataset-name mouse_brain \\
    --output-dir results/mouse/

  # Human: Compare ccAFv2 vs Revelio
  python create_contingency_flexible.py \\
    --tool1-file /path/to/human/ccafv2.csv \\
    --tool1-name ccafv2 \\
    --tool2-file /path/to/human/revelio.csv \\
    --tool2-name revelio \\
    --dataset-name pbmc_human \\
    --output-dir results/human/
        """
    )

    parser.add_argument(
        '--tool1-file',
        type=str,
        required=True,
        help='Path to tool 1 prediction CSV (must have CellID and Predicted columns)'
    )

    parser.add_argument(
        '--tool1-name',
        type=str,
        required=True,
        help='Name of tool 1 (e.g., seurat, tricycle, ccafv2, revelio)'
    )

    parser.add_argument(
        '--tool2-file',
        type=str,
        required=True,
        help='Path to tool 2 prediction CSV'
    )

    parser.add_argument(
        '--tool2-name',
        type=str,
        required=True,
        help='Name of tool 2'
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Dataset name for output files (e.g., mouse_brain, pbmc_human)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/contingency_tables',
        help='Output directory for contingency tables'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Creating Contingency Table")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Tool 1: {args.tool1_name}")
    print(f"Tool 2: {args.tool2_name}")
    print()

    # Load predictions
    df1 = load_predictions(args.tool1_file, args.tool1_name)
    df2 = load_predictions(args.tool2_file, args.tool2_name)

    # Create contingency table
    print(f"\nCreating contingency table...")
    contingency, merged = create_contingency_table(df1, df2, args.tool1_name, args.tool2_name)

    # Display table
    print("\nContingency Table:")
    print(contingency)

    # Save table
    output_file = os.path.join(
        args.output_dir,
        f"contingency_{args.tool1_name}_vs_{args.tool2_name}_{args.dataset_name}.csv"
    )
    contingency.to_csv(output_file)
    print(f"\n✓ Saved to: {output_file}")

    # Save merged data (for heatmap generation)
    merged_file = os.path.join(
        args.output_dir,
        f"merged_{args.tool1_name}_vs_{args.tool2_name}_{args.dataset_name}.csv"
    )
    merged.to_csv(merged_file, index=False)
    print(f"✓ Saved merged data to: {merged_file}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total cells: {len(merged)}")
    print(f"  {args.tool1_name} phases: {list(contingency.index)}")
    print(f"  {args.tool2_name} phases: {list(contingency.columns)}")

    print("\n" + "=" * 80)
    print("✅ Contingency table created successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
