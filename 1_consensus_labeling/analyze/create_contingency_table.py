"""
Create Contingency Tables for Tool Comparison
==============================================

This script creates contingency tables comparing predictions from different
cell cycle prediction tools to understand phase co-occurrences.

Usage:
    python create_contingency_table.py --tool1 seurat --tool2 tricycle --dataset reh --output results/

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_predictions(file_path, tool_name, cell_id_col='CellID', pred_col='Predicted'):
    """
    Load predictions from a tool output file.

    Parameters:
    -----------
    file_path : str
        Path to the prediction CSV file
    tool_name : str
        Name of the tool (for column naming)
    cell_id_col : str
        Column name for cell IDs
    pred_col : str
        Column name for predictions

    Returns:
    --------
    pd.DataFrame
        DataFrame with cell IDs and predictions
    """
    df = pd.read_csv(file_path, index_col=0)

    # Handle NA values
    if pred_col in df.columns:
        df[pred_col] = df[pred_col].fillna('NA')

    return df[[cell_id_col, pred_col]].rename(columns={pred_col: f'Predicted_{tool_name}'})


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
    merged = df1.merge(df2, on='CellID', suffixes=(f'_{tool1_name}', f'_{tool2_name}'))

    # Create contingency table
    contingency = pd.crosstab(
        merged[f'Predicted_{tool1_name}'],
        merged[f'Predicted_{tool2_name}'],
        margins=False
    )

    return contingency


def main():
    parser = argparse.ArgumentParser(
        description='Create contingency tables comparing cell cycle prediction tools'
    )
    parser.add_argument(
        '--tool1',
        type=str,
        required=True,
        choices=['seurat', 'tricycle', 'revelio', 'ccafv2'],
        help='First tool to compare'
    )
    parser.add_argument(
        '--tool2',
        type=str,
        required=True,
        choices=['seurat', 'tricycle', 'revelio', 'ccafv2'],
        help='Second tool to compare'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['reh', 'sup', 'gse146773', 'gse64016'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/predictions',
        help='Directory containing prediction files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../results/contingency_tables',
        help='Output directory for contingency tables'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Creating Contingency Table")
    print("=" * 80)
    print(f"Tool 1: {args.tool1}")
    print(f"Tool 2: {args.tool2}")
    print(f"Dataset: {args.dataset}")
    print()

    # Define file naming conventions for different tools
    tool_file_patterns = {
        'seurat': 'cellcyclescore_',
        'tricycle': 'tricycle_',
        'revelio': 'revelio_',
        'ccafv2': 'ccAFV2_'
    }

    # Load tool 1 predictions
    tool1_pattern = tool_file_patterns[args.tool1]
    tool1_files = list(Path(args.data_dir).glob(f"{tool1_pattern}*{args.dataset}*.csv"))

    if not tool1_files:
        raise FileNotFoundError(f"No {args.tool1} prediction files found for {args.dataset}")

    tool1_file = tool1_files[0]
    print(f"Loading {args.tool1} predictions from: {tool1_file}")
    df1 = load_predictions(tool1_file, args.tool1)
    print(f"  Loaded {len(df1)} cells")

    # Load tool 2 predictions
    tool2_pattern = tool_file_patterns[args.tool2]
    tool2_files = list(Path(args.data_dir).glob(f"{tool2_pattern}*{args.dataset}*.csv"))

    if not tool2_files:
        raise FileNotFoundError(f"No {args.tool2} prediction files found for {args.dataset}")

    tool2_file = tool2_files[0]
    print(f"Loading {args.tool2} predictions from: {tool2_file}")
    df2 = load_predictions(tool2_file, args.tool2)
    print(f"  Loaded {len(df2)} cells")

    # Create contingency table
    print(f"\nCreating contingency table...")
    contingency = create_contingency_table(df1, df2, args.tool1, args.tool2)

    # Display table
    print("\nContingency Table:")
    print(contingency)

    # Save table
    output_file = os.path.join(
        args.output_dir,
        f"contingency_{args.tool1}_vs_{args.tool2}_{args.dataset}.csv"
    )
    contingency.to_csv(output_file)
    print(f"\n✓ Saved to: {output_file}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total cells: {contingency.sum().sum()}")
    print(f"  {args.tool1} phases: {list(contingency.index)}")
    print(f"  {args.tool2} phases: {list(contingency.columns)}")

    print("\n" + "=" * 80)
    print("✅ Contingency table created successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
