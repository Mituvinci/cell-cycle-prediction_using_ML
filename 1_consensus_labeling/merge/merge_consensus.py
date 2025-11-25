#!/usr/bin/env python3
"""
Merge Consensus Labels
======================
Merges predictions from 4 cell cycle prediction tools (Seurat, Tricycle, ccAFv2, Revelio)
to create consensus labels based on agreement thresholds.

Functions:
1. identify_unique_rows() - Identifies cells with conflicting or unique predictions
2. find_common_rows() - Finds cells where ≥N tools agree on the same prediction

EXACT implementation from 1_2_Merge_three_tools.ipynb (Feb 19, 2025)
"""

import pandas as pd
import glob
import argparse
import os
from pathlib import Path


def identify_unique_rows(df1, df2, df3):
    """
    Identify rows where CellID appears in only one dataframe OR
    where the 'Predicted' values differ across all occurrences.

    Args:
        df1, df2, df3: DataFrames with 'CellID' and 'Predicted' columns

    Returns:
        DataFrame with unique/conflicting rows (Predicted='Unique' for NaN)
    """
    # Concatenate all dataframes and count occurrences of each CellID
    all_dfs = pd.concat([df1, df2, df3])
    cellid_counts = all_dfs['CellID'].value_counts()

    # Filter out CellIDs that occur in more than one file
    unique_cellids = cellid_counts[cellid_counts == 1].index

    # Get rows with unique CellIDs
    unique_rows = all_dfs[all_dfs['CellID'].isin(unique_cellids)]

    # For non-unique CellIDs, check if 'Predicted' values do not match across files
    non_unique_cellids = cellid_counts[cellid_counts > 1].index
    for cellid in non_unique_cellids:
        rows = all_dfs[all_dfs['CellID'] == cellid]

        # If all 'Predicted' values are different, add to unique_rows
        if len(rows['Predicted'].unique()) == len(rows):
            unique_rows = pd.concat([unique_rows, rows.iloc[:1]])

    # Fill NaN in 'Predicted' column with 'Unique'
    unique_rows['Predicted'].fillna('Unique', inplace=True)

    return unique_rows.drop_duplicates()


def find_common_rows(dfs, min_overlap):
    """
    Find common rows appearing in at least min_overlap files.
    Rows are considered common if they have the same CellID AND Predicted phase.

    Args:
        dfs: Dictionary of DataFrames (e.g., {"seurat": df1, "tricycle": df2, ...})
        min_overlap: Minimum number of tools that must agree (2, 3, or 4)

    Returns:
        DataFrame with cells where ≥min_overlap tools agree
    """
    df_list = list(dfs.values())  # Convert dictionary values to list
    combined = pd.concat(df_list).groupby(['CellID', 'Predicted']).filter(
        lambda x: len(x) >= min_overlap
    ).drop_duplicates()
    return combined


def load_reassigned_csvs(base_path, sample_name, file_patterns):
    """
    Load the reassigned CSV files from the assign/ directory.

    Args:
        base_path: Path to directory containing reassigned CSVs
        sample_name: Sample identifier (e.g., "1_GD428_21136_Hu_REH_Parental")
        file_patterns: Dictionary mapping tool names to file patterns

    Returns:
        Dictionary of loaded DataFrames
    """
    dfs = {}
    for key, pattern in file_patterns.items():
        csv_files = glob.glob(f"{base_path}/{pattern}")
        if csv_files:
            dfs[key] = pd.read_csv(csv_files[0])
            print(f"✓ Loaded {key}: {csv_files[0]}")
        else:
            print(f"⚠ Warning: No file found for {key} with pattern {pattern}")

    # Ensure at least two files are available
    if len(dfs) < 2:
        raise ValueError("Error: Less than two datasets found. Cannot proceed.")

    return dfs


def merge_consensus(input_dir, output_dir, sample_name, dataset_type):
    """
    Main function to merge consensus labels from 4 tools.

    Args:
        input_dir: Directory containing reassigned CSV files
        output_dir: Directory to save merged consensus files
        sample_name: Sample identifier (e.g., "1_GD428_21136_Hu_REH_Parental")
        dataset_type: "reh" or "sup"
    """
    # File patterns for each tool
    file_patterns = {
        "cellcyclescore": "cellcyclescore*.csv",
        "tricycle": "tricycle*.csv",
        "ccAFV2": "ccAFV2*.csv",
        "revelio": "revelio*.csv"
    }

    # Load CSV files
    print(f"\n📂 Loading reassigned CSV files from: {input_dir}")
    dfs = load_reassigned_csvs(input_dir, sample_name, file_patterns)
    print(f"✓ Loaded {len(dfs)} datasets")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find common rows appearing in at least 2, 3, and 4 files
    print("\n🔗 Merging consensus labels...")
    at_least_two = find_common_rows(dfs, 2)
    at_least_three = find_common_rows(dfs, 3)
    all_four = find_common_rows(dfs, 4) if len(dfs) == 4 else None

    # Save results
    output_prefix = os.path.join(output_dir, f"{sample_name}_")

    at_least_two_path = output_prefix + "overlapped_at_least_two_regions.csv"
    at_least_three_path = output_prefix + "overlapped_at_least_three_regions.csv"

    at_least_two.to_csv(at_least_two_path, index=False)
    at_least_three.to_csv(at_least_three_path, index=False)
    print(f"✓ Saved ≥2 tools agreement: {at_least_two_path} ({len(at_least_two)} cells)")
    print(f"✓ Saved ≥3 tools agreement: {at_least_three_path} ({len(at_least_three)} cells)")

    if all_four is not None:
        all_four_path = output_prefix + "overlapped_all_four_regions.csv"
        all_four.to_csv(all_four_path, index=False)
        print(f"✓ Saved all 4 tools agreement: {all_four_path} ({len(all_four)} cells)")

    # Identify unique/conflicting rows (requires exactly 3 dataframes for original logic)
    if len(dfs) >= 3:
        df_list = list(dfs.values())[:3]  # Take first 3 for unique row identification
        unique_rows_df = identify_unique_rows(df_list[0], df_list[1], df_list[2])
        unique_rows_path = output_prefix + "no_overlapped.csv"
        unique_rows_df.to_csv(unique_rows_path, index=False)
        print(f"✓ Saved unique/conflicting cells: {unique_rows_path} ({len(unique_rows_df)} cells)")

    # Print summary statistics
    print("\n📊 Summary:")
    print(f"  • At least 2 tools agree: {len(at_least_two)} cells")
    print(f"  • At least 3 tools agree: {len(at_least_three)} cells")
    if all_four is not None:
        print(f"  • All 4 tools agree: {len(all_four)} cells")
    if len(dfs) >= 3:
        print(f"  • Unique/conflicting: {len(unique_rows_df)} cells")

    print("\n✅ Consensus merging complete!")

    return {
        "at_least_two": at_least_two,
        "at_least_three": at_least_three,
        "all_four": all_four,
        "unique": unique_rows_df if len(dfs) >= 3 else None
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge consensus labels from 4 cell cycle prediction tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge REH consensus labels
  python merge_consensus.py \\
    --input ../assign/output/reh \\
    --output ./consensus \\
    --sample 1_GD428_21136_Hu_REH_Parental \\
    --dataset reh

  # Merge SUP-B15 consensus labels
  python merge_consensus.py \\
    --input ../assign/output/sup \\
    --output ./consensus \\
    --sample 2_GD444_21136_Hu_Sup_Parental \\
    --dataset sup
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing reassigned CSV files from assign/ step'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for consensus CSV files'
    )

    parser.add_argument(
        '--sample',
        type=str,
        required=True,
        help='Sample name (e.g., "1_GD428_21136_Hu_REH_Parental")'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['reh', 'sup'],
        required=True,
        help='Dataset type: "reh" or "sup"'
    )

    args = parser.parse_args()

    # Run consensus merging
    merge_consensus(
        input_dir=args.input,
        output_dir=args.output,
        sample_name=args.sample,
        dataset_type=args.dataset
    )


if __name__ == "__main__":
    main()
