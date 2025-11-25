"""
Generate Heatmaps for Phase Mapping Analysis
=============================================

This script:
1. Calculates observed/expected ratios from contingency tables
2. Generates heatmaps to visualize phase associations
3. Saves heatmaps for manual phase assignment decisions

Usage:
    python generate_heatmaps.py --contingency-dir results/contingency_tables/ --output-dir results/phase_assignment_heatmaps/

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_obs_expect(contingency_table):
    """
    Calculate observed/expected ratios from contingency table.

    Formula:
        Expected[i,j] = (Row_Total[i] * Col_Total[j]) / Grand_Total
        Ratio[i,j] = Observed[i,j] / Expected[i,j]

    Parameters:
    -----------
    contingency_table : pd.DataFrame
        Contingency table (observed counts)

    Returns:
    --------
    obs_over_exp : pd.DataFrame
        Observed/expected ratios
    expected_table : pd.DataFrame
        Expected counts under independence
    """
    # Calculate totals
    row_sums = contingency_table.sum(axis=1)
    col_sums = contingency_table.sum(axis=0)
    total = contingency_table.sum().sum()

    # Calculate expected counts
    expected_table = pd.DataFrame(
        index=contingency_table.index,
        columns=contingency_table.columns
    )

    for i in contingency_table.index:
        for j in contingency_table.columns:
            expected_table.at[i, j] = (row_sums[i] * col_sums[j]) / total

    # Calculate observed/expected ratios
    obs_over_exp = contingency_table / expected_table

    # Replace inf and nan with 0
    obs_over_exp = obs_over_exp.replace([np.inf, -np.inf], 0)
    obs_over_exp = obs_over_exp.fillna(0)

    return obs_over_exp, expected_table


def create_heatmap(obs_over_exp, tool1_name, tool2_name, dataset, output_path):
    """
    Create and save heatmap visualization of obs/expect ratios.

    Parameters:
    -----------
    obs_over_exp : pd.DataFrame
        Observed/expected ratios
    tool1_name : str
        Name of reference tool (rows)
    tool2_name : str
        Name of comparison tool (columns)
    dataset : str
        Dataset name
    output_path : str
        Path to save the heatmap
    """
    # Create figure
    plt.figure(figsize=(12, 8))

    # Create heatmap
    # Use coolwarm colormap: blue=low ratio, white=1.0, red=high ratio
    sns.heatmap(
        obs_over_exp,
        annot=True,  # Show values
        fmt='.2f',   # 2 decimal places
        cmap='coolwarm',
        center=1.0,  # Center colormap at 1.0 (random association)
        vmin=0,
        vmax=3.0,   # Cap at 3.0 for better visualization
        cbar_kws={'label': 'Observed / Expected Ratio'},
        linewidths=0.5,
        linecolor='gray'
    )

    # Formatting
    plt.title(
        f'Phase Association Heatmap: {tool1_name.upper()} vs {tool2_name.upper()}\n'
        f'Dataset: {dataset.upper()}',
        fontsize=14,
        fontweight='bold'
    )
    plt.xlabel(f'{tool2_name.upper()} Phases', fontsize=12, fontweight='bold')
    plt.ylabel(f'{tool1_name.upper()} Phases', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved heatmap: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate heatmaps for phase mapping analysis'
    )
    parser.add_argument(
        '--contingency-dir',
        type=str,
        default='../results/contingency_tables',
        help='Directory containing contingency table CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../results/phase_assignment_heatmaps',
        help='Output directory for heatmap images'
    )
    parser.add_argument(
        '--save-obs-expect',
        action='store_true',
        help='Also save obs/expect ratio tables as CSV'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_obs_expect:
        obs_expect_dir = os.path.join(args.output_dir, 'obs_expect_tables')
        os.makedirs(obs_expect_dir, exist_ok=True)

    print("=" * 80)
    print("Generating Phase Mapping Heatmaps")
    print("=" * 80)
    print(f"Input directory: {args.contingency_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Find all contingency table files
    contingency_files = list(Path(args.contingency_dir).glob("contingency_*.csv"))

    if not contingency_files:
        raise FileNotFoundError(f"No contingency tables found in {args.contingency_dir}")

    print(f"Found {len(contingency_files)} contingency tables\n")

    # Process each contingency table
    for contingency_file in contingency_files:
        # Parse filename: contingency_tool1_vs_tool2_dataset.csv
        filename = contingency_file.stem  # Remove .csv
        parts = filename.split('_')

        if len(parts) >= 5 and parts[0] == 'contingency':
            tool1_name = parts[1]
            # tool2_name = parts[3]
            dataset = parts[-1]
            tool2_name = '_'.join(parts[3:-1])  # Handle multi-word tool names

            print(f"Processing: {tool1_name} vs {tool2_name} ({dataset})")

            # Load contingency table
            contingency_table = pd.read_csv(contingency_file, index_col=0)
            print(f"  Shape: {contingency_table.shape}")

            # Calculate obs/expect ratios
            obs_over_exp, expected_table = calculate_obs_expect(contingency_table)

            # Save obs/expect table if requested
            if args.save_obs_expect:
                obs_exp_file = os.path.join(
                    obs_expect_dir,
                    f"obs_expect_{tool1_name}_vs_{tool2_name}_{dataset}.csv"
                )
                obs_over_exp.to_csv(obs_exp_file)
                print(f"  ✓ Saved obs/expect table: {obs_exp_file}")

            # Create heatmap
            output_path = os.path.join(
                args.output_dir,
                f"heatmap_{tool1_name}_vs_{tool2_name}_{dataset}.png"
            )
            create_heatmap(obs_over_exp, tool1_name, tool2_name, dataset, output_path)

            print()

    print("=" * 80)
    print("✅ All heatmaps generated successfully!")
    print("=" * 80)
    print("\n📊 Next Steps:")
    print("  1. View heatmaps in:", args.output_dir)
    print("  2. Look for RED cells (high ratios > 2.0) = strong associations")
    print("  3. Update phase mappings in: configs/phase_mappings/*.yaml")
    print("  4. Run: python apply_phase_reassignment.py")


if __name__ == '__main__':
    main()
