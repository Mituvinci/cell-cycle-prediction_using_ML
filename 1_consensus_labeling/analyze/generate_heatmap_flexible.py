#!/usr/bin/env python3
"""
Generate Obs/Expected Heatmaps - Flexible Version
===================================================

This script generates observed/expected ratio heatmaps from contingency tables.
You will manually inspect these heatmaps to determine phase mappings.

Usage:
    python generate_heatmap_flexible.py \
        --contingency-table results/contingency_seurat_vs_tricycle_mouse.csv \
        --output-dir results/heatmaps/ \
        --dataset-name mouse_brain

Author: Halima Akhter
Date: 2025-11-28
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_obs_expected_ratios(contingency):
    """
    Calculate observed/expected ratios from contingency table.

    The ratio indicates how much more (or less) likely a phase combination
    occurs compared to random chance.

    Parameters:
    -----------
    contingency : pd.DataFrame
        Contingency table (crosstab)

    Returns:
    --------
    pd.DataFrame
        Observed/expected ratio matrix
    """
    total = contingency.sum().sum()
    expected = np.outer(contingency.sum(axis=1), contingency.sum(axis=0)) / total

    # Avoid division by zero
    expected = expected + 1e-10

    # Calculate ratios
    obs_exp_ratio = contingency.values / expected

    # Convert to DataFrame with same index/columns
    ratio_df = pd.DataFrame(
        obs_exp_ratio,
        index=contingency.index,
        columns=contingency.columns
    )

    return ratio_df


def plot_heatmap(ratio_df, tool1_name, tool2_name, output_path, dataset_name):
    """
    Plot observed/expected ratio heatmap.

    Parameters:
    -----------
    ratio_df : pd.DataFrame
        Observed/expected ratio matrix
    tool1_name : str
        Name of tool 1 (rows)
    tool2_name : str
        Name of tool 2 (columns)
    output_path : str
        Path to save figure
    dataset_name : str
        Dataset name for title
    """
    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        ratio_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=1.0,
        vmin=0,
        vmax=3,
        cbar_kws={'label': 'Observed/Expected Ratio'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(f'Obs/Expected Ratios: {tool1_name} vs {tool2_name}\nDataset: {dataset_name}',
              fontsize=14, fontweight='bold')
    plt.xlabel(f'{tool2_name} Phases', fontsize=12, fontweight='bold')
    plt.ylabel(f'{tool1_name} Phases', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate observed/expected ratio heatmaps from contingency tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mouse: Generate heatmap for Seurat vs Tricycle
  python generate_heatmap_flexible.py \\
    --contingency-table results/contingency_seurat_vs_tricycle_mouse_brain.csv \\
    --tool1-name seurat \\
    --tool2-name tricycle \\
    --dataset-name mouse_brain \\
    --output-dir results/heatmaps/

INSTRUCTIONS FOR MANUAL PHASE MAPPING:
========================================
1. Look at the heatmap colors (green = high co-occurrence, red = low)
2. Identify which sub-phases map to G1, S, G2M based on color patterns
3. Create YAML config file with your mappings

Example interpretation:
  - If "G1.S" has high ratio (green) with "S", map G1.S → S
  - If "G2" has high ratio (green) with "G2M", map G2 → G2M
  - If "M.G1" has high ratio (green) with "G1", map M.G1 → G1
        """
    )

    parser.add_argument(
        '--contingency-table',
        type=str,
        required=True,
        help='Path to contingency table CSV file'
    )

    parser.add_argument(
        '--tool1-name',
        type=str,
        required=True,
        help='Name of tool 1 (row labels)'
    )

    parser.add_argument(
        '--tool2-name',
        type=str,
        required=True,
        help='Name of tool 2 (column labels)'
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Dataset name for plot title'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/heatmaps',
        help='Output directory for heatmap figures'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Generating Obs/Expected Ratio Heatmap")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Tool 1: {args.tool1_name}")
    print(f"Tool 2: {args.tool2_name}")
    print()

    # Load contingency table
    print(f"Loading contingency table from: {args.contingency_table}")
    contingency = pd.read_csv(args.contingency_table, index_col=0)
    print(f"  Dimensions: {contingency.shape}")

    # Calculate obs/expected ratios
    print("\nCalculating observed/expected ratios...")
    ratio_df = calculate_obs_expected_ratios(contingency)

    print("\nObs/Expected Ratio Matrix:")
    print(ratio_df)

    # Plot heatmap
    output_filename = f"heatmap_{args.tool1_name}_vs_{args.tool2_name}_{args.dataset_name}.png"
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"\nGenerating heatmap...")
    plot_heatmap(ratio_df, args.tool1_name, args.tool2_name, output_path, args.dataset_name)

    # Save ratio matrix as CSV
    ratio_csv_path = os.path.join(
        args.output_dir,
        f"obs_expected_ratios_{args.tool1_name}_vs_{args.tool2_name}_{args.dataset_name}.csv"
    )
    ratio_df.to_csv(ratio_csv_path)
    print(f"✓ Saved ratio matrix to: {ratio_csv_path}")

    print("\n" + "=" * 80)
    print("✅ Heatmap generated successfully!")
    print("=" * 80)
    print("\n🔍 NEXT STEPS:")
    print("1. Open the heatmap image and inspect color patterns")
    print("2. Identify which sub-phases map to G1, S, G2M")
    print("3. Create YAML config file with your mappings")
    print(f"4. Heatmap location: {output_path}")


if __name__ == '__main__':
    main()
