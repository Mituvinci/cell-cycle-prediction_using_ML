#!/usr/bin/env python3
"""
Measure sparsity (percentage of zeros) in all training datasets.
This determines which scaling method to use during evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Get project root
project_root = Path(__file__).resolve().parent.parent

def load_gene_list(gene_list_path):
    """Load gene list from text file."""
    with open(gene_list_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes

def calculate_sparsity(df):
    """Calculate percentage of zero values in dataframe."""
    total_values = df.size
    zero_count = (df == 0).sum().sum()
    sparsity = (zero_count / total_values) * 100
    return sparsity

def measure_dataset_sparsity(dataset_name, data_path, gene_list_path):
    """
    Measure sparsity for a specific training dataset.

    Returns:
        tuple: (dict with dataset stats, expression dataframe for plotting)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")

    # Load training data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    # First column is cell ID or index, last column is phase label
    if 'phase' in df.columns:
        expression_cols = [col for col in df.columns if col not in ['Unnamed: 0', 'phase', 'Cell', 'cell_id']]
    else:
        # Assume last column is label
        expression_cols = df.columns[1:-1]

    print(f"  Expression columns: {len(expression_cols)}")

    # Load gene list
    print(f"Loading gene list from: {gene_list_path}")
    gene_list = load_gene_list(gene_list_path)
    print(f"  Genes in list: {len(gene_list)}")

    # Filter to training genes (case-insensitive matching)
    gene_map = {gene.upper(): gene for gene in expression_cols}
    matched_genes = []
    for gene in gene_list:
        gene_upper = gene.upper()
        if gene_upper in gene_map:
            matched_genes.append(gene_map[gene_upper])

    print(f"  Matched genes: {len(matched_genes)}")

    if len(matched_genes) == 0:
        print("  ERROR: No genes matched!")
        return None, None

    # Filter expression data to matched genes
    expr_df = df[matched_genes]

    # Calculate sparsity
    sparsity = calculate_sparsity(expr_df)

    # Calculate additional stats
    total_cells = len(expr_df)
    total_genes = len(matched_genes)
    total_values = expr_df.size
    zero_count = (expr_df == 0).sum().sum()
    nonzero_count = total_values - zero_count

    mean_genes_per_cell = (expr_df != 0).sum(axis=1).mean()
    median_genes_per_cell = (expr_df != 0).sum(axis=1).median()

    print(f"\nResults for {dataset_name}:")
    print(f"  Cells: {total_cells:,}")
    print(f"  Genes: {total_genes:,}")
    print(f"  Total values: {total_values:,}")
    print(f"  Zero values: {zero_count:,} ({sparsity:.2f}%)")
    print(f"  Non-zero values: {nonzero_count:,} ({100-sparsity:.2f}%)")
    print(f"  Mean genes detected per cell: {mean_genes_per_cell:.1f}")
    print(f"  Median genes detected per cell: {median_genes_per_cell:.1f}")

    stats = {
        'Dataset': dataset_name,
        'Cells': total_cells,
        'Genes': total_genes,
        'Total_Values': total_values,
        'Zero_Values': zero_count,
        'NonZero_Values': nonzero_count,
        'Sparsity_%': round(sparsity, 2),
        'NonZero_%': round(100 - sparsity, 2),
        'Mean_Genes_Per_Cell': round(mean_genes_per_cell, 1),
        'Median_Genes_Per_Cell': round(median_genes_per_cell, 1)
    }

    return stats, expr_df

def plot_sparsity_comparison(summary_df, output_dir):
    """Plot 1: Sparsity comparison bar plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#d62728' if x >= 65 else '#2ca02c' for x in summary_df['Sparsity_%']]

    bars = ax.bar(summary_df['Dataset'], summary_df['Sparsity_%'], color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=65, color='orange', linestyle='--', linewidth=2, label='Threshold (65%)')
    ax.set_ylabel('Sparsity (% Zeros)', fontweight='bold')
    ax.set_xlabel('Training Dataset', fontweight='bold')
    ax.set_title('Training Data Sparsity Comparison', fontweight='bold', fontsize=13)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    output_file = output_dir / '1_sparsity_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_zero_nonzero_comparison(summary_df, output_dir):
    """Plot 2: Stacked bar plot showing zero vs non-zero percentages."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(summary_df))
    width = 0.6

    p1 = ax.bar(x, summary_df['Sparsity_%'], width, label='Zeros', color='#d62728', alpha=0.7)
    p2 = ax.bar(x, summary_df['NonZero_%'], width, bottom=summary_df['Sparsity_%'],
                label='Non-Zeros', color='#2ca02c', alpha=0.7)

    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xlabel('Training Dataset', fontweight='bold')
    ax.set_title('Zero vs Non-Zero Value Distribution', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Dataset'], rotation=45, ha='right')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / '2_zero_nonzero_stacked.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_gene_detection_rates(summary_df, output_dir):
    """Plot 3: Mean genes detected per cell."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4'] * len(summary_df)
    bars = ax.bar(summary_df['Dataset'], summary_df['Mean_Genes_Per_Cell'],
                  color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('Mean Genes Detected per Cell', fontweight='bold')
    ax.set_xlabel('Training Dataset', fontweight='bold')
    ax.set_title('Gene Detection Rates Across Training Datasets', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_file = output_dir / '3_gene_detection_rates.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_expression_distributions(expression_data_dict, output_dir):
    """Plot 4: Histograms of non-zero expression values for each dataset."""
    n_datasets = len(expression_data_dict)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (dataset_name, expr_df) in enumerate(expression_data_dict.items()):
        ax = axes[idx]

        # Get non-zero values
        nonzero_values = expr_df.values.flatten()
        nonzero_values = nonzero_values[nonzero_values > 0]

        ax.hist(nonzero_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Expression Value (log-normalized)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{dataset_name}\n(n={len(nonzero_values):,} non-zero values)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add statistics text
        stats_text = f'Mean: {nonzero_values.mean():.2f}\nMedian: {np.median(nonzero_values):.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file = output_dir / '4_expression_histograms.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def generate_diagnostic_report(summary_df, output_dir):
    """Generate text diagnostic report."""
    report_file = output_dir / 'TRAINING_SPARSITY_REPORT.txt'

    high_sparsity = summary_df[summary_df['Sparsity_%'] >= 65]
    low_sparsity = summary_df[summary_df['Sparsity_%'] < 65]

    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING DATA SPARSITY DIAGNOSTIC REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("PURPOSE\n")
        f.write("-"*70 + "\n")
        f.write("This report documents sparsity (% of zero values) in all training datasets.\n")
        f.write("Sparsity determines which scaling method to use during model evaluation.\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("PLATFORM GROUPINGS\n")
        f.write("-"*70 + "\n\n")

        f.write("HIGH SPARSITY DATASETS (>=65% zeros):\n")
        f.write(f"  Platform: 10x Chromium multiomics\n")
        f.write(f"  Datasets: {', '.join(high_sparsity['Dataset'].tolist())}\n")
        f.write(f"  Sparsity range: {high_sparsity['Sparsity_%'].min():.1f}% - {high_sparsity['Sparsity_%'].max():.1f}%\n")
        f.write(f"  Mean gene detection: {high_sparsity['Mean_Genes_Per_Cell'].mean():.0f} genes/cell\n\n")

        f.write("LOW SPARSITY DATASETS (<65% zeros):\n")
        f.write(f"  Platform: Smart-seq2, Fluidigm C1, etc.\n")
        f.write(f"  Datasets: {', '.join(low_sparsity['Dataset'].tolist())}\n")
        f.write(f"  Sparsity range: {low_sparsity['Sparsity_%'].min():.1f}% - {low_sparsity['Sparsity_%'].max():.1f}%\n")
        f.write(f"  Mean gene detection: {low_sparsity['Mean_Genes_Per_Cell'].mean():.0f} genes/cell\n\n")

        f.write("SCALING METHOD RECOMMENDATIONS\n")
        f.write("-"*70 + "\n\n")

        f.write("RULE 1: Same Sparsity -> SIMPLE Scaling\n")
        f.write("  When training and benchmark have similar sparsity (difference <10%)\n")
        f.write("  Example: REH model (76% sparsity) on SUP benchmark (71% sparsity)\n")
        f.write("  Action: Use SIMPLE scaling (scaler only)\n\n")

        f.write("RULE 2: Different Sparsity -> DOUBLE or TRIPLE Scaling\n")
        f.write("  When training and benchmark have different sparsity (difference >10%)\n")
        f.write("  Example: REH model (76% sparsity) on GSE146773 benchmark (35% sparsity)\n")
        f.write("  Action: Use DOUBLE or TRIPLE scaling (domain adaptation)\n\n")

        f.write("IMPLEMENTATION PLAN\n")
        f.write("-"*70 + "\n")
        f.write("1. Hard-code sparsity values in evaluate_models.py:\n\n")
        f.write("   TRAINING_SPARSITY = {\n")
        for _, row in summary_df.iterrows():
            f.write(f"       '{row['Dataset'].lower()}': {row['Sparsity_%']},\n")
        f.write("   }\n\n")

        f.write("2. During evaluation:\n")
        f.write("   a. Detect training dataset from model filename\n")
        f.write("   b. Look up training sparsity\n")
        f.write("   c. Calculate benchmark sparsity\n")
        f.write("   d. Compare sparsity difference\n")
        f.write("   e. Auto-select scaling method:\n")
        f.write("      - Difference <10% -> simple\n")
        f.write("      - Difference 10-25% -> double\n")
        f.write("      - Difference >25% -> triple\n\n")

        f.write("3. Allow manual override with --scaling_method argument\n\n")

        f.write("SCIENTIFIC VALIDITY\n")
        f.write("-"*70 + "\n")
        f.write("- Sparsity differences are due to technical platform effects\n")
        f.write("- 10x Chromium has higher dropout rate (70-77% zeros)\n")
        f.write("- Smart-seq2 has higher sensitivity (35-50% zeros)\n")
        f.write("- Domain adaptation (double/triple scaling) addresses this mismatch\n")
        f.write("- This is scientifically valid and improves cross-platform accuracy\n\n")

        f.write("="*70 + "\n")

    print(f"  Saved: {report_file}")


def plot_sparsity_by_platform(summary_df, output_dir):
    """Plot 5: Grouped bar plot by platform type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define platform groups
    high_sparsity = summary_df[summary_df['Sparsity_%'] >= 65].copy()
    low_sparsity = summary_df[summary_df['Sparsity_%'] < 65].copy()

    x_high = np.arange(len(high_sparsity))
    x_low = np.arange(len(low_sparsity)) + len(high_sparsity) + 0.5

    bars1 = ax.bar(x_high, high_sparsity['Sparsity_%'], color='#d62728', alpha=0.7,
                   label='High Sparsity (10x Chromium)', edgecolor='black')
    bars2 = ax.bar(x_low, low_sparsity['Sparsity_%'], color='#2ca02c', alpha=0.7,
                   label='Low Sparsity (Smart-seq2/Other)', edgecolor='black')

    all_datasets = list(high_sparsity['Dataset']) + list(low_sparsity['Dataset'])
    all_x = list(x_high) + list(x_low)

    ax.set_ylabel('Sparsity (% Zeros)', fontweight='bold')
    ax.set_xlabel('Training Dataset', fontweight='bold')
    ax.set_title('Training Data Sparsity Grouped by Platform Type', fontweight='bold', fontsize=13)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_datasets, rotation=45, ha='right')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_file = output_dir / '5_sparsity_by_platform.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    """Measure sparsity for all training datasets."""

    print("="*70)
    print("TRAINING DATA SPARSITY MEASUREMENT")
    print("="*70)

    # Define datasets to analyze
    datasets = [
        {
            'name': 'REH',
            'data_path': project_root / '1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv',
            'gene_list': project_root / 'gene_lists/reh_3benchmark_sup.txt'
        },
        {
            'name': 'SUP-B15',
            'data_path': project_root / '1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv',
            'gene_list': project_root / 'gene_lists/reh_3benchmark_sup.txt'
        },
        {
            'name': 'PBMC',
            'data_path': project_root / '1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv',
            'gene_list': project_root / 'models/human_pbmc/adaboost/adaboost_NFT_pbmc_fld_1_features.txt'
        },
        {
            'name': 'Mouse_Brain',
            'data_path': project_root / '1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data.csv',
            'gene_list': project_root / 'models/mouse_brain/adaboost/adaboost_NFT_mouse_brain_fld_1_features.txt'
        },
        {
            'name': 'Nestorova',
            'data_path': project_root / '1_consensus_labeling/assign/final_training_data_nestorova/nestorova_training_data.csv',
            'gene_list': project_root / 'models/nestorova/adaboost/adaboost_NFT_nestorova_fld_1_features.txt'
        },
        {
            'name': 'HPSC',
            'data_path': project_root / '1_consensus_labeling/assign/final_training_data_gse75748/gse75748_training_data.csv',
            'gene_list': project_root / 'models/human_hpsc_5770_standard/adaboost/adaboost_NFT_hpsc_fld_1_features.txt'
        }
    ]

    # Measure sparsity for each dataset
    results = []
    expression_data_dict = {}
    for ds in datasets:
        result, expr_df = measure_dataset_sparsity(
            dataset_name=ds['name'],
            data_path=ds['data_path'],
            gene_list_path=ds['gene_list']
        )
        if result is not None:
            results.append(result)
            expression_data_dict[ds['name']] = expr_df

    # Create summary dataframe
    summary_df = pd.DataFrame(results)

    # Sort by sparsity
    summary_df = summary_df.sort_values('Sparsity_%', ascending=False)

    # Save results
    output_dir = project_root / 'distribution_analysis' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'training_sparsity_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"CSV Results saved to: {output_file}")
    print(f"{'='*70}")

    # Generate diagnostic report
    print(f"\n{'='*70}")
    print("GENERATING DIAGNOSTIC REPORT")
    print(f"{'='*70}")
    generate_diagnostic_report(summary_df, output_dir)

    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS (300 DPI)")
    print(f"{'='*70}")

    plot_sparsity_comparison(summary_df, output_dir)
    plot_zero_nonzero_comparison(summary_df, output_dir)
    plot_gene_detection_rates(summary_df, output_dir)
    plot_expression_distributions(expression_data_dict, output_dir)
    plot_sparsity_by_platform(summary_df, output_dir)

    print(f"\n{'='*70}")
    print("All plots generated successfully!")
    print(f"{'='*70}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE (sorted by sparsity)")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Print groupings
    print("\n" + "="*70)
    print("GROUPINGS FOR SCALING METHOD SELECTION")
    print("="*70)

    high_sparsity = summary_df[summary_df['Sparsity_%'] >= 65]
    low_sparsity = summary_df[summary_df['Sparsity_%'] < 65]

    print("\nHIGH SPARSITY DATASETS (>=65% zeros):")
    print("  Platform: 10x Chromium multiomics")
    print("  Datasets:", ', '.join(high_sparsity['Dataset'].tolist()))
    print("  Sparsity range:", f"{high_sparsity['Sparsity_%'].min():.1f}% - {high_sparsity['Sparsity_%'].max():.1f}%")
    print("  Scaling recommendation: When evaluating on similar high-sparsity benchmarks -> SIMPLE")
    print("                         When evaluating on low-sparsity benchmarks -> DOUBLE or TRIPLE")

    print("\nLOW SPARSITY DATASETS (<65% zeros):")
    print("  Platform: Smart-seq2, Fluidigm C1, etc.")
    print("  Datasets:", ', '.join(low_sparsity['Dataset'].tolist()))
    print("  Sparsity range:", f"{low_sparsity['Sparsity_%'].min():.1f}% - {low_sparsity['Sparsity_%'].max():.1f}%")
    print("  Scaling recommendation: When evaluating on similar low-sparsity benchmarks -> SIMPLE")
    print("                         When evaluating on high-sparsity benchmarks -> DOUBLE or TRIPLE (reverse)")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review sparsity values above")
    print("2. Hard-code these values in evaluate_models.py")
    print("3. Implement automatic scaling method selection based on sparsity matching")
    print("="*70)

if __name__ == '__main__':
    main()
