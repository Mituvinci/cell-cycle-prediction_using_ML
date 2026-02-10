#!/usr/bin/env python3
"""
Phase 1: Distribution Diagnostic Analysis
==========================================
Compare expression distributions between REH/SUP training data and benchmarks.

Purpose:
- Diagnose why REH-trained models perform poorly on external benchmarks
- Check if normalization methods/parameters are consistent
- Identify distribution mismatches that explain performance gaps

Outputs:
- Summary statistics table (CSV)
- Distribution comparison plots (PNG, 300 DPI for publication)
- Diagnostic report (TXT)

Author: Distribution Analysis for Cross-Platform Cell Cycle Prediction
Date: 2026-02-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('default')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Paths
BASE_DIR = Path("/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq")
CELL_CYCLE_DIR = BASE_DIR / "cell_cycle_prediction"
BENCHMARK_DIR = BASE_DIR / "data" / "benchmarks_preprocessed"
OUTPUT_DIR = CELL_CYCLE_DIR / "distribution_analysis" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gene list (intersection of REH + 3 benchmarks + SUP)
GENE_LIST_FILE = CELL_CYCLE_DIR / "gene_lists" / "reh_3benchmark_sup.txt"

# Dataset paths
DATASETS = {
    'REH': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv",
        'type': 'Training',
        'platform': '10x Chromium',
        'color': '#1f77b4'
    },
    'SUP': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv",
        'type': 'Training',
        'platform': '10x Chromium',
        'color': '#ff7f0e'
    },
    'GSE146773': {
        'path': BENCHMARK_DIR / "GSE146773/GSE146773_expression.csv",
        'type': 'Benchmark',
        'platform': 'Smart-seq2',
        'color': '#2ca02c'
    },
    'GSE64016': {
        'path': BENCHMARK_DIR / "GSE64016/GSE64016_expression.csv",
        'type': 'Benchmark',
        'platform': 'Fluidigm C1',
        'color': '#d62728'
    },
    'Buettner_mESC': {
        'path': BENCHMARK_DIR / "Buettner_mESC/Buettner_mESC_expression.csv",
        'type': 'Benchmark',
        'platform': 'E-MTAB-2805 (mESC-SMARTer)',
        'color': '#9467bd'
    }
}


def load_gene_list(gene_list_file):
    """Load gene list (intersection of training + benchmarks)."""
    print(f"\nLoading gene list: {gene_list_file}")
    with open(gene_list_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(genes)} genes (intersection)")
    return genes


def load_dataset(dataset_name, info, gene_list=None):
    """Load dataset and extract expression matrix, filtered to gene list."""
    print(f"Loading {dataset_name}...")

    try:
        df = pd.read_csv(info['path'], nrows=2)
        print(f"  Preview columns: {list(df.columns[:5])}")

        # Load full data
        df = pd.read_csv(info['path'], index_col=0)

        # Remove label/metadata columns if present
        metadata_cols = ['Predicted', 'gex_barcode', 'CellID', 'paper_phase', 'Labeled']
        expr_df = df.drop(columns=[col for col in metadata_cols if col in df.columns], errors='ignore')

        # Keep only numeric columns
        expr_df = expr_df.select_dtypes(include=[np.number])

        print(f"  Before filtering: {expr_df.shape[0]} cells x {expr_df.shape[1]} genes")

        # Filter to gene list if provided
        if gene_list is not None:
            # Try matching genes in different case formats
            available_genes = set(expr_df.columns)

            # Create case-insensitive mapping
            gene_map = {gene.upper(): gene for gene in available_genes}

            # Find matching genes
            matched_genes = []
            missing_genes = []
            for gene in gene_list:
                gene_upper = gene.upper()
                if gene_upper in gene_map:
                    matched_genes.append(gene_map[gene_upper])
                else:
                    missing_genes.append(gene)

            expr_df = expr_df[matched_genes]

            print(f"  After filtering to gene list: {expr_df.shape[0]} cells x {expr_df.shape[1]} genes")
            print(f"  Matched: {len(matched_genes)} genes, Missing: {len(missing_genes)} genes")

            if len(missing_genes) > 10:
                print(f"  First 10 missing genes: {missing_genes[:10]}")

        return expr_df

    except Exception as e:
        print(f"  ERROR loading {dataset_name}: {e}")
        return None


def compute_statistics(expr_df, dataset_name):
    """Compute comprehensive distribution statistics."""
    stats = {
        'Dataset': dataset_name,

        # Overall statistics
        'N_Cells': expr_df.shape[0],
        'N_Genes': expr_df.shape[1],

        # Expression value statistics (across all values)
        'Global_Min': expr_df.values.min(),
        'Global_Max': expr_df.values.max(),
        'Global_Mean': expr_df.values.mean(),
        'Global_Median': np.median(expr_df.values),
        'Global_Std': expr_df.values.std(),
        'Global_Q25': np.percentile(expr_df.values, 25),
        'Global_Q75': np.percentile(expr_df.values, 75),
        'Global_Q95': np.percentile(expr_df.values, 95),
        'Global_Q99': np.percentile(expr_df.values, 99),

        # Zero proportion (sparsity)
        'Pct_Zeros': (expr_df.values == 0).sum() / expr_df.size * 100,

        # Per-cell statistics (mean across cells)
        'Mean_Genes_Per_Cell': (expr_df > 0).sum(axis=1).mean(),
        'Mean_Total_Counts_Per_Cell': expr_df.sum(axis=1).mean(),

        # Per-gene statistics (mean across genes)
        'Mean_Cells_Per_Gene': (expr_df > 0).sum(axis=0).mean(),
        'Mean_Expression_Per_Gene': expr_df.mean(axis=0).mean(),

        # Check if log-normalized (values typically 0-10 range)
        'Likely_LogNormalized': 'Yes' if expr_df.values.max() < 15 else 'No',
        'Values_In_Log_Range': f"[{expr_df.values.min():.2f}, {expr_df.values.max():.2f}]"
    }

    return stats


def plot_distribution_comparison(data_dict, output_dir):
    """Generate comprehensive distribution comparison plots."""

    print("\nGenerating distribution plots...")

    # Plot 1: Overall expression distribution (histogram)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, expr_df) in enumerate(data_dict.items()):
        ax = axes[idx]

        # Flatten all expression values
        values = expr_df.values.flatten()

        # Remove zeros for clearer visualization
        values_nonzero = values[values > 0]

        ax.hist(values_nonzero, bins=50, alpha=0.7, color=DATASETS[name]['color'], edgecolor='black')
        ax.set_xlabel('Expression Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f"{name}\n({DATASETS[name]['platform']})", fontsize=11, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"Mean: {values.mean():.2f}\nMedian: {np.median(values):.2f}\nMax: {values.max():.2f}"
        ax.text(0.65, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "1_expression_histograms.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: 1_expression_histograms.png")
    plt.close()


    # Plot 2: Box plots comparing distributions
    fig, ax = plt.subplots(figsize=(12, 6))

    all_data = []
    labels = []
    colors = []

    for name, expr_df in data_dict.items():
        # Sample 10000 values per dataset for visualization
        values = expr_df.values.flatten()
        values_nonzero = values[values > 0]
        sample = np.random.choice(values_nonzero, size=min(10000, len(values_nonzero)), replace=False)

        all_data.append(sample)
        labels.append(f"{name}\n({DATASETS[name]['platform']})")
        colors.append(DATASETS[name]['color'])

    bp = ax.boxplot(all_data, labels=labels, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Expression Value (non-zero)', fontsize=12)
    ax.set_title('Expression Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "2_expression_boxplots.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: 2_expression_boxplots.png")
    plt.close()


    # Plot 3: Violin plots
    fig, ax = plt.subplots(figsize=(12, 6))

    parts = ax.violinplot(all_data, positions=range(1, len(all_data)+1), showmeans=True, showmedians=True)

    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[idx])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_ylabel('Expression Value (non-zero)', fontsize=12)
    ax.set_title('Expression Distribution Comparison (Violin Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "3_expression_violinplots.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: 3_expression_violinplots.png")
    plt.close()


    # Plot 4: Quantile-Quantile plots (compare each benchmark to REH)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    reh_values = data_dict['REH'].values.flatten()
    reh_values_nonzero = reh_values[reh_values > 0]
    reh_quantiles = np.percentile(reh_values_nonzero, np.arange(0, 101, 1))

    compare_datasets = ['SUP', 'GSE146773', 'GSE64016', 'Buettner_mESC']

    for idx, name in enumerate(compare_datasets):
        ax = axes[idx]

        other_values = data_dict[name].values.flatten()
        other_values_nonzero = other_values[other_values > 0]
        other_quantiles = np.percentile(other_values_nonzero, np.arange(0, 101, 1))

        ax.scatter(reh_quantiles, other_quantiles, alpha=0.6, s=30, color=DATASETS[name]['color'])

        # Add diagonal line (perfect match)
        min_val = min(reh_quantiles.min(), other_quantiles.min())
        max_val = max(reh_quantiles.max(), other_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')

        ax.set_xlabel('REH Quantiles', fontsize=11)
        ax.set_ylabel(f'{name} Quantiles', fontsize=11)
        ax.set_title(f'Q-Q Plot: REH vs {name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "4_qq_plots.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: 4_qq_plots.png")
    plt.close()


    # Plot 5: Per-cell statistics comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5a: Genes detected per cell
    ax = axes[0]
    for name, expr_df in data_dict.items():
        genes_per_cell = (expr_df > 0).sum(axis=1)
        ax.hist(genes_per_cell, bins=50, alpha=0.5, label=name, color=DATASETS[name]['color'])

    ax.set_xlabel('Genes Detected per Cell', fontsize=11)
    ax.set_ylabel('Number of Cells', fontsize=11)
    ax.set_title('Gene Detection Rate per Cell', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5b: Total counts per cell
    ax = axes[1]
    for name, expr_df in data_dict.items():
        total_counts = expr_df.sum(axis=1)
        ax.hist(total_counts, bins=50, alpha=0.5, label=name, color=DATASETS[name]['color'])

    ax.set_xlabel('Total Expression per Cell', fontsize=11)
    ax.set_ylabel('Number of Cells', fontsize=11)
    ax.set_title('Total Expression per Cell', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "5_per_cell_statistics.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: 5_per_cell_statistics.png")
    plt.close()


    # Plot 6: Sparsity comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    sparsity_data = []
    for name, expr_df in data_dict.items():
        pct_zeros = (expr_df.values == 0).sum() / expr_df.size * 100
        sparsity_data.append({'Dataset': name, 'Platform': DATASETS[name]['platform'],
                             'Type': DATASETS[name]['type'], 'Sparsity_%': pct_zeros})

    sparsity_df = pd.DataFrame(sparsity_data)

    x_pos = np.arange(len(sparsity_df))
    bars = ax.bar(x_pos, sparsity_df['Sparsity_%'],
                  color=[DATASETS[name]['color'] for name in sparsity_df['Dataset']],
                  alpha=0.7, edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['Dataset']}\n({row['Platform']})"
                        for _, row in sparsity_df.iterrows()], fontsize=10)
    ax.set_ylabel('Percentage of Zero Values (%)', fontsize=12)
    ax.set_title('Data Sparsity Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, sparsity_df['Sparsity_%']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "6_sparsity_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: 6_sparsity_comparison.png")
    plt.close()


def generate_diagnostic_report(stats_df, output_dir):
    """Generate text-based diagnostic report for reviewers."""

    report_path = output_dir / "DIAGNOSTIC_REPORT.txt"

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 1: DISTRIBUTION DIAGNOSTIC REPORT\n")
        f.write("Cross-Platform Cell Cycle Prediction Analysis\n")
        f.write("="*80 + "\n\n")

        f.write("IMPORTANT: Analysis performed on TRAINING GENES ONLY\n")
        f.write("-" * 80 + "\n")
        f.write("Gene list: gene_lists/reh_3benchmark_sup.txt\n")
        f.write(f"Number of genes: {stats_df.iloc[0]['N_Genes']:,.0f} (intersection)\n")
        f.write("This ensures fair comparison - analyzing the SAME genes used in training.\n\n")

        f.write("OBJECTIVE:\n")
        f.write("-----------\n")
        f.write("Diagnose why REH-trained models perform well on SUP (~80-90% accuracy)\n")
        f.write("but poorly on external benchmarks GSE146773, GSE64016, Buettner_mESC (<67%).\n\n")

        f.write("HYPOTHESIS:\n")
        f.write("-----------\n")
        f.write("Distribution mismatch between training data (REH/SUP) and benchmarks\n")
        f.write("due to different sequencing platforms and/or normalization parameters.\n\n")

        f.write("="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n\n")

        # Finding 1: Expression range comparison
        f.write("1. EXPRESSION VALUE RANGES:\n")
        f.write("-" * 40 + "\n")
        for _, row in stats_df.iterrows():
            f.write(f"  {row['Dataset']:15s}: [{row['Global_Min']:6.2f}, {row['Global_Max']:6.2f}]")
            f.write(f"  (Mean: {row['Global_Mean']:.2f}, Median: {row['Global_Median']:.2f})\n")

        # Check if ranges are similar
        max_range = stats_df['Global_Max'].max()
        min_range = stats_df['Global_Max'].min()
        if max_range / min_range > 1.5:
            f.write(f"\n  WARNING: Max values differ by {max_range/min_range:.1f}x across datasets!\n")
            f.write("  This suggests different normalization parameters or methods.\n")
        else:
            f.write("\n  OK: Expression ranges are relatively consistent.\n")

        f.write("\n")

        # Finding 2: Log normalization check
        f.write("2. LOG NORMALIZATION STATUS:\n")
        f.write("-" * 40 + "\n")
        for _, row in stats_df.iterrows():
            f.write(f"  {row['Dataset']:15s}: {row['Likely_LogNormalized']:3s}  ")
            f.write(f"(Range: {row['Values_In_Log_Range']})\n")

        if not all(stats_df['Likely_LogNormalized'] == 'Yes'):
            f.write("\n  WARNING: Not all datasets appear to be log-normalized!\n")
            f.write("  Log-normalized data typically has values in range [0, 10].\n")
        else:
            f.write("\n  OK: All datasets appear to be log-normalized.\n")

        f.write("\n")

        # Finding 3: Sparsity comparison
        f.write("3. DATA SPARSITY (% ZEROS):\n")
        f.write("-" * 40 + "\n")
        for _, row in stats_df.iterrows():
            f.write(f"  {row['Dataset']:15s}: {row['Pct_Zeros']:5.1f}%\n")

        max_sparsity = stats_df['Pct_Zeros'].max()
        min_sparsity = stats_df['Pct_Zeros'].min()
        if max_sparsity - min_sparsity > 20:
            f.write(f"\n  WARNING: Sparsity differs by {max_sparsity - min_sparsity:.1f}% across datasets!\n")
            f.write("  Different platforms have different dropout rates.\n")

        f.write("\n")

        # Finding 4: Gene detection
        f.write("4. GENE DETECTION PER CELL:\n")
        f.write("-" * 40 + "\n")
        for _, row in stats_df.iterrows():
            f.write(f"  {row['Dataset']:15s}: {row['Mean_Genes_Per_Cell']:7.0f} genes/cell\n")

        f.write("\n")

        # Finding 5: Platform comparison
        f.write("5. PLATFORM SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write("  Training Data (perform well on each other):\n")
        f.write("    - REH: 10x Chromium, Multiomics\n")
        f.write("    - SUP: 10x Chromium, Multiomics\n")
        f.write("  Benchmarks (poor performance):\n")
        f.write("    - GSE146773: Smart-seq2, Higher depth\n")
        f.write("    - GSE64016:  Fluidigm C1 (H1-Fucci hESCs, 247 cells)\n")
        f.write("    - Buettner:  E-MTAB-2805 (mESC-SMARTer)\n")

        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("="*80 + "\n\n")

        # Generate recommendations based on findings
        f.write("Based on the distribution analysis:\n\n")

        if max_range / min_range > 1.5:
            f.write("[CRITICAL] 1. RE-NORMALIZE ALL DATASETS\n")
            f.write("   Problem: Expression value ranges differ significantly.\n")
            f.write("   Solution: Apply IDENTICAL Seurat log normalization to all datasets:\n")
            f.write("             - Same scale factor (10000)\n")
            f.write("             - Same log base (natural log)\n")
            f.write("             - Same pseudocount (1)\n\n")

        if max_sparsity - min_sparsity > 20:
            f.write("[IMPORTANT] 2. CONSIDER PLATFORM EFFECTS\n")
            f.write("   Problem: Sparsity differs dramatically across platforms.\n")
            f.write("   Options:\n")
            f.write("     a) Train separate models per platform\n")
            f.write("     b) Use platform-aware normalization\n")
            f.write("     c) Apply imputation before training\n\n")

        f.write("[SCIENTIFIC] 3. REPORT CROSS-PLATFORM LIMITATIONS\n")
        f.write("   67% accuracy may be the REALISTIC limit for cross-platform prediction.\n")
        f.write("   Compare your results to published tools on same benchmarks.\n\n")

        f.write("[MANUSCRIPT] 4. INCLUDE THESE ANALYSES IN SUPPLEMENT\n")
        f.write("   Reviewers will appreciate the thorough distribution analysis.\n")
        f.write("   Show that you investigated and addressed platform effects.\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"\nDiagnostic report saved: {report_path}")


def main():
    """Main execution function."""

    print("="*80)
    print("PHASE 1: DISTRIBUTION DIAGNOSTIC ANALYSIS")
    print("Using TRAINING GENE LIST (intersection of REH + 3 benchmarks + SUP)")
    print("="*80)
    print()

    # Load gene list (intersection used during training)
    gene_list = load_gene_list(GENE_LIST_FILE)

    # Load all datasets, filtered to gene list
    data_dict = {}
    for name, info in DATASETS.items():
        expr_df = load_dataset(name, info, gene_list=gene_list)
        if expr_df is not None:
            data_dict[name] = expr_df

    print()
    print("="*80)
    print("COMPUTING STATISTICS")
    print("="*80)

    # Compute statistics for all datasets
    stats_list = []
    for name, expr_df in data_dict.items():
        print(f"\nAnalyzing {name}...")
        stats = compute_statistics(expr_df, name)
        stats_list.append(stats)

    # Create summary table
    stats_df = pd.DataFrame(stats_list)

    # Reorder columns for better readability
    col_order = ['Dataset', 'N_Cells', 'N_Genes',
                 'Global_Min', 'Global_Max', 'Global_Mean', 'Global_Median', 'Global_Std',
                 'Global_Q25', 'Global_Q75', 'Global_Q95', 'Global_Q99',
                 'Pct_Zeros', 'Mean_Genes_Per_Cell', 'Mean_Total_Counts_Per_Cell',
                 'Mean_Cells_Per_Gene', 'Mean_Expression_Per_Gene',
                 'Likely_LogNormalized', 'Values_In_Log_Range']
    stats_df = stats_df[col_order]

    # Save statistics
    stats_path = OUTPUT_DIR / "distribution_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistics saved: {stats_path}")

    # Display summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(stats_df.to_string(index=False))

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    plot_distribution_comparison(data_dict, OUTPUT_DIR)

    # Generate diagnostic report
    print("\n" + "="*80)
    print("GENERATING DIAGNOSTIC REPORT")
    print("="*80)
    generate_diagnostic_report(stats_df, OUTPUT_DIR)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - distribution_statistics.csv       (Summary table)")
    print("  - DIAGNOSTIC_REPORT.txt             (For reviewers)")
    print("  - 1_expression_histograms.png       (Individual distributions)")
    print("  - 2_expression_boxplots.png         (Box plot comparison)")
    print("  - 3_expression_violinplots.png      (Violin plot comparison)")
    print("  - 4_qq_plots.png                    (Quantile-quantile plots)")
    print("  - 5_per_cell_statistics.png         (Gene detection & counts)")
    print("  - 6_sparsity_comparison.png         (Zero proportions)")
    print()


if __name__ == "__main__":
    main()
