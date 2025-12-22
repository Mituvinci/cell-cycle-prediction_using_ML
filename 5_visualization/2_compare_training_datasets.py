#!/usr/bin/env python3
"""
Compare Training and Benchmark Datasets - Cell Cycle Marker Gene Expression
=============================================================================

Analyzes cell cycle marker gene expression across training and benchmark datasets.
Publication-quality figures with Times New Roman font.

Consistent with plot_table1.csv naming and cell counts.

Author: Halima Akhter
Date: 2025-12-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# ============================================================================
# PUBLICATION-QUALITY SETTINGS
# ============================================================================

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

# High-quality output settings
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable in Illustrator)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.format'] = 'pdf'

# Professional styling
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

# Output directory
OUTPUT_DIR = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/5_visualization/diagnostic_results"
os.makedirs("diagnostic_results", exist_ok=True)

# =============================================================================
# DATA PATHS
# =============================================================================

# Marker genes CSV
PATH_MARKER_GENES = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/Cell_Cycle_marker_genes/markergenes.csv"

# Dataset configuration CSV
PATH_TABLE1_CSV = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/5_visualization/plot_table1.csv"

# File path mapping (dataset name prefix -> file path)
DATASET_FILE_PATHS = {
    'REH': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv",
    'SUP-B15': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv",
    'PBMC': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv",
    'Mouse Brain': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_full_consensus_with_genes.csv",
    'GSE75748': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/GSE75748_hPSC_final_training_matrix.csv",
    'GSE146773': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/GSE146773_seurat_normalized_gene_expression.csv",
    'GSE64016': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/GSE64016_seurat_normalized_gene_expression.csv",
    'Buettner mESC': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Buettner_mESC_benchmark_clean.csv"
}

def load_dataset_config():
    """Load dataset configuration from plot_table1.csv."""
    df_config = pd.read_csv(PATH_TABLE1_CSV)

    config = {}
    for _, row in df_config.iterrows():
        dataset_name = row['Dataset']

        # Find matching file path
        file_path = None
        for key, path in DATASET_FILE_PATHS.items():
            if key in dataset_name:
                file_path = path
                break

        if file_path:
            config[dataset_name] = {
                'path': file_path,
                'is_training': row['Is_Training'] == 'Yes',
                'expected_counts': {
                    'total': int(row['After_Consensus_Labeling']) if pd.notna(row['After_Consensus_Labeling']) else None,
                    'G1': int(row['G1']),
                    'S': int(row['S']),
                    'G2M': int(row['G2M'])
                }
            }

    return config

# =============================================================================
# LOAD CELL CYCLE MARKER GENES FROM CSV
# =============================================================================

def load_marker_genes():
    """
    Load ALL unique cell-cycle genes from markergenes.csv
    (Revelio + Seurat, all columns, no filtering)
    """
    print(f"\nLoading marker genes from: {PATH_MARKER_GENES}")
    df = pd.read_csv(PATH_MARKER_GENES)

    all_genes = set()

    for col in df.columns:
        genes = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
        all_genes.update(genes)
        print(f"  {col}: {len(genes)} genes")

    all_genes = sorted(all_genes)

    print(f"\n  ? Total UNIQUE cell-cycle genes loaded: {len(all_genes)}")

    return all_genes

# Load marker genes
ALL_CC_GENES = load_marker_genes()



def load_dataset(path, name, expected_count):
    """Load dataset and standardize column names."""
    print(f"\nLoading {name}...")
    df = pd.read_csv(path)

    # Special handling for Buettner: merge with ground truth
    if 'Buettner' in name:
        print(f"  Merging with ground truth labels...")
        ground_truth_path = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Buettner_mESC_goundTruth.csv"
        df_gt = pd.read_csv(ground_truth_path)
        # Merge on Cell_ID
        df = df.merge(df_gt[['Cell_ID', 'Phase']], on='Cell_ID', how='inner')
        print(f"  After merge: {len(df)} cells")

    # Standardize column names to uppercase
    rename_dict = {}
    for col in df.columns:
        if col not in ['gex_barcode', 'Predicted', 'Cell_ID', 'CellID', 'cell', 'phase', 'Phase',
                       'Unnamed: 0', 'dataset', 'Labeled', 'paper_phase', 'Label']:
            rename_dict[col] = col.upper()
    df = df.rename(columns=rename_dict)

    # Standardize phase column
    phase_col = None
    for col in ['Predicted', 'paper_phase', 'Labeled', 'Phase', 'Label']:
        if col in df.columns:
            if col != 'Predicted':
                df = df.rename(columns={col: 'Predicted'})
            phase_col = 'Predicted'
            break

    if phase_col is None:
        print(f"  WARNING: No phase column found!")
        return None, None

    # Standardize phase names
    df['Predicted'] = df['Predicted'].str.replace(r'^G2.*', 'G2M', regex=True)
    df['Predicted'] = df['Predicted'].str.replace(r'^S.*', 'S', regex=True)
    df['Predicted'] = df['Predicted'].str.replace(r'^G1.*', 'G1', regex=True)

    # Filter to valid phases
    valid_phases = ['G1', 'S', 'G2M']
    df = df[df['Predicted'].isin(valid_phases)]

    print(f"  Loaded: {len(df)} cells")

    # Verify cell counts match expected
    actual_counts = df['Predicted'].value_counts().to_dict()
    print(f"  Phase distribution: G1={actual_counts.get('G1', 0)}, S={actual_counts.get('S', 0)}, G2M={actual_counts.get('G2M', 0)}")

    if expected_count:
        expected_total = expected_count['total']
        if len(df) != expected_total:
            print(f"  WARNING: Expected {expected_total} cells but got {len(df)}")
            print(f"  Expected: G1={expected_count['G1']}, S={expected_count['S']}, G2M={expected_count['G2M']}")

    return df, phase_col


def get_marker_expression(df, genes, dataset_name):
    """Get expression statistics for marker genes."""
    # Find available genes (handle capitalization differences)
    available_genes = []
    for g in genes:
        if g in df.columns:
            available_genes.append(g)
        elif g.capitalize() in df.columns:
            available_genes.append(g.capitalize())
        elif g.lower() in df.columns:
            available_genes.append(g.lower())

    if len(available_genes) == 0:
        print(f"  {dataset_name}: No marker genes found!")
        return None, None, None

    print(f"  {dataset_name}: Found {len(available_genes)}/{len(genes)} marker genes")

    # Extract expression data
    gene_expr = df[available_genes].copy()

    # Calculate statistics
    stats = {
        'Dataset': dataset_name,
        'n_genes': len(available_genes),
        'mean_expr': gene_expr.mean().mean(),
        'median_expr': gene_expr.median().median(),
        'std_expr': gene_expr.std().mean(),
        'max_expr': gene_expr.max().max(),
        'pct_zero': (gene_expr == 0).sum().sum() / gene_expr.size * 100,
        'pct_cells_expressing': ((gene_expr > 0).sum(axis=1) / len(available_genes) * 100).mean()
    }

    return stats, gene_expr, available_genes


def plot_heatmap(data_dict, output_name, title):
    """Create publication-quality heatmap of marker gene expression."""
    # Prepare data for heatmap
    datasets = list(data_dict.keys())

    # Get all available genes across datasets
    all_genes = set()
    for genes in data_dict.values():
        all_genes.update(genes)
    all_genes = sorted(list(all_genes))

    # Create matrix
    matrix_data = []
    for dataset in datasets:
        row = []
        for gene in all_genes:
            if gene in data_dict[dataset]:
                row.append(data_dict[dataset][gene])
            else:
                row.append(np.nan)
        matrix_data.append(row)

    matrix = pd.DataFrame(matrix_data, index=datasets, columns=all_genes)

    # Transpose matrix: genes on Y-axis, datasets on X-axis
    matrix = matrix.T

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 20))

    # Create heatmap
    sns.heatmap(matrix, cmap='RdYlBu_r', center=matrix.median().median(),
                annot=False, fmt='.2f', linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Mean Expression', 'shrink': 0.8},
                ax=ax)

    # Styling
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cell Cycle Marker Genes (Gene Symbols)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Tick labels
    ax.tick_params(axis='both', labelsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

    plt.tight_layout()

    # Save
    plt.savefig(f'{OUTPUT_DIR}/{output_name}.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/{output_name}.png', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_name}.pdf and .png")


def plot_grouped_bars(stats_df, metric, ylabel, output_name, title):
    """Create publication-quality grouped bar plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars - use Is_Training column for color
    x = np.arange(len(stats_df))
    if 'Is_Training' in stats_df.columns:
        colors = ['#2E86AB' if is_train else '#F18F01' for is_train in stats_df['Is_Training']]
    else:
        # Fallback to old logic
        colors = ['#2E86AB' if 'Train' in d else '#F18F01' for d in stats_df['Dataset']]

    bars = ax.bar(x, stats_df[metric], width=0.6, color=colors,
                   edgecolor='black', linewidth=0.8)

    # Labels and styling
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['Dataset'], rotation=45, ha='right', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='y', labelsize=12)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='black', label='Training Set'),
        Patch(facecolor='#F18F01', edgecolor='black', label='Benchmark Set')
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right', frameon=True)

    plt.tight_layout()

    # Save
    plt.savefig(f'{OUTPUT_DIR}/{output_name}.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/{output_name}.png', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_name}.pdf and .png")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("CELL CYCLE MARKER GENE EXPRESSION ANALYSIS")
    print("=" * 80)

    # Load dataset configuration from CSV
    print("\nLoading dataset configuration from plot_table1.csv...")
    dataset_config = load_dataset_config()
    print(f"  Found {len(dataset_config)} datasets in configuration")

    # Load all datasets
    datasets = {}
    dataset_training_status = {}

    for dataset_name, config in dataset_config.items():
        df, phase_col = load_dataset(config['path'], dataset_name, config['expected_counts'])
        if df is not None:
            datasets[dataset_name] = df
            dataset_training_status[dataset_name] = config['is_training']

    print(f"\nSuccessfully loaded {len(datasets)} datasets")

    # ==========================================================================
    # 1. OVERALL CELL CYCLE GENE EXPRESSION STATISTICS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. CELL CYCLE GENE EXPRESSION STATISTICS")
    print("=" * 80)

    all_stats = []
    for name, df in datasets.items():
        stats, gene_expr, available_genes = get_marker_expression(df, ALL_CC_GENES, name)
        if stats:
            # Add Is_Training status to stats
            stats['Is_Training'] = dataset_training_status.get(name, False)
            all_stats.append(stats)

    stats_df = pd.DataFrame(all_stats)
    print("\nOverall Statistics:")
    print(stats_df.to_string(index=False))

    # Save stats
    stats_df.to_csv(f'{OUTPUT_DIR}/cc_gene_expression_stats.csv', index=False)

    # ==========================================================================
    # 2. PLOT: MEAN EXPRESSION COMPARISON
    # ==========================================================================
    print("\n" + "=" * 80)
    print("2. GENERATING MEAN EXPRESSION PLOTS")
    print("=" * 80)

    plot_grouped_bars(
        stats_df,
        'mean_expr',
        'Mean Expression Level',
        'mean_cc_gene_expression',
        'Mean Cell Cycle Gene Expression Across Datasets'
    )

    # ==========================================================================
    # 3. PLOT: SPARSITY (% ZEROS)
    # ==========================================================================
    plot_grouped_bars(
        stats_df,
        'pct_zero',
        'Percentage of Zero Values (%)',
        'cc_gene_sparsity',
        'Cell Cycle Gene Expression Sparsity Across Datasets'
    )

    # ==========================================================================
    # 4. PLOT: % CELLS EXPRESSING MARKERS
    # ==========================================================================
    plot_grouped_bars(
        stats_df,
        'pct_cells_expressing',
        '% Genes Expressed per Cell',
        'pct_cells_expressing_cc_genes',
        'Average Percentage of CC Genes Expressed per Cell'
    )

    # ==========================================================================
    # 5. ALL CC MARKER GENE HEATMAP (ALL UNIQUE GENES FROM MARKERGENES.CSV)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("5. GENERATING ALL CC MARKER GENE HEATMAP")
    print("=" * 80)

    # Calculate mean expression for ALL CC markers from markergenes.csv
    key_marker_data = {}
    all_key_markers = ALL_CC_GENES

    for name, df in datasets.items():
        marker_means = {}
        for marker in all_key_markers:
            if marker in df.columns:
                marker_means[marker] = df[marker].mean()
            elif marker.capitalize() in df.columns:
                marker_means[marker] = df[marker.capitalize()].mean()
        key_marker_data[name] = marker_means

    plot_heatmap(
        key_marker_data,
        'all_cc_marker_gene_heatmap',
        'All Cell Cycle Marker Gene Expression (Revelio + Seurat)'
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("Output formats: PDF (vector) + PNG (raster)")
    print("Font: Times New Roman, 12pt tick labels, 14pt axis labels")


if __name__ == "__main__":
    main()
