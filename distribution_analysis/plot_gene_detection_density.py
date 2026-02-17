#!/usr/bin/env python3
"""
Plot Expression Value Distribution (Probability Density Curves)
===============================================================
For each dataset, flatten all expression values (non-zero),
then plot all datasets as overlaid KDE curves on a single figure.

X-axis: Expression values (same as original histograms)
Y-axis: Probability density (not frequency)

Usage:
    python distribution_analysis/plot_gene_detection_density.py

Output:
    distribution_analysis/results/expression_density_all_datasets.png
    distribution_analysis/results/expression_density_all_datasets.pdf

Author: Distribution Analysis for Cross-Platform Cell Cycle Prediction
Date: 2026-02-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Paths
BASE_DIR = Path("/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq")
CELL_CYCLE_DIR = BASE_DIR / "cell_cycle_prediction"
BENCHMARK_DIR = BASE_DIR / "data" / "benchmarks_preprocessed"
OUTPUT_DIR = CELL_CYCLE_DIR / "distribution_analysis" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gene list: intersection of REH + 3 benchmarks + SUP (12,012 genes)
GENE_LIST_FILE = CELL_CYCLE_DIR / "gene_lists" / "reh_3benchmark_sup.txt"

# Datasets
DATASETS = {
    'REH': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv",
        'type': 'Training',
        'platform': '10x Chromium',
        'color': '#1f77b4',
        'linestyle': '-'
    },
    'SUP-B15': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv",
        'type': 'Training',
        'platform': '10x Chromium',
        'color': '#ff7f0e',
        'linestyle': '-'
    },
    'GSE146773': {
        'path': BENCHMARK_DIR / "GSE146773/GSE146773_expression.csv",
        'type': 'Benchmark',
        'platform': 'Smart-seq2',
        'color': '#2ca02c',
        'linestyle': '--'
    },
    'GSE64016': {
        'path': BENCHMARK_DIR / "GSE64016/GSE64016_expression.csv",
        'type': 'Benchmark',
        'platform': 'Fluidigm C1',
        'color': '#d62728',
        'linestyle': '--'
    },
    'Buettner_mESC': {
        'path': BENCHMARK_DIR / "Buettner_mESC/Buettner_mESC_expression.csv",
        'type': 'Benchmark',
        'platform': 'E-MTAB-2805',
        'color': '#9467bd',
        'linestyle': '--'
    }
}


def load_gene_list(gene_list_file):
    """Load gene list (intersection of training + benchmarks)."""
    with open(gene_list_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_expression_matrix(path, gene_list):
    """Load dataset, filter to gene list, return only numeric expression columns."""
    print(f"  Loading: {path.name}")
    df = pd.read_csv(path, index_col=0)

    # Remove metadata columns
    metadata_cols = ['Predicted', 'gex_barcode', 'CellID', 'paper_phase', 'Labeled', 'phase']
    expr_df = df.drop(columns=[col for col in metadata_cols if col in df.columns], errors='ignore')
    expr_df = expr_df.select_dtypes(include=[np.number])

    # Filter to gene list (case-insensitive matching)
    gene_map = {gene.upper(): gene for gene in expr_df.columns}
    matched_genes = [gene_map[g.upper()] for g in gene_list if g.upper() in gene_map]
    expr_df = expr_df[matched_genes]

    print(f"    {expr_df.shape[0]} cells x {expr_df.shape[1]} genes (filtered to gene list)")
    return expr_df


def main():
    print("=" * 70)
    print("Expression Value Distribution (Probability Density)")
    print("=" * 70)

    # Load gene list (same intersection used in original analysis)
    gene_list = load_gene_list(GENE_LIST_FILE)
    print(f"Gene list: {len(gene_list)} genes from {GENE_LIST_FILE.name}")

    # Load all datasets
    expression_data = {}

    for name, info in DATASETS.items():
        print(f"\n{name} ({info['type']}, {info['platform']}):")

        if not info['path'].exists():
            print(f"    WARNING: File not found, skipping: {info['path']}")
            continue

        expr_df = load_expression_matrix(info['path'], gene_list)

        # Flatten all expression values and keep only non-zero
        values = expr_df.values.flatten()
        values_nonzero = values[values > 0]
        expression_data[name] = values_nonzero
        print(f"    Non-zero values: {len(values_nonzero):,} / {len(values):,} total")

    # =========================================================================
    # PLOT: Overlaid KDE curves - expression values on X, probability on Y
    # =========================================================================
    print("\nGenerating plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    for name, values in expression_data.items():
        info = DATASETS[name]
        label = f"{name} ({info['platform']}, n={len(values):,})"

        # Plot KDE curve
        sns.kdeplot(
            data=values,
            ax=ax,
            label=label,
            color=info['color'],
            linestyle=info['linestyle'],
            linewidth=2.0,
            bw_adjust=0.8
        )

    ax.set_xlabel('Expression Value', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.set_title('Cross-Dataset Expression Value Distribution (Non-Zero)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Legend at upper right
    ax.legend(
        loc='upper right',
        framealpha=0.9,
        edgecolor='black',
        title='Solid = Training, Dashed = Benchmark',
        title_fontsize=9
    )

    plt.tight_layout()

    # Save PNG and PDF
    png_path = OUTPUT_DIR / "expression_density_all_datasets.png"
    pdf_path = OUTPUT_DIR / "expression_density_all_datasets.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
