#!/usr/bin/env python3
"""
Compare ONE training dataset vs THREE benchmarks.
Usage: python phase1_distribution_diagnostic.py <training_dataset>
Example: python phase1_distribution_diagnostic.py reh
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Paths
BASE_DIR = Path("/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq")
CELL_CYCLE_DIR = BASE_DIR / "cell_cycle_prediction"
BENCHMARK_DIR = BASE_DIR / "data" / "benchmarks_preprocessed"

# Training dataset configurations
TRAINING_DATASETS = {
    'reh': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv",
        'gene_list': CELL_CYCLE_DIR / "gene_lists/reh_3benchmark_sup.txt",
        'platform': '10x Chromium'
    },
    'sup': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv",
        'gene_list': CELL_CYCLE_DIR / "gene_lists/reh_3benchmark_sup.txt",
        'platform': '10x Chromium'
    },
    'pbmc': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv",
        'gene_list': CELL_CYCLE_DIR / "models/human_pbmc/adaboost/adaboost_NFT_pbmc_fld_1_features.txt",
        'platform': '10x Chromium'
    },
    'mouse_brain': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data.csv",
        'gene_list': CELL_CYCLE_DIR / "models/mouse_brain/adaboost/adaboost_NFT_mouse_brain_fld_1_features.txt",
        'platform': '10x Chromium'
    },
    'nestorova': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_nestorova/nestorova_training_data.csv",
        'gene_list': CELL_CYCLE_DIR / "models/nestorova/adaboost/adaboost_NFT_nestorova_fld_1_features.txt",
        'platform': 'Smart-seq2'
    },
    'hpsc': {
        'path': CELL_CYCLE_DIR / "1_consensus_labeling/assign/final_training_data_gse75748/gse75748_training_data.csv",
        'gene_list': CELL_CYCLE_DIR / "models/human_hpsc_5770_standard/adaboost/adaboost_NFT_hpsc_fld_1_features.txt",
        'platform': 'Smart-seq2'
    }
}

# Benchmark datasets (fixed)
BENCHMARKS = {
    'GSE146773': {
        'path': BENCHMARK_DIR / "GSE146773/GSE146773_expression.csv",
        'platform': 'Smart-seq2'
    },
    'GSE64016': {
        'path': BENCHMARK_DIR / "GSE64016/GSE64016_expression.csv",
        'platform': 'Fluidigm C1'
    },
    'Buettner_mESC': {
        'path': BENCHMARK_DIR / "Buettner_mESC/Buettner_mESC_expression.csv",
        'platform': 'E-MTAB-2805'
    }
}


def load_gene_list(gene_list_file):
    """Load gene list."""
    with open(gene_list_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_dataset(path, gene_list):
    """Load dataset and filter to gene list."""
    df = pd.read_csv(path, index_col=0)

    # Remove metadata columns
    metadata_cols = ['Predicted', 'gex_barcode', 'CellID', 'paper_phase', 'Labeled', 'phase']
    expr_df = df.drop(columns=[col for col in metadata_cols if col in df.columns], errors='ignore')
    expr_df = expr_df.select_dtypes(include=[np.number])

    # Filter to gene list (case-insensitive)
    gene_map = {gene.upper(): gene for gene in expr_df.columns}
    matched_genes = [gene_map[g.upper()] for g in gene_list if g.upper() in gene_map]
    expr_df = expr_df[matched_genes]

    return expr_df


def calculate_sparsity(df):
    """Calculate percentage of zeros."""
    return (df == 0).sum().sum() / df.size * 100


def main():
    if len(sys.argv) != 2:
        print("Usage: python phase1_distribution_diagnostic.py <training_dataset>")
        print("Available: reh, sup, pbmc, mouse_brain, nestorova, hpsc")
        sys.exit(1)

    training_name = sys.argv[1].lower()

    if training_name not in TRAINING_DATASETS:
        print(f"ERROR: Unknown training dataset '{training_name}'")
        print(f"Available: {', '.join(TRAINING_DATASETS.keys())}")
        sys.exit(1)

    # Create output directory
    output_dir = CELL_CYCLE_DIR / "distribution_analysis" / f"result_{training_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"Distribution Analysis: {training_name.upper()} vs Benchmarks")
    print("="*70)

    # Load training dataset
    train_config = TRAINING_DATASETS[training_name]
    print(f"\nLoading training data: {training_name}")
    gene_list = load_gene_list(train_config['gene_list'])
    print(f"  Gene list: {len(gene_list)} genes")

    train_df = load_dataset(train_config['path'], gene_list)
    print(f"  Loaded: {train_df.shape[0]} cells x {train_df.shape[1]} genes")

    # Calculate training stats
    train_sparsity = calculate_sparsity(train_df)
    train_mean_genes = (train_df != 0).sum(axis=1).mean()

    # Load benchmarks
    benchmark_data = {}
    for bench_name, bench_config in BENCHMARKS.items():
        print(f"\nLoading benchmark: {bench_name}")
        bench_df = load_dataset(bench_config['path'], gene_list)
        print(f"  Loaded: {bench_df.shape[0]} cells x {bench_df.shape[1]} genes")
        benchmark_data[bench_name] = {
            'df': bench_df,
            'sparsity': calculate_sparsity(bench_df),
            'mean_genes': (bench_df != 0).sum(axis=1).mean(),
            'platform': bench_config['platform']
        }

    # Generate summary table
    summary_rows = []
    summary_rows.append({
        'Dataset': training_name.upper(),
        'Type': 'Training',
        'Platform': train_config['platform'],
        'Cells': train_df.shape[0],
        'Genes': train_df.shape[1],
        'Sparsity_%': round(train_sparsity, 2),
        'Mean_Genes_Per_Cell': round(train_mean_genes, 1)
    })

    for bench_name, bench_info in benchmark_data.items():
        summary_rows.append({
            'Dataset': bench_name,
            'Type': 'Benchmark',
            'Platform': bench_info['platform'],
            'Cells': bench_info['df'].shape[0],
            'Genes': bench_info['df'].shape[1],
            'Sparsity_%': round(bench_info['sparsity'], 2),
            'Mean_Genes_Per_Cell': round(bench_info['mean_genes'], 1)
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save summary
    summary_file = output_dir / 'distribution_statistics.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n\nSaved: {summary_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Generate plot: Sparsity comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    datasets = [training_name.upper()] + list(BENCHMARKS.keys())
    sparsities = [train_sparsity] + [benchmark_data[b]['sparsity'] for b in BENCHMARKS]
    colors = ['#1f77b4'] + ['#2ca02c', '#d62728', '#9467bd']

    bars = ax.bar(datasets, sparsities, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Sparsity (% Zeros)', fontweight='bold')
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_title(f'{training_name.upper()} Training Data vs Benchmarks: Sparsity Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_file = output_dir / 'sparsity_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == '__main__':
    main()
