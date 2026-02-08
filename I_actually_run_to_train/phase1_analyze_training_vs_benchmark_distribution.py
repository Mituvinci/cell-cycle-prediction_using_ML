#!/usr/bin/env python3
"""
Distribution Analysis: Training vs Benchmark Data

Analyzes distribution differences between training datasets and benchmark datasets
to understand why models may not generalize well.

Analyses:
1. Gene expression distributions (mean, variance, std)
2. Cell cycle marker gene expression
3. PCA visualization (training vs benchmark)
4. Statistical tests (KS test, Welch's t-test)
5. Coefficient of Variation (CV) analysis

Author: Halima Akhter
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class DistributionAnalyzer:
    """Analyze distribution differences between training and benchmark data."""

    def __init__(self, output_dir="../distribution_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Cell cycle marker genes (UPPERCASE for cross-species compatibility)
        self.cc_markers = {
            'G1': ['CCND1', 'CCND2', 'CCND3', 'CDK4', 'CDK6', 'RB1'],
            'S': ['PCNA', 'MCM2', 'MCM5', 'MCM6', 'RRM2', 'TYMS', 'CCNE1', 'CCNE2'],
            'G2M': ['CCNB1', 'CCNB2', 'CDC20', 'CDK1', 'MKI67', 'TOP2A', 'AURKA', 'AURKB']
        }
        self.all_cc_markers = [g for genes in self.cc_markers.values() for g in genes]

    def load_dataset(self, csv_path, dataset_name):
        """Load dataset and return expression matrix and labels."""
        print(f"Loading {dataset_name}: {csv_path}")
        df = pd.read_csv(csv_path)

        if 'phase' in df.columns or 'paper_phase' in df.columns:
            labels = df.get('phase', df.get('paper_phase'))
            expression = df.drop(columns=[c for c in ['phase', 'paper_phase', 'cell', 'Unnamed: 0'] if c in df.columns])
        elif 'cell_id' in df.columns:
            expression = df.drop(columns=['cell_id'])
            labels = None
        else:
            expression = df
            labels = None

        # Convert all gene names to UPPERCASE for consistency
        expression.columns = [str(col).upper() for col in expression.columns]

        print(f"  Shape: {expression.shape}")
        if labels is not None:
            print(f"  Classes: {labels.value_counts().to_dict()}")

        return expression, labels

    def get_common_genes(self, df1, df2):
        """Get common genes between two datasets."""
        genes1 = set(df1.columns)
        genes2 = set(df2.columns)
        common = genes1.intersection(genes2)
        print(f"  Common genes: {len(common)} (Dataset1: {len(genes1)}, Dataset2: {len(genes2)})")
        return sorted(list(common))

    def calculate_statistics(self, expression_df):
        """Calculate mean, variance, std, CV for each gene."""
        expression_df = expression_df.select_dtypes(include=[np.number])
        stats_df = pd.DataFrame({
            'mean': expression_df.mean(axis=0),
            'variance': expression_df.var(axis=0),
            'std': expression_df.std(axis=0),
            'cv': expression_df.std(axis=0) / (expression_df.mean(axis=0) + 1e-10),
            'median': expression_df.median(axis=0),
            'q25': expression_df.quantile(0.25, axis=0),
            'q75': expression_df.quantile(0.75, axis=0)
        })
        return stats_df

    def compare_distributions(self, train_expr, benchmark_expr, train_name, benchmark_name):
        """Compare distributions between training and benchmark data."""
        print(f"\n{'='*80}")
        print(f"COMPARING: {train_name} vs {benchmark_name}")
        print(f"{'='*80}")

        # Get common genes
        common_genes = self.get_common_genes(train_expr, benchmark_expr)
        train_common = train_expr[common_genes]
        benchmark_common = benchmark_expr[common_genes]

        # Calculate statistics
        print("\nCalculating statistics...")
        train_stats = self.calculate_statistics(train_common)
        benchmark_stats = self.calculate_statistics(benchmark_common)

        # Statistical tests
        print("\nPerforming statistical tests...")
        self.statistical_tests(train_common, benchmark_common, train_stats, benchmark_stats,
                               train_name, benchmark_name)

        # Visualizations
        print("\nGenerating visualizations...")
        self.plot_distribution_comparison(train_stats, benchmark_stats, train_name, benchmark_name)
        self.plot_variance_comparison(train_stats, benchmark_stats, train_name, benchmark_name)
        self.plot_cv_comparison(train_stats, benchmark_stats, train_name, benchmark_name)

        # Cell cycle markers
        if any(g in common_genes for g in self.all_cc_markers):
            print("\nAnalyzing cell cycle marker genes...")
            self.analyze_cc_markers(train_common, benchmark_common, train_name, benchmark_name)

        # PCA visualization
        print("\nPerforming PCA visualization...")
        self.pca_visualization(train_common, benchmark_common, train_name, benchmark_name)

        print(f"\nDONE: {train_name} vs {benchmark_name}")
        print(f"{'='*80}\n")

    def statistical_tests(self, train_expr, benchmark_expr, train_stats, benchmark_stats,
                          train_name, benchmark_name):
        """Perform statistical tests to compare distributions."""

        # Kolmogorov-Smirnov test for each gene
        ks_results = []
        for gene in train_expr.columns[:100]:  # Test first 100 genes (representative sample)
            statistic, pvalue = stats.ks_2samp(train_expr[gene], benchmark_expr[gene])
            ks_results.append({'gene': gene, 'ks_statistic': statistic, 'pvalue': pvalue})

        if len(ks_results) == 0:
            print("  WARNING: No genes tested (columns might be non-numeric)")
            return

        ks_df = pd.DataFrame(ks_results)
        significant_genes = (ks_df['pvalue'] < 0.05).sum()

        print(f"  Kolmogorov-Smirnov test (first 100 genes):")
        print(f"    Significantly different distributions: {significant_genes}/100 genes (p<0.05)")
        print(f"    Mean KS statistic: {ks_df['ks_statistic'].mean():.4f}")

        # Overall statistics comparison
        print(f"\n  Overall Statistics:")
        print(f"    Mean expression:")
        print(f"      Training:  {train_stats['mean'].mean():.4f} +/- {train_stats['mean'].std():.4f}")
        print(f"      Benchmark: {benchmark_stats['mean'].mean():.4f} +/- {benchmark_stats['mean'].std():.4f}")
        print(f"    Variance:")
        print(f"      Training:  {train_stats['variance'].mean():.4f} +/- {train_stats['variance'].std():.4f}")
        print(f"      Benchmark: {benchmark_stats['variance'].mean():.4f} +/- {benchmark_stats['variance'].std():.4f}")
        print(f"    Coefficient of Variation:")
        print(f"      Training:  {train_stats['cv'].mean():.4f} +/- {train_stats['cv'].std():.4f}")
        print(f"      Benchmark: {benchmark_stats['cv'].mean():.4f} +/- {benchmark_stats['cv'].std():.4f}")

        # Save KS test results
        output_file = f"{self.output_dir}/ks_test_{train_name}_vs_{benchmark_name}.csv"
        ks_df.to_csv(output_file, index=False)
        print(f"\n  Saved KS test results to: {output_file}")

    def plot_distribution_comparison(self, train_stats, benchmark_stats, train_name, benchmark_name):
        """Plot distribution comparison of mean and variance."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Mean comparison
        axes[0].scatter(train_stats['mean'], benchmark_stats['mean'], alpha=0.3, s=10)
        axes[0].plot([0, train_stats['mean'].max()], [0, train_stats['mean'].max()],
                     'r--', label='y=x')
        axes[0].set_xlabel(f'{train_name} Mean Expression')
        axes[0].set_ylabel(f'{benchmark_name} Mean Expression')
        axes[0].set_title('Mean Expression Comparison')
        axes[0].legend()

        # Variance comparison
        axes[1].scatter(train_stats['variance'], benchmark_stats['variance'], alpha=0.3, s=10)
        axes[1].plot([0, train_stats['variance'].max()], [0, train_stats['variance'].max()],
                     'r--', label='y=x')
        axes[1].set_xlabel(f'{train_name} Variance')
        axes[1].set_ylabel(f'{benchmark_name} Variance')
        axes[1].set_title('Variance Comparison')
        axes[1].legend()

        plt.tight_layout()
        output_file = f"{self.output_dir}/distribution_comparison_{train_name}_vs_{benchmark_name}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def plot_variance_comparison(self, train_stats, benchmark_stats, train_name, benchmark_name):
        """Plot variance distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        axes[0].hist(train_stats['variance'], bins=50, alpha=0.5, label=train_name, density=True)
        axes[0].hist(benchmark_stats['variance'], bins=50, alpha=0.5, label=benchmark_name, density=True)
        axes[0].set_xlabel('Variance')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Variance Distribution')
        axes[0].legend()
        axes[0].set_xlim(0, np.percentile(train_stats['variance'], 99))

        # Boxplot
        data_to_plot = [train_stats['variance'], benchmark_stats['variance']]
        axes[1].boxplot(data_to_plot, labels=[train_name, benchmark_name])
        axes[1].set_ylabel('Variance')
        axes[1].set_title('Variance Comparison (Boxplot)')
        axes[1].set_ylim(0, np.percentile(train_stats['variance'], 99))

        plt.tight_layout()
        output_file = f"{self.output_dir}/variance_comparison_{train_name}_vs_{benchmark_name}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def plot_cv_comparison(self, train_stats, benchmark_stats, train_name, benchmark_name):
        """Plot Coefficient of Variation comparison."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.scatter(train_stats['cv'], benchmark_stats['cv'], alpha=0.3, s=10)
        max_cv = max(train_stats['cv'].quantile(0.99), benchmark_stats['cv'].quantile(0.99))
        ax.plot([0, max_cv], [0, max_cv], 'r--', label='y=x')
        ax.set_xlabel(f'{train_name} CV')
        ax.set_ylabel(f'{benchmark_name} CV')
        ax.set_title('Coefficient of Variation Comparison')
        ax.legend()
        ax.set_xlim(0, max_cv)
        ax.set_ylim(0, max_cv)

        plt.tight_layout()
        output_file = f"{self.output_dir}/cv_comparison_{train_name}_vs_{benchmark_name}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def analyze_cc_markers(self, train_expr, benchmark_expr, train_name, benchmark_name):
        """Analyze cell cycle marker gene expression."""
        available_markers = [g for g in self.all_cc_markers if g in train_expr.columns]

        if len(available_markers) == 0:
            print("  No cell cycle markers found in common genes.")
            return

        print(f"  Found {len(available_markers)} cell cycle markers")

        # Calculate mean expression for markers
        train_marker_expr = train_expr[available_markers].mean(axis=0)
        benchmark_marker_expr = benchmark_expr[available_markers].mean(axis=0)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        x = np.arange(len(available_markers))
        width = 0.35

        ax.bar(x - width/2, train_marker_expr.values, width, label=train_name, alpha=0.8)
        ax.bar(x + width/2, benchmark_marker_expr.values, width, label=benchmark_name, alpha=0.8)

        ax.set_xlabel('Cell Cycle Marker Genes')
        ax.set_ylabel('Mean Expression')
        ax.set_title('Cell Cycle Marker Gene Expression')
        ax.set_xticks(x)
        ax.set_xticklabels(available_markers, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        output_file = f"{self.output_dir}/cc_markers_{train_name}_vs_{benchmark_name}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def pca_visualization(self, train_expr, benchmark_expr, train_name, benchmark_name):
        """Perform PCA and visualize training vs benchmark data."""
        # Combine data
        train_labeled = train_expr.copy()
        train_labeled['dataset'] = train_name

        benchmark_labeled = benchmark_expr.copy()
        benchmark_labeled['dataset'] = benchmark_name

        combined = pd.concat([train_labeled, benchmark_labeled], axis=0)
        labels = combined['dataset']
        expression = combined.drop(columns=['dataset'])
        expression = expression.select_dtypes(include=[np.number])

        # Standardize
        scaler = StandardScaler()
        expression_scaled = scaler.fit_transform(expression)

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(expression_scaled)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        for dataset_name in labels.unique():
            mask = labels == dataset_name
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                      label=dataset_name, alpha=0.5, s=20)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'PCA: {train_name} vs {benchmark_name}')
        ax.legend()

        plt.tight_layout()
        output_file = f"{self.output_dir}/pca_{train_name}_vs_{benchmark_name}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

        print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("DISTRIBUTION ANALYSIS: TRAINING vs BENCHMARK DATA")
    print("="*80)

    analyzer = DistributionAnalyzer(output_dir="../distribution_analysis")

    # Define datasets
    training_datasets = {
        'concate_4ds_all': '../1_consensus_labeling/assign/final_training_data_concate_4_ds_pbmchealthy_human_mouse_brain_GSE75748_Nestorova/concatenated_training_data.csv',
        'concate_2ds_human': '../1_consensus_labeling/assign/final_training_data_concate_2_ds_only_pbmchealthy_human_GSE75748/concatenated_training_data.csv',
        'concate_mouse': '../1_consensus_labeling/assign/final_training_data_concate_mouse_brain_Nestorova/concatenated_training_data.csv'
    }

    benchmark_datasets = {
        'GSE146773': '/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/GSE146773_seurat_normalized_gene_expression.csv',
        'GSE64016': '/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/GSE64016_seurat_normalized_gene_expression.csv',
        'Buettner_mESC': '/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/Buettner_mESC_SeuratNormalized_ML_ready.csv'
    }

    # Load all datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)

    train_data = {}
    for name, path in training_datasets.items():
        expr, labels = analyzer.load_dataset(path, name)
        train_data[name] = expr

    benchmark_data = {}
    for name, path in benchmark_datasets.items():
        expr, labels = analyzer.load_dataset(path, name)
        benchmark_data[name] = expr

    # Compare each training dataset with each benchmark
    print("\n" + "="*80)
    print("STARTING COMPARISONS")
    print("="*80)

    for train_name, train_expr in train_data.items():
        for benchmark_name, benchmark_expr in benchmark_data.items():
            # Skip cross-species comparisons that don't make sense
            if 'human' in train_name and benchmark_name == 'Buettner_mESC':
                print(f"\nSkipping: {train_name} vs {benchmark_name} (cross-species)")
                continue
            if 'mouse' in train_name and benchmark_name in ['GSE146773', 'GSE64016']:
                print(f"\nSkipping: {train_name} vs {benchmark_name} (cross-species)")
                continue

            analyzer.compare_distributions(train_expr, benchmark_expr, train_name, benchmark_name)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved in: {analyzer.output_dir}/")
    print("\nSummary of outputs:")
    print("  - Distribution comparison plots (mean, variance)")
    print("  - Variance distribution histograms")
    print("  - Coefficient of Variation comparisons")
    print("  - Cell cycle marker gene expression")
    print("  - PCA visualizations")
    print("  - KS test results (CSV)")


if __name__ == "__main__":
    main()
