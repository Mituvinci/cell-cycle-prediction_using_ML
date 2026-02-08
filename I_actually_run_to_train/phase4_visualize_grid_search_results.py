#!/usr/bin/env python3
"""
Visualize Grid Search Results

Creates comparison plots for all 6 configurations:
- Heatmap: Performance across all benchmarks
- Bar plots: Configuration comparison
- Line plots: Metric trends

Author: Halima Akhter
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_comparison_data(csv_path):
    """Load comparison CSV."""
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")
    print(f"  Configurations: {df['configuration'].nunique()}")
    print(f"  Benchmarks: {df['benchmark_name'].nunique()}")
    return df


def plot_heatmap_comparison(df, output_dir):
    """Create heatmap showing performance across configurations and benchmarks."""
    metrics = ['accuracy', 'f1_macro', 'balanced_accuracy', 'mcc']

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Pivot table: rows=configurations, columns=benchmarks
        pivot = df.pivot_table(
            values=metric,
            index='configuration',
            columns='benchmark_name',
            aggfunc='mean'
        )

        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=pivot.min().min(), vmax=pivot.max().max(),
                   linewidths=0.5, ax=ax, cbar_kws={'label': metric.upper()})

        ax.set_title(f'Grid Search: {metric.upper()} Across Configurations')
        ax.set_xlabel('Benchmark Dataset')
        ax.set_ylabel('Configuration (Scaler_Features)')

        plt.tight_layout()
        output_file = f"{output_dir}/heatmap_{metric}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")


def plot_barplot_comparison(df, output_dir):
    """Create bar plots comparing configurations."""
    metrics = ['accuracy', 'f1_macro', 'balanced_accuracy', 'mcc']
    benchmarks = df['benchmark_name'].unique()

    for benchmark in benchmarks:
        benchmark_data = df[df['benchmark_name'] == benchmark]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            if metric not in benchmark_data.columns:
                continue

            ax = axes[idx]

            # Group by configuration and calculate mean
            grouped = benchmark_data.groupby('configuration')[metric].mean().sort_values(ascending=False)

            # Create bar plot
            colors = ['green' if i == 0 else 'steelblue' for i in range(len(grouped))]
            grouped.plot(kind='bar', ax=ax, color=colors, alpha=0.8)

            ax.set_title(f'{metric.upper()}')
            ax.set_xlabel('')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, (config, value) in enumerate(grouped.items()):
                ax.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle(f'Configuration Comparison: {benchmark}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = f"{output_dir}/barplot_{benchmark}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")


def plot_scaler_comparison(df, output_dir):
    """Compare scalers (aggregated across feature selections)."""
    metrics = ['accuracy', 'f1_macro', 'balanced_accuracy', 'mcc']

    if 'scaler' not in df.columns:
        print("WARNING: 'scaler' column not found. Skipping scaler comparison.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if metric not in df.columns:
            continue

        ax = axes[idx]

        # Group by scaler and benchmark, calculate mean
        grouped = df.groupby(['scaler', 'benchmark_name'])[metric].mean().reset_index()

        # Pivot for plotting
        pivot = grouped.pivot(index='benchmark_name', columns='scaler', values=metric)

        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric.upper()}')
        ax.set_xlabel('Benchmark')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Scaler')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Scaler Comparison (Aggregated Across Feature Selections)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = f"{output_dir}/scaler_comparison.png"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_feature_comparison(df, output_dir):
    """Compare feature selection modes (aggregated across scalers)."""
    metrics = ['accuracy', 'f1_macro', 'balanced_accuracy', 'mcc']

    if 'feature_selection' not in df.columns:
        print("WARNING: 'feature_selection' column not found. Skipping feature comparison.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if metric not in df.columns:
            continue

        ax = axes[idx]

        # Group by feature_selection and benchmark, calculate mean
        grouped = df.groupby(['feature_selection', 'benchmark_name'])[metric].mean().reset_index()

        # Pivot for plotting
        pivot = grouped.pivot(index='benchmark_name', columns='feature_selection', values=metric)

        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric.upper()}')
        ax.set_xlabel('Benchmark')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Features')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Feature Selection Comparison (Aggregated Across Scalers)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = f"{output_dir}/feature_comparison.png"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def generate_summary_report(df, output_dir):
    """Generate text summary report."""
    report_file = f"{output_dir}/summary_report.txt"

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GRID SEARCH SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total configurations: {df['configuration'].nunique()}\n")
        f.write(f"Benchmarks tested: {', '.join(df['benchmark_name'].unique())}\n")
        f.write(f"Total models evaluated: {len(df)}\n\n")

        metrics = ['accuracy', 'f1_macro', 'balanced_accuracy', 'mcc']

        f.write("="*80 + "\n")
        f.write("BEST CONFIGURATIONS BY METRIC AND BENCHMARK\n")
        f.write("="*80 + "\n\n")

        for benchmark in df['benchmark_name'].unique():
            benchmark_data = df[df['benchmark_name'] == benchmark]

            f.write(f"\n{benchmark}:\n")
            f.write("-" * 40 + "\n")

            for metric in metrics:
                if metric not in benchmark_data.columns:
                    continue

                # Group by configuration and get mean
                grouped = benchmark_data.groupby('configuration')[metric].mean().sort_values(ascending=False)

                f.write(f"\n  {metric.upper()}:\n")
                for i, (config, value) in enumerate(grouped.head(3).items(), 1):
                    f.write(f"    {i}. {config}: {value:.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL BEST CONFIGURATION (Averaged Across All Benchmarks)\n")
        f.write("="*80 + "\n\n")

        for metric in metrics:
            if metric not in df.columns:
                continue

            # Average across all benchmarks
            overall = df.groupby('configuration')[metric].mean().sort_values(ascending=False)

            f.write(f"\n{metric.upper()}:\n")
            for i, (config, value) in enumerate(overall.head(3).items(), 1):
                f.write(f"  {i}. {config}: {value:.4f}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Saved: {report_file}")


def main():
    """Main visualization pipeline."""
    print("="*80)
    print("VISUALIZING GRID SEARCH RESULTS")
    print("="*80)

    input_csv = "../results/grid_search/comparison_all_configs.csv"
    output_dir = "../results/grid_search/visualizations"

    # Check if input exists
    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        print("Run evaluate_grid_search.sh first!")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_comparison_data(input_csv)

    # Generate visualizations
    print("\nGenerating visualizations...")

    print("\n1. Heatmaps (metric across configs and benchmarks)...")
    plot_heatmap_comparison(df, output_dir)

    print("\n2. Bar plots (config comparison per benchmark)...")
    plot_barplot_comparison(df, output_dir)

    print("\n3. Scaler comparison (aggregated)...")
    plot_scaler_comparison(df, output_dir)

    print("\n4. Feature selection comparison (aggregated)...")
    plot_feature_comparison(df, output_dir)

    print("\n5. Summary report...")
    generate_summary_report(df, output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved in: {output_dir}/")
    print("\nGenerated files:")
    print("  - heatmap_*.png (performance heatmaps)")
    print("  - barplot_*.png (configuration comparisons)")
    print("  - scaler_comparison.png")
    print("  - feature_comparison.png")
    print("  - summary_report.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
