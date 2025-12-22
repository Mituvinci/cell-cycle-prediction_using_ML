#!/usr/bin/env python3
"""
Generate publication-quality benchmark comparison plots with single Y-axis
2x2 grid layout: GSE146773, GSE64016, Buettner_mESC, SUP-B15
Output: Individual panels + Combined 2x2 grid with shared legend
Format: PDF (editable in Adobe Illustrator), PNG, EPS

Note: Kappa and MCC are multiplied by 100 for visualization (converted from -1 to 1 scale to %)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import argparse
import sys

# ============================================================================
# PUBLICATION-QUALITY SETTINGS
# ============================================================================

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# High-quality PDF settings
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable in Illustrator)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600  # High resolution for publication
plt.rcParams['savefig.format'] = 'pdf'

# Professional styling
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Generate benchmark comparison plots')
parser.add_argument('--input-dir', '-i', required=False,
                   default="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/5_visualization",
                   help='Directory containing benchmark CSV files')
parser.add_argument('--training-dataset', '-t', required=False, default='reh',
                   help='Training dataset name (reh, hpsc, pbmc, mouse_brain)')

args = parser.parse_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - use input directory from arguments
BASE_DIR = args.input_dir
training_dataset = args.training_dataset

# CSV file paths - use dynamic training dataset name
csv_files = {
    'GSE146773': os.path.join(BASE_DIR, f'gse146773_results_{training_dataset}.csv'),
    'GSE64016': os.path.join(BASE_DIR, f'gse64016_results_{training_dataset}.csv'),
    'Buettner_mESC': os.path.join(BASE_DIR, f'buettner_mesc_results_{training_dataset}.csv'),
    'SUP-B15': os.path.join(BASE_DIR, f'sup_results_{training_dataset}.csv')
}

# Consistent model order for ALL visualizations
# DNN4 removed, Top 3 DF/SF moved after Embedding3TML (no AUC for DF)
MODEL_ORDER = [
    'DNN3',
    'DNN5',
    'CNN',
    'Hybrid CNN',
    'FE model',
    'Adaboost',
    'LGBM',
    'Random Forest',
    'Embedding3TML',
    'Top 3 DF',
    'Top 3 SF'
]

def sort_models_by_order(df):
    """Sort DataFrame by predefined model order."""
    df['sort_key'] = df['Model'].apply(lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999)
    df = df.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
    return df

# Colorblind-friendly color scheme
COLORS = {
    'Accuracy': '#2E86AB',      # Blue (plotted but not in legend)
    'AUC': '#F18F01',           # Orange
    'F1': '#06A77D',            # Green
    'MCC': '#D90429',           # Red (right axis)
    'Kappa': '#6A4C93',         # Purple
    'Balanced_Accuracy': '#8B4513'  # Brown
}

# Line markers for each metric
MARKERS = {
    'Accuracy': 'o',
    'AUC': 's',
    'F1': '^',
    'MCC': 'D',
    'Kappa': 'v',
    'Balanced_Accuracy': 'p'
}

# Metrics to plot (all on single Y-axis in %)
ALL_METRICS = ['Accuracy', 'AUC', 'F1', 'Kappa', 'Balanced_Accuracy', 'MCC']

# Metrics to show in legend (excluding Accuracy)
LEGEND_METRICS = ['AUC', 'F1', 'Kappa', 'Balanced_Accuracy', 'MCC']

# Metrics that need to be multiplied by 100 (from -1 to 1 scale)
SCALE_TO_PERCENT = ['Kappa', 'MCC']

# ============================================================================
# FUNCTION: CREATE SINGLE PANEL
# ============================================================================

def create_single_panel(df, benchmark_name, ax=None, show_legend=True):
    """
    Create a single panel with single Y-axis line plot

    Parameters:
    -----------
    df : DataFrame
        Must have columns: Model, Accuracy, AUC, F1, MCC, Kappa, Balanced_Accuracy
    benchmark_name : str
        Name of benchmark (e.g., 'GSE146773')
    ax : matplotlib axis
        If provided, plot on this axis. Otherwise create new figure.
    show_legend : bool
        Whether to show legend on this panel (default: True)

    Returns:
    --------
    fig, ax1, lines, labels : matplotlib figure, axis, and legend data
    """

    # Create figure if not provided
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        standalone = True
    else:
        ax1 = ax
        fig = ax1.get_figure()
        standalone = False

    # X-axis positions
    x = np.arange(len(df))

    # Storage for legend data
    all_lines = []
    all_labels = []

    # Plot all metrics on single Y-axis
    for metric in ALL_METRICS:
        # Scale Kappa and MCC to percentage (multiply by 100)
        if metric in SCALE_TO_PERCENT:
            plot_values = df[metric] * 100
        else:
            plot_values = df[metric]

        line, = ax1.plot(x, plot_values,
                        color=COLORS[metric],
                        marker=MARKERS[metric],
                        markersize=8,
                        linewidth=2.5,
                        alpha=0.9)

        # Only add to legend if it's in LEGEND_METRICS
        if metric in LEGEND_METRICS:
            all_lines.append(line)
            all_labels.append(f"{metric.replace('_', ' ')} (%)")

    # ========================================================================
    # AXIS FORMATTING
    # ========================================================================

    # X-axis: Model names
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=12)

    # Y-axis: Percentages
    ax1.set_ylabel('Performance Metrics (%)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=12, labelcolor='black')

    # Panel label (A, B, C, or D)
    panel_labels = {'GSE146773': 'A', 'GSE64016': 'B', 'Buettner_mESC': 'C', 'SUP-B15': 'D'}
    panel_letter = panel_labels.get(benchmark_name, '')

    # Add panel label and benchmark name as title
    title_text = f"{panel_letter}. {benchmark_name}"
    ax1.text(-0.1, 1.05, title_text, transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='bottom')

    # ========================================================================
    # STYLING
    # ========================================================================

    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Set spine colors
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')

    # Add gridlines
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
    ax1.set_axisbelow(True)

    # Set Y-axis limits with padding
    # Calculate min/max across all plotted values
    all_values = []
    for metric in ALL_METRICS:
        if metric in SCALE_TO_PERCENT:
            all_values.extend((df[metric] * 100).tolist())
        else:
            all_values.extend(df[metric].tolist())

    y_min = min(all_values)
    y_max = max(all_values)
    padding = (y_max - y_min) * 0.1
    ax1.set_ylim(y_min - padding, y_max + padding)

    # ========================================================================
    # LEGEND (only for standalone plots)
    # ========================================================================

    if show_legend and standalone:
        legend = ax1.legend(all_lines, all_labels,
                           loc='best',
                           fontsize=12,
                           frameon=True,
                           fancybox=False,
                           edgecolor='black',
                           framealpha=0.95)
        legend.get_frame().set_linewidth(1.5)

    if standalone:
        plt.tight_layout()

    return fig, ax1, all_lines, all_labels


# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading benchmark data...")
data = {}
for benchmark, filepath in csv_files.items():
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Ensure required columns exist
        required_cols = ['Model', 'Accuracy', 'AUC', 'F1', 'MCC', 'Kappa', 'Balanced_Accuracy']
        if not all(col in df.columns for col in required_cols):
            print(f"WARNING: {filepath} missing required columns!")
            print(f"Required: {required_cols}")
            print(f"Found: {list(df.columns)}")
        else:
            # Sort by predefined model order
            df = sort_models_by_order(df)
            data[benchmark] = df
            print(f"  Loaded {benchmark}: {len(df)} models")
    else:
        print(f"WARNING: {filepath} not found!")
        print(f"  Creating dummy data for demonstration...")
        # Create dummy data for demonstration
        models = ['DNN3', 'DNN5', 'FE model', 'CNN', 'Hybrid CNN',
                 'Top 3 DF', 'Top 3 SF', 'LGBM', 'Adaboost', 'Random Forest', 'Embedding3TML']
        dummy_df = pd.DataFrame({
            'Model': models,
            'Accuracy': np.random.uniform(70, 76, len(models)),
            'AUC': np.random.uniform(82, 89, len(models)),
            'F1': np.random.uniform(70, 76, len(models)),
            'MCC': np.random.uniform(0.57, 0.64, len(models)),
            'Kappa': np.random.uniform(57, 64, len(models)),
            'Balanced_Accuracy': np.random.uniform(70, 76, len(models))
        })
        data[benchmark] = dummy_df
        print(f"  Created dummy data for {benchmark}")

if len(data) == 0:
    print("\nERROR: No data available to plot!")
    print("Please provide CSV files with the following format:")
    print("Model,Accuracy,AUC,F1,MCC,Kappa,Balanced_Accuracy")
    exit(1)

# ============================================================================
# GENERATE INDIVIDUAL PANELS
# ============================================================================

print("\nGenerating individual panels...")

panel_files = {}
panel_labels = {'GSE146773': 'A', 'GSE64016': 'B', 'Buettner_mESC': 'C', 'SUP-B15': 'D'}

for benchmark, df in data.items():
    print(f"\n  Creating Panel for {benchmark}...")

    # Create panel
    fig, ax1, lines, labels = create_single_panel(df, benchmark, show_legend=True)

    # Output filename
    panel_letter = panel_labels[benchmark]
    output_base = os.path.join(BASE_DIR, f'panel_{panel_letter}_{benchmark.lower()}')

    # Save as PDF
    pdf_file = f"{output_base}.pdf"
    plt.savefig(pdf_file, format='pdf', dpi=600, bbox_inches='tight',
                transparent=False, facecolor='white')
    print(f"    PDF saved: {pdf_file}")

    # Save as PNG
    png_file = f"{output_base}.png"
    plt.savefig(png_file, format='png', dpi=600, bbox_inches='tight',
                transparent=False, facecolor='white')
    print(f"    PNG saved: {png_file}")

    # Save as EPS
    eps_file = f"{output_base}.eps"
    plt.savefig(eps_file, format='eps', dpi=600, bbox_inches='tight')
    print(f"    EPS saved: {eps_file}")

    panel_files[benchmark] = {'pdf': pdf_file, 'png': png_file, 'eps': eps_file}

    plt.close()

# ============================================================================
# GENERATE COMBINED 2x2 GRID LAYOUT WITH SHARED LEGEND
# ============================================================================

print("\n\nGenerating combined 2x2 grid layout (A + B + C + D) with shared legend...")

# Create figure with 2x2 grid
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35,
                      left=0.08, right=0.92, top=0.92, bottom=0.15)

# Define panel order: Row 1: GSE146773, GSE64016; Row 2: Buettner_mESC, SUP-B15
benchmark_grid = [
    ['GSE146773', 'GSE64016'],      # Row 1
    ['Buettner_mESC', 'SUP-B15']    # Row 2
]

# Create subplots
legend_lines = None
legend_labels = None

for row in range(2):
    for col in range(2):
        benchmark = benchmark_grid[row][col]
        if benchmark in data:
            print(f"  Adding Panel {panel_labels[benchmark]}: {benchmark}...")
            ax = fig.add_subplot(gs[row, col])
            fig_temp, ax1, lines, labels = create_single_panel(
                data[benchmark], benchmark, ax=ax, show_legend=False
            )
            # Store legend data from first panel
            if legend_lines is None:
                legend_lines = lines
                legend_labels = labels
        else:
            print(f"  WARNING: {benchmark} data not available, skipping...")
            ax = fig.add_subplot(gs[row, col])
            ax.text(0.5, 0.5, f'{benchmark}\nData Not Available',
                   ha='center', va='center', fontsize=16,
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

# Add single shared legend at the bottom center
if legend_lines is not None:
    fig.legend(legend_lines, legend_labels,
              loc='lower center',
              ncol=5,  # 5 metrics in one row
              fontsize=14,
              frameon=True,
              fancybox=False,
              edgecolor='black',
              framealpha=0.95,
              bbox_to_anchor=(0.5, 0.02))

# Output filename
output_base = os.path.join(BASE_DIR, 'combined_panels_ABCD')

# Save as PDF
pdf_file = f"{output_base}.pdf"
plt.savefig(pdf_file, format='pdf', dpi=600, bbox_inches='tight',
            transparent=False, facecolor='white')
print(f"\n  Combined PDF saved: {pdf_file}")

# Save as PNG
png_file = f"{output_base}.png"
plt.savefig(png_file, format='png', dpi=600, bbox_inches='tight',
            transparent=False, facecolor='white')
print(f"  Combined PNG saved: {png_file}")

# Save as EPS
eps_file = f"{output_base}.eps"
plt.savefig(eps_file, format='eps', dpi=600, bbox_inches='tight')
print(f"  Combined EPS saved: {eps_file}")

plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("PUBLICATION-QUALITY FIGURE GENERATION COMPLETE!")
print("="*70)

print("\nINDIVIDUAL PANELS:")
for benchmark, files in panel_files.items():
    panel_letter = panel_labels[benchmark]
    print(f"\n  Panel {panel_letter} ({benchmark}):")
    for fmt, filepath in files.items():
        print(f"    {fmt.upper()}: {os.path.basename(filepath)}")

print("\nCOMBINED LAYOUT (2x2 Grid):")
print(f"  PDF: {os.path.basename(pdf_file)}")
print(f"  PNG: {os.path.basename(png_file)}")
print(f"  EPS: {os.path.basename(eps_file)}")

print("\nGRID LAYOUT:")
print("  Row 1: Panel A (GSE146773) | Panel B (GSE64016)")
print("  Row 2: Panel C (Buettner_mESC) | Panel D (SUP-B15)")
print("  Shared legend at bottom center")

print("\nFIGURE SPECIFICATIONS:")
print("  Font: Times New Roman")
print("  Resolution: 600 DPI")
print("  Formats: PDF (vector), PNG (raster), EPS (vector)")
print("  PDF Font Type: 42 (editable in Adobe Illustrator)")

print("\nFONT SIZES:")
print("  Tick labels: 12pt")
print("  Axis labels: 14pt (bold)")
print("  Legend: 14pt")
print("  Panel titles: 16pt (bold)")

print("\nMETRICS PLOTTED:")
print("  Y-axis (all in %): Accuracy (not in legend), AUC, F1, Kappa, Balanced Accuracy, MCC")
print("  Note: Kappa and MCC multiplied by 100 for visualization")

print("\nLEGEND METRICS (Accuracy excluded):")
print("  AUC, F1, Kappa, Balanced Accuracy, MCC")

print("\nCOLOR SCHEME (Colorblind-friendly):")
for metric, color in COLORS.items():
    legend_status = "" if metric in LEGEND_METRICS or metric == 'Accuracy' else ""
    print(f"  {metric.replace('_', ' ')}: {color}{legend_status}")

print("\n" + "="*70)
print("Files ready for journal submission!")
print("="*70 + "\n")
