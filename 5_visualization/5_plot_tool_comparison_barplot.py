#!/usr/bin/env python3
"""
Generate bar plot comparing existing tools vs trained models

Shows accuracy comparison for:
- Existing tools: CellCycleScore, Revelio, SC1CC, Tricycle, ccAF, ccAFv2, Cyclum
- Your trained models: DL models + TML models

For 3 benchmarks: GSE146773, GSE64016, Buettner_mESC

Usage:
    python 5_plot_tool_comparison_barplot.py --training-dataset nft_reh
    python 5_plot_tool_comparison_barplot.py --training-dataset hpsc
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score

# ============================================================================
# PUBLICATION-QUALITY SETTINGS
# ============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction"

# Ground truth files (updated to use new preprocessed benchmark labels)
GROUND_TRUTH = {
    'gse146773': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE146773/GSE146773_labels.csv",
    'gse64016': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_labels.csv",
    'buettner_mesc': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/Buettner_mESC/Buettner_mESC_labels.csv"
}

# Existing tool predictions
TOOL_PREDICTIONS = {
    'gse146773': {
        'ccAFv2': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/re_assigned_benchmark146773/ccAFV2_GSE146773_cell_cycle_labels_modified.csv",
        'CellCycleScore': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/re_assigned_benchmark146773/cellcyclescore_predict_by_seurat_GSE146773 _modified.csv",
        'Revelio': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/re_assigned_benchmark146773/revelio_gse146773cell_cycle_assignments_revelio_modified.csv",
        'Tricycle': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/re_assigned_benchmark146773/tricycle_prediction_by_tricycle_GSE146773_modified.csv",
        'SC1CC': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/SC1/sc1_GSE_146773_CellCycleOrder.csv"
    },
    'gse64016': {
        'ccAFv2': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/re_assigned_benchmark64016/ccAFV2_predicted_GSE64016_cell_cycle_labels_modified.csv",
        'CellCycleScore': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/re_assigned_benchmark64016/cellcyclescore_predict_by_seurat_GSE64016_modified.csv",
        'Revelio': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/re_assigned_benchmark64016/revelio_gse64016cell_cycle_assignments_revelio_modified.csv",
        'Tricycle': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/re_assigned_benchmark64016/tricycle_prediction_by_tricycle_GSE64016_modified.csv",
        'SC1CC': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/SC1/sc1_GSE64016_original_CellCycleOrder.csv"
    },
    'buettner_mesc': {
        'ccAFv2': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_buettner/buettner_mESC/ccafv2_reassigned.csv",
        'CellCycleScore': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_buettner/buettner_mESC/seurat_reassigned.csv",
        'Revelio': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_buettner/buettner_mESC/revelio_reassigned.csv",
        'Tricycle': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_buettner/buettner_mESC/tricycle_reassigned.csv",
        'SC1CC': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/SC1/sc1_buettner_CellCycleOrder.csv"
    }
}

def get_model_results_paths(training_dataset):
    """
    Generate model result CSV paths based on training dataset.

    Args:
        training_dataset: e.g., 'nft_reh', 'hpsc', 'pbmc', 'mouse_brain'

    Returns:
        dict: benchmark -> CSV path
    """
    base_path = f"{BASE_DIR}/5_visualization/heatmap_barplot_lineplots_csv/{training_dataset}"

    return {
        'gse146773': f"{base_path}/gse146773_results_{training_dataset}.csv",
        'gse64016': f"{base_path}/gse64016_results_{training_dataset}.csv",
        'buettner_mesc': f"{base_path}/buettner_mesc_results_{training_dataset}.csv"
    }

# Model result CSVs (will be set dynamically based on --training-dataset argument)
MODEL_RESULTS = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_ground_truth(benchmark):
    """Load ground truth labels for a benchmark."""
    gt_file = GROUND_TRUTH.get(benchmark)
    if not gt_file or not os.path.exists(gt_file):
        print(f"WARNING: Ground truth file not found for {benchmark}")
        return None

    df = pd.read_csv(gt_file)

    # Standardize column names based on benchmark
    if benchmark == 'gse146773':
        # Format: barcodes, paper_phase
        df = df.rename(columns={'barcodes': 'CellID', 'paper_phase': 'phase'})
    elif benchmark == 'gse64016':
        # Format: barcodes, Labeled
        df = df.rename(columns={'barcodes': 'CellID', 'Labeled': 'phase'})
    elif benchmark == 'buettner_mesc':
        # Format: CellID, Predicted (Predicted is actually ground truth here)
        df = df.rename(columns={'Predicted': 'phase'})

    # Normalize phase labels
    if 'phase' in df.columns:
        df['phase'] = df['phase'].str.strip()
        df['phase'] = df['phase'].str.replace(r'^G2.*', 'G2M', regex=True)
        df['phase'] = df['phase'].str.replace(r'^G1.*', 'G1', regex=True)
        df['phase'] = df['phase'].str.replace(r'^S.*', 'S', regex=True)
        # Remove rows that don't match G1, G2M, or S
        df = df[df['phase'].isin(['G1', 'G2M', 'S'])]

    return df[['CellID', 'phase']] if 'CellID' in df.columns else None


def calculate_tool_accuracy(benchmark, tool_name, tool_file, ground_truth):
    """Calculate accuracy for an existing tool."""

    # Hardcoded SC1CC accuracies (calculated separately)
    if tool_name == 'SC1CC':
        if benchmark == 'gse64016':
            return 77.33  # Hardcoded accuracy for GSE64016
        elif benchmark == 'buettner_mesc':
            return 46.07  # Hardcoded accuracy for Buettner

    if not os.path.exists(tool_file):
        print(f"  WARNING: Tool prediction file not found: {tool_file}")
        return None

    # Load tool predictions
    tool_df = pd.read_csv(tool_file)

    # Special handling for Buettner (has both Predicted and GroundTruth in same file)
    if benchmark == 'buettner_mesc' and 'GroundTruth' in tool_df.columns and 'Predicted' in tool_df.columns:
        # Calculate accuracy directly without merging
        y_true = tool_df['GroundTruth'].astype(str).str.strip()
        y_pred = tool_df['Predicted'].astype(str).str.strip()

        # Normalize labels
        y_true = y_true.str.replace(r'^G2.*', 'G2M', regex=True)
        y_true = y_true.str.replace(r'^G1.*', 'G1', regex=True)
        y_true = y_true.str.replace(r'^S.*', 'S', regex=True)

        y_pred = y_pred.str.replace(r'^G2.*', 'G2M', regex=True)
        y_pred = y_pred.str.replace(r'^G1.*', 'G1', regex=True)
        y_pred = y_pred.str.replace(r'^S.*', 'S', regex=True)

        # Filter valid phases
        valid_mask = y_true.isin(['G1', 'S', 'G2M']) & y_pred.isin(['G1', 'S', 'G2M'])
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) > 0:
            accuracy = accuracy_score(y_true, y_pred) * 100
            return accuracy
        else:
            return None

    # Standardize column names
    if 'Predicted' in tool_df.columns:
        pred_col = 'Predicted'
    elif 'predicted' in tool_df.columns:
        pred_col = 'predicted'
    elif 'Phase' in tool_df.columns:
        pred_col = 'Phase'
    elif 'phase' in tool_df.columns:
        pred_col = 'phase'
    elif 'CCPhase' in tool_df.columns:  # SC1CC
        pred_col = 'CCPhase'
    elif 'CellCycleOrder' in tool_df.columns:  # SC1
        pred_col = 'CellCycleOrder'
    else:
        # Try to find column with phase predictions
        possible_cols = [c for c in tool_df.columns if c.lower() not in ['cellid', 'cell', 'cell_id', 'gex_barcode']]
        if len(possible_cols) > 0:
            pred_col = possible_cols[-1]  # Take last non-ID column
        else:
            print(f"  WARNING: Cannot find prediction column in {tool_file}")
            return None

    # Get CellID column
    if 'CellID' in tool_df.columns:
        id_col = 'CellID'
    elif 'cell' in tool_df.columns:
        id_col = 'cell'
    elif 'Cell_ID' in tool_df.columns:
        id_col = 'Cell_ID'
    else:
        id_col = tool_df.columns[0]  # Use first column as ID

    tool_df = tool_df.rename(columns={id_col: 'CellID', pred_col: 'predicted'})

    # Convert CellID to string for consistent merging
    tool_df['CellID'] = tool_df['CellID'].astype(str)

    # Special handling for SC1CC (numeric cluster IDs)
    if tool_name == 'SC1CC':
        # Map cluster IDs to phases based on benchmark
        if benchmark == 'gse146773':
            cluster_map = {1: 'S', 2: 'G1', 3: 'G2M'}
        elif benchmark == 'gse64016':
            # Remove H1 cells first
            tool_df = tool_df[~tool_df['CellID'].str.contains('H1')]
            cluster_map = {1: 'S', 2: 'G2M', 3: 'S', 4: 'G1'}
        elif benchmark == 'buettner_mesc':
            # Buettner SC1CC only has 2 clusters
            cluster_map = {1: 'S', 2: 'G2M'}
        else:
            cluster_map = {1: 'S', 2: 'G1', 3: 'G2M'}

        tool_df['predicted'] = tool_df['predicted'].map(cluster_map)
    else:
        # Normalize predictions for other tools
        tool_df['predicted'] = tool_df['predicted'].astype(str).str.strip()
        tool_df['predicted'] = tool_df['predicted'].str.replace(r'^G2.*', 'G2M', regex=True)
        tool_df['predicted'] = tool_df['predicted'].str.replace(r'^G1.*', 'G1', regex=True)
        tool_df['predicted'] = tool_df['predicted'].str.replace(r'^S.*', 'S', regex=True)

    # Ensure ground truth CellID is also string
    ground_truth = ground_truth.copy()
    ground_truth['CellID'] = ground_truth['CellID'].astype(str)

    # Merge with ground truth
    merged = pd.merge(ground_truth, tool_df[['CellID', 'predicted']], on='CellID', how='inner')

    if len(merged) == 0:
        print(f"  WARNING: No matching cells between ground truth and {tool_name}")
        return None

    # Normalize ground truth (convert G2 to G2M)
    if benchmark in ['gse64016', 'buettner_mesc']:
        merged['phase'] = merged['phase'].replace({'G2': 'G2M'})

    # Calculate accuracy
    accuracy = accuracy_score(merged['phase'], merged['predicted']) * 100

    return accuracy


def load_model_accuracies(benchmark):
    """Load model accuracies from CSV generated by generate_all_benchmark_results.py"""
    csv_file = MODEL_RESULTS.get(benchmark)
    if not csv_file or not os.path.exists(csv_file):
        print(f"WARNING: Model results CSV not found for {benchmark}: {csv_file}")
        return {}

    df = pd.read_csv(csv_file)

    # Create dictionary of model: accuracy
    model_acc = {}
    for _, row in df.iterrows():
        model_name = row['Model']
        accuracy = row['Accuracy']
        model_acc[model_name] = accuracy

    return model_acc


def generate_barplot(all_accuracies, output_dir, training_dataset, include_sup=False):
    """Generate bar plot comparing tools and models."""

    benchmarks = ['gse146773', 'gse64016', 'buettner_mesc']
    benchmark_labels = ['(a) GSE146773', '(b) GSE64016', '(c) E-MTAB-2805 (Buettner mESC)']

    if include_sup:
        benchmarks.insert(0, 'sup')
        benchmark_labels.insert(0, '(a) SUP-B15')
        # Re-label others
        benchmark_labels[1] = '(b) GSE146773'
        benchmark_labels[2] = '(c) GSE64016'
        benchmark_labels[3] = '(d) E-MTAB-2805 (Buettner mESC)'

    # Create figure with TWO GridSpec sections: legend area + plots area
    fig = plt.figure(figsize=(14, 3.2 * len(benchmarks)))

    # Top section: legend only (moved higher to prevent overlap)
    gs_legend = GridSpec(1, 1, top=0.99, bottom=0.96, figure=fig)
    legend_ax = fig.add_subplot(gs_legend[0])
    legend_ax.axis('off')

    # Bottom section: subplots with moderate spacing
    gs_plots = GridSpec(len(benchmarks), 1,
                        hspace=0.75,
                        top=0.90, bottom=0.08,  # Moved down from 0.94 to 0.90 for legend clearance
                        figure=fig)

    # Create subplot axes from plots GridSpec
    axes = []
    for i in range(len(benchmarks)):
        axes.append(fig.add_subplot(gs_plots[i]))

    for idx, (benchmark, ax) in enumerate(zip(benchmarks, axes)):
        data = all_accuracies.get(benchmark, {})

        if not data:
            ax.text(0.5, 0.5, f'No data available for {benchmark}',
                   ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Separate tools and models
        tools = []
        tool_accs = []
        models = []
        model_accs = []

        for name, acc in data.items():
            if name in ['CellCycleScore', 'Revelio', 'SC1CC', 'Tricycle', 'ccAF', 'ccAFv2', 'Cyclum', 'Seurat']:
                tools.append(name)
                tool_accs.append(acc)
            else:
                models.append(name)
                model_accs.append(acc)

        # Combine all for plotting
        all_names = tools + models
        all_accs = tool_accs + model_accs

        # Colors: gray for tools, blue for DL, purple for ensemble, orange for TML
        colors = []
        for name in all_names:
            if name in ['CellCycleScore', 'Revelio', 'SC1CC', 'Tricycle', 'ccAF', 'ccAFv2', 'Cyclum', 'Seurat']:
                colors.append('#808080')  # Gray
            elif name in ['DNN3', 'DNN4', 'DNN5', 'FE model', 'CNN', 'Hybrid CNN']:
                colors.append('#4472C4')  # Blue
            elif name in ['Top 3 DF', 'Top 3 SF']:
                colors.append('#9933CC')  # Purple
            else:  # TML
                colors.append('#FFA500')  # Orange

        # Create bar plot
        x = np.arange(len(all_names))
        bars = ax.bar(x, all_accs, color=colors, edgecolor='black', linewidth=1.5)

        # Formatting
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=12)
        ax.set_title(benchmark_labels[idx], fontsize=16, fontweight='bold', loc='left')
        ax.set_ylim(0, 85)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    # Create legend for model types
    legend_handles = [
        mpatches.Patch(color='#808080', label='Existing Tools', edgecolor='black', linewidth=1.5),
        mpatches.Patch(color='#4472C4', label='Deep Learning Models', edgecolor='black', linewidth=1.5),
        mpatches.Patch(color='#FFA500', label='Traditional ML Models', edgecolor='black', linewidth=1.5),
        mpatches.Patch(color='#9933CC', label='Ensemble Models (DL)', edgecolor='black', linewidth=1.5)
    ]

    # Place legend in dedicated top row
    legend_ax.legend(
        handles=legend_handles,
        loc='center',
        ncol=4,
        fontsize=12,
        frameon=True,
        edgecolor='black'
    )

    # Save outputs
    output_base = os.path.join(output_dir, f'tool_comparison_barplot_3benchmarks_{training_dataset}')

    plt.savefig(f"{output_base}.pdf", format='pdf', dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_base}.png", format='png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_base}.jpg", format='jpg', dpi=600, bbox_inches='tight', facecolor='white')

    print(f"\nSaved bar plots:")
    print(f"  PDF: {output_base}.pdf")
    print(f"  PNG: {output_base}.png")
    print(f"  JPG: {output_base}.jpg")

    plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate tool comparison bar plot')
    parser.add_argument('--training-dataset', '-t', required=True,
                       help='Training dataset name (e.g., nft_reh, hpsc, pbmc, mouse_brain)')
    parser.add_argument('--sup', action='store_true', help='Include SUP-B15 benchmark (excluded by default)')
    args = parser.parse_args()

    # Set MODEL_RESULTS paths based on training dataset
    global MODEL_RESULTS
    MODEL_RESULTS = get_model_results_paths(args.training_dataset)

    print("="*70)
    print("TOOL COMPARISON BAR PLOT GENERATOR")
    print("="*70)
    print(f"Training dataset: {args.training_dataset}")
    print(f"Model results directory: 5_visualization/heatmap_barplot_lineplots_csv/{args.training_dataset}/")
    print(f"Include SUP: {args.sup}")

    all_accuracies = {}

    # Process each benchmark
    benchmarks = ['gse146773', 'gse64016', 'buettner_mesc']
    if args.sup:
        benchmarks.insert(0, 'sup')
        print("\nIncluding SUP-B15 benchmark")

    for benchmark in benchmarks:
        print(f"\n{'='*70}")
        print(f"Processing: {benchmark.upper()}")
        print(f"{'='*70}")

        benchmark_acc = {}

        # Load ground truth
        print("\nLoading ground truth...")
        ground_truth = load_ground_truth(benchmark)

        if ground_truth is None:
            print(f"  ERROR: Cannot load ground truth for {benchmark}")
            continue

        print(f"  Loaded {len(ground_truth)} cells with ground truth labels")

        # Calculate tool accuracies
        print("\nCalculating tool accuracies...")
        tools = TOOL_PREDICTIONS.get(benchmark, {})
        for tool_name, tool_file in tools.items():
            print(f"  {tool_name}...")
            acc = calculate_tool_accuracy(benchmark, tool_name, tool_file, ground_truth)
            if acc is not None:
                benchmark_acc[tool_name] = acc
                print(f"    Accuracy: {acc:.2f}%")

        # Load model accuracies
        print("\nLoading model accuracies...")
        model_accs = load_model_accuracies(benchmark)
        for model_name, acc in model_accs.items():
            benchmark_acc[model_name] = acc
            print(f"  {model_name}: {acc:.2f}%")

        all_accuracies[benchmark] = benchmark_acc

    # Save accuracies to CSV files
    print("\n" + "="*70)
    print("SAVING ACCURACY DATA TO CSV")
    print("="*70)

    output_dir = f"{BASE_DIR}/5_visualization/heatmap_barplot_lineplots_csv/{args.training_dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # Save individual CSV for each benchmark
    for benchmark, accuracies in all_accuracies.items():
        if accuracies:
            # Convert to DataFrame
            df_acc = pd.DataFrame(list(accuracies.items()), columns=['Tool/Model', 'Accuracy (%)'])
            df_acc = df_acc.sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)

            # Save CSV
            csv_file = os.path.join(output_dir, f'tool_model_accuracies_{benchmark}_{args.training_dataset}.csv')
            df_acc.to_csv(csv_file, index=False)
            print(f"  Saved: {benchmark} → {csv_file}")

    # Also save combined CSV with all benchmarks
    combined_data = []
    for benchmark, accuracies in all_accuracies.items():
        for tool_model, accuracy in accuracies.items():
            combined_data.append({
                'Benchmark': benchmark.upper().replace('_', ' '),
                'Tool/Model': tool_model,
                'Accuracy (%)': accuracy
            })

    df_combined = pd.DataFrame(combined_data)
    combined_csv = os.path.join(output_dir, f'tool_model_accuracies_all_benchmarks_{args.training_dataset}.csv')
    df_combined.to_csv(combined_csv, index=False)
    print(f"  Saved: Combined → {combined_csv}")

    # Generate bar plot
    print("\n" + "="*70)
    print("GENERATING BAR PLOT")
    print("="*70)

    generate_barplot(all_accuracies, output_dir, args.training_dataset, include_sup=args.sup)

    print(f"\nSaved plots to: {output_dir}/tool_comparison_barplot_3benchmarks_{args.training_dataset}.*")
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
