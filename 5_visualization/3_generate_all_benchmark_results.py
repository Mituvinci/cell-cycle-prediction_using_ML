#!/usr/bin/env python3
"""
Comprehensive Benchmark Results Generator
==========================================

This script does EVERYTHING in one go:
1. Extracts best models from 5-fold CV for ALL benchmarks
2. Calculates Top3DF and Top3SF ensemble fusion
3. Generates CSV files for each benchmark
4. Generates precision/recall heatmap for all benchmarks

Usage:
    python generate_all_benchmark_results.py --input <csv_file> [--skip-ensemble] [--skip-heatmap]

Arguments:
    --input: Path to consolidated CSV file with all model results
    --skip-ensemble: Skip Top3 ensemble fusion calculation (faster)
    --skip-heatmap: Skip heatmap generation
    --model-dir: Base directory for .pt model files (optional)

Output:
    - sup_results_reh.csv
    - gse146773_results_reh.csv
    - gse64016_results_reh.csv
    - buettner_mesc_results_reh.csv
    - precision_recall_heatmap_3benchmarks.pdf/png/jpg
"""

import pandas as pd
import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import subprocess

# Add paths for ensemble fusion
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../3_evaluation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../2_model_training'))

from ensemble_fusion import score_level_fusion, decision_level_fusion

# ============================================================================
# PUBLICATION-QUALITY SETTINGS FOR HEATMAP
# ============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600

# ============================================================================
# MODEL NAME MAPPING AND ORDERING
# ============================================================================

MODEL_NAME_MAPPING = {
    'simpledense': 'DNN3',
    'sd': 'DNN3',
    'enhancedense': 'DNN5',
    'ed': 'DNN5',
    'deepdense': 'DNN4',
    'dd': 'DNN4',
    'featureembedding': 'FE model',
    'fe': 'FE model',
    'cnn': 'CNN',
    'hybridcnn': 'Hybrid CNN',
    'hbdcnn': 'Hybrid CNN',
    'ensemble': 'Embedding3TML',
    'lightgbm': 'LGBM',
    'lgbm': 'LGBM',
    'adaboost': 'Adaboost',
    'randomforest': 'Random Forest',
    'random': 'Random Forest',
    'rf': 'Random Forest'
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


def fix_metric_scale(value):
    """
    Fix MCC/Kappa scale inconsistency.
    Both should be -1 to +1, but some models incorrectly output 0-100.
    If value > 1, divide by 100.
    """
    if pd.isna(value):
        return value
    if abs(value) > 1:
        return value / 100.0
    return value

MODEL_DISPLAY_NAMES_HEATMAP = {
    'DNN3': 'DNN3',
    'DNN4': 'DNN4',
    'DNN5': 'DNN5',
    'FE model': 'FE',
    'CNN': 'CNN',
    'Hybrid CNN': 'Hybrid CNN',
    'Top 3 DF': 'Top_3_D.F.',
    'Top 3 SF': 'Top_3_S.F.',
    'LGBM': 'LGBM',
    'Adaboost': 'Adaboost',
    'Random Forest': 'RF',
    'Embedding3TML': 'Embedding3TML'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_training_dataset(prefix_name):
    """
    Extract training dataset name from prefix_name.

    Pattern: *_DATASET_fld_*

    Examples:
        - "fe_NFT_reh_fld_1" -> "reh"
        - "fe_NFT_hpsc_fld_1" -> "hpsc"
        - "fe_NFT_mouse_brain_fld_1" -> "mouse_brain"
        - "simpledense_pbmc_fld_2" -> "pbmc"
    """
    match = re.search(r'_([a-z_]+)_fld', prefix_name.lower())
    if match:
        return match.group(1)
    return None


def extract_model_architecture(model_name):
    """Extract base model architecture from full model name."""
    return model_name.lower().split('_')[0]


def map_to_display_name(architecture):
    """Map architecture name to display name."""
    return MODEL_NAME_MAPPING.get(architecture, architecture)


def find_model_file(prefix_name, base_search_dir=None):
    """Find .pt model file by prefix_name."""
    if base_search_dir is None:
        base_search_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/models"

    search_pattern = f"{base_search_dir}/**/{prefix_name}.pt"
    matches = glob.glob(search_pattern, recursive=True)

    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        return matches[0]
    else:
        return matches[0]


def extract_best_models_for_benchmark(df, benchmark, base_search_dir=None):
    """
    Extract best performing model for each architecture on specified benchmark.

    Returns:
        tuple: (best_models_df, top3_info, best_models_full_data)
    """
    df['architecture'] = df['Model'].apply(extract_model_architecture)

    best_models = []
    all_models_info = []
    best_models_full = []  # Store full data for heatmap

    for architecture in df['architecture'].unique():
        # Skip DNN4 (deepdense) - removed from all visualizations
        if architecture.lower() in ['deepdense', 'dd', 'dnn4']:
            continue

        arch_models = df[df['architecture'] == architecture].copy()
        best_idx = arch_models[f'{benchmark}_accuracy'].idxmax()
        best_model = arch_models.loc[best_idx]

        display_name = map_to_display_name(architecture)

        # Metrics for CSV output
        # Fix MCC/Kappa scale (should be -1 to +1, not 0-100)
        mcc_value = fix_metric_scale(best_model[f'{benchmark}_mcc'])
        kappa_value = fix_metric_scale(best_model[f'{benchmark}_kappa'])

        model_data = {
            'Model': display_name,
            'Accuracy': round(best_model[f'{benchmark}_accuracy'], 2),
            'AUC': round(best_model[f'{benchmark}_roc_auc'], 2) if pd.notna(best_model[f'{benchmark}_roc_auc']) else None,
            'Precision': round(best_model[f'{benchmark}_precision'], 2),
            'Recall': round(best_model[f'{benchmark}_recall'], 2),
            'F1': round(best_model[f'{benchmark}_f1'], 2),
            'MCC': round(mcc_value, 4) if pd.notna(mcc_value) else None,
            'Kappa': round(kappa_value, 4) if pd.notna(kappa_value) else None,
            'Balanced_Accuracy': round(best_model[f'{benchmark}_balanced_acc'], 2),
            'Training_Time': round(best_model['training_time'], 2),
            'CPU_Memory_MB': round(best_model['cpu_memory_used'], 2) if pd.notna(best_model['cpu_memory_used']) else None,
            'GPU_Memory_MB': round(best_model['gpu_memory_used'], 2) if pd.notna(best_model['gpu_memory_used']) else None
        }

        # Add per-class precision and recall (if available)
        # DL models have these in 0-1 scale, need to multiply by 100
        dl_models = ['DNN3', 'DNN5', 'FE model', 'CNN', 'Hybrid CNN']
        is_dl_model = display_name in dl_models

        for phase in ['g1', 'g2m', 's']:
            prec_col = f'{benchmark}_precision_{phase}'
            rec_col = f'{benchmark}_recall_{phase}'

            if prec_col in best_model.index:
                prec_val = best_model[prec_col]
                # DL models: always multiply by 100 (they're in 0-1 scale)
                if pd.notna(prec_val) and is_dl_model:
                    prec_val = prec_val * 100
                model_data[f'precision_{phase}'] = round(prec_val, 2) if pd.notna(prec_val) else None

            if rec_col in best_model.index:
                rec_val = best_model[rec_col]
                # DL models: always multiply by 100 (they're in 0-1 scale)
                if pd.notna(rec_val) and is_dl_model:
                    rec_val = rec_val * 100
                model_data[f'recall_{phase}'] = round(rec_val, 2) if pd.notna(rec_val) else None

        best_models.append(model_data)

        # Store full data for heatmap (includes per-class metrics)
        full_data = {'Model': display_name}
        for col in best_model.index:
            if col.startswith(f'{benchmark}_'):
                # Remove benchmark prefix for easier access
                clean_col = col.replace(f'{benchmark}_', '')
                value = best_model[col]

                # DL models: multiply per-class precision/recall by 100
                if is_dl_model and any(clean_col.startswith(prefix) for prefix in ['precision_', 'recall_']):
                    if pd.notna(value):
                        value = value * 100

                full_data[clean_col] = value
        best_models_full.append(full_data)

        # Store for top 3 selection
        all_models_info.append({
            'display_name': display_name,
            'prefix_name': best_model['prefix_name'],
            'accuracy': best_model[f'{benchmark}_accuracy']
        })

    output_df = pd.DataFrame(best_models).sort_values('Accuracy', ascending=False).reset_index(drop=True)

    # Find top 3 DL models
    dl_models = [m for m in all_models_info if m['display_name'] not in ['LGBM', 'Adaboost', 'Random Forest', 'Embedding3TML']]
    top3_models = sorted(dl_models, key=lambda x: x['accuracy'], reverse=True)[:3]

    top3_info = []
    for model_info in top3_models:
        model_path = find_model_file(model_info['prefix_name'], base_search_dir)
        if model_path:
            top3_info.append((model_path, model_info['display_name'], model_info['accuracy']))

    return output_df, top3_info, pd.DataFrame(best_models_full)


def run_ensemble_fusion(top3_info, benchmark):
    """Run ensemble fusion and return metrics."""
    if len(top3_info) != 3:
        return None, None

    model_paths = [info[0] for info in top3_info]
    benchmark_map = {
        'sup': 'SUP',
        'gse146773': 'GSE146773',
        'gse64016': 'GSE64016',
        'buettner_mesc': 'Buettner_mESC'
    }
    fusion_benchmark = benchmark_map[benchmark.lower()]

    try:
        df_result = decision_level_fusion(model_paths, fusion_benchmark)

        # Fix MCC/Kappa scale
        mcc_val = fix_metric_scale(df_result[f'{benchmark}_mcc'].values[0])
        kappa_val = fix_metric_scale(df_result[f'{benchmark}_kappa'].values[0])

        top3df_metrics = {
            'Model': 'Top 3 DF',
            'Accuracy': round(df_result[f'{benchmark}_accuracy'].values[0], 2),
            'AUC': None,
            'Precision': round(df_result[f'{benchmark}_precision'].values[0], 2),
            'Recall': round(df_result[f'{benchmark}_recall'].values[0], 2),
            'F1': round(df_result[f'{benchmark}_f1'].values[0], 2),
            'MCC': round(mcc_val, 4) if pd.notna(mcc_val) else None,
            'Kappa': round(kappa_val, 4) if pd.notna(kappa_val) else None,
            'Balanced_Accuracy': round(df_result[f'{benchmark}_balanced_acc'].values[0], 2),
            'Training_Time': None,
            'CPU_Memory_MB': None,
            'GPU_Memory_MB': None
        }

        # Add per-class precision and recall
        for phase in ['g1', 'g2m', 's']:
            prec_col = f'{benchmark}_precision_{phase}'
            rec_col = f'{benchmark}_recall_{phase}'

            if prec_col in df_result.columns:
                top3df_metrics[f'precision_{phase}'] = round(df_result[prec_col].values[0], 2)
            if rec_col in df_result.columns:
                top3df_metrics[f'recall_{phase}'] = round(df_result[rec_col].values[0], 2)

    except Exception as e:
        print(f"  ERROR in Decision Fusion: {e}")
        top3df_metrics = None

    try:
        sf_result = score_level_fusion(model_paths, fusion_benchmark)

        # Fix MCC/Kappa scale
        mcc_val = fix_metric_scale(sf_result[f'{benchmark}_mcc'].values[0])
        kappa_val = fix_metric_scale(sf_result[f'{benchmark}_kappa'].values[0])

        top3sf_metrics = {
            'Model': 'Top 3 SF',
            'Accuracy': round(sf_result[f'{benchmark}_accuracy'].values[0], 2),
            'AUC': round(sf_result[f'{benchmark}_roc_auc'].values[0], 2) if f'{benchmark}_roc_auc' in sf_result.columns else None,
            'Precision': round(sf_result[f'{benchmark}_precision'].values[0], 2),
            'Recall': round(sf_result[f'{benchmark}_recall'].values[0], 2),
            'F1': round(sf_result[f'{benchmark}_f1'].values[0], 2),
            'MCC': round(mcc_val, 4) if pd.notna(mcc_val) else None,
            'Kappa': round(kappa_val, 4) if pd.notna(kappa_val) else None,
            'Balanced_Accuracy': round(sf_result[f'{benchmark}_balanced_acc'].values[0], 2),
            'Training_Time': None,
            'CPU_Memory_MB': None,
            'GPU_Memory_MB': None
        }

        # Add per-class precision and recall
        for phase in ['g1', 'g2m', 's']:
            prec_col = f'{benchmark}_precision_{phase}'
            rec_col = f'{benchmark}_recall_{phase}'

            if prec_col in sf_result.columns:
                top3sf_metrics[f'precision_{phase}'] = round(sf_result[prec_col].values[0], 2)
            if rec_col in sf_result.columns:
                top3sf_metrics[f'recall_{phase}'] = round(sf_result[rec_col].values[0], 2)

    except Exception as e:
        print(f"  ERROR in Score Fusion: {e}")
        top3sf_metrics = None

    return top3df_metrics, top3sf_metrics


def generate_heatmap(all_benchmark_data, output_dir):
    """Generate precision/recall heatmap for all benchmarks."""

    print("\n" + "="*70)
    print("GENERATING PRECISION/RECALL HEATMAP")
    print("="*70)

    # Prepare data
    benchmarks = ['gse146773', 'gse64016', 'buettner_mesc']
    phases = ['g1', 'g2m', 's']
    phase_labels = ['G1', 'G2M', 'S']

    # Benchmark display names for heatmap
    benchmark_display_names = {
        'gse146773': 'GSE146773',
        'gse64016': 'GSE64016',
        'buettner_mesc': 'Buettner mESC'
    }

    # Get model order from first benchmark
    first_benchmark = benchmarks[0]
    if first_benchmark not in all_benchmark_data:
        print("ERROR: No data available for heatmap")
        return

    # Sort by predefined model order (including Top 3 DF/SF)
    df_first = all_benchmark_data[first_benchmark].copy()
    df_first['sort_key'] = df_first['Model'].apply(lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999)
    df_first = df_first.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)

    models = df_first['Model'].tolist()
    display_models = [MODEL_DISPLAY_NAMES_HEATMAP.get(m, m) for m in models]

    # Build data matrix
    all_columns = []
    col_structure = []

    for benchmark in benchmarks:
        if benchmark not in all_benchmark_data:
            continue

        # Sort by predefined model order (including Top 3 DF/SF)
        df = all_benchmark_data[benchmark].copy()
        df['sort_key'] = df['Model'].apply(lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999)
        df = df.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)

        # Precision columns
        for i, phase in enumerate(phases):
            col_name = f'precision_{phase}'
            if col_name in df.columns:
                all_columns.append(df[col_name].values)
                col_structure.append((benchmark_display_names[benchmark], 'Precision', phase_labels[i]))

        # Recall columns
        for i, phase in enumerate(phases):
            col_name = f'recall_{phase}'
            if col_name in df.columns:
                all_columns.append(df[col_name].values)
                col_structure.append((benchmark_display_names[benchmark], 'Recall', phase_labels[i]))

    if len(all_columns) == 0:
        print("ERROR: No precision/recall data found")
        return

    heatmap_data = np.column_stack(all_columns)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.heatmap(heatmap_data,
                cmap='Blues',
                vmin=0,
                vmax=100,
                cbar_kws={'label': '', 'pad': 0.02, 'aspect': 40},
                linewidths=1.5,
                linecolor='black',
                ax=ax,
                square=False)

    # Row labels
    ax.set_yticks(np.arange(len(display_models)) + 0.5)
    ax.set_yticklabels(display_models, fontsize=14, fontweight='bold', rotation=0)

    # Column labels (bottom)
    ax.set_xticks(np.arange(len(col_structure)) + 0.5)
    ax.set_xticklabels([c[2] for c in col_structure], fontsize=12, fontweight='bold', rotation=0)

    # Benchmark labels (top)
    benchmark_positions = []
    current_benchmark = None
    start_idx = 0

    for i, (benchmark, metric, phase) in enumerate(col_structure):
        if benchmark != current_benchmark:
            if current_benchmark is not None:
                mid_point = (start_idx + i) / 2
                benchmark_positions.append((current_benchmark, mid_point))
                start_idx = i
            current_benchmark = benchmark

    mid_point = (start_idx + len(col_structure)) / 2
    benchmark_positions.append((current_benchmark, mid_point))

    for benchmark, pos in benchmark_positions:
        ax.text(pos, -2.5, benchmark, ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Metric labels (middle) with red lines
    metric_positions = []
    current_key = None
    start_idx = 0

    for i, (benchmark, metric, phase) in enumerate(col_structure):
        key = (benchmark, metric)
        if key != current_key:
            if current_key is not None:
                mid_point = (start_idx + i) / 2
                metric_positions.append((current_key[1], mid_point, start_idx, i))
                start_idx = i
            current_key = key

    mid_point = (start_idx + len(col_structure)) / 2
    metric_positions.append((current_key[1], mid_point, start_idx, len(col_structure)))

    for metric, pos, start, end in metric_positions:
        ax.text(pos, -1.2, metric, ha='center', va='bottom', fontsize=14)
        # Red bar removed per user request
        # if metric == 'Precision':
        #     ax.plot([start, end], [-0.5, -0.5], color='red', linewidth=3, clip_on=False)

    # Vertical separators
    current_benchmark = col_structure[0][0]
    for i, (benchmark, metric, phase) in enumerate(col_structure):
        if benchmark != current_benchmark:
            ax.axvline(x=i, color='black', linewidth=3)
            current_benchmark = benchmark

    # Colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)

    # Save
    output_base = os.path.join(output_dir, 'precision_recall_heatmap_3benchmarks')

    plt.savefig(f"{output_base}.pdf", format='pdf', dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_base}.png", format='png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_base}.jpg", format='jpg', dpi=600, bbox_inches='tight', facecolor='white')

    print(f"\n  Saved heatmap:")
    print(f"    PDF: {output_base}.pdf")
    print(f"    PNG: {output_base}.png")
    print(f"    JPG: {output_base}.jpg")

    plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate all benchmark results + heatmap')
    parser.add_argument('--input', '-i', required=True, help='Path to consolidated CSV file')
    parser.add_argument('--model-dir', '-m', default=None, help='Base directory for model files')
    parser.add_argument('--skip-ensemble', action='store_true', help='Skip ensemble fusion')
    parser.add_argument('--skip-heatmap', action='store_true', help='Skip heatmap generation')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print("="*70)
    print("COMPREHENSIVE BENCHMARK RESULTS GENERATOR")
    print("="*70)
    print(f"\nInput: {args.input}")
    print(f"Skip ensemble: {args.skip_ensemble}")
    print(f"Skip heatmap: {args.skip_heatmap}")

    # Load consolidated data
    print("\nLoading consolidated data...")
    df_all = pd.read_csv(args.input)
    print(f"  Total models: {len(df_all)}")

    # Extract training dataset name from first row
    training_dataset = extract_training_dataset(df_all['prefix_name'].iloc[0])
    if training_dataset is None:
        print("ERROR: Could not extract training dataset name from prefix_name")
        sys.exit(1)
    print(f"  Detected training dataset: {training_dataset}")

    # Process each benchmark
    benchmarks = ['sup', 'gse146773', 'gse64016', 'buettner_mesc']
    all_benchmark_data = {}

    #output_dir = os.path.dirname(args.input) if os.path.dirname(args.input) else '.'
    output_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/5_visualization/heatmap_barplot_lineplots_csv/"+training_dataset
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in benchmarks:
        print(f"\n{'='*70}")
        print(f"PROCESSING: {benchmark.upper()}")
        print(f"{'='*70}")

        best_df, top3_info, full_data = extract_best_models_for_benchmark(df_all, benchmark, args.model_dir)

        print(f"\nBest models extracted: {len(best_df)}")

        # Ensemble fusion
        if not args.skip_ensemble and len(top3_info) == 3:
            print(f"\nRunning ensemble fusion...")
            print(f"  Top 3 models: {', '.join([t[1] for t in top3_info])}")

            top3df, top3sf = run_ensemble_fusion(top3_info, benchmark)

            if top3df:
                best_df = pd.concat([best_df, pd.DataFrame([top3df])], ignore_index=True)
                # Also add to full_data for heatmap
                full_data = pd.concat([full_data, pd.DataFrame([top3df])], ignore_index=True)

            if top3sf:
                best_df = pd.concat([best_df, pd.DataFrame([top3sf])], ignore_index=True)
                # Also add to full_data for heatmap
                full_data = pd.concat([full_data, pd.DataFrame([top3sf])], ignore_index=True)

        # Sort models by predefined order
        best_df = sort_models_by_order(best_df)

        # Save CSV
        output_file = os.path.join(output_dir, f'{benchmark}_results_{training_dataset}.csv')
        best_df.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")

        # Store for heatmap (only if has per-class data)
        if benchmark != 'sup':  # SUP doesn't need heatmap
            all_benchmark_data[benchmark] = full_data

    # Generate heatmap
    if not args.skip_heatmap and len(all_benchmark_data) > 0:
        generate_heatmap(all_benchmark_data, output_dir)

    # Automatically call plotting script
    print("\n" + "="*70)
    print("GENERATING BENCHMARK COMPARISON PLOTS")
    print("="*70)

    plot_script = os.path.join(os.path.dirname(__file__), '4_plot_benchmark_results.py')
    if os.path.exists(plot_script):
        try:
            print(f"\nCalling: {plot_script}")
            print(f"  Input directory: {output_dir}")
            print(f"  Training dataset: {training_dataset}")

            subprocess.run([
                'python', plot_script,
                '--input-dir', output_dir,
                '--training-dataset', training_dataset
            ], check=True)

            print("\nBenchmark plots generated successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\nWARNING: Plotting script failed with error: {e}")
        except Exception as e:
            print(f"\nWARNING: Could not run plotting script: {e}")
    else:
        print(f"\nWARNING: Plotting script not found: {plot_script}")

    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"\nGenerated files in: {output_dir}/")
    print(f"  - sup_results_{training_dataset}.csv")
    print(f"  - gse146773_results_{training_dataset}.csv")
    print(f"  - gse64016_results_{training_dataset}.csv")
    print(f"  - buettner_mesc_results_{training_dataset}.csv")
    if not args.skip_heatmap:
        print(f"  - precision_recall_heatmap_3benchmarks.pdf/png/jpg")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
