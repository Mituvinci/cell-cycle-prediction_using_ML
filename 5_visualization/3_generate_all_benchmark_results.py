#!/usr/bin/env python3
"""
FIXED Comprehensive Benchmark Results Generator
================================================

IMPORTANT CHANGES FROM ORIGINAL:
1. Uses FIXED Top-3 models for ALL benchmarks (not different per benchmark)
2. NEVER uses DeepDense (always excluded)
3. For REH: Uses SimpleDense, EnhanceDense, FeatureEmbedding by default
4. For other datasets: Can specify different fixed models

This script does EVERYTHING in one go:
1. Extracts best models from 5-fold CV for ALL benchmarks
2. Calculates Top3DF and Top3SF ensemble fusion with FIXED models
3. Generates CSV files for each benchmark
4. Generates precision/recall heatmap for all benchmarks

Usage:
    # REH models (uses default: SimpleDense, EnhanceDense, FE)
    python 3_generate_all_benchmark_results_FIXED.py \
        --input ../results/double/consolidated_reh_7ds.csv \
        --model-dir ../models/submitted_human_models/Deeplearning_ML/REH

    # Custom fixed models
    python 3_generate_all_benchmark_results_FIXED.py \
        --input ../results/double/consolidated_reh_7ds.csv \
        --model-dir ../models/submitted_human_models/Deeplearning_ML/REH \
        --fixed-models simpledense_NFT_reh_fld_2 enhancedense_NFT_reh_fld_3 fe_NFT_reh_fld_5

Arguments:
    --input: Path to consolidated CSV file with all model results
    --model-dir: Base directory for .pt model files (required for ensemble)
    --fixed-models: Space-separated list of 3 model prefix names (optional, auto-detected if not provided)
    --skip-ensemble: Skip Top3 ensemble fusion calculation (faster)
    --skip-heatmap: Skip heatmap generation

Output:
    - sup_results_<dataset>.csv
    - gse146773_results_<dataset>.csv
    - gse64016_results_<dataset>.csv
    - buettner_mesc_results_<dataset>.csv
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
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Model name mapping
MODEL_NAME_MAP = {
    'simpledense': 'DNN3',
    'sd': 'DNN3',
    'dnn3': 'DNN3',
    'deepdense': 'DNN4',  # EXCLUDED - NEVER USE
    'dd': 'DNN4',  # EXCLUDED - NEVER USE
    'dnn4': 'DNN4',  # EXCLUDED - NEVER USE
    'enhancedense': 'DNN5',
    'ed': 'DNN5',
    'dnn5': 'DNN5',
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
}

# Consistent model order for ALL visualizations (DNN4/DeepDense REMOVED)
MODEL_ORDER = [
    'DNN3',
    'DNN5',
    'CNN',
    'Hybrid CNN',
    'FE model',
    'LGBM',
    'Adaboost',
    'Random Forest',
    'Embedding3TML',
    'Top 3 DF',
    'Top 3 SF'
]

def sort_models_by_order(df):
    """Sort DataFrame by predefined model order."""
    df['sort_key'] = df['Model'].apply(lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999)
    df = df.sort_values('sort_key').drop('sort_key', axis=1)
    return df

def fix_metric_scale(value):
    """Fix MCC/Kappa scale if needed (should be -1 to +1, not 0-100)."""
    if pd.isna(value):
        return value
    if value > 1.5:
        return value / 100.0
    return value

# Compact display names for heatmap
HEATMAP_MODEL_NAMES = {
    'DNN3': 'DNN3',
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

def extract_training_dataset(prefix_name):
    """
    Extract training dataset name from prefix_name.

    Examples:
        - "fe_NFT_reh_fld_1" -> "reh"
        - "fe_NFT_hpsc_fld_1" -> "hpsc"
        - "fe_NFT_mouse_brain_fld_1" -> "mouse_brain"
        - "simpledense_pbmc_fld_2" -> "pbmc"
    """
    match = re.search(r'_([a-z_]+)_fld', prefix_name.lower())
    if match:
        dataset = match.group(1)
        # Add 'nft_' prefix for consistency
        if not dataset.startswith('nft_'):
            return f"nft_{dataset}"
        return dataset
    return None

def find_model_file(prefix_name, base_search_dir=None):
    """Find .pt model file by prefix_name."""
    if base_search_dir is None:
        base_search_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/models"

    search_pattern = f"{base_search_dir}/**/{prefix_name}.pt"
    matches = glob.glob(search_pattern, recursive=True)

    if len(matches) == 0:
        return None
    else:
        return matches[0]

def extract_model_architecture(model_name):
    """Extract architecture name from model column."""
    model_lower = model_name.lower()
    if 'simpledense' in model_lower or 'dnn3' in model_lower:
        return 'simpledense'
    elif 'deepdense' in model_lower or 'dnn4' in model_lower:
        return 'deepdense'
    elif 'enhancedense' in model_lower or 'dnn5' in model_lower:
        return 'enhancedense'
    elif 'featureembedding' in model_lower or 'fe_' in model_lower or model_lower == 'fe':
        return 'featureembedding'
    elif 'hybrid' in model_lower or 'hbdcnn' in model_lower:
        return 'hybridcnn'
    elif 'cnn' in model_lower and 'hybrid' not in model_lower:
        return 'cnn'
    elif 'ensemble' in model_lower:
        return 'ensemble'
    elif 'lightgbm' in model_lower or 'lgbm' in model_lower:
        return 'lightgbm'
    elif 'adaboost' in model_lower:
        return 'adaboost'
    elif 'random' in model_lower or 'rf' in model_lower:
        return 'randomforest'
    return model_name

def map_to_display_name(architecture):
    """Map architecture to display name."""
    arch_lower = architecture.lower()
    return MODEL_NAME_MAP.get(arch_lower, architecture)

def extract_best_models_for_benchmark(df, benchmark, base_search_dir=None):
    """
    Extract best performing model for each architecture on specified benchmark.

    IMPORTANT: DeepDense (DNN4) is ALWAYS excluded.

    Returns:
        tuple: (best_models_df, best_models_full_data)
    """
    df['architecture'] = df['Model'].apply(extract_model_architecture)

    best_models = []
    best_models_full = []

    for architecture in df['architecture'].unique():
        # CRITICAL: Skip DeepDense (DNN4) - ALWAYS excluded
        if architecture.lower() in ['deepdense', 'dd', 'dnn4']:
            print(f"  Skipping DeepDense (DNN4) - excluded from analysis")
            continue

        arch_models = df[df['architecture'] == architecture].copy()
        best_idx = arch_models[f'{benchmark}_accuracy'].idxmax()
        best_model = arch_models.loc[best_idx]

        display_name = map_to_display_name(architecture)

        # Metrics for CSV output
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

        # Add per-class precision and recall
        dl_models = ['DNN3', 'DNN5', 'FE model', 'CNN', 'Hybrid CNN']
        is_dl_model = display_name in dl_models

        for phase in ['g1', 'g2m', 's']:
            prec_col = f'{benchmark}_precision_{phase}'
            rec_col = f'{benchmark}_recall_{phase}'

            if prec_col in best_model.index:
                prec_val = best_model[prec_col]
                if pd.notna(prec_val) and is_dl_model:
                    prec_val = prec_val * 100
                model_data[f'precision_{phase}'] = round(prec_val, 2) if pd.notna(prec_val) else None

            if rec_col in best_model.index:
                rec_val = best_model[rec_col]
                if pd.notna(rec_val) and is_dl_model:
                    rec_val = rec_val * 100
                model_data[f'recall_{phase}'] = round(rec_val, 2) if pd.notna(rec_val) else None

        best_models.append(model_data)

        # Store full data for heatmap
        full_data = {'Model': display_name}
        for col in best_model.index:
            if col.startswith(f'{benchmark}_'):
                clean_col = col.replace(f'{benchmark}_', '')
                value = best_model[col]

                if is_dl_model and any(clean_col.startswith(prefix) for prefix in ['precision_', 'recall_']):
                    if pd.notna(value):
                        value = value * 100

                full_data[clean_col] = value
        best_models_full.append(full_data)

    output_df = pd.DataFrame(best_models).sort_values('Accuracy', ascending=False).reset_index(drop=True)

    return output_df, pd.DataFrame(best_models_full)


def find_fixed_top3_models(df_all, fixed_model_names, base_search_dir):
    """
    Find FIXED Top-3 model paths that will be used for ALL benchmarks.

    Args:
        df_all: Full consolidated DataFrame
        fixed_model_names: List of 3 model prefix names (e.g., ['simpledense_NFT_reh_fld_2', ...])
        base_search_dir: Base directory to search for model files

    Returns:
        list: [(model_path, display_name, model_prefix), ...] for 3 models
    """
    if fixed_model_names is None or len(fixed_model_names) != 3:
        print("ERROR: Must provide exactly 3 fixed model names")
        return []

    top3_info = []

    for model_prefix in fixed_model_names:
        # Find the model file
        model_path = find_model_file(model_prefix, base_search_dir)

        if model_path is None:
            print(f"  WARNING: Model file not found for {model_prefix}")
            continue

        # Get display name from CSV
        matching_rows = df_all[df_all['prefix_name'] == model_prefix]
        if len(matching_rows) == 0:
            print(f"  WARNING: No CSV entry found for {model_prefix}")
            continue

        model_row = matching_rows.iloc[0]
        architecture = extract_model_architecture(model_row['Model'])
        display_name = map_to_display_name(architecture)

        top3_info.append((model_path, display_name, model_prefix))
        print(f"  Found: {model_prefix} -> {display_name}")

    return top3_info


def auto_detect_top3_models(df_all, training_dataset, base_search_dir):
    """
    Auto-detect Top-3 models based on training dataset.

    For REH: Uses SimpleDense, EnhanceDense, FeatureEmbedding (best fold each)
    For others: Uses top 3 DL models by average accuracy across all benchmarks

    CRITICAL: DeepDense is NEVER included.
    """
    print(f"\nAuto-detecting Top-3 models for {training_dataset}...")

    # Filter out DeepDense completely
    df_dl = df_all[df_all['Model'].apply(lambda x: extract_model_architecture(x).lower() not in ['deepdense', 'dd', 'dnn4'])].copy()

    # For REH: Use SimpleDense, EnhanceDense, FeatureEmbedding
    if 'reh' in training_dataset.lower():
        print("  Using REH default: SimpleDense, EnhanceDense, FeatureEmbedding")

        target_archs = ['simpledense', 'enhancedense', 'featureembedding']
        top3_info = []

        for arch in target_archs:
            arch_models = df_dl[df_dl['Model'].apply(lambda x: extract_model_architecture(x).lower() == arch)].copy()

            if len(arch_models) == 0:
                print(f"  WARNING: No models found for {arch}")
                continue

            # Find best fold by average accuracy across all benchmarks
            benchmarks = ['sup', 'gse146773', 'gse64016', 'buettner_mesc']
            arch_models['avg_accuracy'] = arch_models[[f'{b}_accuracy' for b in benchmarks]].mean(axis=1)

            best_idx = arch_models['avg_accuracy'].idxmax()
            best_model = arch_models.loc[best_idx]

            model_prefix = best_model['prefix_name']
            model_path = find_model_file(model_prefix, base_search_dir)

            if model_path:
                display_name = map_to_display_name(arch)
                top3_info.append((model_path, display_name, model_prefix))
                print(f"  Selected: {model_prefix} (avg accuracy: {best_model['avg_accuracy']:.2f}%)")

        return top3_info

    else:
        # For other datasets: Use top 3 DL models by average accuracy
        print("  Using top 3 DL models by average accuracy across all benchmarks")

        # Calculate average accuracy across all benchmarks
        benchmarks = ['sup', 'gse146773', 'gse64016', 'buettner_mesc']
        df_dl['avg_accuracy'] = df_dl[[f'{b}_accuracy' for b in benchmarks]].mean(axis=1)

        # Exclude traditional ML models
        dl_only = df_dl[~df_dl['Model'].apply(lambda x: extract_model_architecture(x).lower() in ['lightgbm', 'lgbm', 'adaboost', 'randomforest', 'random', 'ensemble'])].copy()

        # Sort by average accuracy and get top 3
        top3_models = dl_only.nlargest(3, 'avg_accuracy')

        top3_info = []
        for _, model_row in top3_models.iterrows():
            model_prefix = model_row['prefix_name']
            model_path = find_model_file(model_prefix, base_search_dir)

            if model_path:
                architecture = extract_model_architecture(model_row['Model'])
                display_name = map_to_display_name(architecture)
                top3_info.append((model_path, display_name, model_prefix))
                print(f"  Selected: {model_prefix} (avg accuracy: {model_row['avg_accuracy']:.2f}%)")

        return top3_info


def run_ensemble_fusion(top3_info, benchmark):
    """
    Run ensemble fusion using FIXED Top-3 models.

    Returns metrics for both Decision Fusion and Score Fusion.
    """
    if len(top3_info) != 3:
        print(f"  ERROR: Need exactly 3 models for ensemble, got {len(top3_info)}")
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

        for phase in ['g1', 'g2m', 's']:
            prec_col = f'{benchmark}_precision_{phase}'
            rec_col = f'{benchmark}_recall_{phase}'

            if prec_col in df_result.columns:
                top3df_metrics[f'precision_{phase}'] = round(df_result[prec_col].values[0], 2)
            if rec_col in df_result.columns:
                top3df_metrics[f'recall_{phase}'] = round(df_result[rec_col].values[0], 2)

    except Exception as e:
        print(f"  ERROR in Decision Fusion: {e}")
        import traceback
        traceback.print_exc()
        top3df_metrics = None

    try:
        sf_result = score_level_fusion(model_paths, fusion_benchmark)

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

        for phase in ['g1', 'g2m', 's']:
            prec_col = f'{benchmark}_precision_{phase}'
            rec_col = f'{benchmark}_recall_{phase}'

            if prec_col in sf_result.columns:
                top3sf_metrics[f'precision_{phase}'] = round(sf_result[prec_col].values[0], 2)
            if rec_col in sf_result.columns:
                top3sf_metrics[f'recall_{phase}'] = round(sf_result[rec_col].values[0], 2)

    except Exception as e:
        print(f"  ERROR in Score Fusion: {e}")
        import traceback
        traceback.print_exc()
        top3sf_metrics = None

    return top3df_metrics, top3sf_metrics


def generate_heatmap(all_benchmark_data, output_dir):
    """
    Generate precision/recall heatmap for all benchmarks.

    EXACT COPY from original 3_generate_all_benchmark_results.py
    Creates single large heatmap with Blues colormap and vertical separators.
    """

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
    display_models = [HEATMAP_MODEL_NAMES.get(m, m) for m in models]

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

    # Metric labels (middle)
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

    # Vertical separators between benchmarks
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
    parser = argparse.ArgumentParser(description='Generate all benchmark results + heatmap (FIXED Top-3 models)')
    parser.add_argument('--input', '-i', required=True, help='Path to consolidated CSV file')
    parser.add_argument('--model-dir', '-m', default=None, help='Base directory for model files')
    parser.add_argument('--fixed-models', '-f', nargs=3, default=None,
                        help='Fixed model prefix names (3 required, e.g., simpledense_NFT_reh_fld_2 enhancedense_NFT_reh_fld_3 fe_NFT_reh_fld_5)')
    parser.add_argument('--skip-ensemble', action='store_true', help='Skip ensemble fusion')
    parser.add_argument('--skip-heatmap', action='store_true', help='Skip heatmap generation')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print("="*70)
    print("FIXED COMPREHENSIVE BENCHMARK RESULTS GENERATOR")
    print("="*70)
    print(f"\nInput: {args.input}")
    print(f"Model directory: {args.model_dir}")
    print(f"Fixed models: {args.fixed_models if args.fixed_models else 'Auto-detect'}")
    print(f"Skip ensemble: {args.skip_ensemble}")
    print(f"Skip heatmap: {args.skip_heatmap}")

    # Load consolidated data
    print("\nLoading consolidated data...")
    df_all = pd.read_csv(args.input)
    print(f"  Total models: {len(df_all)}")

    # Extract training dataset name
    training_dataset = extract_training_dataset(df_all['prefix_name'].iloc[0])
    if training_dataset is None:
        print("ERROR: Could not extract training dataset name from prefix_name")
        sys.exit(1)
    print(f"  Detected training dataset: {training_dataset}")

    # Find FIXED Top-3 models (ONCE for ALL benchmarks)
    top3_info = None
    if not args.skip_ensemble:
        if args.fixed_models:
            print("\nUsing user-specified fixed Top-3 models:")
            top3_info = find_fixed_top3_models(df_all, args.fixed_models, args.model_dir)
        else:
            top3_info = auto_detect_top3_models(df_all, training_dataset, args.model_dir)

        if len(top3_info) != 3:
            print(f"\nWARNING: Could not find 3 models, ensemble fusion will be skipped")
            top3_info = None
        else:
            print(f"\nFIXED Top-3 models for ALL benchmarks:")
            for i, (path, name, prefix) in enumerate(top3_info, 1):
                print(f"  {i}. {name} ({prefix})")

    # Process each benchmark
    benchmarks = ['sup', 'gse146773', 'gse64016', 'buettner_mesc']
    all_benchmark_data = {}

    output_dir = f"/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/5_visualization/heatmap_barplot_lineplots_csv/{training_dataset}"
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in benchmarks:
        print(f"\n{'='*70}")
        print(f"PROCESSING: {benchmark.upper()}")
        print(f"{'='*70}")

        best_df, full_data = extract_best_models_for_benchmark(df_all, benchmark, args.model_dir)

        print(f"\nBest models extracted: {len(best_df)}")

        # Ensemble fusion using FIXED Top-3 models
        if top3_info is not None and len(top3_info) == 3:
            print(f"\nRunning ensemble fusion with FIXED Top-3 models...")
            print(f"  Models: {', '.join([t[1] for t in top3_info])}")

            top3df, top3sf = run_ensemble_fusion(top3_info, benchmark)

            if top3df:
                best_df = pd.concat([best_df, pd.DataFrame([top3df])], ignore_index=True)
                full_data = pd.concat([full_data, pd.DataFrame([top3df])], ignore_index=True)
                print(f"  Top-3 DF Accuracy: {top3df['Accuracy']}%")

            if top3sf:
                best_df = pd.concat([best_df, pd.DataFrame([top3sf])], ignore_index=True)
                full_data = pd.concat([full_data, pd.DataFrame([top3sf])], ignore_index=True)
                print(f"  Top-3 SF Accuracy: {top3sf['Accuracy']}%")

        # Sort models by predefined order
        best_df = sort_models_by_order(best_df)

        # Save CSV
        output_file = os.path.join(output_dir, f'{benchmark}_results_{training_dataset}.csv')
        best_df.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")

        # Store for heatmap
        if benchmark != 'sup':
            all_benchmark_data[benchmark] = full_data

    # Generate heatmap
    if not args.skip_heatmap and len(all_benchmark_data) > 0:
        print("\n" + "="*70)
        print("GENERATING PRECISION/RECALL HEATMAP")
        print("="*70)
        generate_heatmap(all_benchmark_data, output_dir)

    # Automatically call line plot script (4_plot_benchmark_results.py)
    print("\n" + "="*70)
    print("GENERATING BENCHMARK LINE PLOTS")
    print("="*70)

    plot_script = os.path.join(os.path.dirname(__file__), '4_plot_benchmark_results.py')
    if os.path.exists(plot_script):
        try:
            print(f"\nCalling: {plot_script}")
            print(f"  Input directory: {output_dir}")
            print(f"  Training dataset: {training_dataset}")

            subprocess.run([
                sys.executable,
                plot_script,
                '--input-dir', output_dir,
                '--training-dataset', training_dataset
            ], check=True)

            print("  Line plots generated successfully!")
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Line plot generation failed: {e}")
        except Exception as e:
            print(f"  WARNING: Could not call line plot script: {e}")
    else:
        print(f"  WARNING: Plot script not found: {plot_script}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nNOTE: Run 5_plot_tool_comparison_barplot.py separately to compare with existing tools:")
    print(f"  python 5_plot_tool_comparison_barplot.py --training-dataset {training_dataset}")


if __name__ == "__main__":
    main()
