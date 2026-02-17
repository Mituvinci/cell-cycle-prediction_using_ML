#!/usr/bin/env python3
"""
Validate Consensus Labeling Approach on Benchmark Datasets

This script:
1. Merges tool predictions for benchmarks (≥3 tools must agree)
2. Compares consensus predictions vs ground truth labels
3. Generates validation table for manuscript

Usage:
    python validate_consensus_on_benchmarks.py --min_agreement 3 --output_dir validation_results
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef,
    classification_report, confusion_matrix
)


def load_benchmark_tool_predictions(benchmark_name, base_dir):
    """
    Load reassigned tool predictions for a benchmark.

    Args:
        benchmark_name: 'gse146773', 'gse64016', or 'buettner_mesc'
        base_dir: Base directory containing tool predictions

    Returns:
        dict: {tool_name: DataFrame}
    """
    predictions = {}

    if benchmark_name == 'gse146773':
        tool_files = {
            'seurat': f"{base_dir}/data/Training_data/Benchmark_data/re_assigned_benchmark146773/cellcyclescore_predict_by_seurat_GSE146773 _modified.csv",
            'tricycle': f"{base_dir}/data/Training_data/Benchmark_data/re_assigned_benchmark146773/tricycle_prediction_by_tricycle_GSE146773_modified.csv",
            'ccafv2': f"{base_dir}/data/Training_data/Benchmark_data/re_assigned_benchmark146773/ccAFV2_GSE146773_cell_cycle_labels_modified.csv",
            'revelio': f"{base_dir}/data/Training_data/Benchmark_data/re_assigned_benchmark146773/revelio_gse146773cell_cycle_assignments_revelio_modified.csv"
        }
        cell_id_col = 'barcodes'  # GSE146773 uses 'barcodes' column
        phase_col = 'paper_phase'  # After reassignment

    elif benchmark_name == 'gse64016':
        tool_files = {
            'seurat': f"{base_dir}/data/Training_data/Benchmark_data/re_assigned_benchmark64016/cellcyclescore_predict_by_seurat_GSE64016_modified.csv",
            'tricycle': f"{base_dir}/data/Training_data/Benchmark_data/re_assigned_benchmark64016/tricycle_prediction_by_tricycle_GSE64016_modified.csv",
            'ccafv2': f"{base_dir}/data/Training_data/Benchmark_data/re_assigned_benchmark64016/ccAFV2_predicted_GSE64016_cell_cycle_labels_modified.csv",
            'revelio': f"{base_dir}/data/Training_data/Benchmark_data/re_assigned_benchmark64016/revelio_gse64016cell_cycle_assignments_revelio_modified.csv"
        }
        cell_id_col = 'barcodes'  # GSE64016 uses 'barcodes'
        phase_col = 'Labeled'  # After reassignment

    elif benchmark_name == 'buettner_mesc':
        tool_files = {
            'seurat': f"{base_dir}/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_buettner/buettner_mESC/seurat_reassigned.csv",
            'tricycle': f"{base_dir}/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_buettner/buettner_mESC/tricycle_reassigned.csv",
            'ccafv2': f"{base_dir}/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_buettner/buettner_mESC/ccafv2_reassigned.csv",
            'revelio': f"{base_dir}/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_buettner/buettner_mESC/revelio_reassigned.csv"
        }
        cell_id_col = 'CellID'  # Buettner uses 'CellID'
        phase_col = 'Predicted'  # Already in Predicted column

    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    print(f"\nLoading tool predictions for {benchmark_name}:")
    for tool_name, file_path in tool_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Standardize column names (only if they don't already exist)
            if 'CellID' not in df.columns and cell_id_col in df.columns:
                df = df.rename(columns={cell_id_col: 'CellID'})
            if 'Predicted' not in df.columns and phase_col in df.columns:
                df = df.rename(columns={phase_col: 'Predicted'})

            # Keep only CellID and Predicted
            if 'CellID' in df.columns and 'Predicted' in df.columns:
                df = df[['CellID', 'Predicted']].copy()
            else:
                print(f"  ERROR: Missing required columns in {tool_name}")
                print(f"  Available columns: {df.columns.tolist()}")
                continue

            predictions[tool_name] = df
            print(f"  {tool_name}: {len(df)} cells")
        else:
            print(f"  WARNING: {tool_name} file not found: {file_path}")

    return predictions


def load_ground_truth(benchmark_name, base_dir):
    """
    Load ground truth labels for a benchmark.

    Args:
        benchmark_name: 'gse146773', 'gse64016', or 'buettner_mesc'
        base_dir: Base directory

    Returns:
        DataFrame with columns: CellID, GroundTruth
    """
    if benchmark_name == 'gse146773':
        gt_file = f"{base_dir}/data/benchmarks_preprocessed/GSE146773/GSE146773_labels.csv"
        cell_id_col = 'barcodes'
        phase_col = 'paper_phase'

    elif benchmark_name == 'gse64016':
        gt_file = f"{base_dir}/data/benchmarks_preprocessed/GSE64016/GSE64016_labels.csv"
        cell_id_col = 'barcodes'
        phase_col = 'Labeled'

    elif benchmark_name == 'buettner_mesc':
        gt_file = f"{base_dir}/data/benchmarks_preprocessed/Buettner_mESC/Buettner_mESC_labels.csv"
        cell_id_col = 'CellID'
        phase_col = 'Predicted'

    print(f"\nLoading ground truth for {benchmark_name}:")
    print(f"  File: {gt_file}")

    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    df = pd.read_csv(gt_file)

    # Standardize column names
    rename_dict = {}
    if cell_id_col in df.columns and 'CellID' not in df.columns:
        rename_dict[cell_id_col] = 'CellID'
    if phase_col in df.columns and 'GroundTruth' not in df.columns:
        rename_dict[phase_col] = 'GroundTruth'

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Verify required columns exist
    if 'CellID' not in df.columns or 'GroundTruth' not in df.columns:
        print(f"  ERROR: Missing required columns")
        print(f"  Available columns: {df.columns.tolist()}")
        raise ValueError(f"Ground truth file missing required columns")

    df = df[['CellID', 'GroundTruth']].copy()
    print(f"  Loaded {len(df)} cells with ground truth")

    return df


def merge_consensus(predictions, min_agreement=3):
    """
    Create consensus labels from multiple tool predictions.

    Args:
        predictions: dict of {tool_name: DataFrame with CellID, Predicted}
        min_agreement: Minimum tools that must agree (default: 3)

    Returns:
        DataFrame with consensus labels
    """
    all_cells = set()
    for df in predictions.values():
        all_cells.update(df['CellID'].tolist())

    print(f"\nCreating consensus (min {min_agreement} tools must agree):")
    print(f"  Total cells across all tools: {len(all_cells)}")

    consensus_data = []

    for cell_id in all_cells:
        # Collect predictions from all tools
        cell_predictions = []

        for tool_name, df in predictions.items():
            cell_df = df[df['CellID'] == cell_id]
            if len(cell_df) > 0:
                phase = cell_df.iloc[0]['Predicted']
                cell_predictions.append((tool_name, phase))

        # Count votes
        phase_votes = [pred[1] for pred in cell_predictions]
        vote_counts = Counter(phase_votes)

        # Get most common phase
        if vote_counts:
            most_common_phase, vote_count = vote_counts.most_common(1)[0]

            # Only include if consensus reached
            if vote_count >= min_agreement:
                consensus_data.append({
                    'CellID': cell_id,
                    'ConsensusPrediction': most_common_phase,
                    'n_tools_agree': vote_count,
                    'n_tools_total': len(cell_predictions),
                    'agreeing_tools': ','.join([t for t, p in cell_predictions if p == most_common_phase])
                })

    consensus_df = pd.DataFrame(consensus_data)

    if len(consensus_df) > 0:
        consensus_df = consensus_df.sort_values('CellID').reset_index(drop=True)

    print(f"  Consensus cells: {len(consensus_df)} ({len(consensus_df)/len(all_cells)*100:.1f}%)")

    # Show agreement distribution
    if len(consensus_df) > 0:
        agreement_dist = consensus_df['n_tools_agree'].value_counts().sort_index()
        print(f"\n  Agreement distribution:")
        for n_tools, count in agreement_dist.items():
            print(f"    {n_tools} tools: {count} cells ({count/len(consensus_df)*100:.1f}%)")

    return consensus_df


def calculate_metrics(consensus_df, ground_truth_df):
    """
    Compare consensus predictions vs ground truth.

    Args:
        consensus_df: DataFrame with CellID, ConsensusPrediction
        ground_truth_df: DataFrame with CellID, GroundTruth

    Returns:
        dict: Performance metrics
    """
    # Merge consensus with ground truth
    merged = consensus_df.merge(ground_truth_df, on='CellID', how='inner')

    if len(merged) == 0:
        print("  WARNING: No overlapping cells between consensus and ground truth!")
        print(f"  Sample consensus Cell IDs: {consensus_df['CellID'].head(3).tolist()}")
        print(f"  Sample ground truth Cell IDs: {ground_truth_df['CellID'].head(3).tolist()}")
        return None, None

    print(f"\n  Cells with both consensus and ground truth: {len(merged)}")

    # Drop any rows with NaN values and convert to string
    merged = merged.dropna(subset=['GroundTruth', 'ConsensusPrediction'])
    merged['GroundTruth'] = merged['GroundTruth'].astype(str)
    merged['ConsensusPrediction'] = merged['ConsensusPrediction'].astype(str)

    if len(merged) == 0:
        print("  WARNING: No valid rows after removing NaN values!")
        return None, None

    print(f"  Valid cells for comparison (after removing NaN): {len(merged)}")

    y_true = merged['GroundTruth']
    y_pred = merged['ConsensusPrediction']

    # Calculate metrics
    metrics = {
        'n_cells_total': len(ground_truth_df),
        'n_cells_consensus': len(consensus_df),
        'n_cells_compared': len(merged),
        'pct_cells_consensus': (len(consensus_df) / len(ground_truth_df)) * 100,
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'f1_weighted': f1_score(y_true, y_pred, average='weighted') * 100,
        'precision_weighted': precision_score(y_true, y_pred, average='weighted') * 100,
        'recall_weighted': recall_score(y_true, y_pred, average='weighted') * 100,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred) * 100,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred)
    }

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for phase in ['G1', 'S', 'G2M']:
        if phase in report:
            metrics[f'{phase}_precision'] = report[phase]['precision'] * 100
            metrics[f'{phase}_recall'] = report[phase]['recall'] * 100
            metrics[f'{phase}_f1'] = report[phase]['f1-score'] * 100

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['G1', 'S', 'G2M'])
    metrics['confusion_matrix'] = cm

    return metrics, merged


def validate_benchmark(benchmark_name, base_dir, min_agreement, output_dir):
    """
    Run full validation pipeline for one benchmark.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING: {benchmark_name.upper()}")
    print(f"{'='*80}")

    # Load tool predictions
    predictions = load_benchmark_tool_predictions(benchmark_name, base_dir)

    if len(predictions) == 0:
        print(f"ERROR: No tool predictions found for {benchmark_name}")
        return None

    # Load ground truth
    ground_truth = load_ground_truth(benchmark_name, base_dir)

    # Create consensus
    consensus_df = merge_consensus(predictions, min_agreement)

    if len(consensus_df) == 0:
        print(f"ERROR: No consensus labels created for {benchmark_name}")
        return None

    # Calculate metrics
    result = calculate_metrics(consensus_df, ground_truth)

    if result is None:
        print(f"ERROR: Could not calculate metrics for {benchmark_name}")
        return None

    metrics, merged_df = result

    # Print results
    print(f"\n  VALIDATION RESULTS:")
    print(f"  {'-'*60}")
    print(f"  Consensus Agreement Rate: {metrics['accuracy']:.1f}%")
    print(f"  F1 Score: {metrics['f1_weighted']:.1f}%")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.1f}%")
    print(f"  Cohen's Kappa: {metrics['kappa']:.3f}")
    print(f"  MCC: {metrics['mcc']:.3f}")

    print(f"\n  Per-class Precision:")
    for phase in ['G1', 'S', 'G2M']:
        if f'{phase}_precision' in metrics:
            print(f"    {phase}: {metrics[f'{phase}_precision']:.1f}%")

    print(f"\n  Confusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")
    print(f"  {'-'*60}")

    # Save outputs
    output_path = Path(output_dir) / benchmark_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Save consensus with ground truth comparison
    merged_df.to_csv(output_path / f"{benchmark_name}_consensus_vs_groundtruth.csv", index=False)

    # Save full consensus (without ground truth)
    consensus_df.to_csv(output_path / f"{benchmark_name}_consensus_full.csv", index=False)

    # Save metrics as CSV
    metrics_df = pd.DataFrame([{k: v for k, v in metrics.items() if k != 'confusion_matrix'}])
    metrics_df.to_csv(output_path / f"{benchmark_name}_metrics.csv", index=False)

    print(f"\n  Saved outputs to: {output_path}/")

    return metrics


def create_summary_table(all_metrics, output_dir):
    """
    Create summary table for manuscript.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY TABLE")
    print(f"{'='*80}\n")

    # Create table
    table_data = []

    for benchmark_name, metrics in all_metrics.items():
        if metrics is not None:
            # Map internal names to display names
            display_names = {
                'gse146773': 'GSE146773',
                'gse64016': 'GSE64016',
                'buettner_mesc': 'Buettner mESC'
            }

            species = 'Human' if benchmark_name in ['gse146773', 'gse64016'] else 'Mouse'

            table_data.append({
                'Benchmark': display_names.get(benchmark_name, benchmark_name),
                'Species': species,
                'Total Cells': metrics['n_cells_total'],
                'Consensus Cells': metrics['n_cells_compared'],
                'Pct with Consensus': f"{(metrics['n_cells_compared']/metrics['n_cells_total']*100):.1f}%",
                'Consensus Accuracy': f"{metrics['accuracy']:.1f}%",
                'F1 Score': f"{metrics['f1_weighted']:.1f}%",
                'Balanced Accuracy': f"{metrics['balanced_accuracy']:.1f}%",
                'Kappa': f"{metrics['kappa']:.3f}",
                'MCC': f"{metrics['mcc']:.3f}"
            })

    table_df = pd.DataFrame(table_data)

    # Print table
    print(table_df.to_string(index=False))

    # Save table
    output_path = Path(output_dir)
    table_df.to_csv(output_path / "validation_summary_table.csv", index=False)

    print(f"\nTable saved to: {output_path}/validation_summary_table.csv")
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate consensus labeling approach on benchmark datasets"
    )

    parser.add_argument(
        '--base_dir',
        type=str,
        default='/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq',
        help='Base project directory'
    )

    parser.add_argument(
        '--benchmarks',
        nargs='+',
        default=['gse146773', 'gse64016', 'buettner_mesc'],
        choices=['gse146773', 'gse64016', 'buettner_mesc'],
        help='Benchmarks to validate'
    )

    parser.add_argument(
        '--min_agreement',
        type=int,
        default=3,
        help='Minimum tools that must agree (default: 3)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./consensus_validation',
        help='Output directory'
    )

    args = parser.parse_args()

    # Validate each benchmark
    all_metrics = {}

    for benchmark_name in args.benchmarks:
        metrics = validate_benchmark(
            benchmark_name,
            args.base_dir,
            args.min_agreement,
            args.output_dir
        )
        all_metrics[benchmark_name] = metrics

    # Create summary table
    create_summary_table(all_metrics, args.output_dir)

    print("\nVALIDATION COMPLETE!")
    print(f"All results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
