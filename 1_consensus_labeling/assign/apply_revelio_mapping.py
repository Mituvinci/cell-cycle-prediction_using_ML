#!/usr/bin/env python3
"""
Apply Phase Mappings to Revelio Predictions
============================================

This script applies YAML phase mappings to Revelio predictions for Buettner mESC
and 10k mouse brain datasets.

Usage:
    python apply_revelio_mapping.py

Author: Halima Akhter
Date: 2025-12-16
"""

import pandas as pd
import yaml
import os

def load_yaml_mapping(yaml_file):
    """Load phase mappings from YAML file."""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config['mappings']['revelio']

def apply_mapping(input_file, output_file, yaml_file, dataset_name):
    """
    Apply phase mapping to Revelio predictions.

    Args:
        input_file: Path to input CSV (with GroundTruth column)
        output_file: Path to output CSV
        yaml_file: Path to YAML mapping file
        dataset_name: Name of dataset for display
    """
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*80}")

    # Load mapping
    print(f"\nLoading mappings from: {yaml_file}")
    mappings = load_yaml_mapping(yaml_file)
    print("Mappings:")
    for old_phase, new_phase in mappings.items():
        print(f"  {old_phase} -> {new_phase}")

    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} cells")

    # Show original phase distribution
    print("\nOriginal Revelio phase distribution:")
    orig_counts = df['Predicted'].value_counts().sort_index()
    for phase, count in orig_counts.items():
        print(f"  {phase}: {count}")

    # Apply mappings
    print("\nApplying phase mappings...")
    df_reassigned = df.copy()

    for old_phase, new_phase in mappings.items():
        if old_phase in df_reassigned['Predicted'].values:
            count = (df_reassigned['Predicted'] == old_phase).sum()
            df_reassigned.loc[df_reassigned['Predicted'] == old_phase, 'Predicted'] = new_phase
            print(f"  {old_phase} -> {new_phase}: {count} cells")

    # Show reassigned phase distribution
    print("\nReassigned phase distribution:")
    new_counts = df_reassigned['Predicted'].value_counts().sort_index()
    for phase, count in new_counts.items():
        print(f"  {phase}: {count}")

    # Save reassigned predictions
    df_reassigned.to_csv(output_file, index=False)
    print(f"\nSaved reassigned predictions to: {output_file}")

    return df_reassigned

def main():
    """Main function."""

    base_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq"

    print("="*80)
    print("REVELIO PHASE MAPPING APPLICATION")
    print("="*80)

    # ===========================================================================
    # 1. BUETTNER mESC
    # ===========================================================================

    # Use ORIGINAL file (not merged with groundtruth) to keep all cells
    buettner_input = f"{base_dir}/data/Training_data/BuettnerESCData/predicted_by_revelio_Buettener.csv"
    buettner_output = f"{base_dir}/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_buettner/buettner_mESC/revelio_reassigned.csv"
    buettner_yaml = f"{base_dir}/cell_cycle_prediction/configs/phase_mappings/buettner_mESC_revelio.yaml"

    # Create output directory
    os.makedirs(os.path.dirname(buettner_output), exist_ok=True)

    # Apply mapping
    df_buettner = apply_mapping(buettner_input, buettner_output, buettner_yaml, "Buettner mESC")

    # ===========================================================================
    # 2. 10K MOUSE BRAIN - Use ORIGINAL file (not merged with groundtruth)
    # ===========================================================================

    mouse_input = f"{base_dir}/data/Training_data/10-k-brain-cells_healthy_mouse/revelio_10-k-brain-cells_healthy_mouse.csv"
    mouse_output = f"{base_dir}/cell_cycle_prediction/1_consensus_labeling/assign/reassigned_predictions_mouse/mouse_brain/revelio_reassigned.csv"
    mouse_yaml = f"{base_dir}/cell_cycle_prediction/configs/phase_mappings/mouse_brain_revelio.yaml"

    # Create output directory
    os.makedirs(os.path.dirname(mouse_output), exist_ok=True)

    # Apply mapping
    df_mouse = apply_mapping(mouse_input, mouse_output, mouse_yaml, "10k Mouse Brain")

    # ===========================================================================
    # SUMMARY
    # ===========================================================================

    print("\n" + "="*80)
    print("SUCCESS! PHASE MAPPINGS APPLIED")
    print("="*80)
    print("\nOutput files:")
    print(f"  1. {buettner_output}")
    print(f"  2. {mouse_output}")
    print("\nThese files are ready to be:")
    print("  - Buettner: Used directly for tool comparison (has experimental ground truth)")
    print("  - Mouse brain: Merged with other tools (Seurat, Tricycle, ccAFv2) for consensus")
    print("="*80)

if __name__ == '__main__':
    main()
