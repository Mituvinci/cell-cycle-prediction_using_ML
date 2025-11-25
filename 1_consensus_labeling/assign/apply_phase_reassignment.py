"""
Apply Phase Reassignment Using YAML Config
===========================================

This script applies phase mappings from YAML config files to tool predictions,
converting sub-phases to the 3 canonical phases (G1, S, G2M).

Usage:
    python apply_phase_reassignment.py --config configs/phase_mappings/training_data.yaml --tool tricycle --dataset reh

Author: Halima Akhter
Date: 2025-11-24
"""

import os
import argparse
import pandas as pd
import yaml
from pathlib import Path


def load_phase_mapping_config(config_path):
    """
    Load phase mapping configuration from YAML file.

    Parameters:
    -----------
    config_path : str
        Path to YAML config file

    Returns:
    --------
    dict
        Configuration dictionary with mappings for each tool
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_phase_mapping(df, tool_name, mappings, pred_col='Predicted'):
    """
    Apply phase mapping to a dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with predictions
    tool_name : str
        Name of the tool
    mappings : dict
        Phase mapping dictionary
    pred_col : str
        Column name containing predictions

    Returns:
    --------
    pd.DataFrame
        DataFrame with reassigned phases
    """
    # Get tool-specific mappings
    if tool_name not in mappings:
        raise ValueError(f"No mappings found for tool: {tool_name}")

    tool_mappings = mappings[tool_name]

    # Apply mappings
    df_copy = df.copy()
    df_copy[pred_col] = df_copy[pred_col].replace(tool_mappings)

    # Count changes
    n_changed = (df[pred_col] != df_copy[pred_col]).sum()
    n_unchanged = (df[pred_col] == df_copy[pred_col]).sum()

    print(f"  ✓ Applied {tool_name} mappings:")
    print(f"    - Changed: {n_changed} cells")
    print(f"    - Unchanged: {n_unchanged} cells")

    # Show phase distribution before and after
    print(f"\n  Phase distribution:")
    print(f"    Before: {df[pred_col].value_counts().to_dict()}")
    print(f"    After:  {df_copy[pred_col].value_counts().to_dict()}")

    return df_copy


def main():
    parser = argparse.ArgumentParser(
        description='Apply phase reassignment using YAML config mappings'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to phase mapping YAML config file'
    )
    parser.add_argument(
        '--tool',
        type=str,
        required=True,
        choices=['seurat', 'tricycle', 'revelio', 'ccafv2'],
        help='Tool name to apply mappings for'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., reh, sup, gse146773)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../data/predictions',
        help='Directory containing original prediction files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/predictions/reassigned',
        help='Output directory for reassigned predictions'
    )
    parser.add_argument(
        '--pred-col',
        type=str,
        default='Predicted',
        help='Column name for predictions'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Applying Phase Reassignment")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Tool: {args.tool}")
    print(f"Dataset: {args.dataset}")
    print()

    # Load config
    print(f"Loading phase mapping config...")
    config = load_phase_mapping_config(args.config)

    # Validate config
    if 'mappings' not in config:
        raise ValueError("Config file must contain 'mappings' section")

    mappings = config['mappings']
    print(f"  ✓ Loaded mappings for {len(mappings)} tools")

    # Define tool file patterns
    tool_file_patterns = {
        'seurat': 'cellcyclescore_',
        'tricycle': 'tricycle_',
        'revelio': 'revelio_',
        'ccafv2': 'ccAFV2_'
    }

    # Find input file
    pattern = tool_file_patterns[args.tool]
    input_files = list(Path(args.input_dir).glob(f"{pattern}*{args.dataset}*.csv"))

    if not input_files:
        raise FileNotFoundError(
            f"No {args.tool} prediction files found for {args.dataset} in {args.input_dir}"
        )

    input_file = input_files[0]
    print(f"\nLoading predictions from: {input_file}")

    # Load predictions
    df = pd.read_csv(input_file, index_col=0)
    print(f"  Loaded {len(df)} cells")

    # Handle NA values
    if args.pred_col in df.columns:
        df[args.pred_col] = df[args.pred_col].fillna('NA')

    # Apply phase mappings
    print(f"\nApplying phase reassignment...")
    df_reassigned = apply_phase_mapping(df, args.tool, mappings, args.pred_col)

    # Save reassigned predictions
    output_filename = f"{pattern}{args.dataset}_reassigned.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    df_reassigned.to_csv(output_path)

    print(f"\n✓ Saved reassigned predictions to: {output_path}")

    # Validate output
    unique_phases = df_reassigned[args.pred_col].unique()
    expected_phases = {'G1', 'S', 'G2M'}

    unexpected_phases = set(unique_phases) - expected_phases
    if unexpected_phases:
        print(f"\n⚠️  Warning: Found unexpected phases: {unexpected_phases}")
        print("   Expected only: G1, S, G2M")
        print("   Check your YAML config file!")
    else:
        print(f"\n✅ Validation passed: All phases are in {expected_phases}")

    print("\n" + "=" * 80)
    print("✅ Phase reassignment completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
