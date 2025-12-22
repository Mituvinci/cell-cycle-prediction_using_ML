#!/usr/bin/env python3
"""
Apply Phase Mappings to Tool Predictions

This script applies the manual phase mappings (from YAML file) to the original
tool prediction CSV files, creating reassigned versions ready for merging.

Usage:
    python apply_phase_mapping.py --mapping <yaml_file> --data_dir <path> --output_dir <path>

Example:
    python apply_phase_mapping.py \\
        --mapping phase_mappings_new_data.yaml \\
        --data_dir /path/to/Training_data \\
        --output_dir ./reassigned_predictions
"""

import os
import argparse
import pandas as pd
import yaml
from pathlib import Path


def load_mapping_config(yaml_file):
    """
    Load phase mapping configuration from YAML file.

    Args:
        yaml_file: Path to YAML mapping file

    Returns:
        dict: Mapping configuration
    """
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Remove instruction section
    if '_INSTRUCTIONS' in config:
        del config['_INSTRUCTIONS']

    return config


def apply_mapping(df, mappings, tool_name, dataset_name):
    """
    Apply phase mappings to a dataframe.

    Args:
        df: DataFrame with columns [CellID, Predicted]
        mappings: Dictionary of phase mappings
        tool_name: Name of the tool
        dataset_name: Name of the dataset

    Returns:
        DataFrame: DataFrame with reassigned phases
    """
    df_reassigned = df.copy()

    # Get mappings for this specific tool (if exists)
    if tool_name in mappings:
        tool_mappings = {k: v for k, v in mappings[tool_name].items()
                        if not k.startswith('_')}  # Skip metadata keys

        # Apply mappings
        mapping_count = 0
        for old_phase, new_phase in tool_mappings.items():
            if old_phase in df_reassigned['Predicted'].values:
                count = (df_reassigned['Predicted'] == old_phase).sum()
                df_reassigned.loc[df_reassigned['Predicted'] == old_phase, 'Predicted'] = new_phase
                mapping_count += count
                print(f"  {old_phase} -> {new_phase}: {count} cells")

        if mapping_count > 0:
            print(f"  Total reassigned: {mapping_count} cells")
        else:
            print(f"  No mappings applied (phases already standard)")
    else:
        print(f"  No mappings defined for {tool_name} (assuming standard 3-class output)")

    return df_reassigned


def process_dataset(dataset_config, dataset_name, data_dir, output_dir):
    """
    Process all tool predictions for a dataset.

    Args:
        dataset_config: Configuration for this dataset
        dataset_name: Name of the dataset (e.g., 'mouse_brain')
        data_dir: Root data directory
        output_dir: Output directory for reassigned predictions
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {dataset_name}")
    print(f"{'='*80}\n")

    # Determine dataset path based on name
    if 'buettner' in dataset_name.lower():
        dataset_path = Path(data_dir) / "BuettnerESCData"
        sample_name = "Buettner"
    elif 'mouse' in dataset_name.lower():
        dataset_path = Path(data_dir) / "10-k-brain-cells_healthy_mouse"
        sample_name = "10-k-brain-cells_healthy_mouse"
    elif 'human' in dataset_name.lower() or 'pbmc' in dataset_name.lower():
        dataset_path = Path(data_dir) / "pbmc_healthy_human"
        sample_name = "pbmc_healthy_human"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if not dataset_path.exists():
        print(f"WARNING: Dataset directory not found: {dataset_path}")
        print(f"Skipping {dataset_name}...")
        return

    # Create output directory
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Tool file mapping
    if 'buettner' in dataset_name.lower() or 'mouse' in dataset_name.lower():
        # Mouse data (including Buettner mESC)
        if 'buettner' in dataset_name.lower():
            # Buettner has different file naming
            tool_files = {
                'seurat': f"predicted_by_seurat_{sample_name}.csv",
                'tricycle': f"predicted_by_tricycle_{sample_name}.csv",
                'ccafv2': f"predicted_by_ccAFv2_{sample_name}.csv",
            }
        else:
            # Mouse brain standard naming
            tool_files = {
                'seurat': f"trainingdata_cellcylescore_{sample_name}_cc_phases.csv",
                'tricycle': f"tricycle_mouse_brain_10k.csv",
                'ccafv2': f"ccAFV2_{sample_name}.csv",
            }
    else:
        # Human data
        tool_files = {
            'seurat': f"trainingdata_cellcylescore_{sample_name}_cc_phases.csv",
            'tricycle': f"tricycle_{sample_name}.csv",
            'ccafv2': f"ccAFV2_{sample_name}.csv",
            'revelio': f"revelio_{sample_name}.csv"  # Only for human
        }

    # Process each tool
    for tool, filename in tool_files.items():
        file_path = dataset_path / filename

        # Skip Revelio for mouse (doesn't work on mouse)
        if tool == 'revelio' and ('mouse' in dataset_name.lower() or 'buettner' in dataset_name.lower()):
            print(f"\nSkipping Revelio (not applicable for mouse)")
            continue

        if not file_path.exists():
            print(f"\nWARNING: Tool prediction file not found: {file_path}")
            continue

        print(f"\nProcessing: {tool.upper()}")
        print(f"  Input: {filename}")

        # Load predictions
        df = pd.read_csv(file_path)

        # Ensure standard column names
        if 'CellID' not in df.columns and 'Cell_ID' in df.columns:
            df = df.rename(columns={'Cell_ID': 'CellID'})
        if 'Predicted' not in df.columns and 'Phase' in df.columns:
            df = df.rename(columns={'Phase': 'Predicted'})

        # Apply mappings
        df_reassigned = apply_mapping(df, dataset_config, tool, dataset_name)

        # Save reassigned predictions
        output_file = output_path / f"{tool}_reassigned.csv"
        df_reassigned.to_csv(output_file, index=False)
        print(f"  Output: {output_file}")

        # Show phase distribution
        phase_counts = df_reassigned['Predicted'].value_counts().to_dict()
        print(f"  Phase distribution: {phase_counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply manual phase mappings to tool predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Apply mappings with default paths
    python apply_phase_mapping.py --mapping phase_mappings_new_data.yaml

    # Specify custom paths
    python apply_phase_mapping.py \\
        --mapping my_mappings.yaml \\
        --data_dir /path/to/Training_data \\
        --output_dir /path/to/output
        """
    )

    parser.add_argument(
        '--mapping',
        type=str,
        required=True,
        help='YAML file containing phase mappings'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data',
        help='Root directory containing training data'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./reassigned_predictions',
        help='Output directory for reassigned predictions'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.mapping):
        raise FileNotFoundError(f"Mapping file not found: {args.mapping}")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Load mapping configuration
    print(f"\nLoading mappings from: {args.mapping}")
    config = load_mapping_config(args.mapping)

    # Process each dataset
    for dataset_name, dataset_config in config.items():
        if dataset_name.startswith('_'):  # Skip metadata
            continue
        process_dataset(dataset_config, dataset_name, args.data_dir, args.output_dir)

    print(f"\n{'='*80}")
    print(f"PHASE REASSIGNMENT COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Reassigned predictions saved to: {args.output_dir}")
    print(f"\nNEXT STEP:")
    print(f"  Run: python merge_reassigned.py --input_dir {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
