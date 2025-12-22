#!/usr/bin/env python3
"""
Merge Reassigned Tool Predictions into Consensus Labels

This script merges reassigned predictions from multiple tools, keeping only cells
where at least N tools agree on the phase assignment.

Usage:
    python merge_reassigned.py --input_dir <path> --output_dir <path> --min_agreement <N>

Example:
    # Require at least 2 tools to agree
    python merge_reassigned.py \\
        --input_dir ./reassigned_predictions \\
        --output_dir ./final_training_data \\
        --min_agreement 2
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from collections import Counter


def load_tool_predictions(dataset_dir):
    """
    Load all tool predictions for a dataset.

    Args:
        dataset_dir: Directory containing reassigned tool predictions

    Returns:
        dict: {tool_name: DataFrame}
    """
    predictions = {}
    dataset_path = Path(dataset_dir)

    for csv_file in dataset_path.glob("*_reassigned.csv"):
        tool_name = csv_file.stem.replace('_reassigned', '')
        df = pd.read_csv(csv_file)

        # Ensure standard column names
        if 'CellID' not in df.columns and 'Cell_ID' in df.columns:
            df = df.rename(columns={'Cell_ID': 'CellID'})
        if 'Predicted' not in df.columns and 'Phase' in df.columns:
            df = df.rename(columns={'Phase': 'Predicted'})

        predictions[tool_name] = df
        print(f"  Loaded {tool_name}: {len(df)} cells")

    return predictions


def merge_predictions(predictions, min_agreement=2):
    """
    Merge predictions from multiple tools based on consensus.

    Args:
        predictions: Dictionary of {tool_name: DataFrame}
        min_agreement: Minimum number of tools that must agree

    Returns:
        DataFrame: Merged consensus predictions
    """
    # Get all unique cell IDs
    all_cells = set()
    for df in predictions.values():
        all_cells.update(df['CellID'].tolist())

    print(f"\n  Total unique cells across all tools: {len(all_cells)}")
    print(f"  Minimum agreement threshold: {min_agreement} tools")

    # Create consensus for each cell
    consensus_data = []

    for cell_id in all_cells:
        # Collect predictions from all tools for this cell
        cell_predictions = []

        for tool_name, df in predictions.items():
            cell_df = df[df['CellID'] == cell_id]
            if len(cell_df) > 0:
                phase = cell_df.iloc[0]['Predicted']
                cell_predictions.append((tool_name, phase))

        # Count votes for each phase
        phase_votes = [pred[1] for pred in cell_predictions]
        vote_counts = Counter(phase_votes)

        # Get the most common phase and its count
        if vote_counts:
            most_common_phase, vote_count = vote_counts.most_common(1)[0]

            # Only include if enough tools agree
            if vote_count >= min_agreement:
                consensus_data.append({
                    'CellID': cell_id,
                    'Predicted': most_common_phase,
                    'n_tools_agree': vote_count,
                    'n_tools_total': len(cell_predictions),
                    'agreeing_tools': ','.join([t for t, p in cell_predictions if p == most_common_phase])
                })

    # Create DataFrame
    consensus_df = pd.DataFrame(consensus_data)

    if len(consensus_df) > 0:
        consensus_df = consensus_df.sort_values('CellID').reset_index(drop=True)

    return consensus_df


def process_dataset(dataset_name, dataset_dir, output_dir, min_agreement):
    """
    Process a single dataset: load predictions and create consensus.

    Args:
        dataset_name: Name of the dataset
        dataset_dir: Directory containing reassigned predictions
        output_dir: Output directory for final consensus data
        min_agreement: Minimum number of tools that must agree
    """
    print(f"\n{'='*80}")
    print(f"MERGING: {dataset_name}")
    print(f"{'='*80}\n")

    # Load all tool predictions
    predictions = load_tool_predictions(dataset_dir)

    if len(predictions) == 0:
        print(f"WARNING: No predictions found in {dataset_dir}")
        return

    print(f"\n  Number of tools: {len(predictions)}")

    # Merge predictions
    consensus_df = merge_predictions(predictions, min_agreement)

    if len(consensus_df) == 0:
        print(f"\nWARNING: No consensus labels created (no cells with ≥{min_agreement} agreement)")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full consensus (with metadata)
    full_output_file = output_path / f"{dataset_name}_consensus_full.csv"
    consensus_df.to_csv(full_output_file, index=False)
    print(f"\n  Full consensus saved: {full_output_file}")

    # Save simplified version (just CellID and Predicted)
    simple_df = consensus_df[['CellID', 'Predicted']]
    simple_output_file = output_path / f"{dataset_name}_consensus.csv"
    simple_df.to_csv(simple_output_file, index=False)
    print(f"  Simplified consensus saved: {simple_output_file}")

    # Print statistics
    print(f"\n  CONSENSUS STATISTICS:")
    print(f"  {'─'*60}")
    print(f"  Total consensus cells: {len(consensus_df)}")

    # Phase distribution
    phase_counts = consensus_df['Predicted'].value_counts().to_dict()
    print(f"\n  Phase distribution:")
    for phase, count in sorted(phase_counts.items()):
        pct = (count / len(consensus_df)) * 100
        print(f"    {phase}: {count} ({pct:.1f}%)")

    # Agreement distribution
    agreement_counts = consensus_df['n_tools_agree'].value_counts().sort_index().to_dict()
    print(f"\n  Agreement distribution:")
    for n_tools, count in sorted(agreement_counts.items()):
        pct = (count / len(consensus_df)) * 100
        print(f"    {n_tools} tools agree: {count} cells ({pct:.1f}%)")

    print(f"  {'─'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge reassigned tool predictions into consensus labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Require at least 2 tools to agree
    python merge_reassigned.py --input_dir ./reassigned_predictions --min_agreement 2

    # Require at least 3 tools to agree (more stringent)
    python merge_reassigned.py --input_dir ./reassigned_predictions --min_agreement 3

    # Specify custom output directory
    python merge_reassigned.py \\
        --input_dir ./reassigned_predictions \\
        --output_dir /path/to/final_data \\
        --min_agreement 2
        """
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing reassigned tool predictions'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./final_training_data',
        help='Output directory for consensus training data (default: ./final_training_data)'
    )

    parser.add_argument(
        '--min_agreement',
        type=int,
        default=2,
        help='Minimum number of tools that must agree (default: 2)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    if args.min_agreement < 2:
        raise ValueError("min_agreement must be at least 2")

    # Process each dataset subdirectory
    input_path = Path(args.input_dir)
    datasets_processed = 0

    for dataset_dir in input_path.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            process_dataset(dataset_name, dataset_dir, args.output_dir, args.min_agreement)
            datasets_processed += 1

    if datasets_processed == 0:
        print(f"\nWARNING: No dataset directories found in {args.input_dir}")
        print("Expected structure: input_dir/dataset_name/*_reassigned.csv")
    else:
        print(f"\n{'='*80}")
        print(f"CONSENSUS MERGING COMPLETE!")
        print(f"{'='*80}\n")
        print(f"Processed {datasets_processed} dataset(s)")
        print(f"Final consensus data saved to: {args.output_dir}")
        print(f"\nNEXT STEP:")
        print(f"  Train models using the consensus CSV files:")
        print(f"  - For mouse: {args.output_dir}/mouse_brain_consensus.csv")
        print(f"  - For human: {args.output_dir}/pbmc_human_consensus.csv")
        print(f"\n  Example:")
        print(f"    python ../../2_model_training/train_deep_learning.py \\")
        print(f"        --model simpledense \\")
        print(f"        --data {args.output_dir}/mouse_brain_consensus.csv \\")
        print(f"        --trials 50 \\")
        print(f"        --cv 5")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
