#!/usr/bin/env python3
"""
Create YAML Mapping Template for Manual Phase Assignment

This script reads the observed/expected ratio CSV files from heatmap analysis
and generates a YAML template for manual phase assignment based on visual
heatmap inspection.

Usage:
    python create_mapping_template.py --results_dir <path> --output <yaml_file>

Example:
    python create_mapping_template.py \\
        --results_dir ../results \\
        --output phase_mappings_new_data.yaml
"""

import os
import argparse
import pandas as pd
import yaml
from pathlib import Path


def extract_phases_from_heatmap(csv_file):
    """
    Extract phase labels from observed/expected ratio CSV.

    Args:
        csv_file: Path to obs_expected_ratios CSV file

    Returns:
        tuple: (reference_phases, comparison_phases)
    """
    df = pd.read_csv(csv_file, index_col=0)
    reference_phases = df.index.tolist()  # Seurat phases (rows)
    comparison_phases = df.columns.tolist()  # Other tool phases (columns)
    return reference_phases, comparison_phases


def identify_submappings_needed(comparison_phases, reference_phases):
    """
    Identify which comparison phases need to be mapped to reference phases.

    Args:
        comparison_phases: List of phases from comparison tool
        reference_phases: List of phases from reference tool (Seurat)

    Returns:
        list: Phases that need mapping (not already in reference)
    """
    # Standard 3-class labels
    standard_phases = {'G1', 'S', 'G2M'}

    # Find phases that need mapping
    needs_mapping = []
    for phase in comparison_phases:
        if phase not in standard_phases and phase not in ['NA', 'Unknown']:
            needs_mapping.append(phase)

    return needs_mapping


def create_template(results_dir, output_file):
    """
    Create YAML template for phase mappings.

    Args:
        results_dir: Directory containing consensus analysis results
        output_file: Output YAML file path
    """
    results_path = Path(results_dir)
    template = {}

    # Process mouse data
    mouse_heatmap_dir = results_path / "mouse" / "heatmaps"
    if mouse_heatmap_dir.exists():
        template['mouse_brain'] = {}
        template['mouse_brain']['_description'] = (
            "Manual phase assignment for mouse brain 10k cells dataset. "
            "Based on heatmap color similarity analysis."
        )
        template['mouse_brain']['reference_tool'] = 'seurat'

        # Find all obs_expected_ratios CSV files
        for csv_file in mouse_heatmap_dir.glob("obs_expected_ratios_*.csv"):
            filename = csv_file.name
            # Extract comparison tool name
            # Format: obs_expected_ratios_seurat_vs_TOOL_mouse_brain.csv
            parts = filename.replace('obs_expected_ratios_', '').replace('.csv', '').split('_')
            tool_idx = parts.index('vs') + 1
            tool_name = parts[tool_idx]

            ref_phases, comp_phases = extract_phases_from_heatmap(csv_file)
            needs_mapping = identify_submappings_needed(comp_phases, ref_phases)

            if needs_mapping:
                template['mouse_brain'][tool_name] = {}
                template['mouse_brain'][tool_name]['_heatmap'] = str(csv_file.name)
                for phase in needs_mapping:
                    template['mouse_brain'][tool_name][phase] = "???"  # Placeholder

                # Add comment about available target phases
                template['mouse_brain'][tool_name]['_available_targets'] = "G1, S, G2M"

    # Process human data
    human_heatmap_dir = results_path / "human" / "heatmaps"
    if human_heatmap_dir.exists():
        template['pbmc_human'] = {}
        template['pbmc_human']['_description'] = (
            "Manual phase assignment for human PBMC dataset. "
            "Based on heatmap color similarity analysis."
        )
        template['pbmc_human']['reference_tool'] = 'seurat'

        for csv_file in human_heatmap_dir.glob("obs_expected_ratios_*.csv"):
            filename = csv_file.name
            parts = filename.replace('obs_expected_ratios_', '').replace('.csv', '').split('_')
            tool_idx = parts.index('vs') + 1
            tool_name = parts[tool_idx]

            ref_phases, comp_phases = extract_phases_from_heatmap(csv_file)
            needs_mapping = identify_submappings_needed(comp_phases, ref_phases)

            if needs_mapping:
                template['pbmc_human'][tool_name] = {}
                template['pbmc_human'][tool_name]['_heatmap'] = str(csv_file.name)
                for phase in needs_mapping:
                    template['pbmc_human'][tool_name][phase] = "???"

                template['pbmc_human'][tool_name]['_available_targets'] = "G1, S, G2M"

    # Add instructions at the top
    final_template = {
        '_INSTRUCTIONS': {
            'step1': 'Examine the heatmaps in results/mouse/heatmaps/ and results/human/heatmaps/',
            'step2': 'For each sub-phase (e.g., G1.S, G2.M), look at the color intensity in the heatmap',
            'step3': 'Assign the sub-phase to the main phase (G1, S, or G2M) with the closest color',
            'step4': 'Replace "???" with your assignment (G1, S, or G2M)',
            'step5': 'Save the file and run apply_phase_mapping.py',
            'example': {
                'G1.S': 'G1  # if G1.S heatmap color is closer to G1 column',
                'G2.M': 'G2M  # if G2.M heatmap color is closer to G2M column',
                'M.G1': 'G2M  # if M.G1 heatmap color is closer to G2M column'
            }
        }
    }
    final_template.update(template)

    # Write YAML
    with open(output_file, 'w') as f:
        yaml.dump(final_template, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\n{'='*80}")
    print(f"YAML TEMPLATE CREATED: {output_file}")
    print(f"{'='*80}\n")
    print("NEXT STEPS:")
    print(f"1. Open the heatmap PNG files in: {results_dir}/mouse/heatmaps/")
    print(f"                                   {results_dir}/human/heatmaps/")
    print(f"2. Edit {output_file} and replace '???' with your mappings")
    print(f"   based on heatmap color similarity")
    print(f"3. Run: python apply_phase_mapping.py --mapping {output_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create YAML template for manual phase assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create template from default results directory
    python create_mapping_template.py --results_dir ../results --output mappings.yaml

    # Specify custom paths
    python create_mapping_template.py \\
        --results_dir /path/to/results \\
        --output /path/to/phase_mappings_new.yaml
        """
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='../results',
        help='Directory containing consensus analysis results (default: ../results)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='phase_mappings_new_data.yaml',
        help='Output YAML file for phase mappings (default: phase_mappings_new_data.yaml)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.results_dir):
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    # Create template
    create_template(args.results_dir, args.output)


if __name__ == "__main__":
    main()
