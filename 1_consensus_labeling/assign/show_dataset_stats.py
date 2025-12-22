#!/usr/bin/env python3
"""
Display dataset statistics in a formatted table and save as CSV
"""
import pandas as pd
import os

DATA_DIR = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data"
OUTPUT_CSV = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/5_visualization/plot_table1.csv"

# Dataset info: [name, species, file_path, before_consensus, is_training]
# is_training = True means has consensus labeling (show both before/after)
# is_training = False means benchmark (show before count, NA for after)
datasets = [
    # Training Datasets (with consensus labeling)
    ["REH (Training Set 1)", "Human", f"{DATA_DIR}/filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv", 7433, True],
    ["PBMC Human (Training Set 2)", "Human", "final_training_data_human/pbmc_human_consensus.csv", 7840, True],
    ["GSE75748 hPSC", "Human", f"{DATA_DIR}/GSE75748_hPSC_final_training_matrix.csv", 1776, True],
    ["Mouse Brain (Training Set 2)", "Mouse", "final_training_data_mouse/mouse_brain_consensus.csv", 10014, True],

    # Benchmark Datasets (NO consensus labeling)
    ["SUP (Benchmark Set 1)", "Human", f"{DATA_DIR}/filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv", 7728, False],
    ["GSE146773 (Benchmark Set 2)", "Human", f"{DATA_DIR}/GSE146773_seurat_normalized_gene_expression.csv", 1067, False],
    ["GSE64016 (Benchmark Set 3)", "Human", f"{DATA_DIR}/GSE64016_seurat_normalized_gene_expression.csv", 247, False],
    ["Buettner mESC (Benchmark Set 4)", "Mouse", f"{DATA_DIR}/Buettner_mESC_goundTruth.csv", 288, False],
]

def get_phase_counts(file_path):
    """Read CSV and return total cells and phase counts"""
    try:
        df = pd.read_csv(file_path)
        total = len(df)

        # Check for phase column
        phase_col = None
        possible_names = ['Predicted', 'Phase', 'Label', 'Labeled', 'predicted', 'phase', 'label', 'labeled', 'paper_phase']

        for col_name in possible_names:
            if col_name in df.columns:
                phase_col = col_name
                break

        if phase_col is None:
            print(f"Error: No phase column found in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return 0, 0, 0, 0

        phases = df[phase_col].value_counts().to_dict()
        g1 = phases.get('G1', 0)
        s = phases.get('S', 0)
        g2m = phases.get('G2M', 0)
        return total, g1, s, g2m
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0, 0, 0

def is_imbalanced(total, g1, s, g2m, threshold=0.40):
    """Check if any phase is >40% of total (imbalanced)"""
    if total == 0:
        return False
    return (g1/total > threshold) or (s/total > threshold) or (g2m/total > threshold)

# Collect data for CSV
csv_data = []

# Print header
print("\nDataset                          Species    Before Consensus    After Consensus    G1        S        G2M      Imbalanced")
print("-" * 120)

# Process each dataset
for dataset_name, species, file_path, before_count, is_training in datasets:
    total, g1, s, g2m = get_phase_counts(file_path)

    # For training: show both before and after
    # For benchmark: show before, NA for after
    if is_training:
        before_str = f"{before_count:,}"
        after_str = f"{total:,}"
    else:
        before_str = f"{before_count:,}"
        after_str = "NA"

    # Check if imbalanced
    imbalanced = is_imbalanced(total, g1, s, g2m)
    imbalanced_mark = "*" if imbalanced else ""

    # Print to console
    print(f"{dataset_name:32} {species:10} {before_str:>19} {after_str:>18} {g1:>9,} {s:>8,} {g2m:>8,}     {imbalanced_mark}")

    # Add to CSV data
    csv_data.append({
        'Dataset': dataset_name,
        'Species': species,
        'Before_Consensus_Labeling': before_count,
        'After_Consensus_Labeling': total if is_training else None,
        'G1': g1,
        'S': s,
        'G2M': g2m,
        'Imbalanced': imbalanced_mark
    })

print()

# Save to CSV
df_output = pd.DataFrame(csv_data)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_output.to_csv(OUTPUT_CSV, index=False)
print(f"Table saved to: {OUTPUT_CSV}")
print()
