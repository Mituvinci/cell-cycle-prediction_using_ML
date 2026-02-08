#!/usr/bin/env python3
"""
Add ground truth labels to Revelio predictions using inner join
Handles:
1. Buettner mESC - merge with Seurat file
2. 10k mouse brain - merge with final training data
"""

import pandas as pd
import os

print("="*80)
print("ADDING GROUND TRUTH TO REVELIO PREDICTIONS")
print("="*80)

# ==============================================================================
# 1. BUETTNER mESC - Add ground truth from Seurat file
# ==============================================================================

print("\n1. Processing Buettner mESC Revelio predictions...")

# File paths
buettner_revelio = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/BuettnerESCData/predicted_by_revelio_Buettener.csv"
buettner_seurat = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/BuettnerESCData/predicted_by_seurat_Buettner.csv"
buettner_output = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/BuettnerESCData/predicted_by_revelio_Buettener_with_groundtruth.csv"

# Load files
df_revelio = pd.read_csv(buettner_revelio)
df_seurat = pd.read_csv(buettner_seurat)

print(f"  Loaded Revelio predictions: {len(df_revelio)} cells")
print(f"  Loaded Seurat (with GroundTruth): {len(df_seurat)} cells")

# Inner join on CellID to add GroundTruth
# Keep only cells that Revelio successfully predicted
df_merged = pd.merge(df_revelio, df_seurat[['CellID', 'GroundTruth']], on='CellID', how='inner')

print(f"  After inner join: {len(df_merged)} cells")
print(f"  Cells lost (Revelio couldn't predict): {len(df_revelio) - len(df_merged)}")

# Reorder columns: CellID, Predicted, GroundTruth
df_merged = df_merged[['CellID', 'Predicted', 'GroundTruth']]

# Save
df_merged.to_csv(buettner_output, index=False)
print(f"  Saved: {buettner_output}")

# Show phase distribution
print("\n  Phase distribution in final Buettner Revelio file:")
print(df_merged['Predicted'].value_counts().sort_index())

# ==============================================================================
# 2. 10K MOUSE BRAIN - Add ground truth from final training data
# ==============================================================================

print("\n2. Processing 10k mouse brain Revelio predictions...")

# File paths
mouse_revelio = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/10-k-brain-cells_healthy_mouse/revelio_10-k-brain-cells_healthy_mouse.csv"
mouse_training = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data.csv"
mouse_output = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/10-k-brain-cells_healthy_mouse/revelio_10-k-brain-cells_healthy_mouse_with_groundtruth.csv"

# Load Revelio predictions
df_revelio = pd.read_csv(mouse_revelio)
print(f"  Loaded Revelio predictions: {len(df_revelio)} cells")

# Load training data (only first column and predicted column to save memory)
# First, find column names
with open(mouse_training, 'r') as f:
    header_line = f.readline().strip()
    columns = header_line.split(',')

# Find the Predicted column (could be at the end)
# Read only CellID column (gex_barcode) and Predicted column
print(f"  Loading mouse brain training data (large file, this may take a moment)...")

# Load only relevant columns to save memory
df_training = pd.read_csv(mouse_training, usecols=[columns[0], 'Predicted'] if 'Predicted' in columns else None)

# Rename gex_barcode to CellID for consistency
if 'gex_barcode' in df_training.columns:
    df_training = df_training.rename(columns={'gex_barcode': 'CellID'})
elif df_training.columns[0] != 'CellID':
    df_training = df_training.rename(columns={df_training.columns[0]: 'CellID'})

# Find the phase column (should be named 'Predicted' or similar)
phase_cols = [col for col in df_training.columns if col.lower() in ['predicted', 'phase', 'label']]
if len(phase_cols) == 0:
    # Try to find last column that might be phase
    print(f"  WARNING: Could not find phase column. Using last column: {df_training.columns[-1]}")
    df_training = df_training.rename(columns={df_training.columns[-1]: 'GroundTruth'})
else:
    df_training = df_training.rename(columns={phase_cols[0]: 'GroundTruth'})

print(f"  Loaded training data: {len(df_training)} cells")

# Inner join on CellID
df_merged = pd.merge(df_revelio, df_training[['CellID', 'GroundTruth']], on='CellID', how='inner')

print(f"  After inner join: {len(df_merged)} cells")
print(f"  Cells lost (Revelio couldn't predict): {len(df_revelio) - len(df_merged)}")

# Reorder columns: CellID, Predicted, GroundTruth
df_merged = df_merged[['CellID', 'Predicted', 'GroundTruth']]

# Save
df_merged.to_csv(mouse_output, index=False)
print(f"  Saved: {mouse_output}")

# Show phase distribution
print("\n  Phase distribution in final 10k mouse brain Revelio file:")
print(df_merged['Predicted'].value_counts().sort_index())

print("\n" + "="*80)
print("COMPLETED!")
print("="*80)
print("\nOutput files:")
print(f"  1. {buettner_output}")
print(f"  2. {mouse_output}")
print("\nThese files now have CellID, Predicted, GroundTruth columns")
print("Ready for contingency table and heatmap generation!")
print("="*80)
