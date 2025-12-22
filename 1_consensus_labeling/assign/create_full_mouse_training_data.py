#!/usr/bin/env python3
"""
Merge Mouse Brain Consensus Labels with Gene Expression
========================================================
Creates full training data with all 5,524 consensus cells.
"""

import pandas as pd

BASE = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq"

# Load consensus labels (5,524 cells)
consensus = pd.read_csv(f"{BASE}/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_consensus.csv")
print(f"Consensus labels: {len(consensus)} cells")
print(f"  G1={len(consensus[consensus['Predicted']=='G1'])}, S={len(consensus[consensus['Predicted']=='S'])}, G2M={len(consensus[consensus['Predicted']=='G2M'])}")

# Load original gene expression (11,843 cells)
gene_expr = pd.read_csv(f"{BASE}/data/Training_data/10-k-brain-cells_healthy_mouse/trainingdata_cellcylescore_10-k-brain-cells_healthy_mouse_normalized_gene_expression.csv", index_col=0)
print(f"\nOriginal gene expression: {len(gene_expr)} cells, {len(gene_expr.columns)} genes")

# Merge on CellID
gene_expr['CellID'] = gene_expr.index
merged = consensus.merge(gene_expr, on='CellID', how='inner')
print(f"\nMerged: {len(merged)} cells")

# Save
output = f"{BASE}/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_full_consensus_with_genes.csv"
merged.to_csv(output, index=False)
print(f"\nSaved: {output}")
print(f"  Columns: CellID, Predicted, + {len(merged.columns)-2} genes")
