#!/usr/bin/env python3
"""
Standardize column names for GSE75748 tool predictions
All tools will have: CellID, Predicted
"""

import pandas as pd
import os

DATA_DIR = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/GSE75748_hPSC"

print("=" * 80)
print("STANDARDIZING COLUMN NAMES FOR GSE75748")
print("=" * 80)

# =============================================================================
# 1. SEURAT: barcode_RNA -> CellID, Phase -> Predicted
# =============================================================================
print("\n1. Processing Seurat...")
seurat = pd.read_csv(f"{DATA_DIR}/seurat_GSE75748_cc_phases.csv")
print(f"   Original columns: {list(seurat.columns)}")

seurat = seurat.rename(columns={
    'barcode_RNA': 'CellID',
    'Phase': 'Predicted'
})
seurat = seurat[['CellID', 'Predicted']]

output_file = f"{DATA_DIR}/seurat_GSE75748_cc_phases.csv"
seurat.to_csv(output_file, index=False)
print(f"   Standardized columns: {list(seurat.columns)}")
print(f"   Saved: {output_file}")
print(f"   Rows: {len(seurat)}")

# =============================================================================
# 2. TRICYCLE: Cell -> CellID, CCStage -> Predicted
# =============================================================================
print("\n2. Processing Tricycle...")
tricycle = pd.read_csv(f"{DATA_DIR}/tricycle_GSE75748.csv")
print(f"   Original columns: {list(tricycle.columns)}")

tricycle = tricycle.rename(columns={
    'Cell': 'CellID',
    'CCStage': 'Predicted'
})
tricycle = tricycle[['CellID', 'Predicted']]

output_file = f"{DATA_DIR}/tricycle_GSE75748.csv"
tricycle.to_csv(output_file, index=False)
print(f"   Standardized columns: {list(tricycle.columns)}")
print(f"   Saved: {output_file}")
print(f"   Rows: {len(tricycle)}")

# =============================================================================
# 3. REVELIO: cellID -> CellID, ccPhase -> Predicted
# =============================================================================
print("\n3. Processing Revelio...")
revelio = pd.read_csv(f"{DATA_DIR}/revelio_GSE75748.csv")
print(f"   Original columns: {list(revelio.columns)}")

revelio = revelio.rename(columns={
    'cellID': 'CellID',
    'ccPhase': 'Predicted'
})
revelio = revelio[['CellID', 'Predicted']]

output_file = f"{DATA_DIR}/revelio_GSE75748.csv"
revelio.to_csv(output_file, index=False)
print(f"   Standardized columns: {list(revelio.columns)}")
print(f"   Saved: {output_file}")
print(f"   Rows: {len(revelio)}")

# =============================================================================
# 4. CCAFV2: first column (unnamed) -> CellID, ccAFv2 -> Predicted
# =============================================================================
print("\n4. Processing ccAFv2...")
ccafv2 = pd.read_csv(f"{DATA_DIR}/ccAFV2_GSE75748.csv")
print(f"   Original columns: {list(ccafv2.columns)}")

# First column is the cell ID (even if unnamed)
ccafv2 = ccafv2.rename(columns={
    ccafv2.columns[0]: 'CellID',
    'ccAFv2': 'Predicted'
})
ccafv2 = ccafv2[['CellID', 'Predicted']]

output_file = f"{DATA_DIR}/ccAFV2_GSE75748.csv"
ccafv2.to_csv(output_file, index=False)
print(f"   Standardized columns: {list(ccafv2.columns)}")
print(f"   Saved: {output_file}")
print(f"   Rows: {len(ccafv2)}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("STANDARDIZATION COMPLETE")
print("=" * 80)
print("\nAll files now have columns: CellID, Predicted")
print("\nPhase distributions:")
print("\nSeurat:")
print(seurat['Predicted'].value_counts())
print("\nTricycle:")
print(tricycle['Predicted'].value_counts())
print("\nRevelio:")
print(revelio['Predicted'].value_counts())
print("\nccAFv2:")
print(ccafv2['Predicted'].value_counts())
print("\n" + "=" * 80)
