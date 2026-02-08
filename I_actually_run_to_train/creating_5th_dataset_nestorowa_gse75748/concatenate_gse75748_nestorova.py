#!/usr/bin/env python3
"""
Concatenate GSE75748 (human) + Nestorova (mouse) training datasets
Finds common genes (intersection) and creates final training file
"""

import pandas as pd
import numpy as np
import os

print("=" * 80)
print("CONCATENATING: GSE75748 (human) + Nestorova (mouse)")
print("=" * 80)

# Input files
GSE75748_FILE = "../../1_consensus_labeling/assign/final_training_data_gse75748/gse75748_training_data.csv"
NESTOROVA_FILE = "../../1_consensus_labeling/assign/final_training_data_nestorova/nestorova_training_data.csv"
OUTPUT_DIR = "../../1_consensus_labeling/assign/final_training_data_GSE75748_Nestorova"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1: Load datasets
# =============================================================================
print("\nSTEP 1: Loading datasets...")
print("-" * 80)

gse75748 = pd.read_csv(GSE75748_FILE)
print(f"GSE75748: {len(gse75748)} cells, {len(gse75748.columns)} columns")
print(f"  Phase distribution: {gse75748['Predicted'].value_counts().to_dict()}")

nestorova = pd.read_csv(NESTOROVA_FILE)
print(f"Nestorova: {len(nestorova)} cells, {len(nestorova.columns)} columns")
print(f"  Phase distribution: {nestorova['Predicted'].value_counts().to_dict()}")

# =============================================================================
# STEP 2: Find common genes
# =============================================================================
print("\nSTEP 2: Finding common genes...")
print("-" * 80)

# Get gene columns (exclude metadata)
metadata_cols = ['gex_barcode', 'Predicted', 'CellID', 'Cell_ID']

gse75748_genes = [col for col in gse75748.columns if col not in metadata_cols]
nestorova_genes = [col for col in nestorova.columns if col not in metadata_cols]

print(f"GSE75748 genes: {len(gse75748_genes)}")
print(f"Nestorova genes: {len(nestorova_genes)}")

# Find intersection (all genes are already UPPERCASE)
common_genes = set(gse75748_genes) & set(nestorova_genes)
common_genes = sorted(list(common_genes))

print(f"\nCommon genes (intersection): {len(common_genes)}")
print(f"  GSE75748 unique: {len(set(gse75748_genes) - set(nestorova_genes))}")
print(f"  Nestorova unique: {len(set(nestorova_genes) - set(gse75748_genes))}")

# =============================================================================
# STEP 3: Subset to common genes
# =============================================================================
print("\nSTEP 3: Subsetting to common genes...")
print("-" * 80)

# Keep: gex_barcode + common_genes + Predicted
cols_to_keep = ['gex_barcode'] + common_genes + ['Predicted']

gse75748_subset = gse75748[cols_to_keep].copy()
nestorova_subset = nestorova[cols_to_keep].copy()

print(f"GSE75748 subset: {gse75748_subset.shape}")
print(f"Nestorova subset: {nestorova_subset.shape}")

# =============================================================================
# STEP 4: Add dataset identifier
# =============================================================================
print("\nSTEP 4: Adding dataset identifiers...")
print("-" * 80)

gse75748_subset['Dataset'] = 'GSE75748'
nestorova_subset['Dataset'] = 'Nestorova'

# =============================================================================
# STEP 5: Concatenate
# =============================================================================
print("\nSTEP 5: Concatenating datasets...")
print("-" * 80)

concatenated = pd.concat([gse75748_subset, nestorova_subset], axis=0, ignore_index=True)

print(f"Concatenated: {concatenated.shape}")
print(f"  Total cells: {len(concatenated)}")
print(f"  Total genes: {len(common_genes)}")
print(f"\nPhase distribution:")
print(concatenated.groupby(['Dataset', 'Predicted']).size().unstack(fill_value=0))

# =============================================================================
# STEP 6: Save outputs
# =============================================================================
print("\nSTEP 6: Saving outputs...")
print("-" * 80)

# Save concatenated data
output_file = os.path.join(OUTPUT_DIR, "concatenated_training_data.csv")
concatenated.to_csv(output_file, index=False)
print(f"Saved: {output_file}")

# Save gene list
gene_list_file = os.path.join(OUTPUT_DIR, "gene_list.txt")
with open(gene_list_file, 'w') as f:
    for gene in common_genes:
        f.write(f"{gene}\n")
print(f"Saved: {gene_list_file}")

# Save report
report_file = os.path.join(OUTPUT_DIR, "concatenation_report.txt")
with open(report_file, 'w') as f:
    f.write("GSE75748 + Nestorova Concatenation Report\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Input 1: GSE75748 (human hPSC)\n")
    f.write(f"  Cells: {len(gse75748)}\n")
    f.write(f"  Original genes: {len(gse75748_genes)}\n\n")
    f.write(f"Input 2: Nestorova (mouse)\n")
    f.write(f"  Cells: {len(nestorova)}\n")
    f.write(f"  Original genes: {len(nestorova_genes)}\n\n")
    f.write(f"Output:\n")
    f.write(f"  Total cells: {len(concatenated)}\n")
    f.write(f"  Common genes: {len(common_genes)}\n")
    f.write(f"  Columns: gex_barcode + {len(common_genes)} genes + Predicted + Dataset\n\n")
    f.write(f"Phase distribution by dataset:\n")
    f.write(concatenated.groupby(['Dataset', 'Predicted']).size().unstack(fill_value=0).to_string())
    f.write("\n")

print(f"Saved: {report_file}")

print("\n" + "=" * 80)
print("CONCATENATION COMPLETE!")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Files created:")
print(f"  1. concatenated_training_data.csv ({len(concatenated)} cells × {len(concatenated.columns)} columns)")
print(f"  2. gene_list.txt ({len(common_genes)} genes)")
print(f"  3. concatenation_report.txt")
print("=" * 80)
