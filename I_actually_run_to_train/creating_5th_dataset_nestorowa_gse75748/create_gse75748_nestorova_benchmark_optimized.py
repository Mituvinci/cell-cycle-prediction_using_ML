#!/usr/bin/env python3
"""
Create GSE75748 + Nestorova dataset with genes that overlap with benchmarks
Gene intersection: GSE75748 ∩ Nestorova ∩ GSE146773 ∩ GSE64016 ∩ Buettner_mESC
"""

import pandas as pd
import numpy as np

print("="*80)
print("Creating GSE75748 + Nestorova dataset (benchmark-optimized)")
print("="*80)

# Paths
gse75748_path = "../1_consensus_labeling/assign/final_training_data_hpsc/GSE75748_hPSC_expression_with_consensus_labels.csv"
nestorova_path = "../1_consensus_labeling/assign/final_training_data_nestorova/nestorova_expression_with_consensus_labels.csv"

benchmark_paths = {
    'GSE146773': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/GSE146773_seurat_normalized_gene_expression.csv",
    'GSE64016': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/GSE64016_seurat_normalized_gene_expression.csv",
    'Buettner': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/Buettner_mESC_SeuratNormalized_ML_ready.csv"
}

# Load training data
print("\nLoading training datasets...")
gse75748 = pd.read_csv(gse75748_path)
nestorova = pd.read_csv(nestorova_path)

# Extract expression (remove phase column)
gse75748_expr = gse75748.drop(columns=['phase'] if 'phase' in gse75748.columns else [])
nestorova_expr = nestorova.drop(columns=['phase'] if 'phase' in nestorova.columns else [])

# Convert to UPPERCASE
gse75748_expr.columns = [str(c).upper() for c in gse75748_expr.columns]
nestorova_expr.columns = [str(c).upper() for c in nestorova_expr.columns]

print(f"  GSE75748: {gse75748_expr.shape[0]} cells, {gse75748_expr.shape[1]} genes")
print(f"  Nestorova: {nestorova_expr.shape[0]} cells, {nestorova_expr.shape[1]} genes")

# Load benchmarks (just to get gene names)
print("\nLoading benchmark datasets (for gene intersection)...")
benchmark_genes = []
for name, path in benchmark_paths.items():
    df = pd.read_csv(path)
    # Drop non-gene columns
    drop_cols = ['cell', 'paper_phase', 'Unnamed: 0']
    gene_cols = [c for c in df.columns if c not in drop_cols]
    genes_upper = [str(g).upper() for g in gene_cols]
    benchmark_genes.append(set(genes_upper))
    print(f"  {name}: {len(genes_upper)} genes")

# Find common genes across ALL datasets
print("\nFinding common genes...")
all_gene_sets = [
    set(gse75748_expr.columns),
    set(nestorova_expr.columns),
    benchmark_genes[0],
    benchmark_genes[1],
    benchmark_genes[2]
]

common_genes = set.intersection(*all_gene_sets)
common_genes = sorted(list(common_genes))

print(f"  Common genes across all 5 datasets: {len(common_genes)}")

# Filter to common genes
gse75748_filtered = gse75748_expr[common_genes].copy()
nestorova_filtered = nestorova_expr[common_genes].copy()

# Add phase labels back
gse75748_filtered.insert(0, 'phase', gse75748['phase'].values)
nestorova_filtered.insert(0, 'phase', nestorova['phase'].values)

# Concatenate
print("\nConcatenating datasets...")
concatenated = pd.concat([gse75748_filtered, nestorova_filtered], axis=0, ignore_index=True)

print(f"  Final shape: {concatenated.shape}")
print(f"  Phase distribution: {concatenated['phase'].value_counts().to_dict()}")

# Save
output_dir = "../1_consensus_labeling/assign/final_training_data_GSE75748_Nestorova"
import os
os.makedirs(output_dir, exist_ok=True)

output_csv = f"{output_dir}/concatenated_training_data.csv"
concatenated.to_csv(output_csv, index=False)
print(f"\n✓ Saved: {output_csv}")

# Save gene list
gene_list_path = f"{output_dir}/gene_list.txt"
with open(gene_list_path, 'w') as f:
    for gene in common_genes:
        f.write(f"{gene}\n")
print(f"✓ Saved: {gene_list_path}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
