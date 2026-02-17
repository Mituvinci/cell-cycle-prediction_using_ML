#!/usr/bin/env python3
"""
Transpose GSE64016 from Cells x Genes to Genes x Cells format
for UNIVERSAL_SEURAT.R processing
"""

import pandas as pd

print("="*80)
print("TRANSPOSE GSE64016: Cells x Genes -> Genes x Cells")
print("="*80)

# Input: cells x genes
input_file = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv"

# Output: genes x cells (for Seurat)
output_file = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression_raw_for_seurat.csv"

print(f"\nLoading: {input_file}")
df = pd.read_csv(input_file, index_col=0)

print(f"  Current shape (Cells x Genes): {df.shape}")
print(f"  First 5 cells: {list(df.index[:5])}")
print(f"  First 5 genes: {list(df.columns[:5])}")

# Check values to confirm it's not log-normalized
print(f"\n  Value range: [{df.values.min():.2f}, {df.values.max():.2f}]")
print(f"  Mean: {df.values.mean():.2f}")

if df.values.max() > 100:
    print("  ✓ Confirmed: NOT log-normalized (max > 100)")
else:
    print("  WARNING: Max value < 100, might already be normalized!")

# Transpose: Genes x Cells
df_transposed = df.T

print(f"\n  Transposed shape (Genes x Cells): {df_transposed.shape}")
print(f"  First 5 genes: {list(df_transposed.index[:5])}")
print(f"  First 5 cells: {list(df_transposed.columns[:5])}")

# Save transposed data
print(f"\nSaving transposed data: {output_file}")
df_transposed.to_csv(output_file)

print("\n" + "="*80)
print("DONE! Now run UNIVERSAL_SEURAT.R on the transposed file:")
print("="*80)
print(f"""
Rscript 0_preprocessing/training_data/UNIVERSAL_SEURAT.R \\
  --input {output_file} \\
  --output data/benchmarks_preprocessed/GSE64016/normalized_seurat/ \\
  --sample GSE64016 \\
  --format csv \\
  --species human
""")
print("="*80)
