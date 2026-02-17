#!/usr/bin/env bash
################################################################################
# Renormalize GSE64016 using UNIVERSAL_SEURAT.R
# This applies the SAME normalization as REH/SUP training data
################################################################################

set -e

BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq"
cd "$BASE_DIR/cell_cycle_prediction"

echo "================================================================================"
echo "RENORMALIZE GSE64016 BENCHMARK"
echo "================================================================================"
echo ""
echo "Problem: GSE64016 is NOT log-normalized (max value: 129,617)"
echo "Solution: Apply Seurat log normalization (same as REH/SUP)"
echo ""
echo "================================================================================"
echo ""

# Step 1: Transpose GSE64016 (Cells x Genes -> Genes x Cells)
echo "STEP 1: Transpose GSE64016 to Seurat format (Genes x Cells)"
echo "================================================================================"
python 0_preprocessing/transpose_GSE64016.py

if [ $? -ne 0 ]; then
    echo "ERROR: Transpose failed!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 2: Apply Seurat Log Normalization (Python)"
echo "================================================================================"

# Run Python normalization (same formula as Seurat!)
python 0_preprocessing/seurat_normalize_python.py

if [ $? -ne 0 ]; then
    echo "ERROR: Normalization failed!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 3: Replace old GSE64016_expression.csv with normalized version"
echo "================================================================================"

# Python output file (already in Cells x Genes format)
NORMALIZED_FILE="$BASE_DIR/data/benchmarks_preprocessed/GSE64016/GSE64016_expression_normalized_python.csv"

if [ ! -f "$NORMALIZED_FILE" ]; then
    echo "ERROR: Normalized file not found: $NORMALIZED_FILE"
    exit 1
fi

# Backup original
BACKUP_FILE="$BASE_DIR/data/benchmarks_preprocessed/GSE64016/GSE64016_expression_ORIGINAL_NOT_NORMALIZED.csv"
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backing up original file..."
    cp "$BASE_DIR/data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv" "$BACKUP_FILE"
    echo "  Saved: $BACKUP_FILE"
fi

# Replace with normalized version
echo "Replacing GSE64016_expression.csv with normalized version..."
cp "$NORMALIZED_FILE" "$BASE_DIR/data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv"

echo ""
echo "================================================================================"
echo "DONE! GSE64016 has been re-normalized"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  - Original file backed up to: GSE64016_expression_ORIGINAL_NOT_NORMALIZED.csv"
echo "  - New normalized file: GSE64016_expression.csv"
echo "  - Normalization method: Seurat log1p(counts/total_counts * 10000)"
echo "  - Same method used for REH/SUP training data"
echo ""
echo "Next steps:"
echo "  1. Re-run evaluation: bash I_actually_run_to_train/run_all_new_evaluate_visu.sh"
echo "  2. Check if accuracy improves on GSE64016"
echo ""
echo "================================================================================"
