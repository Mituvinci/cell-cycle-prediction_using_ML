#!/bin/bash
################################################################################
# STEP 1: Generate Contingency Tables + Heatmaps for GSE75748 dataset
# 4 tools: Seurat (reference), Tricycle, ccAFv2, Revelio
# 3 comparisons: Seurat vs Tricycle, Seurat vs ccAFv2, Seurat vs Revelio
################################################################################

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq"
DATA_DIR="$BASE_DIR/data/GSE75748_hPSC"
OUTPUT_DIR="$BASE_DIR/cell_cycle_prediction/1_consensus_labeling/results/gse75748"
ANALYZE_DIR="$BASE_DIR/cell_cycle_prediction/1_consensus_labeling/analyze"

CONTINGENCY_SCRIPT="$ANALYZE_DIR/create_contingency_flexible.py"
HEATMAP_SCRIPT="$ANALYZE_DIR/generate_heatmap_flexible.py"

# Tool prediction files
SEURAT="$DATA_DIR/seurat_GSE75748_cc_phases.csv"
TRICYCLE="$DATA_DIR/tricycle_GSE75748.csv"
CCAFV2="$DATA_DIR/ccAFV2_GSE75748.csv"
REVELIO="$DATA_DIR/revelio_GSE75748.csv"

DATASET_NAME="gse75748"

# =============================================================================
# FUNCTIONS
# =============================================================================

run_contingency_and_heatmap() {
    local TOOL1_FILE=$1
    local TOOL1_NAME=$2
    local TOOL2_FILE=$3
    local TOOL2_NAME=$4

    echo ""
    echo "=============================================================================="
    echo "Processing: $TOOL1_NAME vs $TOOL2_NAME"
    echo "=============================================================================="

    # Create contingency table
    python $CONTINGENCY_SCRIPT \
        --tool1-file "$TOOL1_FILE" \
        --tool1-name "$TOOL1_NAME" \
        --tool2-file "$TOOL2_FILE" \
        --tool2-name "$TOOL2_NAME" \
        --dataset-name "$DATASET_NAME" \
        --output-dir "$OUTPUT_DIR/contingency_tables"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create contingency table for $TOOL1_NAME vs $TOOL2_NAME"
        return 1
    fi

    # Generate heatmap
    CONTINGENCY_FILE="$OUTPUT_DIR/contingency_tables/contingency_${TOOL1_NAME}_vs_${TOOL2_NAME}_${DATASET_NAME}.csv"

    python $HEATMAP_SCRIPT \
        --contingency-table "$CONTINGENCY_FILE" \
        --tool1-name "$TOOL1_NAME" \
        --tool2-name "$TOOL2_NAME" \
        --dataset-name "$DATASET_NAME" \
        --output-dir "$OUTPUT_DIR/heatmaps"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate heatmap for $TOOL1_NAME vs $TOOL2_NAME"
        return 1
    fi

    echo "SUCCESS: $TOOL1_NAME vs $TOOL2_NAME complete"
}

# =============================================================================
# MAIN
# =============================================================================

echo "================================================================================"
echo "GSE75748 - STEP 1: ANALYZE (Contingency Tables + Heatmaps)"
echo "================================================================================"
echo "Reference tool: Seurat"
echo "Comparisons: Seurat vs Tricycle, Seurat vs ccAFv2, Seurat vs Revelio"
echo "Output: $OUTPUT_DIR"
echo "================================================================================"

# Create output directories
mkdir -p "$OUTPUT_DIR/contingency_tables"
mkdir -p "$OUTPUT_DIR/heatmaps"

# =============================================================================
# STEP 0: Standardize column names (CellID, Predicted)
# =============================================================================

echo ""
echo "================================================================================"
echo "STEP 0: Standardizing column names to CellID and Predicted"
echo "================================================================================"

python "$ANALYZE_DIR/standardize_columns_GSE75748.py"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to standardize column names"
    exit 1
fi

echo "Column standardization complete"

# =============================================================================
# 3 COMPARISONS (all vs Seurat)
# =============================================================================

# 1. Seurat vs Tricycle
run_contingency_and_heatmap "$SEURAT" "seurat" "$TRICYCLE" "tricycle"

# 2. Seurat vs ccAFv2
run_contingency_and_heatmap "$SEURAT" "seurat" "$CCAFV2" "ccafv2"

# 3. Seurat vs Revelio
run_contingency_and_heatmap "$SEURAT" "seurat" "$REVELIO" "revelio"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "================================================================================"
echo "STEP 1 COMPLETE"
echo "================================================================================"
echo ""
echo "Heatmaps saved to:"
echo "  $OUTPUT_DIR/heatmaps/"
echo "    - heatmap_seurat_vs_tricycle_gse75748.png"
echo "    - heatmap_seurat_vs_ccafv2_gse75748.png"
echo "    - heatmap_seurat_vs_revelio_gse75748.png"
echo ""
echo "Contingency tables saved to:"
echo "  $OUTPUT_DIR/contingency_tables/"
echo ""
echo "NEXT STEP: Open the 3 heatmap PNG files and inspect colors."
echo "  Green = sub-phase aligns with that Seurat phase"
echo "  Then tell me the mappings for tricycle, ccafv2, revelio."
echo "================================================================================"
