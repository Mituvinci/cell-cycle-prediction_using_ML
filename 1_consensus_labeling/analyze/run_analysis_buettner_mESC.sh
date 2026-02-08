#!/usr/bin/bash
################################################################################
# Buettner mESC Analysis: Generate Contingency Tables + Heatmaps
################################################################################
#
# This script processes Buettner mESC dataset to create:
# 1. Contingency tables comparing OTHER tools vs SEURAT (reference)
# 2. Obs/expected ratio heatmaps for manual inspection
#
# IMPORTANT: Seurat is used as the REFERENCE/GROUND TRUTH.
# We only compare other tools (Tricycle, ccAFv2) against Seurat.
#
# Author: Halima Akhter
# Date: 2025-12-05
################################################################################

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directories
BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq"
BUETTNER_DIR="$BASE_DIR/data/Training_data/BuettnerESCData"
OUTPUT_DIR="$BASE_DIR/cell_cycle_prediction/1_consensus_labeling/results"

# Script locations
ANALYZE_DIR="$BASE_DIR/cell_cycle_prediction/1_consensus_labeling/analyze"
CONTINGENCY_SCRIPT="$ANALYZE_DIR/create_contingency_flexible.py"
HEATMAP_SCRIPT="$ANALYZE_DIR/generate_heatmap_flexible.py"

# Buettner files (3 tools: Seurat, Tricycle, ccAFv2)
BUETTNER_SEURAT="$BUETTNER_DIR/predicted_by_seurat_Buettner.csv"
BUETTNER_TRICYCLE="$BUETTNER_DIR/predicted_by_tricycle_Buettner.csv"
BUETTNER_CCAFV2="$BUETTNER_DIR/predicted_by_ccAFv2_Buettner.csv"

# =============================================================================
# FUNCTIONS
# =============================================================================

run_contingency_and_heatmap() {
    local TOOL1_FILE=$1
    local TOOL1_NAME=$2
    local TOOL2_FILE=$3
    local TOOL2_NAME=$4
    local DATASET_NAME=$5
    local OUT_DIR=$6

    echo ""
    echo "=============================================================================="
    echo "Processing: $TOOL1_NAME vs $TOOL2_NAME (Dataset: $DATASET_NAME)"
    echo "=============================================================================="

    # Create contingency table
    python $CONTINGENCY_SCRIPT \
        --tool1-file "$TOOL1_FILE" \
        --tool1-name "$TOOL1_NAME" \
        --tool2-file "$TOOL2_FILE" \
        --tool2-name "$TOOL2_NAME" \
        --dataset-name "$DATASET_NAME" \
        --output-dir "$OUT_DIR/contingency_tables"

    if [ $? -ne 0 ]; then
        echo "ERROR: Error creating contingency table!"
        return 1
    fi

    # Generate heatmap
    CONTINGENCY_FILE="$OUT_DIR/contingency_tables/contingency_${TOOL1_NAME}_vs_${TOOL2_NAME}_${DATASET_NAME}.csv"

    python $HEATMAP_SCRIPT \
        --contingency-table "$CONTINGENCY_FILE" \
        --tool1-name "$TOOL1_NAME" \
        --tool2-name "$TOOL2_NAME" \
        --dataset-name "$DATASET_NAME" \
        --output-dir "$OUT_DIR/heatmaps"

    if [ $? -ne 0 ]; then
        echo "ERROR: Error generating heatmap!"
        return 1
    fi

    echo "SUCCESS: Completed: $TOOL1_NAME vs $TOOL2_NAME"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "================================================================================"
echo "CONSENSUS LABELING - BUETTNER mESC ANALYSIS"
echo "================================================================================"
echo "This script will create contingency tables and heatmaps for:"
echo "  - BUETTNER: 2 comparisons (Seurat vs Tricycle, Seurat vs ccAFv2)"
echo ""
echo "IMPORTANT: Seurat is used as the REFERENCE. We compare other tools AGAINST Seurat."
echo ""
echo "Input directory: $BUETTNER_DIR"
echo "Output directory: $OUTPUT_DIR/buettner_mESC"
echo "================================================================================"
echo ""

# Verify input files exist
echo "Verifying input files..."
if [ ! -f "$BUETTNER_SEURAT" ]; then
    echo "ERROR: Seurat file not found: $BUETTNER_SEURAT"
    exit 1
fi
if [ ! -f "$BUETTNER_TRICYCLE" ]; then
    echo "ERROR: Tricycle file not found: $BUETTNER_TRICYCLE"
    exit 1
fi
if [ ! -f "$BUETTNER_CCAFV2" ]; then
    echo "ERROR: ccAFv2 file not found: $BUETTNER_CCAFV2"
    exit 1
fi
echo "All input files found!"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/buettner_mESC/contingency_tables"
mkdir -p "$OUTPUT_DIR/buettner_mESC/heatmaps"

# =============================================================================
# BUETTNER mESC DATASET (2 comparisons - only vs Seurat)
# =============================================================================

echo ""
echo "################################################################################"
echo "# BUETTNER mESC DATASET"
echo "################################################################################"

# Seurat vs Tricycle
run_contingency_and_heatmap \
    "$BUETTNER_SEURAT" "seurat" \
    "$BUETTNER_TRICYCLE" "tricycle" \
    "buettner_mESC" "$OUTPUT_DIR/buettner_mESC"

# Seurat vs ccAFv2
run_contingency_and_heatmap \
    "$BUETTNER_SEURAT" "seurat" \
    "$BUETTNER_CCAFV2" "ccafv2" \
    "buettner_mESC" "$OUTPUT_DIR/buettner_mESC"

echo ""
echo "SUCCESS: Buettner mESC dataset analysis complete!"
echo "   Results: $OUTPUT_DIR/buettner_mESC/heatmaps/"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "================================================================================"
echo "SUCCESS SUCCESS SUCCESS  ANALYSIS COMPLETE! SUCCESS SUCCESS SUCCESS"
echo "================================================================================"
echo ""
echo "Output Locations:"
echo "   Buettner heatmaps: $OUTPUT_DIR/buettner_mESC/heatmaps/"
echo "     - seurat_vs_tricycle"
echo "     - seurat_vs_ccafv2"
echo ""
echo "   Buettner contingency tables: $OUTPUT_DIR/buettner_mESC/contingency_tables/"
echo "     - contingency_seurat_vs_tricycle_buettner_mESC.csv"
echo "     - contingency_seurat_vs_ccafv2_buettner_mESC.csv"
echo ""
echo "NEXT STEPS:"
echo "   1. Open the heatmap images and inspect color patterns"
echo "   2. Identify which sub-phases map to G1, S, G2M based on Seurat alignment"
echo "   3. Create YAML config file with your mappings:"
echo "      - configs/phase_mappings/buettner_mESC.yaml"
echo "   4. Run phase reassignment using your YAML config"
echo "   5. Run merge_consensus.py to create final consensus labels"
echo ""
echo "================================================================================"
