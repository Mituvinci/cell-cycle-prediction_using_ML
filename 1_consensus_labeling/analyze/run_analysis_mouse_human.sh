#!/usr/bin/bash
"""
Master Script: Generate Contingency Tables + Heatmaps
======================================================

This script processes MOUSE and HUMAN datasets to create:
1. Contingency tables comparing OTHER tools vs SEURAT (reference)
2. Obs/expected ratio heatmaps for manual inspection

IMPORTANT: Seurat is used as the REFERENCE/GROUND TRUTH.
We only compare other tools (Tricycle, ccAFv2, Revelio) against Seurat.

Author: Halima Akhter
Date: 2025-11-28
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directories
BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq"
MOUSE_DIR="$BASE_DIR/data/Training_data/10-k-brain-cells_healthy_mouse"
HUMAN_DIR="$BASE_DIR/data/Training_data/pbmc_healthy_human"
OUTPUT_DIR="$BASE_DIR/cell_cycle_prediction/1_consensus_labeling/results"

# Script locations
ANALYZE_DIR="$BASE_DIR/cell_cycle_prediction/1_consensus_labeling/analyze"
CONTINGENCY_SCRIPT="$ANALYZE_DIR/create_contingency_flexible.py"
HEATMAP_SCRIPT="$ANALYZE_DIR/generate_heatmap_flexible.py"

# Mouse files (3 tools: Seurat, Tricycle, ccAFv2)
MOUSE_SEURAT="$MOUSE_DIR/trainingdata_cellcylescore_10-k-brain-cells_healthy_mouse_cc_phases.csv"
MOUSE_TRICYCLE="$MOUSE_DIR/tricycle_mouse_brain_10k.csv"
MOUSE_CCAFV2="$MOUSE_DIR/ccAFV2_10-k-brain-cells_healthy_mouse.csv"

# Human files (4 tools: Seurat, Tricycle, ccAFv2, Revelio)
HUMAN_SEURAT="$HUMAN_DIR/trainingdata_cellcylescore_pbmc_healthy_human_cc_phases.csv"
HUMAN_TRICYCLE="$HUMAN_DIR/tricycle_pbmc_healthy_human.csv"
HUMAN_CCAFV2="$HUMAN_DIR/ccAFV2_pbmc_healthy_human.csv"
HUMAN_REVELIO="$HUMAN_DIR/revelio_pbmc_healthy_human_1.csv"

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
        echo "❌ Error creating contingency table!"
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
        echo "❌ Error generating heatmap!"
        return 1
    fi

    echo "✅ Completed: $TOOL1_NAME vs $TOOL2_NAME"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "================================================================================"
echo "CONSENSUS LABELING - STEP 1: ANALYZE"
echo "================================================================================"
echo "This script will create contingency tables and heatmaps for:"
echo "  - MOUSE: 2 comparisons (Seurat vs Tricycle, Seurat vs ccAFv2)"
echo "  - HUMAN: 3 comparisons (Seurat vs Tricycle, Seurat vs ccAFv2, Seurat vs Revelio)"
echo ""
echo "IMPORTANT: Seurat is used as the REFERENCE. We compare other tools AGAINST Seurat."
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/contingency_tables"
mkdir -p "$OUTPUT_DIR/heatmaps"
mkdir -p "$OUTPUT_DIR/mouse/contingency_tables"
mkdir -p "$OUTPUT_DIR/mouse/heatmaps"
mkdir -p "$OUTPUT_DIR/human/contingency_tables"
mkdir -p "$OUTPUT_DIR/human/heatmaps"

# =============================================================================
# MOUSE DATASET (2 comparisons - only vs Seurat)
# =============================================================================

echo ""
echo "################################################################################"
echo "# MOUSE DATASET: 10-k-brain-cells_healthy_mouse"
echo "################################################################################"

# Seurat vs Tricycle
run_contingency_and_heatmap \
    "$MOUSE_SEURAT" "seurat" \
    "$MOUSE_TRICYCLE" "tricycle" \
    "mouse_brain" "$OUTPUT_DIR/mouse"

# Seurat vs ccAFv2
run_contingency_and_heatmap \
    "$MOUSE_SEURAT" "seurat" \
    "$MOUSE_CCAFV2" "ccafv2" \
    "mouse_brain" "$OUTPUT_DIR/mouse"

echo ""
echo "✅ Mouse dataset analysis complete!"
echo "   Results: $OUTPUT_DIR/mouse/heatmaps/"

# =============================================================================
# HUMAN DATASET (3 comparisons - only vs Seurat)
# =============================================================================

echo ""
echo "################################################################################"
echo "# HUMAN DATASET: pbmc_healthy_human"
echo "################################################################################"

# Seurat vs Tricycle
run_contingency_and_heatmap \
    "$HUMAN_SEURAT" "seurat" \
    "$HUMAN_TRICYCLE" "tricycle" \
    "pbmc_human" "$OUTPUT_DIR/human"

# Seurat vs ccAFv2
run_contingency_and_heatmap \
    "$HUMAN_SEURAT" "seurat" \
    "$HUMAN_CCAFV2" "ccafv2" \
    "pbmc_human" "$OUTPUT_DIR/human"

# Seurat vs Revelio
run_contingency_and_heatmap \
    "$HUMAN_SEURAT" "seurat" \
    "$HUMAN_REVELIO" "revelio" \
    "pbmc_human" "$OUTPUT_DIR/human"

echo ""
echo "✅ Human dataset analysis complete!"
echo "   Results: $OUTPUT_DIR/human/heatmaps/"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "================================================================================"
echo "✅ ✅ ✅  ANALYSIS COMPLETE! ✅ ✅ ✅"
echo "================================================================================"
echo ""
echo "📁 Output Locations:"
echo "   Mouse heatmaps: $OUTPUT_DIR/mouse/heatmaps/"
echo "     - seurat_vs_tricycle"
echo "     - seurat_vs_ccafv2"
echo ""
echo "   Human heatmaps: $OUTPUT_DIR/human/heatmaps/"
echo "     - seurat_vs_tricycle"
echo "     - seurat_vs_ccafv2"
echo "     - seurat_vs_revelio"
echo ""
echo "🔍 NEXT STEPS:"
echo "   1. Open the heatmap images and inspect color patterns"
echo "   2. Identify which sub-phases map to G1, S, G2M based on Seurat alignment"
echo "   3. Create YAML config files with your mappings:"
echo "      - Mouse: configs/phase_mappings/mouse_brain.yaml"
echo "      - Human: configs/phase_mappings/pbmc_human.yaml"
echo "   4. Run phase reassignment using your YAML configs"
echo "   5. Run merge_consensus.py to create final consensus labels"
echo ""
echo "================================================================================"
