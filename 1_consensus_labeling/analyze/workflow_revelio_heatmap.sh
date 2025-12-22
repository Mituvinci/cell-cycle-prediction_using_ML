#!/bin/bash
################################################################################
# Revelio Heatmap Generation Workflow
################################################################################
#
# This script:
# 1. Adds ground truth to Revelio predictions (inner join on CellID)
# 2. Creates contingency tables (Revelio vs GroundTruth)
# 3. Generates heatmaps for manual inspection
#
# Author: Halima Akhter
# Date: 2025-12-16
################################################################################

set -e

echo "================================================================================"
echo "REVELIO HEATMAP GENERATION WORKFLOW"
echo "================================================================================"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directories
BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq"
ANALYZE_DIR="$BASE_DIR/cell_cycle_prediction/1_consensus_labeling/analyze"
BUETTNER_DIR="$BASE_DIR/data/Training_data/BuettnerESCData"
MOUSE_DIR="$BASE_DIR/data/Training_data/10-k-brain-cells_healthy_mouse"
OUTPUT_DIR="$BASE_DIR/cell_cycle_prediction/1_consensus_labeling/results"

# Script locations
CONTINGENCY_SCRIPT="$ANALYZE_DIR/create_contingency_flexible.py"
HEATMAP_SCRIPT="$ANALYZE_DIR/generate_heatmap_flexible.py"
ADD_GT_SCRIPT="$ANALYZE_DIR/add_groundtruth_to_revelio.py"

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

# =============================================================================
# STEP 1: Add ground truth to Revelio predictions
# =============================================================================

echo ""
echo "STEP 1: Adding ground truth to Revelio predictions..."
echo ""

python $ADD_GT_SCRIPT

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to add ground truth!"
    exit 1
fi

# =============================================================================
# STEP 2: Create separate files for contingency analysis
# =============================================================================

echo ""
echo "STEP 2: Creating separate files for contingency analysis..."
echo ""

python3 << 'EOF'
import pandas as pd

# Buettner
buettner_merged = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/BuettnerESCData/predicted_by_revelio_Buettener_with_groundtruth.csv"
df = pd.read_csv(buettner_merged)

# Create Revelio-only file
df_revelio = df[['CellID', 'Predicted']].copy()
df_revelio.to_csv("/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/BuettnerESCData/revelio_predictions_only.csv", index=False)

# Create GroundTruth-only file (rename GroundTruth to Predicted for contingency script)
df_gt = df[['CellID', 'GroundTruth']].copy()
df_gt = df_gt.rename(columns={'GroundTruth': 'Predicted'})
df_gt.to_csv("/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/BuettnerESCData/groundtruth_only.csv", index=False)

print("Created Buettner separate files")

# Mouse brain
mouse_merged = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/10-k-brain-cells_healthy_mouse/revelio_10-k-brain-cells_healthy_mouse_with_groundtruth.csv"
df = pd.read_csv(mouse_merged)

# Create Revelio-only file
df_revelio = df[['CellID', 'Predicted']].copy()
df_revelio.to_csv("/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/10-k-brain-cells_healthy_mouse/revelio_predictions_only.csv", index=False)

# Create GroundTruth-only file
df_gt = df[['CellID', 'GroundTruth']].copy()
df_gt = df_gt.rename(columns={'GroundTruth': 'Predicted'})
df_gt.to_csv("/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/10-k-brain-cells_healthy_mouse/groundtruth_only.csv", index=False)

print("Created Mouse brain separate files")
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create separate files!"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR/buettner_revelio/contingency_tables"
mkdir -p "$OUTPUT_DIR/buettner_revelio/heatmaps"
mkdir -p "$OUTPUT_DIR/mouse_brain_revelio/contingency_tables"
mkdir -p "$OUTPUT_DIR/mouse_brain_revelio/heatmaps"

# =============================================================================
# STEP 3: BUETTNER mESC - Revelio vs GroundTruth
# =============================================================================

echo ""
echo "################################################################################"
echo "# BUETTNER mESC DATASET"
echo "################################################################################"

run_contingency_and_heatmap \
    "$BUETTNER_DIR/groundtruth_only.csv" "groundtruth" \
    "$BUETTNER_DIR/revelio_predictions_only.csv" "revelio" \
    "buettner_mESC" "$OUTPUT_DIR/buettner_revelio"

echo ""
echo "SUCCESS: Buettner mESC Revelio analysis complete!"
echo "   Results: $OUTPUT_DIR/buettner_revelio/heatmaps/"

# =============================================================================
# STEP 4: 10K MOUSE BRAIN - Revelio vs GroundTruth
# =============================================================================

echo ""
echo "################################################################################"
echo "# 10K MOUSE BRAIN DATASET"
echo "################################################################################"

run_contingency_and_heatmap \
    "$MOUSE_DIR/groundtruth_only.csv" "groundtruth" \
    "$MOUSE_DIR/revelio_predictions_only.csv" "revelio" \
    "mouse_brain" "$OUTPUT_DIR/mouse_brain_revelio"

echo ""
echo "SUCCESS: 10k mouse brain Revelio analysis complete!"
echo "   Results: $OUTPUT_DIR/mouse_brain_revelio/heatmaps/"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "================================================================================"
echo "SUCCESS SUCCESS SUCCESS  ANALYSIS COMPLETE! SUCCESS SUCCESS SUCCESS"
echo "================================================================================"
echo ""
echo "Output Locations:"
echo "   Buettner heatmap: $OUTPUT_DIR/buettner_revelio/heatmaps/"
echo "     - heatmap_groundtruth_vs_revelio_buettner_mESC.png (Y-axis: GroundTruth, X-axis: Revelio)"
echo ""
echo "   Mouse brain heatmap: $OUTPUT_DIR/mouse_brain_revelio/heatmaps/"
echo "     - heatmap_groundtruth_vs_revelio_mouse_brain.png (Y-axis: GroundTruth, X-axis: Revelio)"
echo ""
echo "   Contingency tables:"
echo "     - $OUTPUT_DIR/buettner_revelio/contingency_tables/contingency_groundtruth_vs_revelio_buettner_mESC.csv"
echo "     - $OUTPUT_DIR/mouse_brain_revelio/contingency_tables/contingency_groundtruth_vs_revelio_mouse_brain.csv"
echo ""
echo "NEXT STEPS:"
echo "   1. Open the heatmap images and inspect color patterns"
echo "   2. Identify which Revelio phases map to G1, S, G2M based on GroundTruth alignment"
echo "   3. Tell Claude the mappings (e.g., 'Buettner: G1.S->G1, S->S, G2->G2M')"
echo "   4. Claude will create YAML config files with your mappings"
echo "   5. Claude will run apply_phase_mapping.py for both datasets"
echo ""
echo "================================================================================"
