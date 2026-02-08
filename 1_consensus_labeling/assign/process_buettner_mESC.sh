#!/usr/bin/bash
################################################################################
# Process Buettner mESC Dataset - Apply Phase Mappings
################################################################################
#
# This script applies phase mappings to Buettner mESC predictions from:
# - Seurat (reference/baseline)
# - Tricycle
# - ccAFv2
#
# Creates reassigned predictions for benchmark evaluation (NOT training)
#
# Author: Halima Akhter
# Date: 2025-12-05
################################################################################

set -e

# Base directories
BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq"
DATA_DIR="$BASE_DIR/data/Training_data/BuettnerESCData"
ASSIGN_DIR="$BASE_DIR/cell_cycle_prediction/1_consensus_labeling/assign"
OUTPUT_DIR="$ASSIGN_DIR/reassigned_predictions_buettner"

echo "================================================================================"
echo "BUETTNER mESC - PHASE REASSIGNMENT"
echo "================================================================================"
echo ""
echo "Input directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Config: phase_mappings_buettner_mESC.yaml"
echo ""
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Apply phase mappings
echo "Applying phase mappings to Buettner mESC predictions..."
echo ""

python "$ASSIGN_DIR/apply_phase_mapping.py" \
    --mapping "$ASSIGN_DIR/phase_mappings_buettner_mESC.yaml" \
    --data_dir "$BASE_DIR/data/Training_data" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "SUCCESS: Buettner mESC phase reassignment complete!"
    echo "================================================================================"
    echo ""
    echo "Output files:"
    echo "  - $OUTPUT_DIR/buettner_mESC/seurat_reassigned.csv"
    echo "  - $OUTPUT_DIR/buettner_mESC/tricycle_reassigned.csv"
    echo "  - $OUTPUT_DIR/buettner_mESC/ccafv2_reassigned.csv"
    echo ""
    echo "These files can now be used as benchmark ground truth for model evaluation"
    echo ""
    echo "================================================================================"
else
    echo ""
    echo "ERROR: Phase reassignment failed!"
    exit 1
fi
