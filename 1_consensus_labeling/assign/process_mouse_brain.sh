#!/usr/bin/env bash
################################################################################
# Process Mouse Brain 10k Dataset
# Applies manual phase mappings and creates consensus training data
################################################################################

set -e

echo "================================================================================"
echo "PROCESSING: Mouse Brain 10k Cells Dataset"
echo "================================================================================"
echo ""

# Configuration
MAPPING_FILE="phase_mappings_mouse_brain.yaml"
DATA_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data"
REASSIGNED_DIR="./reassigned_predictions_mouse"
FINAL_DIR="./final_training_data_mouse"
MIN_AGREEMENT=2

# Step 1: Apply phase mappings
echo "STEP 1: Applying phase mappings to tool predictions"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""
python apply_phase_mapping.py \
    --mapping "$MAPPING_FILE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$REASSIGNED_DIR"

echo ""
echo "Reassigned predictions saved to: $REASSIGNED_DIR"
echo ""

# Step 2: Merge consensus labels
echo "================================================================================"
echo "STEP 2: Merging consensus labels (minimum $MIN_AGREEMENT tools agreement)"
echo "================================================================================"
echo ""
python merge_reassigned.py \
    --input_dir "$REASSIGNED_DIR" \
    --output_dir "$FINAL_DIR" \
    --min_agreement "$MIN_AGREEMENT"

echo ""
echo "================================================================================"
echo "SUCCESS! Mouse Brain Training Data Ready"
echo "================================================================================"
echo ""
echo "Final training data: $FINAL_DIR/mouse_brain_consensus.csv"
echo ""
echo "Phase distribution and consensus statistics shown above."
echo ""
echo "NEXT STEPS:"
echo "  1. Review the consensus statistics"
echo "  2. Check phase distribution (should be balanced)"
echo "  3. Train models on the consensus data"
echo ""
echo "Example training commands:"
echo ""
echo "# Deep Learning (DNN3)"
echo "python ../../2_model_training/train_deep_learning.py \\"
echo "    --model simpledense \\"
echo "    --data $FINAL_DIR/mouse_brain_consensus.csv \\"
echo "    --trials 50 \\"
echo "    --cv 5 \\"
echo "    --output results_mouse_dnn3"
echo ""
echo "# Traditional ML (Random Forest)"
echo "python ../../2_model_training/train_traditional_ml.py \\"
echo "    --model randomforest \\"
echo "    --data $FINAL_DIR/mouse_brain_consensus.csv \\"
echo "    --trials 50 \\"
echo "    --cv 5 \\"
echo "    --output results_mouse_rf"
echo ""
echo "================================================================================"
