#!/usr/bin/env bash
################################################################################
# Manual Phase Assignment Workflow
#
# This script guides you through the 3-step phase assignment process:
#   1. Create YAML mapping template
#   2. Manual editing (you fill in the mappings based on heatmaps)
#   3. Apply mappings to tool predictions
#   4. Merge consensus labels
#
# Usage:
#   bash run_phase_assignment.sh
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "MANUAL PHASE ASSIGNMENT WORKFLOW"
echo "================================================================================"
echo ""

# Configuration
RESULTS_DIR="../results"
MAPPING_FILE="phase_mappings_new_data.yaml"
DATA_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data"
REASSIGNED_DIR="./reassigned_predictions"
FINAL_DIR="./final_training_data"
MIN_AGREEMENT=3

# Step 1: Create YAML template
echo "STEP 1: Creating YAML mapping template"
echo "────────────────────────────────────────────────────────────────────────────────"
python create_mapping_template.py \
    --results_dir "$RESULTS_DIR" \
    --output "$MAPPING_FILE"

echo ""
echo "Template created: $MAPPING_FILE"
echo ""

# Step 2: Manual editing
echo "================================================================================"
echo "STEP 2: MANUAL EDITING REQUIRED"
echo "================================================================================"
echo ""
echo "Please complete the following steps:"
echo ""
echo "1. Open the heatmap PNG files:"
echo "   - $RESULTS_DIR/mouse/heatmaps/heatmap_seurat_vs_*.png"
echo "   - $RESULTS_DIR/human/heatmaps/heatmap_seurat_vs_*.png"
echo ""
echo "2. Edit $MAPPING_FILE:"
echo "   - Replace '???' with G1, S, or G2M"
echo "   - Based on heatmap color similarity"
echo "   - Warmer colors = stronger association"
echo ""
echo "3. Save the file"
echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
read -p "Press ENTER when you have finished editing $MAPPING_FILE..."
echo ""

# Check if file was edited
if grep -q "???" "$MAPPING_FILE"; then
    echo "WARNING: Found '???' placeholders in $MAPPING_FILE"
    echo "Please complete all mappings before continuing."
    read -p "Continue anyway? (y/n): " continue_choice
    if [[ "$continue_choice" != "y" ]]; then
        echo "Exiting. Please complete the mappings and run the script again."
        exit 1
    fi
fi

# Step 3: Apply mappings
echo "================================================================================"
echo "STEP 3: Applying phase mappings"
echo "================================================================================"
echo ""
python apply_phase_mapping.py \
    --mapping "$MAPPING_FILE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$REASSIGNED_DIR"

echo ""

# Step 4: Merge consensus
echo "================================================================================"
echo "STEP 4: Merging consensus labels"
echo "================================================================================"
echo ""
echo "Minimum agreement threshold: $MIN_AGREEMENT tools"
echo ""
python merge_reassigned.py \
    --input_dir "$REASSIGNED_DIR" \
    --output_dir "$FINAL_DIR" \
    --min_agreement "$MIN_AGREEMENT"

echo ""
echo "================================================================================"
echo "WORKFLOW COMPLETE!"
echo "================================================================================"
echo ""
echo "Final training data saved to: $FINAL_DIR"
echo ""
echo "Next steps:"
echo "  1. Review consensus statistics above"
echo "  2. Check phase distributions (should be balanced)"
echo "  3. Train models on the consensus CSV files"
echo ""
echo "Example training command:"
echo "  python ../../2_model_training/train_deep_learning.py \\"
echo "      --model simpledense \\"
echo "      --data $FINAL_DIR/mouse_brain_consensus.csv \\"
echo "      --trials 50 \\"
echo "      --cv 5"
echo ""
echo "================================================================================"
