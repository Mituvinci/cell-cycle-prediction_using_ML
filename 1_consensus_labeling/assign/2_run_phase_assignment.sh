#!/usr/bin/env bash
################################################################################
#   Merge consensus labels for Human and Mouse datasets
#   AND create full training data with expression matrix
################################################################################

# Configuration
REASSIGNED_HUMAN="./reassigned_predictions_human"
REASSIGNED_MOUSE="./reassigned_predictions_mouse"
FINAL_HUMAN="./final_training_data_human"
FINAL_MOUSE="./final_training_data_mouse"
DATA_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data"

# STEP 1: Create consensus labels
echo "================================================================================"
echo "STEP 1: Creating Consensus Labels"
echo "================================================================================"

# Human (PBMC): minimum agreement = 4
echo ""
echo "Processing HUMAN (PBMC) - Minimum agreement: 4"
echo "--------------------------------------------------------------------------------"
python merge_reassigned.py \
    --input_dir "$REASSIGNED_HUMAN" \
    --output_dir "$FINAL_HUMAN" \
    --min_agreement 3

echo ""

# Mouse (Mouse Brain): minimum agreement = 3
echo "Processing MOUSE (Mouse Brain) - Minimum agreement: 3"
echo "--------------------------------------------------------------------------------"
python merge_reassigned.py \
    --input_dir "$REASSIGNED_MOUSE" \
    --output_dir "$FINAL_MOUSE" \
    --min_agreement 3

echo ""
echo "================================================================================"
echo "STEP 2: Merging Expression Data with Consensus Labels"
echo "================================================================================"

# Human PBMC
echo ""
echo "Merging Human PBMC Expression Data"
echo "--------------------------------------------------------------------------------"
python merge_expression_with_labels.py \
    --expression "$DATA_DIR/pbmc_healthy_human/trainingdata_cellcylescore_pbmc_healthy_human_normalized_gene_expression.csv" \
    --labels "$FINAL_HUMAN/pbmc_human_consensus_full.csv" \
    --output "$FINAL_HUMAN/pbmc_human_training_data.csv" \
    --min_agreement 3

echo ""

# Mouse Brain
echo "Merging Mouse Brain Expression Data"
echo "--------------------------------------------------------------------------------"
python merge_expression_with_labels.py \
    --expression "$DATA_DIR/10-k-brain-cells_healthy_mouse/trainingdata_cellcylescore_10-k-brain-cells_healthy_mouse_normalized_gene_expression.csv" \
    --labels "$FINAL_MOUSE/mouse_brain_consensus_full.csv" \
    --output "$FINAL_MOUSE/mouse_brain_training_data.csv" \
    --min_agreement 3

echo ""
echo "================================================================================"
echo "ALL TRAINING DATA CREATED!"
echo "================================================================================"
echo ""
echo "Final training files:"
echo "  Human: $FINAL_HUMAN/pbmc_human_training_data.csv"
echo "  Mouse: $FINAL_MOUSE/mouse_brain_training_data.csv"
echo ""
echo "Format: [gex_barcode, gene1, gene2, ..., geneN, CellID, Predicted]"
echo "================================================================================"

