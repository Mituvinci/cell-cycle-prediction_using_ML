#!/usr/bin/env bash
################################################################################
# Create Final Training Data by Merging Expression + Labels
#
# This script merges gene expression matrices with consensus labels
# to create final training data ready for model training
################################################################################

set -e

echo "================================================================================"
echo "CREATING FINAL TRAINING DATA"
echo "================================================================================"
echo ""

DATA_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data"

# Human PBMC
echo "STEP 1: Merging Human PBMC Data"
echo "────────────────────────────────────────────────────────────────────────────────"
python merge_expression_with_labels.py \
    --expression "$DATA_DIR/pbmc_healthy_human/trainingdata_cellcylescore_pbmc_healthy_human_normalized_gene_expression.csv" \
    --labels final_training_data_human/pbmc_human_consensus_full.csv \
    --output final_training_data_human/pbmc_human_training_data.csv \
    --min_agreement 2

echo ""

# Mouse Brain
echo "================================================================================"
echo "STEP 2: Merging Mouse Brain Data"
echo "────────────────────────────────────────────────────────────────────────────────"
python merge_expression_with_labels.py \
    --expression "$DATA_DIR/10-k-brain-cells_healthy_mouse/trainingdata_cellcylescore_10-k-brain-cells_healthy_mouse_normalized_gene_expression.csv" \
    --labels final_training_data_mouse/mouse_brain_consensus_full.csv \
    --output final_training_data_mouse/mouse_brain_training_data.csv \
    --min_agreement 2

echo ""
echo "================================================================================"
echo "ALL TRAINING DATA CREATED!"
echo "================================================================================"
echo ""
echo "Final training files:"
echo "  Human: final_training_data_human/pbmc_human_training_data.csv"
echo "  Mouse: final_training_data_mouse/mouse_brain_training_data.csv"
echo ""
echo "Format: [gex_barcode, gene1, gene2, ..., geneN, CellID, Predicted]"
echo ""
echo "Ready to train models!"
echo "================================================================================"
