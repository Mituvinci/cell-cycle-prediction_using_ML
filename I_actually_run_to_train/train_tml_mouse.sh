#!/usr/bin/env bash
################################################################################
# Train ALL Traditional ML Models on Mouse Brain Data
# Uses merged training data: expression matrix + consensus labels
# Uses pre-computed gene list from mousebrain_3benchmark_sup.txt
################################################################################

set -e

echo "================================================================================"
echo "TRAINING ALL TRADITIONAL ML MODELS ON MOUSE BRAIN DATA"
echo "================================================================================"
echo ""

# Configuration
TRAINING_DATA="1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data_UPPERCASE.csv"
GENE_LIST="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/gene_lists/mousebrain_3benchmark_sup.txt"
OUTPUT_BASE="models/mouse_brain_new"
TRIALS=15
CV_FOLDS=5
SCALING="robust"

# Check if training data exists
if [ ! -f "$TRAINING_DATA" ]; then
    echo "ERROR: Training data not found: $TRAINING_DATA"
    echo "Please run: cd 1_consensus_labeling/assign && bash create_final_training_data.sh"
    exit 1
fi

# Check if gene list exists
if [ ! -f "$GENE_LIST" ]; then
    echo "ERROR: Gene list not found: $GENE_LIST"
    echo "Please run: python scripts/find_common_genes.py"
    exit 1
fi

echo "Training data: $TRAINING_DATA"
echo "Gene list: $GENE_LIST"
echo "Output directory: $OUTPUT_BASE"
echo "Optuna trials: $TRIALS"
echo "CV folds: $CV_FOLDS"
echo ""

# Train all TML models

MODELS=("adaboost" "random_forest" "lgbm" "ensemble")



for MODEL in "${MODELS[@]}"; do
    echo "================================================================================"
    echo "Training: $MODEL (Mouse Brain)"
    echo "================================================================================"

    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs

    # Submit SLURM job
    # Using --gene-list for pre-computed gene list (load_and_preprocess_data_v3)
    sbatch --export=ALL,\
TRAINING_SCRIPT="2_model_training/train_traditional_ml.py",\
TRAINING_ARGS="--model $MODEL --dataset mouse_brain --gene-list $GENE_LIST --trials $TRIALS --cv $CV_FOLDS --scaling $SCALING --output $OUTPUT_DIR" \
           --job-name="tml_mouse_${MODEL}" \
           scripts/train_model_generic.slurm

    echo "Job submitted for $MODEL"
    echo ""
    sleep 2  # Avoid overwhelming scheduler
done

echo "================================================================================"
echo "ALL TRADITIONAL ML JOBS SUBMITTED"
echo "================================================================================"
echo ""
echo "Models: ${MODELS[@]}"
echo "Monitor with: squeue -u $USER"
echo "Check logs: ls -lh logs/"
echo ""
echo "================================================================================"
