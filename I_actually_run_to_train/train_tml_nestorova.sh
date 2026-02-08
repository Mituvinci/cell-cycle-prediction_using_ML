#!/usr/bin/env bash
################################################################################
# Train ALL Traditional ML Models on Nestorova Mouse Data
# Uses merged training data: expression matrix + consensus labels
# Uses pre-computed gene list from nestorova_3benchmark_sup.txt
################################################################################

set -e

echo "================================================================================"
echo "TRAINING ALL TRADITIONAL ML MODELS ON NESTOROVA MOUSE DATA"
echo "================================================================================"
echo ""

# Configuration
TRAINING_DATA="1_consensus_labeling/assign/final_training_data_nestorova/nestorova_training_data.csv"
GENE_LIST="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/gene_lists/nestorova_3benchmark_sup.txt"
OUTPUT_BASE="models/nestorova"
TRIALS=15
CV_FOLDS=5
SCALING="robust"

# Check if training data exists
if [ ! -f "$TRAINING_DATA" ]; then
    echo "ERROR: Training data not found: $TRAINING_DATA"
    echo "Please run: bash 1_consensus_labeling/assign/process_nestorova.sh"
    exit 1
fi

# Check if gene list exists
if [ ! -f "$GENE_LIST" ]; then
    echo "ERROR: Gene list not found: $GENE_LIST"
    echo "Please run: python scripts/find_common_genes.py"
    exit 1
fi

echo "Training data: $TRAINING_DATA"
echo "Gene list: $GENE_LIST (6479 genes)"
echo "Output directory: $OUTPUT_BASE"
echo "Optuna trials: $TRIALS"
echo "CV folds: $CV_FOLDS"
echo "Scaling: $SCALING"
echo ""

# Train all TML models
MODELS=("adaboost" "random_forest" "lgbm" "ensemble")

for MODEL in "${MODELS[@]}"; do
    echo "================================================================================"
    echo "Training: $MODEL (Nestorova)"
    echo "================================================================================"

    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs

    # Submit SLURM job
    sbatch --export=ALL,\
TRAINING_SCRIPT="2_model_training/train_traditional_ml.py",\
TRAINING_ARGS="--model $MODEL --dataset nestorova --gene-list $GENE_LIST --trials $TRIALS --cv $CV_FOLDS --scaling $SCALING --output $OUTPUT_DIR" \
           --job-name="tml_nestorova_${MODEL}" \
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
