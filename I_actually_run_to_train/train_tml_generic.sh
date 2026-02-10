#!/usr/bin/env bash
################################################################################
# Generic Traditional ML Training Script
# Usage: bash train_tml_generic.sh <dataset_name> <training_data_path> <gene_list_path> <scaling_method> [feature_selection_method]
# Example: bash train_tml_generic.sh reh 1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv gene_lists/reh_3benchmark_sup.txt robust
# Example: bash train_tml_generic.sh reh 1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv gene_lists/reh_3benchmark_sup.txt robust SelectKBest
################################################################################

set -e

# Check arguments
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "ERROR: Wrong number of arguments"
    echo "Usage: $0 <dataset_name> <training_data_path> <gene_list_path> <scaling_method> [feature_selection_method]"
    echo ""
    echo "Arguments:"
    echo "  dataset_name              : Name of dataset (e.g., reh, sup, nestorova)"
    echo "  training_data_path        : Path to training CSV file"
    echo "  gene_list_path            : Path to gene list text file"
    echo "  scaling_method            : Scaling method (standard or robust)"
    echo "  feature_selection_method  : Optional feature selection (SelectKBest, ElasticCV, or omit for none)"
    echo ""
    echo "Examples:"
    echo "  $0 reh 1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv gene_lists/reh_3benchmark_sup.txt robust"
    echo "  $0 reh 1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv gene_lists/reh_3benchmark_sup.txt robust SelectKBest"
    exit 1
fi

DATASET_NAME="$1"
TRAINING_DATA="$2"
GENE_LIST="$3"
SCALING="$4"
FEATURE_SELECTION="${5:-none}"

# Convert relative paths to absolute if needed
if [[ "$GENE_LIST" != /* ]]; then
    GENE_LIST="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/${GENE_LIST}"
fi

echo "================================================================================"
echo "TRAINING ALL TRADITIONAL ML MODELS ON ${DATASET_NAME^^} DATA"
echo "================================================================================"
echo ""

# Configuration
TRIALS=15
CV_FOLDS=5
K_FEATURES=300  # For SelectKBest feature selection

# Set output directory based on scaling and feature selection
if [ "$FEATURE_SELECTION" != "none" ]; then
    OUTPUT_BASE="models/${DATASET_NAME}_${SCALING}_${FEATURE_SELECTION}"
else
    OUTPUT_BASE="models/${DATASET_NAME}_${SCALING}"
fi

# Check if training data exists
if [ ! -f "$TRAINING_DATA" ]; then
    echo "ERROR: Training data not found: $TRAINING_DATA"
    exit 1
fi

# Check if gene list exists
if [ ! -f "$GENE_LIST" ]; then
    echo "ERROR: Gene list not found: $GENE_LIST"
    exit 1
fi

# Validate scaling method
if [[ "$SCALING" != "standard" && "$SCALING" != "robust" ]]; then
    echo "ERROR: Scaling method must be 'standard' or 'robust', got: $SCALING"
    exit 1
fi

# Count genes
GENE_COUNT=$(wc -l < "$GENE_LIST")

echo "Dataset name: $DATASET_NAME"
echo "Training data: $TRAINING_DATA"
echo "Gene list: $GENE_LIST ($GENE_COUNT genes)"
echo "Output directory: $OUTPUT_BASE"
echo "Scaling method: $SCALING"
echo "Feature selection: $FEATURE_SELECTION"
if [ "$FEATURE_SELECTION" != "none" ]; then
    echo "K features: $K_FEATURES"
fi
echo "Optuna trials: $TRIALS"
echo "CV folds: $CV_FOLDS"
echo ""

# Train all TML models
MODELS=("adaboost" "random_forest" "lgbm" "ensemble")

for MODEL in "${MODELS[@]}"; do
    echo "================================================================================"
    echo "Training: $MODEL (${DATASET_NAME})"
    echo "================================================================================"

    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs

    # Build training arguments
    TRAINING_ARGS="--model $MODEL --dataset $DATASET_NAME --gene-list $GENE_LIST --trials $TRIALS --cv $CV_FOLDS --scaling $SCALING --output $OUTPUT_DIR"

    # Add feature selection if specified
    if [ "$FEATURE_SELECTION" != "none" ]; then
        TRAINING_ARGS="$TRAINING_ARGS --feature-selection $FEATURE_SELECTION --select-k $K_FEATURES"
    fi

    # Submit SLURM job
    sbatch --export=ALL,\
TRAINING_SCRIPT="2_model_training/train_traditional_ml.py",\
TRAINING_ARGS="$TRAINING_ARGS" \
           --job-name="tml_${DATASET_NAME}_${MODEL}" \
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
