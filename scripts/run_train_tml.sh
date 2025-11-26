#!/bin/bash
# ==============================================================================
# Traditional ML Model Training Wrapper Script
# ==============================================================================
# Submits SLURM jobs for training traditional ML models using generic template
#
# Usage:
#   ./run_train_tml.sh <model> <dataset> <output_dir> [optional_args]
# ==============================================================================

if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments!"
    echo "Usage: $0 <model> <dataset> <output_dir> [optional_args]"
    echo ""
    echo "Available models: adaboost, random_forest, lgbm, ensemble"
    echo "Available datasets: reh, sup"
    exit 1
fi

MODEL=$1
DATASET=$2
OUTPUT_DIR=$3
shift 3

# Build training arguments
TRAINING_SCRIPT="2_model_training/train_traditional_ml.py"
TRAINING_ARGS="--model $MODEL --dataset $DATASET --output $OUTPUT_DIR $@"

echo "=========================================="
echo "Submitting Traditional ML Training Job"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Additional args: $@"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Submit SLURM job with exported variables
sbatch --export=ALL,TRAINING_SCRIPT="$TRAINING_SCRIPT",TRAINING_ARGS="$TRAINING_ARGS" \
       --job-name="train_${MODEL}_${DATASET}" \
       scripts/train_model_generic.slurm

echo ""
echo "Job submitted! Monitor with: squeue -u $USER"
