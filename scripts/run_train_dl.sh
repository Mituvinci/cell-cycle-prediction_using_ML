#!/bin/bash
# ==============================================================================
# Deep Learning Model Training Wrapper Script
# ==============================================================================
# This script submits SLURM jobs for training deep learning models
#
# Usage:
#   ./run_train_dl.sh <model> <dataset> <output_dir> [optional_args]
#
# Examples:
#   # Train CNN on SUP with default settings
#   ./run_train_dl.sh cnn sup ./models/CNN/
#
#   # Train DNN3 on REH with custom Optuna trials
#   ./run_train_dl.sh simpledense reh ./models/DNN3/ --trials 50 --epochs-end 2000
#
#   # Train with feature selection
#   ./run_train_dl.sh deepdense sup ./models/DNN5/ --feature-selection ElasticCV
#
# Available models: simpledense, deepdense, cnn, hbdcnn, fe
# ==============================================================================

# Check minimum arguments
if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments!"
    echo ""
    echo "Usage: $0 <model> <dataset> <output_dir> [optional_args]"
    echo ""
    echo "Examples:"
    echo "  $0 cnn sup ./models/CNN/"
    echo "  $0 simpledense reh ./models/DNN3/ --trials 50"
    echo ""
    echo "Available models: simpledense, deepdense, cnn, hbdcnn, fe"
    echo "Available datasets: reh, sup"
    exit 1
fi

# Parse required arguments
MODEL=$1
DATASET=$2
OUTPUT_DIR=$3
shift 3  # Remove first 3 args, leaving optional args

# Build training arguments
TRAINING_SCRIPT="2_model_training/train_deep_learning.py"
TRAINING_ARGS="--model $MODEL --dataset $DATASET --output $OUTPUT_DIR $@"

echo "=========================================="
echo "Submitting Deep Learning Training Job"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Additional args: $@"
echo "=========================================="
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Submit SLURM job with exported variables
sbatch --export=ALL,TRAINING_SCRIPT="$TRAINING_SCRIPT",TRAINING_ARGS="$TRAINING_ARGS" \
       --job-name="train_${MODEL}_${DATASET}" \
       scripts/train_model_generic.slurm

echo ""
echo "Job submitted! Check logs/ directory for output."
echo "Monitor with: squeue -u $USER"
