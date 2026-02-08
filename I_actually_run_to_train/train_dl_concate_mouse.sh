#!/usr/bin/env bash
################################################################################
# Train ALL Deep Learning Models on Concatenated Mouse Datasets
# Datasets: Mouse Brain + Nestorova
# Uses merged training data: expression matrix + consensus labels
# Uses pre-computed gene list from concatenation
################################################################################

set -e

echo "================================================================================"
echo "TRAINING ALL DEEP LEARNING MODELS ON CONCATENATED MOUSE DATASETS"
echo "================================================================================"
echo ""

# Configuration
TRAINING_DATA="1_consensus_labeling/assign/final_training_data_concate_mouse_brain_Nestorova/concatenated_training_data.csv"
GENE_LIST="1_consensus_labeling/assign/final_training_data_concate_mouse_brain_Nestorova/gene_list.txt"
OUTPUT_BASE="models/concate_mouse"
TRIALS=5
CV_FOLDS=5
SCALING="robust"

# Check if training data exists
if [ ! -f "$TRAINING_DATA" ]; then
    echo "ERROR: Training data not found: $TRAINING_DATA"
    echo "Please run: python concatenate_training_datasets.py"
    exit 1
fi

# Check if gene list exists
if [ ! -f "$GENE_LIST" ]; then
    echo "ERROR: Gene list not found: $GENE_LIST"
    echo "Please run: python concatenate_training_datasets.py"
    exit 1
fi

echo "Training data: $TRAINING_DATA"
echo "Gene list: $GENE_LIST"
echo "Output directory: $OUTPUT_BASE"
echo "Optuna trials: $TRIALS (TEST MODE)"
echo "CV folds: $CV_FOLDS"
echo "Scaling: $SCALING"
echo ""
echo "NOTE: TRIALS=5 for quick testing. Increase to 60-200 for production."
echo ""

# Count genes and cells
GENE_COUNT=$(wc -l < "$GENE_LIST")
CELL_COUNT=$(tail -n +2 "$TRAINING_DATA" | wc -l)
echo "Dataset statistics:"
echo "  Cells: $CELL_COUNT"
echo "  Genes: $GENE_COUNT"
echo ""

# Train all DL models
MODELS=("simpledense" "cnn" "hbdcnn" "fe" "enhancedense")

for MODEL in "${MODELS[@]}"; do
    echo "================================================================================"
    echo "Training: $MODEL (Concatenated Mouse Datasets)"
    echo "================================================================================"

    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs

    # Submit SLURM job
    # Using --gene-list for pre-computed gene list, --scaling for scaler
    sbatch --export=ALL,\
TRAINING_SCRIPT="2_model_training/train_deep_learning.py",\
TRAINING_ARGS="--model $MODEL --dataset concate_mouse --gene-list $GENE_LIST --trials $TRIALS --cv $CV_FOLDS --scaling $SCALING --output $OUTPUT_DIR" \
           --job-name="dl_mouse_${MODEL}" \
           scripts/train_model_generic.slurm

    echo "Job submitted for $MODEL"
    echo ""
    sleep 2
done

echo "================================================================================"
echo "ALL DEEP LEARNING JOBS SUBMITTED (CONCATENATED MOUSE DATASETS)"
echo "================================================================================"
echo ""
echo "Models: ${MODELS[@]}"
echo "Monitor with: squeue -u $USER"
echo "Check logs: ls -lh logs/"
echo ""
echo "After training completes, run benchmark evaluation:"
echo "  cd 3_evaluation"
echo "  python consolidate_training_and_benchmark.py --input ../models/concate_mouse --output results/concate_mouse.csv"
echo ""
echo "================================================================================"
