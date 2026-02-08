#!/usr/bin/env bash
################################################################################
# Train ALL Deep Learning Models on Human PBMC Data
# Uses merged training data: expression matrix + consensus labels
# Uses pre-computed gene list from pbmc_3benchmark_sup.txt
################################################################################

set -e

echo "================================================================================"
echo "TRAINING ALL DEEP LEARNING MODELS ON HUMAN PBMC DATA"
echo "================================================================================"
echo ""

# Configuration
TRAINING_DATA="1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv"
GENE_LIST="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/gene_lists/pbmc_3benchmark_sup.txt"
OUTPUT_BASE="models/human_pbmc"
TRIALS=60
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

# Train all DL models
MODELS=("simpledense" "cnn" "hbdcnn" "fe" "enhancedense")


for MODEL in "${MODELS[@]}"; do
    echo "================================================================================"
    echo "Training: $MODEL (Human PBMC)"
    echo "================================================================================"

    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs

    # Submit SLURM job
    # Using --gene-list for pre-computed gene list (load_and_preprocess_data_v3)
    sbatch --export=ALL,\
TRAINING_SCRIPT="2_model_training/train_deep_learning.py",\
TRAINING_ARGS="--model $MODEL --dataset pbmc --gene-list $GENE_LIST --trials $TRIALS --cv $CV_FOLDS --scaling $SCALING  --output $OUTPUT_DIR" \
           --job-name="dl_human_${MODEL}" \
           scripts/train_model_generic.slurm

    echo "Job submitted for $MODEL"
    echo ""
    sleep 2  # Avoid overwhelming scheduler
done

echo "================================================================================"
echo "ALL DEEP LEARNING JOBS SUBMITTED"
echo "================================================================================"
echo ""
echo "Models: ${MODELS[@]}"
echo "Monitor with: squeue -u $USER"
echo "Check logs: ls -lh logs/"
echo ""
echo "================================================================================"
