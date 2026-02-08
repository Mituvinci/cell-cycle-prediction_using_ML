#!/usr/bin/env bash
################################################################################
# Train ALL Deep Learning Models on Human hPSCData
# Uses merged training data: expression matrix + consensus labels
# Uses pre-computed gene list from hPSC_3benchmark_sup.txt
################################################################################

set -e

echo "================================================================================"
echo "TRAINING ALL DEEP LEARNING MODELS ON HUMAN hPCSS DATA"
echo "================================================================================"
echo ""

# Configuration
TRAINING_DATA="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/GSE75748_hPSC_final_training_matrix.csv"
GENE_LIST="gene_lists/hpsc_3benchmark_sup.txt"
OUTPUT_BASE="models/human_hpsc_5770_standard"
TRIALS=200
CV_FOLDS=5
SELECT_K=5770
SCALING="standard"

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
echo "SelectKBest (k): $SELECT_K"
echo "Scaling: $SCALING"
echo ""

# Train all DL models
MODELS=("simpledense" "cnn" "hbdcnn" "fe" "enhancedense")


for MODEL in "${MODELS[@]}"; do
    echo "================================================================================"
    echo "Training: $MODEL (Human hPCSS)"
    echo "================================================================================"

    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs

    # Submit SLURM job
    # Using --gene-list for pre-computed gene list, --select-k for SelectKBest, --scaling for scaler
    sbatch --export=ALL,\
TRAINING_SCRIPT="2_model_training/train_deep_learning.py",\
TRAINING_ARGS="--model $MODEL --dataset hpsc --gene-list $GENE_LIST --trials $TRIALS --cv $CV_FOLDS --select-k $SELECT_K --scaling $SCALING --output $OUTPUT_DIR" \
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
