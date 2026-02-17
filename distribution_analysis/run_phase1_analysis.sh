#!/bin/bash

# Usage: bash run_phase1_analysis.sh <training_dataset>
# Example: bash run_phase1_analysis.sh reh

if [ $# -eq 0 ]; then
    echo "ERROR: No training dataset specified"
    echo "Usage: bash run_phase1_analysis.sh <training_dataset>"
    echo "Available: reh, sup, pbmc, mouse_brain, nestorova, hpsc"
    exit 1
fi

TRAINING_DATASET=$1

echo "Running distribution analysis for: $TRAINING_DATASET"

source ~/.bashrc
conda activate pytorch

python distribution_analysis/phase1_distribution_diagnostic_only_sparsity.py $TRAINING_DATASET

echo ""
echo "Done! Results saved to: distribution_analysis/result_${TRAINING_DATASET}/"
