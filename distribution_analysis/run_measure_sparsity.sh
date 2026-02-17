#!/bin/bash

# Measure sparsity for all training datasets

echo "========================================"
echo "Measuring Training Data Sparsity"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate pytorch

# Run Python script
python distribution_analysis/measure_training_sparsity.py

echo ""
echo "Done! Results saved to:"
echo "  distribution_analysis/results/training_sparsity_summary.csv"
