#!/usr/bin/env bash
################################################################################
# Run Phase 1: Distribution Diagnostic Analysis
################################################################################

BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction"
cd "$BASE_DIR"

echo "================================================================================"
echo "PHASE 1: DISTRIBUTION DIAGNOSTIC ANALYSIS"
echo "================================================================================"
echo ""
echo "This analysis will:"
echo "  1. Load REH, SUP, and all 3 benchmark datasets"
echo "  2. Compare expression value distributions"
echo "  3. Check normalization consistency"
echo "  4. Generate publication-quality plots (300 DPI)"
echo "  5. Create diagnostic report for reviewers"
echo ""
echo "Output directory: distribution_analysis/results/"
echo "================================================================================"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate pytorch

# Run analysis
python distribution_analysis/phase1_distribution_diagnostic.py

echo ""
echo "================================================================================"
echo "ANALYSIS COMPLETE!"
echo "================================================================================"
echo ""
echo "Check results in: distribution_analysis/results/"
echo ""
