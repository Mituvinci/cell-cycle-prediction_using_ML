#!/usr/bin/env bash
################################################################################
# Run Cell Cycle Scoring on All Consensus Labeled Datasets
################################################################################
#
# Purpose:
#   Runs pca_with_scoring_from_csv.R on all consensus labeled training datasets
#   to verify cell cycle effects and compare Seurat scoring with consensus labels
#
# Usage:
#   bash run_scoring_on_consensus_labeled.sh              # Run all datasets
#   bash run_scoring_on_consensus_labeled.sh pbmc_human   # Run single dataset
#
################################################################################

set -e

# Fix library path for MKL on DOLLY SODS cluster
export LD_LIBRARY_PATH=/shared/software/conda/envs/intelpython_310/lib:$LD_LIBRARY_PATH

echo "================================================================================"
echo "CELL CYCLE SCORING ON CONSENSUS LABELED DATASETS"
echo "================================================================================"
echo ""

# Define datasets to process
DATASETS=(
  "pbmc_human"
  "mouse_brain"
  "nestorova"
  "gse75748_hpsc"
  "reh"
  "sup_b15"
)

# If argument provided, run only that dataset
if [ $# -eq 1 ]; then
  DATASETS=("$1")
  echo "Running single dataset: $1"
  echo ""
fi

# Output directory
OUTPUT_BASE="pca_results_consensus_labeled"
mkdir -p "$OUTPUT_BASE"

# Count datasets
TOTAL=${#DATASETS[@]}
CURRENT=0

echo "Datasets to process: ${TOTAL}"
echo ""

# Process each dataset
for DATASET in "${DATASETS[@]}"; do
  CURRENT=$((CURRENT + 1))

  echo "================================================================================"
  echo "[$CURRENT/$TOTAL] Processing: $DATASET"
  echo "================================================================================"

  # Run R script
  Rscript pca_with_scoring_from_csv.R "$DATASET"

  if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS: $DATASET completed"
    echo ""
  else
    echo ""
    echo "ERROR: $DATASET failed"
    echo ""
  fi

  # Small delay between datasets
  sleep 2
done

echo "================================================================================"
echo "ALL DATASETS PROCESSED"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "Summary of outputs:"
for DATASET in "${DATASETS[@]}"; do
  if [ -d "$OUTPUT_BASE/$DATASET" ]; then
    echo "  $DATASET:"
    ls -lh "$OUTPUT_BASE/$DATASET/"
    echo ""
  fi
done

echo "================================================================================"
echo "QUICK RESULTS SUMMARY"
echo "================================================================================"
echo ""

# Display cell cycle effect results from JSON files
for DATASET in "${DATASETS[@]}"; do
  JSON_FILE="$OUTPUT_BASE/$DATASET/${DATASET}_summary.json"
  if [ -f "$JSON_FILE" ]; then
    EFFECT=$(grep -o '"cell_cycle_effect": [a-z]*' "$JSON_FILE" | awk '{print $2}')
    CELLS=$(grep -o '"cells": [0-9]*' "$JSON_FILE" | awk '{print $2}')
    COR_G2M=$(grep -o '"pc1_cor_g2m_score": -\?[0-9.]*' "$JSON_FILE" | awk '{print $2}')
    echo "$DATASET: cells=$CELLS, effect=$EFFECT, PC1_cor_G2M=$COR_G2M"
  fi
done

echo ""
echo "Done!"
echo ""
