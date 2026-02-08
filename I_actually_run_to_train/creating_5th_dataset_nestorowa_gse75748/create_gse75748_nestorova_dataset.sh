#!/usr/bin/env bash
# Create concatenated dataset: GSE75748 (human hPSC) + Nestorova (mouse)

set -e

OUTPUT_DIR="../../1_consensus_labeling/assign/final_training_data_GSE75748_Nestorova"
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "Creating GSE75748 + Nestorova concatenated dataset"
echo "================================================================================"
echo ""
echo "Input datasets:"
echo "  1. GSE75748 (human hPSC): ../../1_consensus_labeling/assign/final_training_data_gse75748/gse75748_training_data.csv"
echo "  2. Nestorova (mouse):     ../../1_consensus_labeling/assign/final_training_data_nestorova/nestorova_training_data.csv"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"

python concatenate_gse75748_nestorova.py

echo ""
echo "================================================================================"
echo "CONCATENATION COMPLETE"
echo "================================================================================"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "================================================================================"
