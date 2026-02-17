#!/bin/bash

# Run consensus validation on all 3 benchmarks
# This validates that the consensus approach is reliable by comparing
# consensus predictions vs ground truth labels

echo "=============================================================================="
echo "Validating Consensus Labeling Approach on Benchmarks"
echo "=============================================================================="

cd /users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling

# Run validation (requires at least 3 tools to agree)
python validate_consensus_on_benchmarks.py \
    --benchmarks gse146773 gse64016 buettner_mesc \
    --min_agreement 3 \
    --output_dir consensus_validation_results

echo ""
echo "=============================================================================="
echo "VALIDATION COMPLETE!"
echo "=============================================================================="
echo ""
echo "Results saved to: consensus_validation_results/"
echo ""
echo "Key files:"
echo "  - validation_summary_table.csv (Table for manuscript)"
echo "  - gse146773/gse146773_consensus_vs_groundtruth.csv"
echo "  - gse64016/gse64016_consensus_vs_groundtruth.csv"
echo "  - buettner_mesc/buettner_mesc_consensus_vs_groundtruth.csv"
echo ""
echo "=============================================================================="
