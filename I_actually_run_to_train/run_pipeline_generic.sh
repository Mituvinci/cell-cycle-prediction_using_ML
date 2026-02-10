#!/usr/bin/env bash
################################################################################
# Generic Pipeline: Evaluation -> Consolidation -> Visualization
#
# STEP 1: Evaluate all trained models on benchmarks
# STEP 2: Consolidate training + benchmark metrics into one CSV
#
# Usage:
#   bash run_pipeline_generic.sh --model-base models/final_best_k_ft/concate_2ds_human
#   bash run_pipeline_generic.sh --model-base models/final_best_marker_genes/concate_4ds_all_marker --method simple
#   bash run_pipeline_generic.sh --model-base models/final_best_k_ft/concate_mouse --step 1
################################################################################

BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction"
cd "$BASE_DIR"

# Default values
RUN_STEP="all"
METHOD="simple"
BENCHMARKS="SUP GSE146773 GSE64016 Buettner_mESC"
MODEL_BASE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-base)
            MODEL_BASE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --step)
            RUN_STEP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_BASE" ]; then
    echo "ERROR: --model-base is required"
    echo "Usage: bash run_pipeline_generic.sh --model-base models/final_best_k_ft/concate_2ds_human"
    exit 1
fi

# Extract last two folder names from MODEL_BASE for consolidated output name
# Example: models/final_best_k_ft/concate_2ds_human -> final_best_k_ft_concate_2ds_human
dataset_path=$(echo "$MODEL_BASE" | awk -F'/' '{print $(NF-1)"_"$NF}')
CONSOLIDATED_OUTPUT="results/$METHOD/consolidated_${dataset_path}.csv"

echo "================================================================================"
echo "GENERIC PIPELINE CONFIGURATION"
echo "================================================================================"
echo "Model directory     : $MODEL_BASE"
echo "Consolidated output : $CONSOLIDATED_OUTPUT"
echo "Benchmarks          : $BENCHMARKS"
echo "Scaling method      : $METHOD"
echo "Run step            : $RUN_STEP"
echo "================================================================================"
echo ""

################################################################################
# STEP 1: Evaluate all models on all benchmarks
################################################################################

if [ "$RUN_STEP" == "all" ] || [ "$RUN_STEP" == "1" ]; then

echo "================================================================================"
echo "STEP 1: BENCHMARK EVALUATION"
echo "================================================================================"
echo ""

total=0
success=0
failed=0
failed_files=""

for model_dir in "$MODEL_BASE"/*/; do
    [ -d "$model_dir" ] || continue
    model_name=$(basename "$model_dir")
    echo "--- $model_name ---"

    for model_file in "$model_dir"*_fld_*.pt "$model_dir"*_fld_*.joblib; do
        [ -e "$model_file" ] || continue
        total=$((total + 1))
        basename_file=$(basename "$model_file")

        echo "  [$total] $basename_file"

        python 3_evaluation/evaluate_models.py \
            --model_path "$model_file" \
            --benchmarks $BENCHMARKS \
            --scaling_method $METHOD

        if [ $? -eq 0 ]; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
            failed_files="$failed_files  $basename_file\n"
            echo "  FAILED: $basename_file"
        fi
    done
    echo ""
done

echo "================================================================================"
echo "STEP 1 SUMMARY"
echo "================================================================================"
echo "  Total   : $total"
echo "  Success : $success"
echo "  Failed  : $failed"

if [ $failed -gt 0 ]; then
    echo ""
    echo "  Failed files:"
    echo -e "$failed_files"
    echo "WARNING: $failed evaluations failed. Fix before running STEP 2."
    exit 1
fi

echo "================================================================================"
echo ""

fi

################################################################################
# STEP 2: Consolidate training + benchmark results
################################################################################

if [ "$RUN_STEP" == "all" ] || [ "$RUN_STEP" == "2" ]; then

echo "================================================================================"
echo "STEP 2: CONSOLIDATION"
echo "================================================================================"
echo "Input  : $MODEL_BASE"
echo "Output : $CONSOLIDATED_OUTPUT"
echo "================================================================================"
echo ""

python 3_evaluation/consolidate_training_and_benchmark.py \
    --input "$MODEL_BASE" \
    --output "$CONSOLIDATED_OUTPUT" \
    --species all

if [ $? -ne 0 ]; then
    echo "ERROR: Consolidation failed. Check output above."
    exit 1
fi

echo ""
echo "STEP 2 DONE: $CONSOLIDATED_OUTPUT"
echo "================================================================================"
echo ""

fi

echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo "Model directory     : $MODEL_BASE"
echo "Consolidated output : $CONSOLIDATED_OUTPUT"
echo "================================================================================"
