#!/usr/bin/env bash
################################################################################
# run_all_datasets.sh
#
# Runs detect_cell_cycle_effect.R on all 7 datasets.
# Each dataset gets its own output directory inside pca_results/.
# Output files are prefixed with the dataset name.
#
# Usage:
#   bash run_all_datasets.sh          # run all 7 datasets
#   bash run_all_datasets.sh REH      # run only REH
################################################################################

SCRIPT_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/Has_cell_cycle_effect_or_not"
DATA_BASE="$SCRIPT_DIR/scRNA_data"
OUTPUT_BASE="$SCRIPT_DIR/pca_results_df"

# MKL shared library is in the conda package cache, not the environment lib.
# Rscript needs this path to load spam.so (Seurat dependency).
export LD_LIBRARY_PATH="/users/ha00014/.conda/pkgs/mkl-2025.3.0-h0e700b2_463/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$SCRIPT_DIR"

################################################################################
# Helper function
################################################################################
run_dataset() {
    local input_path="$1"
    local organism="$2"
    local dataset_name="$3"
    local output_dir="$OUTPUT_BASE/$dataset_name"

    echo "================================================================================"
    echo "  Processing: $dataset_name  (organism: $organism)"
    echo "================================================================================"
    echo "  Input : $input_path"
    echo "  Output: $output_dir"
    echo ""

    Rscript "$SCRIPT_DIR/pca_cell_cycle_from_scoring.R" \
        "$input_path" \
        "$output_dir" \
        "$organism" \
        "$dataset_name"

    if [ $? -ne 0 ]; then
        echo ""
        echo "  FAILED: $dataset_name"
        echo "================================================================================"
        return 1
    fi

    return 0
}

################################################################################
# Dataset definitions
################################################################################
# If a specific dataset name is passed as argument, run only that one.
# Otherwise run all 7.

FILTER="${1:-all}"

total=0
success=0
failed=0

run_if_selected() {
    local dataset_name="$3"
    if [ "$FILTER" == "all" ] || [ "$FILTER" == "$dataset_name" ]; then
        total=$((total + 1))
        run_dataset "$1" "$2" "$3"
        if [ $? -eq 0 ]; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    fi
}

# 1. Mouse brain (10x, mouse)
run_if_selected \
    "$DATA_BASE/10-k-brain-cells_healthy_mouse/filtered_feature_bc_matrix" \
    "mouse" \
    "mouse_brain"

# 2. REH (10x, human)
run_if_selected \
    "$DATA_BASE/1_GD428_21136_Hu_REH_Parental/filtered_feature_bc_matrix" \
    "human" \
    "REH"

# 3. SUP-B15 (10x, human)
run_if_selected \
    "$DATA_BASE/2_GD444_21136_Hu_Sup_Parental/filtered_feature_bc_matrix" \
    "human" \
    "SUP_B15"

# 4. GSE64016 (CSV, human)
run_if_selected \
    "$DATA_BASE/3_GSE64016_H1andFUCCI_normalized_EC_original/GSE64016_H1andFUCCI_normalized_EC.csv" \
    "human" \
    "GSE64016"

# 5. GSE146773 (10x, human)
run_if_selected \
    "$DATA_BASE/GSE146773/filtered_feature_bc_matrix" \
    "human" \
    "GSE146773"

# 6. GSE75748 (CSV, human)
run_if_selected \
    "$DATA_BASE/GSE75748/GSE75748_sc_cell_type_ec.csv" \
    "human" \
    "GSE75748"

# 7. Nestorova (TXT tab-delimited, mouse)
run_if_selected \
    "$DATA_BASE/nestorova/nestorawa_forcellcycle_expressionMatrix.txt" \
    "mouse" \
    "nestorova"

#8 healthy human
run_if_selected \
    "$DATA_BASE/pbmc_healthy_human/filtered_feature_bc_matrix" \
    "human" \
    "pbmc_healthy_human"


################################################################################
# Summary
################################################################################
echo "================================================================================"
echo "  ALL DONE"
echo "================================================================================"
echo "  Total   : $total"
echo "  Success : $success"
echo "  Failed  : $failed"
echo ""
echo "  Results : $OUTPUT_BASE"
echo "  Each folder contains:"
echo "    {name}_pca_cell_cycle_phase.png   -- PCA colored by G1 / S / G2M"
echo "    {name}_pca_cell_cycle_scores.png  -- PCA colored by S.Score / G2M.Score"
echo "    {name}_cell_cycle_summary.json    -- marker counts, PC1 correlations, verdict"
echo "================================================================================"
