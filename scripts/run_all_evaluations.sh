#!/usr/bin/env bash
################################################################################
# Comprehensive Model Evaluation Script - Array-Based
#
# Evaluates ALL trained models on ALL benchmarks:
# - GSE146773 (human)
# - GSE64016 (human)
# - Buettner_mESC (mouse)
#
# Automatically finds all _fld_*.pt and _fld_*.joblib files in each directory
################################################################################

set -e

BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction"
cd "$BASE_DIR"

echo "================================================================================"
echo "COMPREHENSIVE MODEL EVALUATION ON ALL BENCHMARKS"
echo "================================================================================"
echo ""

# Create output directories
mkdir -p results/evaluations/mouse_brain
mkdir -p results/evaluations/human_pbmc
mkdir -p results/evaluations/human_hpsc


# Benchmarks to evaluate
BENCHMARKS="SUP GSE146773 GSE64016 Buettner_mESC"


################################################################################
# Function to evaluate all models in a directory
################################################################################

evaluate_models_in_dir() {
    local model_dir=$1
    local species=$2
    local model_name=$(basename "$model_dir")

    echo "--- ${species} ${model_name} ---"

    # Check if directory exists
    if [ ! -d "$model_dir" ]; then
        echo "  WARNING: Directory not found: $model_dir"
        return
    fi

    # Count models
    local model_count=$(find "$model_dir" -name "*_fld_*.pt" -o -name "*_fld_*.joblib" 2>/dev/null | wc -l)

    if [ "$model_count" -eq 0 ]; then
        echo "  WARNING: No model files found in $model_dir"
        return
    fi

    echo "  Found $model_count model files"

    # Process each model file
    local fold_num=0
    for model_file in "$model_dir"/*_fld_*.pt "$model_dir"/*_fld_*.joblib; do
        [ -e "$model_file" ] || continue  # Skip if glob doesn't match

        fold_num=$((fold_num + 1))
        local basename_file=$(basename "$model_file")

        # Extract fold number from filename
        if [[ "$basename_file" =~ _fld_([0-9]+)\. ]]; then
            local fold="${BASH_REMATCH[1]}"
        else
            local fold="$fold_num"
        fi

        echo "    Fold $fold: $basename_file"

        # Run evaluation
        python 3_evaluation/evaluate_models.py \
            --model_path "$model_file" \
            --benchmarks $BENCHMARKS  \
	    --scaling_method double	

        if [ $? -eq 0 ]; then
            echo "      ✓ Evaluation complete"
        else
            echo "      ✗ Evaluation FAILED"
        fi
    done

    echo ""
}


################################################################################
# Define all model directories
################################################################################


# Base path provided by you
human_dir="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/models/human_hpsc_5770_standard"

# Initialize array
HUMAN_MODEL_DIRS=()

# Automatically fill array with all subfolders inside BASE_DIR
for folder in "$human_dir"/*/; do
    [ -d "$folder" ] || continue
    HUMAN_MODEL_DIRS+=("$folder")
done


#echo "${HUMAN_MODEL_DIRS[@]}"


total_human=0
for model_dir in "${HUMAN_MODEL_DIRS[@]}"; do
    evaluate_models_in_dir "$model_dir" "human"
    count=$(find "$model_dir" -name "*_fld_*.pt" -o -name "*_fld_*.joblib" 2>/dev/null | wc -l)
    total_human=$((total_human + count))
done

echo "✓ Human models complete! ($total_human models evaluated)"
echo ""


# Base path provided by you
mouse_dir="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/models/mouse_brain"

# Initialize array
MOUSE_MODEL_DIRS=()

# Automatically fill array with all subfolders inside BASE_DIR
for folder in "$mouse_dir"/*/; do
    [ -d "$folder" ] || continue
    MOUSE_MODEL_DIRS+=("$folder")
done

#echo "${MOUSE_MODEL_DIRS[@]}"


total_mouse=0
for model_dir in "${MOUSE_MODEL_DIRS[@]}"; do
    #evaluate_models_in_dir "$model_dir" "mouse"
    count=$(find "$model_dir" -name "*_fld_*.pt" -o -name "*_fld_*.joblib" 2>/dev/null | wc -l)
    total_mouse=$((total_mouse + count))
done

echo "✓ Mouse models complete! ($total_mouse models evaluated)"
echo ""



new_REH_HUMAN_MODEL_DIRS=()
new_reh_human="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/models/reh_7ds"

# Automatically fill array with all subfolders inside BASE_DIR
for folder in "$new_reh_human"/*/; do
    [ -d "$folder" ] || continue
    new_REH_HUMAN_MODEL_DIRS+=("$folder")
done


#echo "${new_REH_HUMAN_MODEL_DIRS[@]}"

total_new_reh_human=0
for model_dir in "${new_REH_HUMAN_MODEL_DIRS[@]}"; do
    #evaluate_models_in_dir "$model_dir" "new_reh_models"
    count=$(find "$model_dir" -name "*_fld_*.pt" -o -name "*_fld_*.joblib" 2>/dev/null | wc -l)
    total_new_reh_human=$((total_new_reh_human + count))
done


# Concatenated 4 datasets (all 4 - cross-species)
CONCATE_4DS_DIRS=()
concate_4ds_dir="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/models/concate_4ds_all"

for folder in "$concate_4ds_dir"/*/; do
    [ -d "$folder" ] || continue
    CONCATE_4DS_DIRS+=("$folder")
done

total_concate_4ds=0
for model_dir in "${CONCATE_4DS_DIRS[@]}"; do
    evaluate_models_in_dir "$model_dir" "concate_4ds_all"
    count=$(find "$model_dir" -name "*_fld_*.pt" -o -name "*_fld_*.joblib" 2>/dev/null | wc -l)
    total_concate_4ds=$((total_concate_4ds + count))
done

echo "✓ Concatenated 4DS models complete! ($total_concate_4ds models evaluated)"
echo ""


# Concatenated 2 datasets (human only)
CONCATE_2DS_HUMAN_DIRS=()
concate_2ds_human_dir="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/models/concate_2ds_human"

for folder in "$concate_2ds_human_dir"/*/; do
    [ -d "$folder" ] || continue
    CONCATE_2DS_HUMAN_DIRS+=("$folder")
done

total_concate_2ds_human=0
for model_dir in "${CONCATE_2DS_HUMAN_DIRS[@]}"; do
    evaluate_models_in_dir "$model_dir" "concate_2ds_human"
    count=$(find "$model_dir" -name "*_fld_*.pt" -o -name "*_fld_*.joblib" 2>/dev/null | wc -l)
    total_concate_2ds_human=$((total_concate_2ds_human + count))
done

echo "✓ Concatenated 2DS Human models complete! ($total_concate_2ds_human models evaluated)"
echo ""


# Concatenated mouse datasets
CONCATE_MOUSE_DIRS=()
concate_mouse_dir="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/models/concate_mouse"

for folder in "$concate_mouse_dir"/*/; do
    [ -d "$folder" ] || continue
    CONCATE_MOUSE_DIRS+=("$folder")
done

total_concate_mouse=0
for model_dir in "${CONCATE_MOUSE_DIRS[@]}"; do
    evaluate_models_in_dir "$model_dir" "concate_mouse"
    count=$(find "$model_dir" -name "*_fld_*.pt" -o -name "*_fld_*.joblib" 2>/dev/null | wc -l)
    total_concate_mouse=$((total_concate_mouse + count))
done

echo "✓ Concatenated Mouse models complete! ($total_concate_mouse models evaluated)"
echo ""


################################################################################
# SUMMARY
################################################################################

echo "================================================================================"
echo "ALL EVALUATIONS COMPLETE!"
echo "================================================================================"
echo ""
echo "Total evaluations:"
echo "  - Human models: $total_human models × 4 benchmarks"
echo "  - Mouse models: $total_mouse models × 4 benchmarks"
echo "  - New REH models: $total_new_reh_human models × 4 benchmarks"
echo "  - Concatenated 4DS models: $total_concate_4ds models × 4 benchmarks"
echo "  - Concatenated 2DS Human models: $total_concate_2ds_human models × 4 benchmarks"
echo "  - Concatenated Mouse models: $total_concate_mouse models × 4 benchmarks"
echo "  - Grand total: $((total_human + total_mouse + total_new_reh_human + total_concate_4ds + total_concate_2ds_human + total_concate_mouse)) model files evaluated"
echo ""
echo ""
echo "Next step: Run consolidation script to create master CSV files"
echo "  python consolidate_training_and_benchmark.py"
echo ""
echo "================================================================================"
