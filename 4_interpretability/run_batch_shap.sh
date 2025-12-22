#!/usr/bin/env bash
################################################################################
# Batch SHAP Analysis Script
#
# Recursively finds all models in a directory and runs SHAP analysis
# Supports cross-species analysis (human models on mouse data)
#
# Usage:
#   bash run_batch_shap.sh \
#       --model_dir models/submitted_human_models \
#       --benchmark Buettner_mESC \
#       --output results/shap_cross_species
################################################################################

set -e

# Default values
MODEL_DIR=""
BENCHMARK="Buettner_mESC"
OUTPUT_DIR="results/shap_batch"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash run_batch_shap.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_dir DIR     Root directory containing models (REQUIRED)"
            echo "  --benchmark NAME    Benchmark dataset (default: Buettner_mESC)"
            echo "                      Choices: GSE146773, GSE64016, SUP, Buettner_mESC"
            echo "  --output DIR        Output directory (default: results/shap_batch)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Cross-species: Human models on Mouse data"
            echo "  bash run_batch_shap.sh \\"
            echo "      --model_dir models/submitted_human_models \\"
            echo "      --benchmark Buettner_mESC \\"
            echo "      --output results/shap_cross_species"
            echo ""
            echo "  # Single species: Human models on Human data"
            echo "  bash run_batch_shap.sh \\"
            echo "      --model_dir models/robust_human_hpsc \\"
            echo "      --benchmark GSE146773 \\"
            echo "      --output results/shap_gse146773"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: --model_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo "================================================================================"
echo "BATCH SHAP ANALYSIS"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Model directory: $MODEL_DIR"
echo "  Benchmark dataset: $BENCHMARK"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all model files recursively
MODEL_FILES=()

echo "Searching for model files in: $MODEL_DIR"
echo ""

# Find .pt files (Deep Learning)
while IFS= read -r -d '' file; do
    MODEL_FILES+=("$file")
done < <(find "$MODEL_DIR" -type f -name "*.pt" -print0)

# Find .pkl files (Traditional ML) - exclude scalers and label encoders
while IFS= read -r -d '' file; do
    # Skip scaler and label encoder files
    if [[ ! "$file" =~ _scl\.pkl$ ]] && [[ ! "$file" =~ _le\.pkl$ ]]; then
        MODEL_FILES+=("$file")
    fi
done < <(find "$MODEL_DIR" -type f -name "*.pkl" -print0)

# Find .joblib files (Traditional ML) - exclude scalers and label encoders
while IFS= read -r -d '' file; do
    # Skip scaler and label encoder files
    if [[ ! "$file" =~ _scl\.joblib$ ]] && [[ ! "$file" =~ _le\.joblib$ ]]; then
        MODEL_FILES+=("$file")
    fi
done < <(find "$MODEL_DIR" -type f -name "*.joblib" -print0)

# Check if any models found
if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "ERROR: No model files found in $MODEL_DIR"
    echo "Searched for: *.pt, *.pkl, *.joblib"
    exit 1
fi

echo "Found ${#MODEL_FILES[@]} model(s):"
for model in "${MODEL_FILES[@]}"; do
    echo "  - $(basename $model)"
done
echo ""

# Process each model
TOTAL=${#MODEL_FILES[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0

for MODEL_PATH in "${MODEL_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))

    # Extract model name (without extension)
    MODEL_BASENAME=$(basename "$MODEL_PATH")
    MODEL_NAME="${MODEL_BASENAME%.*}"

    # Create model-specific output directory
    MODEL_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_OUTPUT_DIR"

    echo "================================================================================"
    echo "[$CURRENT/$TOTAL] Processing: $MODEL_NAME"
    echo "================================================================================"
    echo "Model path: $MODEL_PATH"
    echo "Output dir: $MODEL_OUTPUT_DIR"
    echo ""

    # Build command with optional cross-species flag
    SHAP_CMD="python run_shap_analysis.py --model_path \"$MODEL_PATH\" --benchmark \"$BENCHMARK\" --output_dir \"$MODEL_OUTPUT_DIR\""

    # Auto-enable cross-species mode for Buettner_mESC (mouse data)
    if [ "$BENCHMARK" = "Buettner_mESC" ]; then
        echo "[CROSS-SPECIES] Human model on Mouse data - enabling zero imputation"
        SHAP_CMD="$SHAP_CMD --cross_species"
    fi

    # Run SHAP analysis
    if eval $SHAP_CMD; then

        echo ""
        echo "SUCCESS: SHAP analysis completed for $MODEL_NAME"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo ""
        echo "FAILED: SHAP analysis failed for $MODEL_NAME"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# Final summary
echo "================================================================================"
echo "BATCH SHAP ANALYSIS COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  Total models processed: $TOTAL"
echo "  Successful: $SUCCESSFUL"
echo "  Failed: $FAILED"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output structure:"
echo "  $OUTPUT_DIR/"
for MODEL_PATH in "${MODEL_FILES[@]}"; do
    MODEL_BASENAME=$(basename "$MODEL_PATH")
    MODEL_NAME="${MODEL_BASENAME%.*}"
    echo "    ├── $MODEL_NAME/"
    echo "    │   ├── ${MODEL_NAME}_shap_summary.png"
    echo "    │   ├── ${MODEL_NAME}_SHAP.txt"
    echo "    │   └── ${MODEL_NAME}_top_features.csv"
done
echo ""
echo "================================================================================"
