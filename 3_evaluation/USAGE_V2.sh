#!/bin/bash
################################################################################
# PROPER BENCHMARK EVALUATION WORKFLOW (V2)
# No double normalization!
################################################################################

# =============================================================================
# STEP 1: Preprocess raw benchmarks (ONE-TIME SETUP)
# =============================================================================

echo "STEP 1: Preprocessing raw benchmarks..."
echo "This applies Seurat log-normalization (SAME AS TRAINING)"
echo "Does NOT apply StandardScaler (evaluation will use trained scaler)"

conda activate pytorch

# Preprocess all benchmarks
python preprocess_benchmarks.py --benchmark all

# Or preprocess individually:
# python preprocess_benchmarks.py --benchmark GSE146773
# python preprocess_benchmarks.py --benchmark GSE64016
# python preprocess_benchmarks.py --benchmark Buettner_mESC

# =============================================================================
# STEP 2: Evaluate models on preprocessed benchmarks
# =============================================================================

echo ""
echo "STEP 2: Evaluating models..."
echo "This applies TRAINED scaler (no refitting), then predicts"

# Example: Evaluate single model fold
python evaluate_models_v2.py \
  --model_dir ../models/final_best/concate_2ds_human/simpledense \
  --fold 0 \
  --benchmark GSE146773 \
  --model_type simpledense \
  --output results_v2/concate_2ds_human_simpledense_fold0_GSE146773.csv

# Example: Evaluate all folds for a model
for FOLD in 0 1 2; do
  python evaluate_models_v2.py \
    --model_dir ../models/final_best/concate_2ds_human/simpledense \
    --fold $FOLD \
    --benchmark GSE146773 \
    --model_type simpledense \
    --output results_v2/concate_2ds_human_simpledense_fold${FOLD}_GSE146773.csv
done

# Example: Evaluate on all benchmarks
for BENCHMARK in GSE146773 GSE64016 Buettner_mESC; do
  python evaluate_models_v2.py \
    --model_dir ../models/final_best/concate_2ds_human/simpledense \
    --fold 0 \
    --benchmark $BENCHMARK \
    --model_type simpledense \
    --output results_v2/concate_2ds_human_simpledense_fold0_${BENCHMARK}.csv
done

echo ""
echo "DONE!"
echo "Results saved to: results_v2/"
