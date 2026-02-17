#!/bin/bash
#SBATCH --job-name=shap_4analyses
#SBATCH --partition=inter_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/shap_4analyses_%j.out
#SBATCH --error=logs/shap_4analyses_%j.err

# ============================================================================
# SLURM Script: Run 4 SHAP Analyses
# ============================================================================
# Purpose: Run SHAP interpretability analysis for:
#   1. DNN5 (REH) on GSE146773 (human-human)
#   2. DNN5 (REH) on Buettner_mESC (human-mouse cross-species)
#   3. DNN3 (Nestorowa) on GSE146773 (mouse-human cross-species)
#   4. DNN3 (Nestorowa) on Buettner_mESC (mouse-mouse)
# ============================================================================

set -e  # Exit on error

echo "========================================================================"
echo "SHAP Analysis - 4 Experiments"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load conda environment
echo "Loading conda environment: pytorch"
source ~/.bashrc
conda activate pytorch

# Fix libstdc++ version issue - use conda's libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# Check CUDA availability
echo "Checking CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Create logs directory
mkdir -p logs

# Base paths
BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction"
INTERP_DIR="$BASE_DIR/4_interpretability"
MODEL_DIR="$BASE_DIR/models"
TRAINING_DIR="$BASE_DIR/1_consensus_labeling/assign"

cd $INTERP_DIR

# ============================================================================
# ANALYSIS 1: DNN5 (REH) → GSE146773 (Human-Human)
# ============================================================================
echo "========================================================================"
echo "ANALYSIS 1/4: DNN5 (REH) → GSE146773 (Human-Human)"
echo "========================================================================"
echo "Start time: $(date)"

python run_shap_analysis.py \
  --model_path "$MODEL_DIR/reh_robust/enhancedense/enhancedense_NFT_reh_fld_1.pt" \
  --benchmark GSE146773 \
  --background_data "$TRAINING_DIR/final_training_data_reh/reh_training_data.csv" \
  --output_dir results/shap_dnn5_reh/human_to_human_gse146773

echo "Analysis 1 completed: $(date)"
echo ""

# ============================================================================
# ANALYSIS 2: DNN5 (REH) → Buettner_mESC (Human-Mouse Cross-Species)
# ============================================================================
echo "========================================================================"
echo "ANALYSIS 2/4: DNN5 (REH) → Buettner_mESC (Human-Mouse Cross-Species)"
echo "========================================================================"
echo "Start time: $(date)"

python run_shap_analysis.py \
  --model_path "$MODEL_DIR/reh_robust/enhancedense/enhancedense_NFT_reh_fld_1.pt" \
  --benchmark Buettner_mESC \
  --background_data "$TRAINING_DIR/final_training_data_reh/reh_training_data.csv" \
  --cross_species \
  --output_dir results/shap_dnn5_reh/human_to_mouse_buettner

echo "Analysis 2 completed: $(date)"
echo ""

# ============================================================================
# ANALYSIS 3: DNN3 (Nestorowa) → GSE146773 (Mouse-Human Cross-Species)
# ============================================================================
echo "========================================================================"
echo "ANALYSIS 3/4: DNN3 (Nestorowa) → GSE146773 (Mouse-Human Cross-Species)"
echo "========================================================================"
echo "Start time: $(date)"

python run_shap_analysis.py \
  --model_path "$MODEL_DIR/nestorova/simpledense/simpledense_NFT_nestorova_fld_1.pt" \
  --benchmark GSE146773 \
  --background_data "$TRAINING_DIR/final_training_data_nestorova/nestorova_training_data.csv" \
  --cross_species \
  --output_dir results/shap_dnn3_nestorowa/mouse_to_human_gse146773

echo "Analysis 3 completed: $(date)"
echo ""

# ============================================================================
# ANALYSIS 4: DNN3 (Nestorowa) → Buettner_mESC (Mouse-Mouse)
# ============================================================================
echo "========================================================================"
echo "ANALYSIS 4/4: DNN3 (Nestorowa) → Buettner_mESC (Mouse-Mouse)"
echo "========================================================================"
echo "Start time: $(date)"

python run_shap_analysis.py \
  --model_path "$MODEL_DIR/nestorova/simpledense/simpledense_NFT_nestorova_fld_1.pt" \
  --benchmark Buettner_mESC \
  --background_data "$TRAINING_DIR/final_training_data_nestorova/nestorova_training_data.csv" \
  --output_dir results/shap_dnn3_nestorowa/mouse_to_mouse_buettner

echo "Analysis 4 completed: $(date)"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "========================================================================"
echo "ALL 4 SHAP ANALYSES COMPLETED SUCCESSFULLY!"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  1. $INTERP_DIR/results/shap_dnn5_reh/human_to_human_gse146773/"
echo "  2. $INTERP_DIR/results/shap_dnn5_reh/human_to_mouse_buettner/"
echo "  3. $INTERP_DIR/results/shap_dnn3_nestorowa/mouse_to_human_gse146773/"
echo "  4. $INTERP_DIR/results/shap_dnn3_nestorowa/mouse_to_mouse_buettner/"
echo ""
echo "Output files per analysis:"
echo "  - *_shap_summary.png (visualization)"
echo "  - *_SHAP.txt (ranked gene list)"
echo "  - *_top_features.csv (importance scores)"
echo "========================================================================"
