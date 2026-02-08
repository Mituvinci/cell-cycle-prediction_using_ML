#!/usr/bin/env bash
# Train DL models on GSE75748 + Nestorova WITH SELECTKBEST (K=300)

DATASET="gse75748_nestorova"
TRAINING_DATA="../1_consensus_labeling/assign/final_training_data_GSE75748_Nestorova/concatenated_training_data_GSE75748_Nestorova.csv"
GENE_LIST="../1_consensus_labeling/assign/final_training_data_GSE75748_Nestorova/gene_list.txt"
MODELS=("simpledense" "cnn" "hbdcnn" "fe" "enhancedense")
TRIALS=50
CV=3
SCALER="standard"
FEATURE_SELECT="SelectKBest"
SELECT_K=300

echo "================================================================================"
echo "Training DL models on GSE75748 + Nestorova"
echo "Trials: $TRIALS, CV: $CV, Feature Selection: $FEATURE_SELECT, K: $SELECT_K"
echo "================================================================================"

for MODEL in "${MODELS[@]}"; do
  output_dir="../models/final_best_k_ft/gse75748_nestorova/${MODEL}"
  mkdir -p "$output_dir"

  TRAINING_ARGS="--model $MODEL --dataset $DATASET --gene-list $GENE_LIST --trials $TRIALS --cv $CV --scaling $SCALER --feature-selection $FEATURE_SELECT --select-k $SELECT_K --output $output_dir"

  echo "  Submitting: $MODEL"
  sbatch --export=ALL,TRAINING_SCRIPT="../2_model_training/train_deep_learning.py",TRAINING_ARGS="$TRAINING_ARGS" --job-name="DL_K300_GSE75748Nest_${MODEL}" ../scripts/train_model_generic.slurm
  sleep 1
done

echo "================================================================================"
echo "SUBMITTED: 5 DL jobs (with SelectKBest k=300)"
echo "================================================================================"
