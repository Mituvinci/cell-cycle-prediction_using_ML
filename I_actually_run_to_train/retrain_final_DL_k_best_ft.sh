#!/usr/bin/env bash
# FINAL TRAINING - ALL DEEP LEARNING MODELS WITH SELECTKBEST (K=300)

DATASETS=("concate_mouse" "concate_4ds_all" "concate_2ds_human")
TRIALS=(50 20 20)
TRAINING_PATHS=(
  "../1_consensus_labeling/assign/final_training_data_concate_mouse_brain_Nestorova/concatenated_training_data_mouse_brain_Nestorova.csv"
  "../1_consensus_labeling/assign/final_training_data_concate_4_ds_pbmchealthy_human_mouse_brain_GSE75748_Nestorova/concatenated_training_data_4_ds_pbmchealthy_human_mouse_brain_GSE75748_Nestorova.csv"
  "../1_consensus_labeling/assign/final_training_data_concate_2_ds_only_pbmchealthy_human_GSE75748/concatenated_training_data_2_ds_only_pbmchealthy_human_GSE75748.csv"
)
GENE_LISTS=(
  "../1_consensus_labeling/assign/final_training_data_concate_mouse_brain_Nestorova/gene_list.txt"
  "../1_consensus_labeling/assign/final_training_data_concate_4_ds_pbmchealthy_human_mouse_brain_GSE75748_Nestorova/gene_list.txt"
  "../1_consensus_labeling/assign/final_training_data_concate_2_ds_only_pbmchealthy_human_GSE75748/gene_list.txt"
)

MODELS=("simpledense" "cnn" "hbdcnn" "fe" "enhancedense")
SCALER="standard"
CV=3
FEATURE_SELECT="SelectKBest"
SELECT_K=300

for i in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$i]}"
  TRIAL="${TRIALS[$i]}"
  TRAINING_DATA="${TRAINING_PATHS[$i]}"
  GENE_LIST="${GENE_LISTS[$i]}"

  echo "================================================================================"
  echo "DATASET: $DATASET (trials=$TRIAL, cv=$CV, feature-selection=$FEATURE_SELECT, k=$SELECT_K)"
  echo "================================================================================"

  for MODEL in "${MODELS[@]}"; do
    output_dir="../models/final_best_k_ft/${DATASET}/${MODEL}"
    mkdir -p "$output_dir"

    TRAINING_ARGS="--model $MODEL --dataset $DATASET --gene-list $GENE_LIST --trials $TRIAL --cv $CV --scaling $SCALER --feature-selection $FEATURE_SELECT --select-k $SELECT_K --output $output_dir"

    echo "  Submitting: $MODEL"
    sbatch --export=ALL,TRAINING_SCRIPT="../2_model_training/train_deep_learning.py",TRAINING_ARGS="$TRAINING_ARGS" --job-name="DL_K300_${DATASET}_${MODEL}" ../scripts/train_model_generic.slurm
    sleep 1
  done
done

echo "================================================================================"
echo "SUBMITTED: 3 datasets × 5 DL models = 15 jobs (with SelectKBest k=300)"
echo "================================================================================"
