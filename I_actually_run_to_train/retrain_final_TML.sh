#!/usr/bin/env bash
# FINAL TRAINING - ALL TRADITIONAL ML MODELS

DATASETS=("concate_mouse" "concate_4ds_all" "concate_2ds_human")
TRIALS=(50 20 20)
TRAINING_PATHS=(
  "../1_consensus_labeling/assign/final_training_data_concate_mouse_brain_Nestorova/concatenated_training_data.csv"
  "../1_consensus_labeling/assign/final_training_data_concate_4_ds_pbmchealthy_human_mouse_brain_GSE75748_Nestorova/concatenated_training_data.csv"
  "../1_consensus_labeling/assign/final_training_data_concate_2_ds_only_pbmchealthy_human_GSE75748/concatenated_training_data.csv"
)
GENE_LISTS=(
  "../1_consensus_labeling/assign/final_training_data_concate_mouse_brain_Nestorova/gene_list.txt"
  "../1_consensus_labeling/assign/final_training_data_concate_4_ds_pbmchealthy_human_mouse_brain_GSE75748_Nestorova/gene_list.txt"
  "../1_consensus_labeling/assign/final_training_data_concate_2_ds_only_pbmchealthy_human_GSE75748/gene_list.txt"
)

MODELS=("rf" "adaboost" "lgbm" "ensemble")
SCALER="standard"
CV=3

for i in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$i]}"
  TRIAL="${TRIALS[$i]}"
  TRAINING_DATA="${TRAINING_PATHS[$i]}"
  GENE_LIST="${GENE_LISTS[$i]}"
  
  echo "================================================================================"
  echo "DATASET: $DATASET (trials=$TRIAL, cv=$CV)"
  echo "================================================================================"
  
  for MODEL in "${MODELS[@]}"; do
    output_dir="../models/final_best/${DATASET}/${MODEL}"
    mkdir -p "$output_dir"
    
    TRAINING_ARGS="--model $MODEL --dataset $DATASET --gene-list $GENE_LIST --trials $TRIAL --cv $CV --scaling $SCALER --output $output_dir"
    
    echo "  Submitting: $MODEL"
    sbatch --export=ALL,TRAINING_SCRIPT="../2_model_training/train_traditional_ml.py",TRAINING_ARGS="$TRAINING_ARGS" --job-name="TML_${DATASET}_${MODEL}" ../scripts/train_model_generic.slurm
    sleep 1
  done
done

echo "================================================================================"
echo "SUBMITTED: 3 datasets × 3 TML models = 9 jobs"
echo "================================================================================"
