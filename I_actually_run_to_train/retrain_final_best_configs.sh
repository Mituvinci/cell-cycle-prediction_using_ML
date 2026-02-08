#!/usr/bin/env bash
# Retrain best configs with higher trials

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

MODEL="simpledense"
SCALER="standard"
CV=3

for i in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$i]}"
  TRIAL="${TRIALS[$i]}"
  TRAINING_DATA="${TRAINING_PATHS[$i]}"
  GENE_LIST="${GENE_LISTS[$i]}"
  
  config_name="${DATASET}_${SCALER}_all_FINAL"
  output_dir="../models/final_best/${config_name}/${MODEL}"
  mkdir -p "$output_dir"
  
  echo "================================================================================"
  echo "FINAL TRAINING: $DATASET"
  echo "Trials: $TRIAL | CV: $CV | Scaler: $SCALER"
  echo "================================================================================"
  
  TRAINING_ARGS="--model $MODEL --dataset $DATASET --gene-list $GENE_LIST --trials $TRIAL --cv $CV --scaling $SCALER --output $output_dir"
  
  sbatch --export=ALL,TRAINING_SCRIPT="../2_model_training/train_deep_learning.py",TRAINING_ARGS="$TRAINING_ARGS" --job-name="FINAL_${DATASET}" ../scripts/train_model_generic.slurm
  
  sleep 2
done

echo "================================================================================"
echo "SUBMITTED 3 FINAL TRAINING JOBS"
echo "concate_mouse: trials=50, cv=3"
echo "concate_4ds_all: trials=20, cv=3"
echo "concate_2ds_human: trials=20, cv=3"
echo "================================================================================"
