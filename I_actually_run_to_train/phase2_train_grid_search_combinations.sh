#!/usr/bin/env bash
# Train grid search on ALL 3 datasets

DATASETS=("concate_4ds_all" "concate_2ds_human" "concate_mouse")
TRAINING_PATHS=(
  "../1_consensus_labeling/assign/final_training_data_concate_4_ds_pbmchealthy_human_mouse_brain_GSE75748_Nestorova/concatenated_training_data.csv"
  "../1_consensus_labeling/assign/final_training_data_concate_2_ds_only_pbmchealthy_human_GSE75748/concatenated_training_data.csv"
  "../1_consensus_labeling/assign/final_training_data_concate_mouse_brain_Nestorova/concatenated_training_data.csv"
)
GENE_LISTS=(
  "../1_consensus_labeling/assign/final_training_data_concate_4_ds_pbmchealthy_human_mouse_brain_GSE75748_Nestorova/gene_list.txt"
  "../1_consensus_labeling/assign/final_training_data_concate_2_ds_only_pbmchealthy_human_GSE75748/gene_list.txt"
  "../1_consensus_labeling/assign/final_training_data_concate_mouse_brain_Nestorova/gene_list.txt"
)

MODEL="simpledense"
TRIALS=5
CV_FOLDS=2
SCALERS=("standard" "minmax" "robust")
FEATURE_MODES=("all" "300")

for i in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$i]}"
  TRAINING_DATA="${TRAINING_PATHS[$i]}"
  GENE_LIST="${GENE_LISTS[$i]}"
  
  echo "================================================================================"
  echo "DATASET: $DATASET"
  echo "================================================================================"
  
  for scaler in "${SCALERS[@]}"; do
    for feature_mode in "${FEATURE_MODES[@]}"; do
      config_name="${DATASET}_${scaler}_${feature_mode}"
      output_dir="../models/grid_search/${config_name}/${MODEL}"
      mkdir -p "$output_dir"
      
      if [ "$feature_mode" == "all" ]; then
        TRAINING_ARGS="--model $MODEL --dataset $DATASET --gene-list $GENE_LIST --trials $TRIALS --cv $CV_FOLDS --scaling $scaler --output $output_dir"
      else
        TRAINING_ARGS="--model $MODEL --dataset $DATASET --gene-list $GENE_LIST --trials $TRIALS --cv $CV_FOLDS --scaling $scaler --feature-selection selectkbest --num-features $feature_mode --output $output_dir"
      fi
      
      echo "[$config_name] Submitting..."
      sbatch --export=ALL,TRAINING_SCRIPT="../2_model_training/train_deep_learning.py",TRAINING_ARGS="$TRAINING_ARGS" --job-name="grid_${DATASET}_${scaler}_${feature_mode}" ../scripts/train_model_generic.slurm
      sleep 2
    done
  done
done

echo "ALL JOBS SUBMITTED (3 datasets × 6 configs = 18 jobs)"
