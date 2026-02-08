#!/usr/bin/env bash
# Train DL models on GSE75748 + Nestorova

DATASET="gse75748_nestorova"
TRAINING_DATA="../1_consensus_labeling/assign/final_training_data_GSE75748_Nestorova/concatenated_training_data.csv"
GENE_LIST="../1_consensus_labeling/assign/final_training_data_GSE75748_Nestorova/gene_list.txt"
MODELS=("simpledense" "cnn" "hbdcnn" "fe" "enhancedense")
TRIALS=20
CV=3
SCALER="standard"

echo "Training DL models on GSE75748 + Nestorova (trials=$TRIALS, cv=$CV)"

for MODEL in "${MODELS[@]}"; do
  output_dir="../models/final_best/gse75748_nestorova/${MODEL}"
  mkdir -p "$output_dir"
  
  TRAINING_ARGS="--model $MODEL --dataset $DATASET --gene-list $GENE_LIST --trials $TRIALS --cv $CV --scaling $SCALER --output $output_dir"
  
  echo "  Submitting: $MODEL"
  sbatch --export=ALL,TRAINING_SCRIPT="../2_model_training/train_deep_learning.py",TRAINING_ARGS="$TRAINING_ARGS" --job-name="DL_GSE75748Nest_${MODEL}" ../scripts/train_model_generic.slurm
  sleep 1
done

echo "SUBMITTED: 5 DL jobs"
