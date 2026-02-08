#!/usr/bin/env bash
# Evaluate ALL datasets on ALL 3 benchmarks

GRID_SEARCH_BASE="../models/grid_search"
BENCHMARKS="GSE146773 GSE64016 Buettner_mESC"
SCALING_METHOD="simple"
MODEL="simpledense"
DATASETS=("concate_4ds_all" "concate_2ds_human" "concate_mouse")
SCALERS=("standard" "minmax" "robust")
FEATURE_MODES=("all" "300")

echo "Evaluating ALL datasets on ALL 3 benchmarks"
total=0; success=0; failed=0

for dataset in "${DATASETS[@]}"; do
  for scaler in "${SCALERS[@]}"; do
    for feature_mode in "${FEATURE_MODES[@]}"; do
      config_name="${dataset}_${scaler}_${feature_mode}"
      model_dir="${GRID_SEARCH_BASE}/${config_name}/${MODEL}"
      
      [ -d "$model_dir" ] || continue
      
      for model_file in "$model_dir"/*_fld_*.pt; do
        [ -e "$model_file" ] || continue
        total=$((total + 1))
        echo "[$total] $(basename "$model_file")"
        
        python ../3_evaluation/evaluate_models.py --model_path "$model_file" --benchmarks $BENCHMARKS --scaling_method $SCALING_METHOD && success=$((success + 1)) || failed=$((failed + 1))
      done
    done
  done
done

echo "Total: $total | Success: $success | Failed: $failed"

# Consolidate
mkdir -p ../results/grid_search
for dataset in "${DATASETS[@]}"; do
  for scaler in "${SCALERS[@]}"; do
    for feature_mode in "${FEATURE_MODES[@]}"; do
      config_name="${dataset}_${scaler}_${feature_mode}"
      model_dir="${GRID_SEARCH_BASE}/${config_name}"
      [ -d "$model_dir" ] || continue
      python ../3_evaluation/consolidate_training_and_benchmark.py --input "$model_dir" --output "../results/grid_search/consolidated_${config_name}.csv" --species all 2>/dev/null
    done
  done
done

# Merge all
python - <<'PYTHON'
import pandas as pd, glob
csvs = glob.glob("../results/grid_search/consolidated_*.csv")
df = pd.concat([pd.read_csv(f).assign(configuration=f.split('consolidated_')[1].replace('.csv','')) for f in csvs], ignore_index=True)
df.to_csv("../results/grid_search/comparison_all_configs.csv", index=False)
print(f"Merged {len(csvs)} configs into comparison_all_configs.csv")
PYTHON
