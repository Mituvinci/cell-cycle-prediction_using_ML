# Benchmark Intersection Update

## Summary
Updated dataset concatenation scripts to include gene intersection with benchmark datasets to prevent missing feature errors during evaluation.

## Problems Fixed

### 1. Missing Benchmark Intersection
Previously, training datasets only took intersection of training data genes. This caused issues when:
- Training data had genes NOT present in benchmarks
- Evaluation would fail with missing feature errors
- Would require imputation (which we want to avoid)

### 2. Gene Name Format Mismatch (CRITICAL BUG)
Different datasets had different gene name formats:
- GSE146773, GSE64016: UPPERCASE (ACTB, GAPDH, TP53)
- Buettner_mESC: Capitalized (Actb, Gapdh, Tp53)
- Training data: Capitalized (Actb, Gapdh, Tp53)
- Marker genes: UPPERCASE after loading

**Result**: Intersection of {'ACTB'} and {'Actb'} = {} (EMPTY!) → 0 genes!

**Fix**: Convert ALL gene names to UPPERCASE before intersection

## Solution
Modified `concatenate_datasets_with_genes.py` and `create_marker_gene_datasets.sh` to:
1. Take intersection of genes across training datasets
2. Take intersection with ALL 3 benchmark datasets
3. Then filter to marker genes only

This ensures:
- All training genes exist in ALL benchmarks
- No missing features during evaluation
- No imputation needed

## Changes Made

### 1. `concatenate_datasets_with_genes.py`

**Added Features:**
- New optional parameter: `--benchmarks` (accepts multiple benchmark CSV paths)
- Function `load_benchmark_genes()` to load gene names from benchmarks
- Updated gene intersection logic to include benchmarks
- Enhanced metadata output to document benchmark intersection

**Usage Example:**
```bash
python concatenate_datasets_with_genes.py \
    --datasets training1.csv training2.csv \
    --benchmarks benchmark1_expression.csv benchmark2_expression.csv \
    --gene-lists markers.txt \
    --output-dir output/ \
    --output-name my_dataset
```

**Gene Filtering Order:**
1. Load training datasets
2. Load benchmarks (if provided)
3. Find intersection across all training datasets
4. Intersect with all benchmarks (if provided)
5. Filter to marker genes (if provided)
6. Create final training data

### 2. `create_marker_gene_datasets.sh`

**Added:**
- Benchmark file paths:
  - `$BENCHMARK_GSE146773`: GSE146773_expression.csv
  - `$BENCHMARK_GSE64016`: GSE64016_expression.csv
  - `$BENCHMARK_BUETTNER`: Buettner_mESC_expression.csv

**Updated:**
- All 4 dataset creation commands now include `--benchmarks` parameter
- All datasets use ALL 3 benchmarks for intersection
- Updated header documentation
- Updated summary output

## Benchmark Paths
```bash
BENCHMARK_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed"
BENCHMARK_GSE146773="$BENCHMARK_DIR/GSE146773/GSE146773_expression.csv"
BENCHMARK_GSE64016="$BENCHMARK_DIR/GSE64016/GSE64016_expression.csv"
BENCHMARK_BUETTNER="$BENCHMARK_DIR/Buettner_mESC/Buettner_mESC_expression.csv"
```

## Gene Filtering Workflow

### Old Workflow (INCORRECT)
```
Training Datasets → Intersection → Filter to Markers → Final Data
```
**Problem**: Training genes might not exist in benchmarks

### New Workflow (CORRECT)
```
Training Datasets → Intersection → Benchmark Intersection → Filter to Markers → Final Data
```
**Result**: All training genes guaranteed to exist in benchmarks

## Expected Impact

### Before (Without Benchmark Intersection)
- Nestorova + GSE75748: 124 genes
- 2ds Mouse: 122 genes
- 2ds Human: 71 genes
- 4ds All: 69 genes

### After (With Benchmark Intersection)
Gene counts may be slightly lower, but will guarantee:
- No missing feature errors
- No imputation needed
- Clean evaluation on all 3 benchmarks

## Next Steps
1. Run `create_marker_gene_datasets.sh` to regenerate datasets
2. Check new gene counts in `dataset_metadata.txt`
3. Train models using new datasets
4. Evaluate on benchmarks (no missing feature issues!)

## Files Modified
- `I_actually_run_to_train/concatenate_datasets_with_genes.py`
- `I_actually_run_to_train/create_marker_gene_datasets.sh`

## Flexibility Note
Both scripts remain flexible:
- Benchmarks are OPTIONAL in Python script
- Marker genes are OPTIONAL in Python script
- Scripts can be used for any dataset concatenation task
- Bash script configures specific use case (benchmarks + markers)
