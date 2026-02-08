#!/usr/bin/env bash
################################################################################
# Create 4 Training Datasets Using Cell Cycle Marker Genes
################################################################################
#
# Features:
#   - Uses only marker genes as features:
#     - Seurat S genes (43 genes)
#     - Seurat G2M genes (54 genes)
#     - Revelio G1 genes (top 50 from 148 genes)
#     Total: ~147 marker genes
#
#   - Intersects with ALL 3 benchmarks to ensure no missing features:
#     - GSE146773 (human)
#     - GSE64016 (human)
#     - Buettner_mESC (mouse)
#
# Creates 4 datasets:
#   1. Nestorova + GSE75748
#   2. 4ds (PBMC + Mouse Brain + GSE75748 + Nestorova)
#   3. 2ds Human (PBMC + GSE75748)
#   4. 2ds Mouse (Mouse Brain + Nestorova)
#
# Output: 1_consensus_labeling/assign/final_training_data_marker_genes_*/
################################################################################

# Base directories
BASE_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction"
ASSIGN_DIR="$BASE_DIR/1_consensus_labeling/assign"
MARKER_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/Cell_Cycle_marker_genes"

# Marker gene files
SEURAT_S="$MARKER_DIR/_Seurat_S_genes.txt"
SEURAT_G2M="$MARKER_DIR/_Seurat_G2M_genes.txt"
REVELIO_G1="$MARKER_DIR/_Revelio_G1_genes.txt"

# Benchmark files (for gene intersection)
BENCHMARK_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed"
BENCHMARK_GSE146773="$BENCHMARK_DIR/GSE146773/GSE146773_expression.csv"
BENCHMARK_GSE64016="$BENCHMARK_DIR/GSE64016/GSE64016_expression.csv"
BENCHMARK_BUETTNER="$BENCHMARK_DIR/Buettner_mESC/Buettner_mESC_expression.csv"

# Create combined marker gene list: ALL Seurat S + ALL Seurat G2M + Top 50 Revelio G1
TEMP_MARKER_LIST="$BASE_DIR/I_actually_run_to_train/temp_combined_markers.txt"

echo "Creating combined marker gene list..."
# Take ALL Seurat S genes (43 genes)
cat "$SEURAT_S" > "$TEMP_MARKER_LIST"
# Add ALL Seurat G2M genes (54 genes)
cat "$SEURAT_G2M" >> "$TEMP_MARKER_LIST"
# Add only top 50 Revelio G1 genes (from 148 genes)
head -50 "$REVELIO_G1" >> "$TEMP_MARKER_LIST"

# Count combined markers
TOTAL_MARKERS=$(sort -u "$TEMP_MARKER_LIST" | wc -l)
echo "Combined marker genes: $TOTAL_MARKERS unique genes"
echo "  - Seurat S: 43 genes (ALL)"
echo "  - Seurat G2M: 54 genes (ALL)"
echo "  - Revelio G1: 50 genes (top 50)"
echo ""

# Input dataset files
PBMC_FILE="$ASSIGN_DIR/final_training_data_human/pbmc_human_training_data.csv"
MOUSE_BRAIN_FILE="$ASSIGN_DIR/final_training_data_mouse/mouse_brain_training_data_UPPERCASE.csv"
GSE75748_FILE="$ASSIGN_DIR/final_training_data_gse75748/gse75748_training_data.csv"
NESTOROVA_FILE="$ASSIGN_DIR/final_training_data_nestorova/nestorova_training_data.csv"

echo "================================================================================"
echo "CREATING MARKER GENE DATASETS WITH BENCHMARK INTERSECTION"
echo "================================================================================"
echo "Marker genes:"
echo "  - Seurat S: $SEURAT_S (43 genes)"
echo "  - Seurat G2M: $SEURAT_G2M (54 genes)"
echo "  - Revelio G1: $REVELIO_G1 (top 50 genes)"
echo ""
echo "Benchmark intersection:"
echo "  - GSE146773: $BENCHMARK_GSE146773"
echo "  - GSE64016: $BENCHMARK_GSE64016"
echo "  - Buettner_mESC: $BENCHMARK_BUETTNER"
echo ""
echo "Input datasets:"
echo "  - PBMC: $PBMC_FILE"
echo "  - Mouse Brain: $MOUSE_BRAIN_FILE"
echo "  - GSE75748: $GSE75748_FILE"
echo "  - Nestorova: $NESTOROVA_FILE"
echo "================================================================================"
echo ""

################################################################################
# Dataset 1: Nestorova + GSE75748
################################################################################

echo "================================================================================"
echo "DATASET 1: Nestorova + GSE75748 (marker genes)"
echo "================================================================================"

OUTPUT_DIR_1="$ASSIGN_DIR/final_training_data_marker_genes_nestorova_gse75748"
mkdir -p "$OUTPUT_DIR_1"

python concatenate_datasets_with_genes.py \
    --datasets "$NESTOROVA_FILE" "$GSE75748_FILE" \
    --benchmarks "$BENCHMARK_GSE146773" "$BENCHMARK_GSE64016" "$BENCHMARK_BUETTNER" \
    --gene-lists "$TEMP_MARKER_LIST" \
    --output-dir "$OUTPUT_DIR_1" \
    --output-name "concatenated_training_data_marker_genes_nestorova_gse75748"

echo ""
echo "DATASET 1 COMPLETE"
echo "Output: $OUTPUT_DIR_1"
echo ""

################################################################################
# Dataset 2: 4ds (PBMC + Mouse Brain + GSE75748 + Nestorova)
################################################################################

echo "================================================================================"
echo "DATASET 2: 4ds - All Datasets (marker genes)"
echo "================================================================================"

OUTPUT_DIR_2="$ASSIGN_DIR/final_training_data_marker_genes_4ds_all"
mkdir -p "$OUTPUT_DIR_2"

python concatenate_datasets_with_genes.py \
    --datasets "$PBMC_FILE" "$MOUSE_BRAIN_FILE" "$GSE75748_FILE" "$NESTOROVA_FILE" \
    --benchmarks "$BENCHMARK_GSE146773" "$BENCHMARK_GSE64016" "$BENCHMARK_BUETTNER" \
    --gene-lists "$TEMP_MARKER_LIST" \
    --output-dir "$OUTPUT_DIR_2" \
    --output-name "concatenated_training_data_marker_genes_4ds_all"

echo ""
echo "DATASET 2 COMPLETE"
echo "Output: $OUTPUT_DIR_2"
echo ""

################################################################################
# Dataset 3: 2ds Human (PBMC + GSE75748)
################################################################################

echo "================================================================================"
echo "DATASET 3: 2ds Human - PBMC + GSE75748 (marker genes)"
echo "================================================================================"

OUTPUT_DIR_3="$ASSIGN_DIR/final_training_data_marker_genes_2ds_human"
mkdir -p "$OUTPUT_DIR_3"

python concatenate_datasets_with_genes.py \
    --datasets "$PBMC_FILE" "$GSE75748_FILE" \
    --benchmarks "$BENCHMARK_GSE146773" "$BENCHMARK_GSE64016" "$BENCHMARK_BUETTNER" \
    --gene-lists "$TEMP_MARKER_LIST" \
    --output-dir "$OUTPUT_DIR_3" \
    --output-name "concatenated_training_data_marker_genes_2ds_human"

echo ""
echo "DATASET 3 COMPLETE"
echo "Output: $OUTPUT_DIR_3"
echo ""

################################################################################
# Dataset 4: 2ds Mouse (Mouse Brain + Nestorova)
################################################################################

echo "================================================================================"
echo "DATASET 4: 2ds Mouse - Mouse Brain + Nestorova (marker genes)"
echo "================================================================================"

OUTPUT_DIR_4="$ASSIGN_DIR/final_training_data_marker_genes_2ds_mouse"
mkdir -p "$OUTPUT_DIR_4"

python concatenate_datasets_with_genes.py \
    --datasets "$MOUSE_BRAIN_FILE" "$NESTOROVA_FILE" \
    --benchmarks "$BENCHMARK_GSE146773" "$BENCHMARK_GSE64016" "$BENCHMARK_BUETTNER" \
    --gene-lists "$TEMP_MARKER_LIST" \
    --output-dir "$OUTPUT_DIR_4" \
    --output-name "concatenated_training_data_marker_genes_2ds_mouse"

echo ""
echo "DATASET 4 COMPLETE"
echo "Output: $OUTPUT_DIR_4"
echo ""

################################################################################
# Summary
################################################################################

echo "================================================================================"
echo "ALL DATASETS CREATED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo "Created 4 datasets with marker genes + benchmark intersection:"
echo "  1. Nestorova + GSE75748:  $OUTPUT_DIR_1"
echo "  2. 4ds (all datasets):    $OUTPUT_DIR_2"
echo "  3. 2ds Human:             $OUTPUT_DIR_3"
echo "  4. 2ds Mouse:             $OUTPUT_DIR_4"
echo ""
echo "Each directory contains:"
echo "  - concatenated_training_data_marker_genes_*.csv (training data)"
echo "  - gene_list.txt (genes available in both marker lists AND all benchmarks)"
echo "  - dataset_metadata.txt (dataset statistics)"
echo ""
echo "Gene filtering applied:"
echo "  1. Intersection of training datasets"
echo "  2. Intersection with 3 benchmarks (GSE146773, GSE64016, Buettner_mESC)"
echo "  3. Filter to marker genes only (Seurat S/G2M + top 50 Revelio G1)"
echo ""
echo "Next steps:"
echo "  1. Use these datasets for training with marker genes only"
echo "  2. Compare performance: All genes vs Marker genes vs SelectKBest"
echo "  3. No missing feature issues during evaluation!"
echo "================================================================================"

# Cleanup temp file
rm -f "$TEMP_MARKER_LIST"
echo "Cleaned up temporary files"
