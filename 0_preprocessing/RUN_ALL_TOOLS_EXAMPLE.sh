#!/bin/bash

# EXAMPLE SCRIPT: Run all 4 cell cycle tools on multiple datasets
# This demonstrates how to use the UNIVERSAL scripts for different data formats

# Base directories
DATA_DIR="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/Has_cell_cycle_effect_or_not/scRNA_data"
OUTPUT_BASE="/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data"

# Tool scripts
SEURAT="./UNIVERSAL_SEURAT.R"
TRICYCLE="./UNIVERSAL_TRICYCLE.R"
REVELIO="./UNIVERSAL_REVELIO.R"
CCAFV2="./UNIVERSAL_ccAFv2.py"

# =====================================================
# EXAMPLE 1: GSE75748 (CSV format, human)
# =====================================================
echo "Processing GSE75748 (CSV format, human)..."

INPUT="${DATA_DIR}/GSE75748/GSE75748_sc_cell_type_ec.csv"
OUTPUT="${OUTPUT_BASE}/GSE75748_hPSC"
SAMPLE="GSE75748"
SPECIES="human"

Rscript $SEURAT --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format csv
Rscript $TRICYCLE --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format csv
Rscript $REVELIO --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format csv
# python $CCAFV2 --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format csv  # RUN LOCALLY (requires TensorFlow)

# =====================================================
# EXAMPLE 2: Nestorova (TXT format, mouse)
# =====================================================
echo "Processing Nestorova (TXT format, mouse)..."

INPUT="${DATA_DIR}/nestorova/nestorawa_forcellcycle_expressionMatrix.txt"
OUTPUT="${OUTPUT_BASE}/Nestorova_mouse_new_training_data"
SAMPLE="nestorova"
SPECIES="mouse"

Rscript $SEURAT --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format txt
Rscript $TRICYCLE --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format txt
Rscript $REVELIO --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format txt
python $CCAFV2 --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format txt

# =====================================================
# EXAMPLE 3: PBMC (10X MTX format, human)
# =====================================================
echo "Processing PBMC (10X MTX format, human)..."

INPUT="${DATA_DIR}/pbmc_healthy_human/filtered_feature_bc_matrix"
OUTPUT="${OUTPUT_BASE}/PBMC_healthy_human"
SAMPLE="pbmc_human"
SPECIES="human"

Rscript $SEURAT --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x
Rscript $TRICYCLE --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x
Rscript $REVELIO --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x
python $CCAFV2 --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x

# =====================================================
# EXAMPLE 4: Mouse Brain (10X MTX format, mouse)
# =====================================================
echo "Processing Mouse Brain (10X MTX format, mouse)..."

INPUT="${DATA_DIR}/10-k-brain-cells_healthy_mouse/filtered_feature_bc_matrix"
OUTPUT="${OUTPUT_BASE}/Mouse_brain_10k"
SAMPLE="mouse_brain"
SPECIES="mouse"

Rscript $SEURAT --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x
Rscript $TRICYCLE --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x
Rscript $REVELIO --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x
python $CCAFV2 --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x

# =====================================================
# EXAMPLE 5: GSE64016 (CSV format, human) - BENCHMARK
# =====================================================
echo "Processing GSE64016 (CSV format, human)..."

INPUT="${DATA_DIR}/3_GSE64016_H1andFUCCI_normalized_EC_original/GSE64016_H1andFUCCI_normalized_EC.csv"
OUTPUT="${OUTPUT_BASE}/GSE64016_benchmark"
SAMPLE="GSE64016"
SPECIES="human"

Rscript $SEURAT --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format csv
Rscript $TRICYCLE --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format csv
Rscript $REVELIO --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format csv
# python $CCAFV2 --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format csv  # RUN LOCALLY (requires TensorFlow)

# =====================================================
# EXAMPLE 6: GSE146773 (10X MTX format, human) - BENCHMARK
# =====================================================
echo "Processing GSE146773 (10X MTX format, human)..."

INPUT="${DATA_DIR}/GSE146773/filtered_feature_bc_matrix"
OUTPUT="${OUTPUT_BASE}/GSE146773_benchmark"
SAMPLE="GSE146773"
SPECIES="human"

Rscript $SEURAT --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x
Rscript $TRICYCLE --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x
Rscript $REVELIO --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x
python $CCAFV2 --input "$INPUT" --output "$OUTPUT" --sample "$SAMPLE" --species "$SPECIES" --format 10x

echo "ALL DONE!"
