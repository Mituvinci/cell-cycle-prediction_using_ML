# UNIVERSAL CELL CYCLE PREDICTION TOOLS

This directory contains 4 universal scripts that work with ANY dataset format for cell cycle phase prediction.

## SCRIPTS

1. **UNIVERSAL_SEURAT.R** - Seurat cell cycle scoring
2. **UNIVERSAL_TRICYCLE.R** - Tricycle phase prediction
3. **UNIVERSAL_REVELIO.R** - Revelio phase assignment
4. **UNIVERSAL_ccAFv2.py** - ccAFv2 phase prediction

## SUPPORTED FORMATS

- **10X MTX format**: Directory with matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz
- **CSV format**: Comma-separated file (genes x cells)
- **TXT format**: Tab-separated file (genes x cells)

## USAGE

### Basic Command Structure

```bash
# R scripts (Seurat, Tricycle, Revelio)
Rscript UNIVERSAL_<TOOL>.R \
  --input <path_to_data> \
  --output <output_directory> \
  --sample <sample_name> \
  --species <human|mouse> \
  --format <10x|csv|txt|auto>

# Python script (ccAFv2)
python UNIVERSAL_ccAFv2.py \
  --input <path_to_data> \
  --output <output_directory> \
  --sample <sample_name> \
  --species <human|mouse> \
  --format <10x|csv|txt|auto>
```

### Arguments

| Argument | Short | Required | Description | Default |
|----------|-------|----------|-------------|---------|
| `--input` | `-i` | YES | Input file or directory | - |
| `--output` | `-o` | YES | Output directory | - |
| `--sample` | `-s` | NO | Sample name | sample |
| `--species` | - | NO | Species (human/mouse) | human |
| `--format` | `-f` | NO | Format (10x/csv/txt/auto) | auto |

## EXAMPLES

### Example 1: GSE75748 (CSV, Human)

```bash
Rscript UNIVERSAL_SEURAT.R \
  --input ../Has_cell_cycle_effect_or_not/scRNA_data/GSE75748/GSE75748_sc_cell_type_ec.csv \
  --output /path/to/output/GSE75748_hPSC \
  --sample GSE75748 \
  --species human \
  --format csv
```

### Example 2: Nestorova (TXT, Mouse)

```bash
Rscript UNIVERSAL_TRICYCLE.R \
  --input ../Has_cell_cycle_effect_or_not/scRNA_data/nestorova/nestorawa_forcellcycle_expressionMatrix.txt \
  --output /path/to/output/Nestorova \
  --sample nestorova \
  --species mouse \
  --format txt
```

### Example 3: PBMC (10X MTX, Human)

```bash
python UNIVERSAL_ccAFv2.py \
  --input ../Has_cell_cycle_effect_or_not/scRNA_data/pbmc_healthy_human/filtered_feature_bc_matrix \
  --output /path/to/output/PBMC \
  --sample pbmc_human \
  --species human \
  --format 10x
```

### Example 4: Auto-detect Format

```bash
# Format will be auto-detected from file extension or directory structure
Rscript UNIVERSAL_REVELIO.R \
  --input ../Has_cell_cycle_effect_or_not/scRNA_data/GSE75748/GSE75748_sc_cell_type_ec.csv \
  --output /path/to/output/GSE75748 \
  --sample GSE75748 \
  --species human
```

## OUTPUT FILES

Each tool generates prediction files in the output directory:

- **Seurat**: `seurat_<sample>_cc_phases.csv`
  - Columns: barcode_RNA, Phase (G1, S, G2M)

- **Tricycle**: `tricycle_<sample>.csv`
  - Columns: Cell, tricyclePosition, CCStage (G1, G1.S, S, G2, G2M, M, M.G1)

- **Revelio**: `revelio_<sample>.csv`
  - Columns: cellID, ccPhase (G1, S, G2M)

- **ccAFv2**: `ccAFV2_<sample>.csv`
  - Columns: Cell index, ccAFv2 (G0-qG0, Early G1, G1, G1.S, S, Late S, G2, G2M, M)

## GENE NAME FORMATTING

The scripts automatically handle gene name formatting:

- **Seurat & Revelio**: UPPERCASE (ACTB, GAPDH, TP53)
- **Tricycle & ccAFv2**: Capitalized (Actb, Gapdh, Tp53)

This ensures compatibility with each tool's marker gene databases.

## BATCH PROCESSING

To run all 4 tools on a dataset:

```bash
#!/bin/bash

INPUT="path/to/data"
OUTPUT="path/to/output"
SAMPLE="my_sample"
SPECIES="human"
FORMAT="csv"

Rscript UNIVERSAL_SEURAT.R -i "$INPUT" -o "$OUTPUT" -s "$SAMPLE" --species "$SPECIES" -f "$FORMAT"
Rscript UNIVERSAL_TRICYCLE.R -i "$INPUT" -o "$OUTPUT" -s "$SAMPLE" --species "$SPECIES" -f "$FORMAT"
Rscript UNIVERSAL_REVELIO.R -i "$INPUT" -o "$OUTPUT" -s "$SAMPLE" --species "$SPECIES" -f "$FORMAT"
python UNIVERSAL_ccAFv2.py -i "$INPUT" -o "$OUTPUT" -s "$SAMPLE" --species "$SPECIES" -f "$FORMAT"
```

See `RUN_ALL_TOOLS_EXAMPLE.sh` for a complete batch processing example.

## REQUIREMENTS

### R Packages
- Seurat
- tricycle
- scater
- SingleCellExperiment
- Revelio
- Matrix
- optparse

### Python Packages
- pandas
- numpy
- scanpy
- anndata
- ccAFv2

## TROUBLESHOOTING

1. **"Cannot auto-detect format"**
   - Specify format explicitly using `--format 10x/csv/txt`

2. **"Input path does not exist"**
   - Check that file/directory path is correct
   - Use absolute paths or correct relative paths

3. **Missing genes warning**
   - Some tools may not find all marker genes
   - This is normal and predictions will still work

4. **Memory errors with large datasets**
   - Consider subsetting cells or using HPC with more RAM

## DATASETS IN scRNA_DATA

| Dataset | Format | Species | Type | Path |
|---------|--------|---------|------|------|
| GSE75748 | CSV | Human | Training | GSE75748/GSE75748_sc_cell_type_ec.csv |
| Nestorova | TXT | Mouse | Training | nestorova/nestorawa_forcellcycle_expressionMatrix.txt |
| PBMC | 10X MTX | Human | Training | pbmc_healthy_human/filtered_feature_bc_matrix/ |
| Mouse Brain | 10X MTX | Mouse | Training | 10-k-brain-cells_healthy_mouse/filtered_feature_bc_matrix/ |
| GSE64016 | CSV | Human | Benchmark | 3_GSE64016_H1andFUCCI_normalized_EC_original/GSE64016_H1andFUCCI_normalized_EC.csv |
| GSE146773 | 10X MTX | Human | Benchmark | GSE146773/filtered_feature_bc_matrix/ |

DO NOT USE: REH, SUP-B15 (per user instructions)
