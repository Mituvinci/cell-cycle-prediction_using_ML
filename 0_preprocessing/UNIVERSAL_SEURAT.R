#!/usr/bin/env Rscript

# UNIVERSAL SEURAT CELL CYCLE SCORING
# Handles: 10X MTX, CSV, TXT formats
# Works for: Human and Mouse datasets

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(Matrix)
  library(optparse)
})

# -----------------------------
# Command-line arguments
# -----------------------------
option_list <- list(
  make_option(c("-i", "--input"), type="character", default=NULL,
              help="Input file or directory (for 10X: path to filtered_feature_bc_matrix/)", metavar="PATH"),
  make_option(c("-o", "--output"), type="character", default=NULL,
              help="Output directory", metavar="PATH"),
  make_option(c("-s", "--sample"), type="character", default="sample",
              help="Sample name [default=%default]", metavar="NAME"),
  make_option(c("-f", "--format"), type="character", default="auto",
              help="Format: 10x, csv, txt, or auto [default=%default]", metavar="FORMAT"),
  make_option(c("--species"), type="character", default="human",
              help="Species: human or mouse [default=%default]", metavar="SPECIES")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Validate required arguments
if (is.null(opt$input) || is.null(opt$output)) {
  print_help(opt_parser)
  stop("--input and --output are required arguments", call.=FALSE)
}

input_path  <- opt$input
output_dir  <- opt$output
sample_name <- opt$sample
format_type <- opt$format
species     <- opt$species

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat(paste("Created output directory:", output_dir, "\n"))
}

# -----------------------------
# Auto-detect format
# -----------------------------
if (format_type == "auto") {
  if (dir.exists(input_path)) {
    # Check if it's 10X directory
    if (file.exists(file.path(input_path, "matrix.mtx.gz")) ||
        file.exists(file.path(input_path, "matrix.mtx"))) {
      format_type <- "10x"
    } else {
      stop("Cannot auto-detect format for directory input. Please specify --format")
    }
  } else if (file.exists(input_path)) {
    # Detect by extension
    if (grepl("\\.csv$", input_path, ignore.case=TRUE)) {
      format_type <- "csv"
    } else if (grepl("\\.(txt|tsv)$", input_path, ignore.case=TRUE)) {
      format_type <- "txt"
    } else {
      stop("Cannot auto-detect format. Please specify --format (10x, csv, or txt)")
    }
  } else {
    stop(paste("Input path does not exist:", input_path))
  }
}

cat(paste("Detected format:", format_type, "\n"))
cat(paste("Species:", species, "\n"))
cat(paste("Sample:", sample_name, "\n"))

# -----------------------------
# Read expression matrix based on format
# -----------------------------
if (format_type == "10x") {
  cat("Reading 10X MTX format...\n")
  mat_sparse <- Read10X(data.dir = input_path)

  # If Read10X returns a list (multiple modalities), take the first one
  if (is.list(mat_sparse)) {
    mat_sparse <- mat_sparse[[1]]
  }

} else if (format_type == "csv") {
  cat("Reading CSV format...\n")
  mat <- read.csv(input_path, row.names = 1, check.names = FALSE)
  mat_sparse <- as(as.matrix(mat), "dgCMatrix")

} else if (format_type == "txt") {
  cat("Reading TXT/TSV format...\n")
  mat <- read.delim(input_path, row.names = 1, check.names = FALSE)
  mat_sparse <- as(as.matrix(mat), "dgCMatrix")

} else {
  stop(paste("Unknown format:", format_type))
}

# -----------------------------
# Gene name formatting
# Seurat cc.genes uses UPPERCASE human gene names
# Works for both human and mouse if genes are UPPERCASE
# -----------------------------
rownames(mat_sparse) <- toupper(rownames(mat_sparse))

cat(paste("Matrix dimensions:", nrow(mat_sparse), "genes x", ncol(mat_sparse), "cells\n"))

# -----------------------------
# Create Seurat object
# -----------------------------
srt_obj <- CreateSeuratObject(counts = mat_sparse, min.cells = 3)
srt_obj@meta.data$sample <- paste0("trainingdata_cellcyclescore_", sample_name)
srt_obj[["percent.mt"]] <- PercentageFeatureSet(srt_obj, pattern = "^MT-")

# -- Join layers (Seurat v5 fix) --
# Multiomics data (e.g. scMultiome ATAC+RNA) creates split layers in Seurat v5.
# CellCycleScoring fails to find features across split layers.
# JoinLayers merges them so all features are accessible.
if (inherits(srt_obj[["RNA"]], "Assay5")) {
  cat("Seurat v5 detected -- joining RNA layers\n")
  srt_obj[["RNA"]] <- JoinLayers(srt_obj[["RNA"]])
}

cat("Seurat object created\n")
print(srt_obj)

# -----------------------------
# Normalize
# -----------------------------
srt_obj <- NormalizeData(srt_obj)
cat("Normalization done\n")

# Save normalized expression
normalized_data <- GetAssayData(srt_obj, assay = "RNA", layer = "data")
normalized_df   <- as.data.frame(t(normalized_data))
normalized_file <- file.path(output_dir, paste0("trainingdata_cellcyclescore_", sample_name, "_normalized_gene_expression.csv"))
write.csv(normalized_df, normalized_file, row.names = TRUE)
cat(paste("Saved normalized expression:", normalized_file, "\n"))

# -----------------------------
# Cell cycle scoring
# -----------------------------
srt_obj <- FindVariableFeatures(srt_obj, selection.method = "vst")
srt_obj <- ScaleData(srt_obj, features = rownames(srt_obj))

# Seurat built-in human cell cycle genes (works with UPPERCASE)
s.genes   <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes

cc_score <- CellCycleScoring(srt_obj, s.features = s.genes, g2m.features = g2m.genes, set.ident = TRUE)
cat("Cell cycle scoring done\n")

# -----------------------------
# Save phase labels
# -----------------------------
df  <- cbind(cc_score[["Phase"]])
df  <- cbind(barcode_RNA = rownames(df), df)
rownames(df) <- 1:nrow(df)

cc_file <- file.path(output_dir, paste0("seurat_", sample_name, "_cc_phases.csv"))
write.csv(df, cc_file, row.names = FALSE)
cat(paste("Saved cell cycle phases:", cc_file, "\n"))

cat("\nPhase distribution:\n")
print(table(df$Phase))

cat("\nDONE!\n")
