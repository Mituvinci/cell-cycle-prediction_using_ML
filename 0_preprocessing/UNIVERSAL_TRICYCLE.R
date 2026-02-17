#!/usr/bin/env Rscript

# UNIVERSAL TRICYCLE CELL CYCLE PREDICTION
# Handles: 10X MTX, CSV, TXT formats
# Works for: Human and Mouse datasets

suppressPackageStartupMessages({
  library(tricycle)
  library(scater)
  library(SingleCellExperiment)
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

# Helper: capitalize gene names
capitalize_genes <- function(x) {
  paste0(toupper(substring(x, 1, 1)), tolower(substring(x, 2)))
}

# -----------------------------
# Auto-detect format
# -----------------------------
if (format_type == "auto") {
  if (dir.exists(input_path)) {
    if (file.exists(file.path(input_path, "matrix.mtx.gz")) ||
        file.exists(file.path(input_path, "matrix.mtx"))) {
      format_type <- "10x"
    } else {
      stop("Cannot auto-detect format for directory input. Please specify --format")
    }
  } else if (file.exists(input_path)) {
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
  suppressMessages({
    library(Seurat)
    mat_sparse <- Read10X(data.dir = input_path)
  })

  if (is.list(mat_sparse)) {
    mat_sparse <- mat_sparse[[1]]
  }

  # Already sparse
  mat <- as.matrix(mat_sparse)

} else if (format_type == "csv") {
  cat("Reading CSV format...\n")
  mat <- read.csv(input_path, row.names = 1, check.names = FALSE)
  mat <- as.matrix(mat)

} else if (format_type == "txt") {
  cat("Reading TXT/TSV format...\n")
  mat <- read.delim(input_path, row.names = 1, check.names = FALSE)
  mat <- as.matrix(mat)
}

# -----------------------------
# Gene name formatting
# Tricycle gene name requirements:
# - Human: UPPERCASE (GAPDH, ACTB, TP53)
# - Mouse: Capitalized (Gapdh, Actb, Tp53)
# -----------------------------
if (species == "human") {
  rownames(mat) <- toupper(rownames(mat))
  cat("Using UPPERCASE gene names for human\n")
} else {
  rownames(mat) <- sapply(rownames(mat), capitalize_genes)
  cat("Using Capitalized gene names for mouse\n")
}

cat(paste("Matrix dimensions:", nrow(mat), "genes x", ncol(mat), "cells\n"))

# -----------------------------
# Convert to integer sparse matrix (Tricycle requires counts)
# -----------------------------
mat <- round(mat)
storage.mode(mat) <- "integer"
mat_sparse <- as(mat, "dgCMatrix")

# -----------------------------
# Build SingleCellExperiment
# -----------------------------
sce <- SingleCellExperiment(assays = list(counts = mat_sparse))
sce <- logNormCounts(sce)

cat(paste("SingleCellExperiment:", nrow(sce), "genes x", ncol(sce), "cells\n"))

# -----------------------------
# Run Tricycle
# -----------------------------
cat("Running Tricycle cell cycle prediction...\n")

sce <- project_cycle_space(
  sce,
  exprs_values = "logcounts",
  name         = "tricycleEmbedding",
  gname.type   = "SYMBOL",
  species      = species
)

sce <- estimate_cycle_position(
  sce,
  gname.type = "SYMBOL",
  species    = species
)

sce <- estimate_Schwabe_stage(
  sce,
  gname.type = "SYMBOL",
  species    = species
)

cat("Tricycle prediction done\n")

# -----------------------------
# Save output
# -----------------------------
out_df <- data.frame(
  Cell              = colnames(sce),
  tricyclePosition  = sce$tricyclePosition,
  CCStage           = sce$CCStage
)

out_path <- file.path(output_dir, paste0("tricycle_", sample_name, ".csv"))
write.csv(out_df, out_path, row.names = FALSE)
cat(paste("Saved predictions:", out_path, "\n"))

cat("\nStage distribution:\n")
print(table(out_df$CCStage))

cat("\nDONE!\n")
