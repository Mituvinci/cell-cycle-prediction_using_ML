#!/usr/bin/env Rscript

# UNIVERSAL REVELIO CELL CYCLE PHASE ASSIGNMENT
# Handles: 10X MTX, CSV, TXT formats
# Works for: Human and Mouse datasets (uses human cyclic gene markers)

suppressPackageStartupMessages({
  library(Revelio)
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

} else if (format_type == "csv") {
  cat("Reading CSV format...\n")
  mat <- read.csv(input_path, row.names = 1, check.names = FALSE)
  mat_sparse <- as(as.matrix(mat), "dgCMatrix")

} else if (format_type == "txt") {
  cat("Reading TXT/TSV format...\n")
  mat <- read.delim(input_path, row.names = 1, check.names = FALSE)
  mat_sparse <- as(as.matrix(mat), "dgCMatrix")
}

# -----------------------------
# Gene name formatting
# Revelio uses UPPERCASE human cyclic gene markers
# Works for both human and mouse data
# -----------------------------
rownames(mat_sparse) <- toupper(rownames(mat_sparse))

# Convert sparse matrix to regular matrix (Revelio requires regular matrix)
mat_dense <- as.matrix(mat_sparse)

cat(paste("Matrix dimensions:", nrow(mat_dense), "genes x", ncol(mat_dense), "cells\n"))
cat("Running Revelio using HUMAN markers with UPPERCASE gene symbols\n")

# -----------------------------
# Create Revelio object
# -----------------------------
myData <- createRevelioObject(
  rawData     = mat_dense,
  cyclicGenes = revelioTestData_cyclicGenes
)

# -----------------------------
# Assign cell cycle phases
# -----------------------------
cat("Assigning cell cycle phases...\n")
myData <- getCellCyclePhaseAssignInformation(dataList = myData)

# -----------------------------
# Extract and save results
# -----------------------------
cc_phases_df <- myData@cellInfo[, c("cellID", "ccPhase")]
cat("\nFirst 10 predictions:\n")
print(head(cc_phases_df, 10))

output_path <- file.path(output_dir, paste0("revelio_", sample_name, ".csv"))
write.csv(cc_phases_df, output_path, row.names = FALSE)
cat(paste("\nSaved predictions:", output_path, "\n"))

cat("\nPhase distribution:\n")
print(table(cc_phases_df$ccPhase))

cat("\nDONE!\n")
