#!/usr/bin/env Rscript
################################################################################
# pca_cell_cycle_from_scoring.R
#
# Follows the logic of 0_seurat_label_reh_sup.R exactly:
#   Load -> Normalize -> FindVariableFeatures -> ScaleData(ALL genes)
#   -> CellCycleScoring -> PCA(HVGs) -> DimPlot colored by Phase -> save PNG
#
# Phase labels are generated in memory by CellCycleScoring. No CSV read/write.
# If phases cluster separately in PCA = cell cycle effect present.
# If phases are mixed in PCA = no cell cycle effect.
#
# Arguments:
#   args[1] = input_path    : 10x directory, .h5, .csv, .txt, or .tsv
#   args[2] = output_dir    : output directory (created if missing)
#   args[3] = organism      : "human" or "mouse" (default: human)
#   args[4] = dataset_name  : prefix for output PNG
################################################################################

library(Seurat)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript pca_cell_cycle_from_scoring.R <input_path> <output_dir> [organism] [dataset_name]")
}

input_path   <- args[1]
outdir       <- args[2]
organism     <- if (length(args) >= 3) tolower(args[3]) else "human"
dataset_name <- if (length(args) >= 4) args[4] else "dataset"

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================================\n")
cat(sprintf("  PCA Cell Cycle Check: %s (%s)\n", dataset_name, organism))
cat("================================================================================\n\n")

################################################################################
# Load data -- auto-detect format (same as detect_cell_cycle_effect.R)
################################################################################
cat("Loading data...\n")

if (dir.exists(input_path)) {
  cat("  Format: 10x directory\n")
  counts <- Read10X(input_path)
} else if (grepl("\\.h5$", input_path, ignore.case = TRUE)) {
  cat("  Format: HDF5\n")
  counts <- Read10X_h5(input_path)
  # Multiome H5 returns a list -- extract Gene Expression (same as your original code)
  if (is.list(counts)) {
    counts <- counts[["Gene Expression"]]
  }
} else if (grepl("\\.(csv|txt|tsv)$", input_path, ignore.case = TRUE)) {
  cat("  Format: CSV/TXT/TSV (genes as rows, cells as columns)\n")
  sep <- if (grepl("\\.csv$", input_path, ignore.case = TRUE)) "," else "\t"
  counts <- read.table(input_path, sep = sep, header = TRUE, row.names = 1,
                       check.names = FALSE, stringsAsFactors = FALSE)
  counts <- as.matrix(counts)
} else {
  stop("Unsupported format. Use 10x directory, .h5, .csv, .txt, or .tsv")
}

################################################################################
# Create Seurat object (min.cells = 200, same as your original code)
################################################################################
seu <- CreateSeuratObject(counts = counts, project = dataset_name, min.cells = 200)
cat(sprintf("  Seurat object: %d cells, %d genes\n", ncol(seu), nrow(seu)))

# Mitochondrial filter
if (any(grepl("^MT-", rownames(seu)))) {
  mt_pattern <- "^MT-"
} else if (any(grepl("^Mt-", rownames(seu)))) {
  mt_pattern <- "^Mt-"
} else {
  mt_pattern <- "^MT-"
  cat("  NOTE: No MT genes detected. percent.mt will be 0.\n")
}
seu[["percent.mt"]] <- PercentageFeatureSet(seu, pattern = mt_pattern)
seu <- subset(seu, subset = percent.mt < 20)
cat(sprintf("  After MT filter: %d cells\n", ncol(seu)))

################################################################################
# Normalize -- same as your code: NormalizeData() with defaults
################################################################################
cat("\nNormalizing...\n")
seu <- NormalizeData(seu)

################################################################################
# Find Variable Features -- same as your code
################################################################################
seu <- FindVariableFeatures(seu, selection.method = "vst")
hvg_genes <- VariableFeatures(seu)
cat(sprintf("  HVGs: %d\n", length(hvg_genes)))

################################################################################
# ScaleData on ALL genes -- this is the key step.
# Your code does ScaleData(features = rownames(seu)) BEFORE CellCycleScoring.
################################################################################
cat("Scaling all genes...\n")
seu <- ScaleData(seu, features = rownames(seu))

################################################################################
# Cell Cycle Scoring -- same as your code: set.ident = TRUE
################################################################################
cat("\nCell cycle scoring...\n")

# Load markers (human by default from Seurat)
s.genes   <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes

# Convert to mouse format if needed -- auto-detect UPPERCASE vs Capitalized
if (organism == "mouse") {
  sample_genes <- head(rownames(seu), 200)
  n_upper <- sum(sample_genes == toupper(sample_genes))
  n_cap   <- sum(sample_genes == paste0(toupper(substring(sample_genes, 1, 1)),
                                        substring(tolower(sample_genes), 2)))
  if (n_upper > n_cap) {
    s.genes   <- toupper(s.genes)
    g2m.genes <- toupper(g2m.genes)
    cat("  Mouse markers converted to: UPPERCASE\n")
  } else {
    s.genes   <- paste0(toupper(substring(tolower(s.genes), 1, 1)),   substring(tolower(s.genes), 2))
    g2m.genes <- paste0(toupper(substring(tolower(g2m.genes), 1, 1)), substring(tolower(g2m.genes), 2))
    cat("  Mouse markers converted to: Capitalized\n")
  }
}

s_present   <- sum(s.genes %in% rownames(seu))
g2m_present <- sum(g2m.genes %in% rownames(seu))
cat(sprintf("  S markers in data:   %d / %d\n", s_present,   length(s.genes)))
cat(sprintf("  G2M markers in data: %d / %d\n", g2m_present, length(g2m.genes)))

# Score -- set.ident = TRUE (same as your code)
seu <- tryCatch(
  {
    CellCycleScoring(seu, s.features = s.genes, g2m.features = g2m.genes, set.ident = TRUE)
  },
  error = function(e) {
    cat(sprintf("  Scoring error: %s\n", e$message))
    cat("  Assigning placeholder: all cells G1.\n")
    seu$Phase <- factor(rep("G1", ncol(seu)), levels = c("G1", "S", "G2M"))
    seu
  }
)

# Print phase distribution (same info as your code's df)
cat("\n  Phase distribution:\n")
print(table(seu$Phase))

################################################################################
# PCA on HVGs -> DimPlot colored by Phase -> save PNG
################################################################################
cat("\nRunning PCA on HVGs...\n")
seu <- RunPCA(seu, features = hvg_genes, npcs = 30, verbose = FALSE)

# Plot
p <- DimPlot(seu, reduction = "pca", group.by = "Phase") +
  ggtitle(sprintf("%s: PCA -- Cell Cycle Phase", dataset_name)) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))

png(file.path(outdir, paste0(dataset_name, "_pca_cell_cycle_phase.png")),
    width = 1200, height = 800, res = 150)
print(p)
dev.off()

cat(sprintf("\n  Saved: %s_pca_cell_cycle_phase.png\n", dataset_name))
cat("================================================================================\n\n")
