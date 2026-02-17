#!/usr/bin/env Rscript
################################################################################
# detect_cell_cycle_effect.R
#
# Combined single script: Load -> Filter -> Normalize -> Cell Cycle Score -> PCA
#
# Purpose:
#   Detect whether a dataset has cell cycle effect by checking if G1, S, G2M
#   phases separate in PCA space. If marker genes are expressed and the data
#   has cell cycle signal, the 3 phases will form distinct clusters in PCA.
#   If not expressed / no signal, phases will be mixed together.
#   PCA plot is ALWAYS generated regardless of marker presence or scoring outcome.
#
# Arguments:
#   args[1] = input_path    : 10x directory, .h5, .csv, .txt, or .tsv
#   args[2] = output_dir    : output directory (created if missing)
#   args[3] = organism      : "human" or "mouse" (default: human)
#   args[4] = dataset_name  : prefix for all output files
#
# Outputs:
#   {dataset_name}_pca_cell_cycle_phase.png   -- PCA colored by G1 / S / G2M
#   {dataset_name}_pca_cell_cycle_scores.png  -- PCA colored by S.Score and G2M.Score
#   {dataset_name}_cell_cycle_summary.json    -- marker counts, correlations, verdict
################################################################################

library(Seurat)
library(jsonlite)
library(ggplot2)
library(patchwork)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript detect_cell_cycle_effect.R <input_path> <output_dir> [organism] [dataset_name]")
}

input_path   <- args[1]
outdir       <- args[2]
organism     <- if (length(args) >= 3) tolower(args[3]) else "human"
dataset_name <- if (length(args) >= 4) args[4] else "dataset"

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================================\n")
cat(sprintf("  Cell Cycle Effect Detection: %s (%s)\n", dataset_name, organism))
cat("================================================================================\n\n")

################################################################################
# STAGE 1: Load data -- auto-detect format
################################################################################
cat("--- STAGE 1: Loading data ---\n")

if (dir.exists(input_path)) {
  cat("  Format detected: 10x directory\n")
  counts <- Read10X(input_path)

} else if (grepl("\\.h5$", input_path, ignore.case = TRUE)) {
  cat("  Format detected: HDF5\n")
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("hdf5r package is required for .h5 files.")
  }
  counts <- Read10X_h5(input_path)

} else if (grepl("\\.(csv|txt|tsv)$", input_path, ignore.case = TRUE)) {
  cat("  Format detected: CSV / TXT / TSV (genes as rows, cells as columns)\n")
  sep <- if (grepl("\\.csv$", input_path, ignore.case = TRUE)) "," else "\t"
  counts <- read.table(input_path,
                       sep             = sep,
                       header          = TRUE,
                       row.names       = 1,
                       check.names     = FALSE,
                       stringsAsFactors = FALSE)
  counts <- as.matrix(counts)
  cat(sprintf("  Read: %d genes x %d cells\n", nrow(counts), ncol(counts)))

} else {
  stop("Unsupported format. Provide a 10x directory, .h5, .csv, .txt, or .tsv file.")
}

seu <- CreateSeuratObject(counts   = counts,
                          project  = dataset_name,
                          min.cells   = 3,
                          min.features = 200)

cat(sprintf("  Seurat object: %d cells, %d genes\n", ncol(seu), nrow(seu)))

################################################################################
# STAGE 2: QC filter -- mitochondrial percentage only
# (data is already cell-filtered; we only remove high-MT cells)
################################################################################
cat("\n--- STAGE 2: Mitochondrial QC filter ---\n")

# Auto-detect MT gene naming in this dataset
if (any(grepl("^MT-", rownames(seu)))) {
  mt_pattern <- "^MT-"
} else if (any(grepl("^Mt-", rownames(seu)))) {
  mt_pattern <- "^Mt-"
} else {
  mt_pattern <- "^MT-"
  cat("  NOTE: No mitochondrial genes found in data. percent.mt will be 0.\n")
}
cat(sprintf("  MT pattern used: %s\n", mt_pattern))

seu[["percent.mt"]] <- PercentageFeatureSet(seu, pattern = mt_pattern)
cat(sprintf("  Median percent.mt: %.2f%%\n", median(seu$percent.mt)))

percent_mt_max <- 20
cells_before <- ncol(seu)
seu <- subset(seu, subset = percent.mt < percent_mt_max)
cells_after <- ncol(seu)
cat(sprintf("  Filter (percent.mt < %d%%): %d -> %d cells  (%.1f%% removed)\n",
            percent_mt_max, cells_before, cells_after,
            100 * (cells_before - cells_after) / cells_before))

################################################################################
# STAGE 3: Normalize + HVG + Cell Cycle Scoring + PCA + Save Plots
################################################################################
cat("\n--- STAGE 3: Normalize, Score, PCA ---\n")

# -- Normalize --
seu <- NormalizeData(seu, normalization.method = "LogNormalize",
                     scale.factor = 10000, verbose = FALSE)

# -- Highly Variable Genes --
seu <- FindVariableFeatures(seu, selection.method = "vst",
                            nfeatures = 2000, verbose = FALSE)
hvg_genes <- VariableFeatures(seu)

if (length(hvg_genes) < 100) {
  cat(sprintf("  Only %d HVGs found -- using all genes instead.\n", length(hvg_genes)))
  hvg_genes <- rownames(seu)
}
cat(sprintf("  HVGs used: %d\n", length(hvg_genes)))

# -- Cell Cycle Markers --
# Start from Seurat built-in human markers, then convert format to match data
s.genes   <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes

if (organism == "mouse") {
  # Mouse genes can be Capitalized (Cdc45) or UPPERCASE (CDC45) depending on dataset.
  # Detect which format this dataset uses and convert markers accordingly.
  sample_genes <- head(rownames(seu), 200)
  n_upper <- sum(sample_genes == toupper(sample_genes))
  n_capitalized <- sum(sample_genes == paste0(toupper(substring(sample_genes, 1, 1)),
                                              substring(tolower(sample_genes), 2)))

  if (n_upper > n_capitalized) {
    s.genes   <- toupper(s.genes)
    g2m.genes <- toupper(g2m.genes)
    cat("  Mouse gene format: UPPERCASE -- markers converted to UPPERCASE\n")
  } else {
    s.genes   <- paste0(toupper(substring(tolower(s.genes), 1, 1)),   substring(tolower(s.genes), 2))
    g2m.genes <- paste0(toupper(substring(tolower(g2m.genes), 1, 1)), substring(tolower(g2m.genes), 2))
    cat("  Mouse gene format: Capitalized -- markers converted to Capitalized\n")
  }
}

# Filter markers to only those present in the dataset
s.genes   <- s.genes[s.genes %in% rownames(seu)]
g2m.genes <- g2m.genes[g2m.genes %in% rownames(seu)]
cat(sprintf("  S-phase markers in data:   %d / %d\n", length(s.genes),   length(cc.genes$s.genes)))
cat(sprintf("  G2M-phase markers in data: %d / %d\n", length(g2m.genes), length(cc.genes$g2m.genes)))

# -- Join layers (Seurat v5 fix) --
# In Seurat v5, layers can be split and CellCycleScoring fails to find features.
# JoinLayers merges them so all features are accessible.
if (inherits(seu[["RNA"]], "Assay5")) {
  cat("  Seurat v5 detected -- joining RNA layers\n")
  seu[["RNA"]] <- JoinLayers(seu[["RNA"]])
}

# -- Cell Cycle Scoring --
# Always attempted. If it fails, placeholder values are assigned so that
# PCA is still generated (the plot itself is the evidence).
scoring_success <- FALSE

seu_scored <- tryCatch(
  {
    withCallingHandlers(
      {
        CellCycleScoring(seu,
                         s.features   = s.genes,
                         g2m.features = g2m.genes,
                         set.ident    = FALSE)
      },
      warning = function(w) {
        cat(sprintf("  Warning during scoring: %s\n", w$message))
        invokeRestart("muffleWarning")
      }
    )
  },
  error = function(e) {
    cat(sprintf("  Scoring error: %s\n", e$message))
    NULL
  }
)

if (!is.null(seu_scored) &&
    all(c("S.Score", "G2M.Score", "Phase") %in% colnames(seu_scored@meta.data))) {
  seu <- seu_scored
  scoring_success <- TRUE
  cat("  Cell cycle scoring: SUCCESS\n")
  cat("  Phase distribution:\n")
  print(table(seu$Phase))
} else {
  # Assign placeholder -- all cells labeled G1, scores = 0
  seu$S.Score   <- 0
  seu$G2M.Score <- 0
  seu$Phase     <- factor(rep("G1", ncol(seu)), levels = c("G1", "S", "G2M"))
  cat("  Scoring did not succeed -- placeholder assigned (all G1, scores 0).\n")
  cat("  PCA will still be generated.\n")
}

# -- Scale + PCA + UMAP (always runs) --
seu <- ScaleData(seu, features = hvg_genes, verbose = FALSE)
seu <- RunPCA(seu, features = hvg_genes, npcs = 30, verbose = FALSE)
seu <- RunUMAP(seu, dims = 1:20, verbose = FALSE)

################################################################################
# SAVE METADATA TABLE (orig.ident, nCount_RNA, nFeature_RNA, S.Score, G2M.Score, Phase, old.ident)
################################################################################
cat("\n--- Saving cell metadata table ---\n")

meta_cols <- c("orig.ident", "nCount_RNA", "nFeature_RNA", "S.Score", "G2M.Score", "Phase", "old.ident")
# old.ident is created by Seurat automatically (same as orig.ident initially)
if (!"old.ident" %in% colnames(seu@meta.data)) {
  seu$old.ident <- seu$orig.ident
}

meta_table <- seu@meta.data[, meta_cols, drop = FALSE]
meta_csv <- file.path(outdir, paste0(dataset_name, "_cell_metadata.csv"))
write.csv(meta_table, meta_csv, row.names = TRUE)
cat(sprintf("  Saved: %s  (%d cells x %d columns)\n", basename(meta_csv), nrow(meta_table), ncol(meta_table)))

################################################################################
# SAVE PLOTS -- always, no conditions
################################################################################
cat("\n--- Saving plots ---\n")

################################################################################
# STAGE 4: Cell Type Marker Verification (For Dr. Hu)
################################################################################
cat("\n--- STAGE 4: Checking PBMC Cell Type Markers ---\n")

# Markers requested by Dr. Hu
pbmc_markers <- c("PAX5", "BCL11B", "CEBPA")
# Check which markers exist in your specific dataset
existing_markers <- intersect(pbmc_markers, rownames(seu))

if (length(existing_markers) > 0) {
  p_id <- FeaturePlot(seu, reduction = "pca", features = existing_markers, ncol = 3) +
    plot_annotation(title = sprintf("%s: Cell Type Markers (B/T/Mono)", dataset_name))

  png(file.path(outdir, paste0(dataset_name, "_cell_type_markers.png")),
      width = 1800, height = 600, res = 150)
  print(p_id)
  dev.off()
  cat(sprintf("  Saved: %s_cell_type_markers.png\n", dataset_name))
} else {
  cat("  Requested markers not found in this dataset; skipping plot.\n")
}


# Plot 1: PCA colored by Phase (G1 / S / G2M)
p1 <- DimPlot(seu, reduction = "pca", group.by = "Phase") +
  ggtitle(sprintf("%s: PCA -- Cell Cycle Phase", dataset_name)) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))

png(file.path(outdir, paste0(dataset_name, "_pca_cell_cycle_phase.png")),
    width = 1200, height = 800, res = 150)
print(p1)
dev.off()
cat(sprintf("  Saved: %s_pca_cell_cycle_phase.png\n", dataset_name))

# Plot 2: PCA colored by S.Score and G2M.Score (two panels)
p2 <- FeaturePlot(seu, reduction = "pca",
                  features = c("S.Score", "G2M.Score"), ncol = 2) +
  plot_annotation(title = sprintf("%s: Cell Cycle Scores on PCA", dataset_name),
                  theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14)))

png(file.path(outdir, paste0(dataset_name, "_pca_cell_cycle_scores.png")),
    width = 1400, height = 700, res = 150)
print(p2)
dev.off()
cat(sprintf("  Saved: %s_pca_cell_cycle_scores.png\n", dataset_name))

# Plot 3: UMAP colored by Phase (G1 / S / G2M)
p3 <- DimPlot(seu, reduction = "umap", group.by = "Phase") +
  ggtitle(sprintf("%s: UMAP -- Cell Cycle Phase", dataset_name)) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))

png(file.path(outdir, paste0(dataset_name, "_umap_cell_cycle_phase.png")),
    width = 1200, height = 800, res = 150)
print(p3)
dev.off()
cat(sprintf("  Saved: %s_umap_cell_cycle_phase.png\n", dataset_name))

# Plot 4: UMAP colored by S.Score and G2M.Score (two panels)
p4 <- FeaturePlot(seu, reduction = "umap",
                  features = c("S.Score", "G2M.Score"), ncol = 2) +
  plot_annotation(title = sprintf("%s: Cell Cycle Scores on UMAP", dataset_name),
                  theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14)))

png(file.path(outdir, paste0(dataset_name, "_umap_cell_cycle_scores.png")),
    width = 1400, height = 700, res = 150)
print(p4)
dev.off()
cat(sprintf("  Saved: %s_umap_cell_cycle_scores.png\n", dataset_name))

################################################################################
# SAVE SUMMARY JSON
################################################################################
cat("\n--- Saving summary ---\n")

cor_s   <- NA_real_
cor_g2m <- NA_real_

if (scoring_success) {
  pc1     <- seu@reductions$pca@cell.embeddings[, 1]
  cor_s   <- cor(pc1, seu$S.Score)
  cor_g2m <- cor(pc1, seu$G2M.Score)
  cat(sprintf("  PC1 correlation with S.Score:   %.3f\n", cor_s))
  cat(sprintf("  PC1 correlation with G2M.Score: %.3f\n", cor_g2m))
}

# Verdict: cell cycle effect present if PC1 correlates strongly with scores
has_cell_cycle_effect <- if (scoring_success) {
  abs(cor_s) > 0.3 || abs(cor_g2m) > 0.3
} else {
  FALSE
}

summary_json <- list(
  dataset_name         = dataset_name,
  organism             = organism,
  cells_after_filter   = ncol(seu),
  n_hvgs               = length(hvg_genes),
  s_markers_present    = s_present,
  s_markers_total      = length(s.genes),
  g2m_markers_present  = g2m_present,
  g2m_markers_total    = length(g2m.genes),
  scoring_success      = scoring_success,
  cell_cycle_effect    = has_cell_cycle_effect,
  pc1_cor_s_score      = cor_s,
  pc1_cor_g2m_score    = cor_g2m,
  phase_distribution   = if (scoring_success) as.list(table(seu$Phase)) else NULL,
  pca_phase_plot       = paste0(dataset_name, "_pca_cell_cycle_phase.png"),
  pca_scores_plot      = paste0(dataset_name, "_pca_cell_cycle_scores.png"),
  umap_phase_plot      = paste0(dataset_name, "_umap_cell_cycle_phase.png"),
  umap_scores_plot     = paste0(dataset_name, "_umap_cell_cycle_scores.png")
)

write_json(summary_json,
           file.path(outdir, paste0(dataset_name, "_cell_cycle_summary.json")),
           pretty = TRUE,
           auto_unbox = TRUE)

cat(sprintf("\n================================================================================\n"))
cat(sprintf("  %s -- DONE\n", dataset_name))
cat(sprintf("  Cell cycle effect detected: %s\n", if (has_cell_cycle_effect) "YES" else "NO"))
cat(sprintf("================================================================================\n\n"))
