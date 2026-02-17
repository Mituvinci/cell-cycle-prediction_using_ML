#!/usr/bin/env Rscript
################################################################################
# PCA with Cell Cycle Scoring from Consensus Labeled CSV Files
################################################################################
#
# Purpose:
#   Takes already-processed consensus labeled CSV files (from final_training_data_*)
#   and applies ONLY Stage 3 (Normalize + HVG + Cell Cycle Scoring + PCA + Plots)
#
# Usage:
#   Rscript pca_with_scoring_from_csv.R <dataset_name>
#
# Example:
#   Rscript pca_with_scoring_from_csv.R pbmc_human
#   Rscript pca_with_scoring_from_csv.R mouse_brain
#
# Input:
#   Consensus labeled CSV files with structure:
#   - gex_barcode (cell ID)
#   - Predicted (phase: G1, S, G2M)
#   - Gene columns (UPPERCASE)
#
# Output:
#   - {dataset}_pca_phase.png
#   - {dataset}_pca_scores.png
#   - {dataset}_summary.json
#
################################################################################

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(jsonlite)
})

################################################################################
# PARSE COMMAND LINE ARGUMENTS
################################################################################
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  cat("\nUsage: Rscript pca_with_scoring_from_csv.R <dataset_name>\n")
  cat("\nAvailable datasets:\n")
  cat("  - pbmc_human\n")
  cat("  - mouse_brain\n")
  cat("  - nestorova\n")
  cat("  - gse75748_hpsc\n")
  cat("  - reh\n")
  cat("  - sup_b15\n")
  cat("\n")
  quit(status = 1)
}

dataset_name <- args[1]
cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("PCA WITH CELL CYCLE SCORING: %s\n", dataset_name))
cat(sprintf("%s\n\n", paste(rep("=", 80), collapse = "")))

################################################################################
# DEFINE FILE PATHS
################################################################################
base_dir <- "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction"
data_dir <- "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data"

# Dataset path mapping
dataset_paths <- list(
  pbmc_human       = file.path(base_dir, "1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv"),
  mouse_brain      = file.path(base_dir, "1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data.csv"),
  nestorova        = file.path(base_dir, "1_consensus_labeling/assign/final_training_data_nestorova/nestorova_training_data.csv"),
  gse75748_hpsc    = file.path(base_dir, "1_consensus_labeling/assign/final_training_data_gse75748/gse75748_training_data.csv"),
  reh              = file.path(data_dir, "old/filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv"),
  sup_b15          = file.path(data_dir, "old/filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv")
)

if (!dataset_name %in% names(dataset_paths)) {
  cat(sprintf("ERROR: Unknown dataset '%s'\n", dataset_name))
  cat("Available datasets:\n")
  for (name in names(dataset_paths)) {
    cat(sprintf("  - %s\n", name))
  }
  quit(status = 1)
}

csv_path <- dataset_paths[[dataset_name]]

if (!file.exists(csv_path)) {
  cat(sprintf("ERROR: File not found: %s\n", csv_path))
  quit(status = 1)
}

# Output directory
output_dir <- file.path(base_dir, "Has_cell_cycle_effect_or_not/pca_results_consensus_labeled", dataset_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("Input CSV:     %s\n", csv_path))
cat(sprintf("Output dir:    %s\n\n", output_dir))

################################################################################
# LOAD CSV AND CREATE SEURAT OBJECT
################################################################################
cat("--- Loading CSV file ---\n")

# Read CSV WITHOUT setting row.names first (to handle duplicates)
data <- read.csv(csv_path, stringsAsFactors = FALSE)
cat(sprintf("  Initial load: %d rows, %d columns\n", nrow(data), ncol(data)))

# Check for duplicate rows (REH and SUP have this issue)
n_before <- nrow(data)
data <- data[!duplicated(data), ]
n_after <- nrow(data)

if (n_before > n_after) {
  cat(sprintf("  Removed %d duplicate rows (%d -> %d)\n", n_before - n_after, n_before, n_after))
}

# Now set row names to first column (cell barcodes)
rownames(data) <- data[, 1]
data <- data[, -1]  # Remove first column
cat(sprintf("  Loaded: %d cells, %d columns\n", nrow(data), ncol(data)))

# Identify metadata columns
metadata_cols <- c("Predicted", "Phase", "phase", "Cell_ID", "cell", "CellID", "dataset", "Dataset")
metadata_present <- intersect(metadata_cols, colnames(data))

# Extract gene expression matrix (transpose to genes x cells)
gene_cols <- setdiff(colnames(data), metadata_present)
expr_matrix <- t(data[, gene_cols, drop = FALSE])
cat(sprintf("  Gene columns: %d\n", length(gene_cols)))

# Create Seurat object
seu <- CreateSeuratObject(counts = expr_matrix, project = dataset_name, min.cells = 0, min.features = 0)

# Add consensus phase labels to metadata if available
if ("Predicted" %in% metadata_present) {
  seu$Consensus_Phase <- data$Predicted
  cat("  Added consensus phase labels (Predicted column)\n")
}

cat(sprintf("  Seurat object created: %d cells, %d genes\n", ncol(seu), nrow(seu)))

################################################################################
# DETECT ORGANISM FROM GENE NAMES
################################################################################
cat("\n--- Detecting organism ---\n")

sample_genes <- head(rownames(seu), 200)

# Mouse genes: start with uppercase letter, rest lowercase (e.g., Gapdh)
# Human genes: all uppercase (e.g., GAPDH)
# But mouse genes can also be UPPERCASE in some datasets

# Count patterns
n_all_upper <- sum(sample_genes == toupper(sample_genes))
n_capitalized <- sum(grepl("^[A-Z][a-z]", sample_genes))

# Common mouse-specific genes
mouse_markers <- c("Gapdh", "Actb", "Rpl13a", "Tubb5", "Hsp90ab1")
human_markers <- c("GAPDH", "ACTB", "RPL13A", "TUBB", "HSP90AB1")

mouse_present <- sum(toupper(sample_genes) %in% toupper(mouse_markers))
human_present <- sum(sample_genes %in% human_markers)

if (n_capitalized > 50 || grepl("mouse", dataset_name, ignore.case = TRUE)) {
  organism <- "mouse"
  cat("  Organism: MOUSE (detected from gene naming pattern)\n")
} else {
  organism <- "human"
  cat("  Organism: HUMAN (detected from gene naming pattern)\n")
}

################################################################################
# STAGE 3: Normalize + HVG + Cell Cycle Scoring + PCA + Plots
################################################################################
cat("\n--- STAGE 3: Normalize, Score, PCA ---\n")

# -- Normalize --
cat("  Normalizing data (LogNormalize, scale.factor=10000)...\n")
seu <- NormalizeData(seu, normalization.method = "LogNormalize",
                     scale.factor = 10000, verbose = FALSE)

# -- Highly Variable Genes --
cat("  Finding highly variable genes (top 2000)...\n")
seu <- FindVariableFeatures(seu, selection.method = "vst",
                            nfeatures = 2000, verbose = FALSE)
hvg_genes <- VariableFeatures(seu)

if (length(hvg_genes) < 100) {
  cat(sprintf("  Only %d HVGs found -- using all genes instead.\n", length(hvg_genes)))
  hvg_genes <- rownames(seu)
}
cat(sprintf("  HVGs used: %d\n", length(hvg_genes)))

# -- Cell Cycle Markers --
cat("  Preparing cell cycle markers...\n")

# Start from Seurat built-in human markers
s.genes   <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes

if (organism == "mouse") {
  # Detect gene format in this dataset
  sample_genes <- head(rownames(seu), 200)
  n_upper <- sum(sample_genes == toupper(sample_genes))
  n_capitalized <- sum(sample_genes == paste0(toupper(substring(sample_genes, 1, 1)),
                                              tolower(substring(sample_genes, 2))))

  if (n_upper > n_capitalized) {
    # UPPERCASE format
    s.genes   <- toupper(s.genes)
    g2m.genes <- toupper(g2m.genes)
    cat("  Mouse gene format: UPPERCASE -- markers converted to UPPERCASE\n")
  } else {
    # Capitalized format (e.g., Gapdh)
    s.genes   <- paste0(toupper(substring(tolower(s.genes), 1, 1)),   substring(tolower(s.genes), 2))
    g2m.genes <- paste0(toupper(substring(tolower(g2m.genes), 1, 1)), substring(tolower(g2m.genes), 2))
    cat("  Mouse gene format: Capitalized -- markers converted to Capitalized\n")
  }
}

# Count markers present in this dataset
s_present   <- sum(s.genes %in% rownames(seu))
g2m_present <- sum(g2m.genes %in% rownames(seu))
cat(sprintf("  S-phase markers in data:   %d / %d\n", s_present,   length(s.genes)))
cat(sprintf("  G2M-phase markers in data: %d / %d\n", g2m_present, length(g2m.genes)))

# -- Cell Cycle Scoring --
cat("  Running cell cycle scoring...\n")
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
  cat("  Seurat Phase distribution:\n")
  print(table(seu$Phase))

  # Compare with consensus labels if available
  if ("Consensus_Phase" %in% colnames(seu@meta.data)) {
    cat("\n  Consensus Phase distribution (from CSV):\n")
    print(table(seu$Consensus_Phase))

    # Calculate agreement
    agreement <- sum(seu$Phase == seu$Consensus_Phase) / ncol(seu) * 100
    cat(sprintf("\n  Agreement between Seurat and Consensus: %.1f%%\n", agreement))
  }
} else {
  # Assign placeholder
  seu$S.Score   <- 0
  seu$G2M.Score <- 0
  seu$Phase     <- factor(rep("G1", ncol(seu)), levels = c("G1", "S", "G2M"))
  cat("  Scoring did not succeed -- placeholder assigned (all G1, scores 0).\n")
}

# -- Scale + PCA --
cat("  Scaling data and running PCA...\n")
seu <- ScaleData(seu, features = hvg_genes, verbose = FALSE)
seu <- RunPCA(seu, features = hvg_genes, npcs = 30, verbose = FALSE)

# Calculate PC1 correlation with scores
pc1_scores <- seu@reductions$pca@cell.embeddings[, 1]
cor_s <- cor(pc1_scores, seu$S.Score)
cor_g2m <- cor(pc1_scores, seu$G2M.Score)

cat(sprintf("  PC1 correlation with S.Score:   %.3f\n", cor_s))
cat(sprintf("  PC1 correlation with G2M.Score: %.3f\n", cor_g2m))

# Determine cell cycle effect
cell_cycle_effect <- (abs(cor_s) > 0.3 || abs(cor_g2m) > 0.3)
cat(sprintf("\n  Cell cycle effect detected: %s\n", ifelse(cell_cycle_effect, "YES", "NO")))

################################################################################
# SAVE PLOTS
################################################################################
cat("\n--- Saving plots ---\n")

# Plot 1: PCA colored by Seurat Phase
p1 <- DimPlot(seu, reduction = "pca", group.by = "Phase", pt.size = 1.5) +
  ggtitle(paste0(dataset_name, " - PCA colored by Seurat Phase")) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

plot1_file <- file.path(output_dir, paste0(dataset_name, "_pca_seurat_phase.png"))
ggsave(plot1_file, plot = p1, width = 8, height = 6, dpi = 300)
cat(sprintf("  Saved: %s\n", basename(plot1_file)))

# Plot 2: PCA colored by S.Score and G2M.Score
if (scoring_success) {
  p2a <- FeaturePlot(seu, reduction = "pca", features = "S.Score", pt.size = 1.5) +
    ggtitle(paste0(dataset_name, " - S.Score")) +
    theme_bw()

  p2b <- FeaturePlot(seu, reduction = "pca", features = "G2M.Score", pt.size = 1.5) +
    ggtitle(paste0(dataset_name, " - G2M.Score")) +
    theme_bw()

  p2 <- cowplot::plot_grid(p2a, p2b, ncol = 2)

  plot2_file <- file.path(output_dir, paste0(dataset_name, "_pca_scores.png"))
  ggsave(plot2_file, plot = p2, width = 14, height = 6, dpi = 300)
  cat(sprintf("  Saved: %s\n", basename(plot2_file)))
}

# Plot 3: PCA colored by Consensus Phase (if available)
if ("Consensus_Phase" %in% colnames(seu@meta.data)) {
  p3 <- DimPlot(seu, reduction = "pca", group.by = "Consensus_Phase", pt.size = 1.5) +
    ggtitle(paste0(dataset_name, " - PCA colored by Consensus Phase")) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

  plot3_file <- file.path(output_dir, paste0(dataset_name, "_pca_consensus_phase.png"))
  ggsave(plot3_file, plot = p3, width = 8, height = 6, dpi = 300)
  cat(sprintf("  Saved: %s\n", basename(plot3_file)))

  # Plot 4: Side-by-side comparison
  p4 <- cowplot::plot_grid(p3, p1, ncol = 2, labels = c("Consensus Labels", "Seurat Scoring"))
  plot4_file <- file.path(output_dir, paste0(dataset_name, "_pca_comparison.png"))
  ggsave(plot4_file, plot = p4, width = 14, height = 6, dpi = 300)
  cat(sprintf("  Saved: %s\n", basename(plot4_file)))
}

################################################################################
# SAVE JSON SUMMARY
################################################################################
cat("\n--- Saving JSON summary ---\n")

summary_data <- list(
  dataset_name = dataset_name,
  organism = organism,
  cells = ncol(seu),
  genes = nrow(seu),
  n_hvgs = length(hvg_genes),
  s_markers_present = s_present,
  s_markers_total = length(s.genes),
  g2m_markers_present = g2m_present,
  g2m_markers_total = length(g2m.genes),
  scoring_success = scoring_success,
  cell_cycle_effect = cell_cycle_effect,
  pc1_cor_s_score = round(cor_s, 4),
  pc1_cor_g2m_score = round(cor_g2m, 4),
  seurat_phase_distribution = as.list(table(seu$Phase))
)

if ("Consensus_Phase" %in% colnames(seu@meta.data)) {
  summary_data$consensus_phase_distribution <- as.list(table(seu$Consensus_Phase))
  summary_data$agreement_percent <- round(sum(seu$Phase == seu$Consensus_Phase) / ncol(seu) * 100, 2)
}

summary_file <- file.path(output_dir, paste0(dataset_name, "_summary.json"))
write_json(summary_data, summary_file, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("  Saved: %s\n", basename(summary_file)))

################################################################################
# DONE
################################################################################
cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("COMPLETED: %s\n", dataset_name))
cat(sprintf("Results saved to: %s\n", output_dir))
cat(sprintf("%s\n\n", paste(rep("=", 80), collapse = "")))
