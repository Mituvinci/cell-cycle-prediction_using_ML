library(Seurat)
library(Signac)
library(tricycle)
library(scater)
library(SingleCellExperiment)

# -----------------------------
# Detect gene naming convention
# -----------------------------
detect_gene_type <- function(gene_names) {
  # ENSEMBL mouse starts with "ENSMUSG", human with "ENSG"
  if (all(grepl("^ENSG", gene_names))) return("ENSEMBL")
  if (all(grepl("^ENSMUSG", gene_names))) return("ENSEMBL")
  
  # Mixed — treat as SYMBOL
  return("SYMBOL")
}

# -------------------------------------------------------
# Main function: works for BOTH HUMAN and MOUSE datasets
# -------------------------------------------------------
makePredictedCCCSV <- function(path, sample_name, species = c("human", "mouse")) {
  
  species <- match.arg(species)
  message("=== Running Tricycle for sample: ", sample_name, " (", species, ") ===")
  
  inputdata.10x <- Read10X_h5(path)
  if ("Gene Expression" %in% names(inputdata.10x)) {
    rna_counts <- inputdata.10x$`Gene Expression`
  } else {
    rna_counts <- inputdata.10x
  }
  
  # Build SingleCellExperiment
  sce <- SingleCellExperiment(assays = list(counts = rna_counts))
  sce <- logNormCounts(sce)
  
  # Detect whether SYMBOL or ENSEMBL
  gtype <- detect_gene_type(rownames(sce))
  message("Detected gene name type: ", gtype)
  
  # ------------------------------------
  # Run Tricycle projection
  # ------------------------------------
  sce <- project_cycle_space(
    sce,
    exprs_values = "logcounts",
    name = "tricycleEmbedding",
    gname.type = gtype,
    species = species
  )
  
  # ------------------------------------
  # Estimate cycle position
  # ------------------------------------
  sce <- estimate_cycle_position(
    sce,
    gname.type = gtype,
    species = species
  )
  
  # ------------------------------------
  # Estimate Schwabe stages (G1, S, G2/M)
  # ------------------------------------
  sce <- estimate_Schwabe_stage(
    sce,
    gname.type = gtype,
    species = species
  )
  
  # ------------------------------------
  # Save output
  # ------------------------------------
  out_df <- data.frame(
    Cell = colnames(sce),
    tricyclePosition = sce$tricyclePosition,
    CCStage = sce$CCStage
  )
  
  out_path <- paste0(
    "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/tricycle_",
    sample_name,
    ".csv"
  )
  
  write.csv(out_df, out_path, row.names = FALSE)
  
  message("Saved: ", out_path)
  
  return(out_df)
}



makePredictedCCCSV(
  "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Review/pbmc_healthy_human/filtered_feature_bc_matrix.h5",
  "pbmc_healthy_human",
  species = "human"
)

makePredictedCCCSV(
  "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Review/10-k-brain-cells_healthy_mouse/filtered_feature_bc_matrix.h5",
  "mouse_brain_10k",
  species = "mouse"
)


