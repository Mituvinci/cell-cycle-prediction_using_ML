library(SingleCellExperiment)
library(tricycle)
library(scater)
library(dplyr)

###############################################
# 1 — Detect gene name type (SYMBOL vs ENSEMBL)
###############################################
detect_gene_type <- function(genes) {
  if (all(grepl("^ENSG", genes))) return("ENSEMBL")
  if (all(grepl("^ENSMUSG", genes))) return("ENSEMBL")
  return("SYMBOL")
}

###############################################
# 2 — Paths
###############################################
input_file <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Dolly/BuettnerESCData/Buettner_mESC_benchmark_clean.csv"

output_file <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Dolly/BuettnerESCData/predicted_by_tricycle_Buettner.csv"

###############################################
# 3 — Load CSV (already normalized)
###############################################
df <- read.csv(input_file, check.names = FALSE)

# Extract ground truth from Cell_ID
df$GroundTruth <- sub("_.*", "", df$Cell_ID)

# Remove duplicated gene columns
df <- df[, !duplicated(colnames(df))]

###############################################
# 4 — Build expression matrix
###############################################
mat <- df %>% select(-Cell_ID, -GroundTruth)
rownames(mat) <- df$Cell_ID

# Transpose → genes x cells
mat <- t(mat)

###############################################
# 5 — Build SingleCellExperiment
###############################################
sce <- SingleCellExperiment(
  assays = list(counts = as.matrix(mat))
)

# Normalize (log transform) — REQUIRED by Tricycle
sce <- logNormCounts(sce)

###############################################
# 6 — Detect gene type
###############################################
gene_type <- detect_gene_type(rownames(sce))
message("Detected gene type: ", gene_type)

###############################################
# 7 — Run Tricycle (species = mouse)
###############################################
sce <- project_cycle_space(
  sce,
  exprs_values = "logcounts",
  gname.type = gene_type,
  species = "mouse"
)

sce <- estimate_cycle_position(
  sce,
  gname.type = gene_type,
  species = "mouse"
)

sce <- estimate_Schwabe_stage(
  sce,
  gname.type = gene_type,
  species = "mouse"
)

###############################################
# 8 — Build final output
###############################################
result_df <- data.frame(
  CellID = colnames(sce),                      # as you requested
  Predicted = sce$CCStage,                     # G1 / S / G2M
  GroundTruth = df$GroundTruth[match(colnames(sce), df$Cell_ID)]
)

###############################################
# 9 — Save output
###############################################
write.csv(result_df, output_file, row.names = FALSE)

cat("🔥 DONE! Tricycle predictions saved to:\n", output_file, "\n")
