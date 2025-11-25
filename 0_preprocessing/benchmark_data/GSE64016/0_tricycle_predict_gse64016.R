library(SingleCellExperiment)
library(tricycle)
library(org.Hs.eg.db)
library(scater)

# Define the file path for the TPM data
tpm_file <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/GSE64016_H1andFUCCI/GSE64016_H1andFUCCI_normalized_EC.csv"

# Load the TPM data
tpm_data <- read.csv(tpm_file, row.names = 1, check.names = FALSE)

# Transpose the data so that genes are rows and cells are columns
tpm_matrix <- as.matrix(tpm_data)

# Create SingleCellExperiment object
sce_obj <- SingleCellExperiment(assays = list(counts = tpm_matrix))

# Filter for valid gene symbols
valid_genes <- rownames(sce_obj) %in% keys(org.Hs.eg.db, keytype = "SYMBOL")
sce_obj <- sce_obj[valid_genes, ]

# Normalize the counts (log transformation for Tricycle)
sce_obj <- logNormCounts(sce_obj)

# Check the SCE object
print(sce_obj)

# Project the data into cycle space
sce_obj <- project_cycle_space(sce_obj, exprs_values = "logcounts", name = "tricycleEmbedding",
                               gname.type = "SYMBOL", species = "human")

# Estimate the cycle position
sce_obj <- estimate_cycle_position(sce_obj, gname.type = "SYMBOL", species = "human")

# Add Schwabe stage estimation
sce_obj <- estimate_Schwabe_stage(sce_obj, gname.type = "SYMBOL", species = "human")

# Extract cell cycle data
cell_cycle_data <- data.frame(
  CellName = colnames(sce_obj),
  TotalUMIs = colSums(assays(sce_obj)$counts),
  CyclePosition = colData(sce_obj)$tricyclePosition,
  CCStage = colData(sce_obj)$CCStage
)

# Save the results to a CSV file
output_csv_path <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/Benchmark_data/prediction_by_tricycle_GSE64016.csv"
write.csv(cell_cycle_data, output_csv_path, row.names = FALSE)

print(paste("Cell cycle prediction saved to:", output_csv_path))
