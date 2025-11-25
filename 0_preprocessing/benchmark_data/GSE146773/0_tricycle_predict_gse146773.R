library(Seurat)
library(Signac)
library(EnsDb.Hsapiens.v86)
library(dplyr)
library(ggplot2)
library(tricycle)
library(scattermore)
library(scater)
library(SingleCellExperiment)



# Example usage
path <- "D:/Halima's Data/Thesis_2/GSE146773_cell_cycle"
sample_name <- "GSE146773"
print(sample_name)
pathh5 <- paste0(path)

# Use Read10X instead of Read10X_h5 if your data is in the matrix format
rna_counts <- Read10X(pathh5)

sce_obj <- SingleCellExperiment(assays = list(counts = rna_counts))

print(sce_obj)

# Identify genes with rownames that do not start with "ENSG"
valid_genes <- !grepl("^ENSG", rownames(sce_obj))

# Subset the SingleCellExperiment object to keep only valid genes
sce_obj <- sce_obj[valid_genes,]

library(org.Hs.eg.db)
valid_symbols <- rownames(sce_obj) %in% keys(org.Hs.eg.db, keytype = "SYMBOL")
# Filter sce_obj for valid symbols if necessary
sce_obj <- sce_obj[valid_symbols, ]

# Check the filtered rownames
head(rownames(sce_obj))

# Correct calculation of TotalUMIs for each cell
# Ensure you're using the correct object for calculation
total_umis <- colSums(assays(sce_obj)$counts)

# Update the creation of the sample vector to match the number of columns in sce_obj
sample_vector <- rep("GSE146773", ncol(sce_obj))

# Correctly add TotalUMIs and sample name to the colData of sce_obj
colData(sce_obj)$TotalUMIs <- total_umis
colData(sce_obj)$sample <- sample_vector


# Normalize counts
sce_obj <- logNormCounts(sce_obj)

print(sce_obj)

# Assuming sce_obj has been successfully processed with project_cycle_space

# Estimate the cycle position - assuming this is your next step

sce_obj <- project_cycle_space(sce_obj, exprs_values = "logcounts", name = "tricycleEmbedding",
                                                          gname.type = "SYMBOL", species = 'human')
sce_obj <- estimate_cycle_position(sce_obj,gname.type = 'ENSEMBL', species = 'human')
names(colData(sce_obj))
print(sce_obj)
top2a.idx1 <- which(rowData(sce_obj)$Gene == 'Top2a')
print(length(top2a.idx1))
print(top2a.idx1)
print(length(sce_obj$tricyclePosition))
print(sce_obj$tricyclePosition)


# Since your rownames are in SYMBOL format, ensure gname.type aligns with this
sce_obj <- estimate_Schwabe_stage(sce_obj,
                                  gname.type = 'SYMBOL',  # Ensure alignment with your data format
                                  species = 'human')


# Assuming you now have cycle position and Schwabe stage in sce_obj,
# you can prepare your data frame for export.
# Assuming cycle_position is available and Schwabe_stage is not

# Check available colData names
print(names(colData(sce_obj)))

cell_cycle_data <- data.frame(
  CellName = colnames(sce_obj),
  TotalUMIs = colData(sce_obj)$TotalUMIs,
  Sample = colData(sce_obj)$sample,
  CyclePosition = colData(sce_obj)$tricyclePosition,  # Corrected to use the available data
  CCStage = colData(sce_obj)$CCStage  # Including CCStage as it's available
)



# Define the path for the output CSV file
output_csv_path <- paste0(path, "/prediction_by_tricycle_", sample_name, ".csv")

# Save the data to a CSV file
write.csv(cell_cycle_data, output_csv_path, row.names = FALSE)
print(paste0("Cell cycle prediction saved to ", output_csv_path))



