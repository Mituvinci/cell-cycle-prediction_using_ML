
#https://www.bioconductor.org/packages/release/bioc/vignettes/tricycle/inst/doc/tricycle.html

library(Seurat)
library(Signac)
library(EnsDb.Hsapiens.v86)
library(dplyr)
library(ggplot2)
library(tricycle)
library(scattermore)
library(scater)
library(SingleCellExperiment)

makePredcitedCCCSV <- function(path, sample_name_1) {
  print(sample_name_1)
  pathh5 <- paste0(path, "/filtered_feature_bc_matrix.h5")
  cat("H5 path ", pathh5, "\n")
  
  inputdata.10x <- Read10X_h5(pathh5)
  rna_counts <- inputdata.10x$`Gene Expression`

  
  sce_obj <- SingleCellExperiment(assays = list(counts = rna_counts))
  
  
  # Calculate TotalUMIs for each cell
  total_umis <- colSums(rna_counts)
  
  # Create a sample vector with the name "REH" for each cell
  sample_name <- rep("REH", ncol(rna_counts))
  
  # Add TotalUMIs and sample name to the colData of sce_obj
  colData(sce_obj)$TotalUMIs <- total_umis
  colData(sce_obj)$sample <- sample_name
  
  csv_path <- paste0(path, "/analysis/clustering/atac/graphclust/differential_expression.csv")
  cat("ENSMBLE path ", csv_path, "\n")
  
  df <- read.csv(csv_path, stringsAsFactors = FALSE)
  dictionary_to_replace <- setNames(df$`Feature.ID`, df$`Feature.Name`)
  
  missing_names <- rownames(sce_obj)[!rownames(sce_obj) %in% names(dictionary_to_replace)]
  if (length(missing_names) > 0) {
    missing_gene_ensmbl_df <- read.csv("D:/Halima's Data/Thesis_2/RCode/missing_gene_ensmbl.csv", stringsAsFactors = FALSE)
    missing_gene_ensmbl <- setNames(missing_gene_ensmbl_df[[2]], missing_gene_ensmbl_df[[1]])
    dictionary_to_replace <- c(dictionary_to_replace, missing_gene_ensmbl)
  }
  
  # Set the rowData for sce_obj
  rowData(sce_obj)$Gene <- rownames(sce_obj)
  rowData(sce_obj)$Accession <- dictionary_to_replace[rownames(sce_obj)]
  
  rownames(sce_obj) <- dictionary_to_replace[rownames(sce_obj)]
  sce_obj <- logNormCounts(sce_obj)
  colnames(sce_obj) <- paste0("REH_", colnames(sce_obj))
  
  # Print sce_obj to check the added metadata
  print(sce_obj)
  
  
  
  sce_obj <- project_cycle_space(sce_obj,exprs_values = "logcounts",name = "tricycleEmbedding",
                                 gname.type = c("ENSEMBL", "SYMBOL"),species = c('human'))
  
  
  
  sce_obj <- estimate_cycle_position(sce_obj,gname.type = 'ENSEMBL', species = 'human')
  names(colData(sce_obj))
  print(sce_obj)
  top2a.idx1 <- which(rowData(sce_obj)$Gene == 'Top2a')
  print(length(top2a.idx1))
  print(top2a.idx1)
  print(length(sce_obj$tricyclePosition))
  print(sce_obj$tricyclePosition)
  
  
  sce_obj <- estimate_Schwabe_stage(sce_obj,
                                    gname.type = 'ENSEMBL',
                                    species = 'human')
  names(colData(sce_obj))
  
  print(sce_obj$CCStage)
  
  print(typeof(sce_obj))
  
  print(sce_obj)
  
  # Access colData of neurosphere_example
  col_data <- colData(sce_obj)
  print(col_data)
  
  # Create a data frame with index/row names and desired attributes

  
  # Create a data frame with index/row names and desired attributes
  col_data_subset <- data.frame(Index = rownames(col_data),
                                TotalUMIs = col_data$TotalUMIs,
                                sample = col_data$sample,
                                tricyclePosition = col_data$tricyclePosition,
                                CCStage = col_data$CCStage)
  write.csv(col_data_subset, paste0("D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/tricycle_",sample_name_1,".csv"), row.names = FALSE)
  print(paste0("Saved lab_data_",sample_name_1,".csv"))
  
}

# Create an array with paths
paths_arr <- c("D:/Halima's Data/Thesis_2/1_GD428_21136_Hu_REH_Parental/outs", "D:/Halima's Data/Thesis_2/2_GD444_21136_Hu_Sup_Parental/outs")

#paths_arr <- c("D:/Halima's Data/Thesis_2/8_10-k-human-pbm-cs-multiome-v-1-0-chromium-controller-1-standard-2-0-0/outs")

# Iterate over each path in the paths_arr
for (path in paths_arr) {
  sample_name <- sub(".*/Thesis_2/(.*)/outs", "\\1", path)
  makePredcitedCCCSV(path, sample_name)
  
}

data(RevelioGeneList, package = "tricycle")

print(RevelioGeneList)

