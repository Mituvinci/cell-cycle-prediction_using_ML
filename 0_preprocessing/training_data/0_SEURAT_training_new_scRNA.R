library(Seurat)
library(Signac)
library(dplyr)
library(ggplot2)

library(scattermore)
library(scater)




s.genes <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes



makePredcitedCCCSV <- function(path,sample_name){
  
  inputdata.10x <- Read10X_h5(path)
  #rna_counts <- inputdata.10x$`Gene Expression`
  if ("Gene Expression" %in% names(inputdata.10x)) {
    rna_counts <- inputdata.10x$`Gene Expression`
  } else {
    rna_counts <- inputdata.10x
  }
  
  sample_name <- paste0("trainingdata_cellcylescore_",sample_name)
  #print(head(rna_counts))
  srt_obj <- CreateSeuratObject(counts = rna_counts, min.cells = 200)
  # Then, add the annotations as metadata
  srt_obj@meta.data$sample  <- sample_name
  srt_obj[["percent.mt"]] <- PercentageFeatureSet(srt_obj, pattern = "^MT-")

  
  cell_cycle_srt_obj  <-  srt_obj
  print("seurat object start")


  cell_cycle_srt_obj <- NormalizeData(cell_cycle_srt_obj)
  
  
  print("seurat object end")
  
  #print(GetAssayData(cell_cycle_srt_obj, assay = "RNA", slot = "data")[1:5,1:5])
    
  
  # Extract normalized gene expression data
  normalized_data <- GetAssayData(cell_cycle_srt_obj, assay = "RNA", slot = "data")
  
  # Convert to data frame and transpose
  normalized_df <- as.data.frame(t(normalized_data))
  
  where_to_save <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/"
  
  # RNA SEQ Save to CSV file
  normalized_file_name <-  paste0(where_to_save,sample_name,"_normalized_gene_expression.csv")
  write.csv(normalized_df,normalized_file_name, row.names = TRUE)
  print(paste0("Saved ",normalized_file_name))
  
  # View the head of the expression matrix for a few cells and genes
  print(head(normalized_df[, 1:5]))
  
  cell_cycle_srt_obj <- FindVariableFeatures(cell_cycle_srt_obj, selection.method = "vst")
  cell_cycle_srt_obj <- ScaleData(cell_cycle_srt_obj, features = rownames(cell_cycle_srt_obj))
  
  cc_score <- CellCycleScoring(cell_cycle_srt_obj, s.features = s.genes, g2m.features = g2m.genes, set.ident = TRUE)
  print(names(cc_score[[]]))

  df  <-  data.frame(Idents(cc_score))
  df  <-  cbind(cc_score[["Phase"]])
  df <- cbind(barcode_RNA = rownames(df), df)
  rownames(df) <- 1:nrow(df)
  cc_file_name <- paste0(where_to_save,sample_name,"_cc_phases.csv")
  write.csv(df, cc_file_name, row.names = FALSE)
  print(paste0("Saved", cc_file_name))
  
  
  
}


paths_arr <- c(
  "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Review/pbmc_healthy_human/filtered_feature_bc_matrix.h5",
  "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Review/10-k-brain-cells_healthy_mouse/filtered_feature_bc_matrix.h5"
)

# Iterate over each path in the paths_arr
for (path in paths_arr) {
  # Assuming you want to use the path as the sample_name

  sample_name <- basename(dirname(path))
  print(sample_name)
  
  makePredcitedCCCSV(path, sample_name)
  
  
  
}


