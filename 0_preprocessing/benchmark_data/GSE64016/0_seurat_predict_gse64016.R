# Load necessary libraries
library(Seurat)
library(dplyr)


# Identify S phase and G2/M phase marker genes
s_genes <- cc.genes$s.genes
g2m_genes <- cc.genes$g2m.genes

# Set the file path to your TPM data
tpm_file <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/GSE64016_H1andFUCCI/GSE64016_H1andFUCCI_normalized_EC.csv"

# Load the TPM data
tpm_data <- read.csv(tpm_file, row.names = 1, check.names = FALSE)


# Create a Seurat object
seurat_object <- CreateSeuratObject(counts = tpm_data, project = "CellCycle", min.cells = 200)

sample_name<- "GSE64016"
seurat_object@meta.data$sample  <- sample_name
seurat_object[["percent.mt"]] <- PercentageFeatureSet(seurat_object, pattern = "^MT-")


cell_cycle_srt_obj  <-  seurat_object
print("seurat object start")

cell_cycle_srt_obj <- NormalizeData(cell_cycle_srt_obj)



# Extract normalized gene expression data
normalized_data <- GetAssayData(cell_cycle_srt_obj, assay = "RNA", slot = "data")

# Convert to data frame and transpose
normalized_df <- as.data.frame(t(normalized_data))

normalized_df$gex_barcode <- rownames(normalized_df)  # Add cell names as a new column

# Create the `Labeled` column from cell names
normalized_df$Labeled <- sub("_.*", "", normalized_df$gex_barcode)

where_to_save <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/Benchmark_data/"

# RNA SEQ Save to CSV file
normalized_file_name <-  paste0(where_to_save,sample_name,"_seurat_normalized_gene_expression.csv")
write.csv(normalized_df,normalized_file_name, row.names = TRUE)
print(paste0("Saved ",normalized_file_name))


cell_cycle_srt_obj <- FindVariableFeatures(cell_cycle_srt_obj, selection.method = "vst")
cell_cycle_srt_obj <- ScaleData(cell_cycle_srt_obj, features = rownames(cell_cycle_srt_obj))

cc_score <- CellCycleScoring(cell_cycle_srt_obj, s.features = s.genes, g2m.features = g2m.genes, set.ident = TRUE)
print(names(cc_score[[]]))
cc_score <- RunPCA(cc_score, features = c(s.genes, g2m.genes))  

df  <-  data.frame(Idents(cc_score))
df  <-  cbind(cc_score[["Phase"]])
df <- cbind(barcode_RNA = rownames(df), df)
rownames(df) <- 1:nrow(df)

write.csv(df, paste0(where_to_save,"predict_by_seurat_",sample_name,".csv"), row.names = FALSE)
print(paste0("Saved ",sample_name,".csv"))
