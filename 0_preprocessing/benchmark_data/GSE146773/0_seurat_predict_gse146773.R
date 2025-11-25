library(Seurat)
library(Signac)
library(dplyr)
library(ggplot2)
library(tricycle)
library(scattermore)
library(scater)


s.genes <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes


where_to_save <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/Benchmark_data/"

  
path <- "D:/Halima's Data/Thesis_2/GSE146773_cell_cycle/data"
sample_name<- "GSE146773"
gse.data <- Read10X(path)
#print(head(rna_counts))
srt_obj <- CreateSeuratObject(counts = gse.data, project = "cell cycle",min.cells = 200)
srt_obj@meta.data$sample  <- sample_name
srt_obj[["percent.mt"]] <- PercentageFeatureSet(srt_obj, pattern = "^MT-")


cell_cycle_srt_obj  <-  srt_obj

cell_cycle_srt_obj <- NormalizeData(cell_cycle_srt_obj)

normalize_srt_obj <- cell_cycle_srt_obj
file_name=paste(path,'/','paper_phase.txt',sep="")

paper_phase <- read.table(file=file_name,header=TRUE)

normalize_srt_obj[['paper_phase']] <- paper_phase$paper_phase


# Assuming 'paper_phase' has a 'cell' column that matches cell names in 'cell_cycle_srt_obj'
normalized_df <- as.data.frame(t(GetAssayData(normalize_srt_obj, assay = "RNA", slot = "data")))

print(head(normalized_df))

normalized_df$cell <- rownames(normalized_df)

# Merge 'paper_phase' into 'normalized_df' using a common identifier
final_df <- merge(normalized_df, paper_phase, by.x = "cell", by.y = "barcodes", all.y = TRUE)


# RNA SEQ Save to CSV file
normalized_file_name <-  paste0(where_to_save,sample_name,"_seurat_normalized_gene_expression.csv")
write.csv(final_df,normalized_file_name, row.names = TRUE)
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
  

  


