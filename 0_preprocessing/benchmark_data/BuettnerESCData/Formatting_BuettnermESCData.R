############################################################
# Load Libraries
############################################################
library(Seurat)
library(dplyr)

############################################################
# Paths
############################################################
save_path <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/Benchmark_data/Butter_mESCData"

expr_clean_file <- file.path(save_path, "Buettner_mESC_expression_CLEAN.csv")
meta_file       <- file.path(save_path, "Buettner_mESC_metadata.csv")

############################################################
# Step 1 — Load expression
############################################################
expr_df <- read.csv(expr_clean_file, check.names = FALSE)

# Remove duplicate/missing gene symbols
expr_df <- expr_df[expr_df$GeneSymbol != "" & !is.na(expr_df$GeneSymbol), ]
expr_df <- expr_df[!duplicated(expr_df$GeneSymbol), ]

cat("After removing duplicates:", nrow(expr_df), "genes\n")

############################################################
# Step 2 — Set GeneSymbol as rownames
############################################################
gene_symbols <- expr_df$GeneSymbol
expr_df$GeneSymbol <- NULL
expr_df$Ensembl_ID <- NULL  # REMOVE THIS ENTIRE COLUMN

rownames(expr_df) <- gene_symbols

############################################################
# Step 3 — Transpose so rows = cells
############################################################
expr_t <- as.data.frame(t(expr_df))   # now rows are cell IDs
expr_t$Cell_ID <- rownames(expr_t)

############################################################
# Step 4 — Extract Phase from column names (NOT metadata)
############################################################
expr_t$Phase <- sub("_cell.*", "", expr_t$Cell_ID)  
# G1_cell1_count → G1
# S_cell4_count  → S
# G2M_cell9      → G2M

############################################################
# Save before normalization
############################################################
raw_out <- file.path(save_path, "Buettner_mESC_RAW_for_Seurat.csv")
write.csv(expr_t, raw_out, row.names = FALSE)

############################################################
# Step 5 — Run Seurat normalization
############################################################
gex_only <- expr_t[, !(colnames(expr_t) %in% c("Cell_ID", "Phase"))]
rownames(gex_only) <- expr_t$Cell_ID

srt <- CreateSeuratObject(counts = t(gex_only))

srt <- NormalizeData(srt, normalization.method = "LogNormalize", scale.factor = 10000)
srt <- ScaleData(srt)

norm_mat <- GetAssayData(srt, slot = "data")
norm_df <- as.data.frame(t(norm_mat))

norm_df$Cell_ID <- rownames(norm_df)
norm_df$Phase <- expr_t$Phase  # add back phase

norm_df <- norm_df %>% relocate(Cell_ID, Phase)

############################################################
# Step 6 — Save final ML-ready CSV
############################################################
final_out <- file.path(save_path, "Buettner_mESC_SeuratNormalized_ML_ready.csv")
write.csv(norm_df, final_out, row.names = FALSE)

cat("\n🔥 DONE! Final ML-ready file saved at:\n", final_out)


cat("⬅️ Final ML features:", ncol(norm_df) - 2, "\n")

