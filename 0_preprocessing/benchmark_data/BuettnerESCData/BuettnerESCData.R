########################################
# Load Required Libraries
########################################

library(scRNAseq)
library(SingleCellExperiment)
library(AnnotationHub)
library(AnnotationFilter)   # IMPORTANT for GeneIdFilter()

save_path <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/Benchmark_data/Butter_mESCData"


########################################
# Load Buettner ESC dataset
########################################
# Load again from the original R object
sce <- BuettnerESCData()

expr_mat <- counts(sce)
ensembl_ids <- rownames(expr_mat)

gene_symbols <- lookup[ensembl_ids_clean]  # from earlier mapping
gene_symbols[is.na(gene_symbols)] <- ensembl_ids_clean

# Build a clean expression matrix
expr_clean <- as.data.frame(as.matrix(expr_mat))

expr_clean <- cbind(
  Ensembl_ID = ensembl_ids,
  GeneSymbol = gene_symbols,
  expr_clean
)

write.csv(expr_clean,
          file.path(save_path, "Buettner_mESC_expression_CLEAN.csv"),
          row.names = FALSE)


cat("\n=== Summary ===\n")
cat("Expression matrix dimensions:", paste(dim(expr_mat), collapse = " × "), "\n")
cat("Genes with mapped symbols:", sum(!is.na(gene_symbols)), "out of", length(gene_symbols), "\n")
cat("Files saved to:\n", save_path, "\n")
cat("Done!\n")

############################################################
# 1 — Load clean file and remove duplicates
############################################################

clean_file <- file.path(save_path, "Buettner_mESC_expression_CLEAN.csv")
expr_df <- read.csv(clean_file, check.names = FALSE)

# Remove rows where GeneSymbol is missing
expr_df <- expr_df[expr_df$GeneSymbol != "" & !is.na(expr_df$GeneSymbol), ]

# Remove duplicate GeneSymbols: keep the FIRST occurrence
expr_df <- expr_df[!duplicated(expr_df$GeneSymbol), ]

cat("After removing duplicated gene symbols:", nrow(expr_df), "genes left\n")

########################################
# Save Metadata (extracted from column names)
########################################

metadata <- as.data.frame(colData(sce))
write.csv(metadata,
          file = file.path(save_path, "Buettner_mESC_metadata.csv"),
          row.names = TRUE)

cat("\n=== DONE ===\n")
cat("Gene symbols:", length(gene_symbols), "\n")

