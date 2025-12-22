library(Seurat)
library(dplyr)

###############################################
# Paths
###############################################
input_file <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Dolly/BuettnerESCData/Buettner_mESC_benchmark_clean.csv"

output_dir <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Dolly/BuettnerESCData"
output_file <- file.path(output_dir, "predicted_by_seurat_Buettner.csv")

###############################################
# Step 1 — Load benchmark file (already normalized)
###############################################
df <- read.csv(input_file, check.names = FALSE)

###############################################
# Step 2 — Extract GroundTruth from Cell_ID
###############################################
df$GroundTruth <- sub("_.*", "", df$Cell_ID)

ground_truth <- df[, c("Cell_ID", "GroundTruth")]

###############################################
# Step 3 — Prepare gene matrix
# (cells × genes → genes × cells)
###############################################
# Remove duplicate gene columns
df <- df[, !duplicated(colnames(df))]

# Now safe to select genes
gene_mat <- df %>% select(-Cell_ID, -GroundTruth)


rownames(gene_mat) <- df$Cell_ID
gene_mat_t <- t(gene_mat)  # Seurat expects genes × cells

###############################################
# Step 4 — Create Seurat object WITHOUT re-normalizing
###############################################
srt <- CreateSeuratObject(counts = gene_mat_t)

# Put normalized values into the Seurat v5 "data" layer
LayerData(srt, layer = "data") <- gene_mat_t

# Scale using normalized data
srt <- ScaleData(srt, features = rownames(srt), layer = "data")



###############################################
# Step 5 — Seurat Cell Cycle Scoring
###############################################
srt <- CellCycleScoring(
  object = srt,
  s.features = cc.genes$s.genes,
  g2m.features = cc.genes$g2m.genes,
  set.ident = TRUE
)

###############################################
# Step 6 — Extract predictions
###############################################
pred <- srt$Phase
cell_ids <- rownames(srt@meta.data)

result_df <- data.frame(
  Cell_ID = cell_ids,
  Predicted = pred,
  GroundTruth = ground_truth$GroundTruth[match(cell_ids, ground_truth$Cell_ID)]
)

###############################################
# Step 7 — Save output
###############################################
write.csv(result_df, output_file, row.names = FALSE)

cat("🔥 DONE! Saved Seurat predictions to:\n", output_file, "\n")
