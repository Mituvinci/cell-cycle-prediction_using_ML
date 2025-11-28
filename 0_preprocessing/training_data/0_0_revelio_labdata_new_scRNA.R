library(Seurat)
library(Signac)
library(EnsDb.Hsapiens.v86)
library(dplyr)
library(ggplot2)
library(scattermore)
library(scater)
library(Revelio)

### -----------------------------
### 1. Prepare mouse gene lists
### -----------------------------
s.genes.mouse  <- cc.genes.updated.2019$s.genes.mouse
g2m.genes.mouse <- cc.genes.updated.2019$g2m.genes.mouse

mouse_cyclic_list <- list(
  "G1/S"     = s.genes.mouse,
  "G2/M"     = g2m.genes.mouse,
  "G1/S_ext" = s.genes.mouse,    # fallback (Revelio requires ext categories)
  "G2/M_ext" = g2m.genes.mouse,  # fallback
  "other"    = character(0)
)

### --------------------------------------
### 2. Universal Revelio wrapper function
### --------------------------------------
makePredcitedCCCSV <- function(path, sample_name, species = "human") {
  
  # Read 10X HDF5
  inputdata.10x <- Read10X_h5(path)
  rna_counts <- if ("Gene Expression" %in% names(inputdata.10x)) {
    inputdata.10x$`Gene Expression`
  } else {
    inputdata.10x
  }
  
  print(rna_counts)
  
  ### Select appropriate marker list
  if (species == "human") {
    cyclic_genes <- revelioTestData_cyclicGenes   # built-in human markers
    message("Using HUMAN Revelio markers...")
    
  } else if (species == "mouse") {
    cyclic_genes <- mouse_cyclic_list
    message("Using MOUSE Revelio markers...")
    
  } else {
    stop("Species must be 'human' or 'mouse'.")
  }
  
  ### Create Revelio object
  myData <- createRevelioObject(
    rawData = rna_counts,
    cyclicGenes = cyclic_genes
  )
  
  ### Phase assignment
  myData <- getCellCyclePhaseAssignInformation(dataList = myData)
  print(myData@cellInfo)
  
  ### Extract useful columns
  cc_phases_df <- myData@cellInfo[, c("cellID", "ccPhase")]
  
  ### Save CSV
  output_path <- paste0(
    "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/revelio_",
    species, "_",
    sample_name,
    ".csv"
  )
  
  write.csv(cc_phases_df, output_path, row.names = FALSE)
  message("Saved: ", output_path)
}

### -------------------------------------
### 3. Run for mouse or human
### -------------------------------------

# human  "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Review/pbmc_healthy_human/filtered_feature_bc_matrix.h5"
paths_arr <- c(
  "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Review/10-k-brain-cells_healthy_mouse/filtered_feature_bc_matrix.h5"
)

for (path in paths_arr) {
  sample_name <- basename(dirname(path))
  print(sample_name)
  
  makePredcitedCCCSV(
    path = path,
    sample_name = sample_name,
    species = "mouse"      # 👈 CHANGE HERE for human/mouse
  )
}
