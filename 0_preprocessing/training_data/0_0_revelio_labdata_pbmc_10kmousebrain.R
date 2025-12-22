library(Seurat)
library(Signac)
library(dplyr)
library(ggplot2)
library(scattermore)
library(scater)
library(Revelio)

### --------------------------------------
### Universal Revelio wrapper (ONE FORMAT)
### --------------------------------------
makePredcitedCCCSV <- function(path, sample_name) {
  
  # -----------------------------
  # Read 10X HDF5
  # -----------------------------
  inputdata.10x <- Read10X_h5(path)
  rna_counts <- if ("Gene Expression" %in% names(inputdata.10x)) {
    inputdata.10x$`Gene Expression`
  } else {
    inputdata.10x
  }
  
  # -----------------------------
  # FORCE HUMAN-STYLE GENE SYMBOLS
  # -----------------------------
  rownames(rna_counts) <- toupper(rownames(rna_counts))
  
  message("Running Revelio using HUMAN markers with UPPERCASE gene symbols")
  
  # -----------------------------
  # Create Revelio object (HUMAN markers ONLY)
  # -----------------------------
  myData <- createRevelioObject(
    rawData     = rna_counts,
    cyclicGenes = revelioTestData_cyclicGenes
  )
  
  # -----------------------------
  # Assign cell cycle phases
  # -----------------------------
  myData <- getCellCyclePhaseAssignInformation(dataList = myData)
  
  # -----------------------------
  # Extract results
  # -----------------------------
  cc_phases_df <- myData@cellInfo[, c("cellID", "ccPhase")]
  print(head(cc_phases_df))
  
  # -----------------------------
  # Save CSV
  # -----------------------------
  output_path <- paste0(
    "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/revelio_",
    sample_name,
    ".csv"
  )
  
  write.csv(cc_phases_df, output_path, row.names = FALSE)
  message("Saved: ", output_path)
}

### -------------------------------------
### Run (HUMAN OR MOUSE – SAME CODE)
### -------------------------------------
#  "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Review/pbmc_healthy_human/filtered_feature_bc_matrix.h5"

paths_arr <- c(
  "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Review/10-k-brain-cells_healthy_mouse/filtered_feature_bc_matrix.h5"
)

for (path in paths_arr) {
  sample_name <- basename(dirname(path))
  print(sample_name)
  makePredcitedCCCSV(path, sample_name)
}
