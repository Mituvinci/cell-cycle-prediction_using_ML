library(Seurat)
library(Signac)
library(EnsDb.Hsapiens.v86)
library(dplyr)
library(ggplot2)

library(scattermore)
library(scater)

library(Revelio)





makePredcitedCCCSV <- function(path,sample_name){
  
  inputdata.10x <- Read10X_h5(path)
  rna_counts <- inputdata.10x$`Gene Expression`
  
  
  print(rna_counts)
  
  
  # Create a Revelio object with the raw data matrix and cyclic gene lists
  myData <- createRevelioObject(rawData = rna_counts,
                                cyclicGenes = revelioTestData_cyclicGenes)
  
  # Infer cell cycle phases for each cell
  myData <- getCellCyclePhaseAssignInformation(dataList = myData)
  
  
  
  print(myData@cellInfo)
  
  # Extract the relevant columns: cellID and ccPhase
  cc_phases_df <- myData@cellInfo[, c("cellID", "ccPhase")]
  
  
  
  # Save the data frame to a CSV file
  write.csv(cc_phases_df, paste0("D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/revelio_",sample_name,".csv"), row.names = FALSE)
  
 
  
}




paths_arr <- c("D:/Halima's Data/Thesis_2/1_GD428_21136_Hu_REH_Parental/outs","D:/Halima's Data/Thesis_2/2_GD444_21136_Hu_Sup_Parental/outs")
# Concatenate "/filtered_feature_bc_matrix.h5" to each path
paths_arr <- paste0(paths_arr, "/filtered_feature_bc_matrix.h5")
# Iterate over each path in the paths_arr
for (path in paths_arr) {
  # Assuming you want to use the path as the sample_name
  sample_name <- sub(".*/Thesis_2/(.*)/outs/filtered_feature_bc_matrix.h5", "\\1", path)
  # This will extract the last part of the path as the sample name
  #print(sample_name)
  
  #print(path)
  
  makePredcitedCCCSV(path, sample_name)
  
  
  
}


