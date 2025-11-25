
# Load libraries
library(Seurat)
library(Revelio)

# Load Revelio's sample cyclic genes data as it is
data("revelioTestData_cyclicGenes")

# Define paths
matrix_path <- "D:/Halima's Data/Thesis_2/GSE_temporary"

# Load your data
data <- Read10X(data.dir = matrix_path)

print(data[1:5, 1:5])

# Print dimensions to confirm structure (genes x cells)
cat("Dimensions of your data (genes x cells):", dim(data), "\n")

# Create the Revelio object without pre-filtering the data
myData <- createRevelioObject(
  rawData = data,
  cyclicGenes = revelioTestData_cyclicGenes,
  ccPhaseAssignBasedOnIndividualBatches = FALSE
)

# Filter out only relevant cyclic genes in the data after object creation
cyclic_genes <- unique(unlist(revelioTestData_cyclicGenes))
#myData@DGEs$countData <- myData@DGEs$countData[rownames(myData@DGEs$countData) %in% cyclic_genes, ]

# Verify the filtered data structure in myData before proceeding
cat("Dimensions of myData@DGEs$countData after filtering:", dim(myData@DGEs$countData), "\n")

# Proceed to infer cell cycle phases for each cell
myData <- getCellCyclePhaseAssignInformation(dataList = myData)

# Extract the relevant columns: cellID and ccPhase
cc_phases_df <- myData@cellInfo[, c("cellID", "ccPhase")]

# Save the data frame to a CSV file
output_path <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/Benchmark_data/predict_revelio_gse146773cell_cycle_assignments_revelio.csv"
write.csv(cc_phases_df, output_path, row.names = FALSE)

# Print completion message
cat("Cell cycle phase information saved to:", output_path, "\n")
