library(Revelio)

# -----------------------------
# Load benchmark CSV
# -----------------------------
path <- "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/Benchmark_data/Butter_mESCData/Buettner_mESC_benchmark_transposed.csv"

expr <- read.csv(path, row.names = 1, check.names = FALSE)

# FORCE human-style gene symbols
rownames(expr) <- toupper(rownames(expr))

expr <- as.matrix(expr)

#print(dim(expr))
#print(head(rownames(expr)))

# -----------------------------
# Run Revelio (HUMAN markers)
# -----------------------------
message("Running Revelio on mouse benchmark data using HUMAN markers (expected to fail)")

myData <- createRevelioObject(
  rawData = expr,
  cyclicGenes = revelioTestData_cyclicGenes
)


myData <- getCellCyclePhaseAssignInformation(dataList = myData)

#print(myData@cellInfo)

### Extract useful columns
cc_phases_df <- myData@cellInfo[, c("cellID", "ccPhase")]

print(cc_phases_df)


### Save CSV
output_path <- paste0(
  "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data//Benchmark_data/Butter_mESCData/revelio_",
  "mouse", "_",
  "Buettener_mESC",
  ".csv"
)

write.csv(cc_phases_df, output_path, row.names = FALSE)
message("Saved: ", output_path)
