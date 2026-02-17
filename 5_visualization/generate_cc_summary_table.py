import os
import pandas as pd
import numpy as np

# =====================================================
# MARKER GENE DIRECTORY
# =====================================================

MARKER_DIR = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/Cell_Cycle_marker_genes"

# =====================================================
# DATASET FILE PATHS (UPDATED WITH YOUR PATHS)
# =====================================================

DATASET_FILE_PATHS = {
    'REH': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv",
    'SUP-B15': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv",
    'PBMC': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv",
    'Mouse Brain': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data.csv",
    'Nestorova': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_nestorova/nestorova_training_data.csv",
    'GSE75748': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_gse75748/gse75748_training_data.csv",
    'GSE146773': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE146773/GSE146773_expression.csv",
    'GSE64016': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv",
    'Buettner mESC': "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/Buettner_mESC/Buettner_mESC_expression.csv"
}

OUTPUT_FILE = "cell_cycle_expression_summary_table.csv"

# =====================================================
# LOAD SEURAT MARKER GENES
# =====================================================

def load_seurat_markers():
    s_path = os.path.join(MARKER_DIR, "_Seurat_S_genes.txt")
    g2m_path = os.path.join(MARKER_DIR, "_Seurat_G2M_genes.txt")

    with open(s_path) as f:
        s_genes = [line.strip().upper() for line in f if line.strip()]

    with open(g2m_path) as f:
        g2m_genes = [line.strip().upper() for line in f if line.strip()]

    all_genes = sorted(set(s_genes + g2m_genes))
    print(f"Loaded {len(all_genes)} total CC genes.")
    return all_genes


# =====================================================
# COMPUTE METRICS
# =====================================================

def compute_metrics(df, cc_genes, dataset_name):

    df.columns = df.columns.str.upper()

    available_genes = [g for g in cc_genes if g in df.columns]

    if len(available_genes) == 0:
        print(f"{dataset_name}: No CC genes found.")
        return None

    expr = df[available_genes]

    n_genes = len(available_genes)
    mean_expr = expr.values.mean()
    std_expr = expr.values.std()
    pct_detected = (expr.values > 0).sum() / expr.size * 100

    key_markers = ["PCNA", "MKI67", "CDK1"]
    key_results = {}

    for marker in key_markers:
        if marker in df.columns:
            key_results[marker] = round((df[marker] > 0).sum() / len(df) * 100, 1)
        else:
            key_results[marker] = None

    return {
        "Dataset": dataset_name,
        "n_CC_genes": n_genes,
        "Mean_Expr": round(mean_expr, 3),
        "Std_Expr": round(std_expr, 3),
        "Pct_Detected": round(pct_detected, 2),
        "PCNA_%": key_results["PCNA"],
        "MKI67_%": key_results["MKI67"],
        "CDK1_%": key_results["CDK1"]
    }


# =====================================================
# MAIN
# =====================================================

def main():

    cc_genes = load_seurat_markers()
    results = []

    for name, path in DATASET_FILE_PATHS.items():
        print(f"Processing {name}...")

        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        df = pd.read_csv(path, index_col=0)

        metrics = compute_metrics(df, cc_genes, name)
        if metrics:
            results.append(metrics)

    final_table = pd.DataFrame(results)

    final_table.to_csv(OUTPUT_FILE, index=False)

    print("\nFinal Table:\n")
    print(final_table.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
