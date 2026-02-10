#!/usr/bin/env python3
"""
Find common genes across multiple CSV files and save to text files.
Edit the FILE_PATHS list below to specify your input files.
"""

import pandas as pd
import os

# Output directory for gene list files
OUTPUT_DIR = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/gene_lists"

# =============================================================================
# EDIT THIS LIST: Add your CSV file paths here
# =============================================================================
'''
FILE_PATHS1 = [
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/training_data_1_GD428_21136_Hu_REH_Parental_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/training_data_2_GD444_21136_Hu_Sup_Parental_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/scTQuery/with_10k_features/REH_aligned/Integrated_REH_aligned_to_Buettner_mESC.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/scTQuery/with_10k_features/REH_aligned/Integrated_REH_aligned_to_GSE64016.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/scTQuery/with_10k_features/REH_aligned/Integrated_REH_aligned_to_GSE146773.csv"
]

FILE_PATHS2 = [
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/trainingdata_cellcylescore_10-k-brain-cells_healthy_mouse_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/training_data_2_GD444_21136_Hu_Sup_Parental_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/GSE146773_seurat_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/GSE64016_seurat_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Buettner_mESC_benchmark_clean.csv"
]

FILE_PATHS3 = [
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/trainingdata_cellcylescore_pbmc_healthy_human_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/training_data_2_GD444_21136_Hu_Sup_Parental_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/GSE146773_seurat_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data/GSE64016_seurat_normalized_gene_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Buettner_mESC_benchmark_clean.csv"
]
'''

FILE_PATHS2 =  [
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE146773/GSE146773_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/Buettner_mESC/Buettner_mESC_expression.csv"
]

FILE_PATHS3 = [
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE146773/GSE146773_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/Buettner_mESC/Buettner_mESC_expression.csv"
]

FILE_PATHS_NESTOROVA = [
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_nestorova/nestorova_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE146773/GSE146773_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/Buettner_mESC/Buettner_mESC_expression.csv"
]

FILE_PATHS_REH = [
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE146773/GSE146773_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/Buettner_mESC/Buettner_mESC_expression.csv"
]

FILE_PATHS_SUP = [
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_sup/sup_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_reh/reh_training_data.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE146773/GSE146773_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/GSE64016/GSE64016_expression.csv",
    "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/benchmarks_preprocessed/Buettner_mESC/Buettner_mESC_expression.csv"
]




# =============================================================================


def get_genes_from_csv(file_path):
    """Read CSV and return gene names (from columns) converted to UPPERCASE."""
    df = pd.read_csv(file_path, index_col=0, nrows=0)
    # Genes are column names, exclude non-gene columns like 'dataset'
    cols = df.columns.tolist()
    exclude_cols = ['dataset', 'cell_id', 'Dataset', 'Cell_ID']
    genes = [c for c in cols if c not in exclude_cols]
    return set([g.upper() for g in genes])


def main(FILE_PATHS, output_filename):
    print("=" * 60)
    print(f"Processing: {output_filename}")
    print("=" * 60)
    print("GENE COUNT PER FILE")
    print("-" * 60)

    all_gene_sets = []

    for file_path in FILE_PATHS:
        if not os.path.exists(file_path):
            print(f"WARNING: File not found - {file_path}")
            continue

        genes = get_genes_from_csv(file_path)
        all_gene_sets.append(genes)

        file_name = os.path.basename(file_path)
        print(f"{file_name}: {len(genes)} genes")

    print("-" * 60)

    if len(all_gene_sets) < 2:
        print("Need at least 2 valid files to find common genes.")
        return

    # Find intersection of all gene sets
    common_genes = all_gene_sets[0]
    for gene_set in all_gene_sets[1:]:
        common_genes = common_genes.intersection(gene_set)

    print(f"COMMON GENES ACROSS ALL FILES: {len(common_genes)}")

    # Save common genes to text file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Sort genes alphabetically before saving
    sorted_genes = sorted(list(common_genes))

    with open(output_path, 'w') as f:
        for gene in sorted_genes:
            f.write(f"{gene}\n")

    print(f"Saved {len(common_genes)} genes to: {output_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    #main(FILE_PATHS, "hpsc_3benchmark_sup.txt")
    #main(FILE_PATHS2, "mousebrain_3benchmark_sup.txt")
    #main(FILE_PATHS3, "pbmc_3benchmark_sup.txt")
    #main(FILE_PATHS_NESTOROVA, "nestorova_3benchmark_sup.txt")
    main(FILE_PATHS_REH, "reh_3benchmark_sup.txt")
    main(FILE_PATHS_SUP, "sup_3benchmark_sup.txt")
