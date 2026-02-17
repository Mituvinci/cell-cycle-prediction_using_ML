#!/usr/bin/env python3
################################################################################
# pca_cell_cycle_python.py
#
# PCA plots for 8 datasets, each colored by cell cycle phase (G1, S, G2M).
# Shows whether cell cycle phases cluster separately in PCA space.
# Each dataset is plotted independently using its own gene set.
#
# Usage:
#   python pca_cell_cycle_python.py
#   python pca_cell_cycle_python.py --output_dir /custom/output/path
################################################################################

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse

################################################################################
# Paths
################################################################################
DATA_ROOT = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq"
CC_ROOT   = os.path.join(DATA_ROOT, "cell_cycle_prediction")

################################################################################
# Dataset definitions
#   (display_name, file_path, phase_column, metadata_columns_to_drop)
#
# Phase column names are inconsistent across files:
#   Predicted  -- training data (REH, SUP-B15, hPSC, PBMC, Mouse Brain, Nestorova)
#   paper_phase -- GSE146773 benchmark
#   Labeled    -- GSE64016 benchmark  (contains "H1" cell-line rows; filtered below)
################################################################################
DATASETS = [
    ("REH",
     os.path.join(DATA_ROOT,
                  "data/filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv"),
     "Predicted",
     ["gex_barcode", "CellID"]),

    ("SUP-B15",
     os.path.join(DATA_ROOT,
                  "data/filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv"),
     "Predicted",
     ["gex_barcode", "CellID"]),

    ("hPSC (GSE75748)",
     os.path.join(DATA_ROOT, "data/GSE75748_hPSC_final_training_matrix.csv"),
     "Predicted",
     ["gex_barcode"]),

    ("PBMC Human",
     os.path.join(CC_ROOT, "1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv"),
     "Predicted",
     ["gex_barcode", "CellID"]),

    ("Mouse Brain",
     os.path.join(CC_ROOT, "1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data_UPPERCASE.csv"),
     "Predicted",
     ["gex_barcode", "CellID"]),

    ("Nestorova",
     os.path.join(CC_ROOT, "1_consensus_labeling/assign/final_training_data_nestorova/nestorova_training_data.csv"),
     "Predicted",
     ["gex_barcode", "CellID"]),

    ("GSE146773",
     os.path.join(DATA_ROOT, "data/Training_data/Benchmark_data/GSE146773_seurat_normalized_gene_expression.csv"),
     "paper_phase",
     ["Unnamed: 0", "cell"]),

    ("GSE64016",
     os.path.join(DATA_ROOT, "data/Training_data/Benchmark_data/GSE64016_seurat_normalized_gene_expression.csv"),
     "Labeled",
     ["gex_barcode"]),
]

VALID_PHASES = {"G1", "S", "G2M"}
PHASE_ORDER  = ["G1", "S", "G2M"]
PHASE_COLORS = {"G1": "#2196F3", "S": "#F44336", "G2M": "#4CAF50"}


################################################################################
# Core function
################################################################################
def load_and_pca(name, path, phase_col, meta_cols):
    """
    Load CSV, extract phase labels, drop metadata, scale, run PCA.

    Returns:
        pc1, pc2       -- numpy arrays (N,)
        phases         -- numpy array of str (N,)
        explained_var  -- explained variance ratio for PC1 and PC2
    """
    print(f"  Loading {name} ...")
    df = pd.read_csv(path)

    # --- phase labels ---
    phases = df[phase_col].astype(str).str.strip()

    # --- columns to drop: metadata + phase + any Unnamed index columns ---
    drop_cols = list(meta_cols) + [phase_col]
    drop_cols += [c for c in df.columns if c.startswith("Unnamed")]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)

    # --- keep only rows with valid phase labels (G1 / S / G2M) ---
    mask = phases.isin(VALID_PHASES)
    X      = X.loc[mask].copy()
    phases = phases.loc[mask].copy()

    # --- ensure all expression columns are numeric ---
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"    {len(X)} cells, {X.shape[1]} genes")
    print(f"    Phase counts: {phases.value_counts().to_dict()}")

    # --- StandardScaler + PCA ---
    X_scaled = StandardScaler().fit_transform(X)

    n_comp = min(30, X_scaled.shape[0], X_scaled.shape[1])
    pca    = PCA(n_components=n_comp)
    X_pca  = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_[:2]
    print(f"    PC1: {explained_var[0]*100:.1f}%  PC2: {explained_var[1]*100:.1f}%")

    return X_pca[:, 0], X_pca[:, 1], phases.values, explained_var


################################################################################
# Main
################################################################################
def main():
    parser = argparse.ArgumentParser(description="PCA colored by cell cycle phase for all datasets")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(CC_ROOT, "Has_cell_cycle_effect_or_not/pca_results/python_pca"),
                        help="Directory for output figures (default: .../pca_results/python_pca)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n_datasets = len(DATASETS)
    n_cols     = 4
    n_rows     = 2          # 2 x 4 = 8 panels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 12))
    axes_flat = axes.flatten()

    print("=" * 80)
    print("  PCA Cell Cycle Phase -- All Datasets (Python)")
    print("=" * 80)

    for i, (name, path, phase_col, meta_cols) in enumerate(DATASETS):
        print(f"\n[{i+1}/{n_datasets}] {name}")
        pc1, pc2, phases, expl_var = load_and_pca(name, path, phase_col, meta_cols)

        ax = axes_flat[i]
        for phase in PHASE_ORDER:
            mask = (phases == phase)
            ax.scatter(pc1[mask], pc2[mask],
                       c=PHASE_COLORS[phase],
                       label=phase,
                       alpha=0.55, s=12, edgecolors="none")

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel(f"PC1 ({expl_var[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2 ({expl_var[1]*100:.1f}%)", fontsize=10)
        ax.tick_params(labelsize=8)

    # --- shared legend (bottom center) ---
    handles = [mpatches.Patch(color=PHASE_COLORS[p], label=p) for p in PHASE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=14, frameon=True, bbox_to_anchor=(0.5, 0.015))

    fig.suptitle("PCA by Cell Cycle Phase -- All Datasets",
                 fontsize=17, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    # --- save ---
    png_path = os.path.join(args.output_dir, "pca_all_datasets_cell_cycle_phase.png")
    pdf_path = os.path.join(args.output_dir, "pca_all_datasets_cell_cycle_phase.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path,          bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


if __name__ == "__main__":
    main()
