#!/usr/bin/env python3

"""
UNIVERSAL ccAFv2 CELL CYCLE PREDICTION
Handles: 10X MTX, CSV, TXT formats
Works for: Human and Mouse datasets
"""

import argparse
import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import ccAFv2

def capitalize_genes(gene_list):
    """Capitalize gene names (e.g., GAPDH -> Gapdh)"""
    return [g[0].upper() + g[1:].lower() if len(g) > 1 else g.upper() for g in gene_list]

def read_10x_mtx(input_dir):
    """Read 10X MTX format"""
    print(f"Reading 10X MTX format from: {input_dir}")
    adata = sc.read_10x_mtx(input_dir, var_names='gene_symbols', cache=True)
    return adata

def read_csv(input_file):
    """Read CSV format (genes x cells)"""
    print(f"Reading CSV format from: {input_file}")
    df = pd.read_csv(input_file, index_col=0)

    # Capitalize gene names
    df.index = capitalize_genes(df.index.tolist())

    # Transpose to cells x genes
    df = df.T

    # Create AnnData
    adata = ad.AnnData(X=df.values.astype(np.float32))
    adata.obs_names = df.index.astype(str).tolist()
    adata.var_names = df.columns.astype(str).tolist()

    return adata

def read_txt(input_file):
    """Read TXT/TSV format (genes x cells)"""
    print(f"Reading TXT/TSV format from: {input_file}")
    df = pd.read_csv(input_file, sep='\t', index_col=0)

    # Capitalize gene names
    df.index = capitalize_genes(df.index.tolist())

    # Transpose to cells x genes
    df = df.T

    # Create AnnData
    adata = ad.AnnData(X=df.values.astype(np.float32))
    adata.obs_names = df.index.astype(str).tolist()
    adata.var_names = df.columns.astype(str).tolist()

    return adata

def auto_detect_format(input_path):
    """Auto-detect input format"""
    if os.path.isdir(input_path):
        # Check for 10X files
        if (os.path.exists(os.path.join(input_path, 'matrix.mtx.gz')) or
            os.path.exists(os.path.join(input_path, 'matrix.mtx'))):
            return '10x'
        else:
            raise ValueError("Cannot auto-detect format for directory. Please specify --format")
    elif os.path.isfile(input_path):
        # Detect by extension
        if input_path.lower().endswith('.csv'):
            return 'csv'
        elif input_path.lower().endswith(('.txt', '.tsv')):
            return 'txt'
        else:
            raise ValueError("Cannot auto-detect format. Please specify --format (10x, csv, or txt)")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Universal ccAFv2 cell cycle prediction for multiple data formats'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input file or directory (for 10X: path to filtered_feature_bc_matrix/)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory')
    parser.add_argument('-s', '--sample', default='sample',
                        help='Sample name (default: sample)')
    parser.add_argument('-f', '--format', default='auto', choices=['10x', 'csv', 'txt', 'auto'],
                        help='Input format: 10x, csv, txt, or auto (default: auto)')
    parser.add_argument('--species', default='human', choices=['human', 'mouse'],
                        help='Species: human or mouse (default: human)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")

    # Auto-detect format if needed
    format_type = args.format
    if format_type == 'auto':
        format_type = auto_detect_format(args.input)

    print(f"Detected format: {format_type}")
    print(f"Species: {args.species}")
    print(f"Sample: {args.sample}")

    # Read data based on format
    if format_type == '10x':
        adata = read_10x_mtx(args.input)
        # Capitalize gene names for ccAFv2
        adata.var_names = capitalize_genes(adata.var_names.tolist())

    elif format_type == 'csv':
        adata = read_csv(args.input)

    elif format_type == 'txt':
        adata = read_txt(args.input)

    print(f"Expression matrix: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # QC and preprocessing
    print("Preprocessing...")

    # MT genes (human: MT-, mouse: Mt-)
    if args.species == 'human':
        adata.var['mt'] = adata.var_names.str.startswith('Mt-')
    else:
        adata.var['mt'] = adata.var_names.str.startswith('Mt-')

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_genes(adata, min_cells=1)

    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print(f"After preprocessing: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Predict cell cycle phases
    print("Running ccAFv2 prediction...")
    predicted_labels, probabilities = ccAFv2.predict_labels(
        adata,
        species=args.species,
        gene_id='symbol'
    )
    adata.obs['ccAFv2'] = predicted_labels

    print("\nFirst 10 predictions:")
    print(adata.obs[['ccAFv2']].head(10))

    print("\nPhase distribution:")
    print(adata.obs['ccAFv2'].value_counts())

    # Save predictions
    output_file = os.path.join(args.output, f"ccAFV2_{args.sample}.csv")
    adata.obs[['ccAFv2']].to_csv(output_file)
    print(f"\nSaved predictions: {output_file}")

    print("\nDONE!")

if __name__ == "__main__":
    main()
