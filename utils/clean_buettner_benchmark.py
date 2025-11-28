#!/usr/bin/env python3
"""
Clean Buettner mESC Benchmark Data
===================================

Removes the Phase column from Buettner_mESC_SeuratNormalized_ML_ready.csv
and ensures all gene names are capitalized (first letter uppercase, rest lowercase).

Ground truth labels are in separate metadata file: Buettner_mESC_metadata.csv

Author: Halima Akhter
Date: 2025-11-28
"""

import pandas as pd
import os

def capitalize_gene_name(gene_name):
    """
    Capitalize gene name: First letter uppercase, rest lowercase.

    For mouse genes: Gnai3, Pbsn, Cdc45
    For human genes: GAPDH, ACTB, etc. (we'll apply same rule)

    Parameters:
    -----------
    gene_name : str
        Original gene name

    Returns:
    --------
    str
        Capitalized gene name
    """
    if gene_name == "Cell_ID":
        return gene_name
    return gene_name.capitalize()


def main():
    # Paths
    data_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data"
    input_file = os.path.join(data_dir, "Buettner_mESC_SeuratNormalized_ML_ready.csv")
    output_file = os.path.join(data_dir, "Buettner_mESC_benchmark_clean.csv")

    print("=" * 80)
    print("CLEANING BUETTNER mESC BENCHMARK DATA")
    print("=" * 80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    # Read the data
    print("Reading data...")
    df = pd.read_csv(input_file)
    print(f"  Original shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:5])}...")
    print()

    # Check if Phase column exists
    if "Phase" in df.columns:
        print("✓ Found 'Phase' column - will remove it")
        df = df.drop(columns=["Phase"])
        print(f"  New shape after removing Phase: {df.shape}")
    else:
        print("⚠ No 'Phase' column found - data already clean")

    # Capitalize all gene names
    print("\nCapitalizing gene names...")
    original_columns = df.columns.tolist()
    new_columns = [capitalize_gene_name(col) for col in original_columns]
    df.columns = new_columns

    # Show changes
    changed_genes = [(orig, new) for orig, new in zip(original_columns, new_columns) if orig != new and orig != "Cell_ID"]
    if changed_genes:
        print(f"  Capitalized {len(changed_genes)} gene names:")
        for orig, new in changed_genes[:5]:
            print(f"    {orig} → {new}")
        if len(changed_genes) > 5:
            print(f"    ... and {len(changed_genes) - 5} more")
    else:
        print("  All gene names already capitalized correctly")

    # Save cleaned data
    print(f"\nSaving cleaned data to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"  Saved shape: {df.shape}")

    # Verify the output
    print("\nVerifying output:")
    print(f"  Columns (first 5): {list(df.columns[:5])}")
    print(f"  First row (first 5 values): {df.iloc[0, :5].tolist()}")

    # Show distribution
    print("\n✅ Benchmark data cleaned successfully!")
    print("=" * 80)
    print("\n📋 SUMMARY:")
    print(f"  - Removed Phase column: {'Yes' if 'Phase' in original_columns else 'No (already removed)'}")
    print(f"  - Total cells: {len(df)}")
    print(f"  - Total features: {len(df.columns) - 1} genes")
    print(f"  - Ground truth labels: Use Buettner_mESC_metadata.csv")
    print()


if __name__ == "__main__":
    main()
