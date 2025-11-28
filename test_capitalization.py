#!/usr/bin/env python3
"""
Test Gene Name Capitalization
==============================

Verifies that capitalize_gene_names() function works correctly.

Author: Halima Akhter
Date: 2025-11-28
"""

import sys
import os
sys.path.insert(0, '2_model_training')

from utils.data_utils import capitalize_gene_names
import pandas as pd

print("=" * 80)
print("TESTING GENE NAME CAPITALIZATION")
print("=" * 80)

# Test data with various gene name formats
test_df = pd.DataFrame({
    'Cell_ID': ['cell1', 'cell2', 'cell3'],
    'Predicted': ['G1', 'S', 'G2M'],
    'GAPDH': [1.5, 2.3, 3.1],     # Human gene (uppercase) → Gapdh
    'actb': [3.2, 1.8, 2.5],       # Human gene (lowercase) → Actb
    'TP53': [0.5, 1.1, 0.8],       # Human gene (mixed) → Tp53
    'gnai3': [2.1, 3.5, 1.9],      # Mouse gene (lowercase) → Gnai3
    'Pbsn': [1.2, 0.9, 1.7],       # Mouse gene (already correct) → Pbsn
    'CDC45': [2.8, 3.2, 2.1],      # Mixed case → Cdc45
    'h19': [0.3, 0.7, 0.5]         # Lowercase → H19
})

print("\nBEFORE capitalization:")
print(f"  Columns: {test_df.columns.tolist()}")

# Apply capitalization
test_df_cap = capitalize_gene_names(test_df)

print("\nAFTER capitalization:")
print(f"  Columns: {test_df_cap.columns.tolist()}")

# Verify results
print("\n" + "-" * 80)
print("VERIFICATION")
print("-" * 80)

expected_columns = ['Cell_ID', 'Predicted', 'Gapdh', 'Actb', 'Tp53', 'Gnai3', 'Pbsn', 'Cdc45', 'H19']
actual_columns = test_df_cap.columns.tolist()

print(f"Expected: {expected_columns}")
print(f"Actual:   {actual_columns}")

# Check each column
all_correct = True
for expected, actual in zip(expected_columns, actual_columns):
    match = "✓" if expected == actual else "✗"
    print(f"  {match} {expected:20s} == {actual}")
    if expected != actual:
        all_correct = False

print("\n" + "=" * 80)
if all_correct:
    print("✅ ✅ ✅  ALL TESTS PASSED!  ✅ ✅ ✅")
else:
    print("❌ ❌ ❌  SOME TESTS FAILED!  ❌ ❌ ❌")
print("=" * 80)

# Additional test: Check that metadata columns are NOT capitalized
print("\n" + "-" * 80)
print("METADATA COLUMN TEST")
print("-" * 80)

test_df2 = pd.DataFrame({
    'gex_barcode': ['bc1', 'bc2'],
    'cell': ['c1', 'c2'],
    'Phase': ['G1', 'S'],
    'Predicted': ['G1', 'S'],
    'GENE1': [1.0, 2.0],
    'gene2': [3.0, 4.0]
})

print(f"\nBefore: {test_df2.columns.tolist()}")
test_df2_cap = capitalize_gene_names(test_df2)
print(f"After:  {test_df2_cap.columns.tolist()}")

# Verify metadata columns are unchanged
metadata_correct = True
for col in ['gex_barcode', 'cell', 'Phase', 'Predicted']:
    if col in test_df2.columns and col in test_df2_cap.columns:
        print(f"  ✓ '{col}' preserved (not capitalized)")
    else:
        print(f"  ✗ '{col}' missing or changed!")
        metadata_correct = False

# Verify gene columns ARE capitalized
if 'Gene1' in test_df2_cap.columns and 'Gene2' in test_df2_cap.columns:
    print(f"  ✓ Gene columns capitalized: GENE1 → Gene1, gene2 → Gene2")
else:
    print(f"  ✗ Gene columns NOT capitalized correctly!")
    metadata_correct = False

print("\n" + "=" * 80)
if metadata_correct:
    print("✅ ✅ ✅  METADATA TEST PASSED!  ✅ ✅ ✅")
else:
    print("❌ ❌ ❌  METADATA TEST FAILED!  ❌ ❌ ❌")
print("=" * 80)

# Final summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Gene capitalization test: {'PASSED ✅' if all_correct else 'FAILED ❌'}")
print(f"Metadata preservation test: {'PASSED ✅' if metadata_correct else 'FAILED ❌'}")
print("\nGene name capitalization is working correctly!" if all_correct and metadata_correct else "\nPlease fix the capitalization function!")
print("=" * 80)
