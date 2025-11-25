# 1. Consensus Labeling

This directory contains scripts for creating consensus cell cycle labels by harmonizing predictions from 4 different computational tools. Since tools predict different phase granularities (3-8 phases), we use a systematic approach to map sub-phases to 3 main phases (G1, S, G2M) and merge predictions.

## Overview

**Purpose**: Create consensus training labels by:
1. Analyzing agreement between 4 tools using contingency tables
2. Mapping sub-phases to main phases based on observed/expected ratios
3. Requiring unanimous or near-unanimous agreement for final labels

**Why Consensus?**: No ground truth exists for REH/SUP-B15 training data. Consensus from multiple established tools provides more reliable labels than any single method.

## Directory Structure

```
1_consensus_labeling/
├── analyze/                           # Step 1 & 2: Contingency analysis
│   ├── create_contingency_table.py   # Compare tool predictions
│   └── generate_heatmaps.py          # Obs/expect ratios + heatmaps
│
├── assign/                            # Step 3: Manual phase mapping
│   └── apply_phase_reassignment.py   # Apply YAML mappings
│
└── merge/                             # Step 4: Create consensus
    └── merge_consensus.py            # Merge where ≥N tools agree
```

## Workflow

### Step 1: Create Contingency Tables
**Script**: `analyze/create_contingency_table.py`

Compare predictions between pairs of tools to identify agreement patterns.

```bash
python analyze/create_contingency_table.py \
  --tool1 path/to/seurat_predictions.csv \
  --tool2 path/to/tricycle_predictions.csv \
  --output contingency_seurat_tricycle.csv
```

**Output**: Contingency table showing how often each tool pair agrees on phase assignments.

Example:
```
           Seurat_G1  Seurat_S  Seurat_G2M
Tricycle_G1      1200        50         100
Tricycle_S         80       900          20
Tricycle_G2       100        30         850
Tricycle_M         50        10         600
```

### Step 2: Calculate Observed/Expected Ratios & Generate Heatmaps
**Script**: `analyze/generate_heatmaps.py`

Calculate how much each tool pair agrees beyond random chance, visualize with heatmaps.

```bash
python analyze/generate_heatmaps.py \
  --contingency contingency_seurat_tricycle.csv \
  --output heatmap_seurat_tricycle.png
```

**Formula**:
```
Observed/Expected Ratio = (Observed Count) / (Expected Count if Random)
Expected = (Row Total × Column Total) / Grand Total
```

**Output**:
- Heatmap showing obs/expect ratios
- High ratios (>5) = strong agreement (similar colors)
- Low ratios (<1) = disagreement (different colors)

**Key Insight**: Color similarity in heatmaps guides phase mapping decisions.

Example heatmap interpretation:
```
ccAFv2 Phase    | Seurat G1 | Seurat S | Seurat G2M
----------------|-----------|----------|------------
G1              | 🟩 8.2    | 🟥 0.1   | 🟥 0.2     → Map to G1
Late G1         | 🟨 3.5    | 🟨 2.1   | 🟥 0.3     → Ambiguous, use YAML
G1.S            | 🟥 0.5    | 🟩 7.8   | 🟥 0.1     → Map to S
S               | 🟥 0.2    | 🟩 9.1   | 🟥 0.1     → Map to S
G2              | 🟥 0.1    | 🟥 0.3   | 🟩 6.5     → Map to G2M
G2.M            | 🟥 0.1    | 🟥 0.2   | 🟩 8.9     → Map to G2M
M.G1            | 🟨 2.0    | 🟥 0.4   | 🟨 3.2     → Ambiguous, use YAML
```

### Step 3: Manual Phase Reassignment (YAML-Guided)
**Script**: `assign/apply_phase_reassignment.py`

Map sub-phases from each tool to 3 main phases (G1, S, G2M) using YAML configuration files.

**Why Manual?**: Different datasets show different phase relationships. Manual mapping based on heatmap analysis ensures biologically meaningful assignments.

**YAML Configuration Format**:
```yaml
# configs/phase_mappings/training_data.yaml
reh:
  seurat:
    G1: G1
    S: S
    G2M: G2M

  tricycle:
    G1: G1
    S: S
    G2: G2M
    M: G2M

  ccAFv2:
    G1: G1
    "Late G1": G1
    G1.S: S
    S: S
    G2: G2M
    G2.M: G2M
    G2/M: G2M
    M: G2M
    M.G1: G1
    Unknown: Unknown

  revelio:
    G0: G1
    G1: G1
    S: S
    G2: G2M
    M: G2M
```

**Usage**:
```bash
python assign/apply_phase_reassignment.py \
  --input path/to/ccafv2_predictions.csv \
  --config configs/phase_mappings/training_data.yaml \
  --dataset reh \
  --tool ccAFv2 \
  --output reassigned_ccafv2_reh.csv
```

**Output**: CSV with phases mapped to {G1, S, G2M, Unknown}.

### Step 4: Merge Consensus Labels
**Script**: `merge/merge_consensus.py`

Combine reassigned predictions from all 4 tools, keeping only cells where ≥N tools agree.

```bash
python merge/merge_consensus.py \
  --input assign/output/reh/ \
  --output consensus/ \
  --sample 1_GD428_21136_Hu_REH_Parental \
  --dataset reh
```

**Logic**:
1. Find cells where all 4 tools agree → highest confidence
2. Find cells where ≥3 tools agree → high confidence
3. Find cells where ≥2 tools agree → moderate confidence
4. Cells with conflicting predictions → labeled "Unique" (excluded)

**Output Files**:
- `{sample}_overlapped_all_four_regions.csv` - All 4 tools agree (best quality)
- `{sample}_overlapped_at_least_three_regions.csv` - ≥3 tools agree (recommended for training)
- `{sample}_overlapped_at_least_two_regions.csv` - ≥2 tools agree (larger dataset, lower quality)
- `{sample}_no_overlapped.csv` - Conflicting predictions (excluded)

**Recommended**: Use ≥3 tools agreement for training to balance data quality and quantity.

## Complete Pipeline Example

```bash
# 1. Create contingency tables (run for all tool pairs)
python analyze/create_contingency_table.py \
  --tool1 ../0_preprocessing/training_data/seurat_reh.csv \
  --tool2 ../0_preprocessing/training_data/tricycle_reh.csv \
  --output contingency/seurat_tricycle_reh.csv

# 2. Generate heatmaps (run for all tool pairs)
python analyze/generate_heatmaps.py \
  --contingency contingency/seurat_tricycle_reh.csv \
  --output heatmaps/seurat_tricycle_reh.png

# [MANUAL STEP]: Review heatmaps, create YAML mapping files

# 3. Apply phase reassignment (run for each tool)
python assign/apply_phase_reassignment.py \
  --input ../0_preprocessing/training_data/ccafv2_reh.csv \
  --config ../configs/phase_mappings/training_data.yaml \
  --dataset reh \
  --tool ccAFv2 \
  --output assign/output/reh/ccafv2_reh_reassigned.csv

# 4. Merge consensus labels
python merge/merge_consensus.py \
  --input assign/output/reh/ \
  --output consensus/ \
  --sample 1_GD428_21136_Hu_REH_Parental \
  --dataset reh
```

## Key Concepts

### Why Observed/Expected Ratios?
- **Random baseline**: If two tools assigned phases randomly, we'd expect certain counts just by chance
- **Observed/Expected > 1**: Tools agree more than random (good!)
- **Observed/Expected < 1**: Tools disagree more than random (conflicting biology)
- **High ratios (>5)**: Strong evidence for mapping sub-phases together

### Phase Mapping Rationale
Different datasets may have different optimal mappings because:
- Cell cycle speed varies by cell type
- Some phases are better separated in certain datasets
- Technical factors (sequencing depth, platform) affect phase resolution

**Example**: In REH cells, "Late G1" maps to G1, but in SUP-B15 it might map to S if those cells progress faster through G1/S transition.

### Quality Control
- **High agreement (all 4 tools)**: ~40-50% of cells (best quality)
- **Moderate agreement (≥3 tools)**: ~70-80% of cells (recommended)
- **Low agreement (≥2 tools)**: ~90% of cells (lower quality)
- **Conflicting predictions**: ~10-20% of cells (excluded)

## Configuration Files

Phase mapping YAML files are stored in `../configs/phase_mappings/`:
```
configs/phase_mappings/
├── training_data.yaml        # REH & SUP-B15 mappings
├── benchmark_gse146773.yaml  # GSE146773 mappings
└── benchmark_gse64016.yaml   # GSE64016 mappings
```

Each file contains tool-specific mappings for that dataset, informed by heatmap analysis.

## Output Format

Final consensus CSV:
```
CellID,Predicted
AAACAGCCAATATGGA-1,G1
AAACAGCCACATTGCA-1,S
AAACAGCCAGAATGAC-1,G2M
...
```

This file is used as training labels in `../2_model_training/`.

## Methodology Notes

This consensus labeling approach is a **key methodological contribution** of our study:
1. Transparent: YAML configs document all mapping decisions
2. Reproducible: Same inputs → same outputs
3. Dataset-specific: Respects biological differences between cell types
4. Conservative: Requires multiple tools to agree (reduces false positives)

## References

- Contingency table analysis: Standard statistical method for categorical agreement
- Observed/Expected ratios: Chi-square test component, measures deviation from independence
- Consensus labeling: Similar to voting-based ensemble methods in ML

## Next Steps

After creating consensus labels:
1. Move to `../2_model_training/` to train ML/DL models on consensus labels
2. Use `../configs/datasets.yaml` to specify paths to consensus CSV files
3. Train models with nested cross-validation
4. Evaluate on benchmark data with ground truth (GSE146773, GSE64016)
