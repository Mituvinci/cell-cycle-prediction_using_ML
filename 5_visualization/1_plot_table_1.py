#!/usr/bin/env python3
"""
Generate publication-quality grouped bar plot for cell cycle phase distribution
Output: PDF (editable in Adobe Illustrator) with Times New Roman font
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ============================================================================
# PUBLICATION-QUALITY SETTINGS
# ============================================================================

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# High-quality PDF settings
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable in Illustrator)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600  # High resolution for publication
plt.rcParams['savefig.format'] = 'pdf'

# Professional styling
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv("/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/5_visualization/plot_table1.csv")

# Clean dataset names for display
df["Dataset"] = df["Dataset"].astype(str)

# Reorder datasets: training first, then benchmarks
order_keywords = ["GSE75748", "PBMC", "Mouse Brain","Nestorova", "GSE146773", "GSE64016", "Buettner"]
df = df[~df["Dataset"].str.contains("REH|SUP", case=False)]
df["_order"] = df["Dataset"].apply(lambda d: next((i for i, kw in enumerate(order_keywords) if kw in d), 999))
df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

# ============================================================================
# CREATE PUBLICATION-QUALITY FIGURE
# ============================================================================

# Figure size optimized for 2-column journal layout (max width ~7 inches)
fig, ax = plt.subplots(figsize=(10, 6))

# Bar positions and width
gap = 0.6
x = np.arange(len(df), dtype=float)
x[4:] += gap  # add space around the separator line
width = 0.25

# Professional color scheme (colorblind-friendly)
# Based on Nature/Science journal standards
colors = {
    'G1': '#2E86AB',    # Blue
    'S': '#A23B72',     # Purple/Magenta
    'G2M': '#F18F01'    # Orange
}

# Create grouped bars
bars1 = ax.bar(x - width, df["G1"], width, label="G1",
               color=colors['G1'], edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x, df["S"], width, label="S",
               color=colors['S'], edgecolor='black', linewidth=0.8)
bars3 = ax.bar(x + width, df["G2M"], width, label="G2M",
               color=colors['G2M'], edgecolor='black', linewidth=0.8)

# Highlight imbalanced datasets with red boxes
for i, row in df.iterrows():
    if row["Imbalanced"] == "*":
        max_height = max(row["G1"], row["S"], row["G2M"])
        rect = mpl.patches.Rectangle((x[i]-0.45, 0), 0.9, max_height * 1.1,
                                     fill=False, edgecolor='red',
                                     linewidth=2.0, linestyle='--')
        ax.add_patch(rect)

# Vertical dashed line to separate training datasets from benchmarks
ax.axvline(x=(x[3] + x[4]) / 2, color='black', linestyle='--', linewidth=1.5)

# ============================================================================
# FONT SIZES (Publication Standard)
# ============================================================================

# X-axis: Dataset names
ax.set_xticks(x)
ax.set_xticklabels(df["Dataset"], rotation=45, ha="right", fontsize=12)

# Y-axis: Cell count numbers
ax.tick_params(axis='y', labelsize=12)

# Axis labels
ax.set_ylabel("Cell Count", fontsize=14, fontweight='bold')
ax.set_xlabel("Dataset", fontsize=14, fontweight='bold')

# Legend
legend = ax.legend(title="Cell Cycle Phase", fontsize=14, title_fontsize=14,
                   loc='upper right', frameon=True, fancybox=False,
                   edgecolor='black', framealpha=1.0)
legend.get_frame().set_linewidth(1.5)

# Optional title (often omitted in journal figures, added in caption instead)
# Uncomment if you want a title:
# ax.set_title("Cell Cycle Phase Distribution Across Datasets",
#              fontsize=14, fontweight='bold', pad=20)

# ============================================================================
# PROFESSIONAL STYLING
# ============================================================================

# Remove top and right spines (cleaner look)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set spine colors to black
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Add subtle horizontal gridlines for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
ax.set_axisbelow(True)  # Grid behind bars

# Set y-axis to start at 0
ax.set_ylim(bottom=0)

# Tight layout to prevent label cutoff
plt.tight_layout()

# ============================================================================
# SAVE HIGH-QUALITY OUTPUTS
# ============================================================================

output_base = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/5_visualization/diagnostic_results/grouped_phase_counts"

# Save as PDF (vector format, editable in Adobe Illustrator)
pdf_file = f"{output_base}.pdf"
plt.savefig(pdf_file, format='pdf', dpi=600, bbox_inches='tight',
            transparent=False, facecolor='white')
print(f"PDF saved: {pdf_file}")

# Also save as high-res PNG (for presentations/previews)
png_file = f"{output_base}_highres.png"
plt.savefig(png_file, format='png', dpi=600, bbox_inches='tight',
            transparent=False, facecolor='white')
print(f"PNG saved: {png_file}")

# Also save as EPS (alternative vector format for some journals)
eps_file = f"{output_base}.eps"
plt.savefig(eps_file, format='eps', dpi=600, bbox_inches='tight')
print(f"EPS saved: {eps_file}")

plt.close()

print("\nPublication-quality figure generation complete!")
print("Files are ready for Adobe Illustrator and journal submission.")
print("\nFont: Times New Roman")
print("Format: PDF (vector), PNG (raster), EPS (vector)")
print("Resolution: 600 DPI")
print("Font sizes:")
print("  - Tick labels: 12pt")
print("  - Axis labels: 14pt (bold)")
print("  - Legend: 14pt")
