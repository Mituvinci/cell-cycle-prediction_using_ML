#!/usr/bin/env bash


# This script automatically runs EVERYTHING in one command:
#   1. Generates benchmark CSV files
#   2. Generates precision/recall heatmap
#   3. Generates benchmark line plots (calls 4_plot_benchmark_results.py)
#   4. Generates tool comparison bar plot (calls 5_plot_tool_comparison_barplot.py)




python 5_visualization/3_generate_all_benchmark_results.py \
    --input results/ft_5770_std_human_hpsc.csv \
    --output-name "nft_5770_std_human_hpsc_2nd"




python 5_visualization/3_generate_all_benchmark_results.py \
    --input results/ft_6317_human_pbmc.csv \
    --output-name "nft_6317_human_pbmc_2nd"


python 5_visualization/3_generate_all_benchmark_results.py \
    --input results/ft_6466_mouse_brain.csv \
    --output-name "nft_6466_mouse_brain_2nd"


python 5_visualization/3_generate_all_benchmark_results.py \
    --input results/ft_nestorova.csv \
    --output-name "nft_nestorova_2nd"



python 5_visualization/3_generate_all_benchmark_results.py \
    --input results/ft_sup_robust_result_double.csv \
    --output-name "nft_sup_robust_double_2nd"
