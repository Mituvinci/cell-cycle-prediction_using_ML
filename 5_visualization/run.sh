#!/usr/bin/env bash





python 5_visualization/3_generate_all_benchmark_results.py --input results/ft_7408_submitted_human_models_all_result.csv 

python 5_visualization/5_plot_tool_comparison_barplot.py --training-dataset nft_reh




python 5_visualization/3_generate_all_benchmark_results.py --input results/ft_5770_std_human_hpsc.csv 

python 5_visualization/5_plot_tool_comparison_barplot.py --training-dataset nft_hpsc





python 5_visualization/3_generate_all_benchmark_results.py --input results/ft_6317_human_pbmc.csv 

python 5_visualization/5_plot_tool_comparison_barplot.py --training-dataset nft_pbmc





python 5_visualization/3_generate_all_benchmark_results.py --input results/ft_6466_mouse_brain.csv 

python 5_visualization/5_plot_tool_comparison_barplot.py --training-dataset nft_mouse_brain





