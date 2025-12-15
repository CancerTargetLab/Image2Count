module load Anaconda3
conda activate geomx

python -m cellevaluation --vis_label_data cosmx_measuremens_flipped_y_dca.csv --h5ad_dir 'out/' --vis_name_pattern 'cosmx_6_6_[0-5]_all.h5ad' \
        --num_subgraphs_per_graph 36 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clustering_metrics --do_performance_metrics --do_pathway_metrics \
        --performance_metrics --figure_dir figures/cosmx/6_6/metrics_dca/
python -m cellevaluation --vis_label_data cosmx_measuremens_flipped_y_dca.csv --h5ad_dir 'out/' --vis_name_pattern 'cosmx_6_6_mean_all.h5ad' \
        --num_subgraphs_per_graph 36 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clustering_metrics --do_performance_metrics --do_pathway_metrics \
        --performance_metrics --figure_dir figures/cosmx/6_6/metrics_dca/mean/
python -m cellevaluation --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/6_6/mean/dca/ --merge \
        --embed_dir out/cosmx_6_6/ --vis_name cosmx_6_6_mean --visualize_expression --has_expr_data --raw_subset_dir cosmx_dca

python -m cellevaluation --vis_label_data cosmx_measuremens_flipped_y_dca.csv --h5ad_dir 'out/' --vis_name_pattern 'cosmx_17_0_[0-5]_all.h5ad' \
        --num_subgraphs_per_graph 36 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clustering_metrics --do_performance_metrics --do_pathway_metrics \
        --performance_metrics --figure_dir figures/cosmx/17_0/metrics_dca/
python -m cellevaluation --vis_label_data cosmx_measuremens_flipped_y_dca.csv --h5ad_dir 'out/' --vis_name_pattern 'cosmx_17_=_mean_all.h5ad' \
        --num_subgraphs_per_graph 36 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clustering_metrics --do_performance_metrics --do_pathway_metrics \
        --performance_metrics --figure_dir figures/cosmx/17_0/metrics_dca/mean/
python -m cellevaluation --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/17_0/mean/dca/ --merge \
        --embed_dir out/cosmx_17_0/ --vis_name cosmx_17_0_mean --visualize_expression --has_expr_data --raw_subset_dir cosmx_dca

python -m cellevaluation --vis_label_data cosmx_measuremens_flipped_y_dca.csv --h5ad_dir 'out/' --vis_name_pattern 'cosmx_lin_[0-5]_all.h5ad' \
        --num_subgraphs_per_graph 36 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clustering_metrics --do_performance_metrics --do_pathway_metrics \
        --performance_metrics --figure_dir figures/cosmx/lin/metrics_dca/
python -m cellevaluation --vis_label_data cosmx_measuremens_flipped_y_dca.csv --h5ad_dir 'out/' --vis_name_pattern 'cosmx_lin_mean_all.h5ad' \
        --num_subgraphs_per_graph 36 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clustering_metrics --do_performance_metrics --do_pathway_metrics \
        --performance_metrics --figure_dir figures/cosmx/lin/metrics_dca/mean/
python -m cellevaluation --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/lin/mean/dca/ --merge \
        --embed_dir out/cosmx_lin/ --vis_name cosmx_lin_mean --visualize_expression --has_expr_data --raw_subset_dir cosmx_dca
