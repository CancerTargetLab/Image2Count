conda activate geomx

python -m cellcontrast --image_preprocess --preprocess_dir data/raw/hkgmh3_74_50/train/ --cell_cutout 20 --preprocess_workers 20 \
        --calc_mean_std
python -m cellcontrast --image_preprocess --preprocess_dir data/raw/hkgmh3_74_50/test/ --cell_cutout 20 --preprocess_workers 10

python -m cellcontrast --train_image_model --output_name out/models/hkgmh3_74_50_image_model_res18_32_16.pt --early_stopping 1000 \
        --embed 32 --contrast 16 --resnet_model '18'  --num_workers 24 --epochs 950 --warmup_epochs_image 95 --lr 0.1 \
        --batch_size_image 1024 --image_dir data/raw/hkgmh3_74_50/ --n_clusters_image 20 --crop_factor 0.5 --weight_decay 1e-6
python -m cellcontrast --embed_image_data --output_name_image out/models/hkgmh3_74_50_image_model_res18_32_16.pt \
        --early_stopping 1000 --embed 32 --contrast 16 --resnet_model '18'  --num_workers 24 --batch_size_image 4096 \
        --image_dir data/raw/hkgmh3_74_50/

python -m cellprediction --model_type Image2Count --raw_subset_dir hkgmh3_74_50 --batch_size 64 --epochs 300 --num_workers 20 \
        --num_node_features 32 --num_gat_features 8 --num_embed_features 32 --lr 0.0005 --lin_layers 7 --gat_layers 0 \
        --output_name out/models/hkgmh3_74_50_7_0.pt  --label_data hkgmh3_74_50_label.csv \
        --early_stopping 50 --heads 2 --node_dropout 0.05 --embed_dropout 0.5 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 10 --weight_decay 1e-2
python -m cellprediction --model_type Image2Count --raw_subset_dir hkgmh3_74_50 --batch_size 64 --epochs 300 --num_workers 20 \
        --num_node_features 32 --num_gat_features 8 --num_embed_features 32 --lr 0.0005 --lin_layers 3 --gat_layers 3 \
        --output_name out/models/hkgmh3_74_50_3_3.pt  --label_data hkgmh3_74_50_label.csv \
        --early_stopping 50 --heads 2 --node_dropout 0.05 --embed_dropout 0.5 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 10 --weight_decay 1e-2
python -m cellprediction --model_type LIN --raw_subset_dir hkgmh3_74_50 --batch_size 64 --epochs 300 --num_workers 20 \
        --num_node_features 32 --lr 0.01 --lin_layers 3 --output_name out/models/hkgmh3_74_50_lin.pt  --label_data hkgmh3_74_50_label.csv \
        --early_stopping 25 --node_dropout 0.05 --embed_dropout 0.1 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 10 --weight_decay 5e-4


for m in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9';
do

    python -m cellprediction --model_type Image2Count --raw_subset_dir hkgmh3_74_50 --batch_size 64 --num_workers 8 --num_node_features 32 \
        --num_embed_features 32  --lin_layers 7 --gat_layers 0 --output_name out/models/hkgmh3_74_50_7_0/$m.pt --output_graph_embed out/hkgmh3_74_50_7_0/$m/ \
        --label_data hkgmh3_74_50_label.csv --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m cellevaluation --vis_select_cells 50000 --vis_label_data hkgmh3_74_50_label.csv --processed_subset_dir hkgmh3_74_50/test --figure_dir figures/hkgmh3_74_50/3_3/$m/ \
            --embed_dir out/hkgmh3_74_50_7_0/$m/ --vis_name hkgmh3_74_50_7_0_$m --visualize_expression --raw_subset_dir hkgmh3_74_50
    python -m cellevaluation --visualize_model_run --model_path out/models/hkgmh3_74_50_7_0/$m.pt --output_model_name hkgmh3_74_50_"$m"_7_0 --figure_model_dir figures/hkgmh3_74_50/7_0/$m/

    python -m cellprediction --model_type Image2Count --raw_subset_dir hkgmh3_74_50 --batch_size 64 --num_workers 8 --num_node_features 32 --num_gat_features 8 \
        --num_embed_features 32  --lin_layers 3 --gat_layers 3 --output_name out/models/hkgmh3_74_50_24_06/$m.pt --output_graph_embed out/hkgmh3_74_50_3_3/$m/ \
        --label_data hkgmh3_74_50_label.csv --heads 2 --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m cellevaluation --vis_select_cells 50000 --vis_label_data hkgmh3_74_50_label.csv --processed_subset_dir hkgmh3_74_50/test --figure_dir figures/hkgmh3_74_50/3_3/$m/ \
            --embed_dir out/hkgmh3_74_50_3_3/$m/ --vis_name hkgmh3_74_50_3_3_$m --visualize_expression --raw_subset_dir hkgmh3_74_50
    python -m cellevaluation --visualize_model_run --model_path out/models/hkgmh3_74_50_3_3/$m.pt --output_model_name hkgmh3_74_50_"$m"_3_3 --figure_model_dir figures/hkgmh3_74_50/3_3/$m/

    python -m cellprediction --model_type LIN --raw_subset_dir hkgmh3_74_50 --batch_size 64 --num_workers 8 --num_node_features 32 \
        --output_name out/models/hkgmh3_74_50_lin/$m.pt --output_graph_embed out/hkgmh3_74_50_lin/$m/ \
        --label_data hkgmh3_74_50_label.csv --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m cellevaluation --vis_select_cells 50000 --vis_label_data hkgmh3_74_50_label.csv --processed_subset_dir hkgmh3_74_50/test --figure_dir figures/hkgmh3_74_50/lin/$m/ \
            --embed_dir out/hkgmh3_74_50_lin/$m/ --vis_name hkgmh3_74_50_lin_$m --visualize_expression --raw_subset_dir hkgmh3_74_50
    python -m cellevaluation --visualize_model_run --model_path out/models/hkgmh3_74_50_lin/$m.pt --output_model_name hkgmh3_74_50_"$m"_lin --figure_model_dir figures/hkgmh3_74_50/lin/$m/

done

python -m cellevaluation --vis_label_data hkgmh3_74_30_label.csv --h5ad_dir 'out/' --vis_name_pattern 'hkgmh3_74_30_7_0_[0-9]_all.h5ad' \
        --do_clustering_metrics --do_performance_metrics --sum_by_subgraph \
        --performance_metrics --figure_dir figures/hkgmh3_74_30/7_0/metrics/
python -m cellevaluation --merge --embed_dir out/hkgmh3_74_30_7_0/ --h5ad_dir 'out/' --vis_label_data hkgmh3_74_30_label.csv \
        --processed_subset_dir hkgmh3_74_30/test --vis_name hkgmh3_74_30_7_0_mean --embed_to_h5ad
python -m cellevaluation --vis_label_data hkgmh3_74_30_label.csv --h5ad_dir 'out/' --vis_name_pattern 'hkgmh3_74_30_7_0_mean_all.h5ad' \
        --do_clustering_metrics --do_performance_metrics  --sum_by_subgraph \
        --performance_metrics --figure_dir figures/hkgmh3_74_30/7_0/metrics/mean/
python -m cellevaluation --vis_label_data hkgmh3_74_50_label.csv --processed_subset_dir hkgmh3_74_50/test --figure_dir figures/hkgmh3_74_50/7_0/mean/ \
        --embed_dir out/hkgmh3_74_50_7_0/ --vis_name hkgmh3_74_50_7_0_mean --visualize_expression --raw_subset_dir hkgmh3_74_50
python -m cellevaluation --visualize_image --vis_name hkgmh3_74_50_7_0_mean.h5ad --vis_img_raw_subset_dir hkgmh3_74_50 --name_tiff 019-1C54.tiff \
        --figure_img_dir figures/hkgmh3_74_50/7_0/mean/019-1C54/ --vis_all_channels \
        --vis_protein PanCk,Ki-67,SMA,CD8,CD4,CD3,PD-1,CD11c,CD68,CD45,Fibronectin,Pan-AKT,BCL6,BCLXL,BAD,p53,CD163,CD14,CD34,CD45RO,STING,B7-H3,CD44,CD127 \

python -m cellevaluation --vis_label_data hkgmh3_74_30_label.csv --h5ad_dir 'out/' --vis_name_pattern 'hkgmh3_74_30_3_3_[0-9]_all.h5ad' \
        --do_clustering_metrics --do_performance_metrics --sum_by_subgraph \
        --performance_metrics --figure_dir figures/hkgmh3_74_30/3_3/metrics/
python -m cellevaluation --merge --embed_dir out/hkgmh3_74_30_3_3/ --h5ad_dir 'out/' --vis_label_data hkgmh3_74_30_label.csv \
        --processed_subset_dir hkgmh3_74_30/test --vis_name hkgmh3_74_30_3_3_mean --embed_to_h5ad
python -m cellevaluation --vis_label_data hkgmh3_74_30_label.csv --h5ad_dir 'out/' --vis_name_pattern 'hkgmh3_74_30_lin_mean_all.h5ad' \
        --do_clustering_metrics --do_performance_metrics  --sum_by_subgraph \
        --performance_metrics --figure_dir figures/hkgmh3_74_30/3_#/metrics/mean/
python -m cellevaluation --vis_label_data hkgmh3_74_50_label.csv --processed_subset_dir hkgmh3_74_50/test --figure_dir figures/hkgmh3_74_50/3_3/mean/ \
        --embed_dir out/hkgmh3_74_50_3_3/ --vis_name hkgmh3_74_50_3_3_mean --visualize_expression --raw_subset_dir hkgmh3_74_50
python -m cellevaluation --visualize_image --vis_name hkgmh3_74_50_3_3_mean.h5ad --vis_img_raw_subset_dir hkgmh3_74_50 --name_tiff 019-1C54.tiff \
        --figure_img_dir figures/hkgmh3_74_50/3_3/mean/019-1C54/ --vis_all_channels \
        --vis_protein PanCk,Ki-67,SMA,CD8,CD4,CD3,PD-1,CD11c,CD68,CD45,Fibronectin,Pan-AKT,BCL6,BCLXL,BAD,p53,CD163,CD14,CD34,CD45RO,STING,B7-H3,CD44,CD127 \

python -m cellevaluation --vis_label_data hkgmh3_74_30_label.csv --h5ad_dir 'out/' --vis_name_pattern 'hkgmh3_74_30_lin_[0-9]_all.h5ad' \
        --do_clustering_metrics --do_performance_metrics --sum_by_subgraph \
        --performance_metrics --figure_dir figures/hkgmh3_74_30/lin/metrics/
python -m cellevaluation --merge --embed_dir out/hkgmh3_74_30_lin/ --h5ad_dir 'out/' --vis_label_data hkgmh3_74_30_label.csv \
        --processed_subset_dir hkgmh3_74_30/test --vis_name hkgmh3_74_30_lin_mean --embed_to_h5ad
python -m cellevaluation --vis_label_data hkgmh3_74_30_label.csv --h5ad_dir 'out/' --vis_name_pattern 'hkgmh3_74_30_lin_mean_all.h5ad' \
        --do_clustering_metrics --do_performance_metrics  --sum_by_subgraph \
        --performance_metrics --figure_dir figures/hkgmh3_74_30/lin/metrics/mean/
python -m cellevaluation --vis_label_data hkgmh3_74_50_label.csv --processed_subset_dir hkgmh3_74_50/test --figure_dir figures/hkgmh3_74_50/lin/mean/ \
        --embed_dir out/hkgmh3_74_50_lin/ --vis_name hkgmh3_74_50_lin_mean --visualize_expression --raw_subset_dir hkgmh3_74_50
python -m cellevaluation --visualize_image --vis_name hkgmh3_74_50_lin_mean.h5ad --vis_img_raw_subset_dir hkgmh3_74_50 --name_tiff 019-1C54.tiff \
        --figure_img_dir figures/hkgmh3_74_50/lin/mean/019-1C54/ --vis_all_channels \
        --vis_protein PanCk,Ki-67,SMA,CD8,CD4,CD3,PD-1,CD11c,CD68,CD45,Fibronectin,Pan-AKT,BCL6,BCLXL,BAD,p53,CD163,CD14,CD34,CD45RO,STING,B7-H3,CD44,CD127 \
