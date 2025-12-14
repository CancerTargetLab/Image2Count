conda activate geomx

python -m cellcontrast --image_preprocess --preprocess_dir data/raw/CRC_2c_1p/train/ --cell_cutout 34 --preprocess_workers 20 \
        --calc_mean_std --preprocess_channels 0,10
python -m cellcontrast --image_preprocess --preprocess_dir data/raw/CRC_2c_1p/test/ --cell_cutout 34 --preprocess_workers 10 \
        --preprocess_channels 0,10

python -m cellcontrast --train_image_model --output_name out/models/crc_2c_1p_image_model_res18_32_16.pt --early_stopping 1000 \
        --embed 32 --contrast 16 --resnet_model '18'  --num_workers 24 --epochs 100 --warmup_epochs_image 10 --lr 0.1  \
        --batch_size_image 4096 --image_dir data/raw/CRC_2c_1p/ --n_clusters_image 10 --crop_factor 1 --weight_decay 1e-6
python -m cellcontrast --embed_image_data --output_name_image out/models/crc_2c_1p_image_model_res18_32_16.pt \
        --early_stopping 1000 --embed 32 --contrast 16 --resnet_model '18'  --num_workers 24 --batch_size_image 4096 \
        --image_dir data/raw/CRC_2c_1p/ 

python -m cellprediction --model_type Image2Count --raw_subset_dir CRC_2c_1p --batch_size 64 --epochs 300 --num_workers 20 \
        --num_node_features 32 --num_gat_features 32 --num_embed_features 128 --lr 0.0001 --lin_layers 35 --gat_layers 0 \
        --output_name out/models/crc_2c_1p_35_0.pt  --label_data CRC_1p_label.csv \
        --early_stopping 25 --heads 8 --node_dropout 0.05 --embed_dropout 0.1 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 10 --subgraphs_per_graph 49 --num_hops 11 --weight_decay 5e-4
python -m cellprediction --model_type Image2Count --raw_subset_dir CRC_2c_1p --batch_size 64 --epochs 300 --num_workers 20 \
        --num_node_features 32 --num_gat_features 32 --num_embed_features 128 --lr 0.0001 --lin_layers 24 --gat_layers 6 \
        --output_name out/models/crc_2c_1p_24_6.pt  --label_data CRC_1p_label.csv \
        --early_stopping 25 --heads 8 --node_dropout 0.05 --embed_dropout 0.1 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 10 --subgraphs_per_graph 49 --num_hops 11 --weight_decay 5e-4
python -m cellprediction --model_type LIN --raw_subset_dir CRC_2c_1p --batch_size 64 --epochs 300 --num_workers 20 \
        --num_node_features 32 --lr 0.05 --lin_layers 24 --output_name out/models/crc_2c_1p_lin.pt  --label_data CRC_1p_label.csv \
        --early_stopping 25 --node_dropout 0.05 --embed_dropout 0.1 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 10 --subgraphs_per_graph 49 --num_hops 11 --weight_decay 5e-4


for m in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9';
do

    python -m cellprediction --model_type Image2Count --raw_subset_dir CRC_2c_1p --batch_size 64 --num_workers 8 --num_node_features 32 \
        --num_embed_features 128  --lin_layers 35 --gat_layers 0 --output_name out/models/crc_2c_1p_35_0/$m.pt --output_graph_embed out/crc_2c_1p_35_0/$m/ \
        --label_data CRC_1p_label.csv --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m cellevaluation --vis_select_cells 50000 --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_2c_1p/test --figure_dir figures/crc_2c_1p/24_6/$m/ \
            --embed_dir out/crc_2c_1p_35_0/$m/ --vis_name crc_2c_1p_35_0_$m --visualize_expression --has_expr_data --raw_subset_dir CRC_2c_1p
    python -m cellevaluation --visualize_model_run --model_path out/models/crc_2c_1p_35_0/$m.pt --output_model_name CRC_2c_1p_"$m"_35_0 --figure_model_dir figures/crc_2c_1p/35_0/$m/

    python -m cellprediction --model_type Image2Count --raw_subset_dir CRC_2c_1p --batch_size 64 --num_workers 8 --num_node_features 32 --num_gat_features 32 \
        --num_embed_features 128  --lin_layers 24 --gat_layers 6 --output_name out/models/crc_2c_1p_24_06/$m.pt --output_graph_embed out/crc_2c_1p_24_6/$m/ \
        --label_data CRC_1p_label.csv --heads 8 --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m cellevaluation --vis_select_cells 50000 --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_2c_1p/test --figure_dir figures/crc_2c_1p/24_6/$m/ \
            --embed_dir out/crc_2c_1p_24_6/$m/ --vis_name crc_2c_1p_24_6_$m --visualize_expression --has_expr_data --raw_subset_dir CRC_2c_1p
    python -m cellevaluation --visualize_model_run --model_path out/models/crc_2c_1p_24_6/$m.pt --output_model_name CRC_2c_1p_"$m"_24_6 --figure_model_dir figures/crc_2c_1p/24_6/$m/

    python -m cellprediction --model_type LIN --raw_subset_dir CRC_2c_1p --batch_size 64 --num_workers 8 --num_node_features 32 \
        --output_name out/models/crc_2c_1p_lin/$m.pt --output_graph_embed out/crc_2c_1p_lin/$m/ \
        --label_data CRC_1p_label.csv --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m cellevaluation --vis_select_cells 50000 --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_2c_1p/test --figure_dir figures/crc_2c_1p/lin/$m/ \
            --embed_dir out/crc_2c_1p_lin/$m/ --vis_name crc_2c_1p_lin_$m --visualize_expression --has_expr_data --raw_subset_dir CRC_2c_1p
    python -m cellevaluation --visualize_model_run --model_path out/models/crc_2c_1p_lin/$m.pt --output_model_name CRC_2c_1p_"$m"_lin --figure_model_dir figures/crc_2c_1p/lin/$m/

done

python -m cellevaluation --vis_label_data CRC_1p/CRC_1p_measurements.csv --h5ad_dir 'out/' --vis_name_pattern 'crc_2c_1p_35_0_[0-9]_all.h5ad' \
        --num_subgraphs_per_graph 900 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clusterin_metrics --do_performance_metrics \
        --performance_metrics --figure_dir figures/crc_2c_1p/35_0/metrics/
python -m cellevaluation --merge --embed_dir out/crc_2c_1p_35_0/ --h5ad_dir 'out/' --vis_label_data CRC_1p/CRC_1p_measurements.csv \
        --processed_subset_dir CRC_2c_1p/test --vis_name crc_2c_1p_35_0_mean --embed_to_h5ad
python -m cellevaluation --vis_label_data CRC_1p/CRC_1p_measurements.csv --h5ad_dir 'out/' --vis_name_pattern 'crc_2c_1p_35_0_mean_all.h5ad' \
        --num_subgraphs_per_graph 900 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clusterin_metrics --do_performance_metrics \
        --performance_metrics --figure_dir figures/crc_2c_1p/35_0/metrics/mean/
python -m cellevaluation --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_2c_1p/test --figure_dir figures/crc_2c_1p/35_0/mean/ --merge \
        --embed_dir out/crc_2c_1p_35_0/ --vis_name crc_2c_1p_35_0_mean --visualize_expression --has_expr_data --raw_subset_dir CRC_2c_1p
python -m cellevaluation --visualize_image --vis_name crc_2c_1p_35_0_mean.h5ad --vis_img_raw_subset_dir CRC_2c_1p --name_tiff CRC03.ome.tif \
        --figure_img_dir figures/crc_2c_1p/35_0/mean/crc03/ \
        --vis_protein Hoechst1,CD3,Ki67,CD4,CD20,CD163,Ecadherin,LaminABC,PCNA,NaKATPase,Keratin,CD45,CD68,FOXP3,Vimentin,Desmin,Ki67_570,CD45RO,aSMA,PD1,CD8a,PDL1,CDX2,CD31,Collagen \
        --vis_img_xcoords 22201 25893 --vis_img_ycoords 14729 18421 --vis_all_channels

python -m cellevaluation --vis_label_data CRC_1p/CRC_1p_measurements.csv --h5ad_dir 'out/' --vis_name_pattern 'crc_2c_1p_24_6_[0-9]_all.h5ad' \
        --num_subgraphs_per_graph 900 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clusterin_metrics --do_performance_metrics \
        --performance_metrics --figure_dir figures/crc_2c_1p/24_6/metrics/
python -m cellevaluation --merge --embed_dir out/crc_2c_1p_24_6/ --h5ad_dir 'out/' --vis_label_data CRC_1p/CRC_1p_measurements.csv \
        --processed_subset_dir CRC_2c_1p/test --vis_name crc_2c_1p_24_6_mean --embed_to_h5ad
python -m cellevaluation --vis_label_data CRC_1p/CRC_1p_measurements.csv --h5ad_dir 'out/' --vis_name_pattern 'crc_2c_1p_24_6_mean_all.h5ad' \
        --num_subgraphs_per_graph 900 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clusterin_metrics --do_performance_metrics \
        --performance_metrics --figure_dir figures/crc_2c_1p/24_6/metrics/mean/
python -m cellevaluation --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_2c_1p/test --figure_dir figures/crc_2c_1p/24_6/mean/ --merge \
        --embed_dir out/crc_2c_1p_24_6/ --vis_name crc_2c_1p_24_6_mean --visualize_expression --has_expr_data --raw_subset_dir CRC_2c_1p
python -m cellevaluation --visualize_image --vis_name crc_2c_1p_24_6_mean.h5ad --vis_img_raw_subset_dir CRC_2c_1p --name_tiff CRC03.ome.tif \
        --figure_img_dir figures/crc_2c_1p/24_6/mean/crc03/ \
        --vis_protein Hoechst1,CD3,Ki67,CD4,CD20,CD163,Ecadherin,LaminABC,PCNA,NaKATPase,Keratin,CD45,CD68,FOXP3,Vimentin,Desmin,Ki67_570,CD45RO,aSMA,PD1,CD8a,PDL1,CDX2,CD31,Collagen \
        --vis_img_xcoords 22201 25893 --vis_img_ycoords 14729 18421 --vis_all_channels

python -m cellevaluation --vis_label_data CRC_1p/CRC_1p_measurements.csv --h5ad_dir 'out/' --vis_name_pattern 'crc_2c_1p_24_6_[0-9]_all.h5ad' \
        --num_subgraphs_per_graph 900 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clusterin_metrics --do_performance_metrics \
        --performance_metrics --figure_dir figures/crc_2c_1p/24_6/metrics/
python -m cellevaluation --merge --embed_dir out/crc_2c_1p_24_6/ --h5ad_dir 'out/' --vis_label_data CRC_1p/CRC_1p_measurements.csv \
        --processed_subset_dir CRC_2c_1p/test --vis_name crc_2c_1p_24_6_mean --embed_to_h5ad
python -m cellevaluation --vis_label_data CRC_1p/CRC_1p_measurements.csv --h5ad_dir 'out/' --vis_name_pattern 'crc_2c_1p_24_6_mean_all.h5ad' \
        --num_subgraphs_per_graph 900 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clusterin_metrics --do_performance_metrics \
        --performance_metrics --figure_dir figures/crc_2c_1p/24_6/metrics/mean/
python -m cellevaluation --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_2c_1p/test --figure_dir figures/crc_2c_1p/lin/mean/ --merge \
        --embed_dir out/crc_2c_1p_lin/ --vis_name crc_2c_1p_lin_mean --visualize_expression --has_expr_data --raw_subset_dir CRC_2c_1p
python -m cellevaluation --visualize_image --vis_name crc_2c_1p_lin_mean.h5ad --vis_img_raw_subset_dir CRC_2c_1p --name_tiff CRC03.ome.tif \
        --figure_img_dir figures/crc_2c_1p/lin/mean/crc03/ \
        --vis_protein Hoechst1,CD3,Ki67,CD4,CD20,CD163,Ecadherin,LaminABC,PCNA,NaKATPase,Keratin,CD45,CD68,FOXP3,Vimentin,Desmin,Ki67_570,CD45RO,aSMA,PD1,CD8a,PDL1,CDX2,CD31,Collagen \
        --vis_img_xcoords 22201 25893 --vis_img_ycoords 14729 18421 --vis_all_channels
