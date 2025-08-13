module load Anaconda3
conda activate geomx

python -m main --image_preprocess --preprocess_dir data/raw/cosmx/train/ --cell_cutout 100 --preprocess_workers 15 --calc_mean_std
python -m main --image_preprocess --preprocess_dir data/raw/cosmx/test/ --cell_cutout 100 --preprocess_workers 15

python src/utils/cells_kmeans.py

python -m main --train_image_model --output_name out/models/cosmx_image_model_100_512_128.pt --early_stopping 1000 --embed 512 --contrast 128 --resnet '50'  --num_workers 30 --epochs 300 --warmup_epochs 30 --crop_factor 0.5 --batch_size 512 --raw_subset_dir cosmx --n_clusters_image 30 --lr 0.1

python -m main --visualize_model_run --model_path out/models/cosmx_image_model_100_512_128.pt --output_model_name cosmx_image_model_100_512_128 --figure_model_dir figures/image_contrast/
python -m main --embed_image_data --output_name out/models/cosmx_image_model_100_512_128.pt --early_stopping 1000 --embed 512 --contrast 128 --resnet '50'  --num_workers 30 --epochs 300 --warmup_epochs 30 --crop_factor 0.5 --batch_size 512 --raw_subset_dir cosmx

python -m main --model_type Image2Count --raw_subset_dir cosmx --batch_size 64 --epochs 300 --num_workers 4 \
        --num_node_features 512 --num_gat_features 64 --num_embed_features 128 --lr 0.0001 --lin_layers 6 --gat_layers 6 \
        --output_name out/models/cosmx_6_6.pt  --label_data cosmx_label.csv \
        --early_stopping 25 --heads 8 --node_dropout 0.05 --embed_dropout 0.5 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 6 --subgraphs_per_graph 4 --num_hops 11  --weight_decay 5e-3

python -m main --model_type Image2Count --raw_subset_dir cosmx --batch_size 64 --epochs 300 --num_workers 4 \
        --num_node_features 512 --num_gat_features 64 --num_embed_features 128 --lr 0.0001 --lin_layers 17 --gat_layers 0 \
        --output_name out/models/cosmx_17_0.pt  --label_data cosmx_label.csv \
        --early_stopping 25 --heads 8 --node_dropout 0.05 --embed_dropout 0.5 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 6 --subgraphs_per_graph 4 --num_hops 11  --weight_decay 5e-3

python -m main --model_type LIN --raw_subset_dir cosmx --batch_size 64 --epochs 300 --num_workers 4 \
        --num_node_features 512 --lr 0.005 --output_name out/models/cosmx_lin.pt  --label_data cosmx_label.csv \
        --early_stopping 25 --node_dropout 0.05 --embed_dropout 0.5 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 6 --subgraphs_per_graph 4 --num_hops 11 --weight_decay 5e-3

for m in '0' '1' '2' '3' '4' '5';
do

    python -m main --model_type Image2Count --raw_subset_dir cosmx --batch_size 64 --num_workers 8 --num_node_features 512 \
        --num_embed_features 128  --lin_layers 6 --gat_layers 6 --output_name out/models/cosmx_6_6/$m.pt --output_graph_embed out/cosmx_6_6/$m/ \
        --label_data cosmx_label.csv --subgraphs_per_graph 0 --num_gat_features 64 --heads 8 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m main --vis_select_cells 50000 --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/6_6/$m/ \
            --embed_dir out/cosmx_6_6/$m/ --vis_name cosmx_6_6_$m --visualize_expression --has_expr_data --raw_subset_dir cosmx
    python -m main --visualize_model_run --model_path out/models/cosmx_6_6/$m.pt --output_model_name cosmx_"$m"_6_6 --figure_model_dir figures/cosmx/6_6/$m/

    python -m main --model_type Image2Count --raw_subset_dir cosmx --batch_size 64 --num_workers 8 --num_node_features 512 --num_gat_features 64 \
        --num_embed_features 128  --lin_layers 17 --gat_layers 0 --output_name out/models/cosmx_17_0/$m.pt --output_graph_embed out/cosmx_17_0/$m/ \
        --label_data cosmx_label.csv --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m main --vis_select_cells 50000 --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/17_0/$m/ \
            --embed_dir out/cosmx_17_0/$m/ --vis_name cosmx_17_0_$m --visualize_expression --has_expr_data --raw_subset_dir cosmx
    python -m main --visualize_model_run --model_path out/models/cosmx_17_0/$m.pt --output_model_name cosmx_"$m"_17_0 --figure_model_dir figures/cosmx/17_0/$m/

    python -m main --model_type LIN --raw_subset_dir cosmx --batch_size 64 --num_workers 8 --num_node_features 512 \
        --output_name out/models/cosmx_lin/$m.pt --output_graph_embed out/cosmx_lin/$m/ \
        --label_data cosmx_label.csv --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m main --vis_select_cells 50000 --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/lin/$m/ \
            --embed_dir out/cosmx_lin/$m/ --vis_name cosmx_lin_$m --visualize_expression --has_expr_data --raw_subset_dir cosmx
    python -m main --visualize_model_run --model_path out/models/cosmx_lin/$m.pt --output_model_name cosmx_"$m"_lin --figure_model_dir figures/cosmx/lin/$m/

done

python -m main --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/6_6/mean/ --merge \
        --embed_dir out/cosmx_6_6/ --vis_name cosmx_6_6_mean --visualize_expression --has_expr_data --raw_subset_dir cosmx
python -m main --visualize_image --vis_name cosmx_6_6_mean.h5ad --vis_img_raw_subset_dir cosmx/test --name_tiff 20210706_105344_S2_C902_P99_N99_F020_Z001.TIF --vis_all_channels \
        --vis_protein 'PTPRC,CD74,CD9,CD163,CD68,CD14,CD4,CD34,CD3D,CD3E,CD3G,CDH1,MKI67,TP53,ICOS,IDO1,HLA-A,HLA-B,HLA-C,KRT1,KRT14,KRT15,TYK2,KRT17,KRT19,JAG1,MTRNR2L1,CRP,TM4SF1,PTGDR2,C1QA,MS4A1,MS4A4A,MET,CD8A,CD8B,CD84,CD127,FN1,FOXP3,DDR1,CTLA4,CD44,CD274,CD37' \
        --figure_img_dir figures/cosmx/6_6/mean/20210706_105344_S2_C902_P99_N99_F020_Z001/

python -m main --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/17_0/mean/ --merge \
        --embed_dir out/cosmx_17_0/ --vis_name cosmx_17_0_mean --visualize_expression --has_expr_data --raw_subset_dir cosmx
python -m main --visualize_image --vis_name cosmx_17_0_mean.h5ad --vis_img_raw_subset_dir cosmx/test --name_tiff 20210706_105344_S2_C902_P99_N99_F020_Z001.TIF --vis_all_channels \
        --vis_protein 'PTPRC,CD74,CD9,CD163,CD68,CD14,CD4,CD34,CD3D,CD3E,CD3G,CDH1,MKI67,TP53,ICOS,IDO1,HLA-A,HLA-B,HLA-C,KRT1,KRT14,KRT15,TYK2,KRT17,KRT19,JAG1,MTRNR2L1,CRP,TM4SF1,PTGDR2,C1QA,MS4A1,MS4A4A,MET,CD8A,CD8B,CD84,CD127,FN1,FOXP3,DDR1,CTLA4,CD44,CD274,CD37' \
        --figure_img_dir figures/cosmx/17_0/mean/20210706_105344_S2_C902_P99_N99_F020_Z001/

python -m main --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/lin/mean/ --merge \
        --embed_dir out/cosmx_lin/ --vis_name cosmx_lin_mean --visualize_expression --has_expr_data --raw_subset_dir cosmx
python -m main --visualize_image --vis_name cosmx_lin_mean.h5ad --vis_img_raw_subset_dir cosmx/test --name_tiff 20210706_105344_S2_C902_P99_N99_F020_Z001.TIF --vis_all_channels \
        --vis_protein 'PTPRC,CD74,CD9,CD163,CD68,CD14,CD4,CD34,CD3D,CD3E,CD3G,CDH1,MKI67,TP53,ICOS,IDO1,HLA-A,HLA-B,HLA-C,KRT1,KRT14,KRT15,TYK2,KRT17,KRT19,JAG1,MTRNR2L1,CRP,TM4SF1,PTGDR2,C1QA,MS4A1,MS4A4A,MET,CD8A,CD8B,CD84,CD127,FN1,FOXP3,DDR1,CTLA4,CD44,CD274,CD37' \
        --figure_img_dir figures/cosmx/lin/mean/20210706_105344_S2_C902_P99_N99_F020_Z001/
