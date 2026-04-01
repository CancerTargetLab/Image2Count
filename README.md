# Image2Count: Predicting Single Cell Expression from Multiplex-Immunofloresent Imaging and Bulk-Count Data  

## Environment setup

Our models are trained with nvidia GPUs. To run on GPUs appropiate CUDA versions must be installed. Installing via conda can result in cuda version mismatches or in the installation of CPU versions. Appropiate versions can be found [here](https://data.pyg.org/whl/).  
Setup via Anaconda environment:
```
conda install -c anaconda python=3.11.5  
pip install torch -f https://data.pyg.org/whl/torch-2.4.0+cu121.html  
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cu121.html  
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html  
pip install torchvision -f https://data.pyg.org/whl/torch-2.4.0+cu121.html  
pip install squidpy  
pip install decoupler  
#pip install imctools  
```

## Required Data
The following data is required to start training:  
1. .tiff images  
These are images of tissue to investigate. We recommend to save them in a seperate directory, and to link them to a working dir via `ln -s /path/to/images/*.tiff data/raw/{EXPERIMENT NAME}/[train/test]`. `[train/test]` are folders in which to split the data, data in `train/` is used for training and validation, data in `test/` is used for testing.
2. {measurements}.csv  
This CSV should be in the same dir as the linked images: `data/raw/{EXPERIMENT NAME}/`. The csv contains Cellpostion information of all images. It consists out of the following columns with these exact names: The first column is named `Image` and contains the name of the images, e.g. `CRC03.ome.tif`. The second and third column should be named `Centroid.X.px` and `Centroid.Y.px`, which contain Cell positions in pixel(!) values. We expect coordinates to start at `(0,0)` at the upper left corner of each corresponding image. The 4th column should be named `Class` and contains Celltype information, if existing. This can also be empty. Every column after this is optional, and represents count data of the cells. This can be used to create smaller areas of the image to train on. This is only useful when already spatialomics count data for cells exists(e.g. to investigate model performace). This can be normalized counts, but do not use log normalized counts.
3. {label}.csv  
This .csv should be located in `data/raw/`. The csv contains information of the overall count data for an image. It consists out of the following columns: The first column is named `ROI` and contains the name of the image, minus everything after the first `.`(`CRC02.ome.tif`->`CRC02`). The second column is named `Patient_ID` and contains some ID to which the Image corresponds. Every column after this is interpreted as Count data. Each entry has the whole count data of the whole Image. This can be normalized counts, but o not use log normalized counts.  

Now we are ready to go!

## Preprocessing
The first process is to normalize Image data and uniformly cut out Cells:  
```sh
python -m cellcontrast --image_preprocess --preprocess_dir 'data/raw/{EXPERIMENT NAME}/[train/test]/' --cell_cutout 34 --preprocess_workers 26 --preprocess_channels 0,10,14,19 --calc_mean_std
```
- `--image_preprocess`:  
Wether or not to run image preprocessing.
- `--preprocess_dir`:  
Directory path which contains .tiff images with `{measurements}.csv` in direct parent directory.
- `--cell_cutout`:  
Pixel Height und Width to cutout around Cell center postions.
- `--preprocess_workers`:  
Number of Threads to use to cutout Cells. Default 1, Optional.
- `--preprocess_channels`: 
Channels to preprocess. If not specified use all Channels. Seperate Channel indicies, starting from 0, with commas.
- `--calc_mean_stdr`:  
Wether or not calculate mean and std per channel and save `mean.npy` and `std.npy` in direct parent directory of `--preprocess_dir`.  

Preprocessing produces files for each Image: `{IMAGE NAME}_cells.npy` in the shape of `(Number of Cells, Channels, --cell_cutout, --cell_cutout)`.

## General arguments

- `--deterministic`:  
Wether or not to run NNs deterministicly.
- `--seed`:  
Seed for random computations.
- `--root_dir`:  
Where to find the raw/ and processed/ dirs. Default is `data/`.
- `--raw_subset_dir`:  
How the subdir containing experiment data in raw/ and processed/ is called.
- `--batch_size`:  
Number of elements per Batch.
- `--epochs`:  
Number of epochs for which to train.
- `--num_workers`:  
Number of worker processes to be used(loading data etc).
- `--lr`:  
Learning rate of model.
- `--weight_decay`:  
Weight decay of optimizer.
- `--early_stopping`:  
Number of epochs after which to stop model run without improvement to val loss.
- `--output_name`:  
Path/name of moel for saving.

## Cell Contrastive Learning

Next, we learn Visual Representations of Cells via Contrastive learning:  
```sh
python -m cellcontrast --train_image_model --embed_image_data --output_name 'out/models/image_contrast_model.pt' --raw_subset_dir '{EXPERIMENT NAME}' --weight_decay 1e-6 --resnet_model '18' --batch_size 4096 --epochs 100 --warmup_epochs 10 --num_workers 25 --lr 0.1 --embed 32 --contrast 16 --crop_factor 0.2 --n_clusters_image 20
```

- `--train_image_model`:  
Wether or not to train the Image model.
- `--embed_image_data`:  
Wether or not to embed data with a given Image model.
- `--resnet_model`:  
What ResNet model to choose, on of '18', '34', '50' and '101', default `'18'`.
- `--embed`:  
Linear net size used to embed data.
- `--contrast`:  
Linear net size on which to calculate the contrast loss.
- `--crop_factor`:  
Cell Image crop factor for Image augmentation.
- `--n_clusters_image`:  
Number of KMeans clusters to be calculated on the ceontroid pixel(area 0.15 of height/width when image height > 50 px) of each cell cut-out. Used to oversample low abundance clusters during training. Only used when > 1.

We extract visual representations after training a model to learn visual representations. Embedding produces files for each `{IMAGE NAME}_cells.npy` in the same directory: `{IMAGE NAME}_cells_embed.pt` in the shape of `(Number of Cells, --embed`.

## Learning to predict SC Expression

Next, we learn what each Cell contributes to the Count Data of an Image:  
```sh
python -m cellprediction --train_gnn --embed_gnn_data --output_name 'out/models/image_graph_model.pt' --output_graph_embed '/out/graph_model/' --init_image_model 'out/models/image_contrast_model.pt' --init_graph_model 'out/models/graph_model.pt' --root_dir 'data/' --raw_subset_dir '{EXPERIMENT NAME}' --label_data '{label}.csv'  --batch_size 64 --epochsh 1000 --num_workers 12 --lr 0.005 --early_stopping 50 --weight_decay 1e-4 --train_ratio 0.6 --val_ratio 0.2 --node_dropout 0.0 --edge_dropout 0.3 --cell_pos_jitter 40 --cell_n_knn 6 --subgraphs_per_graph 0 --num_hops_subgraph 0 --model_type 'Image2Count' --data_use_log_graph --graph_mse_mult 1 --graph_cos_sim_mult 1  --lin_layers 3 --gat_layers 3 --num_node_features 32 --num_edge_features 1 --num_embed_features 128 --heads 4 --embed_dropout 0.1 --conv_dropout 0.1 --num_cfolds 0
```

- `--train_gnn`:  
Wether or not to train the Graph Model.
- `--embed_gnn_data`:  
Wether or not to embed predicted Cell Expression of test data.
- `--embed_graph_train_data`:  
Wether or not to embed predicted Cell Expression for only train data.
- `--output_graph_embed`:  
Dir in which to embed Cell Expressions.
- `--init_image_model`:  
Name of pre-trained Image model to load. If not used, train from scratch. Only used when `IMAGE` in modeltype.
- `--init_graph_model`:  
Name of pre-trained Graph model to load. If not used, train from scratch. Only used when `IMAGE` in modeltype.
- `--label_data`:  
`{label}.csv` label data in the raw dir containing count data.
- `--train_ratio`:  
Ratio of Patients used for training in `train/` folder.
- `--val_ratio`:  
Ratio of Patients which are used for validation in `train/` folder.
- `--num_cfolds`:  
Number of Crossvalidation folds in `train/` folder split over patients. Only used when greater > 1. `--output_name` of model gets split at `.` and becomes a folder in which one model per split, named after split number `[n].pt`, is saved.
- `--node_dropout`:  
Probability of Graph Node dropout during training.
- `--edge_dropout`:  
Probability of Graph Edge dropout during training.
- `--cell_pos_jitter`:  
Positional Jittering during training of cells in pixel dist.
- `-cell_n_knn`:  
Number of Nearest Neighbours to calculate for each cell in graph and the minimum amount of cells per graph/image necessary to use the model.
- `--subgraphs_per_graph`:  
Number of Subgraphs per Graph to use for training, should be a quadratic integer. If 0, train with entire graph. Can only be non 0 if `{measurements}.csv` contains Cell count data.
- `--num_hops_subgraph`:  
Number of hops to create subgraph neighborhoods.
- `--model_type`:  
Type of Model to train, one of [Image2Count/LIN]. When IMAGE in name, then model is trained together with an Image Model.
- `--data_use_log_graph`:  
Wether or not to log count data when calulating loss.
- `--graph_mse_mult`:  
Multiplier for L1 Loss.
- `--graph_cos_sim_mult`:  
Multiplier for Cosine Similarity Loss.
- `--lin_layers`:  
Number of lin block Layers in model.
- `--gat_layers`:  
Number of gat block Layers in model.
- `-num_node_features`:  
Size of initial Node features.
- `--num_edge_features`:  
Size of edge features.
- `--num_embed_features`:  
Size to embed initial Node features to.
- `--heads`:  
Number of Attention Heads for the Graph NN.
- `--embed_dropout`:  
Percentage of embedded feature dropout chance.
- `--conv_dropout`:  
Percentage of dropout chance between layers.

After training a model to predict the Expression of Single Cells, the predicted Expression of all Single Cells in the specified directory get embeded in the specified output graph embed directory. Each graph gets embedded seperatly in shape `(Number of Cells, Number of Genes/Transcripts/Proteins/...)`. It is important to note that models trained on subgraphs will embed subgraphs, if the embedding is not done in a seperate call where `--subgraphs_per_graph` is set to 0. Subgraphs are stored in `processed/{EXPERIMENT NAME}/subgraphs/` if created.

## Create and save an sc.AnnData object of model prediction

```sh
python -m cellevaluation --merge --embed_dir out/graph_model/ --h5ad_dir 'out/' --vis_label_data {label}.csv --processed_subset_dir dataset/test --vis_name graph_model --embed_to_h5ad
```  

- `--merge`:  
Merge crossfold model predictions by calculating the mean. 
- `embed_dir`:  
Directory in which data is embedded, if --merge is called this should be the parent directory containing all to be merged prediction folders, resulting in os.path.join(args['embed_dir'], 'mean') containing merged predictions.
- `--h5ad_dir`:  
Directory in which to save the sc.AnnData obeject.
- `--vis_label_data`:  
Label csv.
- `--processed_subset_dir`:  
Processed subset dir in which test data is.
- `--vis_name`:  
Name of the sc.AnnData model that is saved. `_all.h5ad` is appended when saving.
- `--embed_to_h5ad`:  
Wether or not to embed data to h5ad.

## Create Metrics of model performance

```sh
python -m cellevaluation --vis_label_data dataset/measurements.csv --h5ad_dir 'out/' --vis_name_pattern 'cosmx_lin_[0-5]_all.h5ad' --num_subgraphs_per_graph 36 --num_hops_per_subgraph 1 2 3 5 8 11 --do_clusterin_metrics --do_performance_metrics --do_pathway_metrics --performance_metrics --figure_dir figures/cosmx/lin/metrics/
```  

- `--vis_label_data`:  
Label or measurement csv, depending on if cellular ground truth is available.
- `--h5ad_dir`:  
Directory in which to save the sc.AnnData obeject.
- `--vis_name_pattern`:  
Pattern which describes `.h5ad` files which to consider. Iterates over all if multiple and creates averages of performance.
- `--num_subgraphs_per_graph`:  
Number of subgraphs per graph to create if ground truth available and niche performance is to be analyzed.
- `--num_hops_per_subgraph`:  
Number of hops to create niches for evaluation. Can be multiple, which are saved accordingly.
- `--do_clusterin_metrics`:  
Wether or not to calculate clustering metrics ARI, NMI.
- `--do_performance_metrics`:  
Wether or not to save performance metrics per marker/cell/niche.
- `--do_pathway_metrics`:  
Wether or not to do enrichment analysis for multiple pathway or tf databases. Only done if `--do_clustering_metrics`.
- `--performance_metrics`:  
Wether or not to calculate metrics.
- `--figure_dir`:  
Directory in which performance metrics are saved.
- `--sum_by_graph`:  
Wether or not 'vis_label_data' is label data or cell measurement data: ROI or Image is ID key for selection. Only used when no ground truth data for sc expression available -> cell expression predictions are summed per graph

## Visualizing Model runs

Model runs can be visualized as follows:  
```sh
python -m cellevaluation --visualize_model_run --model_path 'out/models/graph_model.pt' --output_model_name 'Image Contrast Model' --figure_model_dir 'figures/graph_model/' --is_cs
```

- `--visualize_model_run`:  
Wether or not to Visualize statistics of model run.
- `--model_path`:  
Path and name of model save(works on folder of model as well if trained for same num of epochs).
- `--output_model_name`:  
Name of model in figures.
- `--figure_model_dir`:  
Path to output figures to.
- `--is_cs`:  
Wether or not Cosine Similarity is used or Contrast Loss.

## Visualizing Expression Data

The predicted single cell Expression can be visualized:
```sh
python -m cellevaluation --visualize_expression --vis_label_data '{label}.csv' --processed_subset_dir '{EXPERIMENT NAME}/test' --figure_dir 'figures/graph_model/' --embed_dir 'out/graph_model/' --vis_select_cells 50000 --vis_name '_graph_model' --has_expr_data --merge
```

- `--visualize_expression`:  
Wether or not to visualize predicted sc expression.
- `--vis_label_data`:  
Count data of Images, linked with Patient IDs.
- `--processed_subset_dir`:  
Subset `train/test{/subgraphs}` directory of processed/ and raw/ of data.
- `--figure_dir`:  
Path to save images to.
- `--embed_dir`:  
Path to predicted single cell data per Graph/Image, if --merge is called this should be the parent directory containing all to be merged prediction folders, resulting in os.path.join(args['embed_dir'], 'mean') containing merged predictions.
- `--vis_select_cells`:  
Number of cells to perform dim reduction on. If 0, then all cells get reduced.
- `--vis_name`:  
Name added to figures name, saves processed data as `{NAME}.h5ad` in `out/`. Additional unproccesed save with all cells is named `{NAME}_all.h5ad`.
- `--has_expr_data`:  
Wether or not true Single Cell expression data is in `measurements.csv`, calculates per cell correlation.
- `--raw_subset_dir`:  
How the subdir containing experiment data in raw/ and processed/ is called, used when --has_expr_data is True.
- `--merge`:  
Wether or not to merge predictions of multiple models in seperate dirs in embed_dir. If specified, visualizes expression of merged models(mean merged).

## Visualizing Spatialomics

The predicted single cell Expression can be also visualized on the Images themselfs:  
```sh
python -m cellevaluation --visualize_image --vis_img_raw_subset_dir '{EXPERIMENT NAME}' --name_tiff 'CRC02.ome.tif' --figure_img_dir 'figures/graph_model/' --vis_protein 'CD45,CD8,Keratin,Ki67,Fibronectin,Some.Name' --vis_img_xcoords 0 0 --vis_img_ycoords 0 0 --vis_all_channels --vis_name '_graph_model.h5ad' --vis_name_og 'original_data.h5ad'
```

- `--visualize_image`:  
Wether or not to Visualize an Image.
- `--vis_img_raw_subset_dir`:  
Name of raw/ subsetdir which contains .tiff Images to visualize.
- `--name_tiff`:  
Name of .tiff Image to visualize.
- `--figure_img_dir`:  
Path to output figures to.
- `--vis_protein`:  
Proteins to visualize Expression over Image of, seperated by `,`; `.` converts to space.
- `--vis_img_xcoords`:  
Image x coords, smaller first.
- `--vis_img_ycoords`:  
Image y coords, smaller first.
- `--vis_all_channels`:  
Wether or not to visualize all Image channels on their own.
- `--vis_name`:  
Name of `out/{NAME}` produced via visualizing expression, needs to be given.
- `--vis_name_og`:  
Name of `out/{NAME}` of original single-cell expression data, manualy created for a dataset if given data present. Contrasts predicted expression with observed expression.

## Reproduce

Tutorials to repreduce our results can be found in the `reproduce/` directory.