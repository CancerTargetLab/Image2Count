import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for cell prediciton model")

    # General Model Arguments
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Wether or not to run NNs deterministicly")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random computations")
    parser.add_argument("--root_dir", type=str, default="data/",
                        help="Where to find the raw/ and processed/ dirs")
    parser.add_argument("--raw_subset_dir", type=str, default="TMA1_preprocessed",
                        help="How the subdir in raw/ and processed/ is called")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Number of elements per Batch")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for which to train")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker processes to be used(loading data etc)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate of model")
    parser.add_argument("--weight_decay", type=float, default=5e-6,
                        help="Weight decay of optimizer")
    parser.add_argument("--early_stopping", type=int, default=100,
                        help="Number of epochs after which to stop model run without improvement to val loss")
    parser.add_argument("--output_name", type=str, default="out/models/image_contrast.pt",
                        help="Path/name of moel for saving")
    
    # Arguments for the GNN model
    parser.add_argument("--label_data", type=str, default="OC1_all.csv",
                        help=".csv label data in the raw dir containing count data")
    parser.add_argument("--data_use_log_graph", action="store_true", default=False,
                        help="Wether or not to log count data when calulating loss")
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="Ratio of Patients used for training in train/")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Ratio of Patients which are used for validation in train/")
    parser.add_argument("--num_cfolds", type=int, default=1,
                        help="Number of Crossvalidation folds split over patients in train/")
    parser.add_argument("--node_dropout", type=float, default=0.0,
                        help="Probability of Graph Node dropout during training")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Probability of Graph Edge dropout during training")
    parser.add_argument("--cell_pos_jitter", type=int, default=40,
                        help="Positional Jittering during training of cells in pixel dist")
    parser.add_argument("--n_knn", type=int, default=6,
                        help="Number of Nearest Neighbours to calculate for each cell in graph")
    parser.add_argument("--subgraphs_per_graph", type=int, default=0,
                        help="Number of Subgraphs per Graph to use for training. If 0, train with entire graph")
    parser.add_argument("--num_hops", type=int, default=10,
                        help="Number of hops to create subgraph neighborhoods")
    parser.add_argument("--model_type", type=str, default="Image2Count",
                        help="Type of Model to train, one of [Image2Count/LIN]. When IMAGE in name, then model is trained together with an Image Model.")
    parser.add_argument("--graph_mse_mult", type=float, default=1.0,
                        help="Multiplier for MSE Loss")
    parser.add_argument("--graph_cos_sim_mult", type=float, default=1.0,
                        help="Multiplier for Cosine Similarity Loss")
    parser.add_argument("--lin_layers", type=int, default=1,
                        help="Number of Layers in Graph")
    parser.add_argument("--gat_layers", type=int, default=1,
                        help="Number of Layers in Graph")
    parser.add_argument("--num_node_features", type=int, default=256,
                        help="Size of initial Node features")
    parser.add_argument("--num_edge_features", type=int, default=1,
                        help="Size of edge features")
    parser.add_argument("--num_embed_features", type=int, default=128,
                        help="Size to embed initial Node features to")
    parser.add_argument("--num_gat_features", type=int, default=128,
                        help="Size to embed embeded Node features for GAT layer")
    parser.add_argument("--heads", type=int, default=1,
                        help="Number of Attention Heads for the Graph NN")
    parser.add_argument("--embed_dropout", type=float, default=0.1,
                        help="Percentage of embedded feature dropout chance")
    parser.add_argument("--conv_dropout", type=float, default=0.1,
                        help="Percentage of dropout chance between layers")
    parser.add_argument("--output_graph_embed", type=str, default="out/",
                        help="Dir in which to embed Cell Expressions")
    parser.add_argument("--init_image_model", type=str, default="",
                        help="Name of pre-trained Image model to load. If not used, train from scratch. Only used when IMAGE in modeltype")
    parser.add_argument("--init_graph_model", type=str, default="",
                        help="Name of pre-trained Graph model to load. If empty, train from scratch. Only used when IMAGE in modeltype")
    parser.add_argument("--train_gnn", action="store_true", default=False,
                        help="Wether or not to train the Graph Model")
    parser.add_argument("--embed_gnn_data", action="store_true", default=False,
                        help="Wether or not to embed predicted Cell Expression of test data")
    parser.add_argument("--embed_graph_train_data", action="store_true", default=False,
                        help="Wether or not to embed predicted Cell Expression for only train data")
    
    return parser.parse_args()


def main(**args):
    if args['train_gnn']:
        from src.run.GraphTrain import train as GraphTrain
        GraphTrain(**args)
    if args['embed_gnn_data']:
        from src.run.GraphEmbed import embed as GraphEmbed
        GraphEmbed(**args)

if __name__ == '__main__':
    args = vars(parse_args())
    main(**args)

