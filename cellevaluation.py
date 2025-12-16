import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate results")

    # General arguments
    parser.add_argument("--vis_label_data", type=str, default="OC1_all.csv",
                        help="Count data of Images, linked with Patient IDs")
    parser.add_argument("--figure_dir", type=str, default="figures/",
                        help="Path to save images to")

    # Preprocess arguments
    parser.add_argument("--merge", action="store_true", default=False,
                        help="Wether or not to merge predictions of models in embed_dir")
    parser.add_argument("--embed_to_h5ad", action="store_true", default=False,
                        help="Wether or not to create an sc.AnnData obj and save it to out for predictions")
    parser.add_argument("--performance_metrics", action="store_true", default=False,
                        help="Wether or not to calculate performance metrics of stored sc.AnnData")
    parser.add_argument("--h5ad_dir", type=str, default="out/",
                        help="Path to h5ad object")
    parser.add_argument("--vis_name_pattern", type=str, default="model_[0-9]_all.h5ad",
                        help="Pattern of .5ad objects in 'h5ad_dir' which should be evaluated")
    parser.add_argument("--num_subgraphs_per_graph", type=int, default=0,
                        help="Number of subgraphs per graph to create to evaluate niches")
    parser.add_argument("--num_hops_per_subgraph", type=int, default=[], nargs='*',
                        help="Number of hops per subgraphs to create to evaluate niches")
    parser.add_argument("--sum_by_graph", action="store_true", default=False,
                        help="Wether or not 'vis_label_data' is label data or cell measurement data: ROI or Image key")
    parser.add_argument("--do_clustering_metrics", action="store_true", default=False,
                        help="Wether or not to calculate metrics which need to cluster predicted expression")
    parser.add_argument("--do_pathway_metrics", action="store_true", default=False,
                        help="Wether or not to calculate agreement between enricht CollecTRI, PROGENy and msigdb between true and actual counts base don true counts clusterin")
    parser.add_argument("--do_performance_metrics", action="store_true", default=False,
                        help="Wether or not to save metrics per markers/cell/niche")
    
    
    
    # Visualize Expression 
    parser.add_argument("--visualize_expression", action="store_true", default=False,
                        help="Wether or not to visualize predicted sc expression")
    parser.add_argument("--has_expr_data", action="store_true", default=False,
                        help="Wether or not true Single Cell expression data is in measurements.csv")
    parser.add_argument("--processed_subset_dir", type=str, default="TMA1_preprocessed",
                        help="Subset directory of processed/ and raw/ of data")
    parser.add_argument("--embed_dir", type=str, default="out/",
                        help="Path to predicted single cell data per Graph/Image")
    parser.add_argument("--vis_select_cells", type=int, default=0,
                        help="Number of cells to perform dim reduction on. If 0, then all cells get reduced")
    parser.add_argument("--vis_name", type=str, default="_cells",
                        help="Name added to figures name, saves processed data as NAME.h5ad")   #alo for visualize image

    # Visualize Image
    parser.add_argument("--visualize_image", action="store_true", default=False,
                        help="Wether or not to Visualize an Image")
    parser.add_argument("--vis_name_og", type=str, default="",
                        help="Name of NAME.h5ad containing TRUE single-cell counts(used for validation data)")
    parser.add_argument("--vis_img_raw_subset_dir", type=str, default="TMA1_preprocessed",
                        help="Name of raw/ subsetdir which contains .tiff Images to visualize")
    parser.add_argument("--name_tiff", type=str, default="027-2B27.tiff",
                        help="Name of .tiff Image to visualize")
    parser.add_argument("--figure_img_dir", type=str, default="figures/",
                        help="Path to output figures to")
    parser.add_argument("--vis_protein", type=str, default="",
                        help="Proteins to visualize Expression over Image of, seperated by ,; . converts to space")
    parser.add_argument("--vis_img_xcoords", nargs='+', type=int, default=[0,0],
                        help="Image x coords, smaller first")
    parser.add_argument("--vis_img_ycoords", nargs='+', type=int, default=[0,0],
                        help="Image y coords, smaller first")
    parser.add_argument("--vis_all_channels", action="store_true", default=False,
                        help="Wether or not to visualize all Image channels on their own")

    # Visualize Model Run
    parser.add_argument("--visualize_model_run", action="store_true", default=False,
                        help="Wether or not to Visualize statistics of model run")
    parser.add_argument("--model_path", type=str, default="out/models/ROI.pt",
                        help="Path and name of model save")
    parser.add_argument("--output_model_name", type=str, default="ROI",
                        help="Name of model in figures")
    parser.add_argument("--figure_model_dir", type=str, default="figures/",
                        help="Path to output figures to")
    parser.add_argument("--is_cs", action="store_true", default=False,
                        help="Wether or not Cosine Similarity is used or Contrast Loss")

    return parser.parse_args()


def main(**args):
    if args['merge']:
            from src.utils.utils import merge
            merge(args['embed_dir'])
            args['embed_dir'] = os.path.join(args['embed_dir'], 'mean')
    if args['embed_to_h5ad']:
        from src.utils.create_h5ad import createH5AD
        createH5AD(path=args['h5ad_dir'],
                   processed_dir=args['processed_subset_dir'],
                   embed_dir=args['embed_dir'],
                   label_data=args['vis_label_data'],
                   name=args['vis_name'])
    if args['performance_metrics']:
        from src.explain.Metrics import Metrics
        Metrics(path=args['h5ad_dir'],
                target=args['vis_label_data'],
                name=args['vis_name_pattern'],
                num_subgraphs_per_graph=args['num_subgraphs_per_graph'],
                num_hops_per_subgraph=args['num_hops_per_subgraph'],
                out=args['figure_dir'],
                sum_by_graph=args['sum_by_graph'],
                do_clustering_metrics=args['do_clustering_metrics'],
                do_pathway_metrics=args['do_pathway_metrics'],
                do_performance_metrics=args['do_performance_metrics'])
    if args['visualize_expression']:
        from src.explain.VisualizeExpression import visualizeExpression
        visualizeExpression(processed_dir=args['processed_subset_dir'],
                            embed_dir=args['embed_dir'],
                            label_data=args['vis_label_data'],
                            figure_dir=args['figure_dir'],
                            name=args['vis_name'],
                            select_cells=args['vis_select_cells'])
        if args['has_expr_data']:
            from src.explain.correlation import correlation
            correlation(raw_subset_dir=args['raw_subset_dir'],
                        figure_dir=args['figure_dir'],
                        vis_name=args['vis_name'])
    if args['visualize_image']:
        from src.explain.VisualizeImage import visualizeImage
        visualizeImage(raw_subset_dir=args['vis_img_raw_subset_dir'],
                       name_tiff=args['name_tiff'],
                       figure_dir=args['figure_img_dir'],
                       vis_name=args['vis_name'],
                       args=args)
    if args['visualize_model_run']:
        from src.utils.per_epoch_metrics import epochMetrics
        epochMetrics(model_path=args['model_path'],
                     figure_dir=args['figure_model_dir'],
                     is_cs=args['is_cs'],
                     name=args['output_model_name'])


if __name__ == '__main__':
    args = vars(parse_args())
    main(**args)
