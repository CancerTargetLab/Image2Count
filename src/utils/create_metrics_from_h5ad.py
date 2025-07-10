import scanpy as sc
import os
import numpy as np
import pandas as pd
import torch
import torch_geometric
import squidpy as sq
from src.utils.utils import per_gene_mi, per_gene_corr

target = 'data/raw/CRC_1p/CRC_1p_measurements.csv'
num_subgraphs_per_graph = 900
num_hops_per_subgraph = [1, 2, 3, 5, 8, 11]

name = 'crc_1p_24_6_mean_all.h5ad'

target_df = pd.read_csv(target,
                 header=0,
                 sep=',')
target_df['Image'] = target_df['Image'].apply(lambda x: x.split('.')[0])

similarity = torch.nn.CosineSimilarity()
mse = torch.nn.MSELoss(reduction='none', )

def create_metrics(path, target_df):
    entries = os.listdir(path)
    for entrie in entries:
        if entrie.endswith('.h5ad') and name in entrie and os.path.isfile(os.path.join(path, entrie)):
            adata = sc.read_h5ad(os.path.join(path, entrie))
            var_names = adata.var_names.values
            tmp = pd.DataFrame()
            tmp['files'] = adata.obs['files'].values
            tmp[var_names] = adata.X
            adata = tmp
            adata['files'] = adata['files'].apply(lambda x: x.split('graph_')[-1].split('.')[0])
            adata = adata[adata['files'].isin(target_df['Image'])]     #Selects files that exist in pred, in case of only investiating test data
            target_df = target_df[target_df['Image'].isin(adata['files'])]
            adata = adata.sort_values(by='files', kind='stable', ignore_index=True)
            target_df = target_df.sort_values(by='Image', kind='stable', ignore_index=True)

            x = adata[adata.columns[1:].values].values
            y = target_df[adata.columns[1:].values].values
            for hops in num_hops_per_subgraph:
                subset_x, subset_y = create_subgraphs(x, target_df, num_subgraphs_per_graph, hops, adata.columns[1:].values)
                metrics(subset_x, subset_y, name+f'_{hops}')
            metrics(x, y, name)
        elif os.path.isdir(os.path.join(path, entrie)):
            create_metrics(os.path.join(path, entrie), target_df)

def create_subgraphs(pred, target_df, num_subgraphs_per_graph, hops, columns):
    subgraphs_x = np.empty((target_df['Image'].unique().shape[0]*num_subgraphs_per_graph, pred.shape[1]),
                           dtype=pred.dtype)
    subgraphs_y = np.empty((target_df['Image'].unique().shape[0]*num_subgraphs_per_graph, pred.shape[1]),
                           dtype=pred.dtype)
    for g, subset_name in enumerate(target_df['Image'].unique().tolist()):
        subset = target_df.loc[target_df['Image']==subset_name]
        subset_x = pred[target_df['Image']==subset_name]
        subset_y = subset[columns].values
        xmax, xmin, ymax, ymin = subset['Centroid.X.px'].max(), subset['Centroid.X.px'].min(), subset['Centroid.Y.px'].max(), subset['Centroid.Y.px'].min()
        # Calculate the step sizes for x and y dimensions
        step_x = (xmax - xmin) / (num_subgraphs_per_graph ** 0.5 + 1)
        step_y = (ymax - ymin) / (num_subgraphs_per_graph ** 0.5 + 1)

        # Generate points
        points = []
        for i in range(int(num_subgraphs_per_graph ** 0.5)):
            for j in range(int(num_subgraphs_per_graph ** 0.5)):
                x = xmin + i * step_x + step_x / 2
                y = ymin + j * step_y + step_y / 2
                points.append((x, y))

        # Not the most straight forward way, but this is how i implemented it in GeoMxData :)
        # So just to be sure, lets just do it like that
        counts = np.zeros((subset.shape[0], 1))
        coordinates = np.column_stack((subset["Centroid.X.px"].to_numpy(), subset["Centroid.Y.px"].to_numpy()))
        adata = sc.AnnData(counts, obsm={"spatial": coordinates})
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
        edge_matrix = adata.obsp["spatial_distances"]
        edge_index, _ = torch_geometric.utils.convert.from_scipy_sparse_matrix(edge_matrix)
        
        for p, point in enumerate(points):
            idx = np.argmin(np.abs(subset['Centroid.X.px'].values-point[0]) + np.abs(subset['Centroid.Y.px']-point[1]))
            subgraph_idxs, _, _, _ = torch_geometric.utils.k_hop_subgraph(int(idx),
                                                                        hops,
                                                                        edge_index,
                                                                        relabel_nodes=True, 
                                                                        directed=False)
            subgraphs_x[p+g*num_subgraphs_per_graph] = np.sum(subset_x[subgraph_idxs.numpy()], axis=0)
            subgraphs_y[p+g*num_subgraphs_per_graph] = np.sum(subset_y[subgraph_idxs.numpy()], axis=0)
    
    return subgraphs_x, subgraphs_y

def metrics(x, y, name):
    sim = similarity(torch.from_numpy(x), torch.from_numpy(y)).numpy()
    mi = per_gene_mi(x, y)
    dist = mse(torch.log(torch.from_numpy(x)+1), torch.log(torch.from_numpy(y)+1)).numpy()
    is_g_zero = np.sum(y, axis=-1) > 0
    x = x[is_g_zero]
    y = y[is_g_zero]
    p_stat, _ = per_gene_corr(x, y, mean=False, method='pearsonr')    
    s_stat, _ = per_gene_corr(x, y, mean=False, method='spearmanr')
    k_stat, _ = per_gene_corr(x, y, mean=False, method='kendalltau')

    mean_data = {
        'Metric': ['Pearson', 'Spearman', 'Kendall', 'MI', 'CosSim', 'MSE'],
        'Mean': [p_stat.mean(), s_stat.mean(), k_stat.mean(), mi.mean(), sim.mean(), dist.mean()],
        'Std': [p_stat.std(), s_stat.std(), k_stat.std(), mi.std(), sim.std(), dist.std()],
    }
    df = pd.DataFrame(mean_data)
    df.to_csv(os.path.join('figures/tmp', 'avg_metrics_'+name+'.csv'))


create_metrics('out/', target_df)