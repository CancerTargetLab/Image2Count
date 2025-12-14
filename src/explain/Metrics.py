import scanpy as sc
import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch_geometric
import squidpy as sq
from src.utils.utils import per_gene_mi, per_gene_corr, per_area_ssim, per_area_ari, per_area_nmi, per_area_js_div, per_cluster_pathways, per_cluster_key_coverage, cluster_cell_expression

def _create_metrics(path,
                   target_df,
                   name,
                   target_column_file_name,
                   sum_by_graph,
                   num_subgraphs_per_graph,
                   num_hops_per_subgraph,
                   do_performance_metrics,
                   do_clustering_metrics,
                   do_pathway_metrics,
                   performance_metrics,
                   y_clusters,
                   y_cluster_enrichment,
                   out):
    entries = os.listdir(path)
    pattern = re.compile(name)
    num_models = 0
    for entrie in entries:
        if entrie.endswith('.h5ad') and pattern.match(entrie) and os.path.isfile(os.path.join(path, entrie)):
            print(entrie)
            adata = sc.read_h5ad(os.path.join(path, entrie))
            var_names = adata.var_names.values
            tmp = pd.DataFrame(data=adata.X, columns=var_names)
            tmp['files'] = adata.obs['files'].values
            adata = tmp
            adata['files'] = adata['files'].apply(lambda x: x.split('graph_')[-1].split('.')[0])
            adata = adata[adata['files'].isin(target_df[target_column_file_name])]     #Selects files that exist in pred, in case of only investiating test data
            target_df = target_df[target_df[target_column_file_name].isin(adata['files'])]
            adata = adata.sort_values(by='files', kind='stable', ignore_index=True)
            target_df = target_df.sort_values(by=target_column_file_name, kind='stable', ignore_index=True)

            if sum_by_graph:
                subgraphs_x = np.empty((target_df[target_column_file_name].unique().shape[0], len(var_names)),
                           dtype=adata[var_names[0]].values.dtype)
                files = adata['files'].unique().tolist()    # return in order of appearance, is already sorted
                for i, file in enumerate(files):
                    subgraphs_x[i] = np.sum(adata.loc[adata['files']==file][var_names].values, axis=0)
                adata = pd.DataFrame(data=subgraphs_x, columns=var_names)

            x = adata[var_names].values
            y = target_df[var_names].values
            for hops in num_hops_per_subgraph:
                subset_x, subset_y, mean_edge_l = _create_subgraphs(x, target_df, num_subgraphs_per_graph, hops, var_names)
                if do_performance_metrics:
                    _prepare_performance_metrics(var_names, subset_y, f'{hops}', mean_edge_l, performance_metrics)
                _metrics(x=subset_x,
                        y=subset_y,
                        name=entrie+f'_{hops}',
                        cluster_key=f'{hops}',
                        var_names=var_names,
                        do_performance_metrics=do_performance_metrics,
                        do_clustering_metrics=do_clustering_metrics,
                        do_pathway_metrics=do_pathway_metrics,
                        performance_metrics=performance_metrics,
                        y_clusters=y_clusters,
                        y_cluster_enrichment=y_cluster_enrichment,
                        out=out)
            if do_performance_metrics:
                _, _, mean_edge_l = _create_subgraphs(x, target_df, 0, -1, var_names)
            is_g_zero = np.sum(y, axis=-1) > 0
            x = x[is_g_zero]
            y = y[is_g_zero]
            mean_edge_l = mean_edge_l[is_g_zero]
            _prepare_performance_metrics(var_names, y, 'sc', mean_edge_l)
            _metrics(x=x,
                    y=y,
                    name=entrie,
                    cluster_key='sc',
                    var_names=var_names,
                    do_performance_metrics=do_performance_metrics,
                    do_clustering_metrics=do_clustering_metrics,
                    do_pathway_metrics=do_pathway_metrics,
                    performance_metrics=performance_metrics,
                    y_clusters=y_clusters,
                    y_cluster_enrichment=y_cluster_enrichment,
                    out=out)
    if num_models > 1:
        _calc_subgraph_mean(out=out,
                           subgraphs=num_hops_per_subgraph,)

def _create_subgraphs(pred, target_df, num_subgraphs_per_graph, hops, columns):
    if num_subgraphs_per_graph > 0:
        subgraphs_x = np.empty((target_df['Image'].unique().shape[0]*num_subgraphs_per_graph, pred.shape[1]),
                            dtype=pred.dtype)
        subgraphs_y = np.empty((target_df['Image'].unique().shape[0]*num_subgraphs_per_graph, pred.shape[1]),
                            dtype=pred.dtype)
        subgraphs_x_m_edge_l = np.empty((target_df['Image'].unique().shape[0]*num_subgraphs_per_graph,))
    else:
        subgraphs_x_m_edge_l = np.empty((target_df.shape[0]))
        subgraphs_x = 0
        subgraphs_y = 0
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
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)    #TODO: wastefull, only do once ?
        edge_matrix = adata.obsp["spatial_distances"]
        edge_index, edge_attr = torch_geometric.utils.convert.from_scipy_sparse_matrix(edge_matrix)
        data = torch_geometric.data.Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=subset.shape[0])
        data = torch_geometric.transforms.ToUndirected(merge=False)(data)
        
        if num_subgraphs_per_graph > 0:
            idx_list = []
            rng = np.random.default_rng(0)
            for p, point in enumerate(points):
                idx = np.argmin(np.abs(subset['Centroid.X.px'].values-point[0]) + np.abs(subset['Centroid.Y.px']-point[1]))
                if idx in idx_list:
                    idx = rng.integers(0, subset.shape[0])
                idx_list.append(idx)
                subgraph_idxs, _, _, edge_mask = torch_geometric.utils.k_hop_subgraph(int(idx),
                                                                            hops,
                                                                            data.edge_index,
                                                                            relabel_nodes=True, 
                                                                            directed=False)
                subgraphs_x[p+g*num_subgraphs_per_graph] = np.sum(subset_x[subgraph_idxs.numpy()], axis=0)
                subgraphs_y[p+g*num_subgraphs_per_graph] = np.sum(subset_y[subgraph_idxs.numpy()], axis=0)
                subgraphs_x_m_edge_l[p+g*num_subgraphs_per_graph] = data.edge_attr[edge_mask].mean().numpy()
        else:
            sum_idx = torch.zeros(data.num_nodes, dtype=data.edge_attr.dtype)
            sum_idx.index_add_(dim=0, index=data.edge_index[0], source=data.edge_attr)
            degree = torch_geometric.utils.degree(data.edge_index[0], data.num_nodes)
            subgraphs_x_m_edge_l[target_df['Image']==subset_name] = sum_idx/degree
    
    return subgraphs_x, subgraphs_y, subgraphs_x_m_edge_l

def _prepare_performance_metrics(var_names,
                                y,
                                hops,
                                mean_edge_l,
                                performance_metrics):
    if 'markers' not in performance_metrics['SNR'].columns:
        performance_metrics['SNR']['markers'] = var_names
        performance_metrics['abundance']['markers'] = var_names
    performance_metrics['SNR'][f'SNR{hops}'] = y.mean(axis=0)/(y.std(axis=0)+1e-12)
    performance_metrics['abundance'][f'abundance_{hops}'] = np.mean(np.log1p(y), axis=0)
    if hops == 'sc':
        performance_metrics['mean_edge_l_sc'][f'mean_edge_l_{hops}'] = mean_edge_l
    else:
        performance_metrics['mean_edge_l'][f'mean_edge_l_{hops}'] = mean_edge_l

def _metrics(x,
            y,
            name,
            cluster_key,
            var_names,
            do_performance_metrics,
            do_clustering_metrics,
            do_pathway_metrics,
            performance_metrics,
            y_clusters,
            y_cluster_enrichment,
            out
            ):
    similarity = torch.nn.CosineSimilarity()
    mse = torch.nn.MSELoss(reduction='none')
    sim = similarity(torch.from_numpy(x), torch.from_numpy(y)).numpy()
    ssim = per_area_ssim(x, y)
    js_div = per_area_js_div(x, y)
    mi = per_gene_mi(x, y)
    dist = mse(torch.log(torch.from_numpy(x)+1), torch.log(torch.from_numpy(y)+1)).numpy()
    p_stat, _ = per_gene_corr(x, y, mean=False, method='pearsonr')    
    s_stat, _ = per_gene_corr(x, y, mean=False, method='spearmanr')
    k_stat, _ = per_gene_corr(x, y, mean=False, method='kendalltau')

    if do_performance_metrics:
        if not cluster_key == 'sc':
            performance_metrics['mean_edge_l'][f'sim_{cluster_key}_{name}'] = sim
            performance_metrics['mean_edge_l'][f'js_div_{cluster_key}_{name}'] = js_div
        else:
            performance_metrics['mean_edge_l_sc'][f'sim_{cluster_key}_{name}'] = sim
            performance_metrics['mean_edge_l_sc'][f'js_div_{cluster_key}_{name}'] = js_div
        performance_metrics['SNR'][f'mi_{cluster_key}_{name}'] = mi
        performance_metrics['SNR'][f'dist_{cluster_key}_{name}'] = dist.mean(axis=0)
        performance_metrics['SNR'][f'js_div_{cluster_key}_{name}'] = js_div
        performance_metrics['SNR'][f'pcc_{cluster_key}_{name}'] = p_stat
        performance_metrics['SNR'][f'scc_{cluster_key}_{name}'] = s_stat
        performance_metrics['SNR'][f'kcc_{cluster_key}_{name}'] = k_stat
        performance_metrics['abundance'][f'mi_{cluster_key}_{name}'] = mi
        performance_metrics['abundance'][f'dist_{cluster_key}_{name}'] = dist.mean(axis=0)
        performance_metrics['abundance'][f'js_div_{cluster_key}_{name}'] = js_div
        performance_metrics['abundance'][f'pcc_{cluster_key}_{name}'] = p_stat
        performance_metrics['abundance'][f'scc_{cluster_key}_{name}'] = s_stat
        performance_metrics['abundance'][f'kcc_{cluster_key}_{name}'] = k_stat

    mean_data = {
        'Metric': ['Pearson',  'Spearman', 'Kendall', 'MI', 'CosSim', 'MSE', 'JensenShannonDiv', 'SSIM'],
        'Mean': [p_stat.mean(), s_stat.mean(), k_stat.mean(), mi.mean(), sim.mean(), dist.mean(), js_div.mean(), ssim],
        'Std': [p_stat.std(), s_stat.std(), k_stat.std(), mi.std(), sim.std(), dist.std(), js_div.std(), 0],    # ssim returns singular value
    }
    if do_clustering_metrics:
        x_cluster = cluster_cell_expression(x)
        if cluster_key not in y_clusters.keys():
            y_cluster = cluster_cell_expression(y.copy())
            y_clusters[cluster_key] = y_cluster
        else:
            y_cluster = y_clusters[cluster_key]
        if do_pathway_metrics:
            if cluster_key not in y_cluster_enrichment.keys():
                y_enrichment = per_cluster_pathways(y.copy(), var_names, clusters=y_cluster, top_k=5)
                y_cluster_enrichment[cluster_key] = y_enrichment
            else:
                y_enrichment = y_cluster_enrichment[cluster_key]
            x_enrichment = per_cluster_pathways(x.copy(), var_names, clusters=y_cluster, top_k=5)
            y_cluster_enrichment[f'{cluster_key}_{name}'] = x_enrichment
            tf_coverage = per_cluster_key_coverage(x_enrichment['CollecTRI'], y_enrichment['CollecTRI'])
            pw_coverage = per_cluster_key_coverage(x_enrichment['PROGENy'], y_enrichment['PROGENy'])
            hm_coverage = per_cluster_key_coverage(x_enrichment['hallmark_msigdb'], y_enrichment['hallmark_msigdb'])
            ro_coverage = per_cluster_key_coverage(x_enrichment['reactome_msigdb'], y_enrichment['reactome_msigdb'])
            kegg_coverage = per_cluster_key_coverage(x_enrichment['kegg_msigdb'], y_enrichment['kegg_msigdb'])
            mean_data['Metric'].extend(['CollecTRI_cov', 'PROGENy_cov', 'hallmark_msigdb_cov', 'reactome_msigdb_cov', 'kegg_msigdb'])
            mean_data['Mean'].extend([ari, nmi, tf_coverage, pw_coverage, hm_coverage, ro_coverage, kegg_coverage])
            mean_data['Std'].extend([0, 0, 0, 0, 0]) # We ignore std values for these metrics as we do not calculate multiple
        ari = per_area_ari(x_cluster, y_cluster)
        nmi = per_area_nmi(x_cluster, y_cluster)
        mean_data['Metric'].extend(['ARI', 'NMI'])
        mean_data['Mean'].extend([ari, nmi])
        mean_data['Std'].extend([0, 0, ]) # We ignore std values for these metrics as we do not calculate multiple
        if do_performance_metrics:
            if not cluster_key == 'sc':
                performance_metrics['mean_edge_l'][f'ari_{cluster_key}_{name}'] = ari
                performance_metrics['mean_edge_l'][f'nmi_{cluster_key}_{name}'] = nmi
                performance_metrics['mean_edge_l'][f'true_cluster_{cluster_key}_{name}'] = y_cluster
                performance_metrics['mean_edge_l'][f'pred_cluster_{cluster_key}_{name}'] = x_cluster
            else:
                performance_metrics['mean_edge_l_sc'][f'ari_{cluster_key}_{name}'] = ari
                performance_metrics['mean_edge_l_sc'][f'nmi_{cluster_key}_{name}'] = nmi
                performance_metrics['mean_edge_l_sc'][f'true_cluster_{cluster_key}_{name}'] = y_cluster
                performance_metrics['mean_edge_l_sc'][f'pred_cluster_{cluster_key}_{name}'] = x_cluster
    df = pd.DataFrame(mean_data)
    df.to_csv(os.path.join(out, 'avg_metrics_'+name+'.csv'))

def _calc_subgraph_mean(out,
                       subgraphs,):
    entries = os.listdir(out)
    for sub in subgraphs:
        sub_entrie = [entrie for entrie in entries if entrie.endswith(f'_{sub}.csv')]
        _mean_multiple_model_metrics(sub_entrie, sub, out)
    sub_entrie = [entrie for entrie in entries if entrie.endswith(f'_all.h5ad.csv')]
    _mean_multiple_model_metrics(sub_entrie, 'sc', out)
        
def _mean_multiple_model_metrics(sub_entrie,
                                appendix,
                                out):
    sim = np.empty(len(sub_entrie), dtype=np.float64)
    mi = np.empty(len(sub_entrie), dtype=np.float64)
    dist = np.empty(len(sub_entrie), dtype=np.float64)
    p_stat = np.empty(len(sub_entrie), dtype=np.float64)
    s_stat = np.empty(len(sub_entrie), dtype=np.float64)
    k_stat = np.empty(len(sub_entrie), dtype=np.float64)
    js_div = np.empty(len(sub_entrie), dtype=np.float64)
    ssim = np.empty(len(sub_entrie), dtype=np.float64)
    ari = np.empty(len(sub_entrie), dtype=np.float64)
    nmi = np.empty(len(sub_entrie), dtype=np.float64)
    tf_cov = np.empty(len(sub_entrie), dtype=np.float64)
    pw_cov = np.empty(len(sub_entrie), dtype=np.float64)
    hm_cov = np.empty(len(sub_entrie), dtype=np.float64)
    ro_cov = np.empty(len(sub_entrie), dtype=np.float64)
    kegg_cov = np.empty(len(sub_entrie), dtype=np.float64)
    for i, entrie in enumerate(sub_entrie):
        df = pd.read_csv(os.path.join(out, entrie), index_col=0)
        sim[i] = df.loc[df['Metric']=='CosSim']['Mean'].values[0]
        mi[i] = df.loc[df['Metric']=='MI']['Mean'].values[0]
        dist[i] = df.loc[df['Metric']=='MSE']['Mean'].values[0]
        p_stat[i] = df.loc[df['Metric']=='Pearson']['Mean'].values[0]
        s_stat[i] = df.loc[df['Metric']=='Spearman']['Mean'].values[0]
        k_stat[i] = df.loc[df['Metric']=='Kendall']['Mean'].values[0]
        js_div[i] = df.loc[df['Metric']=='JensenShannonDiv']['Mean'].values[0]
        ssim[i] = df.loc[df['Metric']=='SSIM']['Mean'].values[0]
        if 'ARI' in df.columns.to_list():
            ari[i] = df.loc[df['Metric']=='ARI']['Mean'].values[0]
            nmi[i] = df.loc[df['Metric']=='NMI']['Mean'].values[0]
        if 'tf_cov' in df.columns.to_list():
            tf_cov[i] = df.loc[df['Metric']=='CollecTRI_cov']['Mean'].values[0]
            pw_cov[i] = df.loc[df['Metric']=='PROGENy_cov']['Mean'].values[0]
            hm_cov[i] = df.loc[df['Metric']=='hallmark_msigdb_cov']['Mean'].values[0]
            ro_cov[i] = df.loc[df['Metric']=='reactome_msigdb_cov']['Mean'].values[0]
            kegg_cov[i] = df.loc[df['Metric']=='kegg_msigdb_cov']['Mean'].values[0]
    mean_data = {
        'Metric': ['Pearson', 'Spearman', 'Kendall', 'MI', 'CosSim', 'MSE', 'JensenShannonDiv', 'SSIM'],
        'Mean': [p_stat.mean(), s_stat.mean(), k_stat.mean(), mi.mean(), sim.mean(), dist.mean(), js_div.mean(), ssim.mean()],
        'Std': [p_stat.std(), s_stat.std(), k_stat.std(), mi.std(), sim.std(), dist.std(), js_div.std(), ssim.std()],
    }
    if 'tf_cov' in df.columns.to_list():
        mean_data['Metric'].extend(['CollecTRI_cov', 'PROGENy_cov', 'hallmark_msigdb_cov', 'reactome_msigdb_cov', 'kegg_msigdb_cov'])
        mean_data['Mean'].extend([tf_cov.mean(), pw_cov.mean(), hm_cov.mean(), ro_cov.mean(), kegg_cov.mean()])
        mean_data['Std'].extend([tf_cov.std(), pw_cov.std(), hm_cov.std(), ro_cov.std(), kegg_cov.std()])
    if 'ARI' in df.columns.to_list():
        mean_data['Metric'].extend(['ARI', 'NMI'])
        mean_data['Mean'].extend([ari.mean(), nmi.mean()])
        mean_data['Std'].extend([ari.std(), nmi.std()])
    df = pd.DataFrame(mean_data)
    df.to_csv(os.path.join(out, 'avg_metrics_'+f'{appendix}'+'.csv'))

def Metrics(path,
            target,
            name,
            num_subgraphs_per_graph,
            num_hops_per_subgraph=[],
            out='figures',
            sum_by_graph=False,
            do_clustering_metrics=True,
            do_pathway_metrics=None,
            do_performance_metrics=True):
    target_column_file_name = 'ROI' if sum_by_graph else 'Image'

    target_df = pd.read_csv(target,
                    header=0,
                    sep=',')
    target_df[target_column_file_name] = target_df[target_column_file_name].apply(lambda x: x.split('.')[0])
    if do_pathway_metrics is None:
        do_pathway_metrics = target_df.columns.shape[0] > 300

    y_clusters = {}
    y_cluster_enrichment = {}
    performance_metrics = {
        'SNR': pd.DataFrame(),
        'abundance': pd.DataFrame(),
        'mean_edge_l': pd.DataFrame(),
        'mean_edge_l_sc': pd.DataFrame()
    }

    if not os.path.exists(out):
        os.makedirs(out)

    _create_metrics(path,
                   target_df,
                   name,
                   target_column_file_name,
                   sum_by_graph,
                   num_subgraphs_per_graph,
                   num_hops_per_subgraph,
                   do_performance_metrics,
                   do_clustering_metrics,
                   do_pathway_metrics,
                   performance_metrics,
                   y_clusters,
                   y_cluster_enrichment,
                   out)
    if do_performance_metrics:
        if not os.path.exists(os.path.join(out, 'performance_metrics')):
            os.makedirs(os.path.join(out, 'performance_metrics'))
        for key in performance_metrics.keys():
            performance_metrics[key].to_csv(os.path.join(out, 'performance_metrics.csv'), index=False)
        if do_pathway_metrics:
            with open(os.path.join(out, 'cluster_enrichment.pcikle'), 'wb') as handle:
                pickle.dump(y_cluster_enrichment, handle, protocol=pickle.HIGHEST_PROTOCOL)
