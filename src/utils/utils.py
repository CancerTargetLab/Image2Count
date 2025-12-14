import numpy as np
import scanpy as sc
import decoupler as dc
import random
import torch
import os
from skimage.metrics import structural_similarity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr, kendalltau, false_discovery_control
from sklearn.feature_selection import mutual_info_regression as mi

def per_gene_pcc(x, y, mean=True):
    """
    Calculate pearsonr correlation between x and y on a gene/protein wise level.

    Parameters:
    x (np.array): 2D number array
    y (np.array): 2D number array
    mean (bool): Wether to calculate mean gene/protein correlation

    Returns:
    (Correlation value,  p-value)
    """

    #print(f'Use src.utils.stats.per_gene_corr instead of this')
    return per_gene_corr(x, y, mean=mean, method='pearsonr')

def _get_method(method):
    if method.upper() == 'PEARSONR':
        corr = pearsonr
    elif method.upper() == 'SPEARMANR':
        corr = spearmanr
    elif  method.upper() == 'KENDALLTAU':
        corr = kendalltau
    else:
        raise Exception(f'Method {method} not one of pearsonr, spearmanr or kendalltau')
    return corr
    
def per_gene_corr(x, y, mean=True, method='pearsonr'):
    """
    Calculate correlation of method between x and y on a gene/protein wise level.

    Parameters:
    x (np.array): 2D number array
    y (np.array): 2D number array
    mean (bool): Wether to calculate mean gene/protein correlation
    method (str): String indicating method to be used ('PEARSONR', 'SPEARMANR', 'KENDALLTAU')

    Returns:
    (Correlation value,  p-value)
    """

    corr =  _get_method(method)
    x, y = x.astype(np.float64), y.astype(np.float64)
    statistic = np.ndarray(x.shape[-1])
    pval = np.ndarray(x.shape[-1])
    for gene in range(x.shape[-1]):
        pcc = corr(x[:,gene], y[:,gene])
        statistic[gene], pval[gene] = pcc.statistic, pcc.pvalue
    pval = false_discovery_control(pval, method='bh')
    if mean:
        return np.nanmean(statistic), np.nanmean(pval)
    else:
        return statistic, pval

def total_corr(x, y, method='pearsonr'):
    """
    Calculate correlation of method between x and y on a gene/protein wise level.

    Parameters:
    x (np.array): 2D number array
    y (np.array): 2D number array
    method (str): String indicating method to be used ('PEARSONR', 'SPEARMANR', 'KENDALLTAU')

    Returns:
    (Correlation value,  p-value)
    """

    corr =  _get_method(method)
    x, y = x.astype(np.float64).squeeze(), y.astype(np.float64).squeeze()
    statistic, pval = corr(x, y)
    pval = false_discovery_control(np.nan_to_num(pval, nan=1.), method='bh')
    return np.mean(statistic), np.mean(pval)

def per_gene_mi(x, y):
    """
    Calculate mutual information between x and y on a gene/protein wise level.

    Parameters:
    x (np.array): 2D number array
    y (np.array): 2D number array

    Returns:
    (np.array): mutual information
    """
    score = np.empty(x.shape[1], dtype=x.dtype)
    for gene in range(x.shape[1]):
        score[gene] = mi(x[:,gene].reshape(-1, 1), y[:,gene], n_jobs=-1)
    return score

def per_area_ari(x, y):
    """
    Calculate adjustend rand index between x and y .

    Parameters:
    x (np.array): 1D number array of clusters
    y (np.array): 1D number array of clusters

    Returns:
    (float): ARI
    """
    return adjusted_rand_score(y, x)

def per_area_nmi(x, y):
    """
    Calculate normalized mutual information for labels between x and y .

    Parameters:
    x (np.array): 1D number array of clusters
    y (np.array): 1D number array of clusters

    Returns:
    (float): NMI
    """
    return normalized_mutual_info_score(y, x)

def per_area_js_div(x, y):
    """
    Calculate jason shannon divergence between x and y on a cell/area wise level.

    Parameters:
    x (np.array): 2D number array
    y (np.array): 2D number array

    Returns:
    (np.array): jason shannon divergence
    """
    return jensenshannon(x, y, axis=1)**2

def per_area_ssim(x, y):
    """
    Calculate structural similarity between x and y on a cell/area wise level.

    Parameters:
    x (np.array): 2D number array
    y (np.array): 2D number array

    Returns:
    (np.array): jason shannon divergence
    """
    return structural_similarity(x, y, data_range=max(x.max(), y.max()), channel_axis=1)

def per_cluster_key_coverage(predicted_dict, true_dict, top_k=5):
    identified = 0
    true_total = 0
    for key in true_dict.keys():
        true_total += len(true_dict[key]['name'][:top_k])
        for val in predicted_dict[key]['name'][:top_k]:
            if val in true_dict[key]['name'][:top_k]:
                identified += 1
    return identified / true_total

def corr_all2all(adata, method='pearsonr'):
    """
    Calculate correlation of method between x and y on a gene/protein wise level.

    Parameters:
    adata (sc.AnnData): AnnData obj
    method (str): String indicating method to be used ('PEARSONR', 'SPEARMANR', 'KENDALLTAU')

    Returns:
    (Correlation value,  p-value)
    """

    corr =  _get_method(method)
    statistic = np.zeros((adata.var_names.values.shape[0],adata.var_names.values.shape[0]))
    pval = np.zeros((adata.var_names.values.shape[0],adata.var_names.values.shape[0]))
    for i in range(statistic.shape[0]):
        for j in range(statistic.shape[0]):
            if j >= i:
                statistic[i,j], pval[i,j] = corr(adata.X[i], adata.X[j])
            else:
                statistic[i,j], pval[i,j] = statistic[j,i], pval[j,i]
    return statistic, pval

def cluster_cell_expression(x):
    adata = sc.AnnData(x)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.pp.pca(adata,
                svd_solver='arpack', 
                n_comps=adata.var_names.shape[0]-1 if adata.var_names.shape[0]<51 else 50,
                )
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.varm['PCs'].shape[1])
    sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=4)
    return adata.obs['leiden'].values

def per_cluster_pathways(x, var_names, clusters, top_k=5):
    adata = sc.AnnData(x)
    adata.var_names = var_names
    adata.obs['leiden'] = clusters
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    collectri = dc.op.collectri(organism='human')
    dc.mt.ulm(data=adata, net=collectri, tmin=2 if adata.var_names.shape[0]<400 else 5)
    score = dc.pp.get_obsm(adata=adata, key='score_ulm')
    df = dc.tl.rankby_group(adata=score, groupby='leiden', reference='rest', method='t-test_overestim_var')
    df = df[df['stat'] > 0.0]
    df = df[df['padj'] <= 0.05]
    tf_source_markers = (
        df.groupby('group', observed=False)
        .head(top_k)
        .drop_duplicates('name')
        .groupby('group', observed=False)[['name', 'stat', 'pval', 'padj']]
        .apply(lambda x: {
            'name': x['name'].to_list(),
            'stat': x['stat'].to_list(),
            'pval': x['pval'].to_list(),
            'padj': x['padj'].to_list()
        })
        .to_dict())
    
    progeny = dc.op.progeny(organism="human")
    dc.mt.ulm(data=adata, net=progeny, tmin=2 if adata.var_names.shape[0]<400 else 5)
    score = dc.pp.get_obsm(adata=adata, key='score_ulm')
    df = dc.tl.rankby_group(adata=score, groupby='leiden', reference='rest', method='t-test_overestim_var')
    df = df[df['stat'] > 0.0]
    df = df[df['padj'] <= 0.05]
    pw_source_markers = (
        df.groupby('group', observed=False)
        .head(top_k)
        .drop_duplicates('name')
        .groupby('group', observed=False)[['name', 'stat', 'pval', 'padj']]
        .apply(lambda x: {
            'name': x['name'].to_list(),
            'stat': x['stat'].to_list(),
            'pval': x['pval'].to_list(),
            'padj': x['padj'].to_list()
        })
        .to_dict())

    hallmark = dc.op.hallmark(organism='human')
    dc.mt.ulm(data=adata, net=hallmark, tmin=2 if adata.var_names.shape[0]<400 else 5)
    score = dc.pp.get_obsm(adata=adata, key='score_ulm')
    df = dc.tl.rankby_group(adata=score, groupby='leiden', reference='rest', method='t-test_overestim_var')
    df = df[df['stat'] > 0.0]
    df = df[df['padj'] <= 0.05]
    hall_mark_source_markers = (
        df.groupby('group', observed=False)
        .head(top_k)
        .drop_duplicates('name')
        .groupby('group', observed=False)[['name', 'stat', 'pval', 'padj']]
        .apply(lambda x: {
            'name': x['name'].to_list(),
            'stat': x['stat'].to_list(),
            'pval': x['pval'].to_list(),
            'padj': x['padj'].to_list()
        })
        .to_dict())
    
    msigdb = dc.op.resource('MSigDB', verbose=True, organism='human')
    reactome = msigdb[msigdb['collection']=='reactome_pathways']
    reactome = reactome[~reactome.duplicated(('genese', 'genesymbol'))]
    geneset_size = reactome.groupby('geneset').size()
    ulm_genesets = geneset_size.index[(geneset_size > 15) & (geneset_size < 500)]
    reactome = reactome[reactome['geneset'].isin(ulm_genesets)]
    reactome = reactome.rename(columns={'geneset': 'source', 'genesymbol': 'target'})
    reactome = reactome[['source', 'target']]
    dc.mt.ulm(data=adata, net=reactome, tmin=2 if adata.var_names.shape[0]<400 else 5)
    score = dc.pp.get_obsm(adata=adata, key='score_ulm')
    df = dc.tl.rankby_group(adata=score, groupby='leiden', reference='rest', method='t-test_overestim_var')
    df = df[df['stat'] > 0.0]
    df = df[df['padj'] <= 0.05]
    reactome_source_markers = (
        df.groupby('group', observed=False)
        .head(top_k)
        .drop_duplicates('name')
        .groupby('group', observed=False)[['name', 'stat', 'pval', 'padj']]
        .apply(lambda x: {
            'name': x['name'].to_list(),
            'stat': x['stat'].to_list(),
            'pval': x['pval'].to_list(),
            'padj': x['padj'].to_list()
        })
        .to_dict())

    kegg = msigdb[msigdb['collection']=='kegg_pathways']
    kegg = kegg[~kegg.duplicated(('geneset', 'genesymbol'))]
    geneset_size = kegg.groupby('geneset').size()
    ulm_genesets = geneset_size.index[(geneset_size > 15) & (geneset_size < 500)]
    kegg = kegg[kegg['geneset'].isin(ulm_genesets)]
    kegg = kegg.rename(columns={'geneset': 'source', 'genesymbol': 'target'})
    kegg = kegg[['source', 'target']]
    dc.mt.ulm(data=adata, net=kegg, tmin=2 if adata.var_names.shape[0]<400 else 5)
    score = dc.pp.get_obsm(adata=adata, key='score_ulm')
    df = dc.tl.rankby_group(adata=score, groupby='leiden', reference='rest', method='t-test_overestim_var')
    df = df[df['stat'] > 0.0]
    df = df[df['padj'] <= 0.05]
    kegg_source_markers = (
        df.groupby('group', observed=False)
        .head(top_k)
        .drop_duplicates('name')
        .groupby('group', observed=False)[['name', 'stat', 'pval', 'padj']]
        .apply(lambda x: {
            'name': x['name'].to_list(),
            'stat': x['stat'].to_list(),
            'pval': x['pval'].to_list(),
            'padj': x['padj'].to_list()
        })
        .to_dict())

    return {'CollecTRI': tf_source_markers, 
            'PROGENy': pw_source_markers,
            'hallmark_msigdb': hall_mark_source_markers,
            'reactome_msigdb': reactome_source_markers,
            'kegg_msigdb': kegg_source_markers}

def avg_cell_n(path):
      """
      Calculate average number of cells per graph for graphs in directory.

      Parameters:
      path (str): String path to directory containing torch graphs

      Returns:
      float: Average number of cells per graph
      """
      
      import os
      import torch
      files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.pt')]
      num_cells = 0
      for file in files:
            num_cells += torch.load(file, weights_only=False).x.shape[0]
      return num_cells/len(files)

def avg_edge_len(path):
      """
      Calculate average number of edge lengths per graph for graphs in directory.

      Parameters:
      path (str): String path to directory containing torch graphs

      Returns:
      float: Average number of cells per graph
      """
      
      import os
      import torch
      files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.pt')]
      num_edges = 0
      edge_sum = 0
      for file in files:
            graph = torch.load(file, weights_only=False)
            num_edges += graph.edge_attr.shape[0]
            edge_sum += torch.sum(graph.edge_attr)
      return edge_sum/num_edges

def avg_roi_area(path):
      """
      Calculate average number of area per graph for graphs in directory. Needs work to be correct.

      Parameters:
      path (str): String path to directory containing torch graphs

      Returns:
      float: Average number of area per graph
      """
      
      import os
      import torch
      files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.pt')]
      area = 0
      area_list = []
      for file in files:
            graph = torch.load(file, weights_only=False)
            tmp = (torch.max(graph.pos[:,0])-torch.min(graph.pos[:,0]))*(torch.max(graph.pos[:,1])-torch.min(graph.pos[:,1]))
            area += tmp
            area_list.append(tmp.item())
      return area/len(files), torch.median(torch.tensor(area_list))

def set_seed(seed, cuda_reproduce=True):
    """
    Set seed for random,  numpy, torch and sklearn.
    
    Parameters:
    seed (int): Seed
    cuda_reproduce (bool): Wether or not to use cuda reproducibility
    """
    
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA devices, if available
    
    # Additional settings for reproducibility
    if cuda_reproduce:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def merge(save_dir):
    result_dirs = [result_dir for result_dir in  os.listdir(save_dir)  if result_dir != 'merged' and result_dir != 'mean' and not '_' in result_dir and not result_dir.endswith('.pt')]
    result_files = [os.listdir(os.path.join(save_dir, result_dir)) for result_dir in result_dirs]

    list(map(lambda x: x.sort(), result_files))

    if not os.path.exists(os.path.join(save_dir, 'mean')):
        os.makedirs(os.path.join(save_dir, 'mean'))

    for file in range(len(result_files[0])):
        if result_files[0][file].startswith('cell'):
            file_contents = []
            for result_dir in range(len(result_dirs)):
                file_contents.append(torch.load(os.path.join(save_dir, result_dirs[result_dir], result_files[result_dir][file]), weights_only=True, map_location='cpu'))
            merged = torch.empty((len(result_files), file_contents[0].shape[0], file_contents[0].shape[1]), dtype=file_contents[0].dtype)
            for i in range(len(result_files)):
                merged[i] = file_contents[i]
            merged = torch.mean(merged, dim=0, keepdim=True)[0].squeeze()
            torch.save(merged, os.path.join(save_dir, 'mean', result_files[0][file]))
            torch.save(torch.sum(merged, dim=0), os.path.join(save_dir, 'mean', result_files[0][file]).replace('cell_', 'roi_'))

def load(path, save_keys, device='cpu'):
    """
    Load metrics of torch save dict.

    Parameters:
    path (str): Path to torch save dict
    save_keys (list): list containing keys for metrics to extract
    device (str): Location to load torch save dict to
    """
    
    save = torch.load(path, map_location=device, weights_only=False)
    if type(save_keys) == list and  type(save) == dict:
        out = {}
        for key in save_keys:
            if key in save.keys():
                out[key] = save[key]
            else:
                print(f'{key} not found in in save {path}')
    elif type(save_keys) == str:
        out = save[save_keys]
    return out

def insert_coords(adata, df):
    adata.obs['Image'] = adata.obs['files'].apply(lambda x: x.split('graph_')[-1].split('.pt')[0]+'.tiff')

    adata.obs['x'] = np.nan
    adata.obs['y'] = np.nan

    for img in adata.obs['Image'].unique().tolist():
        adata.obs.loc[adata.obs['Image']==img, ['x', 'y']] = df.loc[df['Image'].isin([img]), ['Centroid.X.px', 'Centroid.Y.px']].values
    
    return adata