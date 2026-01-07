import pandas as pd
import scanpy as sc
import numpy as np
import decoupler as dc
import os
import json
from src.utils.utils import per_cluster_key_coverage, cluster_cell_expression

adata_path = 'save/nanostring/nanostring_100_lin_mean_all.h5ad'
target_df_path = 'save/nanostring/nanostring_measurements_flipped_y_dca.csv.zip'

name = 'nanostring_lin_mean'
out = 'figures/metrics/nanostring_100_dca/celltypes/'

def per_cluster_cell_enrichment(x, var_names, clusters, celltypes, top_k=5):
    adata = sc.AnnData(x)
    adata.var_names = var_names
    adata.obs['leiden'] = clusters
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    dc.mt.ulm(data=adata, net=celltypes, tmin=5, verbose=True)
    score = dc.pp.get_obsm(adata=adata, key='score_ulm')
    df = dc.tl.rankby_group(adata=score, groupby='leiden', reference='rest', method='t-test_overestim_var')
    df = df[df['stat'] > 0.0]
    df = df[df['padj'] <= 0.05]
    cluster_celltypes = (
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
    return cluster_celltypes


target_column_file_name = 'Image'

adata = sc.read_h5ad(adata_path)
target_df = pd.read_csv(target_df_path)


target_df[target_column_file_name] = target_df[target_column_file_name].apply(lambda x: x.split('.')[0])
var_names = adata.var_names.values
tmp = pd.DataFrame(data=adata.X, columns=var_names)
tmp['files'] = adata.obs['files'].values
adata = tmp
adata['files'] = adata['files'].apply(lambda x: x.split('graph_')[-1].split('.')[0])
adata = adata[adata['files'].isin(target_df[target_column_file_name])]     #Selects files that exist in pred, in case of only investiating test data
target_df = target_df[target_df[target_column_file_name].isin(adata['files'])]
adata = adata.sort_values(by='files', kind='stable', ignore_index=True)
target_df = target_df.sort_values(by=target_column_file_name, kind='stable', ignore_index=True)
files = adata['files']

x = adata[var_names].values
y = target_df[var_names].values

is_g_zero = np.sum(y, axis=-1) > 0
x = x[is_g_zero]
y = y[is_g_zero]

x_cluster = cluster_cell_expression(x.copy())
y_cluster = cluster_cell_expression(y.copy())

if not os.path.exists(os.path.join('data', 'raw', 'msigdb')):
    os.makedirs(os.path.join('data', 'raw', 'msigdb'))
pathway_csvs = os.listdir(os.path.join('data', 'raw', 'msigdb'))
celltypes = None
for csv in pathway_csvs:
    if csv.endswith('.csv') and csv.startswith('celltypes1'):
        celltypes =  pd.read_csv(os.path.join('data', 'raw', 'msigdb', csv))
if celltypes is None:
    raise Exception('No celltypes found, go to http://www.bio-bigdata.center/CellMarkerSearch.jsp?quickSearchInfo=lung&index_key=2#framekuang and download csv and create celltype csv!')

y_enrichment = per_cluster_cell_enrichment(y.copy(),
                                           var_names,
                                           clusters=y_cluster,
                                           celltypes=celltypes,
                                           top_k=5)
yx_enrichment = per_cluster_cell_enrichment(y.copy(),
                                           var_names,
                                           clusters=x_cluster,
                                           celltypes=celltypes,
                                           top_k=5)
x_enrichment = per_cluster_cell_enrichment(x.copy(),
                                           var_names,
                                           clusters=x_cluster,
                                           celltypes=celltypes,
                                           top_k=5)
xy_enrichment = per_cluster_cell_enrichment(x.copy(),
                                           var_names,
                                           clusters=y_cluster,
                                           celltypes=celltypes,
                                           top_k=5)

celltype_dict = {
    'y_enrichment': y_enrichment,
    'x_enrichment': x_enrichment,
    'yx_enrichment': yx_enrichment,
    'xy_enrichment': xy_enrichment,
}

celltype_coverage_top5 = per_cluster_key_coverage(xy_enrichment,
                            y_enrichment, top_k=5)/2 + per_cluster_key_coverage(x_enrichment,
                                                            yx_enrichment, top_k=5)/2
celltype_coverage_top3 = per_cluster_key_coverage(xy_enrichment,
                            y_enrichment, top_k=3)/2 + per_cluster_key_coverage(x_enrichment,
                                                            yx_enrichment, top_k=3)/2
celltype_coverage_top1 = per_cluster_key_coverage(xy_enrichment,
                            y_enrichment, top_k=1)/2 + per_cluster_key_coverage(x_enrichment,
                                                            yx_enrichment, top_k=1)/2

print('Celltype coverage Top 5: ', celltype_coverage_top5)
print('Celltype coverage Top 3: ', celltype_coverage_top3)
print('Celltype coverage Top 1: ', celltype_coverage_top1)

if not os.path.exists(out):
    os.makedirs(out)

files = files.values[is_g_zero]
df = pd.DataFrame()
df['files'] = files
df['y_cluster'] = y_cluster
df['x_cluster'] = x_cluster
df.to_csv(os.path.join(out, f'clusters_{name}.csv'),
        index=False,
        header=True,
        sep=',')

with open(os.path.join(out, f'celltypes_per_cluster_{name}.json'), 'w') as handle:
    json.dump(celltype_dict, handle, indent=4)
