import pandas as pd
import numpy as np
import scanpy as sc
import statsmodels.formula.api as smf
from scipy.stats import false_discovery_control
from statsmodels.stats.contrast import _get_pairs_labels, _embed_constraints, t_test_multi
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import os
import glob

paths = ['figures/metrics/crc_1p',]
models = ['6_6', '17_0', 'lin']
levels = ['1', '2', '3', '5', '8', '11', 'sc']
_addage = 'metrics/mean/performance_metrics'
name_performance_metrics = ('abundance.csv', 'mean_edge_l_sc.csv', 'mean_edge_l.csv')
name = 'figures/metrics/nanostring_100'

lower_is_better = ['dist', 'js_div']
metrics = ['sim', 'js_div', 'dist', 'mi', 'pcc', 'scc', 'true_cluster']


df = []
var_names = sc.read_h5ad('save/nanostring/nanostring_100_6_6_mean_all.h5ad').var_names
for csv in os.listdir('data/raw/msigdb/'):
    if 'celltype' not in csv:
        tmp = pd.read_csv(os.path.join('data/raw/msigdb/', csv))
        tmp = tmp[tmp['target'].isin(var_names)]
        geneset_size = tmp.groupby('source').size()
        ulm_genesets = geneset_size.index[(geneset_size > 10) & (geneset_size < 500)]
        tmp = tmp[tmp['source'].isin(ulm_genesets)]
        df.append(tmp)
df = pd.concat(df)[['source', 'target']]
binary_matrix = pd.crosstab(df['target'], df['source'])
jaccard_dist = pdist(binary_matrix.values, metric='jaccard')
dist_matrix = squareform(jaccard_dist)
aglomcluster = AgglomerativeClustering(n_clusters=50, metric='precomputed', linkage='complete', )
binary_matrix['gene_cluster'] = aglomcluster.fit_predict(dist_matrix)

df = pd.DataFrame({'gene': var_names, 'cluster': [-1]*var_names.shape[0]})
cluster_lookup = binary_matrix['gene_cluster']
df.loc[df['gene'].isin(cluster_lookup.index), 'cluster'] = df.loc[df['gene'].isin(cluster_lookup.index), 'gene'].map(cluster_lookup)
gene_df = df

aggregated_results_genes = {}
aggregated_results_areas = {}
aggregated_results_sc = {}

df_metrics_name = [(lambda x: x+'_')(metric) for metric in metrics]
metrics_index = None

for path in paths:
    for model in models:
        model_path = os.path.join(path, model, _addage, 'performance_metrics*.csv')
        files = glob.glob(model_path)

        for f in files: # TODO: as before, append?
            if f.endswith(name_performance_metrics):
                df = pd.read_csv(f)
                if 'markers' in df.columns:
                    assert (gene_df['gene'] == df['markers']).sum() == gene_df['gene'].shape[0]
                df = df[df.columns[df.columns.str.startswith(tuple(df_metrics_name))]]
                df = df.rename(columns=(lambda x: x.split(os.path.basename(path))[0]))
                if f.endswith(name_performance_metrics[0]):
                    aggregated_results_genes[model] = df
                elif f.endswith(name_performance_metrics[1]):
                    aggregated_results_areas[model] = df
                elif f.endswith(name_performance_metrics[2]):
                    aggregated_results_sc[model] = df

def aggregate_results(result_dict):
    _aggregated_results = []
    for key in result_dict.keys():
        df = result_dict[key]
        df['architecture'] = key
        _aggregated_results.append(df)
    return pd.concat(_aggregated_results)

def test_effect(results_df):
    results_summary = []
    result_metrics = []
    for metric in results_df.columns[results_df.columns.str.startswith(tuple(metrics[:-1]))]:
        if 'true_cluster' in results_df.columns:
            group = 'true_cluster'
        else:
            group = 'true_cluster_'+metric.split('_')[-2]+'_'
        result_metrics.append([metric, group])
    for metric, group in result_metrics:
        if metric.startswith(tuple(lower_is_better)):
            results_df[metric] = results_df[metric] * -1

        results_df[group] = results_df[group].astype('category')
        results_df['architecture'] = results_df['architecture'].astype('category')

        model = smf.mixedlm(f"{metric} ~ architecture", results_df, groups=group)
        result = model.fit()
        
        term_name = 'architecture'
        desinfo = result.model.data.design_info
        term_idx = desinfo.term_names.index(term_name)
        term = desinfo.terms[term_idx]
        idx_start = desinfo.term_slices[term].start
        factor = term.factors[0]
        cat = desinfo.factor_infos[factor].categories
        
        k_level = len(cat)
        cm = desinfo.term_codings[term][0].contrast_matrices[factor].matrix

        k_params = len(result.params)
        labels = _get_pairs_labels(k_level, cat)

        import statsmodels.sandbox.stats.multicomp as mc
        c_all_pairs = -mc.contrast_allpairs(k_level)
        contrasts_sub = c_all_pairs.dot(cm)
        contrasts = _embed_constraints(contrasts_sub, k_params, idx_start)  #Error is here somehow, but works like this
        res_df = t_test_multi(result, contrasts[:,:-1], method='fdr_bh', ci_method=None,
                            alpha=0.05, contrast_names=labels)
        res = res_df[['coef', 'pvalue-fdr_bh']]
        res = res.rename(columns={'coef': f'{metric}coef', 'pvalue-fdr_bh': f'{metric}pv_fdr_bh1'}).T
        results_summary.append(res)

    return pd.concat(results_summary)

aggregated_results_genes = aggregate_results(aggregated_results_genes)
aggregated_results_genes['true_cluster'] = gene_df['cluster']
aggregated_results_areas = aggregate_results(aggregated_results_areas)
aggregated_results_sc = aggregate_results(aggregated_results_sc)

result_genes = test_effect(aggregated_results_genes)
result_areas = test_effect(aggregated_results_areas)
result_sc = test_effect(aggregated_results_sc)

agg_results = pd.concat([result_genes, result_areas, result_sc])
idx = agg_results.index.str.contains('pv_fdr_bh1')
pvals = agg_results.iloc[idx].values
pvals_r = pvals.reshape((pvals.shape[0]*pvals.shape[1]))

adj_pvals = false_discovery_control(pvals_r).reshape(pvals.shape)
agg_results.iloc[idx] = adj_pvals

agg_results.to_csv(name+'.csv')
agg_results.to_latex(name+'.txt')
