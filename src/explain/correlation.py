import os
import scanpy as sc
import pandas as pd
import numpy as np
import torch
from src.utils.utils import per_gene_corr, total_corr, per_gene_mi

def correlation(raw_subset_dir='CRC',
                figure_dir='figures/crc_gat_test',
                vis_name='crc_gat_test'):
    adata_name = vis_name
    out = figure_dir

    df = pd.read_csv([os.path.join('data/raw/',
                                   raw_subset_dir,
                                   p) for p in os.listdir(os.path.join('data/raw/',
                                                                       raw_subset_dir)) if p.endswith('.csv')][0],
                     header=0,
                     sep=',')
    df['Image'] = df['Image'].apply(lambda x: x.split('.')[0])

    adata = sc.read_h5ad(os.path.join('out', adata_name+'_all.h5ad'))
    var_names = adata.var_names.values
    tmp = pd.DataFrame()
    tmp['files'] = adata.obs['files'].values
    tmp[var_names] = adata.X
    adata = tmp
    adata['files'] = adata['files'].apply(lambda x: x.split('graph_')[-1].split('.')[0])
    adata = adata[adata['files'].isin(df['Image'])]     #Selects files that exist in pred, in case of only investiating test data
    df = df[df['Image'].isin(adata['files'])]
    adata = adata.sort_values(by='files', kind='stable', ignore_index=True)
    df = df.sort_values(by='Image', kind='stable', ignore_index=True)

    pred = adata[adata.columns[1:].values].values
    y = df[adata.columns[1:].values].values
    is_g_zero = np.sum(y, axis=-1) > 0
    pred = pred[is_g_zero]
    y = y[is_g_zero]

    p_statistic, p_pval = per_gene_corr(pred, y, mean=False, method='PEARSONR')
    s_statistic, s_pval = per_gene_corr(pred, y, mean=False, method='SPEARMANR')
    k_statistic, k_pval = per_gene_corr(pred, y, mean=False, method='KENDALLTAU')

    mi = per_gene_mi(pred, y)
    similarity = torch.nn.CosineSimilarity()
    sim = similarity(torch.from_numpy(pred), torch.from_numpy(y)).squeeze().detach().numpy()
    mse = torch.nn.MSELoss(reduce=None)(torch.from_numpy(pred), torch.from_numpy(y)).squeeze().detach().numpy()
        
    correlation_data = {
        'Variable': var_names,
        'Pearson Correlation Coef.': [corr for corr in p_statistic],
        'Pearson p-value': [corr for corr in p_pval],
        'Spearman Correlation Coef.': [corr for corr in s_statistic],
        'Spearman p-value': [corr for corr in s_pval],
        'Kendall Correlation Coef.': [corr for corr in k_statistic],
        'Kendall p-value': [corr for corr in k_pval],
        'Mutual Information': [corr for corr in mi]
    }

    corr_df = pd.DataFrame(correlation_data)
    mean_values = corr_df[corr_df.columns[1:]].mean()
    mean_row = pd.DataFrame({'Variable': 'mean', **mean_values}, index=[0])
    std_values = corr_df[corr_df.columns[1:]].std()
    std_row = pd.DataFrame({'Variable': 'std', **std_values}, index=[0])
    corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

    corr_df.to_csv(os.path.join(out, 'corr_'+adata_name+'.csv'))

    mean_data = {
         'Metric': ['Pearson', 'Spearman', 'Kendall', 'MI', 'CosSim', 'MSE'],
         'Mean': [p_statistic.mean(), s_statistic.mean(), k_statistic.mean(), mi.mean(), sim.mean(), mse.mean()],
         'Std': [p_statistic.std(), s_statistic.std(), k_statistic.std(), mi.std(), sim.std(), mse.std()],
    }
    df = pd.DataFrame(mean_data)
    df.to_csv(os.path.join(out, 'avg_metrics_'+adata_name+'.csv'))

    corr_p = np.ndarray(adata['files'].unique().shape[0])
    corr_s = np.ndarray(adata['files'].unique().shape[0])
    corr_k = np.ndarray(adata['files'].unique().shape[0])
    pval_p = np.ndarray(adata['files'].unique().shape[0])
    pval_s = np.ndarray(adata['files'].unique().shape[0])
    pval_k = np.ndarray(adata['files'].unique().shape[0])
    sorted_ids = sorted(adata['files'].unique().tolist())
    for i, id in enumerate(sorted_ids):
        corr_p[i], pval_p[i] = total_corr(adata[adata['files']==id][adata.columns[1:].values].values,
                                            df[df['Image']==id][adata.columns[1:].values].values,
                                            method='PEARSONR')
        corr_s[i], pval_s[i] = total_corr(adata[adata['files']==id][adata.columns[1:].values].values,
                                            df[df['Image']==id][adata.columns[1:].values].values,
                                            method='SPEARMANR')
        corr_k[i], pval_k[i] = total_corr(adata[adata['files']==id][adata.columns[1:].values].values,
                                            df[df['Image']==id][adata.columns[1:].values].values,
                                            method='KENDALLTAU')

    correlation_data = {
        'ROIs': sorted_ids,
        'Pearson Correlation Coef.': [corr for corr in corr_p],
        'Pearson p-value': [corr for corr in pval_p],
        'Spearman Correlation Coef.': [corr for corr in corr_s],
        'Spearman p-value': [corr for corr in pval_s],
        'Kendall Correlation Coef.': [corr for corr in corr_k],
        'Kendall p-value': [corr for corr in pval_k]
    }

    corr_df = pd.DataFrame(correlation_data)
    mean_values = corr_df[corr_df.columns[1:]].mean()
    mean_row = pd.DataFrame({'ROIs': 'mean', **mean_values}, index=[0])
    std_values = corr_df[corr_df.columns[1:]].std()
    std_row = pd.DataFrame({'ROIs': 'std', **std_values}, index=[0])
    corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

    corr_df.to_csv(os.path.join(out, 'corr_files_'+adata_name+'.csv'))

    out = os.path.join(out, 'cell_corr_per_roi')
    if not os.path.exists(out) and not os.path.isdir(out):
            os.makedirs(out)
    sorted_ids = sorted(adata['files'].unique().tolist())
    for i, id in enumerate(sorted_ids):
        p_statistic, p_pval = per_gene_corr(adata[adata['files']==id][adata.columns[1:].values].values,
                                            df[df['Image']==id][adata.columns[1:].values].values,
                                            method='PEARSONR',
                                            mean=False)
        s_statistic, s_pval = per_gene_corr(adata[adata['files']==id][adata.columns[1:].values].values,
                                            df[df['Image']==id][adata.columns[1:].values].values,
                                            method='SPEARMANR',
                                            mean=False)
        k_statistic, k_pval = per_gene_corr(adata[adata['files']==id][adata.columns[1:].values].values,
                                            df[df['Image']==id][adata.columns[1:].values].values,
                                            method='KENDALLTAU',
                                            mean=False)
            
        correlation_data = {
            'Variable': var_names,
            'Pearson Correlation Coef.': [corr for corr in p_statistic],
            'Pearson p-value': [corr for corr in p_pval],
            'Spearman Correlation Coef.': [corr for corr in s_statistic],
            'Spearman p-value': [corr for corr in s_pval],
            'Kendall Correlation Coef.': [corr for corr in k_statistic],
            'Kendall p-value': [corr for corr in k_pval]
        }

        corr_df = pd.DataFrame(correlation_data)
        mean_values = corr_df[corr_df.columns[1:]].mean()
        mean_row = pd.DataFrame({'Variable': 'mean', **mean_values}, index=[0])
        std_values = corr_df[corr_df.columns[1:]].std()
        std_row = pd.DataFrame({'Variable': 'std', **std_values}, index=[0])
        corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

        corr_df.to_csv(os.path.join(out, f'corr_{id}_'+adata_name+'.csv'))
