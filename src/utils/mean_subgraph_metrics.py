import os
import pandas as pd
import numpy as np

path = 'figures/tmp/crc_1p/35_0/'
subgraphs = [1, 2, 3, 5, 8, 11]
out = 'figures/tmp/crc_1p/35_0/'

def calc_subgraph_mean(path):
    entries = os.listdir(path)
    for sub in subgraphs:
        sub_entrie = [entrie for entrie in entries if entrie.endswith(f'_{sub}.csv')]
        metrics(sub_entrie, sub)
    sub_entrie = [entrie for entrie in entries if entrie.endswith(f'_all.h5ad.csv')]
    metrics(sub_entrie, 'sc')
        
def metrics(sub_entrie, appendix):
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
    for i, entrie in enumerate(sub_entrie):
        df = pd.read_csv(os.path.join(path, entrie), index_col=0)
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
    mean_data = {
        'Metric': ['Pearson', 'Spearman', 'Kendall', 'MI', 'CosSim', 'MSE', 'JensenShannonDiv', 'SSIM'],
        'Mean': [p_stat.mean(), s_stat.mean(), k_stat.mean(), mi.mean(), sim.mean(), dist.mean(), js_div.mean(), ssim.mean()],
        'Std': [p_stat.std(), s_stat.std(), k_stat.std(), mi.std(), sim.std(), dist.std(), js_div.std(), ssim.std()],
    }
    if 'ARI' in df.columns.to_list():
        mean_data['Metric'].extend(['ARI', 'NMI'])
        mean_data['Mean'].extend([ari.mean(), nmi.mean()])
        mean_data['Std'].extend([ari.std(), nmi.std()])
    df = pd.DataFrame(mean_data)
    df.to_csv(os.path.join(out, 'avg_metrics_'+f'{appendix}'+'.csv'))

calc_subgraph_mean(path)