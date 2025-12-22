import pandas as pd
import os
import matplotlib.pyplot as plt

path = 'figures/metrics/nanostring_100/6_6/metrics/mean/performance_metrics'
out = 'figures/metrics/nanostring_100/6_6/metrics/mean/performance_metrics'

hops = ['1', '2', '3', '5', '8', '11', 'sc']
m_names = ('sim', 'js_div', 'ari', 'nmi', 'mi', 'dist', 'pcc', 'scc', 'kcc')
comparison = ['SNR', 'abundance', 'mean_edge_l', 'mean_edge_l_sc']

if not os.path.exists(out):
    os.makedirs(out)

files = [file for file in os.listdir(path) if file.endswith('.csv')]


for file in files:
    metric = file.split('_metrics_')[-1].split('.')[0]
    df = pd.read_csv(os.path.join(path, file))
    cols = [col for col in df.columns.to_list() if col.startswith(m_names)]
    for col in cols:
        if not metric.endswith('_sc'):
            col_hop = col.split('_')[1] if col.split('_')[1] in hops else col.split('_')[2]
            col_metric = metric+'_'+ col_hop
            if col_metric not in df.columns:
                col_metric = col_metric.split('_')[0]+col_metric.split('_')[1]
        else:
            col_metric = col
        col_metric = metric if metric == 'mean_edge_l_sc' else col_metric
        plt.scatter(df[col_metric], df[col], s=1, alpha=0.5)
        plt.xlabel(f'{metric}')
        y_label = col.split('_')[0]
        plt.ylabel(y_label)
        plt.title(f'{col_metric}_{y_label}')
        plt.savefig(os.path.join(out, f'{col_metric}_{y_label}.png'),
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

