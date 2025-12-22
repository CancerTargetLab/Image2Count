from scipy.stats import ranksums, false_discovery_control
import pandas as pd
import numpy as np
import os

path = ['figures/metrics/nanostring_100', 'figures/metrics/nanostring_100_dca']
mean = True
seperate = False
path_join = 'metrics'
out_name = 'nanostring'
endswith = '.h5ad.csv' # ''

val_dict = {}
stat_name = []



def get_vals(path, subdirs, val_dict, stat_name):
    for subdir in subdirs:
        if path_join:
            subdir = os.path.join(subdir, path_join)
        if os.path.isdir(os.path.join(path, subdir)):
            val_dict[subdir] = []
            if not mean:
                entries = os.listdir(os.path.join(path, subdir))
            else:
                entries = os.listdir(os.path.join(path, subdir, 'mean'))
            for entrie in entries:
                if endswith:
                    if not entrie.endswith(endswith):
                        continue
                if entrie.endswith('.csv') and not mean and entrie.startswith('avg_metrics_'): #and len(entrie) < 19:
                    tmp = pd.read_csv(os.path.join(path, subdir, entrie))
                    val_dict[subdir].append(tmp['Mean'].values)
                    if len(stat_name) == 0:
                        stat_name.extend(tmp['Metric'].tolist())
                elif entrie.endswith('.csv') and mean and entrie.startswith('avg_metrics_'): #and len(entrie) < 19:
                    tmp = pd.read_csv(os.path.join(path, subdir, 'mean', entrie))
                    val_dict[subdir].append(tmp['Mean'].values) 
                    if len(stat_name) == 0:
                        stat_name.extend(tmp['Metric'].tolist())

if isinstance(path, str):
    subdirs = os.listdir(path)
    get_vals(path, subdirs, val_dict, stat_name)
elif isinstance(path, list):
    for p in path:
        tmp_dict = {}
        subdirs = os.listdir(p)
        get_vals(p, subdirs, tmp_dict, stat_name)
        if seperate:
            for key in tmp_dict.keys():
                val_dict[p.split('/')[-1]+'_'+key] = tmp_dict[key]
        else:
            for key in tmp_dict.keys():
                if key in val_dict.keys():
                    val_dict[key].extend(tmp_dict[key])
                else:
                    val_dict[key] = tmp_dict[key]


keys = list(val_dict.keys())
key0 = []
key1 = []
stats = []
pvals = []
for i in range(len(keys)-1):
    for j in range(i+1, len(keys)):
        wrs = ranksums(val_dict[keys[i]], val_dict[keys[j]])
        key0.append(keys[i])
        key1.append(keys[j])
        stats.append(wrs.statistic)
        pvals.append(wrs.pvalue)
stats = np.array(stats)
pvals = false_discovery_control(np.array(pvals))
sol_dict = {'m1': key0, 'm2': key1}
for i in range(stats.shape[1]):
    sol_dict[f'{stat_name[i]}'] = stats[:,i]
    sol_dict[f'pval_{stat_name[i]}'] = pvals[:,i]
df = pd.DataFrame(sol_dict)

if isinstance(path, str):
    addage = ''
    addage += '_mean' if mean else ''
    addage += endswith.split('.csv')[0] if endswith else ''
    df.to_csv(os.path.join(path, f'{out_name}_ranksums{addage}.csv'))
else:
    if isinstance(path, list):
        path = os.path.join(path[0], '..')
        addage = '_'+path.split('/')[-2].split('_')[0]
        addage += '_mean' if mean else ''
        addage += '_sep' if seperate else '_merged_runs'
        addage += endswith.split('.csv')[0] if endswith else ''
        df.to_csv(os.path.join(path, f'{out_name}_ranksums{addage}.csv'))
