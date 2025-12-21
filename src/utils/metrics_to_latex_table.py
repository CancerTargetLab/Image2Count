import os
import pandas as pd

path = 'figures/metrics/'

num_subgraphs_plus_sc = 7

metrics = ['CollecTRI_cov', 'PROGENy_cov', 'hallmark_msigdb_cov', 'reactome_msigdb_cov', 'kegg_msigdb_cov']#['Pearson', 'Spearman', 'MI', 'MSE', 'CosSim', 'JensenShannonDiv', 'SSIM', 'ARI', 'NMI']

dir_is_only_train = ['hkgmh3_74', 'hkgmh3_74_30', 'hkgmh3_74_50']

metric_dir_in_model_dir = 'metrics'

map_avgfile2index = {
    '_1.csv': 1,
    '_2.csv': 2,
    '_3.csv': 3,
    '_5.csv': 4,
    '_8.csv': 5,
    '_11.csv': 6,
    '_mean_all.h5ad.csv': 0,
    '_sc.csv': 0
}

map_dir2name = {
    # 'crc_1p': '',
    # 'crc_2c_1p': '2 Channels',
    # 'crc_6_1p': '8{\\textmu}m',
    # 'crc_14_1p': '20{\\textmu}m',
    # 'crc_1p_nc': 'no KM',
    'nanostring_100': '',
    'nanostring_100_dca': 'DCA',
    # 'hkgmh3_74': '8{\\textmu}m',
    # 'hkgmh3_74_30': '12{\\textmu}m',
    # 'hkgmh3_74_50': '20{\\textmu}m',
}

map_model2name = {
    'lin': 'ResNet',
    # '24_6': 'GAT I2C',
    # '35_0': 'FFW I2C',
    # '3_3': 'GAT I2C',
    # '7_0': 'FFW I2C',
    '6_6': 'GAT I2C',
    '17_0': 'FFW I2C',
}

start_str = """
\\begin{table*}[hbt!]
    \\centering
    \\resizebox{\\textwidth}{!}{%
    \\begin{NiceTabular}{|l|c|c|c|c|c|c|c|}
        \\toprule
        & Single Cell & 1 hop & 2 hop & 3 hop & 5 hop & 8 hop & train size \\\\
"""

map_exp2str = {
    'nanostring': """
        N cells CosMx & 1 & 7.68 & 21.28 & 42.95 & 109.69 & 266.01 & 483.71 \\\\
    """,
    'crc': """
        N cells CycIF & 1 & 7.78 & 19.85 & 38.25 & 95.88 & 234.84& 438.54 \\\\
    """,
    'hkgmh3': """
        N cells GeoMx & - & - & - & - & - & - & 269.83 \\\\
    """
}

end_str = """
        \\bottomrule
    \\end{NiceTabular}
    }
    \\caption{ of predictions of linear evaluation head of the ResNet visual feature backbone, FFW Image2Count (I2C) 
    and graph-based Image2Count models to counts of the test data from CosMx, CycIF and GeoMx datasets, averaged 
    over all cross-validation runs (6 for CosMx, 10 for CycIF and GeoMx) from average per-gene correlation. Merged 
    predictions were obtained by calculating the mean prediction over all cross-validation runs. DCA signifies models 
    trained on raw data with correlation calculated from predicted counts and imputed count data using DCA (deep-count 
    autoencoder). $K$ hop stands for a k-hop subgraph created from a central cell, expanding the graph to neighboring 
    nodes $k$ times. For each image of the CosMx dataset, $6\\times 6=36$ subgraphs for a total of 1800 were created. 
    For the CycIF dataset, $30\\times 30=900$ subgraphs were created for a total of 2700 subgraphs. Edges were 
    determined via k-NN of spatial cell positions with number of neighbors set to 6. N Cells shows the (average) 
    number of cells whose total expression was used to calculate correlation. Correlation is the mean Spearman 
    correlation over all markers per cross-validation fold, averaged over all cross-validation folds.}
    \\label{tab:performance}
\end{table*}
"""

def is_average_metrics_file(file, mean=False):
    if '.h5ad' in file and not mean:
        return False
    elif 'mean' == file or 'performance_metrics' == file:
        return False
    elif file.endswith('.json') or file.startswith('performance_metrics'):
        return False
    else:
        return True

def create_table(path, metric):
    table = start_str
    experiment = [elem for elem in os.listdir(path) if os.path.isdir(os.path.join(path, elem)) and elem in map_dir2name.keys()]
    experiment.sort()
    for exp in experiment:
        exp_str = [map_exp2str[key] for key in map_exp2str.keys() if exp.startswith(key)][0]
        if exp_str not in table:
            table += '\n\t\t\\midrule\n'+exp_str+'\n'
        table += '\t\t\\dotrule\n'
        lin = 'ResNet'
        ffw = 'FFW I2C'
        gat = 'GAT I2C'
        models = [file for file in os.listdir(os.path.join(path, exp,)) if file in map_model2name.keys()]
        for model in models:
            name = map_model2name[model]

            metrics_mean = ['-']*num_subgraphs_plus_sc
            metrics_std = ['-']*num_subgraphs_plus_sc
            avg_metrics_files = [file for file in os.listdir(os.path.join(path, exp, model, metric_dir_in_model_dir)) if is_average_metrics_file(file)]
            for file in avg_metrics_files:
                file_end = map_avgfile2index[[key for key in map_avgfile2index.keys() if file.endswith(key)][0]]
                if exp in dir_is_only_train:
                    file_end = -1
                df = pd.read_csv(os.path.join(path, exp, model, metric_dir_in_model_dir, file))
                metrics_mean[file_end] = float(df.loc[df['Metric']==metric]['Mean'].values[0])
                metrics_std[file_end] = float(df.loc[df['Metric']==metric]['Std'].values[0])
            tmp_str = '\t\t' + name + ' ' + map_dir2name[exp]
            for i in range(num_subgraphs_plus_sc):
                if metrics_mean[i] == '-':
                    tmp_str += f' & -'
                else:
                    tmp_str += f' & {metrics_mean[i]:.3f} $\\pm$ {metrics_std[i]:.3f}'
            tmp_str += ' \\\\\n'

            metrics_mean = ['-']*num_subgraphs_plus_sc
            avg_metrics_files = [file for file in os.listdir(os.path.join(path, exp, model, metric_dir_in_model_dir, 'mean')) if is_average_metrics_file(file, mean=True)]
            for file in avg_metrics_files:
                file_end = map_avgfile2index[[key for key in map_avgfile2index.keys() if file.endswith(key)][0]]
                if exp in dir_is_only_train:
                    file_end = -1
                df = pd.read_csv(os.path.join(path, exp, model, metric_dir_in_model_dir, 'mean', file))
                metrics_mean[file_end] = float(df.loc[df['Metric']==metric]['Mean'].values[0])
            tmp_str += '\t\t' + name + ' ' + map_dir2name[exp] + ' merged'
            for i in range(num_subgraphs_plus_sc):
                if metrics_mean[i] == '-':
                    tmp_str += f' & -'
                else:
                    tmp_str += f' & {metrics_mean[i]:.3f}'
            tmp_str += ' \\\\\n'
            if name == lin:
                lin = tmp_str
            elif name == ffw:
                ffw = tmp_str
            else:
                gat = tmp_str
        table += lin + ffw + gat
    table += end_str
    with open(os.path.join(path, metric+'_table.txt'), mode='w') as f:
        f.write(table)

def create_tables(path):
    for metric in metrics:
        create_table(path, metric)

create_tables(path)