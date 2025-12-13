import torch
import numpy as np
import scanpy as sc
import pandas as pd
import os

def get_true_graph_expression_dict(path):
    """
    Create a dict and populate it with name of graphs and ROI expression.

    Parameters:
    path (str): Path to dir containing ROI graphs

    Return:
    dict: dict of file name keys and corresponding ROI expression
    """
    path = os.path.join(os.getcwd(), path)
    graph_paths = [p for p in os.listdir(path) if 'graph' in p and p.endswith('pt')]
    value_dict = {}
    for graph_p in graph_paths:
        graph = torch.load(os.path.join(path, graph_p), map_location='cpu', weights_only=False)
        value_dict[graph_p] = {'y': graph.y.numpy()}
        if 'Class' in graph.to_dict().keys():
            value_dict[graph_p]['cell_class'] = graph.Class
    return value_dict

def get_predicted_graph_expression(value_dict, path):
    """
    Add predicted ROI expression to value_dict.

    Parameters:
    value_dict: dict of file name keys and corresponding ROI expression
    path (str): Path to dir containing ROI graphs

    Return:
    dict: dict of file name keys and corresponding ROI expression(true and predicted)
    """
    path = os.path.join(os.getcwd(), path)
    roi_pred_paths = [p for p in os.listdir(path) if p.startswith('roi_pred')]
    for roi_pred_p in roi_pred_paths:
        value_dict[roi_pred_p.split('roi_pred_')[1]]['roi_pred'] = torch.load(os.path.join(path, roi_pred_p), 
                                                                            map_location='cpu',
                                                                            weights_only=False).squeeze().detach().numpy()
    return value_dict

def get_predicted_cell_expression(value_dict, path):
    """
    Add predicted sc ROI expression to value_dict, return cell shape metrics.

    Parameters:
    value_dict: dict of file name keys and corresponding ROI expression
    path (str): Path to dir containing ROI graphs

    Return:
    dict: dict of file name keys and corresponding ROI expression and predicted sc ROI expression
    tuple: Number of cells, genes/proteins per cell
    """
    path = os.path.join(os.getcwd(), path)
    cell_pred_paths = [p for p in os.listdir(path) if p.startswith('cell_pred')]
    num_cells = 0
    for roi_pred_p in cell_pred_paths:
        value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'] = torch.load(os.path.join(path, roi_pred_p),
                                                                                map_location='cpu',
                                                                                weights_only=False).squeeze().detach().numpy()
        num_cells += value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'].shape[0]
    cell_shapes = (num_cells, value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'].shape[-1])
    return value_dict, cell_shapes

def get_patient_ids(label_data, keys):
    """
    Calculate numpy array of IDs corresponding to sorted file names of ROIs(value_dict keys), and gene/protein names.

    Parameters:
    label_data (str): Name of .csv in data/raw/ containing label information of ROIs(IDs specificly)
    keys (list): List of value_dict keys, aka ROI graph file names

    Return:
    np.array: IDs of sorted value_dict keys
    np.array: Gene/Protein names
    """
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'raw', label_data), header=0, sep=',')
    IDs = np.array(df[~df.duplicated(subset=['ROI'], keep=False) | ~df.duplicated(subset=['ROI'], keep='first')].sort_values(by=['ROI'])['Patient_ID'].values)
    exps = df.columns.values[2:]

    if len(keys) != IDs.shape[0]:
        tmp = np.ndarray((len(keys)), dtype=object)
        for i_key in range(len(keys)):
            tmp[i_key] = str(df[df['ROI']==keys[i_key].split('graph_')[-1].split('.')[0]]['Patient_ID'].values[0])
        IDs = tmp
    return IDs, exps

def get_bulk_expression_of(value_dict, IDs, exps, key='y'):
    """
    Create scanpy.AnnData obj of ROI expressions(NOT SC!)

    Parameters:
    value_dict (dict): dict of file name keys and corresponding ROI expression and predicted sc ROI expression
    IDs (np.array): IDs of sorted value_dict keys
    exps (np.array): Gene/Protein names
    key (str): key of ROI expression to select

    Return:
    sc.AnnData: obj of ROI expression with corresponding file names, IDs,
    """
    rois = list(value_dict.keys())
    rois.sort()
    rois_np = np.array(rois)
    adata = sc.AnnData(np.zeros((len(rois), value_dict[rois[0]][key].shape[0])))
    adata.obs['ID'] = str(-1)
    adata.var_names = exps
    files = np.array([])

    i = 0
    for id in np.unique(IDs).tolist():  #TODO: when subgraphs what then? how to automate instead of manual label creation?
        id_map = IDs==id
        id_keys = rois_np[id_map].tolist()
        for id_key in id_keys:
            adata.X[i] = value_dict[id_key][key]
            adata.obs.loc[str(i), 'ID'] = str(id)
            files = np.concatenate((files, np.array([id_key])))
            i += 1
    adata.obs['files'] = files
    return adata

def combine_data(value_dict, IDs, exps, name, cell_shapes, path):
    """
    Create scanpy.AnnData obj of sc expressions.

    Parameters:
    value_dict (dict): dict of file name keys and corresponding ROI expression and predicted sc ROI expression
    IDs (np.array): IDs of sorted value_dict keys
    exps (np.array): Gene/Protein names
    name (str): name of .h5ad file to save scanpy.AnnData
    figure_dir (str): Path to save figures to
    cell_shapes (tuple): Tuple of ints(number of cells, number og gene/proteins)
    """
    rois = list(value_dict.keys())
    rois.sort() # VERY IMPORTANT!!! IDs correspond to sorted value_dict keys
    rois_np = np.array(rois)
    counts = np.empty(cell_shapes, dtype=np.float32)
    cell_class = None
    ids = np.array([])
    files = np.array([])

    i = 0
    num_cells = 0
    key = 'cell_pred'
    # Build up counts array, ids, files such that every cell has correct corresponding ids and files asociated with cell ROI
    for id in np.unique(IDs).tolist():
        id_map = IDs==id
        id_keys = rois_np[id_map].tolist()  # Selects all ROI names corresponding to id
        for id_key in id_keys:
            tmp_counts = value_dict[id_key][key]
            counts[num_cells:num_cells+tmp_counts.shape[0],:] = tmp_counts
            if num_cells != 0:
                if ('cell_class' in value_dict[id_key].keys()) and cell_class is not None:
                    cell_class = np.concatenate((cell_class, value_dict[id_key]['cell_class']))
            else:
                if ('cell_class' in value_dict[id_key].keys()):
                    cell_class = value_dict[id_key]['cell_class']
            ids = np.concatenate((ids, np.array([id]*value_dict[id_key][key].shape[0])))
            files = np.concatenate((files, np.array([id_key]*value_dict[id_key][key].shape[0])))
            num_cells += tmp_counts.shape[0]
            i += 1
    counts = np.array(counts)
    
    adata = sc.AnnData(counts)
    if cell_class is not None:
        cell_class = np.array(cell_class)
        adata.obs['cell_class'] = cell_class
    adata.obs['ID'] = ids
    adata.obs['files'] = files
    adata.obs['leiden'] = -1
    adata.var_names = exps
    adata.write(os.path.join(path, name+'_all.h5ad'))

def createH5AD(path,
               processed_dir='TMA1_processed',
               embed_dir='out/',
               label_data='label_data.csv',
               figure_dir='figures/',
               name='_cells',):
    """
    processed_dir (str): Dir name in data/processed/ containing torch_geometric graphs
    embed_dir (str): Path to dir containing predicted sc expression
    label_data (str): Name of .csv in data/raw/ containing label information of ROIs(IDs specificly)
    figure_dir (str): Path to save figures to
    name (str): name of .h5ad file to save/load scanpy.AnnData, in figure path name
    select_cells (int): Number of cells to analyse, if 0 select all, otherwise random specified subset
    """
    value_dict = get_true_graph_expression_dict(os.path.join('data/processed', processed_dir))
    value_dict = get_predicted_graph_expression(value_dict, embed_dir)
    value_dict, cell_shapes = get_predicted_cell_expression(value_dict, embed_dir)
    IDs, exps = get_patient_ids(label_data, list(value_dict.keys()))
    if not os.path.exists(figure_dir) and not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    combine_data(value_dict, IDs, exps, name, figure_dir, cell_shapes, path)
