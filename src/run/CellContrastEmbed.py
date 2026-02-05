import torch
from src.data.CellContrastData import EmbedDataset
from src.models.CellContrastModel import ContrastiveLearning
from src.utils.utils import load
from src.utils.utils import set_seed

def embed(**args):
    """
    Embed visual representations of cells.

    Parameters:
    image_dir (str): Path to dir in which torch.tensors of cell cut outs are
    model_name (str): Path and name of model torch save dict
    args (dict): Arguments
    """

    batch_size = args['batch_size']
    seed = args['seed']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    train_dataset = EmbedDataset(split='train',
                                 save_embed_data=True,
                                 **args)
    test_dataset = EmbedDataset(split='test',
                                save_embed_data=True,
                                **args)

    model = ContrastiveLearning(channels=train_dataset.img_shape[0],
                                **args).to(device, torch.float32)
    model.load_state_dict(load(args['output_name'], save_keys='model', device=device))
    model.eval()
    model.mode = 'embed'

    train_dataset.save_embed_data(model, device=device, batch_size=batch_size)
    test_dataset.save_embed_data(model, device=device, batch_size=batch_size)
