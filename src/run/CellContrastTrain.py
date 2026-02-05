from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models.CellContrastModel import ContrastiveLearning
from src.data.CellContrastData import EmbedDataset
from src.loss.ContrastiveLoss import add_contrastive_loss
from src.optimizer.LARC import LARC
from src.utils.utils import set_seed

def train(**args):
    """
    Train Image model to learn visual representations.

    image_dir (str): Path to dir containing torch.tensor cell cutouts
    output_name (str): Path and name of torch save dict of model+metrics
    args (dict): Arguments
    """

    output_name = args['output_name']
    batch_size = args['batch_size']
    lr = args['lr']
    weight_decay = args['weight_decay']
    warmup_epochs = args['warmup_epochs']
    EPOCH = args['epochs']
    num_workers = args['num_workers']
    seed = args['seed']

    if EPOCH > 900: # Weird bug, when running for more then 982 epochs we get an recursion error
        import sys
        sys.setrecursionlimit(100000)

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args['deterministic']:
        set_seed(seed)

    train_dataset = EmbedDataset(split='train',
                                n_clusters=args['n_clusters_image'],
                                **args)
    test_dataset = EmbedDataset(split='test',
                                n_clusters=1,
                                **args)
    model = ContrastiveLearning(channels=train_dataset.img_shape[0],
                                **args).to(device, dtype=torch.float32)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=True,
                              pin_memory=True,
                              sampler=None if args['n_clusters_image'] <= 1 else WeightedRandomSampler(train_dataset.weight, train_dataset.__len__()))
    val_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=True,
                            pin_memory=True)

    #sc_lr = lr * batch_size / 256
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr,
                                weight_decay=weight_decay,
                                momentum=0.9)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=lr, 
                                                    epochs=EPOCH, 
                                                    steps_per_epoch=len(train_loader), 
                                                    pct_start=warmup_epochs/EPOCH,
                                                    div_factor=25,
                                                    final_div_factor=1e5)
    optimizer = LARC(optimizer, clip=False)
    loss = add_contrastive_loss

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in list(range(EPOCH)):
        running_loss = 0
        running_acc = 0
        model.train()

        with tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}") as train_loader:
            for idx, batch in enumerate(train_loader):
                batch = torch.cat((batch[0], batch[1])).to(device)
                optimizer.zero_grad()
                out = model(batch)
                l, logits, labels = loss(out)
                l.backward()
                optimizer.step()
                scheduler.step()
                running_loss += l.item()

                # Compute element-wise equality between the predicted labels and true labels
                contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
                # Convert the boolean tensor to float32 and compute the mean
                running_acc += torch.mean(contrast_acc.float()).item()

            train_acc = running_acc / len(train_loader)
            train_acc_list.append(train_acc)
            epoch_loss = running_loss / len(train_loader)
            train_loss_list.append(epoch_loss)
            print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        with torch.no_grad():
            running_loss = 0
            running_acc = 0
            model.eval()

            with tqdm(val_loader, total=len(val_loader), desc=f"Validation epoch {epoch}") as val_loader:
                for idx, batch in enumerate(val_loader):
                    batch = torch.cat((batch[0], batch[1])).to(device)
                    out = model(batch)
                    l, logits, labels = loss(out)
                    running_loss += l.item()

                    # Compute element-wise equality between the predicted labels and true labels
                    contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
                    # Convert the boolean tensor to float32 and compute the mean
                    running_acc += torch.mean(contrast_acc.float()).item()

                val_acc = running_acc / len(val_loader)
                val_acc_list.append(val_acc)
                epoch_loss = running_loss / len(val_loader)
                val_loss_list.append(epoch_loss)
                print(f"Val Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        if (epoch+1) % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "train_acc": train_acc_list,
                "train_list": train_loss_list,
                "val_acc": val_acc_list,
                "val_list": val_loss_list,
                "epoch": epoch
            }, output_name)

    torch.save({
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "train_acc": train_acc_list,
                "train_list": train_loss_list,
                "val_acc": val_acc_list,
                "val_list": val_loss_list,
                "epoch": epoch
            }, output_name)
