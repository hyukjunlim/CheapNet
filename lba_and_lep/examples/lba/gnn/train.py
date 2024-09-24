import argparse
import logging
import os
import time
import datetime
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model import GIGN, explain
from pre_model import GNN_LBA
from data import GNNTransformLBA
from atom3d.datasets import LMDBDataset, PTGDataset
from scipy.stats import spearmanr
import warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")

def train_loop(args, model, loader, optimizer, scheduler, epoch, device):
    model.train()

    loss_all = 0
    total = 0
    for it, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()

    return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        output = model(data)
        loss = F.mse_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        y_true.extend(data.y.tolist())
        y_pred.extend(output.tolist())

    r_p = np.corrcoef(y_true, y_pred)[0,1]
    r_s = spearmanr(y_true, y_pred)[0]

    return np.sqrt(loss_all / total), r_p, r_s, y_true, y_pred

# def plot_corr(y_true, y_pred, plot_dir):
#     plt.clf()
#     sns.scatterplot(y_true, y_pred)
#     plt.xlabel('Actual -log(K)')
#     plt.ylabel('Predicted -log(K)')
#     plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None, test_mode=False):

    if args.precomputed:
        train_dataset = PTGDataset(os.path.join(args.data_dir, 'train'))
        val_dataset = PTGDataset(os.path.join(args.data_dir, 'val'))
        test_dataset = PTGDataset(os.path.join(args.data_dir, 'test'))
    else:
        transform=GNNTransformLBA()
        train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=transform)
        val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=transform)
        test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    for data in train_loader:
        num_features = data.num_features
        break
    
    num_clusters = [25, 372] if args.seqid == 30 else [24, 362]
    model = GNN_LBA(num_features, hidden_dim=64)
    # model = GIGN(num_features, hidden_dim=args.hidden_dim, num_clusters=num_clusters).to(device)
    model.to(device)
    print(F'GIGN params # : {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    logger.info(f"GIGN params # : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    best_val_loss = 999
    best_rp = 0
    best_rs = 0

    # maxnum = 0
    # maxnum_lig = 0
    # maxnum_pro = 0
    # for j in [test_loader, val_loader, train_loader]:
    #     sum_intra = 0
    #     sum_lig = 0
    #     sum_pro = 0
    #     lig_list = []
    #     pro_list = []
    #     for i, data in enumerate(j):
    #         data = data.to(device)
    #         for i in range(data.batch.max().item() + 1):
                
    #             mask = data.batch[data.edge_index_intra[0, :]] == i
    #             mask_lig = data.split[data.edge_index_intra[0, :]] == 0
    #             mask_pro = data.split[data.edge_index_intra[0, :]] == 1
    #             comb_lig = mask & mask_lig
    #             comb_pro = mask & mask_pro
    #             edge_index_lig = data.edge_index_intra[:, comb_lig]
    #             edge_index_pro = data.edge_index_intra[:, comb_pro]
    #             unique_nodes_lig = torch.unique(edge_index_lig)
    #             unique_nodes_pro = torch.unique(edge_index_pro)

    #             # pocket_nodes = torch.unique(data.edge_index_inter)
    #             # intra_lig_edges = edge_index_lig[:, torch.isin(edge_index_lig[0, :], pocket_nodes) & torch.isin(edge_index_lig[1, :], pocket_nodes)]
    #             # intra_pro_edges = edge_index_pro[:, torch.isin(edge_index_pro[0, :], pocket_nodes) & torch.isin(edge_index_pro[1, :], pocket_nodes)]
    #             # edge_index_pocket = torch.cat([intra_lig_edges, data.edge_index_inter, intra_pro_edges], dim=1)
    #             # edge_index_pocket = edge_index_pocket[:, data.batch[edge_index_pocket[0, :]] == i]
    #             # unique_nodes_pocket = torch.unique(edge_index_pocket)

    #             maxnum_lig = max(maxnum_lig, unique_nodes_lig.size(0))
    #             maxnum_pro = max(maxnum_pro, unique_nodes_pro.size(0))
    #             maxnum = max(maxnum, unique_nodes_lig.size(0) + unique_nodes_pro.size(0))
    #             sum_intra += unique_nodes_lig.size(0) + unique_nodes_pro.size(0)
    #             sum_lig += unique_nodes_lig.size(0)
    #             sum_pro += unique_nodes_pro.size(0)
    #             lig_list.append(unique_nodes_lig.size(0))
    #             pro_list.append(unique_nodes_pro.size(0))
    #     print(sum_intra / args.batch_size / len(j))
    #     print(sum_lig / args.batch_size / len(j))
    #     print(sum_pro / args.batch_size / len(j))
    #     print(maxnum, maxnum_lig, maxnum_pro)
    #     lig_list = np.array(lig_list)
    #     q1 = np.percentile(lig_list, 25)
    #     q2 = np.percentile(lig_list, 50)
    #     q3 = np.percentile(lig_list, 75)
    #     q4 = np.percentile(lig_list, 100)
    #     print(f'LIG: {q1}, {q2}, {q3}, {q4}')
    #     pro_list = np.array(pro_list)
    #     q1 = np.percentile(pro_list, 25)
    #     q2 = np.percentile(pro_list, 50)
    #     q3 = np.percentile(pro_list, 75)
    #     q4 = np.percentile(pro_list, 100)
    #     print(f'PRO: {q1}, {q2}, {q3}, {q4}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    else:
        scheduler = None
    count = 0
    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(args, model, train_loader, optimizer, scheduler, epoch, device)
        val_loss, r_p, r_s, y_true, y_pred = test(model, val_loader, device)
        if args.use_scheduler:
            scheduler.step(val_loss)
        if val_loss < best_val_loss:
            count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
            # plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}.png'))
            print(f'Best model saved at epoch {epoch} with val rmse: {val_loss:.7f}')
            logger.info(f'Best model saved at epoch {epoch} with val rmse: {val_loss:.7f}')
            best_val_loss = val_loss
            best_rp = r_p
            best_rs = r_s
            if test_mode:
                train_file = os.path.join(log_dir, f'lba-rep{rep}.best.train.pt')
                val_file = os.path.join(log_dir, f'lba-rep{rep}.best.val.pt')
                test_file = os.path.join(log_dir, f'lba-rep{rep}.best.test.pt')
                cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
                model.load_state_dict(cpt['model_state_dict'])
                rmse, pearson, spearman, _, _ = test(model, test_loader, device)
                print(f'\tTest RMSE {rmse:.7f}, Test Pearson {pearson:.7f}, Test Spearman {spearman:.7f}')
        else:
            count += 1
            if count > args.early_stop_patience and epoch > 200:
                print("Early stopping")
                logger.info("Early stopping")
                break
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f}s'.format(epoch, elapsed), end=', ')
        print('Train RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(train_loss, val_loss, r_p, r_s))
        logger.info('Epoch: {:03d}, Train RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(epoch, train_loss, val_loss, r_p, r_s))
        # logger.info('{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, train_loss, val_loss, r_p, r_s))

    if test_mode:
        train_file = os.path.join(log_dir, f'lba-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'lba-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'lba-rep{rep}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        _, _, _, y_true_train, y_pred_train = test(model, train_loader, device)
        torch.save({'targets':y_true_train, 'predictions':y_pred_train}, train_file)
        _, _, _, y_true_val, y_pred_val = test(model, val_loader, device)
        torch.save({'targets':y_true_val, 'predictions':y_pred_val}, val_file)
        rmse, pearson, spearman, y_true_test, y_pred_test = test(model, test_loader, device)
        print(f'\tTest RMSE {rmse:.7f}, Test Pearson {pearson:.7f}, Test Spearman {spearman:.7f}')
        logger.info(f'Test RMSE {rmse:.7f}, Test Pearson {pearson:.7f}, Test Spearman {spearman:.7f}')
        torch.save({'targets':y_true_test, 'predictions':y_pred_test}, test_file)

    return best_val_loss, best_rp, best_rs

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=10e-4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', type=bool, default=True)
    parser.add_argument('--early_stop_patience', type=int, default=100)
    parser.add_argument('--GPU_NUM', type=int, default=None)
    parser.add_argument('--use_scheduler', type=int, default=1)
    parser.add_argument('--seed_set', type=int, default=0)
    parser.add_argument('--rep', type=int, default=None)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f'/data/project/dlagurwns03/GIGN/codes/lba_and_lep/examples/lba/gnn/dataset/split-by-sequence-identity-{args.seqid}/data'
    if args.GPU_NUM is None:
        args.GPU_NUM = 0 if args.seqid == 30 else 1

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime(f"%Y-%m-%d-%H-%M-%S-{args.seqid}")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(args, device, log_dir)
        
    elif args.mode == 'test':
        for repeat in range(100):
            if args.seed_set:
                if args.seqid == 30:
                    seed_random = []
                    seed_always = [370, 679, 261]
                elif args.seqid == 60:
                    seed_random = []
                    seed_always = [437, 245, 927]
                iter_list = seed_always + list(np.random.choice(seed_random, size=3-len(seed_always), replace=False))
            else:
                iter_list = np.random.randint(0, 1000, size=3)
            for rep, seed in enumerate(iter_list):
                if args.rep is not None and rep != args.rep:
                    continue
                else:
                    pass
                log_dir = os.path.join('logs', f'lba_test_withH_{args.seqid}_{explain}_{repeat}_{args.GPU_NUM}')
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                logger = logging.getLogger('lba')
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler(os.path.join(log_dir, f'log_rep{rep}.txt'))
                logger.addHandler(fh)
                print(f'seed: {seed}')
                logger.info(f'seed: {seed}')
                seed_everything(seed)
                train(args, device, log_dir, rep, test_mode=True)
                logger.removeHandler(fh)
