import os
import sys
sys.path.append(os.path.abspath('../atom3d'))
import argparse
import logging
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader as PTGDataLoader
from torch.utils.data import DataLoader
from model import CheapNet, MLP_LEP
from data import CollaterLEP
from atom3d.util.transforms import PairedGraphTransform
from atom3d.datasets import LMDBDataset, PTGDataset
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")

def train_loop(args, epoch, gcn_model, ff_model, loader, criterion, optimizer, scheduler, device):
    gcn_model.train()
    ff_model.train()

    losses = []
    for it, (active, inactive) in enumerate(loader):
        labels = torch.FloatTensor([a == 'A' for a in active.y]).to(device)
        active = active.to(device)
        inactive = inactive.to(device)
        optimizer.zero_grad()
        out_active = gcn_model(active)
        out_inactive = gcn_model(inactive)
        output = ff_model(out_active, out_inactive)
        loss = criterion(output, labels)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if args.use_scheduler:
            scheduler.step((epoch - 1) + it / len(loader))

    return np.mean(losses)


@torch.no_grad()
def test(gcn_model, ff_model, loader, criterion, device):
    gcn_model.eval()
    ff_model.eval()

    losses = []
    y_true = []
    y_pred = []

    for active, inactive in loader:
        labels = torch.FloatTensor([a == 'A' for a in active.y]).to(device)
        active = active.to(device)
        inactive = inactive.to(device)
        out_active = gcn_model(active)
        out_inactive = gcn_model(inactive)
        output = ff_model(out_active, out_inactive)
        loss = criterion(output, labels)
        losses.append(loss.item())
        y_true.extend(labels.tolist())
        y_pred.extend(output.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    return np.mean(losses), auroc, auprc, y_true, y_pred

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None, test_mode=False):
    
    transform = PairedGraphTransform('atoms_active', 'atoms_inactive', label_key='label')
    if args.precomputed:
        train_dataset = PTGDataset(os.path.join(args.data_dir, 'train'))
        val_dataset = PTGDataset(os.path.join(args.data_dir, 'val'))
        test_dataset = PTGDataset(os.path.join(args.data_dir, 'test'))
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, collate_fn=CollaterLEP())
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=CollaterLEP())
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=CollaterLEP())
    else:
        train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=transform)
        val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=transform)
        test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=transform)
        train_loader = PTGDataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
        val_loader = PTGDataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
        test_loader = PTGDataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)
    print(f'Total samples : {len(train_dataset) + len(val_dataset) + len(test_dataset)}')

    for active, inactive in train_loader:
        num_features1 = active.num_features
        num_features2 = inactive.num_features
        assert num_features1 == num_features2
        break

    num_clusters = [49, 312]
    gcn_model = CheapNet(num_features1, hidden_dim=args.hidden_dim, num_clusters=num_clusters)
    gcn_model.to(device)
    ff_model = MLP_LEP(args.hidden_dim).to(device)
    num_params_gcn = sum(p.numel() for p in gcn_model.parameters() if p.requires_grad)
    num_params_ff = sum(p.numel() for p in ff_model.parameters() if p.requires_grad)
    num_params = num_params_gcn + num_params_ff
    print(F'CheapNet params # : {num_params} ({num_params_gcn} + {num_params_ff})')
    logger.info(f"CheapNet params # : {num_params}({num_params_gcn}, {num_params_ff})")

    best_val_loss = 999

    check_quantiles(train_loader, val_loader, test_loader)

    params = [x for x in gcn_model.parameters()] + [x for x in ff_model.parameters()]
    criterion = nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-6)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    else:
        scheduler = None
    count = 0
    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(args, epoch, gcn_model, ff_model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_auroc, val_auprc, _, _ = test(gcn_model, ff_model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            count = 0
            torch.save({
                'epoch': epoch,
                'gcn_state_dict': gcn_model.state_dict(),
                'ff_state_dict': ff_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
            print(f'Best model saved at epoch {epoch} with val loss: {val_loss:.7f}')
            logger.info(f'Best model saved at epoch {epoch} with val loss: {val_loss:.7f}')
            best_val_loss = val_loss
            if test_mode:
                train_file = os.path.join(log_dir, f'lep-rep{rep}.best.train.pt')
                val_file = os.path.join(log_dir, f'lep-rep{rep}.best.val.pt')
                test_file = os.path.join(log_dir, f'lep-rep{rep}.best.test.pt')
                cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
                gcn_model.load_state_dict(cpt['gcn_state_dict'])
                ff_model.load_state_dict(cpt['ff_state_dict'])
                test_loss, test_auroc, test_auprc, y_true_test, y_pred_test = test(gcn_model, ff_model, test_loader, criterion, device)
                print(f'\tTest loss {test_loss}, Test AUROC {test_auroc}, Test AUPRC {test_auprc}')
        else:
            count += 1
            if count > args.early_stop_patience and epoch > 200:
                print(f'Early stopping')
                logger.info(f'Early stopping')
                break
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed), end=', ')
        print(f'Train loss {train_loss:.7f}, Val loss {val_loss:.7f}, Val AUROC {val_auroc:.7f}, Val AUPRC {val_auprc:.7f}')
        logger.info(f'Epoch: {epoch}, Train loss {train_loss:.7f}, Val loss {val_loss:.7f}, Val AUROC {val_auroc:.7f}, Val AUPRC {val_auprc:.7f}')

    if test_mode:
        train_file = os.path.join(log_dir, f'lep-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'lep-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'lep-rep{rep}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        gcn_model.load_state_dict(cpt['gcn_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        _, _, _, y_true_train, y_pred_train = test(gcn_model, ff_model, train_loader, criterion, device)
        torch.save({'targets':y_true_train, 'predictions':y_pred_train}, train_file)
        _, _, _, y_true_val, y_pred_val = test(gcn_model, ff_model, val_loader, criterion, device)
        torch.save({'targets':y_true_val, 'predictions':y_pred_val}, val_file)
        test_loss, auroc, auprc, y_true_test, y_pred_test = test(gcn_model, ff_model, test_loader, criterion, device)
        print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test auprc {auprc}')
        logger.info(f'Test loss {test_loss}, Test AUROC {auroc}, Test auprc {auprc}')
        torch.save({'targets':y_true_test, 'predictions':y_pred_test}, test_file)
            
        return test_loss, auroc, auprc

    return best_val_loss

def check_quantiles(train_loader, val_loader, test_loader):
    lig_list = []
    pro_list = []
    total_list = []
    for j in [test_loader, val_loader, train_loader]:
        for i, (data, data2) in enumerate(j):
            for i in range(data.batch.max().item() + 1):
                mask = data.batch[data.edge_index_intra[0, :]] == i
                mask_lig = data.split[data.edge_index_intra[0, :]] == 0
                mask_pro = data.split[data.edge_index_intra[0, :]] == 1
                edge_index_lig = data.edge_index_intra[:, mask & mask_lig]
                edge_index_pro = data.edge_index_intra[:, mask & mask_pro]
                unique_nodes_lig = torch.unique(edge_index_lig)
                unique_nodes_pro = torch.unique(edge_index_pro)

                lig_list.append(unique_nodes_lig.size(0))
                pro_list.append(unique_nodes_pro.size(0))
                total_list.append(unique_nodes_lig.size(0) + unique_nodes_pro.size(0))

                mask = data2.batch[data2.edge_index_intra[0, :]] == i
                mask_lig = data2.split[data2.edge_index_intra[0, :]] == 0
                mask_pro = data2.split[data2.edge_index_intra[0, :]] == 1
                edge_index_lig = data2.edge_index_intra[:, mask & mask_lig]
                edge_index_pro = data2.edge_index_intra[:, mask & mask_pro]
                unique_nodes_lig = torch.unique(edge_index_lig)
                unique_nodes_pro = torch.unique(edge_index_pro)

                lig_list.append(unique_nodes_lig.size(0))
                pro_list.append(unique_nodes_pro.size(0))
                total_list.append(unique_nodes_lig.size(0) + unique_nodes_pro.size(0))

    lig_list = np.array(lig_list)
    q1 = np.percentile(lig_list, 25)
    q2 = np.percentile(lig_list, 50)
    q3 = np.percentile(lig_list, 75)
    q4 = np.percentile(lig_list, 100)
    avg = np.mean(lig_list)
    std = np.std(lig_list)
    print(f'LIG: {q1}, {q2}, {q3}, {q4}, {avg:.2f}, {std:.2f}')

    pro_list = np.array(pro_list)
    q1 = np.percentile(pro_list, 25)
    q2 = np.percentile(pro_list, 50)
    q3 = np.percentile(pro_list, 75)
    q4 = np.percentile(pro_list, 100)
    avg = np.mean(pro_list)
    std = np.std(pro_list)
    print(f'PRO: {q1}, {q2}, {q3}, {q4}, {avg:.2f}, {std:.2f}')

    total_list = np.array(total_list)
    q1 = np.percentile(total_list, 25)
    q2 = np.percentile(total_list, 50)
    q3 = np.percentile(total_list, 75)
    q4 = np.percentile(total_list, 100)
    avg = np.mean(total_list)
    std = np.std(total_list)
    print(f'TOTAL: {q1}, {q2}, {q3}, {q4}, {avg:.2f}, {std:.2f}')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=f"dataset")
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--precomputed', type=bool, default=True)
    parser.add_argument('--early_stop_patience', type=int, default=100)
    parser.add_argument('--GPU_NUM', type=int, default=0)
    parser.add_argument('--use_scheduler', type=int, default=0)
    parser.add_argument('--seed_set', type=int, default=0)
    parser.add_argument('--rep', type=int, default=None)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(args, device, log_dir)
        
    elif args.mode == 'test':
        for repeat in range(100):
            seed_random = []
            seed_always = [758, 657, 263]
            if args.seed_set:
                iter_list = seed_always + list(np.random.choice(seed_random, size=3-len(seed_always), replace=False))
            else:
                l = [i for i in range(1000) if i not in seed_always]
                iter_list = np.random.choice(l, size=3, replace=False)
            for rep, seed in enumerate(iter_list):
                if args.rep is not None and rep != args.rep:
                    continue
                log_dir = os.path.join('logs', f'lep_test_{repeat}_{args.GPU_NUM}_{rep}')
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                logger = logging.getLogger('lep')
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler(os.path.join(log_dir, f'log_rep{rep}.txt'))
                logger.addHandler(fh)
                print('seed:', seed)
                logger.info(f'seed: {seed}')
                seed_everything(seed)
                train(args, device, log_dir, rep, test_mode=True)
                logger.removeHandler(fh)
