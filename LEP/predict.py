import os
import sys
sys.path.append(os.path.abspath('../atom3d'))
import argparse
import numpy as np
import torch
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

@torch.no_grad()
def test(gcn_model, ff_model, loader, device):
    gcn_model.eval()
    ff_model.eval()

    y_true = []
    y_pred = []

    for active, inactive in loader:
        labels = torch.FloatTensor([a == 'A' for a in active.y]).to(device)
        active = active.to(device)
        inactive = inactive.to(device)
        out_active = gcn_model(active)
        out_inactive = gcn_model(inactive)
        output = ff_model(out_active, out_inactive)
        y_true.extend(labels.tolist())
        y_pred.extend(output.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    return auroc, auprc, y_true, y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=f"dataset")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--precomputed', type=bool, default=True)
    parser.add_argument('--GPU_NUM', type=int, default=0)
    parser.add_argument('--use_scheduler', type=int, default=0)
    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')

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

    for active, inactive in train_loader:
        num_features1 = active.num_features
        num_features2 = inactive.num_features
        assert num_features1 == num_features2
        break

    num_clusters = [49, 312]
    gcn_model = CheapNet(num_features1, hidden_dim=args.hidden_dim, num_clusters=num_clusters)
    gcn_model.to(device)
    ff_model = MLP_LEP(args.hidden_dim).to(device)
    
    # Load model
    auroc_list = []
    auprc_list = []
    for i in range(3):
        model_dir = f'LEP_best_models'
        model_path = os.path.join(model_dir, f'best_weights_rep{i}.pt')  # Adjust rep accordingly

        # Initialize models
        num_features1 = test_dataset[0][0].num_features
        num_clusters = [49, 312]
        gcn_model = CheapNet(num_features1, hidden_dim=args.hidden_dim, num_clusters=num_clusters).to(device)
        ff_model = MLP_LEP(args.hidden_dim).to(device)

        # Load the best saved model
        best_model = torch.load(model_path)
        gcn_model.load_state_dict(best_model['gcn_state_dict'])
        ff_model.load_state_dict(best_model['ff_state_dict'])

        # Test the model
        auroc, auprc, y_true, y_pred = test(gcn_model, ff_model, test_loader, device)
        print(f"Model {i+1} | AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")
        auroc_list.append(auroc)
        auprc_list.append(auprc)

    # Compute average and std
    auroc_avg = np.mean(auroc_list)
    auprc_avg = np.mean(auprc_list)
    auroc_std = np.std(auroc_list)
    auprc_std = np.std(auprc_list)
    print(f"Average | AUROC: {auroc_avg:.3f} ± {auroc_std:.3f}, AUPRC: {auprc_avg:.3f} ± {auprc_std:.3f}")
