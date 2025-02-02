import os
import sys
sys.path.append(os.path.abspath('../atom3d'))
import torch
import argparse
from torch_geometric.data import DataLoader
from model import CheapNet
from data import GNNTransformLBA
from atom3d.datasets import LMDBDataset, PTGDataset
from scipy.stats import spearmanr
import numpy as np
import torch.nn.functional as F

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--GPU_NUM', type=int, default=0)
    parser.add_argument('--precomputed', type=bool, default=True)
    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    # Set data directory
    if args.data_dir is None:
      args.data_dir = f'dataset/split-by-sequence-identity-{args.seqid}/data'

    # Load test data
    if args.precomputed:
        test_dataset = PTGDataset(os.path.join(args.data_dir, 'test'))
    else:
        transform=GNNTransformLBA()
        test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    # Load model
    rmse_list = []
    pearson_list = []
    spearman_list = []
    for i in range(3):
      model_dir = f'LBA_{args.seqid}_best_models'
      model_path = os.path.join(model_dir, f'best_weights_rep{i}.pt')  # Adjust rep accordingly
      num_features = test_dataset[0].num_features
      num_clusters = [25, 372] if args.seqid == 30 else [24, 362]
      
      model = CheapNet(num_features, hidden_dim=args.hidden_dim, num_clusters=num_clusters).to(device)
      best_model = torch.load(model_path)
      model.load_state_dict(best_model['model_state_dict'])

      # Test the model
      rmse, pearson, spearman, y_true, y_pred = test(model, test_loader, device)
      print(f"Model {i+1} | RMSE: {rmse:.3f}, Pearson: {pearson:.3f}, Spearman: {spearman:.3f}")
      rmse_list.append(rmse)
      pearson_list.append(pearson)
      spearman_list.append(spearman)
    rmse_avg = np.mean(rmse_list)
    pearson_avg = np.mean(pearson_list)
    spearman_avg = np.mean(spearman_list)
    rmse_std = np.std(rmse_list)
    pearson_std = np.std(pearson_list)
    spearman_std = np.std(spearman_list)
    print(f"Average | RMSE: {rmse_avg:.3f} ± {rmse_std:.3f}, Pearson: {pearson_avg:.3f} ± {pearson_std:.3f}, Spearman: {spearman_avg:.3f} ± {spearman_std:.3f}")
    