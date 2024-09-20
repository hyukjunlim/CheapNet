
# %%
import os
import pandas as pd
import torch
from CheapNet import CheapNet
from dataset_CheapNet import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

from glob import glob
from collections import defaultdict
import re

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# %%

def val(model, dataloader, device, cluster_idx):
    model.eval()

    pred_list = []
    label_list = []
    cluster_dict = defaultdict(list)  # Dictionary to store predictions and labels by cluster

    for idx, data in enumerate(dataloader):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            label = data.y

            cluster_id = cluster_idx[idx]
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
            cluster_dict[cluster_id].append((pred.item(), label.item()))

    # Calculate Spearman coefficients for each cluster and average over all clusters
    spearman_per_cluster = []
    for cluster_id, values in cluster_dict.items():
        predictions, labels = zip(*values)
        spearman = spearmanr(labels, predictions)[0]  # Spearman for this cluster
        spearman_per_cluster.append(spearman)

    avg_spearman = np.mean(spearman_per_cluster)  # Average Spearman over 57 samples

    model.train()
    
    return avg_spearman



# %%
data_root = './data'
graph_type = 'Graph_GIGN'
batch_size = 128

for red_node in [1]:
    data_df = pd.read_csv("CoreSet.dat", delim_whitespace=True)
    cluster_idx = data_df['target'].values
    casf2016_dir = os.path.join(data_root, 'casf2016')
    casf2016_df = pd.read_csv(os.path.join(data_root, 'casf2016.csv'))
    casf2016_set = GraphDataset(casf2016_dir, casf2016_df, graph_type=graph_type, create=False)
    casf2016_loader = PLIDataLoader(casf2016_set, batch_size=1, shuffle=False, num_workers=4)

    # columns = ['Model', 'CSAR RMSE', 'CSAR R', 'CSAR Spea']
    columns = ['Model', 'CASF2016 Spea']
    # columns = ['Model', 'Valid RMSE', 'Valid R', 'Valid Spea', 'Test2013 RMSE', 'Test2013 R', 'Test2013 Spea', 'Test2016 RMSE', 'Test2016 R', 'Test2016 Spea', 'Test2019 RMSE', 'Test2019 R', 'Test2019 Spea', 'CSAR RMSE', 'CSAR R', 'CSAR Spea', 'CASF2016 RMSE', 'CASF2016 R', 'CASF2016 Spea']
    results_df = pd.DataFrame(columns=columns)

    org = False


    models = ['CheapNet']
    model_root = 'save/g-d-c/q2q2/ours-lrs-0.001-28-156_0'
    device = torch.device('cuda:0')

    # Regular expression to match epoch files
    epoch_file_pattern = re.compile(r'epoch-(\d+)')

    for model_name in models:
        model_dir = sorted(glob(os.path.join(model_root, f'*repeat*')))
        for md in model_dir:
            folder = os.path.join(md, 'model')
            model_list = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            model_list = [f for f in model_list if epoch_file_pattern.match(f)]
            model_list.sort(key=lambda x: int(epoch_file_pattern.search(x).group(1)), reverse=True)
            if model_list:
                latest_epoch_model = model_list[0]
                latest_epoch_model_path = os.path.join(folder, latest_epoch_model)
            model = CheapNet(35, 256).to(device)
            load_model_dict(model, latest_epoch_model_path)
            model = model.cuda()

            casf2016_spea = val(model, casf2016_loader, device, cluster_idx)
            print(md)

            new_row = {
                'Model': md.split('/')[1] + " | " + md[-1],
                'CASF2016 Spea': casf2016_spea,
            }


            new_row_df = pd.DataFrame([new_row])

            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
# 
    metrics = ['CASF2016 Spea']
    mean_values = results_df[metrics].mean()
    std_values = results_df[metrics].std()
    results_df = results_df.append(pd.Series(['Mean'] + list(mean_values), index=results_df.columns), ignore_index=True)
    results_df = results_df.append(pd.Series(['Std'] + list(std_values), index=results_df.columns), ignore_index=True)
    results_df.to_csv(f'results_org_{org}.csv', index=False)
    print(results_df)