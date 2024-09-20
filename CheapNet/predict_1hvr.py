
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
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            label = data.y
            print(pred, label)

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    rmse = np.sqrt(mean_squared_error(label, pred))
    mae = mean_absolute_error(label, pred)

    model.train()

    return rmse, mae
    
# %%
data_root = './data'
graph_type = 'Graph_GIGN'
batch_size = 128

for red_node in [1]:

    casestudy_dir = os.path.join(data_root, 'casestudy')
    casestudy_df = pd.read_csv(os.path.join(data_root, 'casestudy.csv'))
    casestudy_set = GraphDataset(casestudy_dir, casestudy_df, graph_type=graph_type, create=False)
    casestudy_loader = PLIDataLoader(casestudy_set, batch_size=batch_size, shuffle=False, num_workers=4)
    columns = ['Model', '1hvr RMSE', '1hvr MAE']
    results_df = pd.DataFrame(columns=columns)

    models = ['GIGN']
    model_root = f'save/g-d-c/q2q2/ours-lrs-0.001-28-156_0'
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

            casestudy_rmse, casestudy_mae = val(model, casestudy_loader, device)

            new_row = {
                'Model': md.split('/')[2] + " | " + md[-1],
                'casestudy RMSE': casestudy_rmse,
                'casestudy MAE': casestudy_mae
            }

            new_row_df = pd.DataFrame([new_row])

            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
# 
    metrics = ['casestudy RMSE', 'casestudy MAE']
    mean_values = results_df[metrics].mean()
    std_values = results_df[metrics].std()
    results_df = results_df.append(pd.Series(['Mean'] + list(mean_values), index=results_df.columns), ignore_index=True)
    results_df = results_df.append(pd.Series(['Std'] + list(std_values), index=results_df.columns), ignore_index=True)
    results_df.to_csv(f'results.csv', index=False)
    print(results_df)