
# %%
import os
import pandas as pd
import torch
from GIGN import GIGN
from etc.org_GIGN import org_GIGN
from dataset_GIGN import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))
    mae = mean_absolute_error(label, pred)

    model.train()

    return rmse, coff, mae
    
# %%
data_root = './data'
graph_type = 'Graph_GIGN'
batch_size = 128

for red_node in [1]:
    valid_dir = os.path.join(data_root, 'valid')
    test2013_dir = os.path.join(data_root, 'test2013')
    test2016_dir = os.path.join(data_root, 'test2016')
    test2019_dir = os.path.join(data_root, 'test2019')
    csar_dir = os.path.join(data_root, 'csar')
    casf2016_dir = os.path.join(data_root, 'casf2016')

    valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
    test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
    test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
    test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))
    csar_df = pd.read_csv(os.path.join(data_root, 'csar.csv'))
    casf2016_df = pd.read_csv(os.path.join(data_root, 'casf2016.csv'))

    valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False)
    test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
    test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
    test2019_set = GraphDataset(test2019_dir, test2019_df, graph_type=graph_type, create=False)
    csar_set = GraphDataset(csar_dir, csar_df, graph_type=graph_type, create=False)
    casf2016_set = GraphDataset(casf2016_dir, casf2016_df, graph_type=graph_type, create=False)

    valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test2013_loader = PLIDataLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test2019_loader = PLIDataLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=4)
    csar_loader = PLIDataLoader(csar_set, batch_size=batch_size, shuffle=False, num_workers=4)
    casf2016_loader = PLIDataLoader(casf2016_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # columns = ['Model', 'Test2013 RMSE', 'Test2013 R', 'Test2016 RMSE', 'Test2016 R', 'Test2019 RMSE', 'Test2019 R']
    columns = ['Model', 'Valid RMSE', 'Valid R', 'Valid MAE', 'Test2013 RMSE', 'Test2013 R', 'Test2013 MAE', 'Test2016 RMSE', 'Test2016 R', 'Test2016 MAE', 'Test2019 RMSE', 'Test2019 R', 'Test2019 MAE', 'CSAR RMSE', 'CSAR R', 'CSAR MAE', 'CASF2016 RMSE', 'CASF2016 R', 'CASF2016 MAE']
    results_df = pd.DataFrame(columns=columns)

    org = False


    models = ['GIGN']
    if org:
        model_root = '../GIGN/data_saved'
    else:
        # model_root = f'model/ours_0.2'
        model_root = f'save/ours-lrs-0.001-28-156_0'
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
            if org:
                model = org_GIGN(35, 256).to(device)
            else:
                model = GIGN(35, 256).to(device)
            load_model_dict(model, latest_epoch_model_path)
            model = model.cuda()

            valid_rmse, valid_coff, valid_mae = val(model, valid_loader, device)
            test2013_rmse, test2013_coff, test2013_mae = val(model, test2013_loader, device)
            test2016_rmse, test2016_coff, test2016_mae = val(model, test2016_loader, device)
            test2019_rmse, test2019_coff, test2019_mae = val(model, test2019_loader, device)
            csar_rmse, csar_coff, csar_mae = val(model, csar_loader, device)
            casf2016_rmse, casf2016_coff, casf2016_mae = val(model, casf2016_loader, device)

            # msg = "valid_rmse-%.4f, valid_r-%.4f, valid_mae-%.4f, test2013_rmse-%.4f, test2013_r-%.4f, test2013_mae-%.4f, test2016_rmse-%.4f, test2016_r-%.4f, test2016_mae-%.4f, test2019_rmse-%.4f, test2019_r-%.4f, test2019_mae-%.4f, csar_rmse-%.4f, csar_r-%.4f, csar_mae-%.4f" \
            #     % (valid_rmse, valid_coff, valid_mae, test2013_rmse, test2013_coff, test2013_mae, test2016_rmse, test2016_coff, test2016_mae, test2019_rmse, test2019_coff, test2019_mae, csar_rmse, csar_coff, csar_mae)

            # print(msg)
            
            new_row = {
                'Model': md.split('/')[2] + " | " + md[-1],
                'Valid RMSE': valid_rmse,
                'Valid R': valid_coff,
                'Valid MAE': valid_mae,
                'Test2013 RMSE': test2013_rmse,
                'Test2013 R': test2013_coff,
                'Test2013 MAE': test2013_mae,
                'Test2016 RMSE': test2016_rmse,
                'Test2016 R': test2016_coff,
                'Test2016 MAE': test2016_mae,
                'Test2019 RMSE': test2019_rmse,
                'Test2019 R': test2019_coff,
                'Test2019 MAE': test2019_mae,
                'CSAR RMSE': csar_rmse,
                'CSAR R': csar_coff,
                'CSAR MAE': csar_mae,
                'CASF2016 RMSE': casf2016_rmse,
                'CASF2016 R': casf2016_coff,
                'CASF2016 MAE': casf2016_mae,
            }

            new_row_df = pd.DataFrame([new_row])

            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
# 
    # metrics = ['Test2013 RMSE', 'Test2013 R', 'Test2016 RMSE', 'Test2016 R', 'Test2019 RMSE', 'Test2019 R']
    metrics = ['Valid RMSE', 'Valid R', 'Valid MAE', 'Test2013 RMSE', 'Test2013 R', 'Test2013 MAE', 'Test2016 RMSE', 'Test2016 R', 'Test2016 MAE', 'Test2019 RMSE', 'Test2019 R', 'Test2019 MAE', 'CSAR RMSE', 'CSAR R', 'CSAR MAE', 'CASF2016 RMSE', 'CASF2016 R', 'CASF2016 MAE']
    mean_values = results_df[metrics].mean()
    std_values = results_df[metrics].std()
    results_df = results_df.append(pd.Series(['Mean'] + list(mean_values), index=results_df.columns), ignore_index=True)
    results_df = results_df.append(pd.Series(['Std'] + list(std_values), index=results_df.columns), ignore_index=True)
    results_df.to_csv(f'results_org_{org}.csv', index=False)
    print(results_df)