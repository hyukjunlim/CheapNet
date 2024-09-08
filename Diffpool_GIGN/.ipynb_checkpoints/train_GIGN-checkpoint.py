
# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from GIGN import GIGN
from dataset_GIGN import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from torch_geometric.utils import batched_negative_sampling
from parallel import DataParallelModel, DataParallelCriterion
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
    model.train()

    return rmse, coff

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    
    for repeat in range(repeats):
        args['repeat'] = repeat

        train_dir = os.path.join(data_root, 'train')
        valid_dir = os.path.join(data_root, 'valid')
        test2013_dir = os.path.join(data_root, 'test2013')
        test2016_dir = os.path.join(data_root, 'test2016')
        test2019_dir = os.path.join(data_root, 'test2019')

        train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
        test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
        test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
        test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))

        train_set = GraphDataset(train_dir, train_df, graph_type=graph_type, create=False)
        valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False)
        test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
        test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
        test2019_set = GraphDataset(test2019_dir, test2019_df, graph_type=graph_type, create=False)

        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test2013_loader = PLIDataLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test2019_loader = PLIDataLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=4)

        lr = 3e-4
        logger = TrainLogger(args, cfg, create=True)
        # logger.info(f"GIGN default, lr={lr}")
        # logger.info(f"IAM GIGN experiment: concat((q-k)^2, q, k), i->j,r, qkv=qkv, lr={lr}")
        logger.info(f"IAM GIGN experiment: ijk->selfatt, only at featsblock3, lr={lr}")
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2013 data: {len(test2013_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")
        logger.info(f"test2019 data: {len(test2019_set)}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = GIGN(35, 256)
        # model = DataParallelModel(model)
        model.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler_bool = False
        if scheduler_bool:
            scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=5, start_factor=1, end_factor=0.6666666666)
        criterion = nn.MSELoss()
        # criterion = DataParallelCriterion(criterion)

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        
        model.train()
        iters = len(train_loader)
        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                data = data.to(device)
                pred = model(data)
                
                label = data.y
                MSE_loss = criterion(pred, label)
                loss = MSE_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0)) 

            
            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse, valid_pr = val(model, valid_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    test2013_rmse, test2013_pr = val(model, test2013_loader, device)
                    test2016_rmse, test2016_pr = val(model, test2016_loader, device)
                    test2019_rmse, test2019_pr = val(model, test2019_loader, device)
                    msg_train = f"Validation : valid_rmse-{valid_rmse:.4f}, valid_pr-{valid_pr:.4f}, \ntest2013_rmse-{test2013_rmse:.4f}, test2013_pr-{test2013_pr:.4f}, test2016_rmse-{test2016_rmse:.4f}, test2016_pr-{test2016_pr:.4f}, test2019_rmse-{test2019_rmse:.4f}, test2019_pr-{test2019_pr:.4f}"
                    logger.info(msg_train)
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
                    
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break
                
            
            if scheduler_bool:
                scheduler.step()
            
        # final testing
        load_model_dict(model, best_model_list[-1])
        valid_rmse, valid_pr = val(model, valid_loader, device)
        test2013_rmse, test2013_pr = val(model, test2013_loader, device)
        test2016_rmse, test2016_pr = val(model, test2016_loader, device)
        test2019_rmse, test2019_pr = val(model, test2019_loader, device)
        msg_test = f"valid_rmse-{valid_rmse:.4f}, valid_pr-{valid_pr:.4f}, test2013_rmse-{test2013_rmse:.4f}, test2013_pr-{test2013_pr:.4f}, test2016_rmse-{test2016_rmse:.4f}, test2016_pr-{test2016_pr:.4f}, test2019_rmse-{test2019_rmse:.4f}, test2019_pr-{test2019_pr:.4f}"
        logger.info(msg_test)
        

# %%