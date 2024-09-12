import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from utils import AverageMeter
from GIGN import GIGN
from dataset_GIGN import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from GIGN import scheduler_bool, lr, explain, num_clusters, only_rep
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
# from torch.utils.tensorboard import SummaryWriter
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    # writer.add_scalar('valid rmse', rmse, epoch)
    model.train()

    return rmse, coff

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    

        
    for rep in range(10):
        seed_always = [418, 714, 444]
        seed_random = []
        l = [i for i in range(1000) if i not in seed_always]
        # for repeat, seed in enumerate(seed_always + list(np.random.choice(seed_random, size=3-len(seed_always), replace=False))):
        for repeat, seed in enumerate(np.random.choice(l, size=3, replace=False)):
        # for repeat, seed in enumerate(np.random.randint(0, 1000, size=3)):
            if only_rep is not None and repeat not in only_rep:
                continue
            else:
                pass   
            seed_everything(seed)
            save_dir = f"./model/{explain}_{rep}-1"
            msg_info = f"{explain}, lr={lr}, seed={seed}"
        
            # # writer = SummaryWriter()
            
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

            logger = TrainLogger(args, cfg, save_dir, create=True)
            logger.info(msg_info)
            logger.info(__file__)
            logger.info(f"train data: {len(train_set)}")
            logger.info(f"valid data: {len(valid_set)}")
            logger.info(f"test2013 data: {len(test2013_set)}")
            logger.info(f"test2016 data: {len(test2016_set)}")
            logger.info(f"test2019 data: {len(test2019_set)}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model = GIGN(35, 256, num_clusters)
            model.cuda()
            logger.info(f"GIGN params # : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
            if scheduler_bool:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
                # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=50, cycle_mult=1, max_lr=lr, min_lr=0, warmup_steps=5, gamma=0.9)

            criterion = nn.MSELoss()

            running_loss = AverageMeter()
            running_acc = AverageMeter()
            running_best_mse = BestMeter("min")
            best_model_list = []
            
            model.train()
            iters = len(train_loader)

            # maxnum = 0
            # maxnum_lig = 0
            # maxnum_pro = 0
            # for j in [train_loader, valid_loader, test2013_loader, test2016_loader, test2019_loader]:
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
            #     print(sum_intra / 128 / len(j))
            #     print(sum_lig / 128 / len(j))
            #     print(sum_pro / 128 / len(j))
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
            
            for epoch in range(epochs):
                for batch_idx, data in enumerate(train_loader):
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
                if scheduler_bool:
                    scheduler.step(valid_rmse)

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
                
            # final testing
            load_model_dict(model, best_model_list[-1])
            valid_rmse, valid_pr = val(model, valid_loader, device)
            test2013_rmse, test2013_pr = val(model, test2013_loader, device)
            test2016_rmse, test2016_pr = val(model, test2016_loader, device)
            test2019_rmse, test2019_pr = val(model, test2019_loader, device)
            msg_test = f"valid_rmse-{valid_rmse:.4f}, valid_pr-{valid_pr:.4f}, test2013_rmse-{test2013_rmse:.4f}, test2013_pr-{test2013_pr:.4f}, test2016_rmse-{test2016_rmse:.4f}, test2016_pr-{test2016_pr:.4f}, test2019_rmse-{test2019_rmse:.4f}, test2019_pr-{test2019_pr:.4f}"
            logger.info(msg_test)
            
            
            # writer.close()
            
