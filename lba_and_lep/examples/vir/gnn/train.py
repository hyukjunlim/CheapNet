import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils import AverageMeter
from model import GIGN
from DUDE_dataset import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def predict(model, dataloader, device):

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

    model.train()

    return pred, label

def get_roce(pred_list, target_list, roce_rate):

    p = sum(target_list)
    n = len(target_list) - p
    pred_list = [[index, x] for index, x in enumerate(pred_list)]
    pred_list = sorted(pred_list, key=lambda x:x[1], reverse=True)
    tp1 = 0
    fp1 = 0
    for x in pred_list:
        if(target_list[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roce_rate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    
    return roce

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
    lr = args.get("lr")
    fold = args.get("fold")

    for rep in range(10):
        seed_always = []
        seed_random = []
        l = [i for i in range(1000) if i not in seed_always]
        # for repeat, seed in enumerate(seed_always + list(np.random.choice(seed_random, size=3-len(seed_always), replace=False))):
        for repeat, seed in enumerate(np.random.choice(l, size=3, replace=False)):
        # for repeat, seed in enumerate(np.random.randint(0, 1000, size=3)):
            seed_everything(seed)
            save_dir = f"./model/{rep}"
            msg_info = f"lr={lr}, seed={seed}"
            args['repeat'] = repeat

            dataset = GraphDataset(data_root, dis_threshold=5, create=False, mode='train')
            test_dataset = GraphDataset(data_root, dis_threshold=5, create=False, mode='test')
            test_loader = PLIDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            logger = TrainLogger(args, cfg, save_dir, create=True)
            logger.info(msg_info)
            logger.info(__file__)
            

            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            num_clusters = [31, 117]
            model = GIGN(35, 256, num_clusters)
            model.cuda()
            logger.info(f"GIGN params # : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
            # if scheduler_bool:
                # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
            criterion = nn.BCELoss()
            
            for fold_idx in range(3):
                if fold_idx != fold:
                    continue
                
                train_dataset = GraphDataset(data_root, dis_threshold=5, create=False, mode='train', fold=fold_idx)
                valid_dataset = GraphDataset(data_root, dis_threshold=5, create=False, mode='valid', fold=fold_idx)

                train_loader = PLIDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                valid_loader = PLIDataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                logger.info(f"fold-{fold_idx}")
                logger.info(f"train data: {len(train_loader)}")
                logger.info(f"valid data: {len(valid_loader)}")
                logger.info(f"test data: {len(test_loader)}")   

                running_loss = AverageMeter()
                running_acc = AverageMeter()
                running_best_auc = BestMeter("max")
                best_model_list = []
                
                model.train()
                iters = len(train_loader)

                # maxnum = 0
                # maxnum_lig = 0
                # maxnum_pro = 0
                # for j in [train_loader, test_loader]:
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

                #             maxnum_lig = max(maxnum_lig, unique_nodes_lig.size(0))
                #             maxnum_pro = max(maxnum_pro, unique_nodes_pro.size(0))
                #             maxnum = max(maxnum, unique_nodes_lig.size(0) + unique_nodes_pro.size(0))
                #             sum_intra += unique_nodes_lig.size(0) + unique_nodes_pro.size(0)
                #             sum_lig += unique_nodes_lig.size(0)
                #             sum_pro += unique_nodes_pro.size(0)
                #             lig_list.append(unique_nodes_lig.size(0))
                #             pro_list.append(unique_nodes_pro.size(0))
                #     print(sum_intra / batch_size / len(j))
                #     print(sum_lig / batch_size / len(j))
                #     print(sum_pro / batch_size / len(j))
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
                        BCE_loss = criterion(pred, label)
                        loss = BCE_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        running_loss.update(loss.item(), label.size(0)) 
                    
                    epoch_loss = running_loss.get_average()
                    running_loss.reset()

                    # Validation phase
                    y_pred, y_label = predict(model, valid_loader, device)
                    valid_loss_value = criterion(torch.FloatTensor(y_pred), torch.FloatTensor(y_label)).item()
                    roce_0_5 = get_roce(y_pred, y_label, 0.5)
                    roce_1 = get_roce(y_pred, y_label, 1)
                    roce_2 = get_roce(y_pred, y_label, 2)
                    roce_5 = get_roce(y_pred, y_label, 5)
                    AUROC = roc_auc_score(y_label, y_pred)
                    msg_valid = f"epoch-{epoch}, train_loss-{epoch_loss:.4f}, valid_loss-{valid_loss_value:.4f}, AUC-{AUROC:.4f}, ROCE_0_5-{roce_0_5:.4f}, ROCE_1-{roce_1:.4f}, ROCE_2-{roce_2:.4f}, ROCE_5-{roce_5:.4f}"
                    logger.info(msg_valid)
                    
                    if AUROC > running_best_auc.get_best():
                        running_best_auc.update(AUROC)
                        if save_model:
                            y_pred, y_label = predict(model, test_loader, device)
                            roce_0_5 = get_roce(y_pred, y_label, 0.5)
                            roce_1 = get_roce(y_pred, y_label, 1)
                            roce_2 = get_roce(y_pred, y_label, 2)
                            roce_5 = get_roce(y_pred, y_label, 5)
                            AUROC = roc_auc_score(y_label, y_pred)
                            msg_train = f"Test: AUC-{AUROC:.4f}, ROCE_0_5-{roce_0_5:.4f}, ROCE_1-{roce_1:.4f}, ROCE_2-{roce_2:.4f}, ROCE_5-{roce_5:.4f}"
                            print(msg_train)
                            msg = f"epoch-{epoch}, train_loss-{epoch_loss:.4f}, valid_loss-{valid_loss_value:.4f}"
                            # logger.info(msg)
                            model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                            best_model_list.append(model_path)
                            save_model_dict(model, logger.get_model_dir(), msg)
                            
                    else:
                        count = running_best_auc.counter()
                        if count > early_stop_epoch:
                            best_mse = running_best_auc.get_best()
                            msg = "best_rmse: %.4f" % best_mse
                            logger.info(f"early stop in epoch {epoch}")
                            logger.info(msg)
                            break_flag = True
                            break
                    
                # final testing
                load_model_dict(model, best_model_list[-1])
                y_pred, y_label = predict(model, test_loader, device)
                roce_0_5 = get_roce(y_pred, y_label, 0.5)
                roce_1 = get_roce(y_pred, y_label, 1)
                roce_2 = get_roce(y_pred, y_label, 2)
                roce_5 = get_roce(y_pred, y_label, 5)
                AUROC = roc_auc_score(y_label, y_pred)
                msg_test = f"test_AUC-{AUROC:.4f}, ROCE_0_5-{roce_0_5:.4f}, ROCE_1-{roce_1:.4f}, ROCE_2-{roce_2:.4f}, ROCE_5-{roce_5:.4f}"
                logger.info(msg_test)
                
