# %%
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Batch, Data
import warnings
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')
from collections import OrderedDict
import glob

# %%
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter

# %%
def mols2graphs(complex_path, label, save_path, dis_threshold=5.):
    print(complex_path)
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p = mol2graph(pocket)
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
    try:
        edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
    except:
        raise ValueError(f"Error in {complex_path}")
    y = torch.FloatTensor([label])
    pos = torch.concat([pos_l, pos_p], dim=0)
    split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
    
    data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)

    torch.save(data, save_path)
    # return data

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, dis_threshold=5, num_process=8, create=False, mode='train', fold=0, total_folds=3):

        self.data_dir = data_dir
        self.dis_threshold = dis_threshold
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self.mode = mode
        self.fold = fold
        self.total_folds = total_folds
        self._make_key()
        self._pre_process()

    def _make_key(self):
        
        valid_keys = glob.glob('./all/*')
        valid_keys = [v.split('/')[-1] for v in valid_keys]
        dude_gene = list(OrderedDict.fromkeys([v.split('_')[0] for v in valid_keys]))
        test_fold = "tryb1_2zebA_full mcr_2oaxE_full bace1_3h0bA_full cxcr4_3oduA_full thb_1q4xA_full andr_2hvcA_full rxra_3ozjA_full esr2_2fszA_full mmp13_2pjtC_full pparg_2i4zA_full esr1_3dt3B_full prgr_3kbaA_full hivpr_1mtbA_full reni_3g6zB_full fa10_2p16A_full dpp4_2i78A_full adrb2_3ny8A_full fa7_1wqvH_full ppara_2p54A_full thrb_1ypeH_full ada17_2fv5A_full ace_3bklA_full urok_1sqtA_full gcr_3bqdA_full drd3_3pblA_full aa2ar_3emlA_full lkha4_3ftxA_full try1_2zq1A_full ppard_2znpA_full cp2c9_1r9oA_full cp3a4_1w0fA_full casp3_1rhrB_full adrb1_2vt4A_full "
        test_list = test_fold.split()
        test_dude_gene = []
        for item in test_list:
            test_dude_gene.append(item.split("_")[0])
        train_dude_gene = [p for p in dude_gene if p not in test_dude_gene]
        self.train_keys = [k for k in valid_keys if k.split('_')[0] in train_dude_gene]
        self.test_keys = [k for k in valid_keys if k.split('_')[0] in test_dude_gene]

        total_len = len(self.train_keys)
        fold_size = total_len // self.total_folds
        start_idx = self.fold * fold_size
        end_idx = start_idx + fold_size if self.fold != self.total_folds - 1 else total_len

        if self.mode == 'train':
            self.keys = self.train_keys[:start_idx] + self.train_keys[end_idx:]  # Two folds for training
        elif self.mode == 'valid':
            self.keys = self.train_keys[start_idx:end_idx]  # One fold for validation
        elif self.mode == 'test':
            self.keys = self.test_keys

    def _pre_process(self):

        complex_path_list = []
        complex_id_list = []
        y_list = []
        graph_path_list = []
        act_or_dec_list = ['actives', 'decoys']

        for sub_dir in os.listdir(self.data_dir):
            if sub_dir not in self.keys:
                continue
            for act_or_dec in act_or_dec_list:
                complex_dir = os.path.join(self.data_dir, sub_dir)
                for i in os.listdir(os.path.join(complex_dir, act_or_dec)):
                    complex_path = os.path.join(complex_dir, f"{act_or_dec}/{i}/{act_or_dec}_final_{self.dis_threshold}A.rdkit")
                    graph_path = os.path.join(complex_dir, f"{act_or_dec}/{i}/{act_or_dec}_final_{self.dis_threshold}A.pyg")
                    if not os.path.exists(complex_path):
                        continue
                    complex_path_list.append(complex_path)
                    complex_id_list.append(i)
                    y_list.append(0 if act_or_dec == 'decoys' else 1)
                    graph_path_list.append(graph_path)

        self.len = len(complex_path_list)
        if self.create:
            dis_thresholds = repeat(self.dis_threshold, len(complex_path_list))
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, y_list, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx])

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        return self.len

if __name__ == '__main__':
    data_dir = 'all'
    train_dataset = GraphDataset(data_dir, dis_threshold=5, create=True, mode='train')
    train_loader = PLIDataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_dataset = GraphDataset(data_dir, dis_threshold=5, create=True, mode='test')
    test_loader = PLIDataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

# %%
