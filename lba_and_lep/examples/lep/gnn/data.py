import numpy as np
import sys
import os
import torch
from tqdm import tqdm
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform, PairedGraphTransform
sys.path.append(os.path.abspath('/data/project/dlagurwns03/GIGN/codes/lba_and_lep'))
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Batch, DataLoader
import atom3d.util.graph as gr
from atom3d.filters import filters

    
class GNNTransformLEP(object):
    def __init__(self, label_key):
        self.label_key = label_key
    
    def __call__(self, item):
        active = item['atoms_active']
        inactive = item['atoms_inactive']
        # print num of atoms in active and inactive whose chain is not 'L'
        item['protein_active'] = active[active.chain != 'L']
        item['protein_inactive'] = inactive[inactive.chain != 'L']
        item['ligand_active'] = active[active.chain == 'L']
        item['ligand_inactive'] = inactive[inactive.chain == 'L']
        filter1 = filters.distance_filter
        item['protein_active'] = filter1(item['protein_active'], item['ligand_active'][['x','y','z']], 6)
        item['protein_inactive'] = filter1(item['protein_inactive'], item['ligand_inactive'][['x','y','z']], 6)
        # transform protein and/or pocket to PTG graphs
        item = prot_graph_transform(item, atom_keys=['protein_active', 'protein_inactive'], label_key=self.label_key)
        item = mol_graph_transform(item, atom_key='ligand_active', label_key=self.label_key)
        item = mol_graph_transform(item, atom_key='ligand_inactive', label_key=self.label_key)
        
        node_feats, intra_edges, inter_edges, intra_edge_attr, inter_edge_attr, node_pos, split = gr.combine_graphs(item['protein_active'], item['ligand_active'], edges_between=True)
        edges = torch.cat([intra_edges, inter_edges], dim=1)
        combined_active = Data(node_feats, edges, intra_edge_attr=intra_edge_attr, inter_edge_attr=inter_edge_attr, y=item[self.label_key], pos=node_pos, edge_index_intra=intra_edges, edge_index_inter=inter_edges, split=split)
        
        node_feats, intra_edges, inter_edges, intra_edge_attr, inter_edge_attr, node_pos, split = gr.combine_graphs(item['protein_inactive'], item['ligand_inactive'], edges_between=True)
        combined_inactive = Data(node_feats, edges, intra_edge_attr=intra_edge_attr, inter_edge_attr=inter_edge_attr, y=item[self.label_key], pos=node_pos, edge_index_intra=intra_edges, edge_index_inter=inter_edges, split=split)
        
        return combined_active, combined_inactive

class CollaterLEP(object):
    """To be used with pre-computed graphs and atom3d.datasets.PTGDataset"""
    def __init__(self):
        pass
    def __call__(self, data_list):
        batch_1 = Batch.from_data_list([d[0] for d in data_list])
        batch_2 = Batch.from_data_list([d[1] for d in data_list])
        return batch_1, batch_2
    

        
if __name__=="__main__":
    save_dir = '/data/project/dlagurwns03/GIGN/atom3d/examples/lep/gnn/dataset'
    data_dir = '/data/project/dlagurwns03/GIGN/atom3d/dataset_lep/split-by-protein/data'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    transform = GNNTransformLEP(label_key='label')
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=transform)
    
    print('processing train dataset...')
    for i, item in enumerate(tqdm(train_dataset)):
        torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))
    
    print('processing validation dataset...')
    for i, item in enumerate(tqdm(val_dataset)):
        torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))
    
    print('processing test dataset...')
    for i, item in enumerate(tqdm(test_dataset)):
        torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))