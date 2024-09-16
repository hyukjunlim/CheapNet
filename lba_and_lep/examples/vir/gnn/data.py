import os
import pickle
from collections import OrderedDict
import random
import glob
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
from Bio.PDB import *
import deepchem
import pickle
from torch_geometric.data import Data, Batch  # Import PyG Data and Batch classes
import pymol
import os
import gzip
import shutil
from tqdm import tqdm

def unzip_files():
    # Loop through all the directories and subdirectories
    for root, dirs, files in os.walk('./all'):
        # If the file is already unzipped, skip it
        if 'actives_final.mol2' in files and 'decoys_final.mol2' in files:
            continue
        for file in files:
            if file in ['actives_final.mol2.gz', 'decoys_final.mol2.gz']:
                file_path = os.path.join(root, file)
                output_file_path = file_path[:-3] 
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f'Unzipped: {file_path} to {output_file_path}')
  

# unzip_files()
valid_keys = glob.glob('./all/*')
print(valid_keys)
valid_keys = [v.split('/')[-1] for v in valid_keys]

dude_gene = list(OrderedDict.fromkeys([v.split('_')[0] for v in valid_keys]))
test_fold = "tryb1_2zebA_full mcr_2oaxE_full bace1_3h0bA_full cxcr4_3oduA_full thb_1q4xA_full andr_2hvcA_full rxra_3ozjA_full esr2_2fszA_full mmp13_2pjtC_full pparg_2i4zA_full esr1_3dt3B_full prgr_3kbaA_full hivpr_1mtbA_full reni_3g6zB_full fa10_2p16A_full dpp4_2i78A_full adrb2_3ny8A_full fa7_1wqvH_full ppara_2p54A_full thrb_1ypeH_full ada17_2fv5A_full ace_3bklA_full urok_1sqtA_full gcr_3bqdA_full drd3_3pblA_full aa2ar_3emlA_full lkha4_3ftxA_full try1_2zq1A_full ppard_2znpA_full cp2c9_1r9oA_full cp3a4_1w0fA_full casp3_1rhrB_full adrb1_2vt4A_full "
test_list = test_fold.split()
test_dude_gene = []
for item in test_list:
    test_dude_gene.append(item.split("_")[0])
print(test_dude_gene)
train_dude_gene = [p for p in dude_gene if p not in test_dude_gene]
print(len(train_dude_gene))
train_keys = [k for k in valid_keys if k.split('_')[0] in train_dude_gene]
test_keys = [k for k in valid_keys if k.split('_')[0] in test_dude_gene]

node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')

zero = np.eye(2)[1]
one = np.eye(2)[0]

with open("receptor_pdb_dict.pkl", 'rb') as fp:
    receptor_dict = pickle.load(fp)

def make_train(train_keys):
    counter = 0
    for key in train_keys:
        try:
        # active
            with open(f"./all/{key}/actives_final.mol2", 'r') as fr:
                actives_string = fr.read()
            divider = "@<TRIPOS>MOLECULE"
            split_act = actives_string.split(divider)
            actives_string_list = [divider + i for i in split_act if i.strip()]
            for i, act in enumerate(actives_string_list):
                save_path = f"./all/{key}/actives/{i}/actives_final.mol2"
                if not os.path.exists(save_path):
                    os.makedirs(f"./all/{key}/actives/{i}", exist_ok=True)
                    with open(save_path, 'w') as fw:
                        fw.write(act)


        # inactive
            with open(f"./all/{key}/decoys_final.mol2", 'r') as fr:
                inactives_string = fr.read()
            divider = "@<TRIPOS>MOLECULE"
            split_inact = inactives_string.split(divider)
            inactives_string_list = [divider + i for i in split_inact if i.strip()]
            for i, inact in enumerate(inactives_string_list):
                save_path = f"./all/{key}/decoys/{i}/decoys_final.mol2"
                if not os.path.exists(save_path):
                    os.makedirs(f"./all/{key}/decoys/{i}", exist_ok=True)
                    with open(save_path, 'w') as fw:
                        fw.write(inact)

            print(f'done for {key}')
        # generate_pocket(distance, key, cid)
        # generate_complex('act', distance=distance)
        # generate_complex('inact', distance=distance)

        except Exception as e:
            print(e)
            counter += 1
            continue

    print('Num Invalid: ', counter)
    print('Num actual train keys: ', len(train_keys))


def make_test(test_keys):
    counter = 0
    for key in test_keys:
        try:
        # active
            with open(f"./all/{key}/actives_final.mol2", 'r') as fr:
                actives_string = fr.read()
            divider = "@<TRIPOS>MOLECULE"
            split_act = actives_string.split(divider)
            actives_string_list = [divider + i for i in split_act if i.strip()]
            for i, act in enumerate(actives_string_list):
                save_path = f"./all/{key}/actives/{i}/actives_final.mol2"
                if not os.path.exists(save_path):
                    os.makedirs(f"./all/{key}/actives/{i}", exist_ok=True)
                    with open(save_path, 'w') as fw:
                        fw.write(act)
        
        # inactive
            with open(f"./all/{key}/decoys_final.mol2", 'r') as fr:
                inactives_string = fr.read()
            divider = "@<TRIPOS>MOLECULE"
            split_inact = inactives_string.split(divider)
            inactives_string_list = [divider + i for i in split_inact if i.strip()]
            for i, inact in enumerate(inactives_string_list):
                save_path = f"./all/{key}/decoys/{i}/decoys_final.mol2"
                if not os.path.exists(save_path):
                    os.makedirs(f"./all/{key}/decoys/{i}", exist_ok=True)
                    with open(save_path, 'w') as fw:
                        fw.write(inact)
            
            print(f'done for {key}')
        except Exception as e:
            print(e)
            counter += 1
            continue

    print('Num Invalid: ', counter)
    print('Num actual test keys: ', len(test_keys))

# make_train(train_keys)
# make_test(test_keys)

# TODO: Generate pocket and complex for train set

def generate_pocket(distance, protein_path, lig_native_path, key, cid):
    complex_dir = os.path.join('./all', key)
    
    pymol.cmd.load(protein_path)
    pymol.cmd.remove('resn HOH')
    pymol.cmd.load(lig_native_path)
    pymol.cmd.remove('hydrogens')
    pymol.cmd.select('Pocket', f'byres {cid}_ligand around {distance}')
    pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
    pymol.cmd.delete('all')

def generate_complex(cid, distance=5, input_ligand_format='mol2'):
    complex_dir = os.path.join('./all', cid)
    pocket_path = os.path.join('./all', cid, f'Pocket_{distance}A.pdb')
    if input_ligand_format != 'pdb':
        ligand_input_path = os.path.join('./all', cid, f'{cid}_ligand.{input_ligand_format}')
        ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
        os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
    else:
        ligand_path = os.path.join('./all', cid, f'{cid}_ligand.pdb')

    save_path = os.path.join(complex_dir, f"{cid}_{distance}A.rdkit")
    ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
    if ligand == None:
        print(f"Unable to process ligand of {cid}")

    pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
    if pocket == None:
        print(f"Unable to process protein of {cid}")

    complex = (ligand, pocket)
    with open(save_path, 'wb') as f:
        pickle.dump(complex, f)

def process_pocket(pdb_pro, pdb_lig):

    # # ligand
    m = Chem.MolFromPDBFile(pdb_lig)
    am = GetAdjacencyMatrix(m)
    n1 = m.GetNumAtoms()
    c1 = m.GetConformers()[0]
    d1 = np.array(c1.GetPositions())
    H, pos = get_atom_feature([(m.GetAtoms()[i], d1[i]) for i in range(n1)])
    g = nx.convert_matrix.from_numpy_matrix(am)
    edge_index = torch.tensor(list(g.edges)).t().contiguous()
    pyg_graph_lig = Data(x=torch.tensor(H, dtype=torch.float), edge_index=edge_index, pos=pos)

    # protein
    m = Chem.MolFromPDBFile(pdb_pro)
    am = GetAdjacencyMatrix(m)
    pockets = pk.find_pockets(pdb_pro)
    n2 = m.GetNumAtoms()
    c2 = m.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    binding_parts = []
    not_in_binding = [i for i in range(0, n2)]
    constructed_graphs = []
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        binding_parts_atoms = []
        idxs = []
        for idx, atom_cord in enumerate(d2):
            if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                binding_parts_atoms.append((m.GetAtoms()[idx], atom_cord))
                idxs.append(idx)
                if idx in not_in_binding:
                    not_in_binding.remove(idx)

        ami = am[np.array(idxs)[:, None], np.array(idxs)]
        H, pos = get_atom_feature(binding_parts_atoms)
        g = nx.convert_matrix.from_numpy_matrix(ami)
        edge_index = torch.tensor(list(g.edges)).t().contiguous()
        pyg_graph = Data(x=torch.tensor(H, dtype=torch.float), edge_index=edge_index, pos=pos)
        constructed_graphs.append(pyg_graph)
        binding_parts.append(binding_parts_atoms)

    constructed_graphs = Batch.from_data_list(constructed_graphs)

    return binding_parts, not_in_binding, constructed_graphs

for key in train_keys:
    # get all the mol2 files ends with 'actives_final_*.mol2'
    actives = glob.glob(f"./all/{key}/actives_final_*.mol2")
    actives_list = [a.split('/')[-1] for a in actives]
    actives_list = [a.split('.')[0] for a in actives_list]
    actives_list = [a.split('_')[-1] for a in actives_list]
    actives_list = [a.split('_')[0] for a in actives_list]
    
    for a in actives_list:
        protein_path = f"./all/{key}/receptor.pdb"
        lig_native_path = f"./all/{key}/actives_final_{a}.mol2"
        _, _, constructed_graphs = process_pocket(protein_path, lig_native_path)
        print(key, a, constructed_graphs)




# with open('train_new_dude_balanced_all2_active.pkl', 'wb') as f:
#     pickle.dump(actives, f)
# with open('train_new_dude_balanced_all2_decoy.pkl', 'wb') as f:
#     pickle.dump(inactive, f)


# TODO: Generate pocket and complex for train set

# with open('test_new_dude_all_active_none_pdb.pkl', 'wb') as f:
#     pickle.dump(actives, f)
# with open('test_new_dude_all_decoy_none_pdb.pkl', 'wb') as f:
#     pickle.dump(inactive, f)