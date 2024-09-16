import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pymol import cmd
if cmd._COb is None:
    import pymol2
    import pymol.invocation
    pymol2.SingletonPyMOL().start()
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)
distance = 5
base_dir = 'all'  
input_ligand_format = 'mol2'

flag = 'check'
# flag = 'generate'
# 
if flag == 'generate':
    for sub_dir in ['cp3a4']:
    # for sub_dir in os.listdir(base_dir):
        complex_dir = os.path.join(base_dir, sub_dir)
        print(f'Processing {complex_dir}')
        act_or_dec_list = ['actives', 'decoys']

        # num_actives = len(os.listdir(os.path.join(complex_dir, 'actives')))
        # num_decoys = len(os.listdir(os.path.join(complex_dir, 'decoys')))
        # # under sample decoys
        # if num_decoys > num_actives:
        #     decoy_list = os.listdir(os.path.join(complex_dir, 'decoys'))
        #     random.shuffle(decoy_list)
        #     decoy_list = decoy_list[:num_actives]
        for act_or_dec in act_or_dec_list:
            success = 0
            for i in os.listdir(os.path.join(complex_dir, act_or_dec)):
                # if num_decoys > num_actives and act_or_dec == 'decoys' and i not in decoy_list:
                #     continue
                if os.path.exists(os.path.join(complex_dir, f"{act_or_dec}/{i}/Pocket_{distance}A.pdb")):
                    print(f"Pocket already exists for {complex_dir}/{act_or_dec}/{i}")
                    continue
                lig_native_path = os.path.join(complex_dir, f"{act_or_dec}/{i}/{act_or_dec}_final.mol2")
                protein_path= os.path.join(complex_dir, f"receptor.pdb")
                cmd.load(protein_path, 'protein')
                cmd.remove('resn HOH')
                cmd.load(lig_native_path, 'ligand')
                cmd.remove('hydrogens')
                cmd.select('Pocket', f'byres ligand around {distance}')
                cmd.save(os.path.join(complex_dir, f'{act_or_dec}/{i}/Pocket_{distance}A.pdb'), 'Pocket')
                cmd.delete('all')

                pocket_path = os.path.join(complex_dir, f'{act_or_dec}/{i}/Pocket_{distance}A.pdb')
                if input_ligand_format != 'pdb':
                    ligand_input_path = os.path.join(complex_dir, f'{act_or_dec}/{i}/{act_or_dec}_final.{input_ligand_format}')
                    ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
                    os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
                else:
                    ligand_path = os.path.join(complex_dir, f'{act_or_dec}/{i}/{act_or_dec}_final.pdb')

                save_path = os.path.join(complex_dir, f"{act_or_dec}/{i}/{act_or_dec}_final_{distance}A.rdkit")
                ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
                if ligand == None:
                    print(f"Unable to process ligand of {complex_dir}/{act_or_dec}/{i}")
                    continue

                pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
                if pocket == None:
                    print(f"Unable to process protein of {complex_dir}/{act_or_dec}/{i}")
                    continue

                if os.path.exists(save_path):
                    success += 1

                if success > len(os.listdir(os.path.join(complex_dir, 'actives'))):
                    break

                complex = (ligand, pocket)
                with open(save_path, 'wb') as f:
                    pickle.dump(complex, f)

                
elif flag == 'check': # check if the pocket is generated
    total_count = 0
    for sub_dir in os.listdir(base_dir):
        complex_dir = os.path.join(base_dir, sub_dir)
        yes_count = 0
        yes_list = []
        no_count = 0
        for i in os.listdir(os.path.join(complex_dir, 'decoys')):
            rdkit_path = os.path.join(complex_dir, f'decoys/{i}/decoys_final_{distance}A.rdkit')
            pocket_path = os.path.join(complex_dir, f'decoys/{i}/Pocket_{distance}A.pdb')
            # total_count += len(os.listdir(os.path.join(complex_dir, 'decoys')))
            if os.path.exists(rdkit_path) and os.path.exists(pocket_path):
                # remove the directory if the rdkit file is not generated
                # os.system(f'rm -rf {os.path.join(complex_dir, "decoys", i)}')
                yes_count += 1
                total_count += 1
                # print(f"Removed {os.path.join(complex_dir, 'decoys', i)}")
                yes_list.append(i)
            elif not os.path.exists(rdkit_path) and os.path.exists(pocket_path):
                no_count += 1
                total_count += 1
        # if yes_count > len(os.listdir(os.path.join(complex_dir, 'actives'))):
        #     print(f'{complex_dir}: {yes_count} pockets generated, {no_count} pockets not generated, vs {len(os.listdir(os.path.join(complex_dir, "actives")))} actives')
        #     num_actives = len(os.listdir(os.path.join(complex_dir, 'actives')))
        #     num_decoys = len(os.listdir(os.path.join(complex_dir, 'decoys')))
        #     # print(f"Removing {len(yes_list) - num_actives} decoys")
        #     random.shuffle(yes_list)
        #     yes_list = yes_list[:num_actives]
        #     for i in os.listdir(os.path.join(complex_dir, 'decoys')):
        #         if i not in yes_list:
        #             os.system(f'rm -rf {os.path.join(complex_dir, "decoys", i)}')
        #             # print(f"Removed {os.path.join(complex_dir, 'decoys', i)}")
        if yes_count == len(os.listdir(os.path.join(complex_dir, 'actives'))):
            print(f'{complex_dir}: {yes_count} pockets generated, {no_count} pockets not generated vs {len(os.listdir(os.path.join(complex_dir, "actives")))} actives')
# import os
# import pickle
# from rdkit import Chem
# from tqdm import tqdm
# import pymol
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')
# from pymol import cmd
# if cmd._COb is None:
#     import pymol2
#     import pymol.invocation

#     pymol.invocation.parse_args(['pymol', '-q'])
#     pymol2.SingletonPyMOL().start()

# distance = 5
# base_dir = 'all'  # base directory containing multiple folders like 'xiap'
# input_ligand_format = 'mol2'

# # Iterate over all directories in 'all'
#     complex_dir = os.path.join(base_dir, sub_dir)
    
#     # Process all 'mol2' files in each subdirectory
#     for filename in os.listdir(complex_dir):
#         if filename.endswith(f".{input_ligand_format}") and 'actives_final_' in filename:
#             file_index = filename.split('_')[-1].split('.')[0]  # Extract the index of the file
#             ligand_native_path = os.path.join(complex_dir, filename)
#             protein_path = os.path.join(complex_dir, "receptor.pdb")

#             # Create output directories like all/xiap/actives/0/
#             output_dir = complex_dir 
#             # output_dir = os.path.join(complex_dir, 'actives', file_index)
#             # os.makedirs(output_dir, exist_ok=True)

#             # Load protein and ligand
#             cmd.load(protein_path)
#             cmd.remove('resn HOH')
#             cmd.load(ligand_native_path)
#             cmd.remove('hydrogens')
#             print(protein_path, ligand_native_path, file_index)
#             cmd.select('Pocket', f'byres actives_final_{file_index} around {distance}')
#             pocket_pdb_path = os.path.join(output_dir, f'Pocket_{distance}A_{file_index}.pdb')
#             cmd.save(pocket_pdb_path, 'Pocket')
#             cmd.delete('all')
#             print(f'Pocket saved for {pocket_pdb_path}')

#             # Convert ligand to PDB format if it's not already
#             if input_ligand_format != 'pdb':
#                 ligand_input_path = ligand_native_path
#                 ligand_pdb_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
#                 os.system(f'obabel {ligand_input_path} -O {ligand_pdb_path} -d')
#             else:
#                 ligand_pdb_path = ligand_native_path

#             # Save RDKit complex (ligand + pocket)
#             save_path = os.path.join(output_dir, f"{file_index}_{distance}A.rdkit")
#             ligand = Chem.MolFromPDBFile(ligand_pdb_path, removeHs=True)
#             if ligand is None:
#                 print(f"Unable to process ligand of {file_index}")
#                 continue

#             pocket = Chem.MolFromPDBFile(pocket_pdb_path, removeHs=True)
#             if pocket is None:
#                 print(f"Unable to process protein of {file_index}")
#                 continue

#             complex_mol = (ligand, pocket)
#             with open(save_path, 'wb') as f:
#                 pickle.dump(complex_mol, f)
#             print(f'RDKit complex saved for {save_path}')