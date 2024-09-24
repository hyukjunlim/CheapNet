import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import cairosvg
from rdkit.Chem.Draw import IPythonConsole
d = {0: ((2.789, 8.636, 16.129), 18), -1: ((3.14, 8.181, 14.863), 21), -2: ((4.718, 7.555, 14.943), 21), -3: ((4.847, 7.915, 16.613), 18), -4: ((3.759, 8.48, 17.115), 8), -5: ((2.335, 8.209, 13.632), 7), -6: ((2.62, 7.087, 12.757), 16), -7: ((1.677, 6.956, 11.786), 16), -8: ((0.595, 7.51, 11.861), 16), -9: ((2.165, 6.153, 10.792), 19), 10: ((1.386, 6.074, 9.513), 4), 11: ((1.335, 4.609, 9.02), 4), -12: ((0.481, 3.689, 9.894), 21), 13: ((-0.978, 4.164, 10.074), 4), 14: ((2.026, 7.026, 8.458), 4), -15: ((1.945, 8.483, 8.842), 17), 16: ((0.752, 9.189, 8.602), 25), 17: ((0.666, 10.553, 8.971), 25), -18: ((1.758, 11.191, 9.57), 7), -19: ((2.931, 10.483, 9.812), 7), 20: ((3.036, 9.129, 9.44), 25), -21: ((2.662, 4.094, 8.874), 19), -22: ((-1.804, 3.244, 10.999), 16), -23: ((-2.192, 1.875, 10.472), 7), -24: ((-3.268, 1.677, 9.616), 7), -25: ((-3.621, 0.405, 9.162), 7), 26: ((-2.869, -0.698, 9.6), 25), 27: ((-1.792, -0.533, 10.442), 25), 28: ((-1.457, 0.75, 10.873), 25), 29: ((-1.544, 4.265, 8.718), 25), -30: ((-4.244, 4.326, 7.075), 19), -31: ((-3.043, 5.197, 7.036), 19), 32: ((-2.51, 5.212, 8.488), 25), -33: ((-2.916, 5.949, 9.376), 8), -34: ((-3.398, 6.657, 6.656), 7), -35: ((-2.135, 7.512, 6.651), 7), -36: ((-4.049, 6.661, 5.257), 21), 37: ((-4.4, 3.398, 6.059), 25), 38: ((-5.573, 2.643, 6.126), 25), 39: ((-5.852, 1.72, 5.029), 25), 40: ((-3.611, 3.333, 5.119), 25), -41: ((-6.112, 0.321, 5.488), 16), -42: ((-5.126, -0.568, 5.842), 21), -43: ((-5.839, -2.045, 6.317), 21), -44: ((-7.399, -1.407, 6.024), 21), -45: ((-7.394, -0.188, 5.584), 16), -46: ((-8.587, -2.304, 6.29), 10), 47: ((-9.081, -2.971, 5.023), 4), -48: ((-8.279, -3.308, 7.407), 10), 49: ((-6.59, 2.806, 7.159), 4)}
sdf_file = 'data/casf2016/3prs/3prs_ligand.sdf'
save_path = 'figure/3prs_hitatt.png'

# mol_supplier = Chem.SDMolSupplier(sdf_file)
# mol = next(mol_supplier)

with Chem.SDMolSupplier(sdf_file) as suppl:
  ms = [x for x in suppl if x is not None]
# for m in ms: tmp=AllChem.Compute2DCoords(m)
mol = ms[0]
tolerance = 1e-3

# Extract the 3D coordinates of the atoms in the molecule
conf = mol.GetConformer()

cluster_colors = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 1.0, 0.0),  # Yellow
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0)   # Cyan
]
# Dictionary to map atom indices to cluster membership based on 3D positions
atom_colors = {}
highlight_atoms = []
for atom_idx in range(mol.GetNumAtoms()):
    pos = conf.GetAtomPosition(atom_idx)
    atom_pos = (pos.x, pos.y, pos.z)
    
    # Find the corresponding cluster membership in position_dict
    for node, (dict_pos, cluster_id) in d.items():
        if np.allclose(atom_pos, dict_pos, atol=tolerance):
            # Color the atom based on the cluster membership
            if node >= 0:
                highlight_atoms.append(atom_idx)
                atom_colors[atom_idx] = cluster_colors[cluster_id % len(cluster_colors)]
                print(f'Atom {atom_idx} belongs to cluster {cluster_id}')
            break

# Generate 2D coordinates for the molecule
# smi = Chem.MolToSmiles(mol)
# print(smi)
# mol = Chem.MolFromSmiles(smi)
# Draw the molecule with highlighted atoms
AllChem.Compute2DCoords(mol)
drawer = rdMolDraw2D.MolDraw2DCairo(2000, 2000)
drawer.drawOptions().addStereoAnnotation = True
drawer.drawOptions().addAtomIndices = True
rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight_atoms,
                                   highlightAtomColors=atom_colors)
drawer.WriteDrawingText(save_path)