# import atom3d.datasets.datasets as da
# da.download_dataset('lep', 'dataset_lep', split='protein') # Download LBA dataset


import atom3d.datasets as da
dataset = da.load_dataset('dataset_lep/raw/LEP/data', 'lmdb') # Load LMDB format dataset
print(len(dataset))  # Print length
from atom3d.filters import filters
filter1 = filters.distance_filter
df = dataset[0]['atoms_active'] 
protein_active = df[df.chain != 'L']
ligand_active = df[df.chain == 'L']
print(df.columns) # Print keys stored in first structure
filtered_dataset = filter1(protein_active, ligand_active[['x','y','z']], 6)
print(filtered_dataset) # Print filtered dataset

# from atom3d.filters import filters
# import atom3d.datasets as da
# dataset = da.load_dataset('dataset/split-by-sequence-identity-60/data/train', 'lmdb') # Load LMDB format dataset
# from atom3d.filters.filters import distance_filter
# import atom3d.util.formats as fo
# struct = dataset[0] # get first structure in dataset
# atoms_df = struct['atoms_protein'] # load atom data for structure
# print(struct.keys()) # print keys stored in structure

