import os
from dataset_CheapNet import GraphDataset, PLIDataLoader
import pandas as pd

data_root = './data'
graph_type = 'Graph_GIGN'
batch_size = 128
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

# check cid '1hvr' in valid_df, test2013_df, test2016_df, test2019_df, csar_df, casf2016_df
print('valid_df:', '1hvr' in valid_df['pdbid'].values)
print('test2013_df:', '1hvr' in test2013_df['pdbid'].values)
print('test2016_df:', '1hvr' in test2016_df['pdbid'].values)
print('test2019_df:', '1hvr' in test2019_df['pdbid'].values)
print('csar_df:', '1hvr' in csar_df['pdbid'].values)
print('casf2016_df:', '1hvr' in casf2016_df['pdbid'].values)

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