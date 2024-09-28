import os
import numpy as np
import pandas as pd
from dataset_CheapNet import GraphDataset, PLIDataLoader
from CheapNet_nobatch import CheapNet
import torch
from utils import *

data_root = './data'
graph_type = 'Graph_GIGN'
batch_size = 64

test2019_dir = os.path.join(data_root, 'test2016')
test2019_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
test2019_set = GraphDataset(test2019_dir, test2019_df, graph_type=graph_type, create=False)
test2019_loader = PLIDataLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda:0')

model = CheapNet(35, 256).to(device)
model = model.cuda()
model.eval()

total_node_num_list = []
total = []
for i, data in enumerate(test2019_loader):
    data = data.to(device)
    with torch.no_grad():
        for j in range(data.batch.max().item() + 1):
            mask = data.batch == j
            node_num = mask.sum()
            total_node_num_list.append(node_num.item())
            total.append((node_num.item(), test2019_df.iloc[batch_size * i + j - 1]['pdbid']))

############################################################################################################
# # make a histogram of the number of nodes in each batch
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set the style to 'whitegrid' for a professional look
# sns.set(style="whitegrid")

# # Calculate mean and standard deviation
# mean_node_num = np.mean(total_node_num_list)
# std_node_num = np.std(total_node_num_list)

# # Create the histogram with more detailed customization
# plt.figure(figsize=(8, 6))
# plt.hist(total_node_num_list, bins=20, color='steelblue', edgecolor='black')

# # Titles and labels with larger fonts
# plt.title('Distribution of Node Counts per Graph in the Validation Set', fontsize=16)
# plt.xlabel('Number of Nodes in Each Graph', fontsize=14)
# plt.ylabel('Frequency of Graphs', fontsize=14)

# # Add gridlines for better readability
# plt.grid(True, linestyle='--', alpha=0.7)

# # Add mean and std annotations
# plt.axvline(mean_node_num, color='red', linestyle='dashed', linewidth=2)
# plt.text(mean_node_num + 20, plt.ylim()[1] * 0.9, f'Mean: {mean_node_num:.2f}', color='red', fontsize=12)
# plt.text(mean_node_num + 20, plt.ylim()[1] * 0.8, f'STD: {std_node_num:.2f}', color='red', fontsize=12)

# # Save the updated figure
# plt.savefig('histogram.png', bbox_inches='tight', dpi=300)
############################################################################################################

selected_indices = []
for lower_bound in range(0, 400, 100):
    upper_bound = lower_bound + 100
    indices_in_range = [i for i, (count, _) in enumerate(total) if lower_bound <= count < upper_bound]
    if len(indices_in_range) >= 3:
        selected = np.random.choice(indices_in_range, 3, replace=False)  # Select 3 random graphs
    else:
        selected = indices_in_range  # If less than 3, select all available graphs
    
    selected_indices.append(list(selected))

# Print or save the selected indices
print("Selected Indices:", selected_indices)
print(f"Selected count for each group: {[total[i][0] for j in selected_indices for i in j]}")
print(f"Selected pdbid for each group: {[total[i][1] for j in selected_indices for i in j]}")
############################################################################################################

selected_indices = [[206], [112], [205], [82], [20], [272], [169], [223], [197], [9], [94], [129]]
selected_counts = [98, 74, 88, 167, 185, 122, 229, 247, 238, 310, 332, 305]
selected_pdbids = ['3g2z', '3twp', '2al5', '3f3c', '4ty7', '3ueu', '2zda', '2xbv', '3b68', '3prs', '2vkm', '1ydt']

# df = pd.read_csv('../DEAttentionDTA/data/seq_data_core2016.csv')
# df = df[df['PDBname'].isin(selected_pdbids)]
# print(df)
# print(df.shape)

# df = pd.read_csv('../GAABind/dataset/PDBBind/test2016.txt', sep='\t', header=None)
# df.columns = ['pdbid']
# df = df[df['pdbid'].isin(selected_pdbids)]
# print(df)
# print(df.shape)

1027, 1043, 1043, 1043, 
############################################################################################################


class GraphSubsetDataset(GraphDataset):
    def __init__(self, root, df, indices, graph_type='Graph_GIGN', create=False):
        super().__init__(root, df, graph_type=graph_type, create=create)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]  # Map the new idx to the original dataset
        return super().__getitem__(original_idx)
    

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(42)


cfg = 'TrainConfig_GIGN'
graph_type = 'Graph_GIGN'
batch_size = 64
data_root = './data'
epochs = 15

import torch
import torch.nn as nn
import torch.optim as optim
from CheapNet_nobatch import CheapNet
from dataset_CheapNet import GraphDataset, PLIDataLoader
from utils import *
    
test2019_dir = os.path.join(data_root, 'test2019')
test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))
test2019_set = GraphDataset(test2019_dir, test2019_df, graph_type=graph_type, create=False)
test2019_loader = PLIDataLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CheapNet(35, 256, [28, 156])
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = nn.MSELoss()

running_loss = AverageMeter()
running_acc = AverageMeter()
running_best_mse = BestMeter("min")
best_model_list = []

for iteration, indices in enumerate(selected_indices):
    print(f"Iteration {iteration + 1} with indices: {indices}")
    
    # Create a subset dataset and data loader
    test2019_subset = GraphSubsetDataset(test2019_dir, test2019_df, indices)
    test2019_loader = PLIDataLoader(test2019_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Train the model
    model.train()
    l = []  # Store timing
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for batch_idx, data in enumerate(test2019_loader):
            data = data.to(device)
            pred = model(data)
            label = data.y
            MSE_loss = criterion(pred, label)
            loss = MSE_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()