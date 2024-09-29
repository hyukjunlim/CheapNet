import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from CheapNet import CheapNet
from dataset_CheapNet import GraphDataset, PLIDataLoader
import numpy as np
import pandas as pd
import os
import torch_geometric.utils
from mpl_toolkits.mplot3d import Axes3D
import time
from utils import load_model_dict

# big font size
plt.rcParams.update({'font.size': 14})

def visualize_complex_with_edges(data, x, ligand_diffpool_layer, protein_diffpool_layer, att_map, threshold=0.1, save_path="complex_clusters_with_edges.png"):
    """
    Visualizes the clustering of both ligand and protein atoms with edges in a 3D plot after DiffPool and saves the plot as an image.
    Additionally, highlights the protein atoms that are clustered in the cluster with the highest attention score.
    """
    d = {}
    def plot_diffpool(data, x, diffpool_layer, ax, color_map, label, top_attention_idx=None):

        s = diffpool_layer(x, data)[1]
        batch_size, num_nodes, _ = s.size()
        x, mask = torch_geometric.utils.to_dense_batch(x, data.batch, fill_value=0, max_num_nodes=diffpool_layer.max_num)
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        num_nodes = mask.sum(dim=1).squeeze(1).long()

        asdf = s.squeeze(0).cpu().detach().numpy()
        if label == "Ligand":
            pass
            # sort asdf at dim=1, biggest 6 probs
            asdf = np.sort(asdf, axis=1)[:6, -6:]
            print(asdf)
        
        clusters = s.argmax(dim=2).squeeze(0)[:num_nodes]
        
        batch_idx = 0
        pos = data.pos[data.batch == batch_idx]
        pos_ = pos[data.split == 0] if label == "Ligand" else pos[data.split == 1]
        pos = pos.cpu().numpy()
        pos_ = pos_.cpu().numpy()
        clusters = clusters[data.split == 0] if label == "Ligand" else clusters[data.split == 1]

        b, n1, n2 = s.size()
        color_map = plt.cm.Set1 if label == "Ligand" else plt.cm.get_cmap('plasma', n2)

        node_positions = {i: pos[i] for i in range(len(pos))}
        node_positions_ = {i: pos_[i] for i in range(len(pos_))}
        node_clusters = {i: clusters[i].item() for i in range(len(pos_))}
        
        # # unique number of clusters
        # print(f"Unique number of clusters in {label}: {len(set(node_clusters.values()))}")
        
        # # unique value of clusters
        # print(f"Unique clusters in {label}: {set(node_clusters.values())}")
        
        for node, (x, y, z) in node_positions_.items():
            cluster_id = node_clusters[node]
            if node in top_attention_idx:
                # print(f"Node: {node}, Cluster: {cluster_id}", top_attention_cluster) if cluster_id in top_attention_cluster else None
                # color = 'cyan' if label == "Ligand" else 'magenta'
                color = plt.cm.Blues(cluster_id / (n2 - 1)) if label == "Ligand" else plt.cm.autumn(cluster_id / (n2 - 1))
                ax.scatter(x, y, z, color=color, s=50, edgecolor='k', label=f"{label} Top Attention" if node == 0 else "")
                ax.text(x, y, z, f"{node}", color='k', fontsize=12)
                d[node] = ((x, y, z), cluster_id)
            else:
                color = color_map(cluster_id / (n2 - 1))
                ax.scatter(x, y, z, color=color, s=10, edgecolor='k', label=label if node == 0 else "")
                ax.text(x, y, z, f"{node}", color='k', fontsize=8)
                d[-node] = ((x, y, z), cluster_id)

        # Plot intra-ligand edges
        if label == "Ligand":
            G = nx.Graph()
            ra_lig = data.edge_index_intra_lig[:, data.batch[data.edge_index_intra_lig[0, :]] == batch_idx].cpu().numpy()
            G.add_edges_from(ra_lig.T)

            for edge in G.edges():
                x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
                y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
                z_coords = [node_positions[edge[0]][2], node_positions[edge[1]][2]]
                ax.plot(x_coords, y_coords, z_coords, color='y', alpha=0.8)
        else:
            # Plot intra-protein edges
            G = nx.Graph()
            ra_pro = data.edge_index_intra_pro[:, data.batch[data.edge_index_intra_pro[0, :]] == batch_idx].cpu().numpy()
            G.add_edges_from(ra_pro.T)

            for edge in G.edges():
                x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
                y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
                z_coords = [node_positions[edge[0]][2], node_positions[edge[1]][2]]
                ax.plot(x_coords, y_coords, z_coords, color='r', alpha=0.8)

        # Plot inter edges
        if label == "Protein":
            G = nx.Graph()
            er = data.edge_index_inter[:, data.batch[data.edge_index_inter[0, :]] == batch_idx].cpu().numpy()
            G.add_edges_from(er.T)

            for edge in G.edges():
                x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
                y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
                z_coords = [node_positions[edge[0]][2], node_positions[edge[1]][2]]
                ax.plot(x_coords, y_coords, z_coords, color='gray', alpha=0.1)

    k = 15
    flattened_tensor = att_map.view(-1)
    topk_values, topk_indices = torch.topk(flattened_tensor, k)
    ligand_indices, protein_indices = [], []
    for idx in topk_indices:
        ligand_idx, protein_idx = divmod(idx.item(), att_map.size(1))
        ligand_indices.append(ligand_idx)
        protein_indices.append(protein_idx)
    
    for i in range(k):
        print(f"Top-{i+1} attention score: {topk_values[i].item()}")
        print(f"Ligand atom index: {ligand_indices[i]}, Protein atom index: {protein_indices[i]}")

    top_attention_cluster_lig = ligand_indices
    print(f"Top attention cluster for ligand: {top_attention_cluster_lig}")

    top_attention_cluster_pro = protein_indices
    print(f"Top attention cluster for protein: {top_attention_cluster_pro}")
    #

    # Initialize the 3D plot
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=60, azim=30)

    # Plot ligand clusters with a blue color map, highlighting the top attention cluster
    plot_diffpool(data, x, ligand_diffpool_layer, ax, plt.cm.Reds, "Ligand", top_attention_idx=top_attention_cluster_lig)

    # Plot protein clusters with a red color map, highlighting the top attention cluster
    plot_diffpool(data, x, protein_diffpool_layer, ax, plt.cm.Blues, "Protein", top_attention_idx=top_attention_cluster_pro)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())
    # print(d)
    # Save the plot
    plt.savefig(save_path)
    plt.savefig('ablation_study_3prs_vis.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def create_cluster_heatmap(attention_weights, label, save_path="attention_heatmap.png"):
    """
    Creates and saves a heatmap of the attention scores between ligand and protein clusters.
    """
    # Average over the heads dimension and squeeze to get [seqlen_q, seqlen_k]
    # attention_matrix = attention_weights.mean(dim=1).squeeze(0).cpu().detach().numpy()
    attention_matrix = attention_weights.squeeze(0).cpu().detach().numpy()
    # attended_clusters = np.where(attention_matrix.sum(axis=0) > 0.1)[0]
    # print(f"Attended clusters for {label}: {attended_clusters}")
    # # sum over the attended clusters
    # attention_matrix = attention_matrix[:, attended_clusters]
    # sum_attention = attention_matrix.sum(axis=1)
    # print(f"Sum of attention scores for {label}: {sum_attention}")


    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label="Attention Score")
    plt.title("Attention Map Between Ligand and Protein Clusters") if label == "Protein" else plt.title("Attention Map Between Protein and Ligand Clusters")
    plt.xlabel("Protein Cluster Index") if label == "Protein" else plt.xlabel("Ligand Cluster Index")
    plt.ylabel("Ligand Cluster Index") if label == "Protein" else plt.ylabel("Protein Cluster Index")
    plt.savefig(save_path)
    plt.close()

def create_attention_heatmap(attention_weights, label, save_path="attention_heatmap.png"):
    """
    Creates and saves a heatmap of the attention scores between ligand and protein clusters.
    """
    # Average over the heads dimension and squeeze to get [seqlen_q, seqlen_k]
    attention_matrix = attention_weights.mean(dim=1).squeeze(0).cpu().detach().numpy()
    # attended_clusters = np.where(attention_matrix.sum(axis=0) > 0.1)[0]
    # print(f"Attended clusters for {label}: {attended_clusters}")
    # # sum over the attended clusters
    # attention_matrix = attention_matrix[:, attended_clusters]
    # sum_attention = attention_matrix.sum(axis=1)
    # print(f"Sum of attention scores for {label}: {sum_attention}")

    if label == "Protein":
        print(attention_matrix.shape)
        a = attention_matrix[0, :]
        b = attention_matrix[19, :]
        c = attention_matrix[27, :]
        # print top5 clusters
        print(f"Top 5 clusters for {label}: {np.argsort(a)[::-1][:10]}")
        print(f"Top 5 clusters for {label}: {np.argsort(b)[::-1][:10]}")
        print(f"Top 5 clusters for {label}: {np.argsort(c)[::-1][:10]}")

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label="Attention Score")
    plt.title("Attention Map Between Ligand and Protein Clusters") if label == "Protein" else plt.title("Attention Map Between Protein and Ligand Clusters")
    plt.xlabel("Protein Cluster Index") if label == "Protein" else plt.xlabel("Ligand Cluster Index")
    plt.ylabel("Ligand Cluster Index") if label == "Protein" else plt.ylabel("Protein Cluster Index")
    plt.savefig(save_path)
    plt.close()

def create_cross_attention_heatmap(data, att_lig, att_pro, s_lig, s_pro, num_nodes, save_path="cross_attention_heatmap.png"):

    att_lig = att_lig.mean(dim=1).squeeze(0) 
    att_pro = att_pro.mean(dim=1).squeeze(0)
    s_lig = s_lig.squeeze(0)[:num_nodes][data.split == 0] # [node_lig, cluster_lig]
    s_pro = s_pro.squeeze(0)[:num_nodes][data.split == 1] # [node_pro, cluster_pro]
    att_1 = torch.matmul(torch.matmul(s_lig, att_lig), s_pro.transpose(0, 1)) # [node_lig, node_pro]
    att_2 = torch.matmul(torch.matmul(s_pro, att_pro), s_lig.transpose(0, 1)) # [node_pro, node_lig]
    mat = (att_1 + att_2.transpose(0, 1))


    mat_cpu = mat.cpu().detach().numpy()
    # Plot the heatmap
    plt.figure(figsize=(11, 6))
    # wanna make the fontsize bigger
    plt.rcParams.update({'font.size': 18})
    plt.imshow(mat_cpu, cmap='viridis', aspect='auto')
    plt.colorbar(label="Attention Score")
    plt.title("Cross-Attention Map Between Protein and Ligand Atoms", fontsize=24)
    plt.xlabel("Protein Atom Index")
    plt.ylabel("Ligand Atom Index")
    # yticks in 5 intervals
    plt.yticks(np.arange(0, mat_cpu.shape[0], 5))
    # save with high resolution
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return mat

# Load the saved GIGN model
device = "cuda" if torch.cuda.is_available() else "cpu"
num_clusters = [28, 156]
model = CheapNet(35, 256, num_clusters).to("cuda" if torch.cuda.is_available() else "cpu")
# model_name = '/home/dlagurwns03/dlagurwns03_link/GIGN/codes/CheapNet/save/g-d-c/q2q2/ours-lrs-0.001-28-156_0/repeat0/model/epoch-643, train_loss-0.0814, train_rmse-0.2853, valid_rmse-1.1860, valid_pr-0.7674.pt'
# model_name = '/home/dlagurwns03/dlagurwns03_link/GIGN/codes/CheapNet/save/g-d-c/q2q2/ours-lrs-0.001-28-156_0/repeat1/model/epoch-748, train_loss-0.0778, train_rmse-0.2788, valid_rmse-1.2007, valid_pr-0.7604.pt'
model_name = '/home/dlagurwns03/dlagurwns03_link/GIGN/codes/CheapNet/save/g-d-c/q2q2/ours-lrs-0.001-28-156_0/repeat2/model/epoch-627, train_loss-0.0857, train_rmse-0.2927, valid_rmse-1.1922, valid_pr-0.7643.pt'
model_name = model_name.split('CheapNet/')[1]
load_model_dict(model, model_name)
model = model.cuda()
model.eval()
from GIGN import GIGN
GIGN_path = 'GIGN_model/20221121_074758_GIGN_repeat0/model/epoch-532, train_loss-0.1162, train_rmse-0.3408, valid_rmse-1.1564, valid_pr-0.7813.pt'
model2 = GIGN(35, 256)
load_model_dict(model2, GIGN_path)
model2 = model2.cuda()
model2.eval()

# Load the dataset and select the first complex
data_root = 'data'  # Update with the actual path
casestudy_dir = os.path.join(data_root, 'test2016')
casestudy_df = pd.read_csv(os.path.join(data_root, 'pred_repeat2.csv'))

casestudy_set = GraphDataset(casestudy_dir, casestudy_df, graph_type='Graph_GIGN', create=False)
casestudy_loader = PLIDataLoader(casestudy_set, batch_size=1, shuffle=False, num_workers=4)

for i, data in enumerate(casestudy_loader):
    if i != 1:
        continue
    data = data.to("cuda" if torch.cuda.is_available() else "cpu")
    print('-' * 30)
    print(f"Processing complex {casestudy_df.iloc[i]['pdbid']}...")
    gign_pred = model2(data)
    model.make_edge_index(data)
    # Pass through the GIGN blocks
    x = model.embedding(data.x)
    model.make_edge_index(data)
    x = model.GIGNBlock1(x, data)
    x = model.GIGNBlock2(x, data)
    x = model.GIGNBlock3(x, data)
    
    # Visualize the entire complex (both ligand and protein) with edges and save as an image
    
    # Obtain the ligand and protein embeddings after diffpool layers
    # Run through the AttentionBlock and capture attention weights
    x_lig, s_lig = model.diffpool1(x, data)
    x_pro, s_pro = model.diffpool2(x, data)
    l2p, att_lig = model.attblock1(x_lig, x_pro, x_pro)  # Forward pass to capture attention weights
    p2l, att_pro = model.attblock2(x_pro, x_lig, x_lig)  # Forward pass to capture attention weights

    out = l2p + p2l
    out = model.fc(out)
    out = out.view(-1)
    y = data.y.view(-1)

    s = model.diffpool1(x, data)[1]
    batch_size, num_nodes, _ = s.size()
    _, mask = torch_geometric.utils.to_dense_batch(x, data.batch, fill_value=0, max_num_nodes=model.diffpool1.max_num)
    mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    num_nodes = mask.sum(dim=1).squeeze(1).long()
    
    print(f'True: {y.item():.4f}')
    print(f'Predicted: {out.item():.4f}')
    print(f'RMSE: {F.mse_loss(out, y).sqrt().item():.4f}')
    print(f'GIGN Predicted: {gign_pred.item():.4f}')

    # Create and save the heatmap
    att_map = create_cross_attention_heatmap(data, att_lig, att_pro, s_lig, s_pro, num_nodes, save_path=f"cross_attention_heatmap.png")
    visualize_complex_with_edges(data, x, model.diffpool1, model.diffpool2, att_map, save_path=f"complex_clusters_with_edges.png")
    # create_attention_heatmap(att_pro, 'Ligand', save_path="attention_heatmap_lig.png")
    # create_attention_heatmap(att_lig, 'Protein', save_path="attention_heatmap_pro.png")
    # create_cluster_heatmap(s_pro, 'Protein', save_path="cluster_heatmap_pro.png")
    # create_cluster_heatmap(s_lig, 'Ligand', save_path="cluster_heatmap_lig.png")

    # time.sleep(3)
    break

