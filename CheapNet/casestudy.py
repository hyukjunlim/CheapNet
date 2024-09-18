import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from CheapNet import GIGN
from dataset_CheapNet import GraphDataset, PLIDataLoader
import numpy as np
import pandas as pd
import os
import torch_geometric.utils
from mpl_toolkits.mplot3d import Axes3D
import time
def visualize_complex_with_edges(data, x, ligand_diffpool_layer, protein_diffpool_layer, attention_weights_lig, attention_weights_pro, threshold=0.5, save_path="complex_clusters_with_edges.png"):
    """
    Visualizes the clustering of both ligand and protein atoms with edges in a 3D plot after DiffPool and saves the plot as an image.
    Additionally, highlights the protein atoms that are clustered in the cluster with the highest attention score.
    """
    
    def plot_diffpool(data, x, diffpool_layer, ax, color_map, label, top_attention_cluster=None):

        s = diffpool_layer(x, data)[1]
        batch_size, num_nodes, _ = s.size()
        x, mask = torch_geometric.utils.to_dense_batch(x, data.batch, fill_value=0, max_num_nodes=diffpool_layer.max_num)
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        num_nodes = mask.sum(dim=1).squeeze(1).long()
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
            # print(f"Node: {node}, Cluster: {cluster_id}", top_attention_cluster) if cluster_id in top_attention_cluster else None
            if top_attention_cluster is not None and cluster_id in top_attention_cluster:
                color = 'cyan' if label == "Ligand" else 'magenta'
                # color = plt.cm.Set2(cluster_id / (n2 - 1) / 3 + 0.66) if label == "Ligand" else plt.cm.Set2(cluster_id / (n2 - 1) / 3)
                ax.scatter(x, y, z, color=color, s=50, edgecolor='k', label=f"{label} Top Attention" if node == 0 else "")
            else:
                color = color_map(cluster_id / (n2 - 1))
                ax.scatter(x, y, z, color=color, s=10, edgecolor='k', label=label if node == 0 else "")

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
        if label == "Ligand":
            G = nx.Graph()
            er = data.edge_index_inter[:, data.batch[data.edge_index_inter[0, :]] == batch_idx].cpu().numpy()
            G.add_edges_from(er.T)

            for edge in G.edges():
                x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
                y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
                z_coords = [node_positions[edge[0]][2], node_positions[edge[1]][2]]
                ax.plot(x_coords, y_coords, z_coords, color='gray', alpha=0.1)

    # Identify the top 1 attention score cluster for ligand
    attention_scores_lig = attention_weights_lig.mean(dim=1).squeeze(0).cpu().detach().numpy()
    top_attention_cluster_lig = [np.argmax(attention_scores_lig.mean(axis=0), axis=-1)]
    # top_attention_cluster_lig = np.where(attention_scores_lig.max(axis=0) > threshold)[0]
    # print(f"Top attention cluster for ligand: {top_attention_cluster_lig}")

    # Identify the top 1 attention score cluster for protein
    attention_scores_pro = attention_weights_pro.mean(dim=1).squeeze(0).cpu().detach().numpy()
    top_attention_cluster_pro = [np.argmax(attention_scores_pro.mean(axis=0), axis=-1)]
    # top_attention_cluster_pro = np.where(attention_scores_pro.max(axis=0) > threshold)[0]
    # print(f"Top attention cluster for protein: {top_attention_cluster_pro}")

    # Initialize the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ligand clusters with a blue color map, highlighting the top attention cluster
    plot_diffpool(data, x, ligand_diffpool_layer, ax, plt.cm.Reds, "Ligand", top_attention_cluster=top_attention_cluster_lig)

    # Plot protein clusters with a red color map, highlighting the top attention cluster
    plot_diffpool(data, x, protein_diffpool_layer, ax, plt.cm.Blues, "Protein", top_attention_cluster=top_attention_cluster_pro)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    # Save the plot
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

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label="Attention Score")
    plt.title("Attention Map Between Ligand and Protein Clusters") if label == "Protein" else plt.title("Attention Map Between Protein and Ligand Clusters")
    plt.xlabel("Protein Cluster Index") if label == "Protein" else plt.xlabel("Ligand Cluster Index")
    plt.ylabel("Ligand Cluster Index") if label == "Protein" else plt.ylabel("Protein Cluster Index")
    plt.savefig(save_path)
    plt.close()

def create_cross_attention_heatmap(att_lig, att_pro, s_lig, s_pro, save_path="cross_attention_heatmap.png"):
    """
    Creates and saves a heatmap of the cross-attention scores between ligand and protein clusters.
    att_lig: Attention weights from ligand to protein clusters [batch_size, num_heads, num_clusters_lig, num_clusters_pro]
    att_pro: Attention weights from protein to ligand clusters [batch_size, num_heads, num_clusters_pro, num_clusters_lig]
    s_lig: Cluster assignments for ligand clusters [batch_size, max_node, num_clusters_lig]
    s_pro: Cluster assignments for protein clusters [batch_size, max_node, num_clusters_pro]
    """

    att_lig = att_lig.mean(dim=1).squeeze(0) 
    att_pro = att_pro.mean(dim=1).squeeze(0)
    s_lig = s_lig.squeeze(0)
    s_pro = s_pro.squeeze(0)
    att_1 = torch.matmul(torch.matmul(s_lig, att_lig), s_pro.transpose(0, 1))
    # att_2 = torch.matmul(torch.matmul(s_pro, att_pro), s_lig.transpose(0, 1))
    # mat = (att_1 + att_2) / 2
    mat = att_1

    mat = mat.cpu().detach().numpy()
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, cmap='viridis', aspect='auto')
    plt.colorbar(label="Attention Score")
    plt.title("Cross-Attention Map Between Ligand and Protein Clusters")
    plt.xlabel("Protein Cluster Index")
    plt.ylabel("Ligand Cluster Index")
    plt.savefig(save_path)
    plt.close()

# Load the saved GIGN model
num_clusters = [28, 156]
model = GIGN(35, 256, num_clusters).to("cuda" if torch.cuda.is_available() else "cpu")
model_name = '/home/dlagurwns03/dlagurwns03_link/GIGN/codes/Diffpool_GIGN/save/g-d-c/q2q2/ours-lrs-0.001-28-156_0/repeat2/model/epoch-443, train_loss-0.0937, train_rmse-0.3060, valid_rmse-1.1876, valid_pr-0.7662.pt'
# slice mode name from /model
model_name = model_name.split('Diffpool_GIGN/')[1]
model.load_state_dict(torch.load(model_name))
model.eval()


# Load the dataset and select the first complex
data_root = 'data'  # Update with the actual path
test2016_dir = os.path.join(data_root, 'test2016')
test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))

test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type='Graph_GIGN', create=False)
test2016_loader = PLIDataLoader(test2016_set, batch_size=1, shuffle=False, num_workers=4)

for i, data in enumerate(test2016_loader):
    data = data.to("cuda" if torch.cuda.is_available() else "cpu")

    if i + 1 != 14:
        continue
    if i + 1 > 14:
        break
    
    print('-' * 30)
    print(f"Processing complex {i + 1}...")
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
    print(out, y)

    # Create and save the heatmap
    visualize_complex_with_edges(data, x, model.diffpool1, model.diffpool2, att_pro, att_lig, save_path="complex_clusters_with_edges.png")
    create_attention_heatmap(att_pro, 'Ligand', save_path="attention_heatmap_lig.png")
    create_attention_heatmap(att_lig, 'Protein', save_path="attention_heatmap_pro.png")
    create_cross_attention_heatmap(att_lig, att_pro, s_lig, s_pro, save_path="cross_attention_heatmap.png")
    time.sleep(3)

