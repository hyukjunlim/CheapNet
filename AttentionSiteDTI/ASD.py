import random

import torch
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, DenseGCNConv
from layers import MultiHeadAttention
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv, global_mean_pool, global_add_pool
import math
import os

    
def gnn_norm(x, norm):

    batch_size, num_nodes, num_channels = x.size()
    x = x.view(-1, num_channels)
    x = norm(x)
    x = x.view(batch_size, num_nodes, num_channels)

    return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(drop_rate),
        )
        
    def forward(self, x):
        
        return self.mlp(x)
    
class DiffPool(nn.Module):
    def __init__(self, input_dim, output_dim, max_num, red_node, edge, drop_rate):
        super().__init__()

        self.max_num = max_num
        self.red_node = red_node
        self.edge = edge
        self.gnn_p = DenseGCNConv(input_dim, red_node, improved=True, bias=True)
        self.gnn_p_norm = nn.Sequential(
            nn.BatchNorm1d(red_node),
            nn.Mish(),
        )
        self.gnn_e = DenseGCNConv(input_dim, output_dim, improved=True, bias=True)
        self.gnn_e_norm = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
        )
        self.out = nn.Linear(output_dim, output_dim)
        self.out_norm = nn.Sequential(
            nn.BatchNorm1d(output_dim),
        )

    def pooling(self, x, adj, s, mask=None):

        batch_size, num_nodes, _ = x.size()
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        s = F.softmax(s, dim=-1)
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        return out, out_adj, s
    
    def set_edge_index(self, data, edge):

        switch = {
            "intra": data.edge_index_intra,
            "inter": data.edge_index_inter,
            "intra_lig": data.edge_index_intra_lig,
            "intra_pro": data.edge_index_intra_pro,
        }
        data.edge_index = switch.get(edge, None)
    
    def forward(self, x, data):
        
        self.set_edge_index(data, self.edge)
        adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.max_num)
        x, mask = to_dense_batch(x, data.batch, fill_value=0, max_num_nodes=self.max_num)
        
        s = gnn_norm(self.gnn_p(x, adj, mask), self.gnn_p_norm)
        x, adj, s = self.pooling(x, adj, s, mask)
        x = gnn_norm(self.gnn_e(x, adj), self.gnn_e_norm)
        x = gnn_norm(self.out(x), self.out_norm)

        return x, s
    
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, heads, drop_rate):
        super().__init__()

        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = MLP(hidden_dim, hidden_dim, drop_rate)
        
    def forward(self, q, k, v): 

        res = q.sum(dim=1)

        batch_size, seqlen_q, _ = q.shape
        _, seqlen_k, _ = k.shape
        
        Q = self.W_Q(q)  # [batch_size, seqlen_q, hidden_dim]
        K = self.W_K(k)
        V = self.W_V(v)
        
        Q = Q.view(batch_size, seqlen_q, self.heads, self.head_dim).transpose(1, 2)  # [batch_size, heads, seqlen_q, head_dim]
        K = K.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seqlen_q, seqlen_k]
        attention = torch.softmax(energy, dim=-1)  # [batch_size, num_heads, seqlen_q, seqlen_k]
        x = torch.matmul(attention, V)  # [batch_size, num_heads, seqlen_q, head_dim]
        x = x.transpose(1, 2).contiguous().view(batch_size, seqlen_q, self.hidden_dim)  # [batch_size, seqlen_q, hidden_dim]
        x = x.sum(dim=1)

        x = self.W_O(x) + res
        
        return x, attention

    
class CheapNetASD(nn.Module):

    def __init__(self, node_dim, hidden_dim):
        super(CheapNetASD, self).__init__()

        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        self.conv4 = GCNConv(hidden_dim*4, hidden_dim*2)
        self.bn4 = nn.BatchNorm1d(hidden_dim*2)
        self.conv5 = GCNConv(hidden_dim*2, node_dim)
        self.bn5 = nn.BatchNorm1d(node_dim)

        self.dropout = 0.2

        self.bilstm = nn.LSTM(node_dim, node_dim, num_layers=1, bidirectional=True, dropout=self.dropout)

        self.fc_in = nn.Linear(9800, 4340)

        self.fc_out = nn.Linear(4340, 1)
        self.attention = MultiHeadAttention(node_dim * 2, node_dim * 2)

        self.diffpool1 = DiffPool(node_dim, node_dim, 600, num_clusters[0], "intra_lig", 0.2)
        self.diffpool2 = DiffPool(node_dim, node_dim, 600, num_clusters[1], "intra_pro", 0.2)
        self.attblock1 = AttentionBlock(node_dim, 1, 0.2)
        self.attblock2 = AttentionBlock(node_dim, 1, 0.2)
        self.max_num = 600

    def make_edge_index(self, data):

        data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
        data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]

    def forward(self, data):
        
        x = data.x
        self.make_edge_index(data)
        edge_index = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = self.bn5(x)

        x_lig, _ = self.diffpool1(x, data)
        x_pro, _  = self.diffpool2(x, data)
        ligand_rep, _ = self.attblock1(x_lig, x_pro, x_pro)
        protein_rep, _ = self.attblock2(x_pro, x_lig, x_lig)

        # protein_rep = global_add_pool(feature_protein, data.batch) # [bsz, 35]
        # ligand_rep = global_add_pool(feature_smile, data.batch) # [bsz, 35]

        batch_size = protein_rep.size(0)
        sequence = torch.cat((ligand_rep, protein_rep), dim=1).view(batch_size, -1, 35)
        
        mask = torch.eye(140, dtype=torch.uint8).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        mask[:, sequence.size()[1]:140, :] = 0
        mask[:, :, sequence.size()[1]:140] = 0
        mask[:, :, sequence.size()[1] - 1] = 1
        mask[:, sequence.size()[1] - 1, :] = 1
        mask[:,  sequence.size()[1] - 1,  sequence.size()[1] - 1] = 0
        sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - sequence.size()[1]), mode='constant', value=0)
        sequence = sequence.permute(1, 0, 2)
        
        h_0 = Variable(torch.zeros(2, batch_size, 35).cuda())
        c_0 = Variable(torch.zeros(2, batch_size, 35).cuda())

        output, _ = self.bilstm(sequence, (h_0, c_0))
        output = output.permute(1, 0, 2)

        out = self.attention(output, mask=mask)
        
        # Flatten and fully connected layer
        out = F.relu(self.fc_in(out.view(batch_size, -1)))

        # Output layer
        out = self.fc_out(out)
        
        return out.view(-1)

class ASD(nn.Module):

    def __init__(self, node_dim, hidden_dim):
        super(ASD, self).__init__()

        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        self.conv4 = GCNConv(hidden_dim*4, hidden_dim*2)
        self.bn4 = nn.BatchNorm1d(hidden_dim*2)
        self.conv5 = GCNConv(hidden_dim*2, node_dim)
        self.bn5 = nn.BatchNorm1d(node_dim)

        self.dropout = 0.2

        self.bilstm = nn.LSTM(node_dim, node_dim, num_layers=1, bidirectional=True, dropout=self.dropout)

        self.fc_in = nn.Linear(9800, 4340)

        self.fc_out = nn.Linear(4340, 1)
        self.attention = MultiHeadAttention(node_dim * 2, node_dim * 2)

    def make_edge_index(self, data):

        data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
        data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]

    def forward(self, data):
        
        x = data.x
        self.make_edge_index(data)
        edge_index = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = self.bn5(x)

        protein_rep = global_add_pool(x[data.split == 1], data.batch[data.split == 1])
        ligand_rep = global_add_pool(x[data.split == 0], data.batch[data.split == 0])

        batch_size = protein_rep.size(0)
        sequence = torch.cat((ligand_rep, protein_rep), dim=1).view(batch_size, -1, 35)
        
        mask = torch.eye(140, dtype=torch.uint8).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        mask[:, sequence.size()[1]:140, :] = 0
        mask[:, :, sequence.size()[1]:140] = 0
        mask[:, :, sequence.size()[1] - 1] = 1
        mask[:, sequence.size()[1] - 1, :] = 1
        mask[:,  sequence.size()[1] - 1,  sequence.size()[1] - 1] = 0
        sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - sequence.size()[1]), mode='constant', value=0)
        sequence = sequence.permute(1, 0, 2)
        
        h_0 = Variable(torch.zeros(2, batch_size, 35).cuda())
        c_0 = Variable(torch.zeros(2, batch_size, 35).cuda())

        output, _ = self.bilstm(sequence, (h_0, c_0))
        output = output.permute(1, 0, 2)

        out = self.attention(output, mask=mask)
        
        # Flatten and fully connected layer
        out = F.relu(self.fc_in(out.view(batch_size, -1)))

        # Output layer
        out = self.fc_out(out)
        
        return out.view(-1)
    

scheduler_bool = True
lr = 1e-3
# explain = 'CheapNet-ASD'
explain = 'ASD'
num_clusters = [28, 156]
only_rep = [0, 1, 2]
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



if __name__ == '__main__':
    import os
    import pandas as pd
    from thop import profile
    from dataset_ASD import GraphDataset, PLIDataLoader

    data_root = './data'
    graph_type = 'Graph_GIGN'
    test2013_dir = os.path.join(data_root, 'test2013')
    test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
    test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
    test2013_loader = PLIDataLoader(test2013_set, batch_size=1, shuffle=False, num_workers=4)

    data = next(iter(test2013_loader))
    
    model = ASD(35, 256)
    flops, params = profile(model, inputs=(data, ))
    print("params: ", int(params))
    print("flops: ", flops/1e9) 
# %%