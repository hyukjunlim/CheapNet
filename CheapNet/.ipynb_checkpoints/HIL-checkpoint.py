import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, VGAE
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch
import math

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class HIL(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, mode=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HIL, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_node = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.mlp_coord = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        
        self.mode = mode
        self.IAM1 = IAMBlock(in_channels, in_channels)
        self.IAM2 = IAMBlock(in_channels, in_channels)
        

    def forward(self, x, data, edge_index):
        # distance feature
        pos, size = data.pos, None
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        dist = _rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
        radial = self.mlp_coord(dist)
        
        # output
        node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
        out_node = self.mlp_node(x + node_feats)
        return out_node

    def message(self, x_j, x_i, radial, index):
        # IAM
        x_j = self.IAM1(x_i, x_j, x_j)
        radial = self.IAM2(x_i, radial, radial)
        out = x_j * radial
        
        return out

class IAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IAMBlock, self).__init__()
        self.dim_in = in_channels
        self.dim_out = out_channels
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(in_channels, out_channels)
        self.fc3 = nn.Linear(in_channels, out_channels)
        
        self.att = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Linear(in_channels, in_channels), 
            nn.BatchNorm1d(in_channels), 
            nn.LeakyReLU(), 
            nn.Dropout(0.1),
            nn.Linear(in_channels, out_channels))
        
    def forward(self, q, k, v):
        # Q = self.fc1(q)
        # K = self.fc2(k)
        # V = self.fc3(v)
        Q, K, V = q, k, v
        
        # QKV = torch.stack([Q, K, V], dim=1) # (n_batch, 3, d_k)
        # attn = torch.bmm(QKV, QKV.transpose(1, 2)) # (n_batch, 3, 3)
        # attn = self.dropout(F.softmax(attn / (self.dim ** 0.5), dim=-1)) # (n_batch, 3, 3)
        # attn = torch.bmm(attn, QKV) # (n_batch, 3, d_k)
        # Q, K, V = attn.chunk(3, dim=1)
        # Q, K, V = Q.squeeze(1), K.squeeze(1), V.squeeze(1)
        # q = q + Q
        # k = k + K
        # v = v + V
        
#         out = torch.cat([Q, K], dim=-1)
        out = (Q - K) * (Q - K)
        out = self.att(out)
        out = V * out
        # out = self.out(out)
        out = v + out
        
        return out
    
class MultiHeadIAMBlock(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(MultiHeadIAMBlock, self).__init__()
        self.heads = heads
        self.attention_heads = nn.ModuleList([IAMBlock(in_channels, in_channels // heads) for _ in range(heads)])
        self.out = nn.Sequential(nn.Linear(in_channels, in_channels), nn.SiLU(),
                                 nn.Linear(in_channels, in_channels))
        
    def forward(self, q, k):
        head_outputs = [head(q, k) for head in self.attention_heads]
        out = torch.cat(head_outputs, dim=-1)
        # out = self.out(out)
        return out

# #### mask exp 4 ########################################################################################################
# class IAMBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(IAMBlock, self).__init__()
#         self.fc1 = nn.Linear(in_channels, out_channels)
#         self.fc2 = nn.Linear(in_channels, out_channels)
#         self.fc3 = nn.Linear(in_channels, out_channels)
        
#     def forward(self, qkv): # (n_batch, 3, d_k)
#         Q = self.fc1(qkv) 
#         K = self.fc2(qkv) 
#         V = self.fc3(qkv) 
        
#         d_k = V.shape[-1] # d_k
#         attention_score = torch.matmul(Q, K.transpose(-2, -1)) # Q x K^T, (n_batch, 3, 3)
#         attention_score = attention_score / math.sqrt(d_k)
#         attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, 3, 3)
#         out = torch.matmul(attention_prob, V) # (n_batch, 3, d_k)
        
#         return out # (n_batch, d_k)

# class MultiHeadIAMBlock(nn.Module):
#     def __init__(self, in_channels, heads=4):
#         super(MultiHeadIAMBlock, self).__init__()
#         self.heads = heads
#         self.attention_heads = nn.ModuleList([IAMBlock(in_channels, in_channels // heads) for _ in range(heads)])
#         self.fc = nn.Linear(in_channels, in_channels)
        
#     def forward(self, q, k, v, mask=None):
#         qkv = torch.stack([q, k, v], dim=0) # (3, n_batch, d_k)
#         qkv = qkv.transpose(0, 1) # (n_batch, 3, d_k)
        
#         head_outputs = [head(qkv) for head in self.attention_heads]
#         out = torch.cat(head_outputs, dim=-1) # (n_batch, 3, d_k)
#         out = self.fc(out) 
#         out = out + qkv
        
#         return out[:, 0, :], out[:, 1, :], out[:, 2, :]

# #### mask exp 4 ########################################################################################################
# class HIL(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.mlp_node = nn.Sequential(
#             nn.Linear(self.in_channels, self.out_channels),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(self.out_channels))
#         self.mlp_coord = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())

#         self.mid_channels = 16

#     def forward(self, x, data, edge_index, mask=None):
#         pos, size = data.pos, None
#         # propagate
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         dist = _rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
#         radial = self.mlp_coord(dist) 
#         if mask is not None:
#             radial = radial * mask
#         node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
#         # output
#         out_node = self.mlp_node(x + node_feats)
#         return out_node

#     def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
#         x = x_j * radial
#         return x

# #### mask exp 4-1 ########################################################################################################
# class HIL(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.mlp_node = nn.Sequential(
#             nn.Linear(self.in_channels, self.out_channels),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(self.out_channels))
#         self.mlp_coord = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())

#         self.mid_channels = 16

#     def forward(self, x, data, edge_index, mask=None):
#         pos, size = data.pos, None
#         # propagate
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         dist = _rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
#         radial = self.mlp_coord(dist) 
#         if mask is not None:
#             radial = torch.cat([radial, mask], dim=1)
#         node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
#         # output
#         out_node = self.mlp_node(x + node_feats)
#         return out_node

#     def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
#         x = x_j * radial
#         return x
    
#### mask exp 5 ########################################################################################################
# class HIL(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.mlp_node = nn.Sequential(nn.Linear(self.in_channels, self.out_channels), nn.BatchNorm1d(self.out_channels), nn.LeakyReLU(), nn.Dropout(0.1))
#         self.mlp_coord = nn.Sequential(nn.Linear(9, self.out_channels), nn.BatchNorm1d(self.out_channels), nn.LeakyReLU(), nn.Dropout(0.1))
#         self.mid_channels = 16

#         ########
#         self.fc1 = nn.Sequential(nn.Linear(self.in_channels, self.in_channels), nn.SiLU())
#         self.fc2 = nn.Sequential(nn.Linear(self.in_channels, self.in_channels), nn.SiLU())
#         self.att = nn.Linear(2 * self.in_channels, self.in_channels)
        
#     def forward(self, x, data, edge_index):
#         # distance feature
#         pos, size = data.pos, None
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         dist = _rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
#         radial = self.mlp_coord(dist) 
        
#         # feature processing 
#         src = self.fc1(x[edge_index[0]])
#         dst = self.fc2(x[edge_index[1]])
#         feature = torch.cat([src, dst], dim=1)
#         att_score = self.att(feature)
#         radial = radial * att_score
        
#         # output
#         node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
#         out_node = self.mlp_node(x + node_feats)
#         return out_node

#     def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
#         x = x_j * radial
#         return x
    
    
# #### GAT exp 2, no propagation ########################################################################################################
# # heterogeneous interaction layer
# class HIL(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
#         self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
#         self.GATlayer1 = GATv2Conv(self.in_channels, self.in_channels, heads=4, concat=False, edge_dim=self.in_channels)
#         self.GATlayer2 = GATv2Conv(self.in_channels, self.in_channels, heads=4, concat=False, edge_dim=self.in_channels)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.in_channels, self.out_channels),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(self.out_channels))
#         self.ffn = nn.Sequential(
#             nn.BatchNorm1d(self.in_channels),
#             nn.LeakyReLU(),
#             nn.Dropout(0.1))

#     def forward(self, x, edge_index_intra, edge_index_inter, pos=None, size=None):
#         row_cov, col_cov = edge_index_intra
#         coord_diff_cov = pos[row_cov] - pos[col_cov]
#         dist_cov = _rbf(torch.norm(coord_diff_cov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
#         radial_cov = self.mlp_coord_cov(dist_cov)
        
#         out_node_intra = self.GATlayer1(x, edge_index_intra, edge_attr=radial_cov)
        
#         row_ncov, col_ncov = edge_index_inter
#         coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
#         dist_ncov = _rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
#         radial_ncov = self.mlp_coord_ncov(dist_ncov)
        
#         out_node_inter = self.GATlayer2(x, edge_index_inter, edge_attr=radial_ncov)
        
#         out_node = self.mlp(x + out_node_intra) + self.mlp(x + out_node_inter)
#         return out_node

#     def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
#         x = x_j * radial
#         return x

# #### GAT exp 3, link prediction ########################################################################################################
# # heterogeneous interaction layer
# class HIL(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.mlp_node = nn.Sequential(
#             nn.Linear(self.in_channels, self.out_channels),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(self.out_channels))
#         self.mlp_coord = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        

#     def forward(self, x, data, edge_index):
#         pos, size = data.pos, None
#         # propagate
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         dist = _rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
#         radial = self.mlp_coord(dist) 
#         node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
        
#         # output
#         out_node = self.mlp_node(x + node_feats)
#         return out_node

#     def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
#         x = x_j * radial
#         return x
    
#### GAT exp 1, default ########################################################################################################
# # heterogeneous interaction layer
# class HIL(MessagePassing):
#     def __init__(self, in_channels: int,
#                  out_channels: int, 
#                  **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.mlp_node_cov = nn.Sequential(
#             nn.Linear(self.in_channels, self.out_channels),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(self.out_channels))
#         self.mlp_node_ncov = nn.Sequential(
#             nn.Linear(self.in_channels, self.out_channels),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(self.out_channels))
#         self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
#         self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        
#         self.conv1 = GATv2Conv(self.in_channels, self.in_channels, heads=4, concat=False)
#         self.conv2 = GATv2Conv(self.in_channels, self.in_channels, heads=4, concat=False)
#         self.ffn1 = nn.Sequential(
#             nn.Linear(self.in_channels, self.in_channels),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(self.in_channels))
#         self.ffn2 = nn.Sequential(
#             nn.Linear(self.in_channels, self.in_channels),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(self.in_channels))

#     def forward(self, x, edge_index_intra, edge_index_inter, pos=None,
#                 size=None):

#         # x = self.ffn1(self.conv1(x, torch.cat([edge_index_intra, edge_index_inter], dim=1)))
#         x_intra = x + self.ffn1(self.conv1(x, edge_index_intra))
#         x_inter = x + self.ffn2(self.conv2(x, edge_index_inter))
        
#         row_cov, col_cov = edge_index_intra
#         coord_diff_cov = pos[row_cov] - pos[col_cov]
#         dist_cov = _rbf(torch.norm(coord_diff_cov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
#         radial_cov = self.mlp_coord_cov(dist_cov)
#         out_node_intra = self.propagate(edge_index=edge_index_intra, x=x_intra, radial=radial_cov, size=size)
#         row_ncov, col_ncov = edge_index_inter
#         coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
#         dist_ncov = _rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
#         radial_ncov = self.mlp_coord_ncov(dist_ncov)
#         out_node_inter = self.propagate(edge_index=edge_index_inter, x=x_inter, radial=radial_ncov, size=size)
#         out_node = self.mlp_node_cov(x + out_node_intra) + self.mlp_node_ncov(x + out_node_inter)
#         return out_node

#     def message(self, x_j: Tensor, x_i: Tensor, radial,
#                 index: Tensor):
#         x = x_j * radial
#         return x



# def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
#     '''
#     From https://github.com/jingraham/neurips19-graph-protein-design
    
#     Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
#     That is, if `D` has shape [...dims], then the returned tensor will have
#     shape [...dims, D_count].
#     '''
#     D_mu = torch.linspace(D_min, D_max, D_count).to(device)
#     D_mu = D_mu.view([1, -1])
#     D_sigma = (D_max - D_min) / D_count
#     D_expand = torch.unsqueeze(D, -1)

#     RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
#     return RBF


# class BipartiteAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(BipartiteAttention, self).__init__()
#         self.W = nn.Linear(input_dim, input_dim)
#         self.att = nn.Linear(input_dim * 2, 1)
#         self.activation = nn.LeakyReLU()

#     def forward(self, x, idx, data):
#         batch = data.batch
#         num_graphs = data.y.size(0)
#         row = idx[0]
#         col = idx[1]
#         adj_list = []
#         out = torch.zeros_like(x)
#         for i in range(num_graphs):
#             mask = (batch[row] == i) & (batch[col] == i)
#             edge_index_sub = idx[:, mask]
#             edge_index_sub = edge_index_sub - edge_index_sub.min()
#             values = torch.ones(edge_index_sub.shape[1]).to(x.device)
#             num_nodes_sub = (batch == i).sum().item()
#             adj_matrix_dense_sub = torch.sparse_coo_tensor(edge_index_sub, values, (num_nodes_sub, num_nodes_sub)).to(x.device).to_dense()
#             adj_list.append(adj_matrix_dense_sub + torch.eye(adj_matrix_dense_sub.size(0)).to(x.device))
#         # max_node_num = max(i.shape[0] for i in adj_list)
#         # adj = torch.cat([F.pad(x, (0, max_node_num - x.size(0), 0, 0)) for x in adj_list],dim=0)
#         # print(adj.shape)
#         E = [torch.zeros_like(i) for i in adj_list] # (B x N) x N
#         W_X = self.W(x) # B x N x D
#         x_cat = torch.cat([W_X[row], W_X[col]], dim=-1) # B x E x 2D 
#         att = self.activation(self.att(x_cat)).squeeze(-1) # E
#         E[row, col] = att # N x N
#         att_score = F.softmax(E, dim=-1) # N x N
#         att_W_X = torch.matmul(att_score, W_X) # N x D
#         A_att_W_X = torch.matmul(adj.transpose(-2, -1), att_W_X) # N x D
#         out = A_att_W_X
            
#         # batch = data.batch
#         # num_graphs = data.y.size(0)
#         # row = idx[0]
#         # col = idx[1]
#         # adj_list = []
#         # out = torch.zeros_like(x)
#         # for i in range(num_graphs):
#         #     mask = (batch[row] == i) & (batch[col] == i)
#         #     edge_index_sub = idx[:, mask]
#         #     edge_index_sub = edge_index_sub - edge_index_sub.min()
#         #     values = torch.ones(edge_index_sub.shape[1]).to(x.device)
#         #     num_nodes_sub = (batch == i).sum().item()
#         #     adj_matrix_dense_sub = torch.sparse_coo_tensor(edge_index_sub, values, (num_nodes_sub, num_nodes_sub)).to(x.device).to_dense()
#         #     adj_list.append(adj_matrix_dense_sub + torch.eye(adj_matrix_dense_sub.size(0)).to(x.device))
            
#         # for i in range(num_graphs):
#         #     mask = (batch[row] == i) & (batch[col] == i)
#         #     batch_x = x[batch == i] # N x D
#         #     batch_row = row[mask] # E
#         #     batch_col = col[mask] # E
#         #     batch_row = batch_row - batch_row.min()
#         #     batch_col = batch_col - batch_col.min()
#         #     adj = adj_list[i] # N x N
#         #     E = torch.zeros_like(adj).to(x.device) # N x N
            
#         #     W_X = self.W(batch_x) # N x D
#         #     x_cat = torch.cat([W_X[batch_row], W_X[batch_col]], dim=-1) # E x 2D 
#         #     att = self.activation(self.att(x_cat)).squeeze(-1) # E
#         #     E[batch_row, batch_col] = att # N x N
#         #     att_score = F.softmax(E, dim=-1) # N x N
            
#         #     att_W_X = torch.matmul(att_score, W_X) # N x D
#         #     A_att_W_X = torch.matmul(adj.transpose(-2, -1), att_W_X) # N x D
#         #     out[batch == i] = A_att_W_X
        
#         return out
        
        


# class EmbeddingMask(nn.Module):
#     def __init__(self, input_dim):
#         super(EmbeddingMask, self).__init__()
#         self.fc_trans_s = nn.Linear(input_dim, input_dim)
#         self.fc_trans_d = nn.Linear(input_dim, input_dim)
#         self.fc_trans_r = nn.Linear(input_dim, input_dim)
#         self.fc_att = nn.Linear(input_dim * 2, input_dim)
#         self.fc_out = nn.Linear(input_dim, input_dim)
#         self.a_func = nn.LeakyReLU()
#         self.ffn = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(input_dim))
        
#     def calculator(self, x, y):
#         out = torch.cat([x, y], dim=1)
#         out = self.a_func(out)
#         return out
    
#     def calc_att(self, radial_ncov, src, dst):
#         # tr_rad = self.a_func(self.fc_trans_r(radial_ncov))
#         tr_src = self.a_func(self.fc_trans_s(src))
#         tr_dst = self.a_func(self.fc_trans_d(dst))
        
#         # att_s_s = self.fc_att(torch.cat([tr_src, tr_src], dim=1))
#         att_s_d = self.fc_att(torch.cat([tr_src, tr_dst], dim=1))
#         # att_s_r = self.fc_att(torch.cat([tr_src, tr_rad], dim=1))
        
#         # att_d_s = self.fc_att(torch.cat([tr_dst, tr_src], dim=1))
#         # att_d_d = self.fc_att(torch.cat([tr_dst, tr_dst], dim=1))
#         # att_d_r = self.fc_att(torch.cat([tr_dst, tr_rad], dim=1))
        
#         # att_r_s = self.fc_att(torch.cat([tr_rad, tr_src], dim=1))
#         # att_r_d = self.fc_att(torch.cat([tr_rad, tr_dst], dim=1))
#         # att_r_r = self.fc_att(torch.cat([tr_rad, tr_rad], dim=1))
        
#         # tmp_src = F.softmax(torch.stack([att_s_s, att_s_d, att_s_r], dim=0), dim=0)
#         # coeff_s_s, coeff_s_d, coeff_s_r = tmp_src[0], tmp_src[1], tmp_src[2]
#         # out_src = coeff_s_s * src + coeff_s_d * dst + coeff_s_r * radial_ncov
        
#         # tmp_dst = F.softmax(torch.stack([att_d_s, att_d_d, att_d_r], dim=0), dim=0)
#         # coeff_d_s, coeff_d_d, coeff_d_r = tmp_dst[0], tmp_dst[1], tmp_dst[2]
#         # out_dst = coeff_d_s * src + coeff_d_d * dst + coeff_d_r * radial_ncov
        
#         # tmp_rad = F.softmax(torch.stack([att_r_s, att_r_d, att_r_r], dim=0), dim=0)
#         # coeff_r_s, coeff_r_d, coeff_r_r = tmp_rad[0], tmp_rad[1], tmp_rad[2]
#         # out_rad = coeff_r_s * src + coeff_r_d * dst + coeff_r_r * radial_ncov
        
        
#         # out = (out_src + out_dst + out_rad) / 3
#         out = att_s_d * dst
        
#         out = self.ffn(out)
#         return out
    
#     def forward(self, y, src, dst):
#         return self.calc_att(y, src, dst)
    
#     # def forward(self, x, y, row, col):
#     #     return self.calc_att(y, x[row], x[col])
    
# # %%