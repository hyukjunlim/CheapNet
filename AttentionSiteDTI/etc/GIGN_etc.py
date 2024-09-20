





import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
    
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
    
class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, drop_rate, n_tasks):
        super(FC, self).__init__()
        
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.drop_rate = drop_rate
        self.predict = nn.ModuleList()
        self.predict.append(MLP(self.d_graph_layer, self.d_FC_layer, self.drop_rate))
        for _ in range(self.n_FC_layer - 2):
            self.predict.append(MLP(self.d_FC_layer, self.d_FC_layer, self.drop_rate))
        self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))

    def forward(self, h):
        
        for layer in self.predict:
            h = layer(h)
            
        return h

class HIL(MessagePassing):
    def __init__(self, hidden_dim, output_dim, drop_rate, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HIL, self).__init__(**kwargs)
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp_coord = MLP(9, hidden_dim, 0.0)
        self.out = MLP(hidden_dim, output_dim, drop_rate)
        
    def _rbf(self, D, D_min=0., D_max=20., D_count=16, device='cpu'):
        
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        
        return RBF

    def message(self, x_j, x_i, radial, index):
        
        return x_j * radial
    
    def forward(self, x, data, edge_index):
        
        res = x

        pos, size = data.pos, None
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        radial = self.mlp_coord(self._rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        x = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
        
        x = self.out(x) + res

        return x

class GIGNBlock(nn.Module):
    def __init__(self, hidden_dim, output_dim, drop_rate):
        super(GIGNBlock, self).__init__()
        
        self.gconv_intra = HIL(hidden_dim, output_dim, drop_rate)
        self.gconv_inter = HIL(hidden_dim, output_dim, drop_rate)

    def forward(self, x, data):
        
        x_intra = self.gconv_intra(x, data, data.edge_index_intra)
        x_inter = self.gconv_inter(x, data, data.edge_index_inter)
        x = (x_intra + x_inter) / 2

        return x

def gnn_norm(x, norm):

    batch_size, num_nodes, num_channels = x.size()
    x = x.view(-1, num_channels)
    x = norm(x)
    x = x.view(batch_size, num_nodes, num_channels)

    return x

class DiffPool(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_num, red_node, edge, drop_rate):
        super().__init__()

        self.max_num = max_num
        self.red_node = red_node
        self.edge = edge
        self.gnn_p = DenseGCNConv(hidden_dim, red_node, improved=True, bias=True)
        self.gnn_p_norm = nn.Sequential(
            nn.BatchNorm1d(red_node),
            nn.Mish(),
        )
        self.gnn_e = DenseGCNConv(hidden_dim, output_dim, improved=True, bias=True)
        self.gnn_e_norm = nn.Sequential(
            nn.BatchNorm1d(output_dim),
        )


    def pooling(self, x, adj, s, mask=None):

        batch_size, num_nodes, _ = x.size()
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        if mask is not None:
            s = s.masked_fill(mask == 0, 1e-9)

        s = F.softmax(s, dim=-1)
        x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        return out, out_adj

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

        s = self.gnn_p(x, adj, mask)
        s = gnn_norm(s, self.gnn_p_norm)
        x, adj = self.pooling(x, adj, s, mask)
        x = self.gnn_e(x, adj)
        x = gnn_norm(x, self.gnn_e_norm)

        return x
 
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, heads, drop_rate):
        super().__init__()

        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = MLP(hidden_dim, hidden_dim, drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        
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
        drop_attention = self.dropout(attention)
        x = torch.matmul(drop_attention, V)  # [batch_size, num_heads, seqlen_q, head_dim]
        x = x.transpose(1, 2).contiguous().view(batch_size, seqlen_q, self.hidden_dim)  # [batch_size, seqlen_q, hidden_dim]
        x = x.sum(dim=1)

        x = self.W_O(x) + res
        
        return x, attention

class GIGN(nn.Module):
    def __init__(self, node_dim, hidden_dim, red_rate=1, heads=1, drop_rate=0.5):
        super().__init__()
        
        self.embedding = MLP(node_dim, hidden_dim, 0.0)
        self.GIGNBlock1 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock2 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock3 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.red_rate = red_rate
        self.diffpool1 = DiffPool(hidden_dim, hidden_dim, 600, ceil(32 * self.red_rate), "intra_lig", drop_rate)
        self.diffpool2 = DiffPool(hidden_dim, hidden_dim, 600, ceil(160 * self.red_rate), "intra_pro", drop_rate)
        self.attblock1 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.attblock2 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.fc = FC(hidden_dim, hidden_dim, 3, drop_rate, 1)

    def make_edge_index(self, data):

        data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
        data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]

    def forward(self, data):
        
        # GIGN
        x = data.x
        x = self.embedding(x)
        self.make_edge_index(data)
        x = self.GIGNBlock1(x, data)
        x = self.GIGNBlock2(x, data)
        x = self.GIGNBlock3(x, data)

        # DiffPool-Attention
        x_lig = self.diffpool1(x, data)
        x_pro = self.diffpool2(x, data)

        l2p, _ = self.attblock1(x_lig, x_pro, x_pro)
        p2l, _ = self.attblock2(x_pro, x_lig, x_lig)
        x = l2p + p2l

        # FC
        x = self.fc(x)

        return x.view(-1)

scheduler_bool = False
lr = 5e-4
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
explain = f"oursh1-d.5"














# import os
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from math import ceil
# from torch_geometric.utils import to_dense_adj, to_dense_batch
# from torch_geometric.nn import DenseGCNConv, global_mean_pool, global_add_pool
# from torch_geometric.nn.conv import MessagePassing
    
# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim, drop_rate):
#         super(MLP, self).__init__()
        
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.BatchNorm1d(output_dim),
#             nn.Mish(),
#             nn.Dropout(drop_rate),
#         )
        
#     def forward(self, x):
        
#         return self.mlp(x)
    
# class FC(nn.Module):
#     def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, drop_rate, n_tasks):
#         super(FC, self).__init__()
        
#         self.d_graph_layer = d_graph_layer
#         self.d_FC_layer = d_FC_layer
#         self.n_FC_layer = n_FC_layer
#         self.drop_rate = drop_rate
#         self.predict = nn.ModuleList()
#         self.predict.append(MLP(self.d_graph_layer, self.d_FC_layer, self.drop_rate))
#         for _ in range(self.n_FC_layer - 2):
#             self.predict.append(MLP(self.d_FC_layer, self.d_FC_layer, self.drop_rate))
#         self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))

#     def forward(self, h):
        
#         for layer in self.predict:
#             h = layer(h)
            
#         return h


# class HIL(MessagePassing):
#     def __init__(self, hidden_dim, output_dim, drop_rate, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
        
#         self.mlp_coord = MLP(9, hidden_dim, 0.0)
#         self.out = MLP(hidden_dim, output_dim, drop_rate)
        
#     def _rbf(self, D, D_min=0., D_max=20., D_count=16, device='cpu'):
        
#         D_mu = torch.linspace(D_min, D_max, D_count).to(device)
#         D_mu = D_mu.view([1, -1])
#         D_sigma = (D_max - D_min) / D_count
#         D_expand = torch.unsqueeze(D, -1)
#         RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        
#         return RBF
    
#     def forward(self, x, data, edge_index):
        
#         pos, size = data.pos, None
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         radial = self.mlp_coord(self._rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
#         x = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
#         x = self.out(x)

#         return x

#     def message(self, x_j, x_i, radial, index):
        
#         return x_j * radial
    
# class GIGNBlock(nn.Module):
#     def __init__(self, hidden_dim, drop_rate):
#         super(GIGNBlock, self).__init__()
        
#         self.gconv_intra = HIL(hidden_dim, hidden_dim, drop_rate)
#         self.gconv_inter = HIL(hidden_dim, hidden_dim, drop_rate)

#     def forward(self, x, data):
        
#         x_intra = self.gconv_intra(x, data, data.edge_index_intra)
#         x_inter = self.gconv_inter(x, data, data.edge_index_inter)
#         x = x + x_intra + x_inter

#         return x
    
# class GNN(nn.Module): 
#     def __init__(self, hidden_dim, output_dim, improved=True, bias=False):
#         super().__init__()
        
#         self.conv1 = DenseGCNConv(hidden_dim, output_dim, improved=improved, bias=bias)
#         self.ba = nn.Sequential(
#             nn.BatchNorm1d(output_dim),
#             nn.Mish(),
#         )

#     def forward(self, x, adj, mask=None):
        
#         x = self.conv1(x, adj, mask)
#         batch_size, num_nodes, num_channels = x.size()
#         x = x.view(-1, num_channels)
#         x = self.ba(x)
#         x = x.view(batch_size, num_nodes, num_channels)

#         return x

# class DiffPool(nn.Module):
#     def __init__(self, hidden_dim, output_dim, max_num, red_node, edge, drop_rate):
#         super().__init__()

#         self.max_num = max_num
#         self.edge = edge
#         self.gnn_pool = GNN(hidden_dim, red_node)
#         self.gnn_embed = GNN(hidden_dim, output_dim, bias=True)
#         self.out = MLP(hidden_dim, output_dim, drop_rate)
#         self.norm = nn.BatchNorm1d(output_dim)

#     def pooling(self, x, adj, s, mask=None):

#         x = x.unsqueeze(0) if x.dim() == 2 else x
#         adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
#         s = s.unsqueeze(0) if s.dim() == 2 else s
        
#         batch_size, num_nodes, _ = x.size()
#         if mask is None:
#             mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)
#         mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)

#         s = torch.softmax(s, dim=-1)
#         x, s = x * mask, s * mask

#         out = torch.matmul(s.transpose(1, 2), x)
#         out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

#         return out, out_adj

#     def set_edge_index(self, data, edge):

#         switch = {
#             "intra": data.edge_index_intra,
#             "inter": data.edge_index_inter,
#             "complex": data.edge_index_complex,
#             "intra_lig|inter": data.edge_index_intra_lig_inter,
#             "intra_pro|inter": data.edge_index_intra_pro_inter,
#             "intra_lig": data.edge_index_intra_lig,
#             "intra_pro": data.edge_index_intra_pro,
#             "pocket": data.edge_index_pocket
#         }
#         data.edge_index = switch.get(edge, None)
        
#     def forward(self, x, data):

#         if self.edge == "intra_lig":
#             res = global_add_pool(x[data.split == 0], data.batch[data.split == 0])
#         elif self.edge == "intra_pro":
#             res = global_add_pool(x[data.split == 1], data.batch[data.split == 1])

#         self.set_edge_index(data, self.edge)
#         adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.max_num)
#         x, mask = to_dense_batch(x, data.batch, fill_value=0, max_num_nodes=self.max_num)

#         s = self.gnn_pool(x, adj, mask)
#         x, adj = self.pooling(x, adj, s, mask)
#         x = self.gnn_embed(x, adj)

#         x = x.sum(dim=1)
#         x = self.out(x) + res
#         x = self.norm(x)

#         return x
       
# class AttentionBlock(nn.Module):
#     def __init__(self, hidden_dim, heads, drop_rate):
#         super().__init__()

#         self.heads = heads
#         self.hidden_dim = hidden_dim
#         self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.out = MLP(hidden_dim, hidden_dim, drop_rate)

#     def forward(self, q, k, v): # x: [batch, hidden_dim]
        
#         bsz, _ = q.size()
#         Q = self.W_Q(q).view(bsz, self.heads, self.hidden_dim // self.heads, 1) # [batch, heads, self.hidden_dim // heads, 1]
#         K = self.W_K(k).view(bsz, self.heads, self.hidden_dim // self.heads, 1) 
#         V = self.W_V(v).view(bsz, self.heads, self.hidden_dim // self.heads, 1)
        
#         scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(K.size(-2)) # [batch, heads, self.hidden_dim // heads, self.hidden_dim // heads]
#         att_weights = F.softmax(scores, dim=-1)
#         out = torch.matmul(att_weights, V) # [batch, heads, self.hidden_dim // heads, 1]
#         out = out.permute(0, 2, 1, 3).contiguous() # [batch, self.hidden_dim // heads, heads, 1]
#         out = out.view(bsz, self.hidden_dim) # [batch, self.hidden_dim]

#         out = self.out(out)

#         return out

# class GIGN(nn.Module):
#     def __init__(self, node_dim, hidden_dim, heads=8, drop_rate=0.1):
#         super().__init__()
        
#         self.embedding = MLP(node_dim, hidden_dim, 0.0)
#         self.GIGNBlock1 = GIGNBlock(hidden_dim, drop_rate)
#         self.GIGNBlock2 = GIGNBlock(hidden_dim, drop_rate)
#         self.red_rate = 1
#         self.diffpool1 = DiffPool(hidden_dim, hidden_dim, 600, ceil(32 * self.red_rate), "intra_lig", drop_rate)
#         self.diffpool2 = DiffPool(hidden_dim, hidden_dim, 600, ceil(160 * self.red_rate), "intra_pro", drop_rate)
#         self.attblock1 = AttentionBlock(hidden_dim, heads, drop_rate)
#         self.attblock2 = AttentionBlock(hidden_dim, heads, drop_rate)
#         self.fc = FC(hidden_dim, hidden_dim, 3, drop_rate, 1)

#     def create_pocket_sparse_edge_index(self, data):
        
#         pocket_nodes = torch.unique(data.edge_index_inter)
#         intra_lig_edges = data.edge_index_intra_lig[:, torch.isin(data.edge_index_intra_lig[0, :], pocket_nodes) & torch.isin(data.edge_index_intra_lig[1, :], pocket_nodes)]
#         intra_pro_edges = data.edge_index_intra_pro[:, torch.isin(data.edge_index_intra_pro[0, :], pocket_nodes) & torch.isin(data.edge_index_intra_pro[1, :], pocket_nodes)]
#         data.edge_index_pocket = torch.cat([intra_lig_edges, data.edge_index_inter, intra_pro_edges], dim=1)
        
#     def make_edge_index(self, data):

#         data.edge_index_complex = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)
#         data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
#         data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]
#         data.edge_index_intra_lig_inter = torch.cat([data.edge_index_intra_lig, data.edge_index_inter], dim=1)
#         data.edge_index_intra_pro_inter = torch.cat([data.edge_index_intra_pro, data.edge_index_inter], dim=1)
#         self.create_pocket_sparse_edge_index(data)

#     def forward(self, data):
        
#         # GIGN
#         x = data.x
#         x = self.embedding(x)
#         self.make_edge_index(data)
#         x = self.GIGNBlock1(x, data)
#         x = self.GIGNBlock2(x, data)

#         # DiffPool-Attention
#         x_lig = self.diffpool1(x, data) # [batch, hidden_dim]
#         x_pro = self.diffpool2(x, data)
        
#         # x = self.attblock1(x_lig, x_pro, x_pro) + self.attblock2(x_pro, x_lig, x_lig)

#         x = self.attblock1(x_lig, x_lig, x_lig) + self.attblock2(x_pro, x_pro, x_pro)

#         # x = x_lig + x_pro


#         # FC
#         x = self.fc(x)

#         return x.view(-1)

# scheduler_bool = False
# lr = 5e-4
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# explain = f"ours-att-drp0.5-GDss" 




# class Encoding(nn.Module):
#     def __init__(self, hidden_dim, heads=4):
#         super(Encoding, self).__init__()
        
#         self.heads = heads
#         self.semantic_emb = nn.Embedding(2, hidden_dim)
    
#     def trans_dim(self, x1, x2, inverse=False):

#         if not inverse:
#             bsz, hidden_dim = x1.size()
#             x1 = x1.view(bsz, self.heads, hidden_dim // self.heads)
#             x2 = x2.view(bsz, self.heads, hidden_dim // self.heads) 
#         else:
#             bsz, heads, hidden_dim = x1.size()
#             x1 = x1.view(bsz, hidden_dim * heads)
#             x2 = x2.view(bsz, hidden_dim * heads)

#         return x1, x2
    
#     def forward(self, x_lig, x_pro): # x: [batch, hidden_dim]

#         SE = self.semantic_emb(torch.tensor([0, 1], device=x_lig.device)) / math.sqrt(x_lig.size(-1) // self.heads)
#         x_lig, x_pro = x_lig + SE[0], x_pro + SE[1]

#         return x_lig, x_pro
    



# # %%
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from math import ceil
# from torch_geometric.utils import to_dense_adj, to_dense_batch
# from torch_geometric.nn import DenseGCNConv, global_mean_pool, global_add_pool
# from torch_geometric.nn.conv import MessagePassing

# class FC(nn.Module):
#     def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
#         super(FC, self).__init__()
        
#         self.d_graph_layer = d_graph_layer
#         self.d_FC_layer = d_FC_layer
#         self.n_FC_layer = n_FC_layer
#         self.dropout = dropout
#         self.predict = nn.ModuleList()
        
#         for j in range(self.n_FC_layer):
#             if j == 0:
#                 self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))
#             if j == self.n_FC_layer - 1:
#                 self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
#             else:
#                 self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))

#     def forward(self, h):
        
#         for layer in self.predict:
#             h = layer(h)
            
#         return h

# class FeatsBlock(nn.Module):
#     def __init__(self, input_dim):
#         super(FeatsBlock, self).__init__()
        
#         self.gconv_intra = HIL(input_dim, input_dim)
#         self.gconv_inter = HIL(input_dim, input_dim)

#     def forward(self, x, data):

#         set_edge_index(data, "intra")
#         x1 = self.gconv_intra(x, data, data.edge_index)
#         set_edge_index(data, "inter")
#         x2 = self.gconv_inter(x, data, data.edge_index)
#         x = x1 + x2
        
#         return x

# class HIL(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
        
#         self.mlp_coord = nn.Sequential(
#             nn.Linear(9, in_channels),
#             nn.BatchNorm1d(in_channels),
#             nn.Mish(),
#         )
#         self.mlp_node = nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#             nn.Mish(),
#             nn.Dropout(0.1),
#             nn.BatchNorm1d(out_channels),
#         )
        
#     def _rbf(self, D, D_min=0., D_max=20., D_count=16, device='cpu'):
        
#         D_mu = torch.linspace(D_min, D_max, D_count).to(device)
#         D_mu = D_mu.view([1, -1])
#         D_sigma = (D_max - D_min) / D_count
#         D_expand = torch.unsqueeze(D, -1)
#         RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        
#         return RBF
    
#     def forward(self, x, data, edge_index):
        
#         # distance feature
#         pos, size = data.pos, None
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         radial = self.mlp_coord(self._rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
            
#         # output
#         node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
#         out_node = self.mlp_node(x + node_feats)
        
#         return out_node

#     def message(self, x_j, x_i, radial, index):
        
#         return x_j * radial
    
# class GNN(nn.Module): 
#     def __init__(self, in_channels, out_channels, nl_mode=None):
#         super().__init__()
        
#         self.conv1 = DenseGCNConv(in_channels, out_channels, improved=True)
#         if nl_mode != '2e':
#             self.nonlin = nn.Sequential(
#                 nn.BatchNorm1d(out_channels),
#                 nn.Mish(),
#                 # nn.Dropout(0.1),
#             )
#         elif nl_mode == '2e':
#             self.nonlin = nn.Sequential(
#                 nn.BatchNorm1d(out_channels),
#                 nn.Mish(),
#                 nn.Dropout(0.1),
#             )

#     def forward(self, x, adj, mask=None):
        
#         x = self.conv1(x, adj, mask)
#         batch_size, num_nodes, num_channels = x.size()
#         x = x.view(-1, num_channels)
#         x = self.nonlin(x)
#         x = x.view(batch_size, num_nodes, num_channels)
        
#         return x
    

# class DiffPool(nn.Module):
#     def __init__(self, hidden_dim, max_num, red_node, edge):
#         super().__init__()

#         self.max_num = max_num
#         self.edge = edge
#         self.gnn1_pool = GNN(hidden_dim, red_node, nl_mode='1p')
#         self.gnn1_embed = GNN(hidden_dim, hidden_dim, nl_mode='1e')
#         self.gnn2_embed = GNN(hidden_dim, hidden_dim, nl_mode='2e')

#     def pooling(self, x, adj, s, mask=None):

#         x = x.unsqueeze(0) if x.dim() == 2 else x
#         adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
#         s = s.unsqueeze(0) if s.dim() == 2 else s
        
#         batch_size, num_nodes, _ = x.size()
#         if mask is None:
#             mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)
#         mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)

#         s = torch.softmax(s, dim=-1)
#         x, s = x * mask, s * mask

#         out = F.mish(torch.matmul(s.transpose(1, 2), x))
#         out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

#         return out, out_adj

#     def forward(self, x, data):

#         set_edge_index(data, self.edge)
#         adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.max_num)
#         x, mask = to_dense_batch(x, data.batch, fill_value=0, max_num_nodes=self.max_num)

#         s = self.gnn1_pool(x, adj, mask)
#         x = self.gnn1_embed(x, adj, mask)
#         x, adj = self.pooling(x, adj, s, mask)
#         x = self.gnn2_embed(x, adj)
#         x = x.mean(dim=1)

#         return x

# class GIGN(nn.Module):
#     def __init__(self, node_dim, hidden_dim, mode=None):
#         super().__init__()
        
#         self.mode = mode
#         self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)
#         self.lin_node = nn.Sequential(
#             nn.Linear(node_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Mish(),
#         )
#         self.featsblock = FeatsBlock(hidden_dim)
#         self.avg_nodes = [32, 111, 160, 193, 193, 193, 108, 600]
#         self.diffpool1 = DiffPool(hidden_dim, 600, self.avg_nodes[0], edge="intra_lig") if '1' in self.mode else None # avg 32
#         self.diffpool2 = DiffPool(hidden_dim, 600, self.avg_nodes[5], edge="intra_lig|inter") if '2' in self.mode else None # avg 111
#         self.diffpool3 = DiffPool(hidden_dim, 600, self.avg_nodes[1], edge="intra_pro") if '3' in self.mode else None # avg 160
#         self.diffpool4 = DiffPool(hidden_dim, 600, self.avg_nodes[6], edge="intra_pro|inter") if '4' in self.mode else None # avg 190
#         self.diffpool5 = DiffPool(hidden_dim, 600, self.avg_nodes[3], edge="intra") if '5' in self.mode else None # avg 193
#         self.diffpool6 = DiffPool(hidden_dim, 600, self.avg_nodes[4], edge="inter") if '6' in self.mode else None # avg 193
#         self.diffpool7 = DiffPool(hidden_dim, 600, self.avg_nodes[2], edge="complex") if '7' in self.mode else None # avg 193
#         self.diffpool8 = DiffPool(hidden_dim, 600, self.avg_nodes[7], edge="pocket") if '8' in self.mode else None # avg 108

#         self.W_Q = nn.Linear(hidden_dim, hidden_dim)
#         self.W_K = nn.Linear(hidden_dim, hidden_dim)
#         self.W_V = nn.Linear(hidden_dim, hidden_dim)

#     def create_pocket_sparse_edge_index(self, data):
        
#         pocket_nodes = torch.unique(data.edge_index_inter)
#         intra_lig_edges = data.edge_index_intra_lig[:, torch.isin(data.edge_index_intra_lig[0, :], pocket_nodes) & torch.isin(data.edge_index_intra_lig[1, :], pocket_nodes)]
#         intra_pro_edges = data.edge_index_intra_pro[:, torch.isin(data.edge_index_intra_pro[0, :], pocket_nodes) & torch.isin(data.edge_index_intra_pro[1, :], pocket_nodes)]
#         data.edge_index_pocket = torch.cat([intra_lig_edges, data.edge_index_inter, intra_pro_edges], dim=1)
        
#     def make_edge_index(self, data):

#         data.edge_index_complex = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)
#         data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
#         data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]
#         data.edge_index_intra_lig_inter = torch.cat([data.edge_index_intra_lig, data.edge_index_inter], dim=1)
#         data.edge_index_intra_pro_inter = torch.cat([data.edge_index_intra_pro, data.edge_index_inter], dim=1)
#         self.create_pocket_sparse_edge_index(data)

#     def forward(self, data):
        
#         # GIGN
#         x = data.x
#         x = self.lin_node(x)
#         self.make_edge_index(data)
#         x = self.featsblock(x, data)
#         if res_bool:
#             res = x

#         # DiffPool
#         x_list = []
#         x_list.append(self.diffpool1(x, data)) if '1' in self.mode else None
#         x_list.append(self.diffpool2(x, data)) if '2' in self.mode else None
#         x_list.append(self.diffpool3(x, data)) if '3' in self.mode else None
#         x_list.append(self.diffpool4(x, data)) if '4' in self.mode else None
#         x_list.append(self.diffpool5(x, data)) if '5' in self.mode else None
#         x_list.append(self.diffpool6(x, data)) if '6' in self.mode else None
#         x_list.append(self.diffpool7(x, data)) if '7' in self.mode else None
#         x_list.append(self.diffpool8(x, data)) if '8' in self.mode else None
        
#         if att_bool:
#             x = torch.stack(x_list, dim=1)
#             Q = self.W_Q(x)
#             K = self.W_K(x)
#             V = self.W_V(x)
#             x = torch.matmul(F.softmax(torch.matmul(Q, K.transpose(1, 2)) / ceil(K.size(-1) ** 0.5), dim=-1), V).mean(dim=1)
#         else:
#             x = torch.sum(torch.stack(x_list), dim=0)
#         if res_bool:
#             res = global_mean_pool(res, data.batch)
#             x = x + res

#         # FC
#         x = self.fc(x)

#         return x.view(-1)

# def set_edge_index(data, edge):

#     switch = {
#         "intra": data.edge_index_intra,
#         "inter": data.edge_index_inter,
#         "complex": data.edge_index_complex,
#         "intra_lig|inter": data.edge_index_intra_lig_inter,
#         "intra_pro|inter": data.edge_index_intra_pro_inter,
#         "intra_lig": data.edge_index_intra_lig,
#         "intra_pro": data.edge_index_intra_pro,
#         "pocket": data.edge_index_pocket
#     }
    
#     # process
#     data.edge_index = switch.get(edge, None)

# # mode_list = ['1', '2', '3', '4', '5', '12', '13', '14', '15', '23', '24', '25', '34', '35', '45'] # 15
# # mode_list = ['123', '124', '125', '134', '135', '145', '234', '235', '245', '345'] # 10
# # mode_list = ['1234', '1235', '1245', '1345', '2345', '12345'] # 6
# mode_list = ['137']
# att_bool = True
# res_bool = False
# scheduler_bool = False
# lr = 5e-4

# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# explain = "ours-adamw-gadb-1"











# # %%
# import torch
# import torch.nn as nn
# from torch_geometric.utils import to_dense_adj, to_dense_batch
# from torch_geometric.nn import DenseGCNConv, dense_diff_pool, global_mean_pool, global_add_pool, DMoNPooling
# from math import ceil
# import os
# import torch.nn.functional as F
# from torch_geometric.nn.conv import MessagePassing
# from typing import List, Optional, Tuple, Union

# import torch
# import torch.nn.functional as F
# from torch import Tensor

# def pooling(x, adj, s, mask=None):

#     x = x.unsqueeze(0) if x.dim() == 2 else x
#     adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
#     s = s.unsqueeze(0) if s.dim() == 2 else s
    
#     batch_size, num_nodes, _ = x.size()
#     if mask is None:
#         mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)
#     mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)

#     s = torch.softmax(s, dim=-1)
#     x, s = x * mask, s * mask

#     out = F.mish(torch.matmul(s.transpose(1, 2), x))
#     out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

#     return out, out_adj

# def set_edge_index(data, edge):

#     switch = {
#         "intra": data.edge_index_intra,
#         "inter": data.edge_index_inter,
#         "complex": data.edge_index_complex,
#         "intra_lig|inter": data.edge_index_intra_lig_inter,
#         "intra_pro|inter": data.edge_index_intra_pro_inter,
#         "intra_lig": data.edge_index_intra_lig,
#         "intra_pro": data.edge_index_intra_pro,
#         "pocket": data.edge_index_pocket
#     }
    
#     # process
#     data.edge_index = switch.get(edge, None)
    
# class FC(nn.Module):
#     def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
#         super(FC, self).__init__()
        
#         self.d_graph_layer = d_graph_layer
#         self.d_FC_layer = d_FC_layer
#         self.n_FC_layer = n_FC_layer
#         self.dropout = dropout
#         self.predict = nn.ModuleList()
        
#         for j in range(self.n_FC_layer):
#             if j == 0:
#                 self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))
#             if j == self.n_FC_layer - 1:
#                 self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
#             else:
#                 self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))

#     def forward(self, h):
        
#         for layer in self.predict:
#             h = layer(h)
            
#         return h

# class FeatsBlock(nn.Module):
#     def __init__(self, input_dim):
#         super(FeatsBlock, self).__init__()
        
#         self.gconv_intra = HIL(input_dim, input_dim)
#         self.gconv_inter = HIL(input_dim, input_dim)

#     def forward(self, x, data):

#         set_edge_index(data, "intra")
#         x1 = self.gconv_intra(x, data, data.edge_index)
#         set_edge_index(data, "inter")
#         x2 = self.gconv_inter(x, data, data.edge_index)
#         x = x1 + x2
        
#         return x

# class HIL(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
        
#         self.mlp_coord = nn.Sequential(
#             nn.Linear(9, in_channels),
#             nn.BatchNorm1d(in_channels),
#             nn.Mish(),
#         )
#         self.mlp_node = nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#             nn.Mish(),
#             nn.BatchNorm1d(out_channels),
#             nn.Dropout(0.1),
#         )
        
#     def _rbf(self, D, D_min=0., D_max=20., D_count=16, device='cpu'):
        
#         D_mu = torch.linspace(D_min, D_max, D_count).to(device)
#         D_mu = D_mu.view([1, -1])
#         D_sigma = (D_max - D_min) / D_count
#         D_expand = torch.unsqueeze(D, -1)
#         RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        
#         return RBF
    
#     def forward(self, x, data, edge_index):
        
#         # distance feature
#         pos, size = data.pos, None
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         radial = self.mlp_coord(self._rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
            
#         # output
#         node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
#         out_node = self.mlp_node(x + node_feats)
        
#         return out_node

#     def message(self, x_j, x_i, radial, index):
        
#         return x_j * radial
    
# class GNN(nn.Module): 
#     def __init__(self, in_channels, out_channels, nl_mode=None):
#         super().__init__()
        
#         self.conv1 = DenseGCNConv(in_channels, out_channels, improved=True)
#         if nl_mode != '2e':
#             self.nonlin = nn.Sequential(
#                 nn.BatchNorm1d(out_channels),
#                 nn.Mish(),
#             )
#         elif nl_mode == '2e':
#             self.nonlin = nn.Sequential(
#                 nn.Mish(),
#                 nn.BatchNorm1d(out_channels),
#                 nn.Dropout(0.1),
#             )

#     def forward(self, x, adj, mask=None):
        
#         x = self.conv1(x, adj, mask)
#         batch_size, num_nodes, num_channels = x.size()
#         x = x.view(-1, num_channels)
#         x = self.nonlin(x)
#         x = x.view(batch_size, num_nodes, num_channels)
        
#         return x
    

# class DiffPool(nn.Module):
#     def __init__(self, hidden_dim, max_num, red_node, edge):
#         super().__init__()

#         self.max_num = max_num
#         self.edge = edge
#         self.gnn1_pool = GNN(hidden_dim, red_node, nl_mode='1p')
#         self.gnn1_embed = GNN(hidden_dim, hidden_dim, nl_mode='1e')
#         self.gnn2_embed = GNN(hidden_dim, hidden_dim, nl_mode='2e')

#     def _rbf(self, D, D_min=0., D_max=6., D_count=9, device='cpu'):
        
#         D_mu = torch.linspace(D_min, D_max, D_count).to(device)
#         D_mu = D_mu.view([1, -1])
#         D_sigma = (D_max - D_min) / D_count
#         D_expand = torch.unsqueeze(D, -1)
#         RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        
#         return RBF
    
#     def calc_attr(self, data):
        
#         # pos = data.pos
#         # row, col = data.edge_index
#         # coord_diff = pos[row] - pos[col]
#         # dist_attr = self.mlp_coord(self._rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=data.x.device))
#         # dist_attr = dist_attr.squeeze(-1)

#         row, col = data.edge_index
#         dist = torch.norm(data.pos[row] - data.pos[col], dim=1)

#         total_size = data.edge_index.size(1)
#         inter_size = data.edge_index_inter.size(1)
#         intra_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         inter_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         intra_size = total_size if data.edge_index in ["intra", "intra_lig", "intra_pro"] else total_size - inter_size
#         intra_mask[:intra_size] = True
#         inter_mask[intra_size:] = True
        
#         dist[(intra_mask & (dist < 0.5))] = 1.5
#         dist[(inter_mask & (dist < 0.5))] = 4.3
#         dist_attr = 4 / (torch.sqrt(dist) + 1)

#         return dist_attr

#     def process_data(self, x, data):

#         data.edge_attr = self.calc_attr(data) if dist_bool else None
#         adj = to_dense_adj(data.edge_index, data.batch, edge_attr=data.edge_attr, max_num_nodes=self.max_num)
#         x, mask = to_dense_batch(x, data.batch, fill_value=0, max_num_nodes=self.max_num)
        
#         return x, adj, mask

#     def forward(self, x, data):

#         set_edge_index(data, self.edge)
#         x, adj, mask = self.process_data(x, data)
#         s = self.gnn1_pool(x, adj, mask)
#         x = self.gnn1_embed(x, adj, mask)
#         x, adj = pooling(x, adj, s, mask)
#         x = self.gnn2_embed(x, adj)
#         x = x.mean(dim=1)

#         return x

# class GIGN(nn.Module):
#     def __init__(self, node_dim, hidden_dim, red_node=None, mode=None):
#         super().__init__()
        
#         self.mode = mode
#         self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)
#         self.lin_node = nn.Sequential(
#             nn.Linear(node_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Mish(),
#         )
#         self.featsblock = FeatsBlock(hidden_dim)
#         self.diffpool1 = DiffPool(hidden_dim, 600, 30, edge="intra_lig") # avg 30
#         self.diffpool2 = DiffPool(hidden_dim, 600, 100, edge="pocket") # avg 100
#         self.diffpool3 = DiffPool(hidden_dim, 600, 200, edge="intra") # avg 200

#     def create_pocket_sparse_edge_index(self, data):
        
#         pocket_nodes = torch.unique(data.edge_index_inter)
#         intra_lig_edges = data.edge_index_intra_lig[:, torch.isin(data.edge_index_intra_lig[0, :], pocket_nodes) & torch.isin(data.edge_index_intra_lig[1, :], pocket_nodes)]
#         intra_pro_edges = data.edge_index_intra_pro[:, torch.isin(data.edge_index_intra_pro[0, :], pocket_nodes) & torch.isin(data.edge_index_intra_pro[1, :], pocket_nodes)]
#         data.edge_index_pocket = torch.cat([intra_lig_edges, data.edge_index_inter, intra_pro_edges], dim=1)
        
#     def make_edge_index(self, data):

#         # data processing
#         data.edge_index_complex = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)
#         data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
#         data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]
#         data.edge_index_intra_lig_inter = torch.cat([data.edge_index_intra_lig, data.edge_index_inter], dim=1)
#         data.edge_index_intra_pro_inter = torch.cat([data.edge_index_intra_pro, data.edge_index_inter], dim=1)
#         self.create_pocket_sparse_edge_index(data)

#     def forward(self, data):
        
#         self.make_edge_index(data)
        
#         # GIGN
#         x = data.x
#         x = self.lin_node(x)
#         x = self.featsblock(x, data)
#         res = x

#         # DiffPool
#         x1 = self.diffpool1(x, data)
#         x2 = self.diffpool2(x, data)
#         x3 = self.diffpool3(x, data)
#         x = x1 + x2 + x3

#         # global pooling
#         if res_bool:
#             res = global_mean_pool(res, data.batch)
#             x = x + res

#         # FC
#         x = self.fc(x)

#         return x.view(-1)
    
# red_node_list = [0, 1, 2]
# # red_node_list = [5 * i for i in range(1, 5)] 
# # red_node_list = [5 * i for i in range(5, 7)]
# # red_node_list = [5 * i for i in range(7, 11)]

# dist_bool = True
# res_bool = False
# scheduler_bool = False
# lr = 5e-4

# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# explain = "f-025-hbad-gbad-d"


















# # %%
# import torch
# import torch.nn as nn
# from torch_geometric.utils import to_dense_adj, to_dense_batch
# from torch_geometric.nn import DenseGCNConv, dense_diff_pool, global_mean_pool, global_add_pool
# from math import ceil
# import os
# import torch.nn.functional as F
# from torch_geometric.nn.conv import MessagePassing

# class FC(nn.Module):
#     def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
#         super(FC, self).__init__()
        
#         self.d_graph_layer = d_graph_layer
#         self.d_FC_layer = d_FC_layer
#         self.n_FC_layer = n_FC_layer
#         self.dropout = dropout
#         self.predict = nn.ModuleList()
        
#         for j in range(self.n_FC_layer):
#             if j == 0:
#                 self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))
#             if j == self.n_FC_layer - 1:
#                 self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
#             else:
#                 self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))

#     def forward(self, h):
        
#         for layer in self.predict:
#             h = layer(h)
            
#         return h

# class FeatsBlock(nn.Module):
#     def __init__(self, input_dim):
#         super(FeatsBlock, self).__init__()
        
#         self.gconv_intra = HIL(input_dim, input_dim)
#         self.gconv_inter = HIL(input_dim, input_dim)

#     def forward(self, x, data):
        
#         intra_x = self.gconv_intra(x, data, data.edge_index_intra)
#         inter_x = self.gconv_inter(x, data, data.edge_index_inter)
#         x = intra_x + inter_x
        
#         return x

# class HIL(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, mode=None, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
        
#         self.mode = mode
#         self.mlp_coord = nn.Sequential(
#             nn.Linear(9, in_channels),
#             nn.BatchNorm1d(in_channels),
#             nn.Mish(),
#         )
#         self.mlp_node = nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#             nn.Mish(),
#             nn.BatchNorm1d(out_channels),
#             nn.Dropout(0.1),
#         )
        
#     def _rbf(self, D, D_min=0., D_max=20., D_count=16, device='cpu'):
        
#         D_mu = torch.linspace(D_min, D_max, D_count).to(device)
#         D_mu = D_mu.view([1, -1])
#         D_sigma = (D_max - D_min) / D_count
#         D_expand = torch.unsqueeze(D, -1)
#         RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        
#         return RBF
    
#     def forward(self, x, data, edge_index):
        
#         # distance feature
#         pos, size = data.pos, None
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         radial = self.mlp_coord(self._rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
            
#         # output
#         node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
#         out_node = self.mlp_node(x + node_feats)
        
#         return out_node

#     def message(self, x_j, x_i, radial, index):
        
#         return x_j * radial
    
# class GNN(nn.Module): 
#     def __init__(self, in_channels, out_channels, nl_mode=None):
#         super().__init__()
        
#         self.nl_mode = nl_mode
#         self.conv1 = DenseGCNConv(in_channels, out_channels)
#         if nl_mode == '1p':
#             self.nonlin = nn.Sequential(
#                 nn.BatchNorm1d(out_channels),
#                 nn.Mish(),
#             )
#         elif nl_mode == '1e':
#             self.nonlin = nn.Sequential(
#                 nn.BatchNorm1d(out_channels),
#                 nn.Mish(),
#             )
#         elif nl_mode == '2e':
#             self.nonlin = nn.Sequential(
#                 nn.Mish(),
#                 nn.BatchNorm1d(out_channels),
#                 nn.Dropout(0.1),
#             )

#     def bn(self, x):
        
#         batch_size, num_nodes, num_channels = x.size()
#         x = x.view(-1, num_channels)
#         x = self.nonlin(x)
#         x = x.view(batch_size, num_nodes, num_channels)
        
#         return x

#     def forward(self, x, adj, mask=None):
        
#         x = self.conv1(x, adj, mask)
#         x = self.bn(x)
        
#         return x
    

# class DiffPool(nn.Module):
#     def __init__(self, hidden_dim, max_num, red_node, edge):
#         super().__init__()
        
#         self.edge = edge
#         self.max_num = max_num
#         self.gnn1_pool = GNN(hidden_dim, red_node, nl_mode='1p')
#         self.gnn1_embed = GNN(hidden_dim, hidden_dim, nl_mode='1e')
#         self.gnn2_embed = GNN(hidden_dim, hidden_dim, nl_mode='2e')
#         self.mlp_coord = nn.Sequential(
#             nn.Linear(9, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Mish(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid(),
#         )

#     def _rbf(self, D, D_min=0., D_max=6., D_count=9, device='cpu'):
        
#         D_mu = torch.linspace(D_min, D_max, D_count).to(device)
#         D_mu = D_mu.view([1, -1])
#         D_sigma = (D_max - D_min) / D_count
#         D_expand = torch.unsqueeze(D, -1)
#         RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        
#         return RBF
    
#     def calc_attr(self, data):
        
#         pos = data.pos
#         row, col = data.edge_index
#         coord_diff = pos[row] - pos[col]
#         dist_attr = self.mlp_coord(self._rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=data.x.device))
#         dist_attr = dist_attr.squeeze(-1)

#         # row, col = data.edge_index
#         # dist = torch.norm(data.pos[row] - data.pos[col], dim=1)

#         # total_size = data.edge_index.size(1)
#         # inter_size = data.edge_index_inter.size(1)
#         # intra_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         # inter_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         # intra_size = total_size if data.edge_index in ["intra", "intra_lig", "intra_pro"] else total_size - inter_size
#         # intra_mask[:intra_size] = True
#         # inter_mask[intra_size:] = True
        
#         # dist[(intra_mask & (dist < 0.5))] = 1.5
#         # dist[(inter_mask & (dist < 0.5))] = 4.3
#         # dist_attr = 4 / (torch.sqrt(dist) + 1)
        
#         return dist_attr

#     def process_data(self, x, data):
        
#         data.x = x
#         data.edge_attr = self.calc_attr(data)
#         adj = to_dense_adj(data.edge_index, data.batch, edge_attr=data.edge_attr, max_num_nodes=self.max_num)
#         x, mask = to_dense_batch(data.x, data.batch, max_num_nodes=self.max_num, fill_value=0)
        
#         return x, adj, mask

#     def set_edge_index(self, data, edge):
#         switch = {
#             "intra": data.edge_index_intra,
#             "inter": data.edge_index_inter,
#             "complex": data.edge_index_complex,
#             "intra_lig|inter": data.edge_index_intra_lig_inter,
#             "intra_pro|inter": data.edge_index_intra_pro_inter,
#             "intra_lig": data.edge_index_intra_lig,
#             "intra_pro": data.edge_index_intra_pro
#         }
        
#         # process
#         data.edge_index = switch.get(edge, None)

#     def forward(self, x, data):

#         self.set_edge_index(data, self.edge)
#         x, adj, mask = self.process_data(x, data)
#         s = self.gnn1_pool(x, adj, mask)
#         x = self.gnn1_embed(x, adj, mask)
#         x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
#         x = self.gnn2_embed(x, adj)
#         x = x.mean(dim=1)
        
#         return x

# class GIGN(nn.Module):
#     def __init__(self, node_dim, hidden_dim, red_node=None, mode=None):
#         super().__init__()
        
#         self.mode = mode
#         self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)
#         self.lin_node = nn.Sequential(
#             nn.Linear(node_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Mish(),
#         )
#         self.featsblock = FeatsBlock(hidden_dim)
#         self.diffpool1 = DiffPool(hidden_dim, 600, red_node, edge="intra")
#         self.diffpool2 = DiffPool(hidden_dim, 600, red_node, edge="inter")

#     def set_edge_index(self, data, edge):
#         switch = {
#             "intra": data.edge_index_intra,
#             "inter": data.edge_index_inter,
#             "complex": data.edge_index_complex,
#             "intra_lig|inter": data.edge_index_intra_lig_inter,
#             "intra_pro|inter": data.edge_index_intra_pro_inter,
#             "intra_lig": data.edge_index_intra_lig,
#             "intra_pro": data.edge_index_intra_pro
#         }
        
#         # process
#         data.edge_index = switch.get(edge, None)

#     def forward(self, data):
        
#         # data processing
#         data.edge_index_complex = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)
#         data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
#         data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]
#         data.edge_index_intra_lig_inter = torch.cat([data.edge_index_intra_lig, data.edge_index_inter], dim=1)
#         data.edge_index_intra_pro_inter = torch.cat([data.edge_index_intra_pro, data.edge_index_inter], dim=1)
        
#         # GIGN
#         x = data.x
#         x = self.lin_node(x)
#         x = self.featsblock(x, data)

#         # DiffPool
#         x1 = self.diffpool1(x, data)
#         x2 = self.diffpool2(x, data)
#         x = x1 + x2

#         # # global pooling
#         # x = global_add_pool(x, data.batch)

#         # FC
#         x = self.fc(x)

#         return x.view(-1)


# # red_node_list = [60, 120, 180]
# # red_node_list = [240, 300, 360]
# red_node_list = [420, 480, 540, 600]

# scheduler_bool = False
# lr = 5e-4
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# explain = "f56"





# # %%
# import torch
# import torch.nn as nn
# from torch_geometric.utils import to_dense_adj, to_dense_batch
# from torch_geometric.nn import DenseGCNConv, dense_diff_pool, global_mean_pool, global_add_pool
# from math import ceil
# import os
# import torch.nn.functional as F
# from torch_geometric.nn.conv import MessagePassing

# class FC(nn.Module):
#     def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
#         super(FC, self).__init__()
        
#         self.d_graph_layer = d_graph_layer
#         self.d_FC_layer = d_FC_layer
#         self.n_FC_layer = n_FC_layer
#         self.dropout = dropout
#         self.predict = nn.ModuleList()
        
#         for j in range(self.n_FC_layer):
#             if j == 0:
#                 self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))
#             if j == self.n_FC_layer - 1:
#                 self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
#             else:
#                 self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))

#     def forward(self, h):
        
#         for layer in self.predict:
#             h = layer(h)
            
#         return h

# class FeatsBlock(nn.Module):
#     def __init__(self, input_dim):
#         super(FeatsBlock, self).__init__()
        
#         self.gconv_intra = HIL(input_dim, input_dim)
#         self.gconv_inter = HIL(input_dim, input_dim)

#     def forward(self, x, data):
        
#         intra_x = self.gconv_intra(x, data, data.edge_index_intra)
#         inter_x = self.gconv_inter(x, data, data.edge_index_inter)
#         x = intra_x + inter_x
        
#         return x

# class HIL(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, mode=None, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(HIL, self).__init__(**kwargs)
        
#         self.mode = mode
#         self.mlp_coord = nn.Sequential(
#             nn.Linear(9, in_channels),
#             nn.BatchNorm1d(in_channels),
#             nn.Mish(),
#             # nn.Dropout(0.1),
#         )
#         self.mlp_node = nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#             nn.Mish(),
#             nn.BatchNorm1d(out_channels),
#             nn.Dropout(0.1),
#         )
        
#     def _rbf(self, D, D_min=0., D_max=20., D_count=16, device='cpu'):
        
#         D_mu = torch.linspace(D_min, D_max, D_count).to(device)
#         D_mu = D_mu.view([1, -1])
#         D_sigma = (D_max - D_min) / D_count
#         D_expand = torch.unsqueeze(D, -1)
#         RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        
#         return RBF
    
#     def forward(self, x, data, edge_index):
        
#         # distance feature
#         pos, size = data.pos, None
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         radial = self.mlp_coord(self._rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
            
#         # output
#         node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
#         out_node = self.mlp_node(x + node_feats)
        
#         return out_node

#     def message(self, x_j, x_i, radial, index):
        
#         return x_j * radial
    
# class GNN(nn.Module): 
#     def __init__(self, in_channels, out_channels, nl_mode=None):
#         super().__init__()
        
#         self.nl_mode = nl_mode
#         self.conv1 = DenseGCNConv(in_channels, out_channels)
#         if self.nl_mode == '1p':
#             self.nonlin = nn.Sequential(
#                 nn.BatchNorm1d(out_channels),
#                 nn.Mish(),
#                 # nn.Dropout(0.1),
#             )
#         elif self.nl_mode == '1e':
#             self.nonlin = nn.Sequential(
#                 nn.BatchNorm1d(out_channels),
#                 nn.Mish(),
#                 # nn.Dropout(0.1),
#             )
#         else:
#             self.nonlin = nn.Sequential(
#                 nn.BatchNorm1d(out_channels),
#                 nn.Mish(),
#                 # nn.Dropout(0.1),
#             )

#     def bn(self, x):
        
#         batch_size, num_nodes, num_channels = x.size()
#         x = x.view(-1, num_channels)
#         x = self.nonlin(x)
#         x = x.view(batch_size, num_nodes, num_channels)
        
#         return x

#     def forward(self, x, adj, mask=None):
        
#         x = self.conv1(x, adj, mask)
#         x = self.bn(x)
        
#         return x
    

# class DiffPool(nn.Module):
#     def __init__(self, hidden_dim, output_dim, max_num, red_node, edge=None):
#         super().__init__()
        
#         self.max_num = max_num
#         self.edge = edge
        
#         # self.red_num = ceil(max_num * red_rate)
        
#         self.gnn1_pool = GNN(hidden_dim, red_node, nl_mode='1p')
#         self.gnn1_embed = GNN(hidden_dim, hidden_dim, nl_mode='1e')
#         self.gnn2_embed = GNN(hidden_dim, output_dim, nl_mode='2e')

#         # self.trans = nn.Linear(hidden_dim, hidden_dim) 
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Mish(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Dropout(0.1),
#         )
        
#     def calc_attr(self, data):
        
#         row, col = data.edge_index
#         dist = torch.norm(data.pos[row] - data.pos[col], dim=1)

#         total_size = data.edge_index.size(1)
#         inter_size = data.edge_index_inter.size(1)
#         intra_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         inter_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         intra_size = total_size if self.edge in ["intra", "intra_lig", "intra_pro"] else total_size - inter_size
#         intra_mask[:intra_size] = True
#         inter_mask[intra_size:] = True
        
#         dist[(intra_mask & (dist < 0.5))] = 1.5
#         dist[(inter_mask & (dist < 0.5))] = 4.3
#         dist_attr = 4 / (torch.sqrt(dist) + 1)
        
#         return dist_attr

#     def process_data(self, x, data):
        
#         switch = {
#             "intra": data.edge_index_intra,
#             "inter": data.edge_index_inter,
#             "complex": data.edge_index_complex,
#             "intra_lig|inter": data.edge_index_intra_lig_inter,
#             "intra_pro|inter": data.edge_index_intra_pro_inter,
#             "intra_lig": data.edge_index_intra_lig,
#             "intra_pro": data.edge_index_intra_pro
#         }
        
#         # process
#         data.x = x
#         data.edge_index = switch.get(self.edge, None)
#         data.edge_attr = self.calc_attr(data)
#         adj = to_dense_adj(data.edge_index, data.batch, edge_attr=data.edge_attr, max_num_nodes=self.max_num)
#         x, mask = to_dense_batch(data.x, data.batch, max_num_nodes=self.max_num, fill_value=0)
        
#         return x, adj, mask
    
#     def forward(self, x, data):
        
#         # x = self.trans(x)
#         x, adj, mask = self.process_data(x, data)
#         s = self.gnn1_pool(x, adj, mask)
#         x = self.gnn1_embed(x, adj, mask)
#         x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
#         x = self.gnn2_embed(x, adj)
#         x = x.mean(dim=1)
#         # x = x.sum(dim=1) / 600
#         x = self.fc(x)
        
#         return x

# class GIGN(nn.Module):
#     def __init__(self, node_dim, hidden_dim, red_node=None, mode=None):
#         super().__init__()
        
#         self.mode = mode
#         self.lin_node = nn.Sequential(
#             nn.Linear(node_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Mish(),
#             # nn.Dropout(0.1),
#         )
#         self.fc = FC(hidden_dim, hidden_dim, 2, 0.1, 1)
#         self.featsblock1 = FeatsBlock(hidden_dim)
        
#         if agg == 'sum' or agg == 'att':
#             out_dim_list = [hidden_dim for _ in range(7)]
#         elif agg == 'cat':
#             out_dim_list = [hidden_dim // len(mode) if str(i + 1) in mode else 0 for i in range(7)]
#             remaining_value = hidden_dim - sum(out_dim_list)
#             last_nonzero_index = len(out_dim_list) - 1
#             while last_nonzero_index >= 0 and out_dim_list[last_nonzero_index] == 0:
#                 last_nonzero_index -= 1
#             if last_nonzero_index >= 0:
#                 out_dim_list[last_nonzero_index] += remaining_value
        
#         adj_list = [600 for _ in range(7)]

#         self.diffpool1 = DiffPool(hidden_dim, out_dim_list[0], adj_list[0], 150, edge="intra_lig|inter") if '1' in mode else None
#         self.diffpool2 = DiffPool(hidden_dim, out_dim_list[1], adj_list[1], 75, edge="intra_lig") if '2' in mode else None
#         self.diffpool3 = DiffPool(hidden_dim, out_dim_list[2], adj_list[2], 150, edge="intra_pro|inter") if '3' in mode else None
#         self.diffpool4 = DiffPool(hidden_dim, out_dim_list[3], adj_list[3], 150, edge="intra_pro") if '4' in mode else None
#         self.diffpool5 = DiffPool(hidden_dim, out_dim_list[4], adj_list[4], 150, edge="intra") if '5' in mode else None
#         self.diffpool6 = DiffPool(hidden_dim, out_dim_list[5], adj_list[5], 150, edge="inter") if '6' in mode else None
#         self.diffpool7 = DiffPool(hidden_dim, out_dim_list[6], adj_list[6], 150, edge="complex") if '7' in mode else None

#         if agg == 'att':
#             self.emb = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim // 4),
#                 # nn.Mish(),
#                 # nn.Dropout(0.1),
#                 nn.Tanh()
#             ) 
#             self.context = nn.Linear(hidden_dim // 4, 1)
        
#     def forward(self, data):
        
#         # data processing
#         data.edge_index_complex = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)
#         data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
#         data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]
#         data.edge_index_intra_lig_inter = torch.cat([data.edge_index_intra_lig, data.edge_index_inter], dim=1)
#         data.edge_index_intra_pro_inter = torch.cat([data.edge_index_intra_pro, data.edge_index_inter], dim=1)
        
#         # Node
#         x = data.x
#         x = self.lin_node(x)
#         x = self.featsblock1(x, data)
        
#         # DiffPool
#         x_list = []
#         x_list.append(self.diffpool1(x, data)) if '1' in self.mode else None
#         x_list.append(self.diffpool2(x, data)) if '2' in self.mode else None
#         x_list.append(self.diffpool3(x, data)) if '3' in self.mode else None
#         x_list.append(self.diffpool4(x, data)) if '4' in self.mode else None
#         x_list.append(self.diffpool5(x, data)) if '5' in self.mode else None
#         x_list.append(self.diffpool6(x, data)) if '6' in self.mode else None
#         x_list.append(self.diffpool7(x, data)) if '7' in self.mode else None
        
#         if agg == 'sum':
#             x = torch.sum(torch.stack(x_list, dim=0), dim=0)
#         elif agg == 'att':
#             x = torch.stack(x_list, dim=1) # (batch, 3, hidden_dim)
#             att = self.emb(x) # (batch, 3, hidden_dim // 4)
#             att = F.softmax(self.context(att), dim=1) # (batch, 3, 1)
#             x = torch.matmul(x.transpose(1, 2), att).squeeze(-1) # (batch, hidden_dim)
#         elif agg == 'cat':
#             x = torch.cat(x_list, dim=-1)
        
#         # FC
#         x = self.fc(x)

#         return x.view(-1)

# mode_list = ['125']
# swa_lr = 1e-5
# # red_rate = 0.4

# # lr_list = [3.5e-4, 4e-4, 4.5e-4]
# lr_list = [5e-4, 5.5e-4, 6e-4]

# red_node_list = [125, 150, 175]

# agg = 'sum'
# adj = '600'
# scheduler_bool = False
# explain = f""

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# # %%
# import torch
# import torch.nn as nn
# from torch_geometric.utils import to_dense_adj, to_dense_batch
# from torch_geometric.nn import DenseGCNConv, dense_diff_pool, global_mean_pool, global_add_pool
# from HIL import HIL
# from math import ceil
# import os
# import math

# class FC(nn.Module):
#     def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
#         super(FC, self).__init__()
        
#         self.d_graph_layer = d_graph_layer
#         self.d_FC_layer = d_FC_layer
#         self.n_FC_layer = n_FC_layer
#         self.dropout = dropout
#         self.predict = nn.ModuleList()
        
#         for j in range(self.n_FC_layer):
#             if j == 0:
#                 self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))
#             if j == self.n_FC_layer - 1:
#                 self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
#             else:
#                 self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#                 self.predict.append(nn.Mish())
#                 self.predict.append(nn.Dropout(self.dropout))

#     def forward(self, h):
        
#         for layer in self.predict:
#             h = layer(h)
            
#         return h

# class FeatsBlock(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FeatsBlock, self).__init__()
        
#         self.gconv_intra = HIL(input_dim, output_dim)
#         self.gconv_inter = HIL(input_dim, output_dim)

#     def forward(self, x, data):
        
#         intra_x = self.gconv_intra(x, data, data.edge_index_intra)
#         inter_x = self.gconv_inter(x, data, data.edge_index_inter)
#         x = intra_x + inter_x
        
#         return x

# class GNN(nn.Module): 
#     def __init__(self, in_channels, out_channels, nl_bool=True):
#         super().__init__()
        
#         self.nl_bool = nl_bool
#         self.conv1 = DenseGCNConv(in_channels, out_channels)
#         self.nonlin = nn.Sequential(
#             nn.BatchNorm1d(out_channels),
#             nn.Mish(),
#             # nn.Dropout(0.1),
#         )

#     def bn(self, x):
        
#         batch_size, num_nodes, num_channels = x.size()
#         x = x.view(-1, num_channels)
#         x = self.nonlin(x)
#         x = x.view(batch_size, num_nodes, num_channels)
        
#         return x

#     def forward(self, x, adj, mask=None):
        
#         x = self.conv1(x, adj, mask)
#         x = self.bn(x) if self.nl_bool else x
        
#         return x
    

# class DiffPool(nn.Module):
#     def __init__(self, hidden_dim, output_dim, max_num, edge=None):
#         super().__init__()
        
#         self.max_num = max_num
#         self.red_num = ceil(max_num * red_rate)
#         self.edge = edge
        
#         self.gnn1_pool = GNN(hidden_dim, self.red_num, nl_bool=True)
#         self.gnn1_embed = GNN(hidden_dim, hidden_dim, nl_bool=True)
#         self.gnn2_embed = GNN(hidden_dim, output_dim, nl_bool=True)

#     def calc_attr(self, data):
        
#         row, col = data.edge_index
#         dist = torch.norm(data.pos[row] - data.pos[col], dim=1)

#         total_size = data.edge_index.size(1)
#         inter_size = data.edge_index_inter.size(1)
#         intra_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         inter_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         intra_size = total_size if self.edge in ["intra", "intra_lig", "intra_pro"] else total_size - inter_size
#         intra_mask[:intra_size] = True
#         inter_mask[intra_size:] = True
        
#         dist[(intra_mask & (dist < 0.5))] = 1.5
#         dist[(inter_mask & (dist < 0.5))] = 4.3
#         dist_attr = 4 / (torch.sqrt(dist) + 1)
        
#         return dist_attr

#     def process_data(self, x, data):
        
#         switch = {
#             "intra": data.edge_index_intra,
#             "inter": data.edge_index_inter,
#             "complex": data.edge_index_complex,
#             "intra_lig|inter": data.edge_index_intra_lig_inter,
#             "intra_pro|inter": data.edge_index_intra_pro_inter,
#             "intra_lig": data.edge_index_intra_lig,
#             "intra_pro": data.edge_index_intra_pro
#         }
        
#         # process
#         data.x = x
#         data.edge_index = switch.get(self.edge, None)
#         data.edge_attr = self.calc_attr(data)
#         adj = to_dense_adj(data.edge_index, data.batch, edge_attr=data.edge_attr, max_num_nodes=self.max_num)
#         x, mask = to_dense_batch(data.x, data.batch, max_num_nodes=self.max_num, fill_value=0)
        
#         return x, adj, mask
    
#     def forward(self, x, data):
#         x, adj, mask = self.process_data(x, data)
#         s = self.gnn1_pool(x, adj, mask)
#         x = self.gnn1_embed(x, adj, mask)
#         x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
#         x = self.gnn2_embed(x, adj)
#         x = x.sum(dim=1) / math.sqrt(600 * self.red_num)

#         return x

# class TokenEmbedding(nn.Module):
#     def __init__(self, d_model):
#         super(TokenEmbedding, self).__init__()
        
#         self.token_embedding = nn.Embedding(3, d_model)

#     def forward(self, indices, q_size):
        
#         out = self.token_embedding(indices)
#         out = out.unsqueeze(0).repeat(q_size, 1, 1)
        
#         return out

# class GIGN(nn.Module):
#     def __init__(self, node_dim, hidden_dim):
#         super().__init__()
        
#         self.lin_node = nn.Sequential(
#             nn.Linear(node_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Mish(),
#             # nn.Dropout(0.1),
#         )
#         self.token_emb = TokenEmbedding(hidden_dim)
#         self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)
#         self.featsblock1 = FeatsBlock(hidden_dim, hidden_dim)
        
#         if agg == 'sum':
#             out_dim_list = [hidden_dim for _ in range(7)]
#         elif agg == 'cat':
#             out_dim_list = [hidden_dim // len(mode) if str(i + 1) in mode else 0 for i in range(7)]
#             remaining_value = hidden_dim - sum(out_dim_list)
#             last_nonzero_index = len(out_dim_list) - 1
#             while last_nonzero_index >= 0 and out_dim_list[last_nonzero_index] == 0:
#                 last_nonzero_index -= 1
#             if last_nonzero_index >= 0:
#                 out_dim_list[last_nonzero_index] += remaining_value
            
#         self.diffpool1 = DiffPool(hidden_dim, out_dim_list[0], 400, edge="intra_lig|inter") if '1' in mode else None
#         self.diffpool2 = DiffPool(hidden_dim, out_dim_list[1], 200, edge="intra_lig") if '2' in mode else None
#         self.diffpool3 = DiffPool(hidden_dim, out_dim_list[2], 600, edge="intra_pro|inter") if '3' in mode else None
#         self.diffpool4 = DiffPool(hidden_dim, out_dim_list[3], 500, edge="intra_pro") if '4' in mode else None
#         self.diffpool5 = DiffPool(hidden_dim, out_dim_list[4], 600, edge="intra") if '5' in mode else None
#         self.diffpool6 = DiffPool(hidden_dim, out_dim_list[5], 600, edge="inter") if '6' in mode else None
#         self.diffpool7 = DiffPool(hidden_dim, out_dim_list[6], 600, edge="complex") if '7' in mode else None
        
#     def forward(self, data):
        
#         # data processing
#         data.edge_index_complex = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)
#         data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
#         data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]
#         data.edge_index_intra_lig_inter = torch.cat([data.edge_index_intra_lig, data.edge_index_inter], dim=1)
#         data.edge_index_intra_pro_inter = torch.cat([data.edge_index_intra_pro, data.edge_index_inter], dim=1)
        
#         # Node
#         x = data.x
#         x = self.lin_node(x)
#         x = self.featsblock1(x, data)
        
#         token_idx = torch.tensor([0, 1, 2], device=x.device)  # [1, 2, 5]
#         token_idx.requires_grad = False
#         token_emb = self.token_emb(token_idx, x.size(0)) # (1, 3, d_k) -> (n_batch, 3, d_k)
#         x1, x2, x3 = torch.split(token_emb, 1, dim=1)
#         x1, x2, x3 = x1.squeeze(1), x2.squeeze(1), x3.squeeze(1)
        
#         # DiffPool
#         x_list = []
#         x_list.append(self.diffpool1(x+x1, data)) if '1' in mode else None
#         x_list.append(self.diffpool2(x+x2, data)) if '2' in mode else None
#         x_list.append(self.diffpool3(x, data)) if '3' in mode else None
#         x_list.append(self.diffpool4(x, data)) if '4' in mode else None
#         x_list.append(self.diffpool5(x+x3, data)) if '5' in mode else None
#         x_list.append(self.diffpool6(x, data)) if '6' in mode else None
#         x_list.append(self.diffpool7(x, data)) if '7' in mode else None
        
#         if agg == 'sum':
#             x = torch.sum(torch.stack(x_list, dim=0), dim=0)
#         elif agg == 'cat':
#             x = torch.cat(x_list, dim=-1)
        
#         # FC
#         x = self.fc(x)

#         return x.view(-1)

# mode = '125'
# lr = 5e-4
# red_rate = 0.4
# agg = 'sum'
# scheduler_bool = False
# explain = f"{agg}_tokenemb_sqrtreddiv"
# # {red_rate}_4265_explba_gnnba

# msg_info = f"{mode}, {explain}, lr={lr}"
# save_dir = f"./model/{explain}"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################






# # %%
# import torch
# import torch.nn as nn
# from torch_geometric.utils import to_dense_adj, to_dense_batch
# from torch_geometric.nn import DenseGCNConv, dense_diff_pool, global_mean_pool, global_add_pool
# from math import ceil
# from HIL import HIL
# import torch.nn.functional as F

# class FC(nn.Module):
#     def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
#         super(FC, self).__init__()
#         self.d_graph_layer = d_graph_layer
#         self.d_FC_layer = d_FC_layer
#         self.n_FC_layer = n_FC_layer
#         self.dropout = dropout
#         self.predict = nn.ModuleList()
#         for j in range(self.n_FC_layer):
#             if j == 0:
#                 self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
#                 self.predict.append(nn.Dropout(self.dropout))
#                 self.predict.append(nn.LeakyReLU())
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#             if j == self.n_FC_layer - 1:
#                 self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
#             else:
#                 self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
#                 self.predict.append(nn.Dropout(self.dropout))
#                 self.predict.append(nn.LeakyReLU())
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))

#     def forward(self, h):
#         for layer in self.predict:
#             h = layer(h)
#         return h
    
# class GNN(nn.Module): 
#     def __init__(self, in_channels, out_channels, normalize=False):
#         super().__init__()
#         self.conv1 = DenseGCNConv(in_channels, out_channels, normalize)
#         self.bn1 = nn.BatchNorm1d(out_channels)

#     def bn(self, x):
#         batch_size, num_nodes, num_channels = x.size()
#         x = x.view(-1, num_channels)
#         x = self.bn1(x)
#         x = x.view(batch_size, num_nodes, num_channels)
#         return x

#     def forward(self, x, adj, mask=None):
#         x = self.conv1(x, adj, mask).relu()
#         x = self.bn(x)
#         return x

# class FeatsBlock(nn.Module):
#     def __init__(self, hidden_dim):
#         super(FeatsBlock, self).__init__()
#         self.gconv_intra = HIL(hidden_dim, hidden_dim)
#         self.gconv_inter = HIL(hidden_dim, hidden_dim)

#     def forward(self, x, data):
#         intra_x = self.gconv_intra(x, data, data.edge_index_intra)
#         inter_x = self.gconv_inter(x, data, data.edge_index_inter)
#         x = intra_x + inter_x
#         return x

# class DiffPool(nn.Module):
#     def __init__(self, hidden_dim, max_num, edge=None):
#         super().__init__()
#         self.max_num = max_num
#         num1 = ceil(max_num * 0.25)
#         self.edge = edge
        
#         self.featsblock = FeatsBlock(hidden_dim)
#         if 'h' in mode:
#             self.embed = HIL(hidden_dim, hidden_dim)
#             self.pool = HIL(hidden_dim, num1)
#         else:
#             self.gnn1_pool = GNN(hidden_dim, num1)
#             self.gnn1_embed = GNN(hidden_dim, hidden_dim)
#             self.gnn2_embed = GNN(hidden_dim, hidden_dim)
    
#     def calc_attr(self, data):
#         row, col = data.edge_index
#         dist = torch.norm(data.pos[row] - data.pos[col], dim=1)

#         total_size = data.edge_index.size(1)
#         inter_size = data.edge_index_inter.size(1)
#         intra_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         inter_mask = torch.zeros(total_size, dtype=torch.bool, device=data.edge_index.device)
#         intra_size = total_size if self.edge in ["intra", "intra_lig", "intra_pro"] else total_size - inter_size
#         intra_mask[:intra_size] = True
#         inter_mask[intra_size:] = True
#         dist[(intra_mask & (dist < 0.5)) | (inter_mask & (dist < 0.5))] = 1.5
#         dist_attr = 4 / (torch.sqrt(dist) + 1)
        
#         return dist_attr

#     def process(self, data):
#         if 'h' in mode:
#             adj = to_dense_adj(data.edge_index, data.batch, edge_attr=data.edge_attr, max_num_nodes=self.max_num)
#             x, mask = to_dense_batch(data.x, data.batch, max_num_nodes=self.max_num, fill_value=0)
#             s, _ = to_dense_batch(data.s, data.batch, max_num_nodes=self.max_num, fill_value=0)
#             return x, adj, s, mask
#         else:
#             adj = to_dense_adj(data.edge_index, data.batch, edge_attr=data.edge_attr, max_num_nodes=self.max_num)
#             x, mask = to_dense_batch(data.x, data.batch, max_num_nodes=self.max_num, fill_value=0)
#             return x, adj, mask
    
#     def forward(self, x, data):
#         # edge_index
#         switch = {
#             "intra": data.edge_index_intra,
#             "inter": data.edge_index_inter,
#             "complex": torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1),
#             "intra_lig|inter": torch.cat([data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0], data.edge_index_inter], dim=1),
#             "intra_pro|inter": torch.cat([data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1], data.edge_index_inter], dim=1),
#             "intra_lig": data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0],
#             "intra_pro": data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]
#         }
#         data.edge_index = switch.get(self.edge, None)
        
#         x = self.featsblock(x, data)
#         if 'h' in mode:
#             # GNN
#             s = self.pool(x, data, data.edge_index)
#             x = self.embed(x, data, data.edge_index)
            
#             # process
#             data.x, data.s = x, s
#             data.edge_attr = self.calc_attr(data)
#             x, adj, s, mask = self.process(data)
#             x, _, _, _ = dense_diff_pool(x, adj, s, mask)
#             x = x.sum(dim=1)
#         else:
#             # process
#             data.x = x
#             data.edge_attr = self.calc_attr(data)
#             x, adj, mask = self.process(data)
            
#             # pooling
#             s = self.gnn1_pool(x, adj, mask)        
#             x = self.gnn1_embed(x, adj, mask)
#             x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
#             x = self.gnn2_embed(x, adj)
#             x = x.sum(dim=1)
        
#         return x

    
# class GIGN(nn.Module):
#     def __init__(self, node_dim, hidden_dim):
#         super().__init__()
#         self.lin_node = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
#         self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)
        
#         self.diffpool1 = DiffPool(hidden_dim, 600, edge="intra_lig|inter") if '1' in mode else None
#         self.diffpool2 = DiffPool(hidden_dim, 600, edge="intra_lig") if '2' in mode else None
#         self.diffpool3 = DiffPool(hidden_dim, 600, edge="intra_pro|inter") if '3' in mode else None
#         self.diffpool4 = DiffPool(hidden_dim, 600, edge="intra_pro") if '4' in mode else None
#         self.diffpool5 = DiffPool(hidden_dim, 600, edge="intra") if '5' in mode else None
#         self.diffpool6 = DiffPool(hidden_dim, 600, edge="inter") if '6' in mode else None
#         self.diffpool7 = DiffPool(hidden_dim, 600, edge="complex") if '7' in mode else None
        
#     def forward(self, data):
#         # data processing
#         x = data.x
#         x = self.lin_node(x)
        
#         # DiffPool
#         x_list = []
#         x_list.append(self.diffpool1(x, data)) if '1' in mode else None # 128 x 150 x 256 -> 128 x 256
#         x_list.append(self.diffpool2(x, data)) if '2' in mode else None
#         x_list.append(self.diffpool3(x, data)) if '3' in mode else None
#         x_list.append(self.diffpool4(x, data)) if '4' in mode else None
#         x_list.append(self.diffpool5(x, data)) if '5' in mode else None
#         x_list.append(self.diffpool6(x, data)) if '6' in mode else None
#         x_list.append(self.diffpool7(x, data)) if '7' in mode else None
#         x_list.append(global_add_pool(x[data.split == 0], data.batch[data.split == 0])) if 'l' in mode else None
#         x_list.append(global_add_pool(x[data.split == 1], data.batch[data.split == 1])) if 'p' in mode else None
        
#         # FC
#         x = torch.sum(torch.stack(x_list), dim=0)
#         x = self.fc(x)

#         return x.view(-1)

# mode = '125'
# msg_info = f"{mode}, 600, "
# lr = 4e-4








###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################










#         self.attr = False
#     def calc_attr(self, data, edge_index):
#         row, col = edge_index
#         dist = torch.norm(data.pos[row] - data.pos[col], dim=1)

#         total_size = edge_index.size(1)
#         inter_size = data.edge_index_inter.size(1)
#         intra_mask = torch.zeros(total_size, dtype=torch.bool, device=edge_index.device)
#         inter_mask = torch.zeros(total_size, dtype=torch.bool, device=edge_index.device)
#         intra_size = total_size if self.mode in ["intra", "intra_lig", "intra_pro"] else total_size - inter_size
#         intra_mask[:intra_size] = True
#         inter_mask[intra_size:] = True
#         dist[intra_mask & (dist < 0.5)] = 1.5
#         dist[inter_mask & (dist < 0.5)] = 4.3
#         dist_attr = 4 / (torch.sqrt(dist) + 1)
        
#         attr = dist_attr
                
#         return edge_index, attr


# class DiffPool(nn.Module):
#     def __init__(self, hidden_dim, max_num, mode=None):
#         super().__init__()
#         self.max_num = max_num
#         num1 = ceil(0.25 * self.max_num)
#         self.mode = mode
#         self.attr = True
#         self.cos = nn.CosineSimilarity(dim=1)
        
#         self.gnn1_pool = GNN(hidden_dim, num1)
#         self.gnn1_embed = GNN(hidden_dim, hidden_dim)
#         self.gnn2_embed = GNN(hidden_dim, hidden_dim)

#     def process(self, data):
#         if self.attr:
#             adj = to_dense_adj(data.edge_index, data.batch, edge_attr=data.edge_attr, max_num_nodes=self.max_num)
#         else:
#             adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.max_num)
#         x, mask = to_dense_batch(data.x, data.batch, max_num_nodes=self.max_num, fill_value=0)
#         return x, adj, mask

#     def calc_attr(self, data, edge_index):
#         row, col = edge_index
#         dist = torch.norm(data.pos[row] - data.pos[col], dim=1)

#         total_size = edge_index.size(1)
#         inter_size = data.edge_index_inter.size(1)
#         intra_mask = torch.zeros(total_size, dtype=torch.bool, device=edge_index.device)
#         inter_mask = torch.zeros(total_size, dtype=torch.bool, device=edge_index.device)
#         intra_size = total_size if self.mode in ["intra", "intra_lig", "intra_pro"] else total_size - inter_size
#         intra_mask[:intra_size] = True
#         inter_mask[intra_size:] = True
#         dist[intra_mask & (dist < 0.5)] = 1.5
#         dist[inter_mask & (dist < 0.5)] = 4.3
#         dist_attr = 4 / (torch.sqrt(dist) + 1)
        
#         # cos_attr = self.cos(data.x[row], data.x[col]) + 1 
#         attr = dist_attr
                
#         return edge_index, attr
    
#     def forward(self, x, data):
#         if self.mode == "intra":
#             edge_index = data.edge_index_intra
#         elif self.mode == "inter":
#             edge_index = data.edge_index_inter
#         elif self.mode == "complex":
#             edge_index = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)
#         elif self.mode == "intra_lig|inter":
#             edge_index = torch.cat([data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0], data.edge_index_inter], dim=1)
#         elif self.mode == "intra_pro|inter":
#             edge_index = torch.cat([data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1], data.edge_index_inter], dim=1)
#         elif self.mode == "intra_lig":
#             edge_index = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
#         elif self.mode == "intra_pro":
#             edge_index = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]
        
#         # processing
#         data.x = x
#         edge_index, edge_attr = self.calc_attr(data, edge_index)
#         data.edge_index, data.edge_attr = edge_index, edge_attr
#         x, adj, mask = self.process(data)
        
#         # 2step
#         x = self.gnn1_embed(x, adj, mask)
#         s = self.gnn1_pool(x, adj, mask)
#         x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
#         x = self.gnn2_embed(x, adj)

#         return x


# class FeatsBlock(nn.Module):
#     def __init__(self, hidden_dim):
#         super(FeatsBlock, self).__init__()
#         self.gconv_intra = HIL(hidden_dim, hidden_dim, mode="intra")
#         self.gconv_inter = HIL(hidden_dim, hidden_dim, mode="inter")

#     def forward(self, x, data):
#         intra_x = self.gconv_intra(x, data, data.edge_index_intra)
#         inter_x = self.gconv_inter(x, data, data.edge_index_inter)
#         x = intra_x + inter_x
#         return x

# class GIGN(nn.Module):
#     def __init__(self, node_dim, hidden_dim):
#         super().__init__()
#         self.lin_node = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
#         self.featsblock1 = FeatsBlock(hidden_dim)
#         self.diffpool1 = DiffPool(hidden_dim, 600, mode="intra_lig|inter")
#         self.diffpool2 = DiffPool(hidden_dim, 600, mode="intra_lig")
#         self.diffpool3 = DiffPool(hidden_dim, 600, mode="intra")
#         self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

#     def forward(self, data):
#         # data processing
#         x = data.x
#         x = self.lin_node(x)

#         # GIGN
#         x = self.featsblock1(x, data)

#         # DiffPool
#         x1 = self.diffpool1(x, data).mean(dim=1)
#         x2 = self.diffpool2(x, data).mean(dim=1)
#         x3 = self.diffpool3(x, data).mean(dim=1)

#         x = x1 + x2 + x3
#         # x = torch.cat([x1, x2, x3], dim=1)
#         x = self.fc(x)

#         return x.view(-1)











        # clusters = s.argmax(dim=2)
        # batch_idx = 0
        # pos = data.pos[data.batch == batch_idx].cpu().numpy()
        # b, n1, n2 = s.size()
        # # Convert edge_index to a networkx graph
        # tmp = data.edge_index[:, data.batch[data.edge_index[0, :]] == batch_idx].cpu().numpy()
        # G = nx.Graph()
        # G.add_edges_from(tmp.T)

        # # Get clusters for the first batch
        # node_clusters = clusters[batch_idx].cpu().numpy()

        # # Initialize the 3D plot
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # Get node positions and colors
        # node_positions = {i: pos[i] for i in range(len(pos))}
        # node_colors = [node_clusters[node] for node in G.nodes()]


        # # Draw nodes
        # for node, (x, y, z) in node_positions.items():
        #     ax.scatter(x, y, z, color=plt.cm.jet(node_colors[node] / n2), s=10, edgecolor='k')

        # # Draw edges
        # for edge in G.edges():
        #     x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
        #     y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
        #     z_coords = [node_positions[edge[0]][2], node_positions[edge[1]][2]]
        #     ax.plot(x_coords, y_coords, z_coords, color='k', alpha=0.5)

        # # Set labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # # Show plot
        # if self.mode == "intra":
        #     plt.savefig('test.png')




















# class DiffPool(nn.Module):
#     def __init__(self, node_dim, hidden_dim):
#         super().__init__()
#         self.max_num_lig = 200
#         self.max_num_pro = 500
#         num_lig = ceil(0.25 * self.max_num_lig)
#         num_pro = ceil(0.1 * self.max_num_pro)
        
#         self.gnn1_pool_lig = GNN(hidden_dim, num_lig)
#         self.gnn1_embed_lig = GNN(hidden_dim, hidden_dim)
#         self.gnn2_embed_lig = GNN(hidden_dim, hidden_dim)
        
#         self.gnn1_pool_pro = GNN(hidden_dim, num_pro)
#         self.gnn1_embed_pro = GNN(hidden_dim, hidden_dim)
#         self.gnn2_embed_pro = GNN(hidden_dim, hidden_dim)
        
#         self.lin = nn.Linear(2 * hidden_dim, hidden_dim)

#     def process(self, data):
#         adj_lig = to_dense_adj(data.edge_index_lig, data.batch, max_num_nodes=self.max_num_lig)
#         adj_pro = to_dense_adj(data.edge_index_pro, data.batch, max_num_nodes=self.max_num_pro)
#         x_lig, mask_lig = to_dense_batch(data.x, data.batch, max_num_nodes=self.max_num_lig, fill_value=0)
#         x_pro, mask_pro = to_dense_batch(data.x, data.batch, max_num_nodes=self.max_num_pro, fill_value=0)
#         return x_lig, adj_lig, mask_lig, x_pro, adj_pro, mask_pro

#     def forward(self, x, data):
#         # # intra
#         # edge_index = data.edge_index_intra
        
#         # # inter
#         # edge_index = data.edge_index_inter
        
#         # # complex
#         # edge_index = torch.cat([data.edge_index_intra, data.edge_index_inter], dim=1)
        
#         # # intra_lig|inter
#         # edge_index = torch.cat([data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0], data.edge_index_inter], dim=1)
        
#         # intra_lig, intra_pro
#         edge_index_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
#         edge_index_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]
        
#         # processing
#         data.edge_index_lig = edge_index_lig
#         data.edge_index_pro = edge_index_pro
#         data.x = x
#         x_lig, adj_lig, mask_lig, x_pro, adj_pro, mask_pro = self.process(data)
        
#         # 2step
#         s_lig = self.gnn1_pool_lig(x_lig, adj_lig, mask_lig)
#         x_lig = self.gnn1_embed_lig(x_lig, adj_lig, mask_lig)
#         x_lig, adj_lig, l1, e1 = dense_diff_pool(x_lig, adj_lig, s_lig, mask_lig)
#         x_lig = self.gnn2_embed_lig(x_lig, adj_lig)
        
#         s_pro = self.gnn1_pool_pro(x_pro, adj_pro, mask_pro)
#         x_pro = self.gnn1_embed_pro(x_pro, adj_pro, mask_pro)
#         x_pro, adj_pro, l1, e1 = dense_diff_pool(x_pro, adj_pro, s_pro, mask_pro)
#         x_pro = self.gnn2_embed_pro(x_pro, adj_pro)
        
#         x = x_lig + x_pro
                
#         return x
    

# class IAM(MessagePassing):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(IAM, self).__init__(**kwargs)
#         self.mlp_node = nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#             nn.Dropout(0.1),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(out_channels))
#         self.mlp_coord = nn.Sequential(nn.Linear(9, in_channels), nn.SiLU())
#         self.fc = nn.Sequential(
#             nn.Linear(3 * in_channels, out_channels),
#             nn.BatchNorm1d(out_channels),
#             nn.LeakyReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(out_channels, out_channels),
#         )

#     def forward(self, x, data):
#         edge_index = data.edge_index
#         pos, size = data.pos, None
#         row, col = edge_index
#         coord_diff = pos[row] - pos[col]
#         rbf = _rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
#         radial = self.mlp_coord(rbf)
#         node_feats = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
#         out_node = self.mlp_node(x + node_feats)
#         return out_node

#     def message(self, x_j, x_i, radial, index):
#         out = self.fc(torch.cat([x_i, x_j, radial], dim=-1))
#         return out

        # def x3d_to_x2d(self, x_3d):
            #     B, N, F = x_3d.shape
            #     x_2d = x_3d.view(B * N, F)
            #     return x_2d

            # def den_to_spa(self, adj):
            #     edge_index, _ = dense_to_sparse(adj)  # Convert dense adj to edge_index
            #     edge_index = edge_index.to(torch.long)
            #     return edge_index

            # def data_process(self, x, adj, data):
            #     B, N, F = x.shape
            #     x = self.x3d_to_x2d(x) # (B * N, F)
            #     data.edge_index = self.den_to_spa(adj)
            #     data.batch = torch.repeat_interleave(torch.arange(B), N).to(x.device)
            #     return x, data


        # # 3step
        # s = self.gnn1_pool(x, adj, mask)
        # x = self.gnn1_embed(x, adj, mask)
        # x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        # s = self.gnn2_pool(x, adj)
        # x = self.gnn2_embed(x, adj)
        # x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        # x = self.gnn3_embed(x, adj)
        
        
        # # # intra_lig_con|inter
        # edge_index_intra_ligand = data.edge_index_intra[0, data.split[data.edge_index_intra[0, :]] == 0]
        # edge_index_inter_ligand = data.edge_index_inter[0, data.split[data.edge_index_inter[0, :]] == 0]

        # ligand_nodes = torch.unique(edge_index_intra_ligand)
        # inter_ligand_nodes = torch.unique(edge_index_inter_ligand)

        # connected_ligand_nodes = ligand_nodes[torch.isin(ligand_nodes, inter_ligand_nodes)]
        # print("connected_ligand_nodes", connected_ligand_nodes, connected_ligand_nodes.size())
        # print("ligand_nodes", ligand_nodes, ligand_nodes.size())
        # print("inter_ligand_nodes", inter_ligand_nodes, inter_ligand_nodes.size())
        # print('-'*10)
        
        # filtered_intra_ligand_edges = edge_index_intra_ligand[:, torch.isin(edge_index_intra_ligand[:], connected_ligand_nodes)]

        # edge_index = torch.cat([filtered_intra_ligand_edges, data.edge_index_inter], dim=1)

        # max_num = 0
        # for j in range(data.batch.max() + 1):
        #     mask = data.batch[row] == j
        #     max_num = max(max_num, mask.sum().item())
        # print(max_num)



        # row, col = edge_index
        # edge_attr = self.IAM(x, data)
        # batch = data.batch[row]
        # data_tmp = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        # new_data = self.lg(data_tmp)




# class GNN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
#         super().__init__()
#         self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
#         self.bn1 = nn.BatchNorm1d(hidden_channels)
#         self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
#         self.bn2 = nn.BatchNorm1d(hidden_channels)
#         if lin is True:
#             self.lin = nn.Linear(3 * hidden_channels + out_channels, out_channels)
#         else:
#             self.lin = None

#     def bn(self, i, x):
#         batch_size, num_nodes, num_channels = x.size()
#         x = x.view(-1, num_channels)
#         x = getattr(self, f'bn{i}')(x)
#         x = x.view(batch_size, num_nodes, num_channels)
#         return x

#     def forward(self, x, adj, mask=None):
#         x0 = x
#         x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
#         x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
#         x = self.lin(x).relu()
#         return x

# class DiffPool(nn.Module):
#     def __init__(self, node_dim, hidden_dim):
#         super().__init__()
#         num_nodes_1 = ceil(0.25 * max_nodes)
#         num_nodes_2 = ceil(0.25 * 0.25 * max_nodes)
        
#         self.gnn1_pool = GNN(hidden_dim, hidden_dim, num_nodes_1)
#         self.gnn1_embed = GNN(hidden_dim, hidden_dim, hidden_dim, lin=False)
#         self.gnn2_embed = GNN(hidden_dim, hidden_dim, hidden_dim, lin=False)
#         self.IAM = IAM(hidden_dim, hidden_dim)

#     def x3d_to_x2d(self, x_3d):
#         B, N, F = x_3d.shape
#         x_2d = x_3d.view(B * N, F)
#         return x_2d

#     def den_to_spa(self, adj):
#         edge_index, _ = dense_to_sparse(adj)  # Convert dense adj to edge_index
#         edge_index = edge_index.to(torch.long)
#         return edge_index
    
#     def data_process(self, x, adj, data):
#         B, N, F = x.shape
#         x = self.x3d_to_x2d(x) # (B * N, F)
#         data.edge_index = self.den_to_spa(adj)
#         data.batch = torch.repeat_interleave(torch.arange(B), N).to(x.device)
#         return x, data
    
#     def process(self, x, data):
#         lig_mask = data.split[data.edge_index_intra[0, :]] == 0
#         edge_index_intra_lig = data.edge_index_intra[:, lig_mask]
#         edge_index_inter = data.edge_index_inter
#         edge_index_intra = data.edge_index_intra
#         edge_index = torch.cat([edge_index_intra_lig, edge_index_inter], dim=1)
#         # edge_index = data.edge_index_inter
#         adj = to_dense_adj(edge_index, data.batch, max_num_nodes=max_nodes)
#         x, mask = to_dense_batch(x, data.batch, max_num_nodes=max_nodes, fill_value=0)
#         return x, adj, mask
    
#     # def forward(self, x, data):
#     #     x, adj, mask = self.process(x, data)
        
#     #     s = self.gnn1_pool(x, adj, mask)
#     #     x = self.gnn1_embed(x, adj, mask)
#     #     x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

#     #     s = self.gnn2_pool(x, adj)
#     #     x = self.gnn2_embed(x, adj)
#     #     x, adj, l2, e2 = dense_diff_pool(x, adj, s)

#     #     x = self.gnn3_embed(x, adj)
        
#     #     x, data = self.data_process(x, adj, data)
#     #     return x, data

#     def forward(self, x, data):
#         edge_index = data.edge_index_inter
#         row, col = edge_index
#         src = x[row]
#         dst = x[col]

#         s = self.gnn1_pool(x, adj, mask)
#         x = self.gnn1_embed(x, adj, mask)
#         x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

#         s = self.gnn2_pool(x, adj)
#         x = self.gnn2_embed(x, adj)
#         x, adj, l2, e2 = dense_diff_pool(x, adj, s)

#         x = self.gnn3_embed(x, adj)
        
#         # x, data = self.data_process(x, adj, data)
#         return x, data





























# class GIGN(nn.Module):
#     def __init__(self, node_dim, hidden_dim):
#         super().__init__()
#         # base
#         self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
#         self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)
#         # Feats
#         self.featsblock1 = FeatsBlock(hidden_dim)
#         self.featsblock2 = FeatsBlock(hidden_dim)
#         self.featsblock3 = FeatsBlock(hidden_dim)

#     def forward(self, data):
#         # data processing
#         x = data.x
#         x = self.lin_node(x)

#         # iteration
#         x = self.featsblock1(x, data)
#         x = self.featsblock2(x, data)
#         x = self.featsblock3(x, data)

#         # output
#         x = global_add_pool(x, data.batch)
#         x = self.fc(x)

#         return x.view(-1)

# class FeatsBlock(nn.Module):
#     def __init__(self, hidden_dim):
#         super(FeatsBlock, self).__init__()
#         # base
#         self.gconv_intra = HIL(hidden_dim, hidden_dim, mode="intra")
#         self.gconv_inter = HIL(hidden_dim, hidden_dim, mode="inter")

#     def forward(self, x, data):
#         intra_x = self.gconv_intra(x, data, data.edge_index_intra)
#         inter_x = self.gconv_inter(x, data, data.edge_index_inter)
#         x = intra_x + inter_x
#         return x

# class FC(nn.Module):
#     def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
#         super(FC, self).__init__()
#         self.d_graph_layer = d_graph_layer
#         self.d_FC_layer = d_FC_layer
#         self.n_FC_layer = n_FC_layer
#         self.dropout = dropout
#         self.predict = nn.ModuleList()
#         for j in range(self.n_FC_layer):
#             if j == 0:
#                 self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
#                 self.predict.append(nn.Dropout(self.dropout))
#                 self.predict.append(nn.LeakyReLU())
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))
#             if j == self.n_FC_layer - 1:
#                 self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
#             else:
#                 self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
#                 self.predict.append(nn.Dropout(self.dropout))
#                 self.predict.append(nn.LeakyReLU())
#                 self.predict.append(nn.BatchNorm1d(d_FC_layer))

#     def forward(self, h):
#         for layer in self.predict:
#             h = layer(h)
#         return h
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# class FeatsBlock(nn.Module):
#     def __init__(self, hidden_dim):
#         super(FeatsBlock, self).__init__()
#         # base
#         self.gconv1 = HIL(hidden_dim, hidden_dim)
#         self.gconv2 = HIL(hidden_dim, hidden_dim)
        
#     def forward(self, x, data):
        # ### exp4
        # batch = data.batch
        # # Intra graph
        # intra_x = self.gconv1(x, data, data.edge_index_intra)
        # # Interaction-inter
        # src = self.fc1_inter(intra_x[data.edge_index_inter[0]])
        # dst = self.fc2_inter(intra_x[data.edge_index_inter[1]])
        # mask = torch.cat([src, dst], dim=1)
        # mask = self.att_inter(mask)
        # # Inter graph
        # inter_x = self.gconv2(intra_x, data, data.edge_index_inter, mask=mask)
        # x = intra_x + inter_x
        # return x, 0, data
    
        # ### exp8
        # batch = data.batch
        # # Interaction-intra
        # src = self.fc1_intra(x[data.edge_index_intra[0]])
        # dst = self.fc2_intra(x[data.edge_index_intra[1]])
        # mask = torch.cat([src, dst], dim=1)
        # mask = self.att_intra(mask)
        # # mask = F.sigmoid(mask)
        # # Intra graph
        # intra_x = x + self.gconv1(x, data, data.edge_index_intra, mask=mask)
        # # Interaction-inter
        # src = self.fc1_inter(intra_x[data.edge_index_inter[0]])
        # dst = self.fc2_inter(intra_x[data.edge_index_inter[1]])
        # mask = torch.cat([src, dst], dim=1)
        # mask = self.att_inter(mask)
        # # mask = F.sigmoid(mask) 
        # # Inter graph
        # inter_x = intra_x + self.gconv2(intra_x, data, data.edge_index_inter, mask=mask)
        # x = inter_x
        # return x, data
    
        ######################
        # # AE learing
        # pos_edge_index = data.edge_index_inter
        # z = self.AElayer.encode(x, pos_edge_index)
        # neg_edge_index = batched_negative_sampling(pos_edge_index, batch)
        # loss_link = self.AElayer.recon_loss_custom(z, batch, pos_edge_index, neg_edge_index=neg_edge_index) \
        #     + (1 / data.x.size(0)) * self.AElayer.kl_loss()
        ######################
        # mask = self.trans2(self.fc2(torch.cat([src, dst], dim=1)))
        # mask = F.cosine_similarity(src, dst, dim=1).unsqueeze(1)
        # mask_hard = (mask > 0.5).to(mask.dtype)
        # mask = (mask_hard - mask).detach() + mask
        # Inter graph
        ######################
        # edge_index_intra_lig_mask = data.split[data.edge_index_intra[0, :]] == 0
        # edge_index_intra_pro_mask = data.split[data.edge_index_intra[0, :]] == 1
        # edge_index_intra_lig = data.edge_index_intra[:, edge_index_intra_lig_mask]
        # edge_index_intra_pro = data.edge_index_intra[:, edge_index_intra_pro_mask]
        
        # z_lig = self.AElayer.encode(x, edge_index_intra_lig)
        # z_pro = self.AElayer.encode(x, edge_index_intra_pro)
        
        # node_num_cum = 0
        # cossim_adj = torch.zeros((data.x.size(0), data.x.size(0)))
        # for i in range(batch.max().item() + 1):
        #     divider = torch.logical_and(data.split == 0, batch == i).sum().item()
        #     batch_z_lig = z_lig[batch == i]
        #     batch_z_pro = z_pro[batch == i]
        #     node_num = batch_z_lig.size(0)
        #     batch_cossim = F.cosine_similarity(batch_z_lig[None,:,:], batch_z_pro[:,None,:], dim=-1)
        #     cossim_adj[node_num_cum:node_num_cum+node_num, node_num_cum:node_num_cum+node_num] = batch_cossim
        #     batch_cossim[:divider, :divider] = 0
        #     batch_cossim[divider:, divider:] = 0
        #     node_num_cum += node_num
        # inter_edge_mask = cossim_adj > 0.5
        # new_edge_index_inter = dense_to_sparse(inter_edge_mask)[0].to('cuda')
        ######################
    
    # def transform_graph(self, x, mode, data):
        # batch = data.batch
        # edge_index = data.edge_index_intra if mode == "intra" else data.edge_index_inter
        # z = self.AElayer.encode(x, edge_index)
        # loss_link = self.AElayer.recon_loss_custom(z, batch, edge_index) + (1 / data.x.size(0)) * self.AElayer.kl_loss()
        
        
        # def edge_change(data, batch, z, i):
        #     divider = (data.split == 0).sum(-1)
        #     batch_z = z[batch == i, :]
        #     A = torch.sigmoid(torch.matmul(batch_z, batch_z.t()))
        #     A_hard = (A > self.threshold).to(A.dtype)
        #     A = (A_hard - A).detach() + A
        #     if mode == "intra":
        #         A[:divider, divider:] = 0
        #         A[divider:, :divider] = 0
        #     else:
        #         A[:divider, :divider] = 0
        #         A[divider:, divider:] = 0
        #     return dense_to_sparse(A)[0]
        # edge_change_index = torch.cat([edge_change(data_list[i], batch, z, i) for i in range(len(data_list))], dim=1)
        # unique_edge_index = torch.unique(edge_change_index, dim=1)
        # if mode == "intra":
        #     data.edge_index_intra = unique_edge_index
        # else:
        #     data.edge_index_inter = unique_edge_index
        # return data, loss_link
    

# class GAEencoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GAEencoder, self).__init__()
#         self.conv1 = GATv2Conv(in_channels, 2 * out_channels, heads=4, concat=False)
#         self.conv2 = GATv2Conv(2 * out_channels, out_channels, heads=4, concat=False)
#         self.bn1 = nn.BatchNorm1d(2 * out_channels)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.dropout(self.bn1(x))
#         x = self.bn2(self.conv2(x, edge_index))
#         return x

# class VGAEencoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(VGAEencoder, self).__init__()
#         # Encoder
#         self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
#         self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
#         self.conv_logvar = GCNConv(2 * out_channels, out_channels, cached=False)
#         # self.conv1 = GATv2Conv(in_channels, 2 * out_channels, heads=4, concat=False)
#         # self.conv_mu = GATv2Conv(2 * out_channels, out_channels, heads=4, concat=False)
#         # self.conv_logvar = GATv2Conv(2 * out_channels, out_channels, heads=4, concat=False)
#         # BatchNorm
#         self.bn1 = nn.BatchNorm1d(2 * out_channels)
#         self.bn_mu = nn.BatchNorm1d(out_channels)
#         self.bn_logvar = nn.BatchNorm1d(out_channels)
#         # Dropout
#         self.dropout = nn.Dropout(0.5)
#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.dropout(self.bn1(x))
#         return self.bn_mu(self.conv_mu(x, edge_index)), self.bn_logvar(self.conv_logvar(x, edge_index))


# class GAElayer(GAE):
#     def __init__(self, encoder):
#         super(GAElayer, self).__init__(encoder)
#         self.encoder = encoder

# class VGAElayer(VGAE):
#     def __init__(self, encoder):
#         super(VGAElayer, self).__init__(encoder)
#         self.encoder = encoder

#     def recon_loss_custom(self, z, batch, pos_edge_index, neg_edge_index=None):
#         pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-10).mean()
#         if neg_edge_index is None:
#             neg_edge_index = batched_negative_sampling(pos_edge_index, batch)
#         neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-10).mean()
#         return pos_loss + neg_loss
    

        
# %%