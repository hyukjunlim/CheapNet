import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing

def _rbf(D, D_min=0., D_max=6., D_count=9, device='cpu'):
    
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    
    return RBF

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
    
class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, drop_rate, output_dim):
        super(FC, self).__init__()
        
        self.predict = nn.ModuleList()
        self.predict.append(MLP(input_dim, hidden_dim, drop_rate))
        for _ in range(num_layer - 2):
            self.predict.append(MLP(hidden_dim, hidden_dim, drop_rate))
        self.predict.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, h):
        
        for layer in self.predict:
            h = layer(h)
            
        return h

class HIL(MessagePassing):
    def __init__(self, input_dim, output_dim, drop_rate, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HIL, self).__init__(**kwargs)
        
        self.mlp_coord = MLP(9, input_dim, 0.0)
        self.out = MLP(input_dim, output_dim, drop_rate)
        
    def message(self, x_j, x_i, radial, index):
        
        return x_j * radial
    
    def forward(self, x, data, edge_index):
        
        res = x

        pos, size = data.pos, None
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        dist = torch.norm(coord_diff, p=2, dim=-1)
        # print(dist.min() if dist.min() > 2 else None)
        radial = self.mlp_coord(_rbf(dist, device=x.device))
        x = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
        
        x = self.out(x) + res

        return x

class GIGNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(GIGNBlock, self).__init__()
        
        self.gconv_intra = HIL(input_dim, output_dim, drop_rate)
        self.gconv_inter = HIL(input_dim, output_dim, drop_rate)

    def forward(self, x, data):
        
        x_intra = self.gconv_intra(x, data, data.edge_index_intra)
        x_inter = self.gconv_inter(x, data, data.edge_index_inter)
        x = (x_intra + x_inter) / 2

        return x


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

class CheapNet(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_clusters=[28, 156], heads=1, drop_rate=0.1):
        super().__init__()
        
        self.embedding = MLP(node_dim, hidden_dim, 0.0)
        self.GIGNBlock1 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock2 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock3 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.diffpool1 = DiffPool(hidden_dim, hidden_dim, 600, num_clusters[0], "intra_lig", drop_rate)
        self.diffpool2 = DiffPool(hidden_dim, hidden_dim, 600, num_clusters[1], "intra_pro", drop_rate)
        self.attblock1 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.attblock2 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.fc = FC(hidden_dim, hidden_dim, 2, drop_rate, 1)

    def make_edge_index(self, data):

        data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
        data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]

    def forward(self, data):
        
        # Embedding
        x = data.x
        x = self.embedding(x)

        # GIGN
        self.make_edge_index(data)
        x = self.GIGNBlock1(x, data)
        x = self.GIGNBlock2(x, data)
        x = self.GIGNBlock3(x, data)

        # DiffPool-Attention
        x_lig, _ = self.diffpool1(x, data)
        x_pro, _  = self.diffpool2(x, data)

        l2p, _ = self.attblock1(x_lig, x_pro, x_pro)
        p2l, _ = self.attblock2(x_pro, x_lig, x_lig)
        x = l2p + p2l

        # FC
        x = self.fc(x)

        return x.view(-1)

# red_rate_list = [0.1, 0.2, 0.3, 0.4]
# red_rate_list = [0.5, 0.6, 0.7]
# red_rate_list = [0.8, 0.9, 1.0]
q_lig = [0, 20, 28, 37, 177]
q_pro = [0, 130, 156, 186, 500]

q_i_lig = 2
q_i_pro = 2

num_clusters = [q_lig[q_i_lig], q_pro[q_i_pro]]

scheduler_bool = True
lr = 10e-4
# x = 0
x = 1
os.environ["CUDA_VISIBLE_DEVICES"] = f'{x}'
explain = f"ours-lrs-{lr}-{num_clusters[0]}-{num_clusters[1]}-{x}"
only_rep = [0, 1, 2]


# %%

if __name__ == '__main__':
    import os
    import pandas as pd
    from thop import profile
    from dataset_CheapNet import GraphDataset, PLIDataLoader

    data_root = './data'
    graph_type = 'Graph_GIGN'
    test2013_dir = os.path.join(data_root, 'test2013')
    test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
    test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
    test2013_loader = PLIDataLoader(test2013_set, batch_size=1, shuffle=False, num_workers=4)

    data = next(iter(test2013_loader))
    
    model = CheapNet(35, 256)
    flops, params = profile(model, inputs=(data, ))
    print("params: ", int(params))
    print("flops: ", flops/1e9) 
# %%