import torch.nn as nn
import torch

# msg_IAM = f"DiffPool GIGN : 1GIGN - (->150)2step DP - 1layer GNN, DP->intra_lig|inter+intra_lig+intra, 6-6-6*100, (4/(sqrt(dist)+1)-adjattr, Diff-mean-res_sum-fc, "

class IAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IAM, self).__init__()
        
    def forward(self, x_i, x_j, index):
        return x_j  