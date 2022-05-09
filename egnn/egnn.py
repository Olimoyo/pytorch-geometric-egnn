import os.path as osp

import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from typing import Union, Tuple


class ResWrapper(torch.nn.Module):
    def __init__(self, module, dim_res=2):
        super(ResWrapper, self).__init__()
        self.module = module
        self.dim_res = dim_res

    def forward(self, x):
        res = x[:, :self.dim_res]
        out = self.module(x)
        return out + res

class EGNN(MessagePassing):
    """EGNN layer from https://arxiv.org/pdf/2102.09844.pdf"""
    def __init__(self,
                 channels_h: Union[int, Tuple[int, int]],
                 channels_m: Union[int, Tuple[int, int]], 
                 channels_a: Union[int, Tuple[int, int]],
                 aggr: str = 'add', 
                 hidden_channels: int = 64,
                 **kwargs):
        super(EGNN, self).__init__(aggr=aggr, **kwargs)

        self.phi_e = nn.Sequential(
                nn.Linear(2 * channels_h + 1 + channels_a, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, channels_m),
                nn.LayerNorm(channels_m),
                nn.SiLU()
        )
        self.phi_x = nn.Sequential(
                nn.Linear(channels_m, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
        )
        self.phi_h = ResWrapper(
            nn.Sequential(
                nn.Linear(channels_h + channels_m, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, channels_h),
            ),
            dim_res=channels_h
        )
    
    def forward(self, x, h, edge_attr, edge_index, c=None):
        if c is None:
            c = degree(edge_index[0], x.shape[0]).unsqueeze(-1)
        return self.propagate(edge_index=edge_index, x=x, h=h, edge_attr=edge_attr, c=c)

    def message(self, x_i, x_j, h_i, h_j, edge_attr):
        mh_ij = self.phi_e(torch.cat([h_i, h_j, torch.norm(x_i - x_j, dim=-1, keepdim=True)**2, edge_attr], dim=-1))
        mx_ij = (x_i - x_j) * self.phi_x(mh_ij)
        return torch.cat((mx_ij, mh_ij), dim=-1)

    def update(self, aggr_out, x, h, edge_attr, c):
        m_x, m_h = aggr_out[:, :self.m_len], aggr_out[:, self.m_len:]
        h_l1 = self.phi_h(torch.cat([h, m_h], dim=-1))
        x_l1 = x + (m_x / c)
        return x_l1, h_l1

def egnn_test():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    dataset = QM9(path).shuffle()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EGNN(
        channels_h=11, 
        channels_m=32, 
        channels_a=4
    ).to(device=device)

    for data in loader:
        data = data.to(device)

        # Propagate x and h through one layer
        x_l1, h_l1 =  model(
            x=data.pos,
            h=data.x,
            a=data.edge_attr,
            edge_index=data.edge_index
        )

        print("Post propagation x, h: ", x_l1.shape, h_l1.shape)
        break      

if __name__ == "__main__":
    egnn_test()
