import os.path as osp

import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing

from torch import Tensor
from typing import List, Optional, Set, Union, Tuple, Callable
from torch_geometric.typing import Adj, Size

class ResWrapper(torch.nn.Module):
    def __init__(self, module, dim_res=2):
        super(ResWrapper, self).__init__()
        self.module = module
        self.dim_res = dim_res

    def forward(self, x):
        res = x[:, :self.dim_res]
        out = self.module(x)
        return out + res

class SparseEGNN(MessagePassing):
    """Sparse version of the EGNN network, where aggregation is done over neighbours only."""
    def __init__(self,
                 channels_h: Union[int, Tuple[int, int]],
                 channels_m: Union[int, Tuple[int, int]], 
                 channels_a: Union[int, Tuple[int, int]],
                 aggr: str = 'add', 
                 hidden_channels: int = 64,
                 **kwargs):
        super(SparseEGNN, self).__init__(aggr=aggr, **kwargs)

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
    
    def forward(self, x, h, a, edge_index):
        m_x, m_h = self.propagate(edge_index=edge_index, x=x, h=h, edge_attr=a)  
        h_l1 = self.phi_h(torch.cat([h, m_h], dim=-1))
        x_l1 = x + m_x
        return x_l1, h_l1   

    def message(self, x_i: Tensor, x_j: Tensor, h_i: Tensor, h_j: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        C = 1 / (x_i.shape[0] - 1)
        mh_ij = self.phi_e(torch.cat([h_i, h_j, torch.norm(x_i - x_j, dim=-1, keepdim=True), edge_attr], dim=-1))
        mx_ij = C * (x_i - x_j) * self.phi_x(mh_ij)
        return mx_ij, mh_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages. 
        Modified for EGNN update scheme from: 
        https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/message_passing.py
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, 
                                     size, kwargs)
        
        # Propagate messages over complete graph
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        mx_ij, mh_ij = self.message(**msg_kwargs)

        # Save coordinate dimension and propagate once
        x_dim = mx_ij.shape[-1]
        m_ij = torch.cat([mx_ij, mh_ij], dim=-1)

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        m_i = self.aggregate(m_ij, **aggr_kwargs)

        m_x = m_i[:, :x_dim]
        m_h = m_i[:, x_dim:]

        return m_x, m_h

def sparse_egnn_test(loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SparseEGNN(
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
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    dataset = QM9(path).shuffle()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    sparse_egnn_test(train_loader)
