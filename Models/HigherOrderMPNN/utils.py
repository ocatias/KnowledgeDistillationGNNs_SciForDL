"""
Taken from  https://github.com/subgraph23/homomorphism-expressivity and possibly slightly adapted
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter as pys
import torch_geometric as pyg
import torch_geometric.nn as gnn

from torch_geometric import data, datasets
from torch_geometric.transforms import BaseTransform

import os
import numpy as np

from typing import *

class Subgraph(data.Data):
    def __inc__(self, key, *args, **kwargs):
        if key in ("index_u", "index_v"): return self.num_node
        elif "index" in key: return self.num_nodes
        else: return 0


class TwoGNNPreTransform(BaseTransform):
    def __init__(self):
        super(TwoGNNPreTransform, self).__init__()
        
    def __call__(self, graph):
        node = torch.arange((N:=graph.num_nodes) ** 2).view(size=(N, N))
        adj = pyg.utils.to_dense_adj(graph.edge_index, max_num_nodes=N).squeeze(0)

        spd = torch.where(~torch.eye(N, dtype=bool) & (adj == 0), torch.full_like(adj, float("inf")), adj)
        for k in range(N): spd = torch.minimum(spd, spd[:, [k]] + spd[[k], :])

        attr, (dst, src) = graph.edge_attr, graph.edge_index
        if attr is not None and attr.ndim == 1: attr = attr[:, None]
        assert graph.x.ndim == 2
        
        stack = lambda *x: torch.stack(torch.broadcast_tensors(*x)).flatten(start_dim=1)
        
        if "time_stamp" in graph and "original_graph" in graph:
            return Subgraph(
                time_stamp=graph.time_stamp, 
                original_graph = graph.original_graph,
                num_node=N,
                num_nodes=N**2,
                x=graph.x[None].repeat_interleave(N, dim=0).flatten(end_dim=1),
                y=graph.y,
                a=attr[:, None].repeat_interleave(N, dim=1).flatten(end_dim=1) if attr is not None else None,

                e=adj.to(int).flatten(end_dim=1),
                d=spd.to(int).flatten(end_dim=1),

                index_d=node[:, 0] + node[0, :],

                index_u=torch.broadcast_to(node[0, :, None], (N, N)).flatten(),
                index_v=torch.broadcast_to(node[0, None, :], (N, N)).flatten(),

                index_uL=stack(node[:, 0] + dst[:, None], node[:, 0] + src[:, None]),
                index_vL=stack(node[0] + N * dst[:, None], node[0] + N * src[:, None]),

                index_uLF=stack(node[:, 0] + dst[:, None], node[:, 0] + src[:, None], (N * src + dst)[:, None]),
                index_vLF=stack(node[0] + N * dst[:, None], node[0] + N * src[:, None], (N * dst + src)[:, None]),
            )

        else:
            return Subgraph(
                num_node=N,
                num_nodes=N**2,
                x=graph.x[None].repeat_interleave(N, dim=0).flatten(end_dim=1),
                y=graph.y,
                a=attr[:, None].repeat_interleave(N, dim=1).flatten(end_dim=1) if attr is not None else None,

                e=adj.to(int).flatten(end_dim=1),
                d=spd.to(int).flatten(end_dim=1),

                index_d=node[:, 0] + node[0, :],

                index_u=torch.broadcast_to(node[0, :, None], (N, N)).flatten(),
                index_v=torch.broadcast_to(node[0, None, :], (N, N)).flatten(),

                index_uL=stack(node[:, 0] + dst[:, None], node[:, 0] + src[:, None]),
                index_vL=stack(node[0] + N * dst[:, None], node[0] + N * src[:, None]),

                index_uLF=stack(node[:, 0] + dst[:, None], node[:, 0] + src[:, None], (N * src + dst)[:, None]),
                index_vLF=stack(node[0] + N * dst[:, None], node[0] + N * src[:, None], (N * dst + src)[:, None]),
            )

    def __repr__(self) -> str:
        return f'TwoGNNPreTransform'

    
    