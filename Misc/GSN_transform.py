import networkx as nx

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from Misc.utils import edge_tensor_to_list



class GSN_transform(BaseTransform):
    r""" 
    """
    def __init__(self, dim = 5):
        import graph_tool as gt
        import graph_tool.topology as top

        self.gt = gt
        self.top = top

        self.dim = dim
        self.patterns_gt = []
        
        for d in range(3, dim+1):
            pattern = nx.cycle_graph(d)
            pattern_edge_list = list(pattern.edges)
            pattern_gt = gt.Graph(directed=False)
            pattern_gt.add_edge_list(pattern_edge_list)
            self.patterns_gt.append(pattern_gt)

    def __call__(self, data: Data):
        nr_vertices = data.x.shape[0]
        
       # Prepare graph
        if isinstance(data.edge_index, torch.Tensor):
            edge_index = data.edge_index.numpy()

        edge_list = edge_index.T
        graph_gt = self.gt.Graph(directed=False)
        graph_gt.add_edge_list(edge_list)
        self.gt.stats.remove_self_loops(graph_gt)
        self.gt.stats.remove_parallel_edges(graph_gt)
       
        counts_ls = []
        
        # Count patterns
        for pattern_gt in self.patterns_gt:
            
            # Count how often the pattern occurs in a graph
            counts_sorted = set()
            sub_isos = self.top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True, generator=True)
            sub_iso_sets = list(map(lambda isomorphism: tuple(isomorphism.a), sub_isos))
            for iso in sub_iso_sets:
                if tuple(sorted(iso)) not in counts_sorted:
                    counts_sorted.add(tuple(sorted(iso)))
                    
            # Count how often each vertex is part of that pattern
            counts = [0 for _ in range(nr_vertices)]
            for tple in counts_sorted:
                for vertex in tple:
                    counts[vertex] += 1
                  
            counts_ls.append(torch.tensor(counts))
                    
        counts = torch.stack(counts_ls, dim=1)
        data.x = torch.concat((counts, data.x), dim=1)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.dim})'