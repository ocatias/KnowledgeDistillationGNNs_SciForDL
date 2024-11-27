import numpy as np
import time

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_edge, add_random_edge, coalesce, degree

class MutateGraph(BaseTransform):
    r"""Mutates a graph:
        - Changes vertex and edge features
        - Randomly adds and removes edges
    Args:
        :prob_node_feat_mut (float): probability that a feature in a node will be changed 
    """
    def __init__(self, possible_node_features, v_feat_prob, possible_edge_features = None, e_feat_prob = None, prob_node_feat_mut = 0.20, prob_edge_drop = 0.05, prob_edge_add = 0.20, train_set = None):
        self.possible_node_features = possible_node_features
        self.possible_edge_features = possible_edge_features
        self.prob_node_feat_mut = prob_node_feat_mut
        self.prob_edge_drop = prob_edge_drop
        self.prob_edge_add = prob_edge_add
        self.know_feat_distr = train_set is not None
        self.max_node_deg = 10
        
        self.original_graph = 0
        if prob_node_feat_mut == 0 and prob_edge_drop == 0 and prob_edge_add == 0:
            self.original_graph = 1

        self.v_feat_prob = v_feat_prob
        self.e_feat_prob = e_feat_prob
            
    def __call__(self, data: Data) -> Data:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        assert (self.prob_edge_drop == 0) or (edge_attr is None) or (self.possible_edge_features is not None)
        assert (self.prob_node_feat_mut == 0) or (x is None) or (self.possible_node_features is not None)
        is_undirected = data.is_undirected()
                
        # Edge_attr with len 1 are a special case, we deal with this by transforming them to len 2 and after the mutation we transform them back
        if edge_attr is not None:
            edge_attr_have_len_1 = len(edge_attr.shape) == 1
            if edge_attr_have_len_1:
                edge_attr = edge_attr.view(-1, 1)
        
        # Change vertex features
        deg = degree(edge_index[0], data.num_nodes)
        if x is not None:
            for node, dim in np.ndindex(x.shape):
                for dim in range(x.shape[1]):
                    if np.random.random() < (1 - self.prob_node_feat_mut):
                        continue

                    if self.know_feat_distr:
                        x[node, dim] = np.random.choice(self.possible_node_features[dim], p=self.v_feat_prob[int(deg[node])][dim])
                    else:
                        x[node, dim] = np.random.choice(self.possible_node_features[dim])
            data.x = x

        # Drop edges (assumes graph is undirected)
        edge_index, edge_mask = dropout_edge(edge_index, p = self.prob_edge_drop, force_undirected = is_undirected)
        
        if edge_attr is not None:
            edge_attr = edge_attr[edge_mask]
        
        # Only add edges if graph is not complete and edges exist
        if edge_index.shape[1] < x.shape[0]*(x.shape[0] - 1) and edge_index.shape[1] > 0:
        
            # Add edges (assumes graph is undirected)
            edge_index_before = torch.clone(edge_index)
            edge_attr_before = torch.clone(edge_attr)
            edge_index, added_edges = add_random_edge(edge_index, p = self.prob_edge_add, force_undirected = is_undirected)
            if edge_attr is not None:
                new_attr = torch.zeros([added_edges.shape[1] // 2, edge_attr.shape[1]], dtype = edge_attr.dtype)
                for edge, dim in np.ndindex(new_attr.shape):
                    if self.know_feat_distr:
                        new_attr[edge, dim] = np.random.choice(self.possible_edge_features[dim], p=self.e_feat_prob[dim])
                    else:
                        new_attr[edge, dim] = np.random.choice(self.possible_edge_features[dim])
                new_attr = torch.cat([new_attr, new_attr], dim = 0)
                edge_attr = torch.cat([edge_attr, new_attr], dim = 0)
            
            # Coalesce edge indices and attributes to get rid of duplicates
            edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce="min")     
            
        # Transform edge features
        if edge_attr is not None:
            if edge_attr_have_len_1:
                edge_attr = edge_attr.view(-1)
                
            data.edge_attr = edge_attr

        data.edge_index = edge_index
            
        if is_undirected:
            assert data.is_undirected()
                    
        return data
    

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.possible_node_features}, {self.possible_edge_features} {self.prob_node_feat_mut}, {self.prob_edge_drop}, {self.prob_edge_add})')