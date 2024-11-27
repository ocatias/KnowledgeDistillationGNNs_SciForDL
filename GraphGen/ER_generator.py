from collections import defaultdict
import numpy as np

from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Data
import torch

from GraphGen.graph_generator import Generator
from Misc.visualize import visualize

class ER_Graph_Generator(Generator):
    def __init__(self, dataset_name,  edge_prob = 0.2, nodes_range = [10, 18]):
        super(ER_Graph_Generator, self).__init__(dataset_name)
        self.edge_prob = edge_prob
        self.nodes_range = nodes_range
        assert len(nodes_range) == 2
        self.possible_num_nodes = list(range(nodes_range[0], nodes_range[1]))
        
        # For node features sum out the degree relevance (this does not normalize as we only care if a features appear at least once)
        v_feat_prob = [[0 for x in num_feats] for num_feats in self.possible_node_features]
        for dim in range(len(self.possible_node_features)):
            for deg in range(1, len(self.v_feat_prob)):
                for feat in self.possible_node_features[dim]:
                    if self.v_feat_prob[deg][dim][feat] > 0:
                        v_feat_prob[dim][feat] = 1
        
        # Assign uniform feature distributions for all features seen in training set
        for dim in range(len(self.e_feat_prob)):
            e_feat = self.e_feat_prob[dim]
            num_non_zero_feats = len(list(filter(lambda x: x > 0, e_feat)))
            self.e_feat_prob[dim] = list(map(lambda x: 0 if x <= 0 else float(1) / num_non_zero_feats, e_feat))
                        
        for dim in range(len(self.possible_node_features)): 
            v_feat = v_feat_prob[dim]
            num_non_zero_feats = len(list(filter(lambda x: x > 0, v_feat)))
            v_feat_prob[dim] = list(map(lambda x: 0 if x <= 0 else float(1) / num_non_zero_feats, v_feat))
            
        self.v_feat_prob = v_feat_prob
    
    def generate_graph(self):
        num_nodes = np.random.choice(self.possible_num_nodes)
        edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=self.edge_prob, directed=False)
        x = torch.zeros((num_nodes, len(self.possible_node_features)), dtype=torch.int)
        for node in range(num_nodes):
            for dim in range(len(self.possible_node_features)):
                x[node, dim] = np.random.choice(self.possible_node_features[dim], p=self.v_feat_prob[dim])
                
        # Find both variants of each edge:
        edge_pointers_dict = defaultdict(lambda: [])
        for i in range(edge_index.shape[1]):
            x1, x2 = int(edge_index[0, i]), int(edge_index[1, i])
            
            if x1 > x2:
                x1, x2 = x2, x1
                
            edge_pointers_dict[(x1, x2)].append(i)
            
        edge_attr = torch.zeros((edge_index.shape[1], len(self.possible_edge_features)), dtype=torch.int)
        for edge, edge_pointers in edge_pointers_dict.items():
            for dim in range(len(self.possible_edge_features)):
                feat = np.random.choice(self.possible_edge_features[dim], p=self.e_feat_prob[dim])
                for pointer in edge_pointers:
                    edge_attr[pointer, dim] = feat 
                
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return graph
    
if __name__ == "__main__":
    generator = ER_Graph_Generator("ZINC")
    for i in range(20):
        graph = generator()
        visualize(graph, f"ER_{i}", v_feat_dim = 0, e_feat_dim = 0)