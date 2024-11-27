from collections import defaultdict

import torch
from torch_geometric.utils import degree
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from Misc.config import config
from Exp.parser import parse_args
from Exp.preparation import load_dataset
from  GraphGen.add_time_stamp import AddTimeStamp

max_node_deg = 20
enforced_feature_diversity_fraction =  0.02

class Generator():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
        if dataset_name == "ZINC":
            self.possible_node_features = [list(range(21))]
            self.possible_edge_features = [list(range(4))]
        elif dataset_name in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
            self.possible_node_features = list(map(lambda x: list(range(x)), get_atom_feature_dims()))
            self.possible_edge_features = list(map(lambda x: list(range(x)), get_bond_feature_dims()))
        elif dataset_name == "alchemy":
            self.possible_node_features = [[0,1] for _ in range(6)]
            self.possible_edge_features = [[0,1] for _ in range(4)]
        elif dataset_name == "qm9":
            self.possible_node_features = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], list(range(10)), [0], [0], [0], [0], list(range(5))]
            self.possible_edge_features = [[0, 1], [0, 1], [0,1], [0]]
        else: 
            raise NotImplementedError("Mutation not implemented for this dataset")
        
        self.og_graph_has_len1_node_features = False
        self.og_graph_has_len1_edge_features = False
        self.learn_from_dataset(self.possible_node_features, self.possible_edge_features)    
        self.time_stamp = AddTimeStamp(original_graph_stamp_value=0)
        
    def learn_from_dataset(self, possible_node_features, possible_edge_features):
        self.train_set, _, _ = load_dataset(parse_args({"--dataset": self.dataset_name, "--batch_size": 1}), config)
        
        v_feat_counts = [[defaultdict(lambda: 0.0) for _ in possible_node_features] for _ in range(max_node_deg)]
        e_feat_counts = [defaultdict(lambda: 0.0) for _ in possible_edge_features]
        
        self.v_feat_prob = [[[] for _ in possible_node_features] for _ in range(max_node_deg)]
        self.e_feat_prob = [[] for _ in possible_edge_features]
        
        nr_v_feat, nr_e_feat = len(possible_node_features), len(possible_edge_features)
        edges, vertices = 0, 0
        
        print("Computing feature counts")
        for graph in self.train_set:
            deg = degree(graph.edge_index[0], graph.num_nodes)
                        
            for vertex in range(graph.x.shape[0]):
                vertices += 1
                for feat in range(nr_v_feat):
                    v_feat_counts[int(deg[vertex])][feat][str(int(graph.x[vertex, feat]))] += 1    

            edge_attr = torch.clone(graph.edge_attr)
            # Ensure all edges have shape [#edges, #feats]
            if len(edge_attr.shape) == 1:
                    edge_attr = edge_attr.view(-1, 1)
                
            for edge in range(edge_attr.shape[0]):
                edges += 1
                for feat in range(nr_e_feat):
                    e_feat_counts[feat][str(int(edge_attr[edge, feat]))] += 1   
                    
        if len(graph.x.shape) == 1:
            self.og_graph_has_len1_node_features = True
            
        if len(graph.edge_attr.shape) == 1:
            self.og_graph_has_len1_edge_features = True
        
        print("Turning feature counts into probabilities")
        vertices_per_degree = [sum(v_feat_counts[d][0].values()) for d in range(max_node_deg)]
        
        for d in range(max_node_deg):
            for feature in range(nr_v_feat):
                # Enforce feature diversity for node features by ensuring that every feature has a minimum likelihood to be drawn (except never seen features)
                additional_nodes = 0
                nodes_to_add_for_diversity = int(vertices_per_degree[d] * enforced_feature_diversity_fraction)
                for key in v_feat_counts[d][feature].keys():
                    counts = v_feat_counts[d][feature][str(key)]
                    if counts < nodes_to_add_for_diversity:
                        additional_nodes += nodes_to_add_for_diversity - counts
                        v_feat_counts[d][feature][str(key)] = nodes_to_add_for_diversity
                        
                # Normalize counts to probabilites
                for key in v_feat_counts[d][feature].keys():
                    v_feat_counts[d][feature][str(key)] = v_feat_counts[d][feature][str(key)] / (vertices_per_degree[d] + additional_nodes)
        
        for feature in range(nr_e_feat):
            # Enforce feature diversity for edge features by ensuring that every feature has a minimum likelihood to be drawn (except never seen features)
            additional_edges = 0
            edges_to_add_for_diversity = int(edges * enforced_feature_diversity_fraction)
            for key in e_feat_counts[feature].keys():
                counts = e_feat_counts[feature][str(key)]                
                if counts < edges_to_add_for_diversity:
                    additional_edges += edges_to_add_for_diversity - counts
                    e_feat_counts[feature][str(key)] = edges_to_add_for_diversity
                    
            # Normalize counts to probabilites
            for key in e_feat_counts[feature].keys():
                e_feat_counts[feature][str(key)] = e_feat_counts[feature][str(key)] / (edges + additional_edges)
                    
        for d in range(max_node_deg):
            
            # If probabilities are empty for this degree then take probabilities from another close-by degree
            if vertices_per_degree[d] == 0:
                alternative_degrees = [idx for idx, d2 in enumerate(vertices_per_degree) if d2 != 0]
                alternative_degrees.sort(key=lambda x: abs(x - d))
                selection_deg = alternative_degrees[0]
            else:
                selection_deg = d
            
            for feature in range(nr_v_feat):
                for value in possible_node_features[feature]:
                    self.v_feat_prob[d][feature].append(v_feat_counts[selection_deg][feature][str(value)])
                                    
        for feature in range(nr_e_feat):
            for value in possible_edge_features[feature]:
                self.e_feat_prob[feature].append(e_feat_counts[feature][str(value)])

    def __call__(self):
        graph = self.generate_graph()
        if self.og_graph_has_len1_node_features and len(graph.x.shape) > 1:
            graph.x = torch.squeeze(graph.x, 1)
                
        if self.og_graph_has_len1_edge_features and len(graph.edge_attr.shape) > 1: 
            graph.edge_attr = torch.squeeze(graph.edge_attr, 1)
            
        return self.time_stamp(graph)

    def generate_graph(self):
        pass