import numpy as np
import networkx as nx
from collections import defaultdict

import torch_geometric
import torch
from torch_geometric.utils import degree, is_undirected

from Misc.visualize import visualize
from GraphGen.graph_generator import Generator

class Cycle_Tree_Generator(Generator):    
    def __init__(self, 
            dataset_name,
            min_components = 6,
            max_components = 17,
            prob_cycle = 0.2,
            possible_cycle_lens = [4, 5, 6, 8, 9, 10],
            max_chordless_cycle_len = 6):
        
        super(Cycle_Tree_Generator,self).__init__(dataset_name)
        self.min_components = min_components
        self.max_components = max_components
        self.prob_cycle = prob_cycle
        self.possible_cycle_lens = possible_cycle_lens
        self.max_chordless_cycle_len = max_chordless_cycle_len
        
    def create_graph_structure(self):
        # np.random.seed(seed=seed)
        nr_vertices = np.random.randint(self.min_components, self.max_components+1)
        G = nx.generators.trees.random_tree(nr_vertices)#, seed = seed)
        for vertex in range(nr_vertices):
            make_cycle = np.random.random() <= self.prob_cycle
            
            if not make_cycle:
                continue
            
            cycle_len = np.random.choice(self.possible_cycle_lens)
            neighbors = list(G.neighbors(vertex))
            nr_vertices = G.number_of_nodes()
            
            for neighbor in neighbors:
                G.remove_edge(vertex, neighbor)
                
            cycle_edges = [[i, (i+1)%cycle_len] for i in range(cycle_len)]
            
            # Replace edges: 0 becomes the starting vertex, all others get shifted to not conflict with existing edges 
            cycle_edges = list(map(lambda ls: list(map(lambda x: vertex if x == 0 else x + nr_vertices, ls)), cycle_edges))
        
            vertices_in_cycle = list(set([x for xs in cycle_edges for x in xs]))
            
            if cycle_len > self.max_chordless_cycle_len:
                max_distance = cycle_len - 4
                distance = np.random.randint(4, max_distance+1)
                starting_vertex_idx = np.random.randint(0, cycle_len)
                ending_vertex_idx = (starting_vertex_idx + distance) % cycle_len
                G.add_edge(vertices_in_cycle[starting_vertex_idx], vertices_in_cycle[ending_vertex_idx])
                    
            for edge in cycle_edges:
                G.add_edge(edge[0], edge[1])
                
            # Re-attach neighbors
            for neighbor in neighbors:
                G.add_edge(neighbor, int(np.random.choice(vertices_in_cycle)))
        
        data = torch_geometric.utils.from_networkx(G)
        return data

    def generate_graph(self):
        while True:
            graph = self.create_graph_structure()#seed = self.seed)
            # self.seed += 1
            if graph.num_nodes < 50 and graph.edge_index.shape[1] < 200:
                deg = degree(graph.edge_index[0], graph.num_nodes)
                x = torch.zeros((graph.num_nodes, len(self.possible_node_features)), dtype=torch.int)
                for node in range(graph.num_nodes):
                    for dim in range(len(self.possible_node_features)):
                        x[node, dim] = np.random.choice(self.possible_node_features[dim], p=self.v_feat_prob[int(deg[node])][dim])

                # Find both variants of each edge:
                edge_pointers_dict = defaultdict(lambda: [])
                for i in range(graph.edge_index.shape[1]):
                    x1, x2 = int(graph.edge_index[0, i]), int(graph.edge_index[1, i])
                    
                    if x1 > x2:
                        x1, x2 = x2, x1
                       
                    edge_pointers_dict[(x1, x2)].append(i)
                    
                edge_attr = torch.zeros((graph.edge_index.shape[1], len(self.possible_edge_features)), dtype=torch.int)
                for edge, edge_pointers in edge_pointers_dict.items():
                    for dim in range(len(self.possible_edge_features)):
                        feat = np.random.choice(self.possible_edge_features[dim], p=self.e_feat_prob[dim])
                        for pointer in edge_pointers:
                            edge_attr[pointer, dim] = feat     
        
                graph.x = x
                graph.edge_attr = edge_attr
                graph.num_nodes = None
                assert is_undirected(graph.edge_index, graph.edge_attr)
                return graph
                
if __name__ == "__main__":
    generator = Cycle_Tree_Generator("ZINC")
    for i in range(20):
        graph = generator.generate_graph()
        visualize(graph, f"cycle_tree_{i}", v_feat_dim = 0, e_feat_dim = 0)

    