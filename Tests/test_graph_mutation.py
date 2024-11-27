import unittest

from torch_geometric.data import Data
import torch

from GraphGen.mutate_graph import MutateGraph

class GraphMutationTest(unittest.TestCase):
    def mutate_graph(self, prob_node_feat_mut, prob_edge_drop, prob_edge_add, prob_edge_feat_mut):
        edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)
        edge_attr = torch.tensor([[0], [0], [0], [0], [0], [0]], dtype=torch.long)
        x = torch.tensor([[0], [0], [0]])
        graph = Data(x = x, edge_index = edge_index, edge_attr = edge_attr)
        mutate = MutateGraph([[1,2]], 
            possible_edge_features = [[1,2]], 
            prob_node_feat_mut = prob_node_feat_mut, 
            prob_edge_drop = prob_edge_drop, 
            prob_edge_add = prob_edge_add, 
            prob_edge_feat_mut = prob_edge_feat_mut)
        mutated_graph = mutate(graph)
        return graph, mutated_graph

    def test_mutating_feat(self):
        _, mutated_graph = self.mutate_graph(prob_node_feat_mut = 1, prob_edge_drop = 0, prob_edge_add = 0, prob_edge_feat_mut = 1)
        assert not torch.equal(torch.tensor([[0], [0], [0]]), mutated_graph.x)
        assert not torch.equal(torch.tensor([[0], [0], [0], [0], [0], [0]], dtype=torch.long), mutated_graph.edge_attr)

    def test_adding_edges(self):
        for _ in range(100):
            _, mutated_graph = self.mutate_graph(prob_node_feat_mut = 0, prob_edge_drop = 0, prob_edge_add = 1, prob_edge_feat_mut = 0)
            if mutated_graph.edge_index.shape[1] > 4:
                return
        assert False

    def test_dropping_edges(self):
        for _ in range(100):
            _, mutated_graph = self.mutate_graph(prob_node_feat_mut = 0, prob_edge_drop = 1, prob_edge_add = 0, prob_edge_feat_mut = 0)
            if mutated_graph.edge_index.shape[1] < 4:
                return
        assert False



# def main():
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                            [1, 0, 2, 1, 2, 0]], dtype=torch.long)
#     edge_attr = torch.tensor([[0], [0], [0], [0], [0], [0]], dtype=torch.long)
#     x = torch.tensor([[0], [0], [0]])
#     graph = Data(x = x, edge_index = edge_index, edge_attr = edge_attr)

#     print(graph, edge_attr)
    
#     mutate = MutateGraph([[1,2]], possible_edge_features = [[1,2]])
#     mutated_graph = mutate(graph)
#     print(mutated_graph, mutated_graph.edge_attr)
    


if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 5],
                            [1, 0, 2, 1, 5, 0]], dtype=torch.long)
    edge_attr = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    x = torch.tensor([[0], [0], [0]])
    graph = Data(x = x, edge_index = edge_index, edge_attr = edge_attr)

    print(graph, edge_attr)
    
    mutate = MutateGraph([[1,2]], possible_edge_features = [[1,2]])
    mutated_graph = mutate(graph)
    print(mutated_graph, mutated_graph.edge_attr)
    
    
    # unittest.main()