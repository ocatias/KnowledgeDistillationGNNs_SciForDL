from GraphGen.graph_generator import Generator
from GraphGen.mutate_graph import MutateGraph

from Misc.visualize import visualize

class Mutate_Graph_Generator(Generator):
    def __init__(self, dataset_name, prob_node_feat_mut = 0.20, prob_edge_drop = 0.05, prob_edge_add = 0.05):
        super(Mutate_Graph_Generator, self).__init__(dataset_name)
        self.prob_node_feat_mut = prob_node_feat_mut
        self.prob_edge_drop = prob_edge_drop
        self.prob_edge_add = prob_edge_add
        self.mutate = MutateGraph(self.possible_node_features, self.v_feat_prob, self.possible_edge_features, self.e_feat_prob, prob_node_feat_mut = prob_node_feat_mut, prob_edge_drop = prob_edge_drop, prob_edge_add = prob_edge_add)
        self.seed = 0
        
    def generate_graph(self):
        graph = self.train_set.dataset[self.seed].clone()
        self.seed = (self.seed + 1) % len(self.train_set.dataset)
        return self.mutate(graph)
        
if __name__ == "__main__":
    generator = Mutate_Graph_Generator("ZINC")
    for i in range(20):
        graph = generator.generate_graph()
        visualize(graph, f"mutate_graph_{i}", v_feat_dim = 0, e_feat_dim = 0)