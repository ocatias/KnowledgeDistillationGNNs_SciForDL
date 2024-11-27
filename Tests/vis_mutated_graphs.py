import os

import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from Exp.parser import parse_args
from Exp.preparation import load_dataset

from Misc.config import config
from GraphGen.mutate_graph import MutateGraph
from Misc.visualize import visualize

p_feat = 0.25
p_edge_drop = 0.05
p_edge_add = 0.05

def main():
    img_dir = config.IMGS_PATH
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    
    ds_name = "ZINC"
    
    if ds_name == "ZINC":
        node_features = [list(range(21))]
        edge_features = [list(range(4))]
    elif ds_name in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        node_features = list(map(lambda x: list(range(x)), get_atom_feature_dims()))
        edge_features = list(map(lambda x: list(range(x)), get_bond_feature_dims()))
    
    train_loader, _, _ = load_dataset(parse_args({"--dataset": ds_name, "--batch_size": 1}), config)
    
    mutate = MutateGraph(node_features, train_set = train_loader, possible_edge_features = edge_features, prob_node_feat_mut = p_feat, prob_edge_drop = p_edge_drop, prob_edge_add = p_edge_add)
    
    
    
    for i, graph in enumerate(train_loader):
        if i >= 10:
            break
        
        print(graph, torch.max(graph.x))
        visualize(graph, f"graph_{i}", v_feat_dim = 0, e_feat_dim = None)
        mutated_graph = mutate(graph)
        visualize(mutated_graph, f"graph_{i}_mutated", v_feat_dim = 0, e_feat_dim = None)        
        
    
    
    
if __name__ == "__main__":
    main()