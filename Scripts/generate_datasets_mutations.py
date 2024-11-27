"""
Generates both a mutated dataset and embeddings which allow us to filter graphs which are "easy".
"""

import os
import argparse
import time

import tqdm
import torch
import numpy as np

from Misc.config import config
from Exp.parser import parse_args
from Misc.utils import transform_dict_to_args_list, string_to_ls_of_ints

from GraphGen.cycle_tree_generator import Cycle_Tree_Generator
from GraphGen.mutate_graph_generator import Mutate_Graph_Generator
from GraphGen.ER_generator import ER_Graph_Generator
from GraphGen.parser import add_generator_args
    
def get_args(passed_args):
    parser = argparse.ArgumentParser(description='Generator of dataset variants.')
    add_generator_args(parser)
    parser.add_argument('-dataset', type=str)
    parser.add_argument('--x', type=int, default=5,
                    help="Number  of mutations of the dataset to generate (default: 5) ")
   
    if passed_args is None:
        return parser.parse_args(passed_args)
    else:
        return parser.parse_args(transform_dict_to_args_list(passed_args))

def get_ds_name(dataset, args):
    if args.generator == "m":
        return f"{dataset}_{args.generator}_{args.m_p_feat}_{args.m_p_edge_drop}_{args.m_p_edge_add}"
    elif args.generator == "er":
        return f"{dataset}_{args.generator}_{args.er_p_edge}_{args.er_min_num_vertices}_{args.er_max_num_vertices}"
    elif args.generator == "ct":
        return f"{dataset}_{args.generator}_{args.ct_min_num_components}_{args.ct_max_num_components}_{args.ct_p_cycle}_{args.ct_cycle_lens}_{args.ct_max_chordless_cycle_len}"
    else:
        raise Exception("Unknown generator")

def get_ds_partial_path(dataset,  args):
    return os.path.join(config.DATA_PATH, get_ds_name(dataset, args))

def main(passed_args = None):
    np.random.seed()
    print("Generating mutated datasets.")
    
    # Import down here to avoid circular imports
    from Exp.preparation import load_dataset 

    args = get_args(passed_args)

    parameters = parse_args({
        "--dataset": args.dataset,
        "--batch_size": 1
        })

    vanilla_ds = load_dataset(parameters, config)[0].dataset
    nr_graphs_og_dataset = len(vanilla_ds)
            
    if args.generator == "m":
        generator = Mutate_Graph_Generator(args.dataset, prob_node_feat_mut = args.m_p_feat, prob_edge_drop = args.m_p_edge_drop, prob_edge_add = args.m_p_edge_add)
    elif args.generator == "ct":
        generator = Cycle_Tree_Generator(
            args.dataset, 
            min_components = args.ct_min_num_components,
            max_components = args.ct_max_num_components,
            prob_cycle = args.ct_p_cycle,
            possible_cycle_lens = string_to_ls_of_ints(args.ct_cycle_lens),
            max_chordless_cycle_len = args.ct_max_chordless_cycle_len)
    elif args.generator == "er":
        generator = ER_Graph_Generator(args.dataset, edge_prob = args.er_p_edge, nodes_range = [args.er_min_num_vertices, args.er_max_num_vertices])
    else:
        raise Exception("Unknown Generator")
    
    print("Generating new graphs and embeddings")
    for _ in range(args.x):
        nr_graphs_required = nr_graphs_og_dataset
        curr_dataset = []
        
        for i in tqdm.tqdm(range(nr_graphs_required)):
            graph = generator()
            curr_dataset.append(graph)
                
        # Store graph DS
        partial_path = get_ds_partial_path(args.dataset, args)
        path_ending = f"{time.time()}_{np.random.randint(0, 1000)}.pt"
        full_path = partial_path + path_ending
        print(full_path)
        print("Example graph has this shape: ", curr_dataset[-1], "\n")
        torch.save(curr_dataset, full_path)
        
        
if __name__ == "__main__":
    main()