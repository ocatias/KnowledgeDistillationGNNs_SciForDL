"""
Helper functions that do argument parsing for experiments.
"""
import os
import argparse
import yaml
from copy import deepcopy
import ast
import glob
import json

from Misc.config import config
from Misc.utils import transform_dict_to_args_list, find_most_recent_model
from GraphGen.parser import add_generator_args

def parse_args(passed_args=None):
    """
    Parse command line arguments. Allows either a config file (via "--config path/to/config.yaml")
    or for all parameters to be set directly.
    A combination of these is NOT allowed.
    Partially from: https://github.com/twitter-research/cwn/blob/main/exp/parser.py
    """

    parser = argparse.ArgumentParser(description='An experiment.')

    # Config file to load
    parser.add_argument('--config', dest='config_file', type=argparse.FileType(mode='r'),
                        help='Path to a config file that should be used for this experiment. '
                        + 'CANNOT be combined with explicit arguments')

    parser.add_argument('--tracking', type=int, default=config.use_wandb_tracking,
                        help=f'If 0 runs without tracking (Default: {str(config.use_wandb_tracking)})')


    # Parameters to be set directly
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--split', type=int, default=0,
                        help='Split for cross validation (default: 0)')
    parser.add_argument('--dataset', type=str, default="ZINC",
                            help='Dataset name (default: ZINC; other options: CSL and most datasets from ogb, see ogb documentation)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    
    parser.add_argument('--device', type=int, default=0,
                    help='Which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='GIN',
                    help='Model to use (default: GIN; options: GIN, GCN, MLP, DSS, DS, CWN, GSNx, L2GNN)')
                    
    # LR SCHEDULER
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau',
                    help='Learning rate decay scheduler (default: ReduceLROnPlateau; other options: StepLR, None; For details see PyTorch documentation)')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5,
                        help='Strength of lr decay (default: 0.5)')

    # For StepLR
    parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50,
                        help='(For StepLR scheduler) number of epochs between lr decay (default: 50)')

    # For ReduceLROnPlateau
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='(For ReduceLROnPlateau scheduler) mininum learnin rate (default: 1e-5)')
    parser.add_argument('--lr_schedule_patience', type=int, default=10,
                        help='(For ReduceLROnPlateau scheduler) number of epochs without improvement until the LR will be reduced')
    parser.add_argument('--threshold', type=float, default=1e-3,
                        help='(For ReduceLROnPlateau scheduler) threshold for measuring a new optimum (default: 1e-3)')
    parser.add_argument('--warmup_steps', type=int, default=0,
                            help='(For Cosine scheduler) Number of epochs with linear warmup (default: 0)')

    parser.add_argument('--max_time', type=float, default=1000,
                        help='Max time (in hours) for one run')

    parser.add_argument('--drop_out', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Dimensionality of hidden units in models (default: 64)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of message passing layers (default: 5) or number of layers of the MLP')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='Number of layers in the MLP that performs predictions on the embedding computed by the GNN (default: 1)')
    parser.add_argument('--virtual_node', type=int, default=0,
                        help='Set 1 to use a virtual node, that is a node that is adjacent to every node in the graph (default: 0)')
    parser.add_argument('--JK', type=str, default='last',
                    help='Type of jumping knowledge to use (default: last; options: last, sum, concat)') 
    
    parser.add_argument('--cat', type=int, default=0,
                    help="Set to 1 to use cycle adjacency transform (default: 0)")


    parser.add_argument('--pooling', type=str, default="mean",
                        help='Graph pooling operation to use (default: mean; other options: sum, set2set)')
    parser.add_argument('--node_encoder', type=int, default=1,
                        help="Set to 0 to disable to node encoder (default: 1)")

    parser.add_argument('--drop_feat', type=int, default=0,
                        help="Set to 1 to drop all edge and vertex features from the graph (default: 0)")

    parser.add_argument('--selection', type=str, default="best_val",
                        help=' (default: best_val; other options: last)')
    
    # Model specifing things
    parser.add_argument('--policy', type=str, default='',
                    help='(For DS / DSS) Policy to use for extracting subgraphs (default: ; options: edge_deleted, node_deleted, ego_nets)') 
    parser.add_argument('--num_hops', type=int, default=3,
                        help='(For DS / DSS with ego_nets policy) Number of hops to extract (default: 3)') 
    parser.add_argument('--esan_conv', type=str, default='',
                    help='(For DS / DSS) Convolution to use (default: ; options: GIN, originalgin, graphconv, GCN, ZINCGIN)')    

    parser.add_argument('--store_model', type=int, default=0,
                        help="Set to 1 to store trained model (default: 0)")      
    parser.add_argument('--emb', type=str, default="[]",
                        help="List of embs to use (default: [])")   
    parser.add_argument('--graph_emb', type=int, default=0,
                        help="Set to 1 to use graph level embs (default: 0)")         
    parser.add_argument('--node_emb', type=int, default=0,
                        help="Set to 1 to use node level embs (default: 0)")   
    parser.add_argument('--graph_emb_fac', type=float, default=100,
                        help="Multiplicative factor for graph_emb loss (default: 100)")     
    parser.add_argument('--node_emb_fac', type=float, default=10,
                        help="Multiplicative factor for node_emb loss (default: 10)") 
    parser.add_argument('--x', type=int, default=-1,
                        help="Factor that the mutated dataset should be larger than the training set(default: -1 meaning that no mutated dataset is loaded)")       
    parser.add_argument('--pred_loss', type=int, default=1,
                        help="Set to 0 to disable training with the prediction loss (default: 1)")  
    parser.add_argument('--import_mlp_layer', type=int, default=0,
                        help="Set to 1 to import the MLP layer from the model defined in --model_path (default: 0)")  
    parser.add_argument('--freeze_mlp_layer', type=int, default=0,
                        help="Set to 1 to freezes the mlp layer (default: 0)")  
    parser.add_argument('--final_mp_layers', type=int, default=3,
                        help="Layers in the GNN that will not be take into account for the knowledge distillation (default: 3)")
    
    
    parser.add_argument('--lifting_cycle_len', type=int, default=8,
                        help="(For CWN) maximum length of cycle to life (Default: 8)")     
    # For embedding generation
    parser.add_argument('--model_path', type=str, default='',
                    help='Path to model to load')
    parser.add_argument('--sizes', type=str, default="",
                    help="Sizes of the different generated datasets (default: '' this creates 5 datasets each containing the original training set plus 1, 2, 5, 10 and 50 runs of the mutatated training set each) ")
    
    parser.add_argument('--teacher', type=str, default='',
                    help='(Generating embeddings / training a student) use instead of model_path to load the most recent model of the given name  (options: DSS, DS)')
    
    # Graph generation / mutation
    add_generator_args(parser)

    parser.add_argument('--label', type=str, default='',
                    help="Add an additional label for tracking (Default: '')")

    # Load partial args instead of command line args (if they are given)
    if passed_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(transform_dict_to_args_list(passed_args))        

    if args.emb == "":
        args.emb = []
    else:
        args.emb = ast.literal_eval(str(args.emb))
    
    args.__dict__["use_tracking"] = args.tracking == 1
    args.__dict__["use_virtual_node"] = args.virtual_node == 1
    args.__dict__["use_node_encoder"] = args.node_encoder == 1
    args.__dict__["do_drop_feat"] = args.drop_feat == 1
    args.__dict__["do_store_model"] = args.store_model == 1
    args.__dict__["use_emb"] = args.emb != []
    args.__dict__["use_graph_emb"] = args.graph_emb == 1
    args.__dict__["use_node_emb"] = args.node_emb == 1
    args.__dict__["use_pred_loss"] = args.pred_loss == 1
    args.__dict__["use_cat"] = args.cat == 1
    args.__dict__["do_import_mlp_layer"] = args.import_mlp_layer == 1
    args.__dict__["do_freeze_mlp_layer"] = args.freeze_mlp_layer == 1

    assert args.JK in ["last", "sum", "concat"]
    
    # https://codereview.stackexchange.com/a/79015
    # If a config file is provided, write it's values into the arguments
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
                arg_dict[key] = value
                
    # Ensure sizes is a string
    args.sizes = str(args.sizes)
                                
    if args.teacher != "":
        print("Teacher argument used")
        
        if args.model_path == "":
            pth_model_weights, pth_model_config = find_most_recent_model(args.teacher, args.dataset)
            args.model_path = pth_model_weights
            
            with open(pth_model_config, "r") as file:
                model_config = json.load(file)
        else:
            with open(args.model_path.replace(".pt", ".json"), "r") as file:
                model_config = json.load(file)
        

        #
        # GENERATING EMBEDDINGS
        # If we are generating embeddings then we need to configure our model so it can load the teacher model
        if args.sizes != "":   
            print("Overwriting commandline parameters with teacher parameters")
            
            if args.teacher in ["DS", "DSS"]:
                releveant_keys = ["pooling", "emb_dim", "model", "esan_conv", "num_hops", "num_layers", "policy"]
            elif args.teacher == "CWN":
                releveant_keys = ["pooling","emb_dim", "model",  "num_layers", "num_mlp_layers", "JK"]
            elif "GSN" in args.teacher:
                releveant_keys = ["pooling","emb_dim", "model",  "num_layers", "num_mlp_layers", "JK"]
                
            for key in releveant_keys:
                print(f"{key}:\t{args.__dict__[key]} -> {model_config[key]}")
                args.__dict__[key] = model_config[key]

        # 
        # TRAIN A MODEL
        # We are training a student model, ensure MLP dimensions work
        else:
            if args.teacher in ["DS", "DSS", "CWN", "L2GNN"] or "GSN" in args.teacher:
                if args.num_layers < model_config["num_layers"]:
                    raise ValueError("Teacher has more layers than students.")
                
                if args.emb_dim <  model_config["emb_dim"]:
                    print("Student emb_dim is smaller than teacher emb_dim: extending emb_dim to fit teacher")
                    args.emb_dim =  model_config["emb_dim"]

                if args.do_import_mlp_layer:
                    args.num_mlp_layers = 2

                # args.JK = model_config["JK"]
                args.pooling = model_config["pooling"]
            
    if args.x != -1 and args.emb == [] and args.teacher != "":
        print("No embeddings path specified via --emb. Loading the most recently created embeddings")
        possible_embs = glob.glob(os.path.join(config.EMBS_PATH, "*.pt"))
        from Scripts.generate_datasets_mutations import get_ds_name
        from Scripts.generate_datasets_mutations import main as generate_graphs
        from Scripts.generate_datasets_embs import main as generate_embs
        from GraphGen.parser import keys as graph_gen_options

        print(f"emb name: {get_ds_name(args.dataset, args)}")
        possible_embs = list(filter(lambda e: get_ds_name(args.dataset, args) in e and args.teacher in e, possible_embs))
        print(f"Found {len(possible_embs)} relevant embeddings")
        og_dataset_embs = os.path.join(config.EMBS_PATH, f"{args.teacher}_{args.dataset}_{args.teacher}.pt")
        num_missing_embeddings = args.x - len(possible_embs)

        if num_missing_embeddings > 0:
            # Check if we have enough mutated datasets
            possible_ds_files = glob.glob(os.path.join(config.DATA_PATH, f"{args.dataset}_*.pt"))
            possible_ds_files = list(filter(lambda e: len(e.split('_')) > 2, possible_ds_files))
            print(f"Found {len(possible_ds_files)} dataset files")
            num_missing_dataset_files = args.x - len(possible_ds_files)
            if num_missing_dataset_files > 0:
                print("Not enough mutated datasets found. Generating {num_missing_dataset_files} additional ones")
                args_for_creating_graphs = {
                    "-dataset": args.dataset,
                    "--x": num_missing_dataset_files
                }
                for key in graph_gen_options:
                      args_for_creating_graphs[f"--{key}"] = args.__dict__[key]

                generate_graphs(args_for_creating_graphs)

        if num_missing_embeddings > 0 or not os.path.isfile(og_dataset_embs):
            print("Not enough embeddings found (or no embedding for original dataset). Generating additional embeddings.")
            generate_embs({
                "-dataset": args.dataset,
                "-teacher": args.teacher
                })
            
        possible_embs = list(filter(lambda e: get_ds_name(args.dataset, args) in e and args.teacher in e, glob.glob(os.path.join(config.EMBS_PATH, "*.pt"))))
        # Sort by recency to take most recent one
        possible_embs.sort(key=lambda x: os.path.getmtime(x))
        
        args.emb = [og_dataset_embs] + possible_embs[:args.x]
        args.use_emb = True
    
    return args
