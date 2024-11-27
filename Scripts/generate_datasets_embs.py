"""
Generates both a mutated dataset and embeddings which allow us to filter graphs which are "easy".
"""

import os
import argparse
import json 
import glob

import tqdm
import torch
from torch_geometric.loader import DataLoader

from Exp.emb_extraction_utils import EmbStorageObject
from Exp.training_loop_functions import eval
from Misc.config import config
from GraphGen.add_time_stamp import AddTimeStamp
from Exp.parser import parse_args
from Exp.utils import get_classes_tasks_feats
from Misc.utils import transform_dict_to_args_list, find_most_recent_model
from Misc.utils import classification_task
from Misc.dataset_wrapper import DatasetWrapper
from Exp.preparation import load_dataset, get_model, get_transform, get_loss
from Exp.run_model import set_seed

from GraphGen.cycle_tree_generator import Cycle_Tree_Generator
from GraphGen.mutate_graph_generator import Mutate_Graph_Generator
from GraphGen.ER_generator import ER_Graph_Generator

def get_args(passed_args):
    parser = argparse.ArgumentParser(description='Generate embeddings for a dataset and a teacher.')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-teacher', type=str)
    parser.add_argument('--device', type=int, default=0,
                    help='Which gpu to use if any (default: 0)')
    parser.add_argument('-required_num', type=int, default = -1)

    if passed_args is None:
        return parser.parse_args(passed_args)
    else:
        return parser.parse_args(transform_dict_to_args_list(passed_args))

def prepare_model_and_transforms(str_model, args):
    print(f"Preparing: {str_model}")
    model_name = args.__dict__[str_model]

    path_model, path_config = find_most_recent_model(model_name, args.dataset)

    # Load args from the stored model's config
    with open(path_config, "r") as file:
        args_for_model = json.load(file)
        # Add -- to every key
        for key in list(args_for_model.keys()):
            value = args_for_model.pop(key)
            
            # Ignore irrelevant keys
            if (value is None) or (key == "delta") or (key[:3] == "do_") or (key[:4] == "use_"):
                continue
            
            # Transform boolean values to integers
            if value == True:
                value = 1
            elif value == False:
                value = 0

            args_for_model[f"--{key}"] = value
    args_for_model = parse_args(args_for_model)
    
    # Prepare models
    device = args.device
    
    set_seed(42)
    train_loader, _, test_loader = load_dataset(parse_args({"--dataset": args.dataset}), config)
    num_classes, num_tasks, num_vertex_features = get_classes_tasks_feats(train_loader.dataset, args.dataset)
    
    model = get_model(args_for_model, num_classes, num_vertex_features, num_tasks)
    model.to(device)
    test_performance = os.path.split(path_model)[-1].split("_test_")[1].split("_")[0]

    print(f"Loading {str_model} from {path_model}")
    model.load_state_dict(torch.load(path_model))
    transform = get_transform(args_for_model)
    ds_generator = lambda ls: DatasetWrapper(ls, train_loader.dataset.num_classes, train_loader.dataset.num_node_features, num_tasks)    
    
    # Test performance of the model on the test set
    eval_model(model, ds_generator([transform(g) for g in test_loader.dataset]), test_performance, args)

    return model, transform, ds_generator

def eval_model(model, dataset, target_performance, args):
    """
    Evaluate a model on a dataset and ensure that the performance is what was measured when this model was trained
    """
    print("Checking whether the loaded model has the correct performance")
    loss_dict = get_loss(args)
    loss_fn = loss_dict["loss"]
    eval_name = loss_dict["metric"]
    metric_method = loss_dict["metric_method"]

    if args.teacher in ["DS", "DSS"]:
        loader = DataLoader(dataset, batch_size=128, shuffle=False, follow_batch=['subgraph_idx'])
    else:
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

    results_dict = eval(model, args.device, loader, loss_fn, eval_name, metric_method=metric_method)
    measured_performance = f"{results_dict[eval_name]:.4f}"
    print(target_performance, "vs", measured_performance)
    assert target_performance == measured_performance

def get_path_original_dataset(dataset, teacher_name):
    og_dataset_path = os.path.join(config.DATA_PATH, f"{dataset}_{teacher_name}.pt")
    og_embs_path = os.path.join(config.EMBS_PATH, f"{teacher_name}_{dataset}_{teacher_name}.pt")
    return og_dataset_path, og_embs_path

def create_embeddings_for_original_dataset(args, num_tasks, dataset, original_ds, teacher, teacher_name, transform_teacher, ds_generator_teacher):
    og_dataset_path, _ = get_path_original_dataset(dataset, teacher_name)
    time_stamp = AddTimeStamp(original_graph_stamp_value=1)
    original_ds = [time_stamp(graph) for graph in original_ds]
    
    torch.save(original_ds, og_dataset_path)
    ds_file_name = os.path.split(og_dataset_path)[-1]
    create_and_store_embeddings_for_dataset(args, num_tasks, ds_file_name, teacher, transform_teacher, ds_generator_teacher)
                                            
def create_and_store_embeddings_for_dataset(args, num_tasks, dataset_filename, teacher, transform_teacher, ds_generator_teacher):
    dataset = torch.load(os.path.join(config.DATA_PATH, dataset_filename))
    emb_storage = EmbStorageObject(classification_task(args.dataset), num_tasks)
    
    # Annotate original dataset with embeddings
    print("Preparing dataset")
    ls = []
    for graph in tqdm.tqdm(dataset):
                ls.append(transform_teacher(graph))
    teacher_loader = DataLoader(ds_generator_teacher(ls), batch_size=1, shuffle=False, follow_batch=['subgraph_idx'])
    
    print("Computing embeddings")
    for _, teacher_graph in enumerate(tqdm.tqdm(teacher_loader)):
        if teacher_graph.edge_attr is not None and len(teacher_graph.edge_attr.shape) == 1:
            teacher_graph.edge_attr = teacher_graph.edge_attr.view(-1, 1)
        teacher_graph.to(args.device)
        
        if teacher_graph.edge_index is not None:
            teacher_graph.edge_index = teacher_graph.edge_index.long()
        t_predictions, t_pred_graph_emb, t_pred_node_emb = teacher(teacher_graph)
        emb_storage.update(teacher_graph, t_predictions, t_pred_graph_emb, t_pred_node_emb)
    
    # Store embeddings and predictions
    print(f"Storing embeddings of {dataset_filename}")
    embs = emb_storage.finalize()
    
    if not os.path.isdir(config.EMBS_PATH):
        os.mkdir(config.EMBS_PATH)
    torch.save(embs, os.path.join(config.EMBS_PATH, args.teacher + "_" + dataset_filename))
    
    del embs, teacher_loader, dataset
    
def main(passed_args = None):
    print("Generating mutated datasets.")
    
    # Import down here to avoid circular imports
    from Exp.preparation import load_dataset 

    args = get_args(passed_args)
    device = args.device

    with torch.no_grad():
        teacher, transform_teacher, ds_generator_teacher = prepare_model_and_transforms("teacher", args)
        teacher.eval()
                
        # Check if we have embeddings for the original dataset
        original_ds = load_dataset(parse_args({
                "--dataset": args.dataset,
                "--batch_size": 1
                }), 
            config)[0].dataset
        _, num_tasks, _ = get_classes_tasks_feats(original_ds, args.dataset)
        
        if args.dataset.lower() == "qm9":
            num_tasks = 19
            
        og_dataset_path, og_embs_path = get_path_original_dataset(args.dataset, args.teacher)
        if not (os.path.isfile(og_dataset_path) and os.path.isfile(og_embs_path)):
            print("No embeddings for original dataset exist -> generating")
            create_embeddings_for_original_dataset(args, num_tasks, args.dataset, original_ds, teacher, args.teacher, transform_teacher, ds_generator_teacher)
        
        relevant_dataset_files = list(glob.glob(os.path.join(config.DATA_PATH, f"{args.dataset}*.pt")))
        print(f"Found {len(relevant_dataset_files)} files: {relevant_dataset_files}")

        num_created_embs = 0
        for path in relevant_dataset_files:
            if args.required_num >= 0 and num_created_embs > args.required_num:
                break

            filename = os.path.split(path)[-1]
            print(f"\n\n\nNow processing {filename}")

            # Check if we have an embedding for this:
            if os.path.isfile(os.path.join(config.EMBS_PATH, f"{args.teacher}_{filename}")):
                print("Already exists")
                continue
            
            create_and_store_embeddings_for_dataset(args, num_tasks, filename, teacher, transform_teacher, ds_generator_teacher)
            num_created_embs += 1
            
        
if __name__ == "__main__":
    main()