import argparse
from joblib import Parallel, delayed

import torch
import matplotlib.pyplot as plt

from Exp.parser import parse_args
from Misc.config import config
from Misc.utils import get_basic_model_args, string_of_args_to_dict, find_most_recent_model
from Exp.run_model import run as train_model
from Exp.preparation import get_model, load_dataset 
from Exp.utils import get_classes_tasks_feats

radius = [10]
delta = 2


def prepare():
    parser = argparse.ArgumentParser(description='Quickly train a model based on hyperparameters specified under Configs.')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-model', type=str)
    
    args = parser.parse_args()
    model_args = get_basic_model_args(args.model, args.dataset)
    
    params_dict =  string_of_args_to_dict(model_args)
    args = parse_args(params_dict)
    
    
    print("Loading dataset")
    train_loader, val_loader, test_loader = load_dataset(args, config)
    num_classes, num_tasks, num_vertex_features = get_classes_tasks_feats(train_loader.dataset, args.dataset)
    
    print("Loading model")
    model = get_model(args, num_classes, num_vertex_features, num_tasks)
    
    print("Loading stored weights into model")
    path_model = find_most_recent_model(args.model, args.dataset)[0]
    model.load_state_dict(torch.load(path_model))
    return model, test_loader, args

def get_embs_labels(args, model, test_loader):
    batch_size = args.batch_size
    batches = len(test_loader)
    print("Generating embeddings")
    embs = []
    labels = []
    for i, batch in enumerate(test_loader):
        print(f"\r{i} / {batches}", end="")
        
        graphs_in_batch = batch.y.shape[0]
        
        if batch.edge_attr is not None and len(batch.edge_attr.shape) == 1:
            batch.edge_attr = batch.edge_attr.view(-1, 1)
                        
        embs_from_model = model(batch)[2][-1]
            
        for i in range(graphs_in_batch):
            embs.append(torch.squeeze(embs_from_model[i, :], dim=0))
            labels.append(batch.y[i])
    print("\rFinished generating embeddings")
    return embs, labels

avg_ls = lambda ls: sum(ls) / len(ls)

def comput_avg_distance(i, embs):
    emb1 = embs[i]
    distances = []
    for j, emb2 in enumerate(embs):
        if i == j:
            continue
    distances.append(torch.norm(emb1 - emb2, 2))
    return avg_ls(distances)

def main():
    model, test_loader, args = prepare()
    embs, labels = get_embs_labels(args, model, test_loader)
    
    avg_distances = []
    avg_nr_neighbors_closer_than_delta = []
    distances_2d = []
    print("Computing pairwise distances")
    nr_embs = len(embs)
    
    same_label_distances = []
    diff_label_distances = []
    
    for i, emb1 in enumerate(embs):
        print(f"\r{i} / {nr_embs}", end="")
        distances = []
        
        for j, emb2 in enumerate(embs):
            if i == j:
                continue
            
            d = float(torch.norm(emb1 - emb2, 2))
            distances.append(d)
            
            if labels[i] == labels[j]:
                same_label_distances.append(d)
            else:
                diff_label_distances.append(d)
            
        distances_2d.append(distances)
        # avg_nr_neighbors_closer_than_delta.append(len(list(filter(lambda distance: distance <= delta, distances))))
        # avg_distances.append(avg_ls(distances))
        
    print(f"Computing distances_flat_list ")
    distances_flat_list = [i for ls in distances_2d for i in ls] 
    print(f"distances_flat_list: {distances_flat_list}")
    max_distance = max(distances)
    num_bins = 50

    fig, ax = plt.subplots()

    # the histogram of the data
    # n, bins, patches = ax.hist(distances_flat_list, num_bins, density=True)
    
    plt.hist(same_label_distances, num_bins, alpha=0.5, label='Same label', density=True)
    plt.hist(diff_label_distances, num_bins, alpha=0.5, label='Different label', density=True)
    name = f"hist_{args.dataset}_{args.model}"
    ax.legend()
    plt.savefig(f'Imgs/{name}.jpg')

    
    
    print(f"max_distance: {max_distance}")
        
    print(f"\rNr distances: {len(avg_distances)}")
    avg_distance = avg_ls(avg_distances)
    avg_nr_neighbors_closer_than_delta = avg_ls(avg_nr_neighbors_closer_than_delta)
    
    print(f"\r\n\n{args.model} on {args.dataset}")
    print(f"Average pariwise embedding distance: {avg_distance}")
    print(f"Average #graphs embedded closer than {delta}: {avg_nr_neighbors_closer_than_delta}")

            
    
if __name__ == "__main__":
    main()