
from Misc.config import config
from Misc.visualize import visualize
from Exp.parser import parse_args
from Exp.preparation import load_dataset

def main():
    dataset_names = ["ZINC", "ogbg-molhiv"]
    num_graphs = 20
    
    for dataset_name in dataset_names:
        train_set, _, _ = load_dataset(parse_args({"--dataset": "ZINC", "--batch_size": 1}), config)
        for i, graph in enumerate(train_set):
            visualize(graph, f"{dataset_name}_{i}", v_feat_dim = 0, e_feat_dim = 0)
            if i +1 == num_graphs:
                break
    
if __name__ == "__main__":
    main()