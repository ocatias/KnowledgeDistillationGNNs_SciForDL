import os
import csv
import itertools

from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, GNNBenchmarkDataset, TUDataset, QM9
import torch.optim as optim
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from Models.gnn import GNN
from Models.encoder import LinearNodeEncoder, NodeEncoder, EdgeEncoder, ZincAtomEncoder, EgoEncoder
from Models.mlp import MLP
from Misc.drop_features import DropFeatures
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr
from Misc.attach_embs import AttachEmbs
from Misc.dataset_wrapper import DatasetWrapper, DatasetWrapperFromDS, load_from_path
from Misc.CAT import CyclicAdjacencyTransform
from Misc.cosine_scheduler import get_cosine_schedule_with_warmup
from Misc.GSN_transform import GSN_transform
from Misc.config import config

# ESAN
from Models.ESAN.esan_preprocessing import policy2transform
from Models.ESAN.conv import GINConv, OriginalGINConv, GCNConv, ZINCGINConv 
from Models.ESAN.models import DSnetwork, DSSnetwork
from Models.ESAN.models import GNN as ESAN_GNN

# CWN
from Models.CWN.cell_encoding import CellularRingEncoding

# Local-2(folklore)-GNN
from Models.HigherOrderMPNN.utils import TwoGNNPreTransform 
from Models.HigherOrderMPNN.model import GNN as HigherOrderGNN

def get_transform(args, split = None):
    transforms = []
    if args.dataset.lower() == "csl":
        transforms.append(OneHotDegree(5))
        
    # Pad features if necessary (needs to be done after adding additional features from other transformation)
    if args.dataset.lower() == "csl":
        transforms.append(AddZeroEdgeAttr(args.emb_dim))
        transforms.append(PadNodeAttr(args.emb_dim))
      
    if args.do_drop_feat:
        transforms.append(DropFeatures(args.emb_dim))
        
    if args.use_cat:
        assert args.model not in ["DS", "DSS", "CWN"]
        transforms.append(CyclicAdjacencyTransform(spiderweb = False))

    if args.model in ["DS", "DSS"]:
        transforms.append(policy2transform(args.policy, args.num_hops))

    if args.model == "CWN":
        transforms.append(CellularRingEncoding(max_ring_size = args.lifting_cycle_len, aggr_edge_atr = True, 
        aggr_vertex_feat = True, explicit_pattern_enc = True, edge_attr_in_vertices = False))
        
    if "GSN" in args.model:
        dim = int(args.model[3:])
        transforms.append(GSN_transform(dim=dim))
    
    if args.model == "L2GNN":
        transforms.append(TwoGNNPreTransform())
        
    if args.use_emb:
        transforms.append(AttachEmbs(args.emb, attach_graph_emb=args.use_graph_emb, attach_node_emb=args.use_node_emb, overwrite_label=args.x != -1))
    return Compose(transforms)

def load_index(dataset, split):
    path = os.path.join(config.SPLITS_PATH, f"{dataset}_{split}.index")
    infile = open(path, "r")
    for line in infile:
        indices = line.split(",")
        indices = [int(i) for i in indices]
    return indices

def load_dataset(args, config):
    if args.x != -1:
        args.use_emb = False

    transform = get_transform(args)

    if transform is None or args.dataset == "alchemy":
        dir = os.path.join(config.DATA_PATH, args.dataset, "Original")
    else:
        print(repr(transform))
        trafo_str = repr(transform).replace("\n", "")
        dir = os.path.join(config.DATA_PATH, args.dataset, trafo_str)

    if args.dataset.lower() == "zinc":
        datasets = [ZINC(root=dir, subset=True, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cifar10":
        datasets = [GNNBenchmarkDataset(name ="CIFAR10", root=dir, split=split, pre_transform=Compose([ToUndirected(), transform])) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cluster":
        datasets = [GNNBenchmarkDataset(name ="CLUSTER", root=dir, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        dataset = PygGraphPropPredDataset(root=dir, name=args.dataset.lower(), pre_transform=transform)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
    elif args.dataset.lower() == "csl":
        all_idx = {}
        for section in ['train', 'val', 'test']:
            with open(os.path.join(config.SPLITS_PATH, "CSL",  f"{section}.index"), 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        dataset = GNNBenchmarkDataset(name ="CSL", root=dir, pre_transform=transform)
        datasets = [dataset[all_idx["train"][args.split]], dataset[all_idx["val"][args.split]], dataset[all_idx["test"][args.split]]]
    elif args.dataset.lower() in ["exp", "cexp"]:
        dataset = PlanarSATPairsDataset(name=args.dataset, root=dir, pre_transform=transform)
        split_dict = dataset.separate_data(args.seed, args.split)
        datasets = [split_dict["train"], split_dict["valid"], split_dict["test"]]
    elif args.dataset.lower() == "alchemy":
        # Based on: github.com/luis-mueller/towards-principled-gts/ 
        
        dataset = TUDataset(root=dir, name="alchemy_full")[load_index("alchemy", "train") + load_index("alchemy", "val") + load_index("alchemy", "test")]
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        
        datasets = [DatasetWrapperFromDS(dataset[0:10000], pre_transform=transform),
                    DatasetWrapperFromDS(dataset[10000:11000], pre_transform=transform),
                    DatasetWrapperFromDS(dataset[11000:], pre_transform=transform)]
        
        for dataset in datasets:
            dataset.mean, dataset.std = mean, std
    elif args.dataset.lower() == "qm9":
        # Based on https://github.com/chrsmrrs/SpeqNets
        
        dataset = QM9(root=dir, pre_transform=transform)
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        
        datasets = [dataset[load_index("qm9", "train")], dataset[load_index("qm9", "train")], dataset[load_index("qm9", "test")]]
        
        for dataset in datasets:
            dataset.mean, dataset.std = mean, std
            
    else:
        raise NotImplementedError("Unknown dataset")
    
    for ds in datasets:
        ds.has_mean = args.dataset.lower() == "alchemy"
        
    if args.x != -1:
        args.use_emb = True
        transform = get_transform(args) 
        print("Overwritting training set with mutated dataset")
        ls_ds_path = []
        for emb_path in args.emb:
            file_name = os.path.split(emb_path)[-1][len(args.teacher) +1:]
            ls_ds_path.append(os.path.join(config.DATA_PATH, file_name))
            
        print(f"Loading dataset from: {ls_ds_path[:3]}, ...")
        new_training_set = []
        for ds_path in tqdm(ls_ds_path):
            new_training_set.append(torch.load(ds_path))
        new_training_set = list(itertools.chain(*new_training_set))

        try:
            num_tasks = datasets[0].num_tasks
        except:
            num_tasks = 1
        
        ls = []
        for g in tqdm(new_training_set):
            ls.append(transform(g))
            
        new_training_set = DatasetWrapper(ls, datasets[0].num_classes, datasets[0].num_node_features, num_tasks)
        datasets[0] = new_training_set
        
    # We need additional information when batching for ESAN
    if args.model in ["DS", "DSS"]:
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True, follow_batch=['subgraph_idx'])
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
    else:
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_max_hamiltonian_cycle_length(dataset):
    if dataset == "peptides-func":
        return -1
    elif dataset == "ogbg-molhiv":
        return 100
    elif dataset == "ZINC":
        return 25
    return 44

def add_1_to_all_in_ls(ls):
    return list(map(lambda x: x+1, ls))

def get_model(args, num_classes, num_vertex_features, num_tasks):
    node_feature_dims, edge_feature_dims = [], []
    cat_add = 1 if args.use_cat else 0
    
    model = args.model.lower()

    if args.model == "CWN":
        node_feature_dims += [2,2,2]
        
    if "GSN" in args.model:
        dim = int(args.model[3:])
        node_feature_dims += [20]*(dim-2)
        
    if args.use_cat:
        node_feature_dims.append(7)
        edge_feature_dims += [9, get_max_hamiltonian_cycle_length(args.dataset)]

    if args.dataset.lower() == "zinc"and not args.do_drop_feat:
        node_feature_dims.append(21 + cat_add)
        edge_feature_dims.append(4 + cat_add)
        node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
        edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=edge_feature_dims)
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"] and not args.do_drop_feat:
        if args.use_cat:
            edge_feature_dims += add_1_to_all_in_ls(get_bond_feature_dims())
            node_feature_dims += add_1_to_all_in_ls(get_atom_feature_dims())
        else:        
            node_feature_dims += get_atom_feature_dims()
            edge_feature_dims += get_bond_feature_dims()
        print("node_feature_dims: ", node_feature_dims)
        node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=node_feature_dims), EdgeEncoder(args.emb_dim, feature_dims=edge_feature_dims)
    elif args.dataset.lower() == "alchemy":
        node_feature_dims += [2]*6
        edge_feature_dims += [2]*4
        node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
        edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=edge_feature_dims)
    elif args.dataset.lower() == "qm9":
        node_feature_dims += [2, 2, 2, 2, 2, 10, 1, 1, 1, 1, 5]
        edge_feature_dims += [2, 2, 2, 1]
        node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
        edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=edge_feature_dims)
    elif args.dataset.lower() == "cifar10":
        node_encoder = LinearNodeEncoder(3, args.emb_dim)
        edge_encoder =  LinearNodeEncoder(1, args.emb_dim)
    else:
        node_encoder, edge_encoder = lambda x: x, lambda x: x
            
    if model in ["gin", "gcn", "gat", "cwn"] or "gsn" in model:  
        if args.do_import_mlp_layer:
            print("Loading other weights")
            weights = torch.load(args.model_path)
            print(f"Found {len(weights)} different weight tensors")
        else:
            weights = None

        use_dim_pooling = False
        if model == "cwn":
            model = "gin"
            use_dim_pooling = True
        elif "gsn" in model:
            model = "gin"

        return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim, 
                gnn_type = model, virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = args.JK, 
                graph_pooling = args.pooling, edge_encoder=edge_encoder, node_encoder=node_encoder, 
                use_node_encoder = args.use_node_encoder, num_mlp_layers = args.num_mlp_layers, weights=weights, 
                dim_pooling=use_dim_pooling, freeze_mlp=args.do_freeze_mlp_layer)
        
    elif args.model == "L2GNN":
        model = HigherOrderGNN(args, mp_type = "L-G", task = "g", num_tasks = num_tasks, num_classes = num_classes, enc_a = node_encoder, enc_e = edge_encoder, bn = True)
        return model
    
    elif args.model.lower() == "mlp":
            return MLP(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim, 
                    num_classes=num_classes, num_tasks=num_tasks, dropout_rate=args.drop_out, graph_pooling=args.pooling, weights=weights)
    # ESAN models
    elif args.model in ["DS", "DSS"]:
        encoder = lambda x: x
        if 'ogb' in args.dataset:
            encoder = AtomEncoder(args.emb_dim) if args.policy != "ego_nets_plus" else EgoEncoder(AtomEncoder(args.emb_dim))
        elif 'ZINC' in args.dataset:
            encoder = ZincAtomEncoder(policy=args.policy, emb_dim=args.emb_dim)
        if 'ogb' in args.dataset or 'ZINC' in args.dataset:
            in_dim = args.emb_dim if args.policy != "ego_nets_plus" else args.emb_dim + 2
        elif args.dataset in ['CSL', 'EXP', 'CEXP']:
            in_dim = 6 if args.policy != "ego_nets_plus" else 6 + 2  # used deg as node feature
        else:
            in_dim = dataset.num_features

        # DSS
        if args.model == "DSS": 
            if args.esan_conv == 'GIN':
                GNNConv = GINConv
            elif args.esan_conv == 'originalgin':
                GNNConv = OriginalGINConv
            elif args.esan_conv == 'graphconv':
                GNNConv = GraphConv
            elif args.esan_conv == 'GCN':
                GNNConv = GCNConv
            elif args.esan_conv == 'ZINCGIN':
                GNNConv = ZINCGINConv
            else:
                raise ValueError('Undefined GNN type called {}'.format(args.esan_conv))
        
            model = DSSnetwork(num_layers=args.num_layers, in_dim=in_dim, emb_dim=args.emb_dim, num_tasks=num_tasks*num_classes,
                            feature_encoder=encoder, GNNConv=GNNConv)
        # DS
        else:
            subgraph_gnn = ESAN_GNN(gnn_type=args.esan_conv, num_tasks=num_tasks*num_classes, num_layer=args.num_layers, in_dim=in_dim,
                           emb_dim=args.emb_dim, drop_ratio=args.drop_out, graph_pooling='sum' if args.model != 'gin' else 'mean', 
                           feature_encoder=encoder)
            model = DSnetwork(subgraph_gnn=subgraph_gnn, channels=[64, 64], num_tasks=num_tasks*num_classes,
                            invariant=args.dataset == 'ZINC')
        return model
            
    else: # Probably don't need other models
        raise ValueError("Unrecognized model")

def get_optimizer_scheduler(model, args, finetune = False):
    
    if finetune:
        lr = args.lr2
    else:
        lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    elif args.lr_scheduler == "ReduceLROnPlateau":
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='min',
                                                                factor=args.lr_scheduler_decay_rate,
                                                                patience=args.lr_schedule_patience,
                                                                verbose=True,
                                                                threshold=args.threshold)
    elif args.lr_scheduler ==  "Cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps = args.warmup_steps, 
                                                        num_training_steps = args.epochs)   
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    return optimizer, scheduler

def get_loss(args):
    metric_method = None
    if args.dataset.lower() in ["zinc", "alchemy", "qm9"]:
        loss = torch.nn.L1Loss()
        metric = "mae"
    elif args.dataset.lower() in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]:
        loss = torch.nn.L1Loss()
        metric = "rmse (ogb)"
        metric_method = get_evaluator(args.dataset)
    elif args.dataset.lower() in ["cifar10", "csl", "exp", "cexp"]:
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    elif args.dataset in ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "rocauc (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset == "ogbg-ppa":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset in ["ogbg-molpcba", "ogbg-molmuv"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap (ogb)" 
        metric_method = get_evaluator(args.dataset)
    else:
        raise NotImplementedError("No loss for this dataset")
    
    # When learning from a teacher we minimize the L1 distance between their predictions
    if args.teacher != "":
        loss = torch.nn.L1Loss()
    
    return {"loss": loss, "metric": metric, "metric_method": metric_method}

def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method