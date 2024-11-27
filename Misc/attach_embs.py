import json
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class AttachEmbs(BaseTransform):
    r""" 
    """
    def __init__(self, paths_to_embs: str, attach_graph_emb = True, attach_node_emb = False, overwrite_label = False):
        self.path_to_embs = paths_to_embs
        self.attach_graph_emb = attach_graph_emb
        self.attach_node_emb = attach_node_emb
        self.overwrite_label = overwrite_label
        
        if self.overwrite_label:
            print("Overwriting labels with model prediction")
        
        print("Loading embeddings")
        embeddings_ls = []
        for path_to_embs in tqdm(paths_to_embs):
            embeddings_ls.append(torch.load(path_to_embs))
            print(embeddings_ls[-1]["logits"].shape)
        self.embeddings = {}
        keys = list(embeddings_ls[-1].keys())
        for key in keys:
            self.embeddings[key] = []
            
            
        for embedding in embeddings_ls:
            for key in keys:
                self.embeddings[key] += embedding[key]
                    
        print(self.embeddings.keys())
        print(f"y: {len(self.embeddings['y'])}")
        print(f"layer 1: {len(self.embeddings['layer 1'])}")
        print(f"graph_emb: {len(self.embeddings['graph_emb'])}")
        print(f"time_stamp: {len(self.embeddings['time_stamp'])}")
        self.idx = 0 
        
        self.max_layer = max(map(lambda x: int(x.split(" ")[1]), filter(lambda x: "layer" in x, self.embeddings.keys())))

    def __call__(self, data: Data):
        
        # Apply this trafo only to the training set
        if self.idx + 1 > len((self.embeddings["layer 1"])):
            return data
        
        # Check that the number of vertices is the same    
        # assert self.embeddings["layer 1"][self.idx].shape[0] == data.x.shape[0]
        
        # Check that the embeddings correspond to the correct dataset version
        # if data.time_stamp is not None and "time_stamp" in embeddings:
        #     assert data.time_stamp == embeddings["time_stamp"]
        
        if self.attach_graph_emb:
            data.graph_emb = self.embeddings["graph_emb"][self.idx]

        if self.attach_node_emb:
            data.node_emb = [self.embeddings[f"layer {i}"][self.idx] for i in range(1, self.max_layer + 1)]
            
        if self.overwrite_label:
            data.y = self.embeddings["y"][self.idx]
            
        data.logits = self.embeddings["logits"][self.idx]
        
        if len(data.logits.shape) == 1:
            data.logits = torch.unsqueeze(data.logits, dim = 0)
            
        self.idx += 1
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.path_to_embs}, {self.attach_graph_emb}, {self.attach_node_emb}, {self.overwrite_label})'