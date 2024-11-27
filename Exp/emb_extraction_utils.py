import copy

import torch
from collections import defaultdict

class EmbStorageObject:
    """
    This is a wrapper around a dictionary to store embedding information
    """
    def __init__(self, classification, num_tasks):
        self.emb_dict = defaultdict(list)
        self.classification = classification
        self.num_tasks = num_tasks
        
    def update(self, batch, prediction, graph_emb, node_embs):
        """
        Add update from one embedding generation step to the embedding dictionary
        """
        for layer, node_emb in enumerate(node_embs):
            self.emb_dict[f"layer {layer + 1}"].append(node_emb.to("cpu"))

        self.emb_dict["graph_emb"].append(graph_emb.to("cpu"))

        self.emb_dict["logits"].append(prediction.view([-1]))
        
        if batch.original_graph[0] == 0:
        
            # Classification
            if self.classification:
                new_y = torch.nn.functional.relu(torch.round(prediction).view([-1]))
                new_y[new_y > 1] = 1
            # Regression
            else:
                new_y = prediction.view([-1])

        # For original graph we use the original label
        else:
            new_y = batch.y.view([-1])
            
        if self.num_tasks > 1:
            new_y = new_y.view([1, -1])
        else:
            new_y = new_y.view([-1])

        self.emb_dict["y"].append(new_y.to("cpu"))

        # To-Do: implement time_stamp checking
        self.emb_dict["time_stamp"].append(batch.time_stamp.to("cpu"))
        
    def finalize(self):
        output_dict = {}
        
        for key in self.emb_dict.keys():
            if key not in ["graph_emb", "y", "logits"]:
                output_dict[key] = self.emb_dict[key]
            
            
        output_dict["graph_emb"] = torch.cat(self.emb_dict["graph_emb"], dim = 0)
        output_dict["logits"] = torch.stack(self.emb_dict["logits"], dim = 0)
        output_dict["y"] = torch.stack(self.emb_dict["y"], dim = 0)
        
        return output_dict