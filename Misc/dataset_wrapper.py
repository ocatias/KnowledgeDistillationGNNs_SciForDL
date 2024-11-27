from tqdm import tqdm
import torch

class DatasetWrapper:
    def __init__(self, ls, num_classes, num_node_features, num_tasks):
        self.ls = ls
        self.num_classes = num_classes
        self.num_node_features = num_node_features
        self.num_tasks = num_tasks
        self.has_mean = False
        
    def __getitem__(self, idx):
        return self.ls[idx]

    def __len__(self):
         return len(self.ls)
     
def load_from_path(path, pre_transform, num_classes, num_node_features, num_tasks):
    ls = torch.load(path)
    return DatasetWrapperFromDS(ls, pre_transform, num_classes, num_node_features, num_tasks)

class DatasetWrapperFromDS:
    def __init__(self, ds, pre_transform):
        self.ls = []
         
        for g in tqdm(ds):
            self.ls.append(pre_transform(g))
        
        try:
            self.num_classes = ds.num_classes
        except:
            pass
        try:
            self.num_tasks = ds.num_tasks
        except:
            pass
        try:
            self.num_node_features = ds.num_node_features
        except:
            pass
        
    def __getitem__(self, idx):
        return self.ls[idx]

    def __len__(self):
        return len(self.ls)
     
    def save(self, path):
        torch.save(self.ls, path)