import time

import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

class AddTimeStamp(BaseTransform):
    def __init__(self, original_graph_stamp_value = None):
        super(AddTimeStamp, self).__init__()
        self.original_graph_stamp_value = original_graph_stamp_value

    def __call__(self, data: Data) -> Data:
        data.time_stamp = torch.tensor([time.time()])
            
        if self.original_graph_stamp_value is not None:
            data.original_graph = torch.tensor([self.original_graph_stamp_value])
            
        return data