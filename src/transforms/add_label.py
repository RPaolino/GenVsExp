import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

class AddLabel(BaseTransform):
    r"""
    """
    def __init__(self, labels):
        self.labels = labels
        self.state = 0

    def __call__(self, data: Data):
        data.y = torch.tensor(self.labels[self.state], dtype=torch.long)
        self.state += 1
        return data
