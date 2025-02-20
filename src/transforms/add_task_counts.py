import networkx as nx
from homlib import hom
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import numpy as np
import torch

from ..utils import homlib_dragon_graph, homlib_cycle_graph, to_homlib

class AddTaskCounts(BaseTransform):
    r"""
    Create the attribute data.count with the number of counts specified by the task.
    """
    def __init__(self, task: str):
        self.task = task
          
    def __call__(self, data: Data):
        if ("sum_sub_C" in self.task
            or "sum_basis_C" in self.task):
            G = to_networkx(data, to_undirected=True)
            if "_sub_" in self.task:
                length_bound = int(self.task.replace("sum_sub_C", ""))
                cycles = nx.simple_cycles(G, length_bound)
            elif "_basis_" in self.task:
                length_bound = int(self.task.replace("sum_basis_C", ""))
                cycles = nx.cycle_basis(G)
            else:
                raise NotImplementedError
            count = 0
            for C in cycles:
                if (len(C)>2) and (len(C)<= length_bound):
                    count += 1
            count = torch.tensor(count)
        elif self.task=="node_count":
            count = torch.tensor(
                data.num_nodes,
                dtype=torch.float
            )
        elif self.task=="hom_D":
            count = hom(
                homlib_dragon_graph(), 
                to_homlib(data)
            )
        elif self.task.contains("hom_C"):
            count = hom(
                homlib_cycle_graph(int(self.task.replace("hom_C", ""))), 
                to_homlib(data)
            )
        else:
            raise NotImplementedError
        data.task_counts = count
        return data
