import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import networkx as nx
from torch_geometric.utils import to_networkx, to_dense_adj

def compute_subgraph_cycle_counts(data: Data, length_bound: int):
    r"""
    Computes the number of cycles each node is part of.

    Input:
        data (torch_geometric.data.Data): the initial graph
        length_bound (int): if length_bound is an int, generate all simple 
            cycles with length at most length_bound.
    """
    num_nodes = data.x.shape[0]
    G = to_networkx(data, to_undirected=True)
    # initializing counts
    positional_encodings = torch.zeros((num_nodes, length_bound-2))
    cycles = nx.simple_cycles(G, length_bound)
    for cycle in cycles:
        cycle_length = len(cycle)
        if (cycle_length > 2) and (cycle_length <= length_bound):
            positional_encodings[cycle, cycle_length - 3] += 1
    return positional_encodings
                    

class AddSubgraphCycleCounts(BaseTransform):
    r"""
    Concatenate the features data.x with the number of cycle counts.
    """
    def __init__(self, length_bound: int):
        self.length_bound = length_bound
        
        
    def __call__(self, data: Data):        
        subgraph_cycle_counts = compute_subgraph_cycle_counts(
            data=data,
            length_bound=self.length_bound
        )
        data.x = torch.cat(
            [
                data.x, 
                subgraph_cycle_counts.to(dtype=torch.float)
            ], 
            dim=1
        )
        return data




        