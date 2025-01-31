import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import networkx as nx
from torch_geometric.utils import to_networkx, to_dense_adj

def compute_homomorphism_cycle_counts(data: Data, length_bound: int):
    r"""
    Computes the number of homomorphisms from cycles.

    Input:
        data (torch_geometric.data.Data): the initial graph
        length_bound (int): if length_bound is an int, generate all simple 
            cycles with length at most length_bound.
    """
    num_nodes = data.x.shape[0]
    # initializing counts
    positional_encodings = torch.zeros((num_nodes, length_bound-2))
    A = torch.zeros((num_nodes, num_nodes))
    A[data.edge_index[0], data.edge_index[1]] = 1
    A[data.edge_index[1], data.edge_index[0]] = 1
    powA = A @ A
    for cycle_length in range(3, length_bound+1):
        powA = powA @ A
        positional_encodings[:, cycle_length-3] = powA.diag()
    return positional_encodings
                    

class AddHomomorphismCycleCounts(BaseTransform):
    r"""
    Concatenate the features data.x with the number of cycle counts.
    """
    def __init__(self, length_bound: int):
        self.length_bound = length_bound
        
        
    def __call__(self, data: Data):        
        homomorphism_cycle_counts = compute_homomorphism_cycle_counts(
            data=data,
            length_bound=self.length_bound
        )
        data.x = torch.cat(
            [
                data.x, 
                homomorphism_cycle_counts.to(dtype=torch.float)
            ], 
            dim=1
        )
        return data




        