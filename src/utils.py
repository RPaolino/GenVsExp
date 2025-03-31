#from homlib import Graph, hom
from  torch_geometric.utils import is_undirected, to_networkx
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

def seed_all(seed=1000):
    r'''
    Manually set the seeds for torch, numpy
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_num_threads(1)

def to_homlib(data):
    """
    Transforms pytorch geometric data object to the graph object used by homlib
    """
    # Assumes graph is undirected (PyG method for this seems to be broken)
    T = Graph(data.x.shape[0])
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        if i > j:
            T.addEdge(i.item(), j.item())
    return T

def homlib_dragon_graph():
    """
       0
      / \
     3   1
      \ /
       2
       |
       4
       |
       5 - 7
       |
       6
    """
    D = Graph(8)
    D.addEdge(0,1)  
    D.addEdge(1,2)  
    D.addEdge(2,3)  
    D.addEdge(3,0)  
    D.addEdge(2,4)  
    D.addEdge(4,5)  
    D.addEdge(5,6) 
    D.addEdge(5,7)  
    return D

def homlib_cycle_graph(num_nodes: int):
    C = Graph(num_nodes)
    for n in range(num_nodes):
        C.addEdge(n, np.mod(n+1, num_nodes))  
    return C