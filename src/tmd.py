import copy
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import ot
import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj
import tqdm
from itertools import product
from typing import Optional

# Author: Ching-Yao Chuang <cychuang@mit.edu>
# License: MIT License
# Reference 
# Chuang, C.-Y. and Jegelka, S., “Tree Mover's Distance: Bridging Graph Metrics and Stability of Graph Neural Networks”, 
# <i>arXiv e-prints</i>, Art. no. arXiv:2210.01906, 2022. doi:10.48550/arXiv.2210.01906.

def get_neighbors(g):
    '''
    get neighbor indexes for each node

    Parameters
    ----------
    g : input torch_geometric graph


    Returns
    ----------
    adj: a dictionary that store the neighbor indexes

    '''
    adj = {}
    for i in range(len(g.edge_index[0])):
        node1 = g.edge_index[0][i].item()
        node2 = g.edge_index[1][i].item()
        if node1 in adj.keys():
            adj[node1].append(node2)
        else:
            adj[node1] = [node2]
    return adj


def TMD(g1, g2, w, L):
    '''
    return the Tree Mover’s Distance (TMD) between g1 and g2

    Parameters
    ----------
    g1, g2 : two torch_geometric graphs
    w : weighting constant for each depth
         if it is a list, then w[l] is the weight for depth-(l+1) tree
         if it is a constant, then every layer shares the same weight
    L    : Depth of computation trees for calculating TMD

    Returns
    ----------
    wass : The TMD between g1 and g2

    Reference
    ----------
    Chuang et al., Tree Mover’s Distance: Bridging Graph Metrics and
    Stability of Graph Neural Networks, NeurIPS 2022
    '''

    if isinstance(w, list):
        assert(len(w) == L-1)
    else:
        w = [w] * (L-1)

    # get attributes
    n1, n2 = len(g1.x), len(g2.x)
    feat1, feat2 = g1.x.to(torch.float), g2.x.to(torch.float)
    adj1 = get_neighbors(g1)
    adj2 = get_neighbors(g2)

    blank = np.zeros(len(feat1[0]))
    D = np.zeros((n1, n2))

    # level 1 (pair wise distance)
    M = np.zeros((n1+1, n2+1))
    for i in range(n1):
        for j in range(n2):
            D[i, j] = torch.norm(feat1[i] - feat2[j])
            M[i, j] = D[i, j]
    # distance w.r.t. blank node
    M[:n1, n2] = torch.norm(feat1, dim=1)
    M[n1, :n2] = torch.norm(feat2, dim=1)

    # level l (tree OT)
    for l in range(L-1):
        M1 = copy.deepcopy(M)
        M = np.zeros((n1+1, n2+1))

        # calculate pairwise cost between tree i and tree j
        for i in range(n1):
            for j in range(n2):
                try:
                    degree_i = len(adj1[i])
                except:
                    degree_i = 0
                try:
                    degree_j = len(adj2[j])
                except:
                    degree_j = 0

                if degree_i == 0 and degree_j == 0:
                    M[i, j] = D[i, j]
                # if degree of node is zero, calculate TD w.r.t. blank node
                elif degree_i == 0:
                    wass = 0.
                    for jj in range(degree_j):
                        wass += M1[n1, adj2[j][jj]]
                    M[i, j] = D[i, j] + w[l] * wass
                elif degree_j == 0:
                    wass = 0.
                    for ii in range(degree_i):
                        wass += M1[adj1[i][ii], n2]
                    M[i, j] = D[i, j] + w[l] * wass
                # otherwise, calculate the tree distance
                else:
                    max_degree = max(degree_i, degree_j)
                    if degree_i < max_degree:
                        cost = np.zeros((degree_i + 1, degree_j))
                        cost[degree_i] = M1[n1, adj2[j]]
                        dist_1, dist_2 = np.ones(degree_i + 1), np.ones(degree_j)
                        dist_1[degree_i] = max_degree - float(degree_i)
                    else:
                        cost = np.zeros((degree_i, degree_j + 1))
                        cost[:, degree_j] = M1[adj1[i], n2]
                        dist_1, dist_2 = np.ones(degree_i), np.ones(degree_j + 1)
                        dist_2[degree_j] = max_degree - float(degree_j)
                    for ii in range(degree_i):
                        for jj in range(degree_j):
                            cost[ii, jj] =  M1[adj1[i][ii], adj2[j][jj]]
                    wass = ot.emd2(dist_1, dist_2, cost)

                    # summarize TMD at level l
                    M[i, j] = D[i, j] + w[l] * wass

        # fill in dist w.r.t. blank node
        for i in range(n1):
            try:
                degree_i = len(adj1[i])
            except:
                degree_i = 0

            if degree_i == 0:
                M[i, n2] = torch.norm(feat1[i])
            else:
                wass = 0.
                for ii in range(degree_i):
                    wass += M1[adj1[i][ii], n2]
                M[i, n2] = torch.norm(feat1[i]) + w[l] * wass

        for j in range(n2):
            try:
                degree_j = len(adj2[j])
            except:
                degree_j = 0
            if degree_j == 0:
                M[n1, j] = torch.norm(feat2[j])
            else:
                wass = 0.
                for jj in range(degree_j):
                    wass += M1[n1, adj2[j][jj]]
                M[n1, j] = torch.norm(feat2[j]) + w[l] * wass


    # final OT cost
    max_n = max(n1, n2)
    dist_1, dist_2 = np.ones(n1+1), np.ones(n2+1)
    if n1 < max_n:
        dist_1[n1] = max_n - float(n1)
        dist_2[n2] = 0.
    else:
        dist_1[n1] = 0.
        dist_2[n2] = max_n - float(n2)

    wass = ot.emd2(dist_1, dist_2, M)
    return wass

def pairwise_TMD(
    train_dataset: torch_geometric.data.InMemoryDataset,
    test_dataset: Optional[torch_geometric.data.InMemoryDataset] = None,
    depth: int = 1,
    w = 1.
):
    r"""
    Computes the TMD distance between each train graph and test graph. 
    If the test_dataset is None, it computes the pairwise distances between graphs
    in the train_dataset.
    """
    num_train_graphs = len(train_dataset)
    parallel = Parallel(n_jobs=-1, backend="multiprocessing", verbose=1)
    dTMD = delayed(
        partial(TMD, L=depth, w=w)
    )
    if test_dataset is None:
        row, col = np.triu_indices(num_train_graphs, 1)
        progress = tqdm.tqdm(
            zip(row, col),
            desc=f"Computing TMD with depth {depth}",
            total=len(row)
        )
        flatten_distances =  parallel(
            dTMD(
                train_dataset[i], 
                train_dataset[j]
            ) for (i, j) in progress
        )
        distances = torch.zeros((num_train_graphs, num_train_graphs))
        distances[row, col] = torch.tensor(flatten_distances)
        distances = distances + distances.T
    else:
        num_test_graphs = len(test_dataset)
        distances = torch.zeros(
            (num_train_graphs, num_test_graphs)
        )
        idx = list(
            product(
            range(num_train_graphs), 
            range(num_test_graphs)
            )
        )
        progress = tqdm.tqdm(
            idx,
            desc=f"Computing TMD with depth {depth}",
            total=len(idx)
        )
        flatten_distances = parallel(
            dTMD(
                train_dataset[i], 
                test_dataset[j]
            ) for (i, j) in progress
        )
        idx = np.transpose(idx)
        distances[idx[0], idx[1]] = torch.tensor(flatten_distances)
    return distances