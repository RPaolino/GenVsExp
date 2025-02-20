import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import MDS
import sklearn.cluster
import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform, Compose
import tqdm
from typing import Union, Iterable

from .transforms import AddLabel, AddSubgraph
from .tmd import pairwise_TMD

def create_synthetic_data(
    name: str,
    num_nodes_lower: int = 35,
    num_nodes_upper: int = 55,
    p: float = 0.1,
    m: int = 2,
    num_blocks_lower: int = 3,
    num_blocks_upper: int = 6,
    min_num_nodes_per_block: int = 3,
    p_same_block_lower: float = 1e-1,
    p_same_block_upper: float = 3e-1,
    p_different_block_lower: float = 1e-3,
    p_different_block_upper: float = 2e-2,
    seed: int = 42
):
    r""""
    Creates a torch_geometric.data.Data representing a random graph
    with a number of nodes drawn randomly between num_nodes_lower and 
    num_nodes_upper.

    Input:
        name: str
            -"er": Erdos-Renyi graph;
            -"ba": Barabasi-Albert graph;
            -"sbm": Stochastick Block Model graph.
        num_nodes_lower: int
            Lower bound for the number of nodes.
        num_nodes_upper: int
            Upper bound for the number of nodes.
        p: float
            Probability for edge creation. Useful for Eros-Renyi graphs.
        m: int
            Number of edges to attach from a new node to existing nodes.
            Useful for Barabasi-Albert graphs.
        num_blocks_lower: int
            Lower bound on the number of blocks in a SBM graph.
        num_blocks_upper: int
            Upper bound on the number of blocks in a SBM graph.
        min_num_nodes_per_block: int
            Minimal number of nodes per block in a SBM graph.
        p_same_block_lower: float
            Lower bound on the probability for edge creation between nodes in
            the same block of a SBM graph.
        p_same_block_upper: float
            Upper bound on the probability for edge creation between nodes in
            the same block of a SBM graph.
        p_different_block_lower: float
            Lower bound on the probability for edge creation between nodes in
            different blocks of a SBM graph.
        p_different_block_upper: float
            Upper bound on the probability for edge creation between nodes in
            different blocks of a SBM graph.
        seed: int
            Indicator of random number generation state. 
    """
    # Randomly draw of the number of nodes
    rng = np.random.default_rng(
        seed = seed
    )
    num_nodes = rng.integers(
        num_nodes_lower, 
        num_nodes_upper+1
    )
    if name=="er":
        G = nx.erdos_renyi_graph(
            n=num_nodes, 
            p=p, 
            seed=seed, 
            directed=False
        )
    elif name=="ba":
        G = nx.barabasi_albert_graph(
            n=num_nodes, 
            m=m, 
            seed=seed
        )
    elif name=="sbm":
        num_blocks = rng.integers(
            num_blocks_lower, 
            num_blocks_upper+1
        )
        sizes = [
            min_num_nodes_per_block for _ in range(num_blocks)
        ]
        # Each remaining node is assigned randomly to one of the blocks
        blocks_remaining_nodes = rng.choice(
            num_blocks, 
            size=num_nodes - sum(sizes)
        )
        for block in blocks_remaining_nodes:
            sizes[block] += 1     
        assert sum(sizes) == num_nodes
        probs = [
            [0 for _ in range(num_blocks)] for _ in range(num_blocks)
        ]
        for i in range(num_blocks):
            for j in range(i, num_blocks):
                if i == j:
                    probs[i][j] = rng.uniform(
                        p_same_block_lower, 
                        p_same_block_upper
                    )
                else:
                    tmp = rng.uniform(
                        p_different_block_lower, 
                        p_different_block_upper
                    )
                    probs[i][j] = tmp
                    probs[j][i] = tmp 
        G = nx.stochastic_block_model(
            sizes=sizes, 
            p=probs, 
            seed = seed
        )
    edge_index = torch.tensor(
        list(G.edges)
    ).t().contiguous()
    edge_index = torch.cat(
        [
            edge_index,
            edge_index.flip(0)
        ], dim=1
    )
    return Data(num_nodes=num_nodes, edge_index=edge_index)

def median_labels(
    dataset: list[Data]
) -> list[int]:
    counts = [d.task_counts for d in dataset]
    labels = [int(c>=np.median(counts)) for c in counts]
    return labels


def TMD_labels(
    TMD_pairwise
) -> list[int]:
    """
    Generates labels for the dataset based on Tree Mover’s Distance (TMD) and
    clusters them into two classes.

    Args:
        TMD_pairwise:
            pairwise TMD distances
    """
   
    # MDS Embedding
    print("Performing MDS embedding.")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embeddings = mds.fit_transform(TMD_pairwise)
    # Clustering
    print("Clustering")
    clustering = sklearn.cluster.KMeans(n_clusters=2, random_state=42)
    labels = clustering.fit_predict(embeddings)
    return embeddings, labels

class SyntheticDataset(InMemoryDataset):
    r"""
    Creates a torch_geometric.data.InMemoryDataset object of random graphs.

    Input:
        num_graphs: int,
            Number of graphs in the dataset
        transform: torch_geometric.transforms.Compose
            Transform to apply to each element in the dataset
    For the meaning of the other input parameters, please check the documentation
    of create_synthetic_data.
    """
    
    def __init__(
        self,
        num_graphs: int,
        transform: Union[BaseTransform, Compose],
        name: str,
        task: str,
        pe: str,
        num_nodes_lower: int = 35,
        num_nodes_upper: int = 55,
        p: float = 0.1,
        m: int = 2,
        num_blocks_lower: int = 3,
        num_blocks_upper: int = 6,
        min_num_nodes_per_block: int = 3,
        p_same_block_lower: float = 1e-1,
        p_same_block_upper: float = 3e-1,
        p_different_block_lower: float = 1e-3,
        p_different_block_upper: float = 2e-2,
        depth: int = 3
    ):
        
        super().__init__()
        self._data_list = []
        data_list = []
        for seed in tqdm.trange(num_graphs, desc="Synthetic Dataset Creation"):
            data = create_synthetic_data(
                name=name,
                num_nodes_lower=num_nodes_lower,
                num_nodes_upper=num_nodes_upper,
                p=p,
                m=m,
                num_blocks_lower=num_blocks_lower,
                num_blocks_upper=num_blocks_upper,
                min_num_nodes_per_block=min_num_nodes_per_block,
                p_same_block_lower=p_same_block_lower,
                p_same_block_upper=p_same_block_upper,
                p_different_block_lower=p_different_block_lower,
                p_different_block_upper=p_different_block_upper,
                seed=seed
            )
            data_list.append(transform(data))
        
        if "tmd" not in task:
            filename = f"data/TMD_{100}_{pe}_{task}.pt"
            labels = median_labels(data_list)
            if os.path.exists(filename):
                print("Loading TMD")
                pairwise_distances = torch.load(filename)
            else:
                pairwise_distances = pairwise_TMD(
                    data_list[:100],
                    depth=depth
                )
                torch.save(
                    pairwise_distances,
                    filename
                )
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            embeddings = mds.fit_transform(pairwise_distances)
            
            np.savetxt(
                filename.replace("TMD_", "embeddings_").replace(".pt", ".txt"),
                np.concatenate(
                    [embeddings, np.array(labels[:100]).reshape(-1, 1)],
                    axis=1
                )
            )
            cluster_0 = [idx for idx, l in enumerate(labels[:100]) if l==0]
            cluster_1 = [idx for idx, l in enumerate(labels[:100]) if l==1]
            fig, ax = plt.subplots()
            ax.scatter(
                embeddings[cluster_0, 0], 
                embeddings[cluster_0, 1], 
                c=["red" for _ in range(len(cluster_0))], 
                s=10, 
                alpha=.5,
                label=0
            )
            ax.scatter(
                embeddings[cluster_1, 0], 
                embeddings[cluster_1, 1], 
                c=["blue" for _ in range(len(cluster_1))], 
                s=10, 
                alpha=.5,
                marker="D",
                label=1
            )
            ax.legend(title="Label")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.savefig(
                filename.replace(".pt", ".svg").replace("data/", "imgs/"), 
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            filename = f"data/TMD_{num_graphs}_{pe}_{task}.pt"
            if os.path.exists(filename):
                print("Loading TMD")
                pairwise_distances = torch.load(filename)
            else:
                pairwise_distances = pairwise_TMD(
                    data_list,
                    depth=depth
                )
                torch.save(
                    pairwise_distances,
                    filename
                )
            embeddings, labels = TMD_labels(
                pairwise_distances
            )
            cluster_0 = [idx for idx, l in enumerate(labels) if l==0]
            cluster_1 = [idx for idx, l in enumerate(labels) if l==1]
            fig, ax = plt.subplots()
            ax.scatter(
                embeddings[cluster_0, 0], 
                embeddings[cluster_0, 1], 
                c=["red" for _ in range(len(cluster_0))], 
                s=10, 
                alpha=.5
            )
            ax.scatter(
                embeddings[cluster_1, 0], 
                embeddings[cluster_1, 1], 
                c=["blue" for _ in range(len(cluster_1))], 
                s=10, 
                alpha=.5,
                marker="D"
            )
            fig.savefig(filename, bbox_inches="tight")
            plt.close(fig)
                
        new_transforms = Compose([
            AddLabel(labels=labels),
            AddSubgraph()
        ])
        for data in tqdm.tqdm(data_list, desc="Adding Synthetic Labels"):
            self._data_list.append(
                new_transforms(data)
            )
        # Collating all elements of the dataset
        self._data, _ = self.collate(
            self._data_list
        )
            
    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: Union[int, slice, Iterable]) -> list:
        if isinstance(idx, int) or isinstance(idx, np.int64):
            out = self._data_list[idx]
        elif isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = len(self._data_list) if idx.stop is None else idx.stop
            step = 1 if idx.step is None else  idx.step
            out = [
                self._data_list[current_idx] for current_idx in np.arange(start, stop, step)
            ]
        else:
            out = [
                self._data_list[current_idx] for current_idx in idx
            ]
        return out

