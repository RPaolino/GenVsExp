import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import BaseTransform

class Subgraph(Data):
    def __inc__(self, key, *args, **kwargs):
        if key in ("index_u", "index_v"): 
            return self.original_num_nodes
        elif "index" in key: 
            return self.num_nodes
        else: 
            return 0

class AddSubgraph(BaseTransform):
    def __call__(self, data):
        N = data.num_nodes
        node = torch.arange(N ** 2).view(size=(N, N))
        adj = to_dense_adj(
            data.edge_index, 
            max_num_nodes=N
        ).squeeze(0)
        # Computing Shortest Path Distances
        # spd[i, j] = 1 if (i, j) connected, else \infty
        spd = torch.where(
            ~torch.eye(N, dtype=bool) & (adj == 0), # non-connected pairs excluding self-loops
            torch.full_like(adj, float("inf")), 
            adj
        )
        # Update SPD using Floyd-Warshall, chech whether going through k results in
        # a shorter path between i and j
        for k in range(N): 
            spd = torch.minimum(
                spd, 
                spd[:, [k]] + spd[[k], :]
            )
        # Edge attributes
        attr, (dst, src) = data.edge_attr, data.edge_index
        if attr is not None and attr.ndim == 1: 
            attr = attr[:, None]
        assert data.x.ndim == 2
        
        stack = lambda *x: torch.stack(
            torch.broadcast_tensors(*x)
        ).flatten(start_dim=1)

        S = Subgraph(
            # original graph
            original_x = data.x,
            original_edge_index = data.edge_index,
            original_num_nodes=N,
            # number of nodes in the  subgraph space
            num_nodes=N**2,
            # duplicate node features
            x=data.x[None].repeat_interleave(N, dim=0).flatten(end_dim=1),
            y=data.y,
            edge_attr=attr[:, None].repeat_interleave(
                N, 
                dim=1
            ).flatten(end_dim=1) if attr is not None else None,
            shortest_path_distance=spd.to(int).flatten(end_dim=1),
            index_shortest_path_distance=node[:, 0] + node[0, :],
            # flattened representation of source for all subgraphs
            index_u=torch.broadcast_to(
                node[0, :, None], 
                (N, N)
            ).flatten(),
            # flattened representation of destination for all subgraphs
            index_v=torch.broadcast_to(
                node[0, None, :],
                (N, N)
            ).flatten(),
            edge_indicator=adj.to(int).flatten(end_dim=1),
            # index for local aggregation
            index_uL=stack(
                node[:, 0] + dst[:, None], 
                node[:, 0] + src[:, None]
            ),
            index_vL=stack(
                node[0] + N * dst[:, None], 
                node[0] + N * src[:, None]
            ),
            index_uLF=stack(
                node[:, 0] + dst[:, None], 
                node[:, 0] + src[:, None], 
                (N * src + dst)[:, None]
            ),
            index_vLF=stack(
                node[0] + N * dst[:, None], 
                node[0] + N * src[:, None], 
                (N * dst + src)[:, None]),
        ) 
        return S

if __name__ == "__main__":
    data = Data(
        edge_index=torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3],
            [1, 3, 0, 2, 1, 3, 0, 2]]
        ),
        edge_attr=torch.tensor(
            [5, 6, 7, 8]
        ),
        x=torch.tensor(
            [[1], [2], [3], [4]]
        )
    )
    S = AddSubgraph()(data)
    print(S.index_u)
    print(S.index_v)
    print(S.edge_indicator)
    print(S.edge_attr.flatten())
    print(S.shortest_path_distance)
    print(S.index_shortest_path_distance)

    print(S.index_uL)
    print(S.index_vL)
    print(S.index_uLF)
    print(S.index_vLF)