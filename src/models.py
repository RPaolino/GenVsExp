# Code from
# https://github.com/subgraph23/homomorphism-expressivity/blob/main/src/model.py

import torch.nn
from typing import Optional, List
import torch.nn.functional as F
from torch_scatter import scatter


class MLP(torch.nn.Sequential):

    def __init__(self, input_channels: int, output_channels: int, hidden_channels: int=None, norm: bool=True):
        super().__init__()
        hidden_channels = hidden_channels or input_channels
        self.add_module(
            "input",
            torch.nn.Linear(input_channels, hidden_channels, bias=not norm)
        )
        relu = torch.nn.Sequential()
        self.add_module(
            "relu",
            relu
        )
        if norm: 
            relu.add_module(
                "norm", 
                torch.nn.BatchNorm1d(hidden_channels)
            )
        relu.add_module(
            "activate", 
            torch.nn.ReLU()
        )
        self.add_module(
            "output", 
            torch.nn.Linear(hidden_channels, output_channels, bias=not norm)
        )

class NodeEmbedding(torch.nn.Module):
    def __init__(
        self, 
        dim: int, 
        max_distance: int, 
        x_encoder: Optional[torch.nn.Module]
    ):
        super().__init__()
        self.max_distance = max_distance
        self.x_encoder = x_encoder 
        self.distance_embedding = torch.nn.Embedding(max_distance + 1, dim)

    def forward(self, batch):
        x = self.x_encoder(batch.x) if self.x_encoder else 0
        d = self.distance_embedding(
            torch.clamp(batch.shortest_path_distance, 0, max=self.max_distance)
        )
        batch.x = x + d
        del batch.d
        return batch

class EdgeEmbedding(torch.nn.Module):
    def __init__(self, dim: int, edge_attr_encoder: Optional[torch.nn.Module]=None):
        super().__init__()
        self.edge_attr_encoder = edge_attr_encoder

    def forward(self, message, attrs=None):
        if not self.edge_attr_encoder: 
            out = F.relu(message)
        else:
            out = F.relu(message + self.edge_attr_encoder(attrs))
        return out


class L(torch.nn.Module):
    r"""
    Applies the following update rule:
        x_v = MLP(
            (1 + \varepsilon) x_v
            + \sum_{u \in N(v)} EdgeEmbedding( (W x)_u + a_{u, v}, a_{u, v})
        )
    where
        x_v: feature of node v
        W: linear transformation
        N(v): neighborhood of node v
        a_{u, v}: edge attribute
    """
    def __init__(
        self, 
        agg: str, 
        input_channels: int, 
        output_channels: int, 
        edge_attr_encoder: Optional[torch.nn.Module], 
        bn: bool,
        gin: bool=True
    ):
        super().__init__()
        self.agg = agg
        self.gin = gin
        self.edge_embedding = EdgeEmbedding(
            input_channels,
            edge_attr_encoder
        )
        self.linear = torch.nn.Linear(
            input_channels, 
            input_channels
        )
        self.mlp = MLP(
            input_channels, 
            output_channels, 
            norm=bn
        )
        self.eps = torch.nn.Parameter(
            torch.zeros(1)
        )
    def forward(self, batch):
        dst, src = batch[f"index_{self.agg}"]
        attrs = "a" in batch and batch["a"]
        message = self.edge_embedding(
            torch.index_select(
                self.linear(batch.x), 
                dim=0, 
                index=src
            ),
            attrs
        )
        aggregate = scatter(
            message, 
            dim=0, 
            index=dst, 
            dim_size=len(batch.x)
        )
        out =  self.mlp(
            (batch.x * (1. + self.eps) if self.gin else 0.) + aggregate
        )
        return out


class LF(torch.nn.Module):
    r"""
    Applies the following update rule:
        x_v = MLP(
            (1 + \varepsilon) x_v
            + \sum_{f \in aggL}\sum_{u \in N(v)} EdgeEmbedding( (W x)_u + a_{u, v}, a_{u, v})
        )
    where
        x_v: feature of node v
        W: linear transformation
        N(v): neighborhood of node v
        a_{u, v}: edge attribute
    """
    def __init__(
        self, 
        agg: str, 
        aggL: List[str], 
        input_channels: int, 
        output_channels: int, 
        edge_attr_encoder: Optional[torch.nn.Module], 
        bn: bool,
        gin: bool=True
    ):
        super().__init__()
        self.agg = agg
        self.aggL = aggL
        self.gin = gin
        self.edge_embedding = EdgeEmbedding(
            input_channels, 
            edge_attr_encoder
        )
        self.linear = torch.nn.Linear(
            input_channels, 
            input_channels
        )
        self.mlp = MLP(
            input_channels, 
            output_channels, 
            norm=bn
        )
        self.eps = torch.nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, super, batch):
        dst, *src = batch[f"index_{self.agg}"]
        attrs = "a" in batch and batch["a"]
        f = lambda agg_f, src_f: torch.index_select(
            super[agg_f].linear(batch.x), 
            dim=0, 
            index=src_f
        )
        message = self.edge_embedding(
            sum(map(f, self.aggL, src)), 
            attrs
        )
        aggregate = scatter(
            message, 
            dim=0, 
            index=dst, 
            dim_size=len(batch.x)
        )
        out = self.mlp(
            (batch.x * (1. + self.eps) if self.gin else 0.) + aggregate
        )
        return out

class G(torch.nn.Module):
    def __init__(
        self, 
        agg: str,
        input_channels: int, 
        output_channels: int, 
        bn: bool,
        gin: bool=True
    ):
        r"""
        Applies the following update rule:
        x_v = MLP(
            (1 + \varepsilon) x_v
            + \sum_{u \in N(v)} x_u
        )
        where
            x_v: feature of node v
            N(v): neighborhood of node v
        """
        super().__init__()
        self.agg = agg
        self.gin = gin
        # self.enc = Edge(input_channels, False)
        self.mlp = MLP(
            input_channels, 
            output_channels, 
            norm=bn
        )
        self.eps = torch.nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, batch):
        idx = batch[f"index_{self.agg}"]
        x = torch.index_select(
            scatter(
                batch.x, 
                dim=0, 
                index=idx
            ), 
            dim=0, 
            index=idx
        )
        out = self.mlp(
            (batch.x * (1. + self.eps) if self.gin else 0.) + x
        )
        return out

class LWL(torch.nn.ModuleDict):
    def __init__(
        self,
        input_channels: int, 
        output_channels: int, 
        edge_attr_encoder: Optional[torch.nn.Module], 
        bn: bool,
        aggL: List[str]=[], 
        aggG: List[str]=[], 
        gin: bool=True
    ):
        super().__init__()
        self.aggL = aggL
        self.aggG = aggG
        for agg in aggL: 
            self[agg] = L(
                agg=agg,
                input_channels=input_channels, 
                output_channels=output_channels, 
                edge_attr_encoder=edge_attr_encoder, 
                bn=bn,
                gin=gin
            )
        for agg in aggG: 
            self[agg] = G(
                agg=agg,
                input_channels=input_channels, 
                output_channels=output_channels, 
                bn=bn,
                gin=gin
            )
        self.bn = torch.nn.BatchNorm1d(output_channels) if bn else torch.nn.Identity()
        
    def forward(self, batch):
        xL = sum(self[agg](batch) for agg in self.aggL)
        xG = sum(self[agg](batch) for agg in self.aggG)
        batch.x = F.relu(self.bn(xL + xG))
        return batch

class FWL(torch.nn.ModuleDict):
    def __init__(
        self, 
        input_channels: int, 
        output_channels: int, 
        edge_attr_encoder: Optional[torch.nn.Module], 
        bn: bool,
        aggL: List[str]=[], 
        aggG: List[str]=[], 
        gin: bool=True
    ):
        super().__init__()
        self.aggL = aggL
        self.aggG = aggG
        for agg in aggL: 
            self[agg] = LF(
                agg=agg,
                aggL=aggL,
                input_channels=input_channels, 
                output_channels=output_channels, 
                edge_attr_encoder=edge_attr_encoder, 
                bn=bn,
                gin=gin
            )
        for agg in aggG: 
            self[agg] = G(
                agg=agg,
                input_channels=input_channels, 
                output_channels=output_channels, 
                bn=bn,
                gin=gin
            )
        self.bn = torch.nn.BatchNorm1d(output_channels) if bn else torch.nn.Identity()

    def forward(self, batch):
        xL = sum(self[agg](self, batch) for agg in self.aggL)
        xG = sum(self[agg](batch) for agg in self.aggG)
        batch.x = F.relu(self.bn(xL + xG))
        return batch

# ---------------------------------- POOLING --------------------------------- #

class Pooling(torch.nn.Module):

    def __init__(
        self, 
        input_channels: int, 
        output_channels: int, 
        task: str, 
        bn: bool, 
        pool: str,
        gin: bool=True
    ):
        super().__init__()
        self.task = task
        self.pooling = task != "e" and MLP(
            input_channels=input_channels, 
            output_channels=input_channels, 
            norm=bn
        )
        self.predict = MLP(
            input_channels=input_channels, 
            output_channels=output_channels, 
            hidden_channels=2*input_channels, 
            norm=False
        )
        self.eps = torch.nn.Parameter(
            torch.zeros(1)
        )
        self.bn = torch.nn.BatchNorm1d(input_channels) if bn else torch.nn.Identity()
        self.pool = pool
        self.gin = gin

    def forward(self, batch):
        if self.task == "e":
            x = batch.x
        else:
            x = scatter(batch.x, dim=0, index=batch.index_u)
            if self.gin:
                x = torch.index_select(
                    batch.x, 
                    dim=0, 
                    index=batch.index_shortest_path_distance
                ) * (1. + self.eps) + x
            x = F.relu(self.bn(self.pooling(x)))
            if self.task == "g":
                x = scatter(torch.index_select(x, 0, batch.index_u), batch.batch, dim=0, reduce=self.pool)
        return x
        return self.predict(x)
        
# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #

class GNN(torch.nn.Sequential):

    def __init__(
        self, 
        model: str, 
        hidden_channels: int, 
        num_layers: int, 
        task: str,
        max_distance,
        x_encoder: Optional[torch.nn.Module], 
        edge_attr_encoder: Optional[torch.nn.Module], 
        output_channels: int, 
        bn: bool=True, 
        pool="sum"
    ):
        super().__init__()

        self.add_module(
            "node_embedding", 
            NodeEmbedding(
                dim=hidden_channels, 
                max_distance=max_distance, 
                x_encoder=x_encoder
            )
        )

        for i in range(num_layers):
            if model == "MP": 
                layer = LWL(
                    aggL=["vL"],
                    input_channels=hidden_channels, 
                    output_channels=hidden_channels, 
                    edge_attr_encoder=edge_attr_encoder, 
                    bn=bn
                )
            elif model == "Sub": 
                layer = LWL(
                    aggL=["uL"],
                    input_channels=hidden_channels, 
                    output_channels=hidden_channels, 
                    edge_attr_encoder=edge_attr_encoder, 
                    bn=bn
                )
            elif model == "L": 
                layer = LWL(
                    aggL=["uL", "vL"],
                    input_channels=hidden_channels, 
                    output_channels=hidden_channels, 
                    edge_attr_encoder=edge_attr_encoder, 
                    bn=bn
                )
            elif model == "LF": 
                layer = FWL(
                    aggL=["uLF", "vLF"],
                    input_channels=hidden_channels, 
                    output_channels=hidden_channels, 
                    edge_attr_encoder=edge_attr_encoder, 
                    bn=bn
                )
            # elif model == "L-G": 
            #   layer = LWL(
            #       aggL=["uL", "vL"], 
            #       aggG=["u", "v"],
            #        input_channels=hidden_channels, 
            #        output_channels=hidden_channels, 
            #        edge_attr_encoder=edge_attr_encoder, 
            #        bn=bn
            #   )
            # elif model == "LF-G": 
            #   layer = FWL(
            #       aggL=["uLF", "vLF"], 
            #       aggG=["u", "v"],
            #        input_channels=hidden_channels, 
            #        output_channels=hidden_channels, 
            #        edge_attr_encoder=edge_attr_encoder, 
            #        bn=bn
            #    )
            elif model == "Sub-G": 
                layer = LWL(
                    aggL=["uL"], 
                    aggG=["u"],
                    input_channels=hidden_channels, 
                    output_channels=hidden_channels, 
                    edge_attr_encoder=edge_attr_encoder, 
                    bn=bn
                )
            elif model == "L-G": 
                layer = LWL(
                    aggL=["uL", "vL"], 
                    aggG=["v"],
                    input_channels=hidden_channels, 
                    output_channels=hidden_channels, 
                    edge_attr_encoder=edge_attr_encoder, 
                    bn=bn
                )
            elif model == "LF-G": 
                layer = FWL(
                    aggL=["uLF", "vLF"], 
                    aggG=["v"],
                    input_channels=hidden_channels, 
                    output_channels=hidden_channels, 
                    edge_attr_encoder=edge_attr_encoder, 
                    bn=bn
                )
            else: 
                raise NotImplementedError
            self.add_module(
                f"A{i}", 
                layer
            )

        self.add_module(
            "out", 
            Pooling(
                input_channels=hidden_channels, 
                output_channels=output_channels, 
                task=task, 
                bn=bn, 
                pool=pool,
                gin=False
            )
        )
        
        
class GNNnClassifier(torch.nn.Sequential):
        def __init__(self, GNN, classifier):
            super().__init__()
            self.add_module("GNN", GNN)
            self.add_module("Classifier", classifier)