# Code from
# https://github.com/subgraph23/homomorphism-expressivity/blob/main/src/model.py

import torch.nn
from typing import Optional, List
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GINConv, global_add_pool


import torch
import torch.nn as nn

class RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    """
    def __init__(self, depth_of_mlp, in_features, out_features, residual=False):
        super().__init__()

        self.residual = residual
        
        self.mlp1 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp2 = MlpBlock(in_features, out_features, depth_of_mlp)

        self.skip = SkipConnection(in_features+out_features, out_features)
        
        if self.residual:
            self.res_x = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)

        mult = torch.matmul(mlp1, mlp2)

        out = self.skip(in1=inputs, in2=mult)
        
        # if self.residual:            
        #     # Now, changing shapes from [1xdxnxn] to [nxnxd] for Linear() layer
        #     inputs, out = inputs.permute(3,2,1,0).squeeze(), out.permute(3,2,1,0).squeeze()
            
        #     residual_ = self.res_x(inputs)
        #     out = residual_ + out # residual connection
            
        #     # Returning output back to original shape
        #     out = out.permute(2,1,0).unsqueeze(0)

        if self.residual:
            # We have 'inputs' in shape [B, in_features, m, m].
            B, c, m, m2 = inputs.shape
            assert m == m2, "Input is not square along last two dims!"

            # 1) Flatten so each (batch, row, col) is a separate row:
            #    [B, c, m, m] -> [B*m*m, c]
            flat_in = inputs.permute(0,2,3,1).reshape(B*m*m, c)

            # 2) Apply linear: [B*m*m, out_features]
            residual_ = self.res_x(flat_in)

            # 3) Reshape back to [B, m, m, out_features]
            residual_ = residual_.reshape(B, m, m, self.res_x.out_features)

            # 4) Permute to [B, out_features, m, m]
            residual_ = residual_.permute(0,3,1,2)

            # 5) Add to 'out'
            out = out + residual_
        
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for i in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))

        return out


class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)
        _init_weights(self.conv)

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out


class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=nn.functional.relu):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)

        self.activation = activation_fn

    def forward(self, input):
        out = self.fc(input)
        if self.activation is not None:
            out = self.activation(out)

        return out
    
    
def diag_offdiag_maxpool(input):
    N = input.shape[-1]

    max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # BxS

    # with torch.no_grad():
    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * input)
    val = torch.abs(torch.add(max_val, min_val))

    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)

    max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # BxS

    return torch.cat((max_diag, max_offdiag), dim=1)  # output Bx2S

def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    # nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
        

class LayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d).unsqueeze(0).unsqueeze(0)) # shape is 1 x 1 x d
        self.b = nn.Parameter(torch.zeros(d).unsqueeze(0).unsqueeze(0)) # shape is 1 x 1 x d
        
    def forward(self, x):
        # x tensor of the shape n x n x d
        mean = x.mean(dim=(0,1), keepdim=True)
        var = x.var(dim=(0,1), keepdim=True, unbiased=False)
        x = self.a * (x - mean) / torch.sqrt(var + 1e-6) + self.b # shape is n x n x d
        return x

class ThreeWLGNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.in_dim_node = net_params['in_dim']
        self.depth_of_mlp = net_params['depth_of_mlp']
        self.hidden_dim = net_params['hidden_dim']
        self.n_layers = net_params['L']
        self.residual = net_params['residual']

        # Each block:
        self.reg_blocks = nn.ModuleList()
        in_features = self.in_dim_node + 1  # adjacency + node feats
        for _ in range(self.n_layers):
            block = RegularBlock(
                depth_of_mlp=self.depth_of_mlp,
                in_features=in_features,
                out_features=self.hidden_dim,
                residual=self.residual
            )
            self.reg_blocks.append(block)
            in_features = self.hidden_dim

    def forward(self, x_3wlg):
        """
        x_3wlg: [B, (1+in_dim), N, N]
        Return node embeddings of shape [B, N, hidden_dim].
        We'll do an extremely simple approach: after the final block,
        we interpret row i of the [N,N] matrix as the i-th node's features
        and just do a sum across columns for each row.

        Many other 3-WL designs are possible, but this is a minimal illustration.
        """
        out = x_3wlg  # shape [B, d, N, N]
        for block in self.reg_blocks:
            out = block(out)  # still [B, d, N, N]
            # print("Out: ", out.shape)


        # Suppose we sum across columns to get node-level embeddings:
        # out: [B, hidden_dim, N, N]
        # sum along the last dimension => [B, hidden_dim, N]
        out = out.sum(dim=-1)
        # Then transpose to [B, N, hidden_dim]
        out = out.transpose(1, 2)
        # print("Out after transpose: ", out.shape)
        return out  # node features for each node i in [B, N, hidden_dim]


class GINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(current_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(
                GINConv(mlp)
            )
            current_channels = hidden_channels

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_add_pool(x, batch)
        return x


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):
        return self.linear(x)

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
        # d = self.distance_embedding(
        #     torch.clamp(batch.shortest_path_distance, 0, max=self.max_distance)
        # )
        batch.x = x #+ d
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

# class Pooling(torch.nn.Module):

#     def __init__(
#         self, 
#         input_channels: int, 
#         output_channels: int, 
#         task: str, 
#         bn: bool, 
#         pool: str,
#         gin: bool=True
#     ):
#         super().__init__()
#         self.task = task
#         self.pooling = task != "e" and MLP(
#             input_channels=input_channels, 
#             output_channels=input_channels, 
#             norm=bn
#         )
#         self.predict = MLP(
#             input_channels=input_channels, 
#             output_channels=output_channels, 
#             hidden_channels=2*input_channels, 
#             norm=False
#         )
#         self.eps = torch.nn.Parameter(
#             torch.zeros(1)
#         )
#         self.bn = torch.nn.BatchNorm1d(input_channels) if bn else torch.nn.Identity()
#         self.pool = pool
#         self.gin = gin

#     def forward(self, batch):
#         if self.task == "e":
#             x = batch.x
#         else:
#             x = scatter(batch.x, dim=0, index=batch.index_u)
#             if self.gin:
#                 x = torch.index_select(
#                     batch.x, 
#                     dim=0, 
#                     index=batch.index_shortest_path_distance
#                 ) * (1. + self.eps) + x
#             x = F.relu(self.bn(self.pooling(x)))
#             if self.task == "g":
#                 x = scatter(torch.index_select(x, 0, batch.index_u), batch.batch, dim=0, reduce=self.pool)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class Pooling(nn.Module):

    def __init__(
        self, 
        input_channels: int, 
        output_channels: int, 
        task: str, 
        bn: bool, 
        pool: str,
        gin: bool=True
    ):
        """
        If task != "e", we apply an MLP ("pooling") that transforms the node embeddings,
        then do a final MLP ("predict").
        If pool == "sum", we sum across nodes for a graph-level output (for task == "g").
        If pool == "mean", we do an average, etc.
        For 3-GNN inputs, we expect shape [B, N, input_channels].
        """
        super().__init__()
        self.task = task
        
        # MLP that processes the features before final classification
        self.pooling = nn.Sequential()  
        if task != "e":  
            # If you want an MLP:
            # from your code, MLP(...) is a custom class, so adapt the usage as needed
            self.pooling = MLP(
                input_channels=input_channels, 
                output_channels=input_channels, 
                norm=bn
            )

        # Another MLP to produce final predictions
        # If you only want to return the embeddings, remove/modify this
        self.predict = MLP(
            input_channels=input_channels, 
            output_channels=output_channels, 
            hidden_channels=2*input_channels, 
            norm=False
        )

        self.eps = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm1d(input_channels) if bn else nn.Identity()
        self.pool = pool
        self.gin = gin

    def forward(self, data_or_tensor):
        """
        data_or_tensor: either
         - A dictionary-like PyG batch (with fields x, index_u, etc.) for WL-based models;
         - Or a 3D tensor [B, N, D] for 3-GNN models.

        Returns a final pooled feature or prediction, shape [B, output_channels].
        """

        # ------------------------------------------
        # 3-GNN scenario: we have [B, N, D] or [B, D, N, N] etc.
        # In your extended code, you decided that your 3-GNN returns node features
        # of shape [B, N, D]. So let's assume that *here* we do a final pool or MLP.
        # ------------------------------------------
        x_3gnn = data_or_tensor  # [B, N, D]
        B, N, D = x_3gnn.shape

    
        # Graph-level pool across nodes
        if self.pool == "sum":
            x_pool = x_3gnn.sum(dim=1)  # shape [B, D]
        elif self.pool == "mean":
            x_pool = x_3gnn.mean(dim=1) # shape [B, D]
        else:
            raise ValueError(f"Unknown pool={self.pool}")
        # print("Shape after pooling: ", x_pool.shape)
        return x_pool
    
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
        pool="sum", 
        device="cpu"
    ):
        super().__init__()

        # self.add_module(
        #     "node_embedding", 
        #     NodeEmbedding(
        #         dim=hidden_channels, 
        #         max_distance=max_distance, 
        #         x_encoder=x_encoder
        #     )
        # )
        if model in ["MP", "Sub", "L", "LF", "Sub-G", "L-G", "LF-G"]:
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
                # else: 
                #     raise NotImplementedError
                self.add_module(
                    f"A{i}", 
                    layer
                )
                    # 3) Otherwise, if we choose the new "3-GNN" option, we skip the loop
        #    and directly add the ThreeWLGNNNet as a single “big” module.
        elif model == "3-GNN":
            # Example net_params.  Adjust to your needs:
            net_params = {
                'in_dim'     : x_encoder,
                'depth_of_mlp': 2,      # or however many 1x1-conv layers you want per block
                'hidden_dim' : hidden_channels,
                'n_classes'  : hidden_channels,
                'dropout'    : 0.0,
                'L'          : num_layers,    # number of 3WLGNN blocks
                'layer_norm' : bn,           # or use separate param
                'residual'   : True,
                'device'     : device,        # or "cuda" as needed
            }
            # Add your 3WLGNN model here (with final classifier removed if you like)
            layer = ThreeWLGNNNet(net_params)
            self.add_module("3WLGNN", layer)

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
