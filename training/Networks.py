import gym
import torch
from torch import Tensor
import torch.nn as nn
import torch

import torch_geometric
import math
import numpy as np
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import add_self_loops, to_dense_batch

import argparse
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
import copy
from .distributions import Categorical, DiagGaussian, Bernoulli
from .util import init, get_clones 



def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output



def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape 


class EmbedConv(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        layer_N: int,
        use_orthogonal: bool,
        use_ReLU: bool,
        use_layerNorm: bool,
        add_self_loop: bool,
        edge_dim: int = 0,
    ):
        """
            EmbedConv Layer which takes in node features, node_type (entity type)
            and the  edge features (if they exist)
            `entity_embedding` is concatenated with `node_features` and
            `edge_features` and are passed through linear layers.
            The `message_passing` is similar to GCN layer

        Args:
            input_dim (int):
                The node feature dimension
            num_embeddings (int):
                The number of embedding classes aka the number of entity types
            embedding_size (int):
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the linear layers
            layer_N (int):
                Number of linear layers for aggregation
            use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer
            use_ReLU (bool):
                Whether to use reLU for each layer
            use_layerNorm (bool):
                Whether to use layerNorm for each layer
            add_self_loop (bool):
                Whether to add self loops in the graph
            edge_dim (int, optional):
                Edge feature dimension, If zero then edge features are not
                considered. Defaults to 0.
        """
        super(EmbedConv, self).__init__(aggr="add")
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        layer_norm = [nn.Identity(), nn.LayerNorm(hidden_size)][use_layerNorm]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)
        self.lin1 = nn.Sequential(
            init_(nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)),
            active_func,
            layer_norm,
        )
        self.lin_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), active_func, layer_norm
        )

        self.lin2 = get_clones(self.lin_h, self._layer_N)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ):
        if self._add_self_loops and edge_attr is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: OptTensor):
        """
        The node_obs obtained from the environment
        is actually [node_features, node_num, entity_type]
        x_i' = AGG([x_j, EMB(ent_j), e_ij] : j \in \mathcal{N}(i))
        """
        node_feat_j = x_j[:, :-1]
        # dont forget to convert to torch.LongTensor
        entity_type_j = x_j[:, -1].long()
        entity_embed_j = self.entity_embed(entity_type_j)
        if edge_attr is not None:
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        x = self.lin1(node_feat)
        for i in range(self._layer_N):
            x = self.lin2[i](x)
        return x


class TransformerConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        num_heads: int,
        concat_heads: bool,
        layer_N: int,
        use_ReLU: bool,
        graph_aggr: str,
        global_aggr_type: str,
        embed_hidden_size: int,
        embed_layer_N: int,
        embed_use_orthogonal: bool,
        embed_use_ReLU: bool,
        embed_use_layerNorm: bool,
        embed_add_self_loop: bool,
        max_edge_dist: float,
        edge_dim: int = 1,
    ):
        """
            Module for Transformer Graph Conv Net:
            • This will process the adjacency weight matrix, construct the binary
                adjacency matrix according to `max_edge_dist` parameter, assign
                edge weights as the weights in the adjacency weight matrix.
            • After this, the batch data is converted to a PyTorch Geometric
                compatible dataloader.
            • Then the batch is passed through the graph neural network.
            • The node feature output is then either:
                • Aggregated across the graph to get graph encoded data.
                • Pull node specific `message_passed` hidden feature as output.

        Args:
            input_dim (int):
                Node feature dimension
                NOTE: a reduction of `input_dim` by 1 will be carried out
                internally because `node_obs` = [node_feat, entity_type]
            num_embeddings (int):
                The number of embedding classes aka the number of entity types
            embedding_size (int):
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the attention layers
            num_heads (int):
                Number of heads in the attention layer
            concat_heads (bool):
                Whether to concatenate the heads in the attention layer or
                average them
            layer_N (int):
                Number of attention layers for aggregation
            use_ReLU (bool):
                Whether to use reLU for each layer
            graph_aggr (str):
                Whether we want to pull node specific features from the output or
                perform global_pool on all nodes.
                Choices: ['global', 'node']
            global_aggr_type (str):
                The type of aggregation to perform if `graph_aggr` is `global`
                Choices: ['mean', 'max', 'add']
            embed_hidden_size (int):
                Hidden layer size of the linear layers in `EmbedConv`
            embed_layer_N (int):
                Number of linear layers for aggregation in `EmbedConv`
            embed_use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer in `EmbedConv`
            embed_use_ReLU (bool):
                Whether to use reLU for each layer in `EmbedConv`
            embed_use_layerNorm (bool):
                Whether to use layerNorm for each layer in `EmbedConv`
            embed_add_self_loop (bool):
                Whether to add self loops in the graph in `EmbedConv`
            max_edge_dist (float):
                The maximum edge distance to consider while constructing the graph
            edge_dim (int, optional):
                Edge feature dimension, If zero then edge features are not
                considered in `EmbedConv`. Defaults to 1.
        """
        super(TransformerConvNet, self).__init__()
        self.active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.edge_dim = edge_dim
        self.max_edge_dist = max_edge_dist
        self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type
        # NOTE: reducing dimension of input by 1 because
        # node_obs = [node_feat, entity_type]
        self.embed_layer = EmbedConv(
            input_dim=input_dim - 1,
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            hidden_size=embed_hidden_size,
            layer_N=embed_layer_N,
            use_orthogonal=embed_use_orthogonal,
            use_ReLU=embed_use_ReLU,
            use_layerNorm=embed_use_layerNorm,
            add_self_loop=embed_add_self_loop,
            edge_dim=edge_dim,
        )
        self.gnn1 = TransformerConv(
            in_channels=embed_hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
            concat=concat_heads,    
            beta=False,
            dropout=0.0,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True,
        )
        self.gnn2 = nn.ModuleList()
        for i in range(layer_N):
            self.gnn2.append(
                self.addTCLayer(self.getInChannels(hidden_size), hidden_size)
            )

    def forward(self, node_obs: Tensor, adj: Tensor, agent_id: Tensor):
        """
        node_obs: Tensor shape:(batch_size, num_nodes, node_obs_dim)
            Node features in the graph formed wrt agent_i
        adj: Tensor shape:(batch_size, num_nodes, num_nodes)
            Adjacency Matrix for the graph formed wrt agent_i
            NOTE: Right now the adjacency matrix is the distance
            magnitude between all entities so will have to post-process
            this to obtain the edge_index and edge_attr
        agent_id: Tensor shape:(batch_size) or (batch_size, k)
            Node number for agent_i in the graph. This will be used to
            pull out the aggregated features from that node
        """
        # convert adj to edge_index, edge_attr and then collate them into a batch
        batch_size = node_obs.shape[0]
        datalist = []
        for i in range(batch_size):
            edge_index, edge_attr = self.processAdj(adj[i])
            # if edge_attr is only one dimensional
            if len(edge_attr.shape) == 1:
                edge_attr = edge_attr.unsqueeze(1)
            datalist.append(
                Data(x=node_obs[i], edge_index=edge_index, edge_attr=edge_attr)
            )
        loader = DataLoader(datalist, shuffle=False, batch_size=batch_size)
        data = next(iter(loader))
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        if self.edge_dim is None:
            edge_attr = None

        # forward pass through embedConv
        x = self.embed_layer(x, edge_index, edge_attr)

        # forward pass through first transfomerConv
        x = self.active_func(self.gnn1(x, edge_index, edge_attr))

        # forward pass conv layers
        for i in range(len(self.gnn2)):
            x = self.active_func(self.gnn2[i](x, edge_index, edge_attr))
        # x is of shape [batch_size*num_nodes, out_channels]
        # convert to [batch_size, num_nodes, out_channels]
        x, mask = to_dense_batch(x, batch)
        # only pull the node-specific features from output
        if self.graph_aggr == "node":
            x = self.gatherNodeFeats(x, agent_id)  # shape [batch_size, out_channels]
        # perform global pool operation on the node features of the graph
        elif self.graph_aggr == "global":
            x = self.graphAggr(x)
        return x

    def addTCLayer(self, in_channels: int, out_channels: int):
        """
        Add TransformerConv Layer

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        Returns:
            TransformerConv: returns a TransformerConv Layer
        """
        return TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=self.num_heads,
            concat=self.concat_heads,
            beta=False,
            dropout=0.0,
            edge_dim=self.edge_dim,
            root_weight=True,
        )

    def getInChannels(self, out_channels: int):
        """
        Given the out_channels of the previous layer return in_channels
        for the next layer. This depends on the number of heads and whether
        we are concatenating the head outputs
        """
        return out_channels + (self.num_heads - 1) * self.concat_heads * (out_channels)

    def processAdj(self, adj: Tensor):
        """
        Process adjacency matrix to filter far away nodes
        and then obtain the edge_index and edge_weight
        `adj` is of shape (batch_size, num_nodes, num_nodes)
            OR (num_nodes, num_nodes)
        """
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)
        # filter far away nodes and connection to itself
        connect_mask = ((adj < self.max_edge_dist) * (adj > 0)).float()
        adj = adj * connect_mask
        index = adj.nonzero(as_tuple=True)
        edge_attr = adj[index]

        if len(index) == 3:
            batch = index[0] * adj.size(-1)
            index = (batch + index[1], batch + index[2])
        return torch.stack(index, dim=0), edge_attr

    def gatherNodeFeats(self, x: Tensor, idx: Tensor):
        """
        The output obtained from the network is of shape
        [batch_size, num_nodes, out_channels]. If we want to
        pull the features according to particular nodes in the
        graph as determined by the `idx`, use this
        Refer below link for more info on `gather()` method for 3D tensors
        https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

        Args:
            x (Tensor): Tensor of shape (batch_size, num_nodes, out_channels)
            idx (Tensor): Tensor of shape (batch_size) or (batch_size, k)
                indicating the indices of nodes to pull from the graph

        Returns:
            Tensor: Tensor of shape (batch_size, out_channels) which just
                contains the features from the node of interest
        """
        out = []
        batch_size, num_nodes, num_feats = x.shape
        idx = idx.long()


        for i in range(idx.shape[1]):
            idx_tmp = idx[:, i].unsqueeze(-1)  # (batch_size, 1)
            assert idx_tmp.shape == (batch_size, 1)
            idx_tmp = idx_tmp.repeat(1, num_feats)  # (batch_size, out_channels)
            idx_tmp = idx_tmp.unsqueeze(1)  # (batch_size, 1, out_channels)
            gathered_node = x.gather(1, idx_tmp).squeeze(
                1
            )  # (batch_size, out_channels)
            out.append(gathered_node)
        out = torch.cat(out, dim=1)  # (batch_size, out_channels*k)
        # out = out.squeeze(1)    # (batch_size, out_channels*k)
        return out

    def graphAggr(self, x: Tensor):
        """
        Aggregate the graph node features by performing global pool


        Args:
            x (Tensor): Tensor of shape [batch_size, num_nodes, num_feats]
            aggr (str): Aggregation method for performing the global pool

        Raises:
            ValueError: If `aggr` is not in ['mean', 'max']

        Returns:
            Tensor: The global aggregated tensor of shape [batch_size, num_feats]
        """
        if self.global_aggr_type == "mean":
            return x.mean(dim=1)
        elif self.global_aggr_type == "max":
            max_feats, idx = x.max(dim=1)
            return max_feats
        elif self.global_aggr_type == "add":
            return x.sum(dim=1)
        else:
            raise ValueError(f"`aggr` should be one of 'mean', 'max', 'add'")


class GNNBase(nn.Module):
    """
    A Wrapper for constructing the Base graph neural network.
    This uses TransformerConv from Pytorch Geometric
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv
    and embedding layers for entity types
    Params:
    args: (argparse.Namespace)
        Should contain the following arguments
        num_embeddings: (int)
            Number of entity types in the env to have different embeddings
            for each entity type
        embedding_size: (int)
            Embedding layer output size for each entity category
        embed_hidden_size: (int)
            Hidden layer dimension after the embedding layer
        embed_layer_N: (int)
            Number of hidden linear layers after the embedding layer")
        embed_use_ReLU: (bool)
            Whether to use ReLU in the linear layers after the embedding layer
        embed_add_self_loop: (bool)
            Whether to add self loops in adjacency matrix
        gnn_hidden_size: (int)
            Hidden layer dimension in the GNN
        gnn_num_heads: (int)
            Number of heads in the transformer conv layer (GNN)
        gnn_concat_heads: (bool)
            Whether to concatenate the head output or average
        gnn_layer_N: (int)
            Number of GNN conv layers
        gnn_use_ReLU: (bool)
            Whether to use ReLU in GNN conv layers
        max_edge_dist: (float)
            Maximum distance above which edges cannot be connected between
            the entities
        graph_feat_type: (str)
            Whether to use 'global' node/edge feats or 'relative'
            choices=['global', 'relative']
    node_obs_shape: (Union[Tuple, List])
        The node observation shape. Example: (18,)
    edge_dim: (int)
        Dimensionality of edge attributes
    """

    def __init__(
        self,
        args: argparse.Namespace,
        node_obs_shape: Union[List, Tuple],
        edge_dim: int,
        graph_aggr: str,
    ):
        super(GNNBase, self).__init__()

        self.args = args
        self.hidden_size = args.gnn_hidden_size
        self.heads = args.gnn_num_heads
        self.concat = args.gnn_concat_heads
        self.gnn = TransformerConvNet(
            input_dim=node_obs_shape,
            edge_dim=edge_dim,
            num_embeddings=args.num_embeddings,
            embedding_size=args.embedding_size,
            hidden_size=args.gnn_hidden_size,
            num_heads=args.gnn_num_heads,
            concat_heads=args.gnn_concat_heads,
            layer_N=args.gnn_layer_N,
            use_ReLU=args.gnn_use_ReLU,
            graph_aggr=graph_aggr,
            global_aggr_type=args.global_aggr_type,
            embed_hidden_size=args.embed_hidden_size,
            embed_layer_N=args.embed_layer_N,
            embed_use_orthogonal=args.use_orthogonal,
            embed_use_ReLU=args.embed_use_ReLU,
            embed_use_layerNorm=args.use_feature_normalization,
            embed_add_self_loop=args.embed_add_self_loop,
            max_edge_dist=args.max_edge_dist,
        )

    def forward(self, node_obs: Tensor, adj: Tensor, agent_id: Tensor):
        x = self.gnn(node_obs, adj, agent_id)
        return x

    @property
    def out_dim(self):
        return self.hidden_size + (self.heads - 1) * self.concat * (self.hidden_size)


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        layer_N: int,
        use_orthogonal: bool,
        use_ReLU: bool,
    ):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        obs_shape: Union[List, Tuple],
        override_obs_dim: Optional[int] = None,
    ):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        # override_obs_dim is only used for graph-based models
        if override_obs_dim is None:
            obs_dim = obs_shape[0]
        else:
            obs_dim = override_obs_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim,
            self.hidden_size,
            self._layer_N,
            self._use_orthogonal,
            self._use_ReLU,
        )

    def forward(self, x: torch.tensor):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x

class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1))
                .transpose(0, 1)
                .contiguous(),
            )
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flattened to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)  # (1, T, -1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (
                    hxs
                    * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)
                ).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    action_space: (gym.Space) action space.
    inputs_dim: int
        Dimension of network input.
    use_orthogonal: bool
        Whether to use orthogonal weight init or xavier uniform.
    gain: float
        Gain of the output layer of the network.
    """

    def __init__(
        self, action_space, inputs_dim: int, use_orthogonal: bool, gain: float
    ):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False

        action_dim = action_space.shape[0]
        self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        
    def forward(
        self,
        x: torch.tensor,
        available_actions: Optional[torch.tensor] = None,
        deterministic: bool = False,
    ):
        """
        Compute actions and action logprobs from given input.
        x: torch.Tensor
            Input to network.
        available_actions: torch.Tensor
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: bool
            Whether to sample from action distribution or return the mode.

        :return actions: torch.Tensor
            actions to take.
        :return action_log_probs: torch.Tensor
            log probabilities of taken actions.
        """
        if self.mixed_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(
                torch.cat(action_log_probs, -1), -1, keepdim=True
            )

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)

        else:
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs

    def get_probs(
        self, x: torch.Tensor, available_actions: Optional[torch.tensor] = None
    ):
        """
        Compute action probabilities from inputs.
        x: torch.Tensor
            Input to network.
        available_actions: torch.Tensor
            Denotes which actions are available to agent
            (if None, all actions available)

        :return action_probs: torch.Tensor
        """
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x)
            action_probs = action_logits.probs

        return action_probs

    def evaluate_actions(
        self,
        x: torch.tensor,
        action: torch.tensor,
        available_actions: Optional[torch.tensor] = None,
        active_masks: Optional[torch.tensor] = None,
    ):
        """
        Compute log probability and entropy of given actions.
        x: torch.Tensor
            Input to network.
        action: torch.Tensor
            Actions whose entropy and log probability to evaluate.
        available_actions: torch.Tensor
            Denotes which actions are available to agent
            (if None, all actions available)
        active_masks: torch.Tensor
            Denotes whether an agent is active or dead.

        :return action_log_probs: torch.Tensor
            log probabilities of the input actions.
        :return dist_entropy: torch.Tensor
            action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b]
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append(
                            (action_logit.entropy() * active_masks).sum()
                            / active_masks.sum()
                        )
                    else:
                        dist_entropy.append(
                            (action_logit.entropy() * active_masks.squeeze(-1)).sum()
                            / active_masks.sum()
                        )
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.sum(
                torch.cat(action_log_probs, -1), -1, keepdim=True
            )
            dist_entropy = (
                dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98
            )  #! dosen't make sense

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append(
                        (action_logit.entropy() * active_masks.squeeze(-1)).sum()
                        / active_masks.sum()
                    )
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1)  # ! could be wrong
            dist_entropy = torch.tensor(dist_entropy).mean()

        else:
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (
                    action_logits.entropy() * active_masks.squeeze(-1)
                ).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy



def minibatchGenerator(
    obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor, max_batch_size: int
):
    """
    Split a big batch into smaller batches.
    """
    num_minibatches = obs.shape[0] // max_batch_size + 1
    for i in range(num_minibatches):
        yield (
            obs[i * max_batch_size : (i + 1) * max_batch_size],
            node_obs[i * max_batch_size : (i + 1) * max_batch_size],
            adj[i * max_batch_size : (i + 1) * max_batch_size],
            agent_id[i * max_batch_size : (i + 1) * max_batch_size],
        )


class GR_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    args: argparse.Namespace
        Arguments containing relevant model information.
    obs_space: (gym.Space)
        Observation space.
    node_obs_space: (gym.Space)
        Node observation space
    edge_obs_space: (gym.Space)
        Edge dimension in graphs
    action_space: (gym.Space)
        Action space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        action_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(GR_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[
            1
        ]  # returns (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # returns (edge_dim,)
        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.actor_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim  # output shape from gnns
        mlp_base_in_dim = gnn_out_dim + obs_shape[0]
        self.base = MLPBase(args, obs_shape=None, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )
        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )

        self.to(device)

    def forward(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        node_obs (np.ndarray / torch.Tensor):
            Local agent graph node features to the actor.
        adj (np.ndarray / torch.Tensor):
            Adjacency matrix for the graph
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (np.ndarray / torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            Whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor)
            Actions to take.
        :return action_log_probs: (torch.Tensor)
            Log probabilities of taken actions.
        :return rnn_states: (torch.Tensor)
            Updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        return (actions, action_log_probs, rnn_states)

    def evaluate_actions(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability and entropy of given actions.
        obs: (torch.Tensor)
            Observation inputs into network.
        node_obs (torch.Tensor):
            Local agent graph node features to the actor.
        adj (torch.Tensor):
            Adjacency matrix for the graph.
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        action: (torch.Tensor)
            Actions whose entropy and log probability to evaluate.
        rnn_states: (torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        active_masks: (torch.Tensor)
            Denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor)
            Log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor)
            Action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'eval Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        return (action_log_probs, dist_entropy)


class GR_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions
    given centralized input (MAPPO) or local observations (IPPO).
    args: (argparse.Namespace)
        Arguments containing relevant model information.
    cent_obs_space: (gym.Space)
        (centralized) observation space.
    node_obs_space: (gym.Space)
        node observation space.
    edge_obs_space: (gym.Space)
        edge observation space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(GR_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[
            1
        ]  # (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # (edge_dim,)

        # TODO modify output of GNN to be some kind of global aggregation
        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.critic_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim
        # if node aggregation, then concatenate aggregated node features for all agents
        # otherwise, the aggregation is done for the whole graph
        if args.critic_graph_aggr == "node":
            gnn_out_dim *= args.num_agents
        mlp_base_in_dim = gnn_out_dim
        if self.args.use_cent_obs:
            mlp_base_in_dim += cent_obs_shape[0]

        self.base = MLPBase(args, cent_obs_shape, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(
        self, cent_obs, node_obs, adj, agent_id, rnn_states, masks
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        cent_obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        node_obs (np.ndarray):
            Local agent graph node features to the actor.
        adj (np.ndarray):
            Adjacency matrix for the graph.
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if RNN states
            should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (cent_obs.shape[0] > self.max_batch_size):
            # print(f'Cent obs: {cent_obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                cent_obs, node_obs, adj, agent_id, self.max_batch_size
            )
            critic_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                critic_feats_batch = self.base(act_feats_batch)
                critic_features.append(critic_feats_batch)
            critic_features = torch.cat(critic_features, dim=0)
        else:
            nbd_features = self.gnn_base(
                node_obs, adj, agent_id
            )  # CHECK from where are these agent_ids coming
            if self.args.use_cent_obs:
                critic_features = torch.cat(
                    [cent_obs, nbd_features], dim=1
                )  # NOTE can remove concatenation with cent_obs and just use graph_feats
            else:
                critic_features = nbd_features
            critic_features = self.base(critic_features)  # Cent obs here

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)
        return (values, rnn_states)

class PopArt(torch.nn.Module):
    """
    Preserving Outputs Precisely while Adaptively Rescaling Targets
    https://deepmind.com/blog/article/preserving-outputs-precisely-while-adaptively-rescaling-targets
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        norm_axes=1,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        device=torch.device("cpu"),
    ):
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(
            **self.tpdv
        )
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)

        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(
            **self.tpdv
        )
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(
            **self.tpdv
        )
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(
            **self.tpdv
        )
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(
            **self.tpdv
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        old_mean, old_stddev = self.mean, self.stddev

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        # changed the next 3 lines according to this issue:
        # https://github.com/marlbenchmark/on-policy/issues/19#issue-939380562
        self.stddev = nn.Parameter(
            (self.mean_sq - self.mean**2).sqrt().clamp(min=1e-4)
        )

        self.weight = nn.Parameter(self.weight * old_stddev / self.stddev)
        self.bias = nn.Parameter(
            (old_stddev * self.bias + old_mean - self.mean) / self.stddev
        )

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]

        return out

    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (
            input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
            + mean[(None,) * self.norm_axes]
        )

        out = out.cpu().numpy()

        return out
class PopArt(torch.nn.Module):
    """
    Preserving Outputs Precisely while Adaptively Rescaling Targets
    https://deepmind.com/blog/article/preserving-outputs-precisely-while-adaptively-rescaling-targets
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        norm_axes=1,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        device=torch.device("cpu"),
    ):
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(
            **self.tpdv
        )
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)

        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(
            **self.tpdv
        )
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(
            **self.tpdv
        )
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(
            **self.tpdv
        )
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(
            **self.tpdv
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        old_mean, old_stddev = self.mean, self.stddev

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        # changed the next 3 lines according to this issue:
        # https://github.com/marlbenchmark/on-policy/issues/19#issue-939380562
        self.stddev = nn.Parameter(
            (self.mean_sq - self.mean**2).sqrt().clamp(min=1e-4)
        )

        self.weight = nn.Parameter(self.weight * old_stddev / self.stddev)
        self.bias = nn.Parameter(
            (old_stddev * self.bias + old_mean - self.mean) / self.stddev
        )

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]

        return out

    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (
            input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
            + mean[(None,) * self.norm_axes]
        )

        out = out.cpu().numpy()

        return out
