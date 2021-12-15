# graph_model.py

import pickle
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from lib.models.aggregator import GraphAggregator, BehaviorAggregator
from lib.utils import *


class GraphModel(torch.nn.Module):
    def __init__(self, k_hop, graph,
                 name='graph_model'):
        super(GraphModel, self).__init__(name=name)
        self.graph_conv = GraphConv(k_hop=k_hop,
                                    graph=graph)
        self.behavior_aggregator = BehaviorAggregator(input_size=None,
                                                      hidden_size=None)

    def forward(self, inputs):
        # inputs: list of past-click list
        for i in range(len(inputs)):
            # yield node representation
            inputs[i] = self.graph_conv(to_cuda(inputs[i]))

            # yield state representation based on play-history
            inputs[i] = self.behavior_aggregator(inputs[i])

        return inputs


class GraphConv(torch.nn.Module):
    def __init__(self, k_hop, graph,
                 num_embed, embed_size,
                 in_features, out_features, dropout=0.0,
                 name='graph_conv'):
        super(GraphConv, self).__init__(name=name)
        self.aggregator = GraphAggregator(k_hop=k_hop,
                                          num_embed=num_embed,
                                          embed_size=embed_size)
        self.graph = pickle.load(open(graph, 'rb'))

        # linear parameters
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, inputs):
        # inputs: list of past-clicks

        # [0] - aggregate representation of neighboring nodes
        # [1] - get node features
        features = [self.aggregator(self.graph, inputs),
                    self.aggregator.get_embedding(inputs)]

        # linear transform the neighboring node representation
        # and return results
        return self._linear_transform(features=features)

    def _linear_transform(self, features):
        # Function to linearly transform neighbor and node features
        features[0] = torch.matmul(features[0], self.weight)
        features[1] = torch.matmul(features[1], self.bias)

        return torch.nn.functional.relu(
            self.dropout(sum(features)))


class CandidateSelect(torch.nn.Module):
    def __init__(self, name='candidate_select'):
        super(CandidateSelect, self).__init__(name=name)
        pass

    def forward(self, inputs):
        return None