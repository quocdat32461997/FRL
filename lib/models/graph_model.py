# graph_model.py

import torch

from lib.models.aggregator import GraphAggregator, BehaviorAggregator
from lib.utils import *


class GraphModel(torch.nn.Module):
    def __init__(self, args, num_embed, name='graph_model'):
        super(GraphModel, self).__init__()
        self.graph_conv = GraphConv(k_hop=args.k_hop,
                                    graph=args.graph,
                                    num_embed=num_embed,
                                    embed_size=args.embed_size,
                                    in_features=args.in_feature,
                                    out_features=args.out_feature,
                                    dropout=args.dropout)
        self.behavior_aggregator = BehaviorAggregator(input_size=args.out_feature,
                                                      hidden_size=args.out_feature)

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
        super(GraphConv, self).__init__()
        self.aggregator = GraphAggregator(k_hop=k_hop,
                                          num_embed=num_embed,
                                          embed_size=embed_size)
        self.graph = pickle.load(open(graph, 'rb'))

        # linear parameters
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, inputs, candidates):
        # inputs: list of past-clicks
        # candidates: list of past-clicks + their neighbors extracted previously

        # [0] - aggregate representation of neighboring nodes
        # [1] - get node features
        features = [self.aggregator(graph=self.graph, nodes=inputs),
                    self.aggregator.get_embedding(inputs=inputs)]

        # linear transform the neighboring node representation
        # and return results
        features = self._linear_transform(features=features)

        # get neighbor nodes for the lastest item
        candidates.extend(self.aggregator.candidate_select(grap=graph,
                                                      nodes=inputs[-1]))
        # get embeddings of candidates
        cand_features = self.aggregator(graph=self.graph, nodes=candidates)
        cand_features = torch.cat(cand_features, dim=0)

        return features, cand_features

    def _linear_transform(self, features):
        # Function to linearly transform neighbor and node features
        features[0] = torch.matmul(features[0], self.weight)
        features[1] = torch.matmul(features[1], self.bias)

        return torch.nn.functional.relu(
            self.dropout(sum(features)))