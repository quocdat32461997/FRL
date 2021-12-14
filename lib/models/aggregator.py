# aggregator.py

import torch

from lib.utils import *


class GraphAggregator(torch.nn.Module):
    def __init__(self, k_hop, num_embed, embed_size,
                 name='graph_aggregator'):
        super(GraphAggregator, self).__init__(name=name)
        self.k_hop = k_hop  # depth of graph-like tree
        self.entity_embeds = torch.nn.Embedding(
            num_embeddinigs=num_embed,
            embedding_dim=embed_size)

    def get_embedding(self, inputs):
        # Function to get entity_embeddings
        return self.entity_embeds(inputs)

    def forward(self, graph, nodes):
        nodes = [self._aggregate(graph=graph,
                                 nodes=[node],
                                 k_hop=self.k_hop) for node in nodes]
        return nodes

    def _aggregate(self, graph, nodes, k_hop):
        # Function to recursively aggregate k-hop-depth nodes
        if k_hop > 0:
            # find neighbor nodes & loop by batch
            nodes = [list(graph.neighbors(node)) for node in nodes]

            # recursively call to next depth
            nodes = [self._aggregate(graph, _nodes, k_hop - 1)
                     for _nodes in nodes]

        # concat by batch
        nodes = torch.cat(nodes, dim=0)

        # lookup embeddings
        nodes = self.get_embedding(nodes)

        # average the last dim
        nodes = torch.mean(nodes, dim=-1)

        return nodes


class BehaviorAggregator(torch.nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers=1,
                 name='behavior_aggregator'):
        super(BehaviorAggregator, self).__init__(name=name)
        self.gru = torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)

    def forward(self, inputs):
        return inputs