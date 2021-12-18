# aggregator.py

import torch

from lib.utils import *


class GraphAggregator(torch.nn.Module):
    def __init__(self, k_hop, num_embed, embed_size,
                 name='graph_aggregator'):
        super(GraphAggregator, self).__init__()
        self.k_hop = k_hop  # depth of graph-like tree
        self.entity_embeds = torch.nn.Embedding(
            num_embeddings=num_embed,
            embedding_dim=embed_size)

    def get_embedding(self, inputs):
        # Function to get entity_embeddings
        return self.entity_embeds(inputs)

    def forward(self, graph, nodes):
        # Return aggregated node representation and item candidates
        #   - nodes: list of nodes (aka clicks) per user
        #   - graph: networkx object
        nodes = [self._aggregate(graph=graph, nodes=[node],
                                                  k_hop=self.k_hop) for node in nodes]

        return nodes

    def _aggregate(self, graph, nodes, k_hop):
        # Function to recursively aggregate k-hop-depth nodes
        #   - graph: networkx object
        #   - nodes: list of tensors (aka nodes)
        #   - k_hop

        if k_hop > 0:
            # find neighbor nodes & find candidates
            for i, node in enumerate(nodes):
                # get neighbor nodes
                nodes[i] = list(graph.neighbors(node))

                # recursively call to next depth
                nodes[i] = self._aggregate(graph=graph, nodes=nodes[i],
                                           k_hop=k_hop - 1)

        # concat by number of past-clicks
        nodes = torch.cat(nodes, dim=0)

        # lookup embeddings
        nodes = self.get_embedding(nodes)

        # average the last dim
        nodes = torch.mean(nodes, dim=-1)

        return nodes

    def candidate_select(self, graph, nodes):
        return self._candidate_select(graph=graph, nodes=[nodes])

    def _candidate_select(self, graph, nodes, k_hop):
        # Function to recursively aggregate k-hop-depth nodes
        # and select candidates
        #   - graph: networkx object
        #   - nodes: list of tensors (aka nodes)
        #   - k_hop

        # filter item candidtes only
        candidates = [node for node in nodes if graph.nodes[node]['is_item'] is True]

        if k_hop > 0:
            # find neighbor nodes & find candidates
            for i, nodes in enumerate(candidates):
                # get neighbor nodes
                nodes = list(graph.neighbors(nodes))

                # recursively call to next depth
                candidates.extend(self._aggregate(graph=graph, nodes=nodes,
                                           k_hop=k_hop - 1))

        # select unique candidates only
        return list(set(candidates))


class BehaviorAggregator(torch.nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers=1,
                 name='behavior_aggregator'):
        super(BehaviorAggregator, self).__init__()
        self.gru = torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)

    def forward(self, inputs):
        return self.gru(inputs)
