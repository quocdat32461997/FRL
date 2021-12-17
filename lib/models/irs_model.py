# irs.py

import torch
from .graph_model import *
from .mdp_model import *

from lib.utils import *


class IRSModel(torch.nn.Module):
    def __init__(self, args, num_users, num_items, name='irs'):
        super(IRSModel, self).__init__(name=name)
        self.graph = GraphModel(k_hop=args.k_hop, graph=args.graph)
        self.recsys = NeuMF(args=args, num_users=num_users, num_items=num_items)
        pass

    def forward(self, inputs):
        return self.recsys(inputs)

    def recommend(self, inputs):
        return None