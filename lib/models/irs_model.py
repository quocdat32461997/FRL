# irs.py

import torch
from .graph_model import *
from .mdp_model import *

from lib.utils import *


class IRSModel(torch.nn.Module):
    def __init__(self, args, num_users, num_items, name='irs'):
        super(IRSModel, self).__init__()
        self.graph = GraphModel(args=args, num_embed=num_items)
        self.recsys = NeuMF(args=args, num_users=num_users, num_items=num_items)
        pass

    def forward(self, users, items):
        return self.recsys(users, items)

    def recommend(self, inputs):
        return None