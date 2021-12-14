# irs.py

import torch
from .graph_model import *
from .mdp_model import *

from lib.utils import *


class IRSModel(torch.nn.Module):
    def __init__(self, name='irs'):
        super(IRSModel, self).__init__(name=name)
        self.graph_conv = GraphConv()
        pass

    def forward(self, inputs):
        return None

    def recommend(self, inputs):
        return None