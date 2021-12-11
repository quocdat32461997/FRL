# irs.py

import torch
from .graph_model import *
from .mdp_model import *


class IRSModel(torch.nn.Module):
    def __init__(self):
        super(IRSModel, self).__init__()
        pass

    def forward(self, inputs):
        return None

    def recommend(self, inputs):
        return None