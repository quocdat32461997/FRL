# utils.py

import torch


def to_cuda(x):
    # Function to move torch.tensor either CPU or CUDA
    if torch.cuda.is_available():
        return x.to('cuda')
    else:
        return x.to('cpu')