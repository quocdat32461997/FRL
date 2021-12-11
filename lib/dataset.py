# dataset.py

import torch
from torch.utils.data import Dataset


class IRSDataset(Dataset):
    def __init__(self, data):
        super(IRSDataset, self).__init__()
        self.data = data
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float)