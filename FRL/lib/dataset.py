# dataset.py

import pickle
import torch
from torch.utils.data import Dataset


class Rating_Datset(torch.utils.data.Dataset):
    def __init__(self, user_list, item_list, rating_list):
        super(Rating_Datset, self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.rating_list = rating_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item = self.item_list[idx]
        rating = self.rating_list[idx]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
        )


class IRSDataset(Dataset):
    def __init__(self, data):
        super(IRSDataset, self).__init__()
        self.data = data
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float)