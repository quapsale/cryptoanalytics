"""
File: datasets.py
Description: PyTorch Datasets for LSTM e GRU models.
File Created: 06/04/2022
Python Version: 3.9
"""

# Imports
import torch
from torch.utils.data import Dataset


# Sequence Dataset V1
class DatasetV1(Dataset):
    """
    Constructor.
    """
    def __init__(self, dataframe, target, features, sequence_length=30):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]
