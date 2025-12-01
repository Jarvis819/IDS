# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class FlowSequenceDataset(Dataset):
    def __init__(self, x_path, y_path):
        # Load preprocessed numpy arrays
        self.X = np.load(x_path)   # shape: (N, L, F)
        self.y = np.load(y_path)   # shape: (N, L)

        assert self.X.shape[0] == self.y.shape[0], "X and y window counts must match"
        self.num_windows, self.seq_len, self.num_features = self.X.shape

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        # Return tensors for PyTorch
        x_win = torch.tensor(self.X[idx], dtype=torch.float32)  # (L, F)
        y_win = torch.tensor(self.y[idx], dtype=torch.long)     # (L,)
        return x_win, y_win

class FlowSequenceDatasetFromArrays(Dataset):
    """Dataset that directly wraps X_seq, y_seq numpy arrays."""
    def __init__(self, X_seq, y_seq):
        assert X_seq.shape[0] == y_seq.shape[0], "X and y window counts must match"
        self.X = X_seq
        self.y = y_seq
        self.num_windows, self.seq_len, self.num_features = self.X.shape

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        x_win = torch.tensor(self.X[idx], dtype=torch.float32)   # (L, F)
        y_win = torch.tensor(self.y[idx], dtype=torch.long)      # (L,)
        return x_win, y_win