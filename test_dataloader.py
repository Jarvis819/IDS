# test_dataloader.py
import torch
from torch.utils.data import DataLoader
from dataset import FlowSequenceDataset

# 🔥 Adjust paths if needed
X_PATH = r"processed\\preprocessed_X_seq.npy"
Y_PATH = r"processed\\preprocessed_y_seq.npy"

def main():
    ds = FlowSequenceDataset(X_PATH, Y_PATH)
    print("Dataset windows:", len(ds))
    print("One window shape (X, y):", ds[0][0].shape, ds[0][1].shape)

    loader = DataLoader(ds, batch_size=32, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Take one batch
    xb, yb = next(iter(loader))
    print("Batch X shape:", xb.shape)  # (B, L, F)
    print("Batch y shape:", yb.shape)  # (B, L)

    xb = xb.to(device)
    yb = yb.to(device)
    print("Moved batch to device successfully.")

if __name__ == "__main__":
    main()
