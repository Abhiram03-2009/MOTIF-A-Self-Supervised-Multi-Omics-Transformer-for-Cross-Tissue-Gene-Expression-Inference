"""
data/dataset.py

PyTorch Dataset classes for loading multi-omics arrays.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.configs.config import Config

cfg = Config()


class MultiOmicsDataset(Dataset):
    """
    Dataset for multi-omics data (ATAC + methylation → RNA expression).

    Args:
        split:          "train", "val", or "test"
        use_methylation: whether to include methylation as a third modality
        mask_ratio:     fraction of input to mask (for pretraining). 0 = no masking.
    """

    def __init__(self, split: str = "train",
                 use_methylation: bool = True,
                 mask_ratio: float = 0.0):

        processed_dir = Path(cfg.data.PROCESSED_DIR) / split

        self.rna    = torch.FloatTensor(np.load(processed_dir / "rna.npy"))
        self.atac   = torch.FloatTensor(np.load(processed_dir / "atac.npy"))
        self.labels = torch.LongTensor(np.load(processed_dir / "labels.npy"))

        self.use_methylation = use_methylation
        if use_methylation:
            self.meth = torch.FloatTensor(np.load(processed_dir / "meth.npy"))

        self.mask_ratio = mask_ratio
        self.n_samples  = len(self.rna)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        atac  = self.atac[idx]
        rna   = self.rna[idx]
        label = self.labels[idx]

        if self.use_methylation:
            meth = self.meth[idx]
            x = torch.cat([atac, meth], dim=0)  # concatenate modalities
        else:
            x = atac

        mask = torch.zeros(x.shape[0], dtype=torch.bool)
        if self.mask_ratio > 0:
            n_mask = int(self.mask_ratio * x.shape[0])
            mask_idx = torch.randperm(x.shape[0])[:n_mask]
            mask[mask_idx] = True
            # Store original before zeroing
            x_masked = x.clone()
            x_masked[mask] = 0.0
            return {
                "x":        x_masked,   # masked input
                "x_orig":   x,          # original (pretraining target)
                "mask":     mask,       # which positions were masked
                "rna":      rna,        # expression (fine-tune target)
                "label":    label,      # tissue label
            }

        return {
            "x":        x,
            "x_orig":   x,
            "mask":     mask,
            "rna":      rna,
            "label":    label,
        }


def get_dataloader(split: str,
                   batch_size: int = 32,
                   use_methylation: bool = True,
                   mask_ratio: float = 0.0,
                   shuffle: bool = None) -> DataLoader:
    """
    Convenience function to get a DataLoader.

    Args:
        split:          "train", "val", or "test"
        batch_size:     mini-batch size
        use_methylation: include methylation modality
        mask_ratio:     masking ratio for pretraining (0 = disabled)
        shuffle:        default True for train, False otherwise
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = MultiOmicsDataset(
        split=split,
        use_methylation=use_methylation,
        mask_ratio=mask_ratio,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,       # increase if on a cluster with many CPU cores
        pin_memory=torch.cuda.is_available(),
    )

    return loader


if __name__ == "__main__":
    print("Testing DataLoader...")
    loader = get_dataloader("train", batch_size=8, mask_ratio=0.15)
    batch  = next(iter(loader))
    print("x shape:     ", batch["x"].shape)
    print("rna shape:   ", batch["rna"].shape)
    print("mask shape:  ", batch["mask"].shape)
    print("labels:      ", batch["label"])
    print("DataLoader OK.")
