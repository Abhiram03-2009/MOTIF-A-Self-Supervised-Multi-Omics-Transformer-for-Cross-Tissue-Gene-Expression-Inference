"""
utils/helpers.py

Shared utilities: device detection, checkpoint save/load, parameter counting.
"""

import torch
from pathlib import Path


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (M-series GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")
    return device


def count_params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch: int, val_loss: float, path: str):
    """Save model + optimizer state dict."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "val_loss":   val_loss,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)


def load_checkpoint(model, path: str, optimizer=None) -> dict:
    """Load model weights from checkpoint. Optionally restore optimizer."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")
    return ckpt
