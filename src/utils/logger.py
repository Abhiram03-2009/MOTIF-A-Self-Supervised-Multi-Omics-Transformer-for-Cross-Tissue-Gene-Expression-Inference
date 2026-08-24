"""
utils/logger.py

Simple CSV logger for tracking training metrics across epochs.
Saves to outputs/logs/<name>.csv for later plotting.
"""

import csv
import time
from pathlib import Path


class Logger:
    """
    Logs training metrics to a CSV file.

    Usage:
        logger = Logger("pretrain")
        logger.log_epoch(1, train_loss=0.5, val_loss=0.4)
    """

    def __init__(self, name: str, log_dir: str = "outputs/logs"):
        self.name    = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.path    = self.log_dir / f"{name}.csv"
        self._initialized = False

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Log one epoch of metrics. Extra kwargs become extra columns."""
        row = {
            "epoch":      epoch,
            "timestamp":  time.strftime("%H:%M:%S"),
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
        }
        row.update({k: round(float(v), 6) for k, v in kwargs.items()})

        if not self._initialized:
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            self._initialized = True

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def log_step(self, epoch: int, step: int, key: str, value: float):
        """Log a mid-epoch step (prints only, does not save to CSV)."""
        pass  # kept simple — extend if you want step-level logging
