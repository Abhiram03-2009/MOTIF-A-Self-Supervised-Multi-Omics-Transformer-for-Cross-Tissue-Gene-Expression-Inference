"""
models/baseline.py

Simple MLP baseline models.
These must be trained first before the transformer — you need baseline numbers
to prove the transformer is better.

Models:
  - LinearBaseline:  linear regression (simplest possible)
  - MLPBaseline:     3-layer MLP
"""

import torch
import torch.nn as nn


class LinearBaseline(nn.Module):
    """
    Single linear layer: input features → gene expression.
    This is the simplest possible baseline.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, output_dim)"""
        return self.linear(x)


class MLPBaseline(nn.Module):
    """
    3-layer MLP baseline.

    Architecture:
        Input → Linear(256) → LayerNorm → ReLU → Dropout
              → Linear(128) → LayerNorm → ReLU → Dropout
              → Linear(output_dim)
    """
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: tuple = (256, 128),
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, output_dim)"""
        return self.net(x)
