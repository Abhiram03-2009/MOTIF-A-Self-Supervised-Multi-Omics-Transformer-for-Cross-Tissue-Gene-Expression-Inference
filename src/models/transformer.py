"""
models/transformer.py

MOTIF (Multi-Omics Transformer for Inferring Function). Class name kept as
MultiOmicsTransformer in code; "MOTIF" is the project/paper-facing name
(see outputs/results/manuscript_draft.md).

Each token is a genomic region (ATAC peak or methylation site); attention
runs jointly across ATAC and methylation tokens rather than just
concatenating the two modalities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.configs.config import Config

cfg = Config()


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Adds position information to each token so the model knows
    which genomic position each feature comes from.

    This is the same as the original "Attention is All You Need" PE.
    """

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CrossModalEmbedding(nn.Module):
    """
    Embeds multi-omics features into a shared token space.

    Instead of simple concatenation, each omics modality gets its own
    projection, then they are combined with learned weights.

    Inputs:
        atac: (batch, n_peaks)
        meth: (batch, n_sites) or None

    Output:
        tokens: (batch, seq_len, hidden_dim)
        where seq_len = n_peaks (each peak is one token)
    """

    def __init__(self, atac_dim: int, meth_dim: int, hidden_dim: int):
        super().__init__()
        self.atac_proj = nn.Linear(1, hidden_dim)
        self.meth_proj = nn.Linear(1, hidden_dim)
        self.fusion    = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)

        self.use_meth  = (meth_dim > 0)
        self.seq_len   = atac_dim  # number of tokens = number of peaks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, atac_dim + meth_dim) — concatenated omics features

        Returns: (batch, atac_dim, hidden_dim)
        """
        batch = x.shape[0]
        atac_dim = self.seq_len

        atac = x[:, :atac_dim]          # (batch, atac_dim)
        meth = x[:, atac_dim:]          # (batch, meth_dim) or empty

        # Treat each peak position as a token: (batch, atac_dim, 1)
        atac_tokens = atac.unsqueeze(-1)                      # (B, L, 1)
        atac_embed  = self.atac_proj(atac_tokens)             # (B, L, H)

        if self.use_meth and meth.shape[1] > 0:
            meth_tokens = meth.unsqueeze(-1)                  # (B, L, 1)
            meth_embed  = self.meth_proj(meth_tokens)         # (B, L, H)
            combined    = torch.cat([atac_embed, meth_embed], dim=-1)  # (B, L, 2H)
            tokens      = self.fusion(combined)               # (B, L, H)
        else:
            tokens = atac_embed                               # (B, L, H)

        return self.norm(tokens)


class MultiOmicsTransformer(nn.Module):
    """
    Multi-Omics Foundation Transformer. `forward(x, pretrain=True)` runs
    the masked-reconstruction pretraining head; `pretrain=False` runs the
    gene expression fine-tuning head. Both share the same encoder.

    Args:
        input_dim:   total input size (atac_dim + meth_dim)
        atac_dim:    number of ATAC peaks (= sequence length)
        meth_dim:    number of methylation sites
        output_dim:  number of genes to predict
        hidden_dim:  transformer embedding dimension
        num_heads:   number of attention heads
        num_layers:  number of transformer encoder layers
        ff_dim:      feedforward layer size inside transformer
        dropout:     dropout rate
    """

    def __init__(
        self,
        input_dim:  int,
        atac_dim:   int,
        meth_dim:   int,
        output_dim: int,
        hidden_dim: int = 256,
        num_heads:  int = 4,
        num_layers: int = 3,
        ff_dim:     int = 512,
        dropout:    float = 0.1,
        readout:    str = "token",
    ):
        super().__init__()
        assert readout in ("token", "pooled")
        assert readout == "pooled" or atac_dim == output_dim, \
            "token readout needs one token per output gene"
        self.readout   = readout
        self.atac_dim  = atac_dim
        self.meth_dim  = meth_dim
        self.hidden_dim = hidden_dim

        self.embedding = CrossModalEmbedding(atac_dim, meth_dim, hidden_dim)

        self.pos_enc   = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer  = nn.TransformerEncoderLayer(
            d_model         = hidden_dim,
            nhead           = num_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            batch_first     = True,   # (batch, seq, features)
            norm_first      = True,   # Pre-LN (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm       = nn.LayerNorm(hidden_dim),
        )

        self.pretrain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # reconstruct each peak value
        )

        # "pooled": mean-pool all tokens, then map to every gene at once.
        # "token": each gene's own token predicts that gene's expression.
        self.finetune_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )
        self.token_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shared encoder: input → contextualized token embeddings.
        Used by both pretraining and fine-tuning.

        x: (batch, input_dim)
        → tokens: (batch, atac_dim, hidden_dim)
        """
        tokens = self.embedding(x)           # (B, L, H)
        tokens = self.pos_enc(tokens)        # (B, L, H)
        tokens = self.transformer(tokens)    # (B, L, H)
        return tokens

    def forward(self, x: torch.Tensor,
                pretrain: bool = False) -> dict:
        """
        Forward pass.

        Args:
            x:        (batch, input_dim) — concatenated omics features
            pretrain: True = masked reconstruction, False = expression prediction

        Returns:
            dict with:
              "pred":      output predictions
              "embedding": global mean embedding (for downstream analysis)
        """
        tokens = self.encode(x)                         # (B, L, H)
        global_embed = tokens.mean(dim=1)               # (B, H) — mean pooling

        if pretrain:
            # Per-token reconstruction: (B, L, 1) → (B, L)
            recon = self.pretrain_head(tokens).squeeze(-1)
            return {"pred": recon, "embedding": global_embed}
        else:
            if self.readout == "token":
                expr_pred = self.token_head(tokens).squeeze(-1)      # (B, n_genes)
            else:
                expr_pred = self.finetune_head(global_embed)          # (B, n_genes)
            return {"pred": expr_pred, "embedding": global_embed}

    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights for interpretability analysis.
        Returns a list of attention matrices, one per layer.

        NOTE: PyTorch's TransformerEncoder doesn't expose attention by default.
        This method uses a custom extraction approach.
        """
        attn_weights = []
        tokens = self.embedding(x)
        tokens = self.pos_enc(tokens)

        for layer in self.transformer.layers:
            attn_out, attn_w = layer.self_attn(
                tokens, tokens, tokens,
                need_weights=True,
                average_attn_weights=True
            )
            attn_weights.append(attn_w.detach().cpu())
            tokens = layer(tokens)

        return attn_weights


if __name__ == "__main__":
    ATAC_DIM   = cfg.model.ATAC_DIM
    METH_DIM   = cfg.model.METH_DIM
    RNA_DIM    = cfg.model.RNA_DIM
    INPUT_DIM  = ATAC_DIM + METH_DIM
    HIDDEN_DIM = cfg.model.HIDDEN_DIM

    model = MultiOmicsTransformer(
        input_dim  = INPUT_DIM,
        atac_dim   = ATAC_DIM,
        meth_dim   = METH_DIM,
        output_dim = RNA_DIM,
        hidden_dim = HIDDEN_DIM,
        num_heads  = cfg.model.NUM_HEADS,
        num_layers = cfg.model.NUM_LAYERS,
        ff_dim     = cfg.model.FF_DIM,
        dropout    = cfg.model.DROPOUT,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    x = torch.randn(4, INPUT_DIM)
    out = model(x, pretrain=True)
    print(f"Pretrain output: {out['pred'].shape}")   # (4, atac_dim)

    out = model(x, pretrain=False)
    print(f"Finetune output: {out['pred'].shape}")   # (4, n_genes)

    print("Transformer OK.")
