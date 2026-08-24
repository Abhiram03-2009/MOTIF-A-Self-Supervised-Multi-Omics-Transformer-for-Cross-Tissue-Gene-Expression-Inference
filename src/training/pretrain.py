"""
training/pretrain.py

Self-supervised pretraining of MOTIF: masks part of each sample's
multi-omics profile and trains the model to reconstruct the masked
values, so it learns regulatory structure before ever seeing an
expression label.

Usage:
    python training/pretrain.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configs.config   import Config
from src.models.transformer import MultiOmicsTransformer
from data.dataset     import get_dataloader
from src.utils.logger     import Logger
from src.utils.helpers    import get_device, save_checkpoint, count_params

cfg    = Config()
device = get_device()


def pretrain():
    print("\n" + "=" * 60)
    print(" PHASE 1: Self-Supervised Pretraining")
    print("=" * 60)

    train_loader = get_dataloader(
        split           = "train",
        batch_size      = cfg.pretrain.BATCH_SIZE,
        use_methylation = cfg.data.USE_METHYLATION,
        mask_ratio      = cfg.model.MASK_RATIO,
    )
    val_loader = get_dataloader(
        split           = "val",
        batch_size      = cfg.pretrain.BATCH_SIZE,
        use_methylation = cfg.data.USE_METHYLATION,
        mask_ratio      = cfg.model.MASK_RATIO,
    )

    model = MultiOmicsTransformer(
        input_dim  = cfg.model.INPUT_DIM,
        atac_dim   = cfg.model.ATAC_DIM,
        meth_dim   = cfg.model.METH_DIM if cfg.data.USE_METHYLATION else 0,
        output_dim = cfg.model.RNA_DIM,
        hidden_dim = cfg.model.HIDDEN_DIM,
        num_heads  = cfg.model.NUM_HEADS,
        num_layers = cfg.model.NUM_LAYERS,
        ff_dim     = cfg.model.FF_DIM,
        dropout    = cfg.model.DROPOUT,
    ).to(device)

    print(f"Model parameters: {count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.pretrain.LEARNING_RATE,
        weight_decay = cfg.pretrain.WEIGHT_DECAY,
    )

    n_batches = len(train_loader)
    total_steps = max(1, cfg.pretrain.EPOCHS * n_batches)
    warmup_steps = min(cfg.pretrain.WARMUP_STEPS, max(1, total_steps - 1))
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup_sched = LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    def masked_mse(pred: torch.Tensor,
                   target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE only on masked (held-out) positions.

        pred:   (batch, atac_dim) — model reconstruction
        target: (batch, atac_dim) — original (unmasked) input
        mask:   (batch, input_dim) — True = position was masked

        We only take the ATAC portion of the mask (first atac_dim positions).
        """
        atac_mask = mask[:, :cfg.model.ATAC_DIM]  # mask over ATAC positions
        if atac_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return F.mse_loss(pred[atac_mask], target[atac_mask])

    logger = Logger("pretrain")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = cfg.pretrain.EARLY_STOP_PATIENCE

    for epoch in range(1, cfg.pretrain.EPOCHS + 1):
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(tqdm(train_loader,
                                               desc=f"Epoch {epoch}/{cfg.pretrain.EPOCHS}",
                                               leave=False)):
            x      = batch["x"].to(device)        # masked input
            x_orig = batch["x_orig"].to(device)   # original (target)
            mask   = batch["mask"].to(device)      # which positions masked

            optimizer.zero_grad()
            out  = model(x, pretrain=True)
            pred = out["pred"]                     # (batch, atac_dim)

            # Target: original ATAC values
            target = x_orig[:, :cfg.model.ATAC_DIM]
            loss   = masked_mse(pred, target, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.pretrain.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

            if (batch_idx + 1) % cfg.pretrain.LOG_EVERY == 0:
                avg = np.mean(train_losses[-cfg.pretrain.LOG_EVERY:])
                logger.log_step(epoch, batch_idx, "train_loss", avg)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x      = batch["x"].to(device)
                x_orig = batch["x_orig"].to(device)
                mask   = batch["mask"].to(device)

                out    = model(x, pretrain=True)
                pred   = out["pred"]
                target = x_orig[:, :cfg.model.ATAC_DIM]
                loss   = masked_mse(pred, target, mask)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        lr_now     = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {lr_now:.2e}")

        logger.log_epoch(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss,
                            "outputs/checkpoints/pretrained_best.pt")
            print(f"  Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"  Early stopping (no val improvement for {patience} epochs).")
                break

    # Save final
    save_checkpoint(model, optimizer, cfg.pretrain.EPOCHS, val_loss,
                    "outputs/checkpoints/pretrained_final.pt")

    print(f"\nPretraining complete. Best val loss: {best_val_loss:.4f}")
    print("Checkpoint saved → outputs/checkpoints/pretrained_best.pt")
    print("Next step: python training/finetune.py")

    return model


if __name__ == "__main__":
    pretrain()
