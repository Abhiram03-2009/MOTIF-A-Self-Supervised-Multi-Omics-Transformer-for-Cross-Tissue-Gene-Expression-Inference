"""
training/train_diffusion.py

Train conditional DDPM for synthetic multi-omics profile generation.

Usage:
    python training/train_diffusion.py
    python main.py --mode diffusion
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configs.config import Config
from src.models.diffusion import ConditionalDenoiser, GaussianDiffusion
from data.dataset import get_dataloader
from src.utils.logger import Logger
from src.utils.helpers import get_device, count_params

cfg = Config()
device = get_device()


def build_diffusion() -> GaussianDiffusion:
    denoiser = ConditionalDenoiser(
        profile_dim=cfg.model.INPUT_DIM,
        n_tissues=len(cfg.data.TISSUES),
        hidden_dim=cfg.diffusion.HIDDEN_DIM,
        time_dim=cfg.diffusion.TIME_DIM,
        label_dim=cfg.diffusion.LABEL_DIM,
    )
    return GaussianDiffusion(
        denoiser=denoiser,
        profile_dim=cfg.model.INPUT_DIM,
        timesteps=cfg.diffusion.TIMESTEPS,
        beta_start=cfg.diffusion.BETA_START,
        beta_end=cfg.diffusion.BETA_END,
    ).to(device)


def train_diffusion():
    print("\n" + "=" * 60)
    print(" PHASE 3: Train Conditional Diffusion Model")
    print("=" * 60)

    train_loader = get_dataloader(
        "train", cfg.diffusion.BATCH_SIZE,
        use_methylation=cfg.data.USE_METHYLATION,
    )
    val_loader = get_dataloader(
        "val", cfg.diffusion.BATCH_SIZE,
        use_methylation=cfg.data.USE_METHYLATION,
    )

    model = build_diffusion()
    print(f"Diffusion params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.diffusion.LEARNING_RATE,
        weight_decay=cfg.diffusion.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.diffusion.EPOCHS
    )
    logger = Logger("diffusion")

    best_val_loss = float("inf")
    Path("outputs/checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, cfg.diffusion.EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Diffusion Epoch {epoch}/{cfg.diffusion.EPOCHS}",
                          leave=False):
            x = batch["x"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = model.training_loss(x, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.diffusion.GRAD_CLIP)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                labels = batch["label"].to(device)
                val_losses.append(model.training_loss(x, labels).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        logger.log_epoch(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/checkpoints/diffusion_best.pt")
            print(f"  Saved best diffusion model (val_loss={val_loss:.4f})")

    torch.save(model.state_dict(), "outputs/checkpoints/diffusion_final.pt")
    print("\nDiffusion training complete.")
    print("  outputs/checkpoints/diffusion_best.pt")
    return model


def generate_synthetic_profiles(model: GaussianDiffusion, n_samples: int,
                                 tissue_label: int, fast: bool = True) -> np.ndarray:
    """Generate synthetic multi-omics profiles for a tissue."""
    model.eval()
    labels = torch.full((n_samples,), tissue_label, dtype=torch.long, device=device)
    with torch.no_grad():
        if fast:
            samples = model.sample_fast(labels, steps=cfg.diffusion.SAMPLE_STEPS)
        else:
            samples = model.sample(labels)
    return samples.cpu().numpy()


if __name__ == "__main__":
    train_diffusion()
