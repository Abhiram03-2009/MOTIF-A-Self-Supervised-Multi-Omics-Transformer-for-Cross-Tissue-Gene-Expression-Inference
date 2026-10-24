"""
training/finetune.py

Fine-tunes the pretrained MOTIF encoder on gene expression prediction and
compares it against linear/MLP baselines and a from-scratch transformer,
plus a cross-tissue generalization test (train on liver+brain, test on
lung) to see whether pretraining actually transfers.

Usage:
    python training/finetune.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configs.config     import Config
from src.models.transformer import MultiOmicsTransformer
from src.models.baseline    import MLPBaseline, LinearBaseline
from data.dataset       import get_dataloader
from src.utils.logger       import Logger
from src.utils.helpers      import get_device, save_checkpoint, load_checkpoint, count_params

cfg    = Config()
device = get_device()


def build_transformer(pretrained_path: str = None) -> MultiOmicsTransformer:
    """Build transformer, optionally loading pretrained weights."""
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

    if pretrained_path and Path(pretrained_path).exists():
        load_checkpoint(model, pretrained_path)
        print(f"  Loaded pretrained weights from: {pretrained_path}")

        # Optionally freeze encoder and only train fine-tune head
        if cfg.finetune.FREEZE_ENCODER:
            for name, param in model.named_parameters():
                if "finetune_head" not in name:
                    param.requires_grad = False
            print("  Encoder frozen — training fine-tune head only")
    else:
        print("  Training from scratch (no pretrained weights)")

    return model


def train_one_model(model: nn.Module, model_name: str,
                    train_loader, val_loader,
                    epochs: int, lr: float) -> dict:
    """
    Generic training loop for any model that takes batch['x'] → gene expression.

    Returns dict of best metrics.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = lr,
        weight_decay = cfg.finetune.WEIGHT_DECAY,
    )
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if cfg.finetune.USE_COSINE_SCHEDULE else None)
    criterion = nn.MSELoss()
    logger    = Logger(model_name)

    best_val_loss = float("inf")
    best_metrics  = {}
    patience = cfg.finetune.EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader,
                          desc=f"[{model_name}] Epoch {epoch}/{epochs}",
                          leave=False):
            x   = batch["x"].to(device)
            rna = batch["rna"].to(device)

            optimizer.zero_grad()

            if isinstance(model, MultiOmicsTransformer):
                out  = model(x, pretrain=False)
                pred = out["pred"]
            else:
                pred = model(x)

            loss = criterion(pred, rna)
            loss.backward()
            if cfg.finetune.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.finetune.GRAD_CLIP)
            optimizer.step()
            train_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_losses, all_pred, all_true = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x   = batch["x"].to(device)
                rna = batch["rna"].to(device)

                if isinstance(model, MultiOmicsTransformer):
                    out  = model(x, pretrain=False)
                    pred = out["pred"]
                else:
                    pred = model(x)

                loss = criterion(pred, rna)
                val_losses.append(loss.item())
                all_pred.append(pred.cpu().numpy())
                all_true.append(rna.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        pearson  = compute_pearson(all_pred, all_true)
        r2       = compute_r2(all_pred, all_true)

        print(f"[{model_name}] Epoch {epoch:3d} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Pearson: {pearson:.4f} | R²: {r2:.4f}")

        logger.log_epoch(epoch, train_loss, val_loss, pearson=pearson, r2=r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = {"val_loss": val_loss, "pearson": pearson, "r2": r2}
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss,
                            f"outputs/checkpoints/{model_name}_best.pt")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"  [{model_name}] Early stopping ({patience} epochs no improvement).")
                break

    return best_metrics


def compute_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Average Pearson correlation across all genes."""
    correlations = []
    for i in range(pred.shape[1]):
        p = np.corrcoef(pred[:, i], true[:, i])[0, 1]
        if not np.isnan(p):
            correlations.append(p)
    return float(np.mean(correlations))


def compute_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """R² coefficient of determination."""
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))


def finetune():
    print("\n" + "=" * 60)
    print(" PHASE 2: Fine-Tuning (Gene Expression Prediction)")
    print("=" * 60)

    INPUT_DIM = cfg.model.INPUT_DIM

    train_loader = get_dataloader("train", cfg.finetune.BATCH_SIZE,
                                  use_methylation=cfg.data.USE_METHYLATION)
    val_loader   = get_dataloader("val",   cfg.finetune.BATCH_SIZE,
                                  use_methylation=cfg.data.USE_METHYLATION)

    EPOCHS = cfg.finetune.EPOCHS
    LR     = cfg.finetune.LEARNING_RATE
    SEED   = cfg.data.RANDOM_SEED

    results = {}

    print("\n--- Training: Linear Baseline ---")
    torch.manual_seed(SEED)
    linear = LinearBaseline(INPUT_DIM, cfg.model.RNA_DIM).to(device)
    results["linear"] = train_one_model(
        linear, "linear_baseline", train_loader, val_loader,
        cfg.finetune.BASELINE_EPOCHS, cfg.finetune.LINEAR_LR
    )

    print("\n--- Training: MLP Baseline ---")
    torch.manual_seed(SEED)
    mlp = MLPBaseline(INPUT_DIM, cfg.model.RNA_DIM).to(device)
    results["mlp"] = train_one_model(
        mlp, "mlp_baseline", train_loader, val_loader,
        cfg.finetune.BASELINE_EPOCHS, cfg.finetune.MLP_LR
    )

    print("\n--- Training: Transformer (from scratch) ---")
    torch.manual_seed(SEED)
    transformer_scratch = build_transformer(pretrained_path=None)
    results["transformer_scratch"] = train_one_model(
        transformer_scratch, "transformer_scratch",
        train_loader, val_loader, EPOCHS, LR
    )

    print("\n--- Training: Transformer (pretrained) ---")
    torch.manual_seed(SEED)
    transformer_pretrained = build_transformer("outputs/checkpoints/pretrained_best.pt")
    results["transformer_pretrained"] = train_one_model(
        transformer_pretrained, "transformer_pretrained",
        train_loader, val_loader, EPOCHS, LR
    )

    print("\n" + "=" * 60)
    print(" RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Model':<30} {'Val Loss':>10} {'Pearson':>10} {'R²':>10}")
    print("-" * 62)
    for name, m in results.items():
        print(f"{name:<30} {m['val_loss']:>10.4f} {m['pearson']:>10.4f} {m['r2']:>10.4f}")

    # Save results
    import json
    Path("outputs/results").mkdir(exist_ok=True)
    with open("outputs/results/finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved → outputs/results/finetune_results.json")
    print("Next step: python training/train_diffusion.py")

    return results


if __name__ == "__main__":
    finetune()
