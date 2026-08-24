"""
evaluation/cross_tissue.py

Trains a fresh model on one tissue and tests it on another, unseen tissue
(cross-tissue transfer), separate from the per-tissue generalization test
in evaluate.py.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configs.config import Config
from src.models.baseline import MLPBaseline
from src.models.transformer import MultiOmicsTransformer
from data.dataset import MultiOmicsDataset
from src.utils.helpers import get_device, save_checkpoint, load_checkpoint

cfg = Config()
device = get_device()


def _pearson(pred, true):
    corrs = []
    for i in range(pred.shape[1]):
        r = np.corrcoef(pred[:, i], true[:, i])[0, 1]
        if not np.isnan(r):
            corrs.append(r)
    return float(np.mean(corrs))


def _r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))


def _train_model(model, train_loader, val_loader, epochs=5, lr=5e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            rna = batch["rna"].to(device)
            optimizer.zero_grad()
            if isinstance(model, MultiOmicsTransformer):
                pred = model(x, pretrain=False)["pred"]
            else:
                pred = model(x)
            loss = criterion(pred, rna)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                rna = batch["rna"].to(device)
                if isinstance(model, MultiOmicsTransformer):
                    pred = model(x, pretrain=False)["pred"]
                else:
                    pred = model(x)
                val_losses.append(criterion(pred, rna).item())

        val_loss = np.mean(val_losses)
        if val_loss < best_loss:
            best_loss = val_loss

    return model


def _evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            rna = batch["rna"]
            if isinstance(model, MultiOmicsTransformer):
                pred = model(x, pretrain=False)["pred"].cpu().numpy()
            else:
                pred = model(x).cpu().numpy()
            all_pred.append(pred)
            all_true.append(rna.numpy())
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    return {"pearson": _pearson(pred, true), "r2": _r2(pred, true),
            "mse": float(np.mean((pred - true) ** 2))}


def run_cross_tissue_evaluation(epochs: int = 5):
    """
    Train MLP and Transformer on liver → test on brain and lung.
    Also train on liver+brain → test on lung (held-out tissue).
    """
    print("\n[Cross-Tissue] Running transfer experiments...")

    tissue_map = {name: i for i, name in enumerate(cfg.data.TISSUES)}
    train_ds = MultiOmicsDataset("train", use_methylation=cfg.data.USE_METHYLATION)
    val_ds = MultiOmicsDataset("val", use_methylation=cfg.data.USE_METHYLATION)
    test_ds = MultiOmicsDataset("test", use_methylation=cfg.data.USE_METHYLATION)

    def tissue_indices(dataset, tissue_name):
        idx = tissue_map[tissue_name]
        return [i for i in range(len(dataset)) if dataset.labels[i].item() == idx]

    results = {}
    INPUT_DIM = cfg.model.INPUT_DIM

    experiments = [
        ("MLP", "liver", "brain"),
        ("MLP", "liver", "lung"),
        ("Transformer", "liver", "brain"),
        ("Transformer", "liver", "lung"),
    ]

    for model_type, train_tissue, test_tissue in experiments:
        train_idx = tissue_indices(train_ds, train_tissue)
        val_idx = tissue_indices(val_ds, train_tissue)
        test_idx = tissue_indices(test_ds, test_tissue)

        if not train_idx or not test_idx:
            continue

        train_loader = DataLoader(
            Subset(train_ds, train_idx), batch_size=8, shuffle=True
        )
        val_loader = DataLoader(
            Subset(val_ds, val_idx) if val_idx else Subset(val_ds, [0]),
            batch_size=8,
        )
        test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=8)

        if model_type == "MLP":
            model = MLPBaseline(INPUT_DIM, cfg.model.RNA_DIM).to(device)
        else:
            model = MultiOmicsTransformer(
                input_dim=INPUT_DIM, atac_dim=cfg.model.ATAC_DIM,
                meth_dim=cfg.model.METH_DIM if cfg.data.USE_METHYLATION else 0,
                output_dim=cfg.model.RNA_DIM,
                hidden_dim=cfg.model.HIDDEN_DIM, num_heads=cfg.model.NUM_HEADS,
                num_layers=cfg.model.NUM_LAYERS, ff_dim=cfg.model.FF_DIM,
            ).to(device)
            ckpt = Path("outputs/checkpoints/pretrained_best.pt")
            if ckpt.exists():
                load_checkpoint(model, str(ckpt))

        print(f"  Training {model_type}: {train_tissue} → {test_tissue}...")
        model = _train_model(model, train_loader, val_loader, epochs=epochs)
        metrics = _evaluate(model, test_loader)

        key = f"{model_type}_{train_tissue}_to_{test_tissue}"
        results[key] = metrics
        print(f"    {key}: Pearson={metrics['pearson']:.4f}, R²={metrics['r2']:.4f}")

    Path("outputs/results").mkdir(exist_ok=True)
    with open("outputs/results/cross_tissue_transfer.json", "w") as f:
        json.dump(results, f, indent=2)

    print("  → outputs/results/cross_tissue_transfer.json")
    return results
