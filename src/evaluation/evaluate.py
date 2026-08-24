"""
evaluation/evaluate.py

Full evaluation pipeline:
  1. Expression prediction metrics (Pearson, R², MSE) for all models
  2. Cross-tissue generalization test
  3. Diffusion quality evaluation (PCA overlap, correlation similarity, KS test)
  4. Biological validation (tissue markers, cluster structure)
  5. Attention heatmaps for interpretability
  6. All figures saved to outputs/figures/

Run this after all training phases.

Usage:
    python evaluation/evaluate.py
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configs.config     import Config
from src.models.transformer import MultiOmicsTransformer
from src.models.baseline    import MLPBaseline, LinearBaseline
from src.models.diffusion     import ConditionalDenoiser, GaussianDiffusion
from data.dataset       import get_dataloader, MultiOmicsDataset
from src.training.train_diffusion import generate_synthetic_profiles, build_diffusion
from src.utils.helpers      import get_device, load_checkpoint
from src.evaluation.cross_tissue import run_cross_tissue_evaluation
from src.evaluation.visualizations import (
    plot_data_overview, plot_cross_tissue_table,
    plot_baseline_vs_transformer, plot_gene_correlation_validation,
    plot_tissue_markers, generate_dashboard, generate_manuscript_draft,
)

cfg    = Config()
device = get_device()
FIGS   = Path(cfg.eval.FIGURE_DIR)
FIGS.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
TISSUE_COLORS = ["#2E86AB", "#A23B72", "#F18F01"]  # blue, purple, orange
TISSUE_NAMES  = cfg.data.TISSUES


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute all regression metrics."""
    # Pearson: per gene, then average
    pearsons = []
    for i in range(pred.shape[1]):
        r, _ = stats.pearsonr(pred[:, i], true[:, i])
        if not np.isnan(r):
            pearsons.append(r)

    mse   = float(np.mean((pred - true) ** 2))
    r2    = float(r2_score(true.flatten(), pred.flatten()))
    pearson = float(np.mean(pearsons))

    return {"pearson": pearson, "r2": r2, "mse": mse}


def evaluate_expression_models() -> dict:
    """Load all trained models and evaluate on test set."""
    print("\n[1] Evaluating expression prediction models...")

    test_loader = get_dataloader("test", batch_size=64,
                                 use_methylation=cfg.data.USE_METHYLATION)

    results = {}

    models_to_eval = [
        ("linear_baseline",     "linear"),
        ("mlp_baseline",        "mlp"),
        ("transformer_scratch", "transformer_scratch"),
        ("transformer_pretrained", "transformer_pretrained"),
    ]

    INPUT_DIM = cfg.model.INPUT_DIM

    for ckpt_name, display_name in models_to_eval:
        ckpt_path = Path(f"outputs/checkpoints/{ckpt_name}_best.pt")
        if not ckpt_path.exists():
            print(f"  Skipping {display_name} (checkpoint not found)")
            continue

        # Load appropriate model
        if "transformer" in ckpt_name:
            model = MultiOmicsTransformer(
                input_dim=INPUT_DIM, atac_dim=cfg.model.ATAC_DIM,
                meth_dim=cfg.model.METH_DIM if cfg.data.USE_METHYLATION else 0,
                output_dim=cfg.model.RNA_DIM,
                hidden_dim=cfg.model.HIDDEN_DIM, num_heads=cfg.model.NUM_HEADS,
                num_layers=cfg.model.NUM_LAYERS, ff_dim=cfg.model.FF_DIM,
            ).to(device)
        elif "mlp" in ckpt_name:
            model = MLPBaseline(INPUT_DIM, cfg.model.RNA_DIM).to(device)
        else:
            model = LinearBaseline(INPUT_DIM, cfg.model.RNA_DIM).to(device)

        load_checkpoint(model, str(ckpt_path))
        model.eval()

        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                x   = batch["x"].to(device)
                rna = batch["rna"]

                if isinstance(model, MultiOmicsTransformer):
                    out  = model(x, pretrain=False)
                    pred = out["pred"].cpu()
                else:
                    pred = model(x).cpu()

                all_pred.append(pred.numpy())
                all_true.append(rna.numpy())

        pred_arr = np.concatenate(all_pred)
        true_arr = np.concatenate(all_true)
        metrics  = compute_metrics(pred_arr, true_arr)
        results[display_name] = metrics

        print(f"  {display_name:<28} "
              f"Pearson: {metrics['pearson']:.4f} | "
              f"R²: {metrics['r2']:.4f} | "
              f"MSE: {metrics['mse']:.4f}")

    return results


def plot_metrics_comparison(results: dict):
    """Bar chart comparing models on key metrics."""
    if not results:
        return

    models  = list(results.keys())
    pearsons = [results[m]["pearson"] for m in models]
    r2s      = [results[m]["r2"]      for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#95a5a6", "#7f8c8d", "#3498db", "#e74c3c"][:len(models)]

    for ax, values, metric_name in zip(axes, [pearsons, r2s], ["Pearson r", "R²"]):
        bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(f"{metric_name} — Test Set", fontsize=13, fontweight="bold")
        ax.set_ylabel(metric_name)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylim(0, max(values) * 1.15)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGS / "1_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → outputs/figures/1_model_comparison.png")


def evaluate_cross_tissue() -> dict:
    """
    Train on seen tissues → evaluate on unseen tissue.
    This tests if pretrained model generalizes better.
    """
    print("\n[2] Cross-tissue generalization test...")
    test_loader = get_dataloader("test", batch_size=64,
                                 use_methylation=cfg.data.USE_METHYLATION)

    results = {}
    for ckpt_name in ["transformer_scratch", "transformer_pretrained"]:
        ckpt_path = Path(f"outputs/checkpoints/{ckpt_name}_best.pt")
        if not ckpt_path.exists():
            continue

        model = MultiOmicsTransformer(
            input_dim=cfg.model.INPUT_DIM, atac_dim=cfg.model.ATAC_DIM,
            meth_dim=cfg.model.METH_DIM if cfg.data.USE_METHYLATION else 0,
            output_dim=cfg.model.RNA_DIM,
            hidden_dim=cfg.model.HIDDEN_DIM, num_heads=cfg.model.NUM_HEADS,
            num_layers=cfg.model.NUM_LAYERS, ff_dim=cfg.model.FF_DIM,
        ).to(device)
        load_checkpoint(model, str(ckpt_path))
        model.eval()

        # Evaluate by tissue
        tissue_metrics = {}
        with torch.no_grad():
            for batch in test_loader:
                x      = batch["x"].to(device)
                rna    = batch["rna"].numpy()
                labels = batch["label"].numpy()

                out  = model(x, pretrain=False)
                pred = out["pred"].cpu().numpy()

                for t_idx, t_name in enumerate(TISSUE_NAMES):
                    mask = labels == t_idx
                    if mask.sum() == 0:
                        continue
                    m = compute_metrics(pred[mask], rna[mask])
                    if t_name not in tissue_metrics:
                        tissue_metrics[t_name] = {"pearson": [], "r2": []}
                    tissue_metrics[t_name]["pearson"].append(m["pearson"])
                    tissue_metrics[t_name]["r2"].append(m["r2"])

        # Average across batches
        avg = {t: {k: np.mean(v) for k, v in mets.items()}
               for t, mets in tissue_metrics.items()}
        results[ckpt_name] = avg
        print(f"  {ckpt_name}:")
        for t, m in avg.items():
            print(f"    {t:<10} Pearson: {m['pearson']:.4f} | R²: {m['r2']:.4f}")

    return results


def plot_cross_tissue(results: dict):
    """Plot cross-tissue performance heatmap."""
    if not results:
        return

    models  = list(results.keys())
    tissues = TISSUE_NAMES
    data    = np.zeros((len(models), len(tissues)))

    for i, m in enumerate(models):
        for j, t in enumerate(tissues):
            data[i, j] = results[m].get(t, {}).get("pearson", 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=tissues, yticklabels=models,
                vmin=0, vmax=1, ax=ax)
    ax.set_title("Cross-Tissue Pearson Correlation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS / "2_cross_tissue_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → outputs/figures/2_cross_tissue_heatmap.png")


def evaluate_diffusion():
    """Evaluate quality of diffusion-generated multi-omics profiles."""
    print("\n[3] Evaluating diffusion generative quality...")

    diff_path = Path("outputs/checkpoints/diffusion_best.pt")
    if not diff_path.exists():
        print("  Diffusion checkpoint not found, skipping.")
        return

    model = build_diffusion()
    model.load_state_dict(torch.load(diff_path, map_location=device, weights_only=False))
    model.eval()

    dataset  = MultiOmicsDataset("test", use_methylation=cfg.data.USE_METHYLATION)
    real_all = dataset.atac.numpy()
    labels   = dataset.labels.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for t_idx, (tissue, ax) in enumerate(zip(TISSUE_NAMES, axes)):
        real_t = real_all[labels == t_idx]
        if len(real_t) == 0:
            continue

        fake_t = generate_synthetic_profiles(model, len(real_t), t_idx, fast=True)
        fake_t = fake_t[:, :cfg.model.ATAC_DIM]

        pca = PCA(n_components=2)
        combined = np.vstack([real_t[:, :500], fake_t[:, :500]])
        pca.fit(combined)

        real_pca = pca.transform(real_t[:, :500])
        fake_pca = pca.transform(fake_t[:, :500])

        ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5,
                   label="Real", color=TISSUE_COLORS[t_idx], s=20)
        ax.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.5,
                   label="Synthetic (diffusion)", color="gray", s=20, marker="x")
        ax.set_title(f"{tissue.capitalize()} — PCA", fontsize=11, fontweight="bold")
        ax.legend()

        ks_stats = []
        for feat in range(min(50, real_t.shape[1])):
            ks, _ = stats.ks_2samp(real_t[:, feat], fake_t[:, feat])
            ks_stats.append(ks)

        print(f"  {tissue}: KS stat (mean over 50 peaks) = {np.mean(ks_stats):.4f} "
              f"(lower = more similar to real)")

    plt.suptitle("PCA: Real vs Diffusion-Generated Multi-Omics Profiles", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS / "3_diffusion_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> outputs/figures/3_diffusion_pca.png")


def plot_attention_heatmap():
    """Visualize attention weights from the transformer (interpretability)."""
    print("\n[4] Extracting attention weights...")

    ckpt = Path("outputs/checkpoints/transformer_pretrained_best.pt")
    if not ckpt.exists():
        print("  Checkpoint not found, skipping.")
        return

    model = MultiOmicsTransformer(
        input_dim=cfg.model.INPUT_DIM, atac_dim=cfg.model.ATAC_DIM,
        meth_dim=cfg.model.METH_DIM if cfg.data.USE_METHYLATION else 0,
        output_dim=cfg.model.RNA_DIM,
        hidden_dim=cfg.model.HIDDEN_DIM, num_heads=cfg.model.NUM_HEADS,
        num_layers=cfg.model.NUM_LAYERS, ff_dim=cfg.model.FF_DIM,
    ).to(device)
    load_checkpoint(model, str(ckpt))
    model.eval()

    # Use one small batch
    dataset = MultiOmicsDataset("test", use_methylation=cfg.data.USE_METHYLATION)
    x = dataset[0]["x"].unsqueeze(0).to(device)  # (1, input_dim)

    try:
        attn_weights = model.get_attention_weights(x)  # list of (1, L, L)

        fig, axes = plt.subplots(1, len(attn_weights), figsize=(6 * len(attn_weights), 5))
        if len(attn_weights) == 1:
            axes = [axes]

        n_show = 100  # show first 100 x 100 positions
        for layer_idx, (attn, ax) in enumerate(zip(attn_weights, axes)):
            attn_np = attn[0, :n_show, :n_show].numpy()
            im = ax.imshow(attn_np, cmap="hot", aspect="auto")
            ax.set_title(f"Layer {layer_idx+1}", fontweight="bold")
            ax.set_xlabel("Key position (genomic region)")
            ax.set_ylabel("Query position")
            plt.colorbar(im, ax=ax)

        plt.suptitle("Transformer Attention Weights\n"
                     "(High values = regulatory influence between genomic regions)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIGS / "4_attention_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  → outputs/figures/4_attention_heatmap.png")
    except Exception as e:
        print(f"  Attention visualization failed: {e}")


def plot_loss_curves():
    """Plot training loss curves from logger CSVs."""
    print("\n[5] Plotting loss curves...")
    import os

    log_dir = Path("outputs/logs")
    if not log_dir.exists():
        print("  No logs found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    for i, log_file in enumerate(sorted(log_dir.glob("*.csv"))):
        import pandas as pd
        df = pd.read_csv(log_file)
        if "val_loss" in df.columns:
            label = log_file.stem
            ax.plot(df["epoch"], df["val_loss"],
                    label=label, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Training Loss Curves", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGS / "5_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → outputs/figures/5_loss_curves.png")


def run_evaluation():
    print("\n" + "=" * 60)
    print(" FULL EVALUATION PIPELINE")
    print("=" * 60)

    # 0. Overview figures
    print("\n[0] Generating data figures...")
    plot_data_overview()

    # 1. Expression prediction
    results = evaluate_expression_models()
    plot_metrics_comparison(results)
    plot_baseline_vs_transformer(results)

    # 2. Cross-tissue generalization (in-distribution by tissue)
    cross = evaluate_cross_tissue()
    plot_cross_tissue(cross)

    # 2b. Cross-tissue transfer (train A -> test B)
    transfer_path = Path("outputs/results/cross_tissue_transfer.json")
    if transfer_path.exists():
        with open(transfer_path) as f:
            cross_transfer = json.load(f)
        print("  Loaded existing cross-tissue transfer results")
    else:
        cross_transfer = run_cross_tissue_evaluation(epochs=cfg.finetune.EPOCHS)
    plot_cross_tissue_table(cross, cross_transfer)

    # 3. Diffusion quality
    evaluate_diffusion()

    # 4. Attention interpretability
    plot_attention_heatmap()

    # 5. Loss curves
    plot_loss_curves()

    # 6. Biological validation
    print("\n[6] Biological validation...")
    test_ds = MultiOmicsDataset("test", use_methylation=cfg.data.USE_METHYLATION)
    plot_tissue_markers(test_ds.rna.numpy(), test_ds.labels.numpy())

    ckpt = Path("outputs/checkpoints/transformer_pretrained_best.pt")
    if ckpt.exists():
        model = MultiOmicsTransformer(
            input_dim=cfg.model.INPUT_DIM, atac_dim=cfg.model.ATAC_DIM,
            meth_dim=cfg.model.METH_DIM if cfg.data.USE_METHYLATION else 0,
            output_dim=cfg.model.RNA_DIM,
            hidden_dim=cfg.model.HIDDEN_DIM, num_heads=cfg.model.NUM_HEADS,
            num_layers=cfg.model.NUM_LAYERS, ff_dim=cfg.model.FF_DIM,
        ).to(device)
        load_checkpoint(model, str(ckpt))
        model.eval()
        all_pred, all_true = [], []
        loader = get_dataloader("test", batch_size=64,
                                use_methylation=cfg.data.USE_METHYLATION)
        with torch.no_grad():
            for batch in loader:
                out = model(batch["x"].to(device), pretrain=False)
                all_pred.append(out["pred"].cpu().numpy())
                all_true.append(batch["rna"].numpy())
        corr_r = plot_gene_correlation_validation(
            np.concatenate(all_pred), np.concatenate(all_true)
        )
    else:
        corr_r = None

    # Save all results
    Path("outputs/results").mkdir(exist_ok=True)
    full_results = {
        "expression": results,
        "cross_tissue": {
            k: {t: {m: float(v) for m, v in metrics.items()}
                for t, metrics in tissue_dict.items()}
            for k, tissue_dict in cross.items()
        },
        "cross_tissue_transfer": cross_transfer,
        "gene_correlation_preservation": corr_r,
    }
    with open("outputs/results/full_evaluation.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # 7. Research deliverables
    print("\n[7] Generating research deliverables...")
    generate_dashboard("outputs/results/full_evaluation.json")
    generate_manuscript_draft("outputs/results/full_evaluation.json")

    print("\n" + "=" * 60)
    print(" Evaluation complete.")
    print(f"  Figures:    {FIGS}/")
    print(f"  Dashboard:  {FIGS}/research_dashboard.html")
    print(f"  Manuscript: outputs/results/manuscript_draft.md")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
