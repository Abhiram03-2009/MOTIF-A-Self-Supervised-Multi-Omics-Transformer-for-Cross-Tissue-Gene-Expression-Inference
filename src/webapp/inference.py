"""
webapp/inference.py

Backend service layer for the Multi-Omics clinical platform.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional

import sys
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.configs.config import Config
from src.models.transformer import MultiOmicsTransformer
from src.models.baseline import MLPBaseline, LinearBaseline
from src.models.diffusion import ConditionalDenoiser, GaussianDiffusion
from data.dataset import MultiOmicsDataset

cfg = Config()


class MultiOmicsPlatform:
    """Unified inference API for clinicians and biologists."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tissues = cfg.data.TISSUES
        self._transformer = None
        self._linear = None
        self._mlp = None
        self._diffusion = None
        self._results = None

    @property
    def results(self) -> dict:
        if self._results is None:
            path = ROOT / "outputs" / "results" / "full_evaluation.json"
            if path.exists():
                with open(path) as f:
                    self._results = json.load(f)
            else:
                self._results = {}
        return self._results

    def model_status(self) -> dict:
        ckpt = ROOT / "outputs" / "checkpoints"
        return {
            "transformer": (ckpt / "transformer_pretrained_best.pt").exists(),
            "linear": (ckpt / "linear_baseline_best.pt").exists(),
            "mlp": (ckpt / "mlp_baseline_best.pt").exists(),
            "diffusion": (ckpt / "diffusion_best.pt").exists(),
            "data": (ROOT / "data" / "processed" / "train" / "rna.npy").exists(),
        }

    def _load_transformer(self) -> MultiOmicsTransformer:
        if self._transformer is None:
            model = MultiOmicsTransformer(
                input_dim=cfg.model.INPUT_DIM, atac_dim=cfg.model.ATAC_DIM,
                meth_dim=cfg.model.METH_DIM if cfg.data.USE_METHYLATION else 0,
                output_dim=cfg.model.RNA_DIM,
                hidden_dim=cfg.model.HIDDEN_DIM, num_heads=cfg.model.NUM_HEADS,
                num_layers=cfg.model.NUM_LAYERS, ff_dim=cfg.model.FF_DIM,
            ).to(self.device)
            ckpt = ROOT / "outputs" / "checkpoints" / "transformer_pretrained_best.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device, weights_only=False)
                model.load_state_dict(state["model"])
            self._transformer = model
        self._transformer.eval()
        return self._transformer

    def _load_diffusion(self) -> GaussianDiffusion:
        if self._diffusion is None:
            denoiser = ConditionalDenoiser(
                profile_dim=cfg.model.INPUT_DIM,
                n_tissues=len(cfg.data.TISSUES),
                hidden_dim=cfg.diffusion.HIDDEN_DIM,
                time_dim=cfg.diffusion.TIME_DIM,
                label_dim=cfg.diffusion.LABEL_DIM,
            )
            model = GaussianDiffusion(
                denoiser=denoiser,
                profile_dim=cfg.model.INPUT_DIM,
                timesteps=cfg.diffusion.TIMESTEPS,
            ).to(self.device)
            ckpt = ROOT / "outputs" / "checkpoints" / "diffusion_best.pt"
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=False))
            self._diffusion = model
        self._diffusion.eval()
        return self._diffusion

    def predict_expression(self, atac: np.ndarray, meth: np.ndarray,
                           model_name: str = "transformer") -> np.ndarray:
        x = np.concatenate([atac, meth], axis=-1).astype(np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        ckpt_name = {"transformer": "transformer_pretrained_best.pt",
                     "mlp": "mlp_baseline_best.pt",
                     "linear": "linear_baseline_best.pt"}.get(model_name, "linear_baseline_best.pt")
        ckpt_path = ROOT / "outputs" / "checkpoints" / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No trained checkpoint at {ckpt_path} — refusing to predict from an "
                f"untrained model. Run the '{model_name}' training stage first."
            )

        if model_name == "transformer":
            model = self._load_transformer()
            with torch.no_grad():
                t = torch.FloatTensor(x).to(self.device)
                return model(t, pretrain=False)["pred"].cpu().numpy()
        elif model_name == "mlp":
            if self._mlp is None:
                self._mlp = MLPBaseline(cfg.model.INPUT_DIM, cfg.model.RNA_DIM).to(self.device)
                self._mlp.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False)["model"])
            with torch.no_grad():
                return self._mlp(torch.FloatTensor(x).to(self.device)).cpu().numpy()
        else:
            if self._linear is None:
                self._linear = LinearBaseline(cfg.model.INPUT_DIM, cfg.model.RNA_DIM).to(self.device)
                self._linear.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False)["model"])
            with torch.no_grad():
                return self._linear(torch.FloatTensor(x).to(self.device)).cpu().numpy()

    def generate_synthetic(self, tissue: str, n_samples: int = 10,
                           fast: bool = True) -> np.ndarray:
        t_idx = self.tissues.index(tissue)
        model = self._load_diffusion()
        labels = torch.full((n_samples,), t_idx, dtype=torch.long, device=self.device)
        with torch.no_grad():
            if fast:
                samples = model.sample_fast(labels, steps=cfg.diffusion.SAMPLE_STEPS)
            else:
                samples = model.sample(labels)
        return samples.cpu().numpy()

    def load_dataset(self, split: str = "test"):
        return MultiOmicsDataset(split, use_methylation=cfg.data.USE_METHYLATION)

    def get_demo_profile(self, tissue: str = "liver") -> tuple:
        ds = self.load_dataset("test")
        t_idx = self.tissues.index(tissue)
        for i in range(len(ds)):
            if ds.labels[i].item() == t_idx:
                atac = ds.atac[i].numpy()
                meth = ds.meth[i].numpy() if cfg.data.USE_METHYLATION else np.zeros(cfg.model.ATAC_DIM)
                rna = ds.rna[i].numpy()
                return atac, meth, rna
        return ds.atac[0].numpy(), ds.meth[0].numpy(), ds.rna[0].numpy()

    def cross_tissue_results(self) -> Optional[dict]:
        path = ROOT / "outputs" / "results" / "cross_tissue_transfer.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None


_platform: Optional[MultiOmicsPlatform] = None


def get_platform() -> MultiOmicsPlatform:
    global _platform
    if _platform is None:
        _platform = MultiOmicsPlatform()
    return _platform
