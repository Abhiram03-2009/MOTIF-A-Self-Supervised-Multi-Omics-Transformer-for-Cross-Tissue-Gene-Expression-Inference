"""
models/diffusion.py

Conditional DDPM for synthetic multi-omics profile generation.

Replaces WGAN-GP with a diffusion model that:
  - Learns to denoise tissue-specific epigenomic profiles
  - Conditions on tissue type (liver, brain, lung)
  - Generates realistic ATAC + methylation profiles via reverse diffusion

Why diffusion over GAN for biological data:
  - More stable training, less mode collapse
  - Better coverage of the data distribution
  - Interpretable generation process (gradual denoising)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Embed diffusion timestep t into a dense vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class ConditionalDenoiser(nn.Module):
    """
    Predicts noise epsilon given noisy profile x_t, timestep t, tissue label.

    Architecture: tissue embedding + time embedding + MLP over profile.
    """

    def __init__(self, profile_dim: int, n_tissues: int,
                 hidden_dim: int = 256, time_dim: int = 64,
                 label_dim: int = 16):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.label_embed = nn.Embedding(n_tissues, label_dim)

        in_dim = profile_dim + time_dim + label_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, profile_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        x:      (batch, profile_dim) noisy profile
        t:      (batch,) integer timesteps
        labels: (batch,) tissue indices
        -> predicted noise (batch, profile_dim)
        """
        t_emb = self.time_embed(t)
        l_emb = self.label_embed(labels)
        h = torch.cat([x, t_emb, l_emb], dim=-1)
        return self.net(h)


class GaussianDiffusion(nn.Module):
    """
    DDPM wrapper: training loss + sampling.

    Args:
        denoiser:     ConditionalDenoiser instance
        profile_dim:  dimension of multi-omics profile
        timesteps:    number of diffusion steps (T)
        beta_start:   noise schedule start
        beta_end:     noise schedule end
    """

    def __init__(self, denoiser: ConditionalDenoiser, profile_dim: int,
                 timesteps: int = 200, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.denoiser = denoiser
        self.profile_dim = profile_dim
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1 - alpha_bar.roll(1, 0)) / (1 - alpha_bar),
        )
        self.posterior_variance[0] = betas[0]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_1m_ab * noise

    def training_loss(self, x0: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """Sample random t, predict noise, return MSE loss."""
        batch = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.denoiser(x_t, t, labels)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor,
               n_samples: int = None) -> torch.Tensor:
        """
        Reverse diffusion: generate synthetic profiles.

        labels: (n_samples,) tissue indices
        -> (n_samples, profile_dim)
        """
        if n_samples is None:
            n_samples = labels.shape[0]
        device = labels.device

        x = torch.randn(n_samples, self.profile_dim, device=device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), step, device=device, dtype=torch.long)
            pred_noise = self.denoiser(x, t, labels)

            alpha = self.alphas[step]
            alpha_bar = self.alpha_bar[step]
            beta = self.betas[step]

            if step > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            coef1 = 1.0 / alpha.sqrt()
            coef2 = beta / (1 - alpha_bar).sqrt()
            x = coef1 * (x - coef2 * pred_noise) + self.posterior_variance[step].sqrt() * noise

        return x

    @torch.no_grad()
    def sample_fast(self, labels: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Accelerated sampling via stride (for interactive platform use)."""
        n_samples = labels.shape[0]
        device = labels.device
        stride = max(1, self.timesteps // steps)
        timesteps = list(range(0, self.timesteps, stride))[::-1]

        x = torch.randn(n_samples, self.profile_dim, device=device)
        for i, step in enumerate(timesteps):
            t = torch.full((n_samples,), step, device=device, dtype=torch.long)
            pred_noise = self.denoiser(x, t, labels)
            alpha_bar = self.alpha_bar[step]
            if i < len(timesteps) - 1:
                prev_step = timesteps[i + 1]
                prev_alpha_bar = self.alpha_bar[prev_step]
                x = (
                    prev_alpha_bar.sqrt() * (x - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
                    + (1 - prev_alpha_bar).sqrt() * pred_noise
                )
            else:
                x = (x - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        return x
