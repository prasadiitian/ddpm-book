"""Noise schedules for DDPM (Ho et al., 2020)."""
import torch


def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02, dtype=torch.float64):
    """The linear schedule from the paper: beta_1 = 1e-4 up to beta_T = 0.02."""
    return torch.linspace(beta_start, beta_end, T, dtype=dtype)


class Schedule:
    """Holds beta_t, alpha_t and alpha_bar_t for t = 1..T (stored at index t-1)."""

    def __init__(self, betas):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.T = len(betas)

    def at(self, t):
        """Index helper: t is 1-based like the paper."""
        return t - 1


def cosine_beta_schedule(T=1000, s=0.008, max_beta=0.999, dtype=torch.float64):
    """Cosine schedule of Nichol & Dhariwal (2021), not part of the original DDPM paper.

    alpha_bar(t) = f(t) / f(0),  f(t) = cos^2(((t/T + s) / (1 + s)) * pi / 2)
    beta_t = 1 - alpha_bar(t) / alpha_bar(t-1), clipped at max_beta.
    """
    steps = torch.arange(T + 1, dtype=dtype)
    f = torch.cos(((steps / T + s) / (1.0 + s)) * torch.pi / 2) ** 2
    ab = f / f[0]
    betas = 1.0 - ab[1:] / ab[:-1]
    return betas.clamp(max=max_beta)
