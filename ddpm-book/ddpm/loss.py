"""Chapter 4: noise-prediction parameterisation and the simple loss."""
import torch
import torch.nn.functional as F


def _gather(values, t, x):
    """values[t-1] for a batch of 1-based steps t, reshaped to broadcast against x."""
    out = values[t - 1].to(x.dtype)
    return out.view(-1, *([1] * (x.dim() - 1)))


def x0_from_eps(xt, t, eps, schedule):
    """Invert the closed form: x0 = (x_t - sqrt(1-ab) * eps) / sqrt(ab)."""
    ab = _gather(schedule.alpha_bars, t, xt)
    return (xt - (1.0 - ab).sqrt() * eps) / ab.sqrt()


def mean_from_eps(xt, t, eps, schedule):
    """Reverse-step mean in the epsilon form: (x_t - beta/sqrt(1-ab) * eps) / sqrt(alpha)."""
    beta = _gather(schedule.betas, t, xt)
    alpha = _gather(schedule.alphas, t, xt)
    ab = _gather(schedule.alpha_bars, t, xt)
    return (xt - beta / (1.0 - ab).sqrt() * eps) / alpha.sqrt()


def simple_loss(model, x0, schedule, t=None, noise=None):
    """L_simple = E || eps - eps_theta(sqrt(ab) x0 + sqrt(1-ab) eps, t) ||^2."""
    b = x0.shape[0]
    if t is None:
        t = torch.randint(1, schedule.T + 1, (b,), device=x0.device)
    if noise is None:
        noise = torch.randn_like(x0)
    ab = _gather(schedule.alpha_bars, t, x0)
    xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
    return F.mse_loss(model(xt, t), noise)
