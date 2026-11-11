"""Chapter 6: the sampling algorithm (Algorithm 2 of the DDPM paper)."""
import torch
from .loss import mean_from_eps


def posterior_variance(schedule, t):
    """beta_tilde_t = (1 - ab_{t-1}) / (1 - ab_t) * beta_t, and 0 at t = 1."""
    i = schedule.at(t)
    if t == 1:
        return torch.zeros_like(schedule.betas[i])
    ab_t, ab_prev = schedule.alpha_bars[i], schedule.alpha_bars[i - 1]
    return (1.0 - ab_prev) / (1.0 - ab_t) * schedule.betas[i]


def p_sample_step(model, xt, t, schedule, sigma="beta", generator=None):
    """One reverse step x_t -> x_{t-1}. t is a Python int (1-based)."""
    tt = torch.full((xt.shape[0],), t, dtype=torch.long, device=xt.device)
    mean = mean_from_eps(xt, tt, model(xt, tt), schedule)
    if t == 1:
        return mean                                  # the last step adds no noise
    var = schedule.betas[schedule.at(t)] if sigma == "beta" else posterior_variance(schedule, t)
    noise = torch.randn(xt.shape, dtype=xt.dtype, device=xt.device, generator=generator)
    return mean + var.to(xt.dtype).sqrt() * noise


@torch.no_grad()
def sample(model, shape, schedule, sigma="beta", generator=None, dtype=torch.float32, device="cpu"):
    """Start from pure noise x_T and walk back to x_0."""
    x = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    for t in range(schedule.T, 0, -1):
        x = p_sample_step(model, x, t, schedule, sigma, generator)
    return x


@torch.no_grad()
def sample_trajectory(model, shape, schedule, keep=(1000, 800, 600, 400, 200, 100, 50, 0),
                      sigma="beta", generator=None, dtype=torch.float32, device="cpu"):
    """Like sample, but also returns snapshots of x_t at the step numbers listed in `keep`."""
    x = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    snaps = {schedule.T: x.clone()} if schedule.T in keep else {}
    for t in range(schedule.T, 0, -1):
        x = p_sample_step(model, x, t, schedule, sigma, generator)
        if t - 1 in keep:
            snaps[t - 1] = x.clone()
    return x, snaps
