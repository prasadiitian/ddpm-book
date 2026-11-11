"""Forward process and posterior for DDPM. All formulas follow the paper."""
import torch


def q_step(x_prev, beta_t, generator=None):
    """One forward step: x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * noise."""
    noise = torch.randn(x_prev.shape, dtype=x_prev.dtype, generator=generator)
    return (1.0 - beta_t) ** 0.5 * x_prev + beta_t ** 0.5 * noise


def q_sample(x0, t, schedule, noise=None):
    """Closed form: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise.

    t is 1-based (1..T) and may be a Python int.
    """
    if noise is None:
        noise = torch.randn_like(x0)
    ab = schedule.alpha_bars[schedule.at(t)]
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def posterior_mean_var(x0, xt, t, schedule):
    """Mean and variance of q(x_{t-1} | x_t, x_0) using the paper's formulas.

    mean = sqrt(ab_{t-1}) * beta_t / (1 - ab_t) * x0
         + sqrt(alpha_t) * (1 - ab_{t-1}) / (1 - ab_t) * xt
    var  = (1 - ab_{t-1}) / (1 - ab_t) * beta_t
    For t = 1, alpha_bar_0 is defined as 1.
    """
    i = schedule.at(t)
    beta_t = schedule.betas[i]
    alpha_t = schedule.alphas[i]
    ab_t = schedule.alpha_bars[i]
    ab_prev = schedule.alpha_bars[i - 1] if t > 1 else torch.ones_like(ab_t)
    coef_x0 = ab_prev.sqrt() * beta_t / (1.0 - ab_t)
    coef_xt = alpha_t.sqrt() * (1.0 - ab_prev) / (1.0 - ab_t)
    mean = coef_x0 * x0 + coef_xt * xt
    var = (1.0 - ab_prev) / (1.0 - ab_t) * beta_t
    return mean, var
