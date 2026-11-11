"""Tier 1 tests for Chapter 4: epsilon parameterisation and the simple loss."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from torch.distributions import Normal, kl_divergence
from ddpm import (Schedule, linear_beta_schedule, q_sample, posterior_mean_var,
                  x0_from_eps, mean_from_eps, simple_loss)

torch.manual_seed(0)
S = Schedule(linear_beta_schedule())


def test_x0_from_eps_inverts_q_sample():
    x0 = torch.randn(64, 3, dtype=torch.float64)
    eps = torch.randn_like(x0)
    worst = 0.0
    for t in (1, 50, 500, 1000):
        xt = q_sample(x0, t, S, noise=eps)
        tt = torch.full((64,), t)
        worst = max(worst, (x0_from_eps(xt, tt, eps, S) - x0).abs().max().item())
    print("x0 recovery error:", worst)
    assert worst < 1e-9


def test_epsilon_mean_equals_posterior_mean():
    """Plugging the true eps into the epsilon-form mean must give the Chapter 3 posterior mean."""
    x0 = torch.randn(16, dtype=torch.float64)
    eps = torch.randn_like(x0)
    worst = 0.0
    for t in (2, 10, 100, 500, 1000):
        xt = q_sample(x0, t, S, noise=eps)
        mu_post, _ = posterior_mean_var(x0, xt, t, S)
        mu_eps = mean_from_eps(xt, torch.full((16,), t), eps, S)
        worst = max(worst, (mu_post - mu_eps).abs().max().item())
    print("largest mean difference (posterior vs eps form):", worst)
    assert worst < 1e-10


def test_kl_equals_weighted_eps_mse_up_to_constant():
    """KL(q || p_theta) - ||mu_q - mu_p||^2 / (2 sigma^2) must not depend on mu_p,
    and ||mu_q - mu_p||^2 must equal beta^2 / (alpha (1-ab)) * ||eps - eps_theta||^2."""
    x0, eps = torch.tensor(0.7, dtype=torch.float64), torch.tensor(-0.3, dtype=torch.float64)
    for t in (2, 100, 700):
        i = t - 1
        beta, alpha, ab = S.betas[i], S.alphas[i], S.alpha_bars[i]
        xt = q_sample(x0, t, S, noise=eps)
        mu_q, var_q = posterior_mean_var(x0, xt, t, S)
        sigma2 = beta                                    # one of the paper's two choices
        offsets, ratios = [], []
        for eps_theta in (torch.tensor(0.5, dtype=torch.float64), torch.tensor(-1.2, dtype=torch.float64)):
            mu_p = mean_from_eps(xt, torch.tensor([t]), eps_theta, S)
            kl = kl_divergence(Normal(mu_q, var_q.sqrt()), Normal(mu_p, sigma2.sqrt()))
            quad = (mu_q - mu_p) ** 2 / (2 * sigma2)
            offsets.append((kl - quad).item())
            weight = beta ** 2 / (2 * sigma2 * alpha * (1 - ab))
            ratios.append((quad / (weight * (eps - eps_theta) ** 2)).item())
        assert abs(offsets[0] - offsets[1]) < 1e-9, (t, offsets)
        assert all(abs(r - 1.0) < 1e-9 for r in ratios), (t, ratios)


def test_loss_values_for_known_predictors():
    x0 = torch.randn(256, 1, 4, 4)
    t = torch.randint(1, 1001, (256,))
    eps = torch.randn_like(x0)
    # A model that returns the true noise has zero loss.
    oracle = lambda xt, tt: eps
    assert simple_loss(oracle, x0, S, t=t, noise=eps).item() < 1e-12
    # A model that always predicts zero has loss E[eps^2], which is about 1.
    zero = lambda xt, tt: torch.zeros_like(xt)
    val = simple_loss(zero, x0, S, t=t, noise=eps).item()
    print("loss of the zero predictor:", val)
    assert abs(val - 1.0) < 0.05


def test_loss_has_gradient():
    lin = torch.nn.Linear(4, 4)
    model = lambda xt, tt: lin(xt)
    loss = simple_loss(model, torch.randn(32, 4), S)
    loss.backward()
    assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0


def test_weight_table():
    for t in (1, 100, 500, 1000):
        i = t - 1
        w = (S.betas[i] / (2 * S.alphas[i] * (1 - S.alpha_bars[i]))).item()
        print(f"t={t:4d} weight beta/(2 alpha (1-ab)) = {w:.4f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
