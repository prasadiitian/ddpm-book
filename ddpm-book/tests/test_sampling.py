"""Tier 1 tests for Chapter 6: the sampler, checked with exact (untrained) noise predictors."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
import torch
from ddpm import (Schedule, linear_beta_schedule, posterior_mean_var,
                  posterior_variance, p_sample_step, sample, sample_trajectory)

S = Schedule(linear_beta_schedule())
f64 = torch.float64


def gaussian_oracle(m, s2):
    """Exact E[eps | x_t] when x_0 ~ N(m, s2):  sqrt(1-ab) (x_t - sqrt(ab) m) / (ab s2 + 1 - ab)."""
    def model(xt, t):
        ab = S.alpha_bars[t - 1].to(xt.dtype).view(-1, *([1] * (xt.dim() - 1)))
        return (1 - ab).sqrt() * (xt - ab.sqrt() * m) / (ab * s2 + 1 - ab)
    return model


def mixture_oracle(ws, ms, s2s):
    """Exact E[eps | x_t] when x_0 is a mixture of 1-D Gaussians: -sqrt(1-ab) * score of p_t."""
    ws, ms, s2s = [torch.tensor(v, dtype=f64) for v in (ws, ms, s2s)]
    def model(xt, t):
        ab = S.alpha_bars[t - 1].to(xt.dtype).view(-1, *([1] * (xt.dim() - 1)))
        abk = ab.unsqueeze(-1)                    # (..., 1) so it broadcasts against the K components
        mu = abk.sqrt() * ms
        var = abk * s2s + (1 - abk)
        x = xt.unsqueeze(-1)
        logp = torch.log(ws) - 0.5 * torch.log(var) - 0.5 * (x - mu) ** 2 / var
        resp = torch.softmax(logp, dim=-1)
        score = (resp * (-(x - mu) / var)).sum(-1)
        return -(1 - ab).sqrt() * score
    return model


def test_last_step_is_deterministic():
    model = gaussian_oracle(0.0, 1.0)
    x = torch.randn(10, dtype=f64)
    a = p_sample_step(model, x, 1, S, generator=torch.Generator().manual_seed(1))
    b = p_sample_step(model, x, 1, S, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)


def test_same_seed_same_samples():
    model = gaussian_oracle(0.0, 1.0)
    a = sample(model, (50,), S, generator=torch.Generator().manual_seed(5), dtype=f64)
    b = sample(model, (50,), S, generator=torch.Generator().manual_seed(5), dtype=f64)
    assert torch.equal(a, b)


def test_posterior_variance_matches_chapter_3():
    x = torch.tensor(0.0, dtype=f64)
    for t in (1, 2, 100, 1000):
        assert abs(posterior_variance(S, t).item() - posterior_mean_var(x, x, t, S)[1].item()) < 1e-15


def test_mixture_oracle_matches_autograd_score():
    ws, ms, s2s = [0.3, 0.7], [-2.0, 1.5], [0.25, 0.5]
    model = mixture_oracle(ws, ms, s2s)
    t, x = 300, torch.linspace(-3, 3, 7, dtype=f64)
    ab = S.alpha_bars[t - 1]
    mu = ab.sqrt() * torch.tensor(ms, dtype=f64)
    var = ab * torch.tensor(s2s, dtype=f64) + (1 - ab)
    xg = x.clone().requires_grad_(True)
    logp = torch.logsumexp(torch.log(torch.tensor(ws, dtype=f64)) - 0.5 * torch.log(var)
                           - 0.5 * (xg.unsqueeze(-1) - mu) ** 2 / var, dim=-1)
    logp.sum().backward()
    want = -(1 - ab).sqrt() * xg.grad
    got = model(x, torch.full((7,), t))
    print("mixture oracle vs autograd score, largest difference:", (got - want).abs().max().item())
    assert (got - want).abs().max().item() < 1e-10


def test_trajectory_ends_where_sample_ends():
    model = gaussian_oracle(1.0, 0.5)
    a = sample(model, (200,), S, generator=torch.Generator().manual_seed(9), dtype=f64)
    b, snaps = sample_trajectory(model, (200,), S, generator=torch.Generator().manual_seed(9), dtype=f64)
    assert torch.equal(a, b) and torch.equal(snaps[0], b)
    assert sorted(snaps) == [0, 50, 100, 200, 400, 600, 800, 1000]
    assert snaps[1000].shape == (200,)


def run_gaussian(sigma):
    torch.manual_seed(0)
    m, s2, n = 2.0, 0.64, 200000
    x = sample(gaussian_oracle(m, s2), (n,), S, sigma=sigma, dtype=f64)
    print(f"sigma={sigma:5s}: target mean {m} var {s2} | sampled mean {x.mean().item():.4f} var {x.var().item():.4f}")
    return x.mean().item(), x.var().item(), m, s2


def test_gaussian_target_both_variance_choices():
    for sigma in ("beta", "tilde"):
        mean, var, m, s2 = run_gaussian(sigma)
        assert abs(mean - m) < 0.01 and abs(var - s2) < 0.01, sigma


def test_mixture_target():
    """Compare the sampled distribution with the exact mixture CDF on a grid of points."""
    torch.manual_seed(0)
    ws, ms, s2s = [0.3, 0.7], [-2.0, 1.5], [0.25, 0.5]
    x = sample(mixture_oracle(ws, ms, s2s), (200000,), S, dtype=f64)
    cdf = lambda v: sum(w * 0.5 * (1 + math.erf((v - m) / math.sqrt(2 * s2)))
                        for w, m, s2 in zip(ws, ms, s2s))
    worst = 0.0
    for v in [i * 0.25 for i in range(-16, 17)]:
        worst = max(worst, abs((x <= v).double().mean().item() - cdf(v)))
    mean_t = sum(w * m for w, m in zip(ws, ms))
    var_t = sum(w * (s2 + m * m) for w, m, s2 in zip(ws, ms, s2s)) - mean_t ** 2
    print(f"mixture: largest CDF difference {worst:.4f}")
    print(f"         sampled mean {x.mean().item():.4f} var {x.var().item():.4f} | exact mean {mean_t:.4f} var {var_t:.4f}")
    assert worst < 0.01
    assert abs(x.mean().item() - mean_t) < 0.02 and abs(x.var().item() - var_t) < 0.05


def test_zero_predictor_fails():
    """A model that predicts no noise should NOT reproduce the target: the test must be able to fail."""
    torch.manual_seed(0)
    zero = lambda xt, t: torch.zeros_like(xt)
    x = sample(zero, (200000,), S, dtype=f64)
    print(f"zero predictor: sampled mean {x.mean().item():.4f} var {x.var().item():.4f}  (target 2.0, 0.64)")
    assert abs(x.var().item() - 0.64) > 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
