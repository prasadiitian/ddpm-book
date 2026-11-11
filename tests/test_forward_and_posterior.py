"""Tier 1 tests: pure math, run in seconds on CPU. Chapters 2 and 3."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from ddpm import Schedule, linear_beta_schedule, q_step, q_sample, posterior_mean_var

torch.manual_seed(0)
S = Schedule(linear_beta_schedule())


def test_schedule_endpoints():
    assert abs(S.betas[0].item() - 1e-4) < 1e-12
    assert abs(S.betas[-1].item() - 0.02) < 1e-12
    assert (S.alpha_bars[1:] < S.alpha_bars[:-1]).all()  # strictly decreasing
    print("alpha_bar_1  =", S.alpha_bars[0].item())
    print("alpha_bar_500=", S.alpha_bars[499].item())
    print("alpha_bar_T  =", S.alpha_bars[-1].item())
    assert S.alpha_bars[-1].item() < 1e-4  # almost no signal left at t = T


def test_closed_form_matches_step_by_step():
    """Run the Markov chain 500 steps for many samples and compare with the closed form."""
    t, n = 500, 400000
    x0 = torch.full((n,), 1.5, dtype=torch.float64)
    x = x0.clone()
    for k in range(t):
        x = q_step(x, S.betas[k])
    ab = S.alpha_bars[t - 1].item()
    mean_theory = ab ** 0.5 * 1.5
    var_theory = 1.0 - ab
    print(f"step-by-step: mean {x.mean().item():.4f} var {x.var().item():.4f}")
    print(f"closed form : mean {mean_theory:.4f} var {var_theory:.4f}")
    assert abs(x.mean().item() - mean_theory) < 0.01
    assert abs(x.var().item() - var_theory) < 0.01
    xc = q_sample(x0, t, S)
    assert abs(xc.mean().item() - mean_theory) < 0.01
    assert abs(xc.var().item() - var_theory) < 0.01


def test_variance_preserved_for_unit_variance_data():
    """If x0 has variance 1, then var(x_t) = ab + (1 - ab) = 1 at every t."""
    x0 = torch.randn(400000, dtype=torch.float64)
    for t in (1, 100, 500, 1000):
        xt = q_sample(x0, t, S)
        assert abs(xt.var().item() - 1.0) < 0.01, t


def test_posterior_formulas_equal_bayes():
    """Compare the paper's posterior mean/variance with Gaussian conditioning (Bayes).

    p(x_{t-1}|x0) = N(sqrt(ab_{t-1}) x0, 1 - ab_{t-1});  p(xt|x_{t-1}) = N(sqrt(alpha_t) x_{t-1}, beta_t)
    Posterior precision = 1/(1-ab_{t-1}) + alpha_t/beta_t.
    """
    x0, xt = torch.tensor(0.7, dtype=torch.float64), torch.tensor(-0.4, dtype=torch.float64)
    worst = 0.0
    for t in (2, 10, 100, 500, 1000):
        i = t - 1
        beta, alpha = S.betas[i], S.alphas[i]
        ab_prev = S.alpha_bars[i - 1]
        prec = 1.0 / (1.0 - ab_prev) + alpha / beta
        var_bayes = 1.0 / prec
        mean_bayes = var_bayes * (ab_prev.sqrt() * x0 / (1.0 - ab_prev) + alpha.sqrt() * xt / beta)
        mean_f, var_f = posterior_mean_var(x0, xt, t, S)
        worst = max(worst, abs((mean_f - mean_bayes).item()), abs((var_f - var_bayes).item()))
    print("largest posterior difference vs Bayes:", worst)
    assert worst < 1e-12


def test_posterior_at_t1_is_a_point_mass_at_x0():
    x0, xt = torch.tensor(0.3, dtype=torch.float64), torch.tensor(0.5, dtype=torch.float64)
    mean, var = posterior_mean_var(x0, xt, 1, S)
    assert abs(mean.item() - x0.item()) < 1e-12
    assert abs(var.item()) < 1e-12


def test_shapes():
    x0 = torch.randn(8, 1, 28, 28, dtype=torch.float64)
    xt = q_sample(x0, 300, S)
    assert xt.shape == x0.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
