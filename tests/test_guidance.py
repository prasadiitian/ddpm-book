"""Chapter 10 tests: DDIM against exact oracles, and the guidance arithmetic."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from ddpm import (Schedule, linear_beta_schedule, sample, sample_ddim, CondUNet, guided_eps,
                  train_conditional, load_digits16)

S = Schedule(linear_beta_schedule())
f64 = torch.float64


def gaussian_oracle(m, s2):
    def model(xt, t):
        ab = S.alpha_bars[t - 1].to(xt.dtype).view(-1, *([1] * (xt.dim() - 1)))
        return (1 - ab).sqrt() * (xt - ab.sqrt() * m) / (ab * s2 + 1 - ab)
    return model


def mixture_oracle(ws, ms, s2s):
    ws, ms, s2s = [torch.tensor(v, dtype=f64) for v in (ws, ms, s2s)]
    def model(xt, t):
        ab = S.alpha_bars[t - 1].to(xt.dtype).view(-1, *([1] * (xt.dim() - 1)))
        abk = ab.unsqueeze(-1)
        mu, var = abk.sqrt() * ms, abk * s2s + (1 - abk)
        x = xt.unsqueeze(-1)
        resp = torch.softmax(torch.log(ws) - 0.5 * torch.log(var) - 0.5 * (x - mu) ** 2 / var, dim=-1)
        return -(1 - ab).sqrt() * (resp * (-(x - mu) / var)).sum(-1)
    return model


def test_ddim_with_eta_1_and_all_steps_is_the_ddpm_sampler():
    """The identity behind DDIM: eta = 1, S = T must reproduce Chapter 6's sampler with sigma^2 = beta_tilde."""
    model = gaussian_oracle(0.5, 0.8)
    a = sample(model, (300,), S, sigma="tilde", generator=torch.Generator().manual_seed(4), dtype=f64)
    b = sample_ddim(model, (300,), S, steps=1000, eta=1.0, generator=torch.Generator().manual_seed(4), dtype=f64)
    d = (a - b).abs().max().item()
    print("DDIM(eta=1, 1000 steps) vs DDPM sampler, largest difference:", d)
    assert d < 1e-8


def test_ddim_eta_0_is_deterministic():
    model = gaussian_oracle(0.5, 0.8)
    xT = torch.randn(100, dtype=f64, generator=torch.Generator().manual_seed(1))
    a = sample_ddim(model, (100,), S, steps=50, x_T=xT, generator=torch.Generator().manual_seed(10))
    b = sample_ddim(model, (100,), S, steps=50, x_T=xT, generator=torch.Generator().manual_seed(11))
    assert torch.equal(a, b)


def test_ddim_error_shrinks_as_steps_grow():
    """With the exact predictor, fewer DDIM steps cost accuracy. The error must fall as steps rise."""
    m, s2 = 2.0, 0.64
    errs = {}
    for steps in (10, 20, 50, 100, 1000):
        torch.manual_seed(0)
        x = sample_ddim(gaussian_oracle(m, s2), (200000,), S, steps=steps, dtype=f64)
        errs[steps] = abs(x.var().item() - s2)
        print(f"DDIM {steps:4d} steps: mean {x.mean().item():.4f} (2.0)  var {x.var().item():.4f} (0.64)")
        assert abs(x.mean().item() - m) < 0.02, steps
    assert errs[10] > errs[20] > errs[50] > errs[100] > errs[1000]
    assert errs[1000] < 0.02


def test_ddim_mixture_target():
    torch.manual_seed(0)
    ws, ms, s2s = [0.3, 0.7], [-2.0, 1.5], [0.25, 0.5]
    x = sample_ddim(mixture_oracle(ws, ms, s2s), (200000,), S, steps=50, dtype=f64)
    cdf = lambda v: sum(w * 0.5 * (1 + math.erf((v - m) / math.sqrt(2 * s2))) for w, m, s2 in zip(ws, ms, s2s))
    worst = max(abs((x <= v).double().mean().item() - cdf(v)) for v in [i * 0.25 for i in range(-16, 17)])
    print(f"DDIM 50 steps, mixture: largest CDF difference {worst:.4f}")
    assert worst < 0.02


def test_guidance_arithmetic():
    torch.manual_seed(0)
    net = CondUNet().eval()
    x, t, y = torch.randn(4, 1, 16, 16), torch.tensor([5, 50, 500, 900]), torch.tensor([0, 3, 7, 9])
    null = torch.full_like(y, 10)
    with torch.no_grad():
        ec, eu = net(x, t, y), net(x, t, null)
        assert torch.allclose(guided_eps(net, x, t, y, 1.0), ec, atol=1e-5)     # scale 1 = plain conditional
        assert torch.allclose(guided_eps(net, x, t, y, 0.0), eu, atol=1e-5)     # scale 0 = unconditional
        assert torch.allclose(guided_eps(net, x, t, y, 3.0), eu + 3 * (ec - eu), atol=1e-5)
    assert (ec - eu).abs().mean().item() > 1e-4                                  # the label really changes the output


def test_conditional_training_runs_and_drops_labels():
    x, y = load_digits16()
    torch.manual_seed(0)
    model = CondUNet()
    _, losses = train_conditional(model, x[:1500], y[:1500], S, steps=40, batch_size=32, log=lambda s: None)
    assert sum(losses[-10:]) < sum(losses[:10])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
