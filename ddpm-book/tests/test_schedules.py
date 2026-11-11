"""Tier 1 tests for Chapter 5: noise schedules."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
import torch
from ddpm import Schedule, linear_beta_schedule, cosine_beta_schedule

LIN = Schedule(linear_beta_schedule())
COS = Schedule(cosine_beta_schedule())


def test_both_schedules_are_valid():
    for name, S in (("linear", LIN), ("cosine", COS)):
        assert (S.betas > 0).all() and (S.betas < 1).all(), name
        assert (S.alpha_bars[1:] < S.alpha_bars[:-1]).all(), name
        assert S.alpha_bars[-1].item() < 1e-3, name
        print(f"{name}: beta range [{S.betas.min().item():.2e}, {S.betas.max().item():.4f}]"
              f"  ab_T = {S.alpha_bars[-1].item():.3e}")


def test_cosine_matches_its_formula():
    """alpha_bar from cumprod(1-beta) must equal cos^2 formula wherever beta was not clipped."""
    T, s = 1000, 0.008
    f = lambda t: math.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    worst = 0.0
    for t in (1, 100, 250, 500, 750, 900, 990):
        want = f(t) / f(0)
        got = COS.alpha_bars[t - 1].item()
        worst = max(worst, abs(want - got))
    print("largest difference from the formula:", worst)
    assert worst < 1e-12


def test_clipping_only_touches_the_last_step():
    assert (COS.betas[:-1] < 0.999).all()
    assert abs(COS.betas[-1].item() - 0.999) < 1e-12


def test_linear_kills_signal_faster():
    """Report where each schedule crosses a few alpha_bar levels."""
    for level in (0.5, 0.1, 0.01):
        tl = int((LIN.alpha_bars > level).sum().item()) + 1
        tc = int((COS.alpha_bars > level).sum().item()) + 1
        print(f"alpha_bar drops below {level:<4}: linear at t={tl:4d}, cosine at t={tc:4d}")
    # Linear keeps less signal than cosine for t = 1..995; cosine only drops below it in the last 5 steps.
    assert (LIN.alpha_bars[:995] < COS.alpha_bars[:995]).all()
    assert (COS.alpha_bars[995:] < LIN.alpha_bars[995:]).all()


def test_table():
    for t in (1, 100, 250, 500, 750, 900, 1000):
        print(f"t={t:4d}  linear ab={LIN.alpha_bars[t-1].item():.6f}  cosine ab={COS.alpha_bars[t-1].item():.6f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
