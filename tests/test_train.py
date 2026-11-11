"""Chapter 8 tests: EMA arithmetic and a short training run."""
import sys, pathlib, copy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from ddpm import Schedule, linear_beta_schedule, UNet, load_digits16, train, ema_update

S = Schedule(linear_beta_schedule())


def test_ema_extremes_and_middle():
    torch.manual_seed(0)
    a, b = UNet(), UNet()
    for decay, expect in ((0.0, lambda pa, pb: pb), (1.0, lambda pa, pb: pa),
                          (0.9, lambda pa, pb: 0.9 * pa + 0.1 * pb)):
        e = copy.deepcopy(a)
        ema_update(e, b, decay)
        for pe, pa, pb in zip(e.parameters(), a.parameters(), b.parameters()):
            assert torch.allclose(pe, expect(pa, pb), atol=1e-6), decay


def test_ema_does_not_track_gradients():
    a, b = UNet(), UNet()
    ema_update(a, b, 0.5)
    assert all(not p.requires_grad or p.grad is None for p in a.parameters())


def test_training_is_reproducible_and_lowers_loss():
    x, _ = load_digits16()
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        _, losses = train(UNet(), x[:1500], S, steps=60, batch_size=32, log=lambda s: None)
        runs.append(losses)
    print(f"loss: first 10 steps {sum(runs[0][:10])/10:.3f} -> last 10 steps {sum(runs[0][-10:])/10:.3f}")
    assert runs[0] == runs[1]                         # same seed, same losses
    assert sum(runs[0][-10:]) < sum(runs[0][:10])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
