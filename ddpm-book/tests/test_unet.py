"""Chapter 7 tests: shapes, time conditioning, independence across the batch, and a sanity run."""
import sys, pathlib, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from ddpm import (Schedule, linear_beta_schedule, UNet, timestep_embedding,
                  load_digits16, simple_loss)

S = Schedule(linear_beta_schedule())
X, Y = load_digits16()


def test_data():
    assert X.shape == (1797, 1, 16, 16)
    assert X.min().item() == -1.0 and X.max().item() == 1.0
    assert Y.bincount().tolist() == [178, 182, 177, 183, 181, 182, 181, 179, 174, 180]
    print("digits: mean", round(X.mean().item(), 3), "std", round(X.std().item(), 3))


def test_shapes_and_parameter_count():
    torch.manual_seed(0)
    net = UNet()
    out = net(X[:5], torch.tensor([1, 10, 100, 500, 1000]))
    assert out.shape == X[:5].shape
    n = sum(p.numel() for p in net.parameters())
    print("U-Net parameters:", n)


def test_timestep_embedding_values():
    e = timestep_embedding(torch.tensor([0, 1, 7]), 8)
    assert e.shape == (3, 8)
    assert torch.allclose(e[0, :4], torch.zeros(4)) and torch.allclose(e[0, 4:], torch.ones(4))
    assert abs(e[1, 0].item() - math.sin(1.0)) < 1e-6          # lowest index has frequency 1
    assert abs(e[2, 4].item() - math.cos(7.0)) < 1e-5


def test_the_step_number_changes_the_output():
    torch.manual_seed(0)
    net = UNet().eval()
    x = X[:4]
    with torch.no_grad():
        a = net(x, torch.full((4,), 1))
        b = net(x, torch.full((4,), 500))
    d = (a - b).abs().mean().item()
    print("mean |output(t=1) - output(t=500)|:", d)
    assert d > 1e-3


def test_samples_in_a_batch_do_not_interact():
    torch.manual_seed(0)
    net = UNet().eval()
    t = torch.tensor([3, 300, 600, 900])
    with torch.no_grad():
        together = net(X[:4], t)
        alone = net(X[2:3], t[2:3])
    diff = (together[2:3] - alone).abs().max().item()
    print("batch vs alone, largest difference:", diff)
    assert diff < 1e-5


def test_every_parameter_gets_a_gradient():
    torch.manual_seed(0)
    net = UNet()
    simple_loss(net, X[:16], S).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum().item() > 0, name


def test_sanity_overfit_one_batch():
    """Tier 2: a network that cannot memorise 16 fixed examples cannot be trained at all."""
    torch.manual_seed(0)
    net = UNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    x0 = X[:16]
    t = torch.randint(1, 1001, (16,))
    noise = torch.randn_like(x0)
    first = simple_loss(net, x0, S, t=t, noise=noise).item()
    start = time.time()
    for step in range(400):
        loss = simple_loss(net, x0, S, t=t, noise=noise)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"overfit: loss {first:.3f} -> {loss.item():.4f} in 400 steps ({time.time()-start:.0f} s)")
    assert first > 0.5
    assert loss.item() < 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
