"""Chapter 8: the training loop (Algorithm 1 of the DDPM paper) with an EMA copy of the weights."""
import copy
import time
import torch
from .loss import simple_loss


@torch.no_grad()
def ema_update(ema_model, model, decay):
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def train(model, data, schedule, steps, batch_size=64, lr=5e-4, ema_decay=0.995,
          seed=0, log_every=200, log=print):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    ema = copy.deepcopy(model)
    losses, start = [], time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (batch_size,), generator=g)
        t = torch.randint(1, schedule.T + 1, (batch_size,), generator=g)
        noise = torch.randn(data[idx].shape, generator=g)
        loss = simple_loss(model, data[idx], schedule, t=t, noise=noise)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, ema_decay)
        losses.append(loss.item())
        if step % log_every == 0:
            recent = sum(losses[-log_every:]) / log_every
            log(f"step {step:5d}  loss {recent:.4f}  elapsed {time.time() - start:.0f}s")
    return ema, losses
