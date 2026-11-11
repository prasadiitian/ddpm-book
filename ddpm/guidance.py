"""Chapter 10: faster sampling (DDIM) and classifier-free guidance."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet, timestep_embedding
from .loss import _gather
from .train import ema_update


@torch.no_grad()
def sample_ddim(model, shape, schedule, steps=50, eta=0.0, generator=None, x_T=None,
                dtype=torch.float32, device="cpu"):
    """DDIM sampling (Song, Meng and Ermon, 2021) on a sub-sequence of `steps` time steps.

    eta = 0 is deterministic DDIM; eta = 1 with steps = T is the DDPM sampler with sigma^2 = beta_tilde.
    """
    T = schedule.T
    ts = [round(T * i / steps) for i in range(steps + 1)]          # 0 = ts[0] < ... < ts[-1] = T
    x = x_T if x_T is not None else torch.randn(shape, dtype=dtype, device=device, generator=generator)
    ones = torch.ones((), dtype=schedule.alpha_bars.dtype)
    for i in range(steps, 0, -1):
        t, t_prev = ts[i], ts[i - 1]
        ab_t = schedule.alpha_bars[t - 1]
        ab_prev = schedule.alpha_bars[t_prev - 1] if t_prev > 0 else ones
        tt = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
        eps = model(x, tt)
        x0 = (x - (1 - ab_t).sqrt().to(x.dtype) * eps) / ab_t.sqrt().to(x.dtype)
        sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
        x = ab_prev.sqrt().to(x.dtype) * x0 + (1 - ab_prev - sigma ** 2).sqrt().to(x.dtype) * eps
        if t_prev > 0 and eta > 0:
            noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            x = x + sigma.to(x.dtype) * noise
    return x


class CondUNet(UNet):
    """The Chapter 7 U-Net plus a class label. Label `num_classes` means 'no label'."""
    def __init__(self, num_classes=10, **kw):
        super().__init__(**kw)
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes + 1, self.tdim)

    def forward(self, x, t, y):
        temb = self.time_mlp(timestep_embedding(t, self.tdim)) + self.class_emb(y)
        h0 = self.inc(x)
        h1 = self.down1(h0, temb)
        h2 = self.down2(F.avg_pool2d(h1, 2), temb)
        m = self.mid2(self.attn(self.mid1(h2, temb)), temb)
        u = self.up2(torch.cat([m, h2], dim=1), temb)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.up1(torch.cat([u, h1], dim=1), temb)
        return self.out(u)


def guided_eps(model, x, t, y, scale):
    """Classifier-free guidance: eps_uncond + scale * (eps_cond - eps_uncond)."""
    null = torch.full_like(y, model.num_classes)
    both = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))
    e_cond, e_uncond = both.chunk(2)
    return e_uncond + scale * (e_cond - e_uncond)


def train_conditional(model, data, labels, schedule, steps, batch_size=64, lr=5e-4, ema_decay=0.995,
                      p_uncond=0.1, seed=0, log_every=200, log=print):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    ema = copy.deepcopy(model)
    losses = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (batch_size,), generator=g)
        x0, y = data[idx], labels[idx].clone()
        y[torch.rand(batch_size, generator=g) < p_uncond] = model.num_classes   # drop the label
        t = torch.randint(1, schedule.T + 1, (batch_size,), generator=g)
        noise = torch.randn(x0.shape, generator=g)
        ab = _gather(schedule.alpha_bars, t, x0)
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        loss = F.mse_loss(model(xt, t, y), noise)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, ema_decay)
        losses.append(loss.item())
        if step % log_every == 0:
            log(f"step {step:5d}  loss {sum(losses[-log_every:]) / log_every:.4f}")
    return ema, losses
