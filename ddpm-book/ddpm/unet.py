"""Chapter 7: a small U-Net with time conditioning, for 16x16 images."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the step number, as in the Transformer's position encoding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, groups=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.time = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(temb)[:, :, None, None]      # inject the step number
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention over all pixel positions (the Attention Is All You Need layer)."""
    def __init__(self, ch, groups=4):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(1, 2)[:, None], k.transpose(1, 2)[:, None],
                                             v.transpose(1, 2)[:, None])[:, 0]
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, in_ch=1, ch=32, tdim=64):
        super().__init__()
        self.tdim = tdim
        self.time_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch, tdim)               # 16 x 16
        self.down2 = ResBlock(ch, 2 * ch, tdim)           # 8 x 8
        self.mid1 = ResBlock(2 * ch, 2 * ch, tdim)
        self.attn = AttentionBlock(2 * ch)
        self.mid2 = ResBlock(2 * ch, 2 * ch, tdim)
        self.up2 = ResBlock(4 * ch, ch, tdim)             # skip connection doubles the channels
        self.up1 = ResBlock(2 * ch, ch, tdim)
        self.out = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.tdim))
        h0 = self.inc(x)
        h1 = self.down1(h0, temb)                              # 16 x 16
        h2 = self.down2(F.avg_pool2d(h1, 2), temb)             # 8 x 8
        m = self.mid2(self.attn(self.mid1(h2, temb)), temb)
        u = self.up2(torch.cat([m, h2], dim=1), temb)          # 8 x 8
        u = F.interpolate(u, scale_factor=2, mode="nearest")   # 16 x 16
        u = self.up1(torch.cat([u, h1], dim=1), temb)
        return self.out(u)
