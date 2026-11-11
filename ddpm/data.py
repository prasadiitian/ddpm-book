"""Small datasets that need no download."""
import torch
import torch.nn.functional as F


def load_digits16():
    """The 1797 handwritten digits that ship with scikit-learn, upscaled from 8x8 to 16x16
    and scaled to [-1, 1] as in the DDPM paper. Returns (images [N,1,16,16], labels [N])."""
    from sklearn.datasets import load_digits
    d = load_digits()
    x = torch.tensor(d.images, dtype=torch.float32).unsqueeze(1) / 16.0       # [0, 1]
    x = F.interpolate(x, size=16, mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0, torch.tensor(d.target)
