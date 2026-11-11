import sys, json, torch, numpy as np
sys.path.insert(0, ".")
from ddpm import *
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = "outputs"
torch.set_num_threads(2)
x, y = load_digits16(); tr_x, te_x, tr_y, te_y = x[:1500], x[1500:], y[:1500], y[1500:]
clf = LogisticRegression(max_iter=2000).fit(tr_x.flatten(1).numpy(), tr_y.numpy())
def stats(imgs):
    imgs = imgs.clamp(-1, 1); p = clf.predict_proba(imgs.flatten(1).numpy())
    d = torch.cdist(imgs.flatten(1), tr_x.flatten(1)).min(1).values
    return {"confidence": float(p.max(1).mean()), "nn_dist_median": float(d.median()),
            "pixel_mean": float(imgs.mean()), "pixel_std": float(imgs.std())}
LIN, COS = Schedule(linear_beta_schedule()), Schedule(cosine_beta_schedule())
def load(path, key="ema"):
    n = UNet(); n.load_state_dict(torch.load(path)[key]); return n.eval()
net = load(f"{out}/linear.pt")
N = 64
res = {}

# 1. trajectory of 8 samples
g = torch.Generator().manual_seed(1)
final, snaps = sample_trajectory(net, (8, 1, 16, 16), LIN, generator=g)
keep = sorted(snaps, reverse=True)
fig, ax = plt.subplots(8, len(keep), figsize=(len(keep) * 1.2, 8 * 1.2), dpi=150)
for j, t in enumerate(keep):
    for i in range(8):
        ax[i, j].imshow(((snaps[t][i, 0].clamp(-1, 1) + 1) / 2).numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[i, j].axis("off")
    ax[0, j].set_title(f"t = {t}", fontsize=8)
plt.tight_layout(); plt.savefig(f"{out}/trajectory.png")
res["trajectory_pixel_std"] = {str(t): float(snaps[t].std()) for t in keep}

# 2. failure modes, 64 samples each, same seed
def gen(model, sched, sigma="beta", seed=2):
    return sample(model, (N, 1, 16, 16), sched, sigma=sigma, generator=torch.Generator().manual_seed(seed))
panels = {}
panels["Trained 6000 steps (reference)"] = gen(net, LIN)
panels["Trained 1500 steps"] = gen(load(f"{out}/short_1500.pt"), LIN)
panels["Trained 300 steps"] = gen(load(f"{out}/short_300.pt"), LIN)
panels["Sampled with the cosine schedule (trained with linear)"] = gen(net, COS)

# no noise added in the sampler
@torch.no_grad()
def sample_no_noise(model, shape, S, seed=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g)
    for t in range(S.T, 0, -1):
        tt = torch.full((shape[0],), t, dtype=torch.long)
        x = mean_from_eps(x, tt, model(x, tt), S)
    return x
panels["No noise added during sampling"] = sample_no_noise(net, (N, 1, 16, 16), LIN)

# off-by-one in the step number given to the network
class Shifted(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x, t): return self.m(x, t - 1)
panels["Network given t-1 instead of t"] = gen(Shifted(net), LIN)

for k, v in panels.items():
    res[k] = stats(v); print(k, res[k], flush=True)

def grid(imgs, n=8):
    imgs = F.pad((imgs[:n*n].clamp(-1, 1) + 1) / 2, (1, 1, 1, 1), value=0.45); s = imgs.shape[-1]
    return imgs.reshape(n, n, s, s).permute(0, 2, 1, 3).reshape(n * s, n * s).numpy()
fig, ax = plt.subplots(2, 3, figsize=(12, 8.4), dpi=150)
for a, (k, v) in zip(ax.flat, panels.items()):
    a.imshow(grid(v), cmap="gray", vmin=0, vmax=1, interpolation="nearest"); a.set_title(k, fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig(f"{out}/failures.png")
json.dump(res, open(f"{out}/eval9.json", "w"), indent=1)
print("DONE")
