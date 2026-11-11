import sys, json, time, torch
sys.path.insert(0, ".")
from ddpm import *
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = "outputs"
torch.set_num_threads(1)
x, y = load_digits16(); tr_x, te_x, tr_y = x[:1500], x[1500:], y[:1500]
clf = LogisticRegression(max_iter=2000).fit(tr_x.flatten(1).numpy(), tr_y.numpy())
S = Schedule(linear_beta_schedule())
net = UNet(); net.load_state_dict(torch.load(f"{out}/linear.pt")["ema"]); net.eval()
model = lambda xt, t: net(xt, t)

def stats(imgs):
    imgs = imgs.clamp(-1, 1); p = clf.predict_proba(imgs.flatten(1).numpy())
    d = torch.cdist(imgs.flatten(1), tr_x.flatten(1)).min(1).values
    pd = torch.cdist(imgs.flatten(1), imgs.flatten(1)); n = len(imgs)
    counts = torch.bincount(torch.tensor(p.argmax(1)), minlength=10)
    return {"confidence": float(p.max(1).mean()), "nn_dist_median": float(d.median()),
            "pixel_mean": float(imgs.mean()), "pixel_std": float(imgs.std()),
            "pairwise_dist": float(pd.sum() / (n * (n - 1))), "min_class": int(counts.min()), "max_class": int(counts.max())}

N = 297
xT = torch.randn((N, 1, 16, 16), generator=torch.Generator().manual_seed(1))
res, picks = {}, {}
configs = [("DDPM, 1000 steps", None, None)] + [(f"DDIM eta=0, {s} steps", s, 0.0) for s in (100, 50, 20, 10)] \
          + [("DDIM eta=1, 50 steps", 50, 1.0)]
for name, steps, eta in configs:
    t0 = time.time()
    if steps is None:
        imgs = sample(net, (N, 1, 16, 16), S, sigma="tilde", generator=torch.Generator().manual_seed(2))
    else:
        with torch.no_grad():
            imgs = sample_ddim(model, (N, 1, 16, 16), S, steps=steps, eta=eta, x_T=xT.clone(),
                               generator=torch.Generator().manual_seed(2))
    secs = time.time() - t0
    res[name] = stats(imgs); res[name]["seconds"] = secs; picks[name] = imgs
    print(name, res[name], flush=True)
res["real_heldout"] = stats(te_x)
json.dump(res, open(f"{out}/eval10_ddim.json", "w"), indent=1)

def grid(imgs, n=6):
    imgs = F.pad((imgs[:n*n].clamp(-1, 1) + 1) / 2, (1, 1, 1, 1), value=0.45); s = imgs.shape[-1]
    return imgs.reshape(n, n, s, s).permute(0, 2, 1, 3).reshape(n * s, n * s).numpy()
show = ["DDIM eta=0, 100 steps", "DDIM eta=0, 50 steps", "DDIM eta=0, 20 steps", "DDIM eta=0, 10 steps"]
fig, ax = plt.subplots(1, 4, figsize=(14, 3.8), dpi=150)
for a, k in zip(ax, show):
    a.imshow(grid(picks[k]), cmap="gray", vmin=0, vmax=1, interpolation="nearest"); a.set_title(k + " (same start noise)", fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig(f"{out}/ddim_grids.png")
print("DONE")
