import sys, json, time, torch
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
S = Schedule(linear_beta_schedule())
net = CondUNet(); net.load_state_dict(torch.load(f"{out}/conditional.pt")["ema"]); net.eval()

PER = 30
labels = torch.arange(10).repeat_interleave(PER)                      # 30 images of each digit
xT = torch.randn((len(labels), 1, 16, 16), generator=torch.Generator().manual_seed(1))

def stats(imgs):
    imgs = imgs.clamp(-1, 1)
    p = clf.predict_proba(imgs.flatten(1).numpy())
    hit = (p.argmax(1) == labels.numpy()).mean()
    p_req = p[range(len(labels)), labels.numpy()].mean()
    d = torch.cdist(imgs.flatten(1), tr_x.flatten(1)).min(1).values
    within = []
    for c in range(10):
        g = imgs[labels == c].flatten(1); dd = torch.cdist(g, g)
        within.append(float(dd.sum() / (len(g) * (len(g) - 1))))
    return {"hit_rate": float(hit), "prob_of_requested_class": float(p_req), "nn_dist_median": float(d.median()),
            "within_class_pairwise": sum(within) / 10, "pixel_std": float(imgs.std()), "pixel_mean": float(imgs.mean())}

res, picks = {}, {}
for scale in (0.0, 1.0, 2.0, 4.0, 8.0):
    model = lambda xt, t, s=scale: guided_eps(net, xt, t, labels[: xt.shape[0]], s)
    with torch.no_grad():
        imgs = sample_ddim(model, xT.shape, S, steps=50, eta=0.0, x_T=xT.clone())
    res[f"scale {scale}"] = stats(imgs); picks[scale] = imgs
    print(scale, res[f"scale {scale}"], flush=True)

# reference: how well the classifier does on real digits, and how varied a real class is
real_hit = float((clf.predict(te_x.flatten(1).numpy()) == te_y.numpy()).mean())
rw = []
for c in range(10):
    g = tr_x[tr_y == c][:PER].flatten(1); dd = torch.cdist(g, g); rw.append(float(dd.sum() / (len(g) * (len(g) - 1))))
res["real"] = {"hit_rate": real_hit, "within_class_pairwise": sum(rw) / 10}
json.dump(res, open(f"{out}/eval10_guidance.json", "w"), indent=1)

def grid(imgs, scale):
    sel = torch.cat([imgs[c * PER: c * PER + 8] for c in range(10)])     # 8 per digit, one row per digit
    sel = F.pad((sel.clamp(-1, 1) + 1) / 2, (1, 1, 1, 1), value=0.45); s = sel.shape[-1]
    return sel.reshape(10, 8, s, s).permute(0, 2, 1, 3).reshape(10 * s, 8 * s).numpy()
fig, ax = plt.subplots(1, 4, figsize=(13, 4.8), dpi=150)
for a, sc in zip(ax, (0.0, 1.0, 2.0, 4.0)):
    a.imshow(grid(picks[sc], sc), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a.set_title(f"guidance scale {sc:g}", fontsize=10); a.axis("off")
plt.tight_layout(); plt.savefig(f"{out}/guidance_grids.png")
print("DONE")
