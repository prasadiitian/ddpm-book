import sys, json, time, torch
sys.path.insert(0, ".")
import numpy as np
from ddpm import *
from sklearn.linear_model import LogisticRegression

out = "outputs"
torch.set_num_threads(2)
x, y = load_digits16()
tr_x, te_x, tr_y, te_y = x[:1500], x[1500:], y[:1500], y[1500:]

clf = LogisticRegression(max_iter=2000).fit(tr_x.flatten(1).numpy(), tr_y.numpy())
res = {"classifier_heldout_accuracy": float(clf.score(te_x.flatten(1).numpy(), te_y.numpy()))}

def stats(imgs):
    imgs = imgs.clamp(-1, 1)
    p = clf.predict_proba(imgs.flatten(1).numpy())
    pred = p.argmax(1)
    d = torch.cdist(imgs.flatten(1), tr_x.flatten(1)).min(1).values
    return {
        "confidence": float(p.max(1).mean()),
        "class_counts": np.bincount(pred, minlength=10).tolist(),
        "nn_dist_median": float(d.median()),
        "pixel_mean": float(imgs.mean()), "pixel_std": float(imgs.std()),
        "outside_range": float(((imgs.abs() > 1).float().mean())),
    }

res["real_heldout"] = stats(te_x)
res["real_train"] = stats(tr_x[:297])
res["gaussian_noise"] = stats(torch.randn(297, 1, 16, 16))

N = 297
def gen(name, sched, which, sigma="beta", seed=0, untrained=False):
    S = Schedule(linear_beta_schedule() if sched == "linear" else cosine_beta_schedule())
    net = UNet()
    if not untrained:
        ck = torch.load(f"{out}/{name}.pt")
        net.load_state_dict(ck[which])
    net.eval()
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    imgs = sample(net, (N, 1, 16, 16), S, sigma=sigma, generator=g)
    return imgs, time.time() - t0

runs = {}
configs = [
    ("linear_ema_beta", "linear", "linear", "ema", "beta"),
    ("linear_raw_beta", "linear", "linear", "model", "beta"),
    ("linear_ema_tilde", "linear", "linear", "ema", "tilde"),
    ("cosine_ema_beta", "cosine", "cosine", "ema", "beta"),
]
for key, name, sched, which, sigma in configs:
    imgs, secs = gen(name, sched, which, sigma)
    s = stats(imgs); s["sample_seconds"] = secs
    res[key] = s
    runs[key] = imgs
    print(key, s, flush=True)

torch.manual_seed(0)
imgs, _ = gen(None, "linear", None, untrained=True)
res["untrained_net_linear"] = stats(imgs)
print("untrained", res["untrained_net_linear"], flush=True)

torch.save(runs, f"{out}/samples.pt")
json.dump(res, open(f"{out}/eval.json", "w"), indent=1)
print("DONE")
