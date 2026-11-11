import sys, json, torch
sys.path.insert(0, ".")
from ddpm import *

name, steps = sys.argv[1], int(sys.argv[2])
out = "outputs"
torch.set_num_threads(1)
torch.manual_seed(0)
x, y = load_digits16()
train_x = x[:1500]                      # the last 297 images are held out for evaluation
beta = linear_beta_schedule() if name == "linear" else cosine_beta_schedule()
S = Schedule(beta)
model = UNet()
logf = open(f"{out}/{name}.log", "w")
def log(s):
    logf.write(s + "\n"); logf.flush()
ema, losses = train(model, train_x, S, steps, log=log)
torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, f"{out}/{name}.pt")
json.dump(losses, open(f"{out}/{name}_losses.json", "w"))
log("DONE")
