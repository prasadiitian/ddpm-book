import sys, json, torch
sys.path.insert(0, ".")
from ddpm import *

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
out = "outputs"
torch.set_num_threads(1); torch.manual_seed(0)
x, y = load_digits16()
S = Schedule(linear_beta_schedule())
logf = open(f"{out}/conditional.log", "w")
def log(s): logf.write(s + "\n"); logf.flush()
model = CondUNet()
ema, losses = train_conditional(model, x[:1500], y[:1500], S, steps, log=log)
torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, f"{out}/conditional.pt")
json.dump(losses, open(f"{out}/conditional_losses.json", "w"))
log("DONE")
