import sys, torch
sys.path.insert(0, ".")
from ddpm import *
steps = int(sys.argv[1])
out = "outputs"
torch.set_num_threads(1); torch.manual_seed(0)
x, y = load_digits16()
S = Schedule(linear_beta_schedule())
ema, losses = train(UNet(), x[:1500], S, steps, log=lambda s: None)
torch.save({"ema": ema.state_dict(), "final_loss": sum(losses[-100:]) / min(100, len(losses))}, f"{out}/short_{steps}.pt")
print("done", steps, sum(losses[-100:]) / min(100, len(losses)))
