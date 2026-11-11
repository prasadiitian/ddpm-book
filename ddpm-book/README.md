# Denoising Diffusion Explained: code

Code for the book *Denoising Diffusion Explained*, which walks through the DDPM paper
(Ho, Jain and Abbeel, 2020) with derivations and tested PyTorch code.

Everything runs on a CPU. The dataset is scikit-learn's 8x8 `load_digits`, upscaled to 16x16,
so no downloads are needed.

## Layout

| Folder | What is in it |
|---|---|
| `ddpm/` | The library: schedules, forward process, loss, sampler, U-Net, training, DDIM and guidance |
| `tests/` | 41 tests (math checks, sanity checks, exact-oracle sampler tests, failure-mode controls) |
| `examples/` | The scripts behind the numbers and figures in Chapters 8 to 10 |
| `outputs/` | Small trained checkpoints (about 2.8 MB each) and the result files the book quotes |
| `notebooks/` | A Colab notebook for longer training runs |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests          # 41 tests; takes a couple of minutes on 2 CPU cores
python examples/ch10_guidance_experiment.py   # uses outputs/conditional.pt
```

To retrain from scratch on a CPU (this overwrites the shipped checkpoints in `outputs/`, and takes a while):

```bash
python examples/ch8_train.py linear 6000
python examples/ch8_train.py cosine 6000
python examples/ch10_train_conditional.py 6000
```

## Honest limits

- This is a small model on small images. It shows the method, not state-of-the-art quality.
- The pure-noise control in `ch8_evaluate.py` is not seeded, so its numbers vary slightly between runs; all other quality numbers reproduced exactly when re-run. Timings depend on your machine.
- Results come from a single trained model and a single seed unless the book says otherwise.
- The latent diffusion and Stable Diffusion material in Chapter 10 is explanation only; it is not implemented here.

## License

MIT. See `LICENSE`.
