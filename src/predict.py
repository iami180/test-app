"""Visualise FNO predictions against the ground-truth solver.

Loads a trained checkpoint, runs the operator on a few fresh test samples,
and saves a panel figure per sample:

    initial field  |  true uT (solver)  |  FNO prediction  |  error map

These are exactly the figures used for the paper and the pitch deck. If no
checkpoint exists yet, train one first with:

    python src/train.py --samples 1000 --epochs 50
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from dataset import generate
from model import FNO2d


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """Per-sample relative L2 error ||pred - target|| / ||target||."""
    num = np.linalg.norm(pred - target)
    den = max(np.linalg.norm(target), 1e-8)
    return float(num / den)


def load_model(ckpt_path: str, device: str) -> tuple[FNO2d, dict, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model = FNO2d(
        modes=saved.get("modes", 12),
        width=saved.get("width", 32),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, saved, ckpt.get("val_err", float("nan"))


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise FNO heat predictions")
    p.add_argument("--ckpt", type=str, default="checkpoints/fno_heat.pt")
    p.add_argument("--n-samples", type=int, default=4, help="panels to render")
    p.add_argument("--seed", type=int, default=123, help="held-out test seed")
    p.add_argument("--out", type=str, default="outputs/predictions.png")
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(
            f"checkpoint not found: {args.ckpt}\n"
            "train one first: python src/train.py --samples 1000 --epochs 50"
        )

    # matplotlib is only needed here, so import lazily.
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, saved, val_err = load_model(args.ckpt, device)
    print(f"device: {device} | loaded {args.ckpt} (val relL2 was "
          f"{val_err:.4f})")

    # Fresh samples the model never saw (different seed from training).
    grid = saved.get("grid", 64)
    alpha = saved.get("alpha", 1e-2)
    t_final = saved.get("t_final", 0.02)
    X, Y = generate(args.n_samples, grid, alpha, t_final, seed=args.seed)

    x_t = torch.from_numpy(X).float().unsqueeze(1).to(device)
    with torch.no_grad():
        pred = model(x_t).squeeze(1).cpu().numpy()

    n = args.n_samples
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["initial u0", "true uT (solver)", "FNO prediction", "abs error"]

    for i in range(n):
        u0, true, p_pred = X[i], Y[i], pred[i]
        err = np.abs(p_pred - true)
        rel = relative_l2(p_pred, true)

        # Shared colour scale for the first three (temperature) panels.
        vmax = max(u0.max(), true.max(), p_pred.max())
        for j, (field, cmap, vlim) in enumerate([
            (u0, "inferno", vmax),
            (true, "inferno", vmax),
            (p_pred, "inferno", vmax),
            (err, "viridis", err.max() or 1e-8),
        ]):
            ax = axes[i, j]
            im = ax.imshow(field, cmap=cmap, vmin=0, vmax=vlim, origin="lower")
            if i == 0:
                ax.set_title(col_titles[j])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[i, 0].set_ylabel(f"sample {i}\nrelL2={rel:.3f}", fontsize=10)

    fig.suptitle(
        f"FNO vs solver — heat diffusion (alpha={alpha}, t={t_final})",
        fontsize=13,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")

    mean_rel = np.mean([relative_l2(pred[i], Y[i]) for i in range(n)])
    print(f"mean relL2 over {n} test samples: {mean_rel:.4f}")
    print(f"figure saved: {args.out}")


if __name__ == "__main__":
    main()
