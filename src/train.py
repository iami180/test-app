"""Train the FNO to emulate the heat-diffusion solver.

Generates (or loads) a dataset, trains an FNO2d, and reports the relative
L2 error on a held-out validation split. The week-1 goal is to drive the
validation relative L2 error below ~2%, matching the canonical FNO benchmark.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset import generate
from model import FNO2d


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of ||pred - target|| / ||target|| (per sample)."""
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    num = torch.linalg.norm(pred - target, dim=1)
    den = torch.linalg.norm(target, dim=1).clamp_min(1e-8)
    return (num / den).mean()


def load_data(args) -> tuple[np.ndarray, np.ndarray]:
    if args.data and os.path.exists(args.data):
        blob = np.load(args.data)
        print(f"loaded dataset from {args.data}")
        return blob["X"], blob["Y"]
    print(f"generating {args.samples} samples on the fly...")
    return generate(args.samples, args.grid, args.alpha, args.t_final, args.seed)


def main() -> None:
    p = argparse.ArgumentParser(description="Train FNO on heat diffusion")
    p.add_argument("--data", type=str, default="", help="optional .npz dataset")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1e-2)
    p.add_argument("--t-final", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--out", type=str, default="checkpoints/fno_heat.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    X, Y = load_data(args)
    X_t = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, H, W)
    Y_t = torch.from_numpy(Y).float().unsqueeze(1)

    n_val = int(len(X_t) * args.val_frac)
    n_train = len(X_t) - n_val
    train_ds = TensorDataset(X_t[:n_train], Y_t[:n_train])
    val_ds = TensorDataset(X_t[n_train:], Y_t[n_train:])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = FNO2d(modes=args.modes, width=args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} | train={n_train} val={n_val}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_err = 0.0
        for xb, yb in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = relative_l2(model(xb), yb)
            loss.backward()
            opt.step()
            train_err += loss.item() * len(xb)
        train_err /= n_train
        sched.step()

        model.eval()
        val_err = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_err += relative_l2(model(xb), yb).item() * len(xb)
        val_err /= max(n_val, 1)

        print(
            f"epoch {epoch:3d} | train relL2 {train_err:.4f} "
            f"| val relL2 {val_err:.4f}"
        )

        if val_err < best_val:
            best_val = val_err
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "val_err": val_err},
                args.out,
            )

    print(f"\nbest val relL2: {best_val:.4f}  (target < 0.02)")
    print(f"checkpoint: {args.out}")


if __name__ == "__main__":
    main()
