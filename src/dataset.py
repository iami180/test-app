"""Generate (initial field -> evolved field) training pairs.

Each sample is a pair (u0, uT) where uT = HeatSolver(u0) at a fixed
horizon. We draw random smooth initial conditions by summing a handful of
Gaussian "hot spots", which gives a rich but band-limited distribution that
the Fourier Neural Operator can learn from.
"""

from __future__ import annotations

import argparse

import numpy as np

from simulate import solve


def random_field(n: int, rng: np.random.Generator, n_blobs: int = 5) -> np.ndarray:
    """Random smooth initial temperature field as a sum of Gaussian blobs."""
    ys, xs = np.mgrid[0:n, 0:n] / (n - 1)
    field = np.zeros((n, n))
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0.15, 0.85, size=2)
        sigma = rng.uniform(0.04, 0.12)
        amp = rng.uniform(0.5, 1.0)
        field += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    # Normalise to [0, 1] for a stable training target range.
    field -= field.min()
    if field.max() > 0:
        field /= field.max()
    return field


def generate(
    n_samples: int,
    grid: int = 64,
    alpha: float = 1e-2,
    t_final: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (X, Y) each of shape (n_samples, grid, grid)."""
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, grid, grid), dtype=np.float32)
    Y = np.empty((n_samples, grid, grid), dtype=np.float32)
    for i in range(n_samples):
        u0 = random_field(grid, rng)
        uT = solve(u0, alpha=alpha, t_final=t_final)
        X[i] = u0.astype(np.float32)
        Y[i] = uT.astype(np.float32)
    return X, Y


def main() -> None:
    p = argparse.ArgumentParser(description="Generate heat-diffusion dataset")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1e-2)
    p.add_argument("--t-final", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/heat_dataset.npz")
    args = p.parse_args()

    X, Y = generate(args.samples, args.grid, args.alpha, args.t_final, args.seed)

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        Y=Y,
        alpha=args.alpha,
        t_final=args.t_final,
        grid=args.grid,
    )
    print(f"wrote {args.out}: X={X.shape}, Y={Y.shape}")


if __name__ == "__main__":
    main()
