"""Quantitative experiments for the write-up (multi-seed, with error bars).

Currently implements the headline calibration study:

    F1 -- diffusivity-recovery error vs measurement noise, comparing the fast
          finite-difference least-squares estimator against the two-phase PINN,
          aggregated over several seeds.

The hypothesis: the FD estimator uses a discrete Laplacian that amplifies
measurement noise, while the PINN reads alpha off a *smooth* fitted surrogate,
so it should degrade much more gracefully as noise grows.

Outputs a table to stdout and a figure with mean +/- std bands.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from digital_twin import DigitalTwin
from pinn import HeatPINN
from simulate import solve_sequence


def make_frames(grid: int, alpha: float, t_final: float, n_snap: int, seed: int):
    """Clean diffusion snapshots and their timestamps."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:grid, 0:grid] / (grid - 1)
    u0 = np.zeros((grid, grid))
    for _ in range(4):
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        u0 += np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 0.08**2))
    u0 /= u0.max()
    frames = solve_sequence(u0, alpha, t_final, n_snap)
    times = np.linspace(0.0, t_final, n_snap)
    return frames, times, (xs, ys)


def fd_calibrate(frames: np.ndarray, dt: float, grid: int) -> float:
    """Finite-difference least-squares diffusivity estimate."""
    return DigitalTwin(grid=grid).calibrate(frames, dt)


def pinn_calibrate(frames, times, grid, coords_xy, iters, seed) -> float:
    """Two-phase PINN: fit a smooth surrogate, then read alpha off its derivatives."""
    torch.manual_seed(seed)
    xs, ys = coords_xy
    coords, values = [], []
    for k, t in enumerate(times):
        coords.append(np.stack([xs.ravel(), ys.ravel(), np.full(grid * grid, t)], axis=1))
        values.append(frames[k].ravel()[:, None])
    Xd = torch.from_numpy(np.concatenate(coords)).float()
    Yd = torch.from_numpy(np.concatenate(values)).float()

    model = HeatPINN(width=64, depth=4)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    t_final = float(times[-1])
    coll = torch.rand(4000, 3)
    coll[:, 2] = 0.1 * t_final + 0.8 * t_final * coll[:, 2]

    for _ in range(iters):
        opt.zero_grad()
        loss = torch.mean((model(Xd) - Yd) ** 2)
        loss.backward()
        opt.step()
    return model.estimate_alpha(coll)


def run_f1(args) -> None:
    true_alpha = args.true_alpha
    dt = args.t_final / (args.snapshots - 1)
    noises = [float(x) for x in args.noises]

    print(f"F1: diffusivity recovery vs noise | true alpha = {true_alpha} "
          f"| {args.seeds} seeds | PINN iters={args.iters}\n")
    print(f"{'noise':>7} | {'FD err %':>18} | {'PINN err %':>18}")
    print("-" * 52)

    fd_mean, fd_std, pn_mean, pn_std = [], [], [], []
    for sigma in noises:
        fd_errs, pn_errs = [], []
        for seed in range(args.seeds):
            frames, times, xy = make_frames(args.grid, true_alpha, args.t_final, args.snapshots, seed)
            rng = np.random.default_rng(1000 + seed)
            noisy = frames + rng.normal(0, sigma, frames.shape)

            a_fd = fd_calibrate(noisy, dt, args.grid)
            a_pn = pinn_calibrate(noisy, times, args.grid, xy, args.iters, seed)
            fd_errs.append(abs(a_fd - true_alpha) / true_alpha * 100)
            pn_errs.append(abs(a_pn - true_alpha) / true_alpha * 100)

        fd_mean.append(np.mean(fd_errs)); fd_std.append(np.std(fd_errs))
        pn_mean.append(np.mean(pn_errs)); pn_std.append(np.std(pn_errs))
        print(f"{sigma:>7.3f} | {np.mean(fd_errs):>7.1f} +/- {np.std(fd_errs):<7.1f} "
              f"| {np.mean(pn_errs):>7.1f} +/- {np.std(pn_errs):<7.1f}")

    if args.plot:
        _plot_f1(args.plot, noises, fd_mean, fd_std, pn_mean, pn_std)
        print(f"\nfigure saved: {args.plot}")


def _plot_f1(path, noises, fd_m, fd_s, pn_m, pn_s):
    import os

    import matplotlib.pyplot as plt

    fd_m, fd_s = np.array(fd_m), np.array(fd_s)
    pn_m, pn_s = np.array(pn_m), np.array(pn_s)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(noises, fd_m, "o-", label="finite-difference LS", color="C1")
    ax.fill_between(noises, fd_m - fd_s, fd_m + fd_s, alpha=0.2, color="C1")
    ax.plot(noises, pn_m, "s-", label="two-phase PINN", color="C0")
    ax.fill_between(noises, pn_m - pn_s, pn_m + pn_s, alpha=0.2, color="C0")
    ax.set_xlabel("measurement noise (std)")
    ax.set_ylabel("diffusivity error (%)")
    ax.set_title("Noise-robust calibration: PINN vs finite differences")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser(description="Write-up experiments")
    p.add_argument("--true-alpha", type=float, default=0.01)
    p.add_argument("--grid", type=int, default=28)
    p.add_argument("--t-final", type=float, default=0.1)   # dt = 0.02 (FD-friendly)
    p.add_argument("--snapshots", type=int, default=6)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--noises", nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
    p.add_argument("--plot", type=str, default="outputs/f1_calibration_noise.png")
    run_f1(p.parse_args())


if __name__ == "__main__":
    main()
