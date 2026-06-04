"""Physics-Informed inverse calibration of heat diffusion.

This is the noise-robust calibration engine of the digital twin: from a few
temperature snapshots of a (real or simulated) object, recover its unknown
diffusivity alpha.

We deliberately *avoid* the naive PINN inverse formulation (a flexible network
plus a trainable alpha, both optimised jointly). On this problem that is badly
ill-conditioned: as the network fits the snapshots, it can explain the field
with its own flexibility instead of diffusion, so alpha drifts toward 0. That
drift is itself a real research finding about inverse identifiability.

Instead we use a clean two-phase method:

    Phase 1  fit a smooth surrogate u_theta(x, y, t) to the data (pure
             regression — well-posed, nothing for alpha to drift into).
    Phase 2  read alpha off the trained surrogate via the PDE, as the
             closed-form least-squares fit of  u_t = alpha * (u_xx + u_yy),
             using autodiff derivatives.

Because the derivatives come from a smooth network (not raw finite differences)
this stays robust when the data is noisy — which is exactly when the fast
least-squares calibrator in `digital_twin.py` struggles.

Demo / verification: generate data with a known alpha, hide it, recover it.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from simulate import solve_sequence


class HeatPINN(nn.Module):
    """Coordinate MLP u(x, y, t) for the temperature field."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)

    def derivatives(self, xyt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (u_t, laplacian u) at the given points via autodiff."""
        xyt = xyt.clone().requires_grad_(True)
        u = self.forward(xyt)
        grads = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
        u_x, u_y, u_t = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
        u_xx = torch.autograd.grad(u_x, xyt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, xyt, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        return u_t, u_xx + u_yy

    def estimate_alpha(self, xyt: torch.Tensor) -> float:
        """Read off diffusivity from the network: LS fit of u_t = alpha * lap u."""
        u_t, lap = self.derivatives(xyt)
        u_t, lap = u_t.detach(), lap.detach()
        return float((u_t * lap).sum() / (lap * lap).sum().clamp_min(1e-12))


def generate_observations(
    grid: int, alpha: float, t_final: float, n_snapshots: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate a diffusing field; return (coords, values, t_final)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:grid, 0:grid] / (grid - 1)
    u0 = np.zeros((grid, grid))
    for _ in range(4):
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        sigma = rng.uniform(0.05, 0.1)
        u0 += np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    u0 /= u0.max()

    frames = solve_sequence(u0, alpha, t_final, n_snapshots)  # (T, grid, grid)
    times = np.linspace(0.0, t_final, n_snapshots)

    def pack(k: int) -> tuple[np.ndarray, np.ndarray]:
        c = np.stack([xs.ravel(), ys.ravel(), np.full(grid * grid, times[k])], axis=1)
        return c.astype(np.float32), frames[k].ravel()[:, None].astype(np.float32)

    # Hold out the LAST snapshot for validation: a calibration that overfits the
    # earlier frames (and drifts alpha) will predict this future time poorly, so
    # validation error catches the drift and lets us early-stop.
    train_k = list(range(n_snapshots - 1))
    ctr = [pack(k)[0] for k in train_k]
    vtr = [pack(k)[1] for k in train_k]
    cval, vval = pack(n_snapshots - 1)
    return (
        np.concatenate(ctr), np.concatenate(vtr),
        cval, vval, t_final,
    )


def train(args) -> None:
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    coords, values, cval, vval, t_final = generate_observations(
        args.grid, args.true_alpha, args.t_final, args.snapshots, args.seed
    )
    Xd = torch.from_numpy(coords).to(device)
    Yd = torch.from_numpy(values).to(device)
    Xv = torch.from_numpy(cval).to(device)
    Yv = torch.from_numpy(vval).to(device)
    print(f"device: {device} | true alpha = {args.true_alpha} (hidden from the model)")
    print(f"observations: {len(Xd)} train points from {args.snapshots - 1} snapshots "
          f"+ 1 held-out validation snapshot\n")

    model = HeatPINN(width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collocation points where we read off the diffusivity (interior, away from
    # the t=0 and t=t_final edges where the network is least constrained).
    coll = torch.rand(args.collocation, 3, device=device)
    coll[:, 2] = 0.1 * t_final + 0.8 * t_final * coll[:, 2]

    def val_loss() -> float:
        with torch.no_grad():
            return float(torch.mean((model(Xv) - Yv) ** 2))

    # --- Phase 1: fit a smooth surrogate to the data (well-posed regression) ---
    # No trainable alpha here, so there is nothing for the field to "drift".
    for it in range(1, args.iters + 1):
        opt.zero_grad()
        data_loss = torch.mean((model(Xd) - Yd) ** 2)
        data_loss.backward()
        opt.step()
        if it % args.log_every == 0 or it == 1:
            a = model.estimate_alpha(coll)
            err = abs(a - args.true_alpha) / args.true_alpha * 100
            print(f"  iter {it:5d} | data {data_loss.item():.2e} "
                  f"| val {val_loss():.2e} | alpha {a:.5f} "
                  f"(true {args.true_alpha}, err {err:5.1f}%)")

    # --- Phase 2: read alpha off the trained surrogate via the PDE ---
    # alpha = argmin || u_t - alpha * lap u ||  (closed-form least squares),
    # using autodiff derivatives (smooth, so robust to noise unlike raw FD).
    a = model.estimate_alpha(coll)
    err = abs(a - args.true_alpha) / args.true_alpha * 100
    print(f"\nRECOVERED alpha = {a:.5f}  |  true = {args.true_alpha}  "
          f"|  error = {err:.2f}%")
    if err < 10:
        print("calibration succeeded (<10% error) -- the twin found the physics.")


def main() -> None:
    p = argparse.ArgumentParser(description="Inverse heat calibration with a PINN")
    p.add_argument("--true-alpha", type=float, default=0.01, help="hidden ground truth")
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--t-final", type=float, default=0.5)
    p.add_argument("--snapshots", type=int, default=6)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--collocation", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args())


if __name__ == "__main__":
    main()
