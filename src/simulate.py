"""2D heat-diffusion solver (explicit finite differences).

Solves the heat equation

    du/dt = alpha * (d2u/dx2 + d2u/dy2)

on the unit square [0, 1] x [0, 1] with a uniform grid and zero-Neumann
(insulated) boundaries by default. This is the "ground truth" simulator we
use to generate training pairs for the Fourier Neural Operator.

The explicit scheme is stable while

    alpha * dt * (1/dx^2 + 1/dy^2) <= 0.5

so `step_dt` is clamped accordingly and the requested time interval is
covered with as many sub-steps as needed.
"""

from __future__ import annotations

import numpy as np


def _laplacian(u: np.ndarray, dx: float, boundary: str) -> np.ndarray:
    """Five-point Laplacian with the given boundary treatment."""
    if boundary == "neumann":
        # Insulated walls: pad by replicating the edge (zero gradient).
        up = np.pad(u, 1, mode="edge")
    elif boundary == "periodic":
        up = np.pad(u, 1, mode="wrap")
    elif boundary == "dirichlet":
        # Fixed zero temperature outside the domain.
        up = np.pad(u, 1, mode="constant", constant_values=0.0)
    else:
        raise ValueError(f"unknown boundary: {boundary!r}")

    lap = (
        up[2:, 1:-1]
        + up[:-2, 1:-1]
        + up[1:-1, 2:]
        + up[1:-1, :-2]
        - 4.0 * up[1:-1, 1:-1]
    ) / (dx * dx)
    return lap


def solve(
    u0: np.ndarray,
    alpha: float,
    t_final: float,
    domain_size: float = 1.0,
    boundary: str = "neumann",
    cfl: float = 0.4,
) -> np.ndarray:
    """Integrate the heat equation from u0 to time t_final.

    Parameters
    ----------
    u0 : (N, N) array
        Initial temperature field.
    alpha : float
        Thermal diffusivity.
    t_final : float
        Physical time to integrate to.
    domain_size : float
        Side length of the square domain.
    boundary : {"neumann", "periodic", "dirichlet"}
        Boundary condition.
    cfl : float
        Stability fraction in (0, 0.5]. Lower is safer/slower.

    Returns
    -------
    (N, N) array
        Temperature field at t_final.
    """
    if u0.ndim != 2 or u0.shape[0] != u0.shape[1]:
        raise ValueError("u0 must be a square 2D array")

    n = u0.shape[0]
    dx = domain_size / (n - 1)

    # Largest stable dt for the explicit scheme (2D => factor 2/dx^2).
    dt_max = cfl * dx * dx / (4.0 * alpha)
    n_steps = max(1, int(np.ceil(t_final / dt_max)))
    dt = t_final / n_steps

    u = u0.astype(np.float64, copy=True)
    for _ in range(n_steps):
        u = u + dt * alpha * _laplacian(u, dx, boundary)
    return u


def solve_sequence(
    u0: np.ndarray,
    alpha: float,
    t_final: float,
    n_snapshots: int,
    **kwargs,
) -> np.ndarray:
    """Return `n_snapshots` evenly spaced frames from 0 to t_final (inclusive)."""
    times = np.linspace(0.0, t_final, n_snapshots)
    frames = [u0.astype(np.float64, copy=True)]
    for prev_t, t in zip(times[:-1], times[1:]):
        frames.append(solve(frames[-1], alpha, t - prev_t, **kwargs))
    return np.stack(frames, axis=0)


if __name__ == "__main__":
    # Tiny smoke test: a hot square should diffuse and conserve heat
    # (Neumann boundaries => total energy is preserved).
    rng = np.random.default_rng(0)
    n = 64
    u0 = np.zeros((n, n))
    u0[24:40, 24:40] = 1.0

    uT = solve(u0, alpha=1e-2, t_final=0.05)
    print(f"initial sum  = {u0.sum():.4f}")
    print(f"final sum    = {uT.sum():.4f}  (should match under Neumann BC)")
    print(f"peak cooled  = {u0.max():.3f} -> {uT.max():.3f}")
