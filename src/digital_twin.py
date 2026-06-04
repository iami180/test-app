"""Digital twin of a heat-diffusing object.

A digital twin is a live virtual replica of a real system that (1) is
calibrated to it, (2) stays in sync with incoming sensor data, and (3)
forecasts its future state. This module wires those three pieces together:

    calibrate()   estimate the physics (diffusivity alpha) from observations
    assimilate()  fuse a sparse/noisy sensor reading into the full-field state
    forecast()    advance the state forward in time with a fast engine

Calibration here uses a closed-form least-squares fit (du/dt = alpha * lap u),
which is instantaneous; `pinn.py` is the heavy-duty, noise-robust alternative.
Forecasting can use the classical solver (always correct) or the trained FNO
surrogate (fast). Assimilation nudges the model state toward a low-resolution
sensor field (e.g. the 8x8 AMG8833), interpolated up to the model grid.

Run `python src/digital_twin.py` for an end-to-end self-test: a synthetic
"real" object is tracked by the twin from noisy 8x8 readings, and we report
how well the twin's full-field estimate follows the hidden ground truth.
"""

from __future__ import annotations

import argparse

import numpy as np

from simulate import _laplacian, solve


def _upsample(coarse: np.ndarray, n: int) -> np.ndarray:
    """Bilinear-upsample a coarse field to an n x n grid (numpy-only)."""
    cy, cx = coarse.shape
    ys = np.linspace(0, cy - 1, n)
    xs = np.linspace(0, cx - 1, n)
    y0 = np.floor(ys).astype(int).clip(0, cy - 2)
    x0 = np.floor(xs).astype(int).clip(0, cx - 2)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x0 + 1] * wx
    bot = coarse[y0 + 1][:, x0] * (1 - wx) + coarse[y0 + 1][:, x0 + 1] * wx
    return top * (1 - wy) + bot * wy


class DigitalTwin:
    """Live virtual replica of a heat-diffusing object on an n x n grid."""

    def __init__(
        self,
        grid: int,
        alpha: float | None = None,
        domain_size: float = 1.0,
        boundary: str = "neumann",
    ):
        self.grid = grid
        self.alpha = alpha
        self.domain_size = domain_size
        self.boundary = boundary
        self.state: np.ndarray | None = None

    # --- 1. calibration -----------------------------------------------------
    def calibrate(self, frames: np.ndarray, dt: float) -> float:
        """Estimate alpha from a sequence of full-field snapshots.

        Closed-form least squares on the heat equation: stacking all interior
        points and time pairs, alpha = <du/dt, lap u> / <lap u, lap u>.
        """
        dx = self.domain_size / (self.grid - 1)
        num = den = 0.0
        for a, b in zip(frames[:-1], frames[1:]):
            dudt = (b - a) / dt
            lap = _laplacian(a, dx, self.boundary)
            num += float(np.sum(dudt * lap))
            den += float(np.sum(lap * lap))
        self.alpha = num / max(den, 1e-12)
        return self.alpha

    # --- 2. state assimilation ---------------------------------------------
    def set_state(self, field: np.ndarray) -> None:
        self.state = field.astype(np.float64, copy=True)

    def assimilate(self, sensor: np.ndarray, gain: float = 0.5) -> None:
        """Nudge the state toward a low-res sensor reading (coarse correction).

        Rather than overwriting the field with a blocky upsample of the sensor
        (which destroys the model's fine structure), we correct only the
        *coarse-scale* discrepancy: sample the model at the sensor locations,
        form the residual there, upsample that residual, and add it back. This
        preserves the high-frequency structure that the physics forecast
        provides while pulling the observed scales into agreement.

        `gain` in [0, 1] sets how aggressively to trust the sensor.
        """
        if self.state is None:
            self.state = _upsample(sensor.astype(np.float64), self.grid)
            return
        s = sensor.shape[0]
        step = self.grid // s
        model_coarse = self.state[::step, ::step][:s, :s]  # observation operator H
        delta = sensor.astype(np.float64) - model_coarse
        self.state = self.state + gain * _upsample(delta, self.grid)

    # --- 3. forecasting -----------------------------------------------------
    def forecast(self, t: float) -> np.ndarray:
        """Advance the state forward by physical time `t` (classical solver)."""
        if self.state is None or self.alpha is None:
            raise RuntimeError("twin needs a state and a calibrated alpha first")
        self.state = solve(
            self.state, self.alpha, t,
            domain_size=self.domain_size, boundary=self.boundary,
        )
        return self.state

    def step(self, sensor: np.ndarray | None, dt: float, gain: float = 0.5) -> np.ndarray:
        """One twin cycle: assimilate a reading (if any), then forecast dt."""
        if sensor is not None:
            self.assimilate(sensor, gain=gain)
        return self.forecast(dt)


def _self_test(args) -> None:
    """Track a synthetic 'real' object from noisy low-res readings."""
    rng = np.random.default_rng(args.seed)
    n = args.grid

    # Hidden ground-truth object.
    ys, xs = np.mgrid[0:n, 0:n] / (n - 1)
    truth = np.zeros((n, n))
    for _ in range(4):
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        truth += np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 0.07**2))
    truth /= truth.max()
    true_alpha = args.true_alpha

    def sense(field: np.ndarray) -> np.ndarray:
        """Coarse, noisy sensor: downsample to s x s + Gaussian noise."""
        step = n // args.sensor
        coarse = field[::step, ::step][: args.sensor, : args.sensor]
        return coarse + rng.normal(0, args.noise, coarse.shape)

    # Calibration phase: a few dense reference snapshots (e.g. thermal camera).
    dt = args.dt
    cal_frames = [truth.copy()]
    for _ in range(args.cal_frames - 1):
        cal_frames.append(solve(cal_frames[-1], true_alpha, dt))
    cal_frames = np.stack(cal_frames)

    twin = DigitalTwin(grid=n)
    est_alpha = twin.calibrate(cal_frames, dt)
    print(f"true alpha = {true_alpha} | calibrated alpha = {est_alpha:.5f} "
          f"({abs(est_alpha-true_alpha)/true_alpha*100:.1f}% error)\n")

    # Live tracking: the real object keeps evolving; the twin only sees noisy
    # low-res readings and must keep its full-field estimate in sync. The twin
    # starts from the dense calibration reference (a realistic initial state).
    real = cal_frames[-1].copy()
    twin.set_state(cal_frames[-1].copy())

    print(f"live tracking from {args.sensor}x{args.sensor} noisy readings "
          f"(noise={args.noise}):")
    errs, last = [], None
    for k in range(1, args.steps + 1):
        real = solve(real, true_alpha, dt)            # hidden truth advances
        reading = sense(real)                          # cheap sensor sees it
        est = twin.step(reading, dt, gain=args.gain)   # twin assimilate+forecast
        rel = np.linalg.norm(est - real) / max(np.linalg.norm(real), 1e-8)
        errs.append(rel)
        last = (real.copy(), est.copy(), reading.copy())
        print(f"  step {k:2d} | twin-vs-truth relL2 = {rel:.4f}")

    if args.plot:
        _plot_tracking(args.plot, last, errs, est_alpha, true_alpha)
        print(f"\nfigure saved: {args.plot}")


def _plot_tracking(path, last, errs, est_alpha, true_alpha):
    import os

    import matplotlib.pyplot as plt

    real, est, reading = last
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    vmax = max(real.max(), est.max())
    panels = [
        (real, "hidden truth", "inferno", 0, vmax),
        (est, "twin estimate", "inferno", 0, vmax),
        (reading, f"{reading.shape[0]}x{reading.shape[0]} sensor (noisy)", "inferno", 0, vmax),
        (np.abs(est - real), "abs error", "viridis", 0, None),
    ]
    for ax, (field, title, cmap, vmn, vmx) in zip(axes, panels):
        im = ax.imshow(field, cmap=cmap, vmin=vmn, vmax=vmx, origin="lower")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Digital twin tracking — calibrated alpha={est_alpha:.4f} "
        f"(true {true_alpha}), final relL2={errs[-1]:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser(description="Digital twin self-test")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--sensor", type=int, default=8, help="sensor resolution (s x s)")
    p.add_argument("--true-alpha", type=float, default=0.01)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--cal-frames", type=int, default=4)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--gain", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", type=str, default="", help="save a tracking figure here")
    _self_test(p.parse_args())


if __name__ == "__main__":
    main()
