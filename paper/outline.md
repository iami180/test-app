# Preprint outline

**Working title:** A cheap-sensor digital twin for heat diffusion:
neural-operator forecasting and noise-robust physics calibration

**Target:** arXiv preprint (cs.LG / physics.comp-ph), workshop-extensible.

---

## Abstract (draft)

Neural operators promise fast PDE surrogates but are almost always evaluated on
clean simulation. We build a small, fully reproducible testbed for a *digital
twin* of a heat-diffusing object that must (i) calibrate the unknown
diffusivity from observations, (ii) reconstruct and track the full temperature
field from a low-resolution, noisy sensor, and (iii) forecast forward with a
learned operator. We report two findings. First, the naive inverse PINN
(a flexible network with a jointly-trained diffusivity) is ill-conditioned: as
the network fits the data it explains the dynamics with its own flexibility and
the diffusivity drifts toward zero; a simple two-phase estimator (fit a smooth
surrogate, then read the parameter off its autodiff derivatives via the PDE)
removes the drift and is markedly more robust to measurement noise than a
finite-difference baseline. Second, a Fourier Neural Operator forecaster gives
up to ~9x speed-up over the classical solver, with the speed-up growing with
resolution as predicted by the O(N^4) vs O(N^2 log N) cost scaling. The twin
tracks a hidden full-resolution field from an 8x8 noisy sensor at 3-9% relative
L2. Code and experiments are released.

---

## Contributions

1. **Open testbed** for cheap-sensor digital twins of a diffusion field
   (calibration + sparse assimilation + neural-operator forecasting).
2. **Identifiability finding + remedy**: naive joint inverse PINN drifts; a
   two-phase fit-then-read-off estimator converges and is noise-robust.
3. **Cost-scaling analysis** of FNO vs classical solver, with an honest account
   of the regime where each wins.

## Sections

1. Introduction — neural operators are sim-only; digital twins need calibration
   + sparse sensing + forecasting; we provide a reproducible testbed.
2. Setup — 2D heat equation, FD solver (ground truth), FNO surrogate.
3. Forecasting & cost scaling — benchmark, F3 (speed-up vs resolution).
4. Inverse calibration — naive PINN drift (identifiability), two-phase method,
   F1 (error vs noise, FD vs PINN, multi-seed).
5. Digital twin — assimilation of an 8x8 noisy sensor, F2 (error vs sensor
   resolution and noise, vs interpolation baseline).
6. Related work — FNO/neural operators; PINNs & inverse problems; data
   assimilation; digital twins.
7. Limitations & future work — simulation-only (real AMG8833 sim-to-real next);
   single PDE; explicit solver baseline.

## Figures

- **F1** calibration error vs measurement noise: FD LS vs two-phase PINN,
  mean +/- std over seeds. `docs/f1_calibration_noise.png` (the key result).
- **F2** twin tracking relL2 vs sensor resolution (4/8/16/32) with a
  bilinear-interpolation baseline. `docs/f2_sensor_resolution.png` (done) — the
  twin stays ~3-8% while naive interpolation reaches ~84% at 4x4.
- **F3** runtime + speed-up vs grid resolution. `docs/benchmark_speedup.png` (done).
- (qualitative) twin tracking panels `docs/digital_twin.png`; FNO vs solver
  fields `docs/predictions.png`.

## Claims / scope (honesty)

- We claim noise-robust *calibration* and a working *simulation* testbed. We do
  NOT yet claim real-hardware sim-to-real (explicitly future work).
- Speed-up is reported on CPU with an explicit solver baseline; we state the
  regime where the solver wins (trivial / short-horizon).
- All numbers are multi-seed with error bars.

## Reproducibility checklist

- [x] Open code, deterministic seeds, CPU-only runnable.
- [x] F1 multi-seed figure + table.
- [x] F2 sensor-resolution sweep with baseline.
- [ ] Pinned dependency versions + one-command repro script.
- [ ] Prose draft of the preprint.
