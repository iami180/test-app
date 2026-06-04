# Research statement

## In one sentence

We study how to build a **digital twin** of a physical system (heat diffusion)
that tracks and forecasts reality accurately from a **cheap, noisy, sparse
sensor**, while **recovering the system's unknown physical parameters**.

## Problem and motivation

Neural operators (e.g. the Fourier Neural Operator, FNO) can emulate physics
orders of magnitude faster than classical solvers, but the literature validates
them almost exclusively on **clean simulated data**, never on noisy real
measurements. Two ingredients are missing for practice:

1. **Calibration** — a real system's physical parameters (e.g. the thermal
   diffusivity α) are unknown and must be inferred from measurements.
2. **Sparse, noisy sensing** — we rarely observe the full field, only a few
   cheap sensor points (here an 8×8 thermal array) corrupted by noise.

Together these define a **digital twin**: a live virtual replica that
calibrates, tracks, and forecasts the real object.

## Research questions

- **Q1** Can the unknown physical parameter (α) be recovered from a few
  measurements, and how well does it tolerate **measurement noise**?
- **Q2** Can the full, high-resolution field be reconstructed and forecast from
  an **ultra-cheap sparse sensor** (8×8)?
- **Q3** When, and by how much, is the learned operator (FNO) actually faster
  than the classical solver?

## System under study

We use **2D heat diffusion** (`∂u/∂t = α·∇²u`) as the test system because the
physics is known and checkable (we have ground truth), it is cheaply
measurable with real hardware (a planned AMG8833 8×8 thermal sensor on a metal
plate), and the classical solver is a canonical FNO benchmark. Simulation
provides the teacher and the ground truth; the 8×8 sensor is currently
**simulated** (real hardware is the next, hardware phase).

## Two main results (the novelty)

1. **Identifiability finding + remedy.** The naive inverse PINN (a flexible
   network with a jointly-trained α) is ill-conditioned: as the network fits
   the data it explains the dynamics with its own flexibility instead of
   diffusion, and α drifts toward zero. We use a **two-phase** estimator — fit a
   smooth surrogate, then read α off its autodiff derivatives via the PDE —
   which removes the drift and is **noise-robust**, unlike the finite-difference
   baseline that amplifies noise.

2. **Working cheap-sensor digital twin.** The twin tracks the hidden,
   full-resolution truth while seeing only the noisy 8×8 sensor (3–9% relative
   L2), tying calibration, assimilation, and FNO forecasting into one loop.

Additionally, the FNO forecaster is up to ~9× faster than the solver (speed-up
grows with resolution), reported honestly including the regime where the solver
wins.

## Output / contribution

- An **open, reproducible testbed** (code + experiments) for cheap-sensor
  digital twins.
- A **negative result + remedy** (the PINN identifiability drift and the
  two-phase fix) — a citable methods result on its own.
- An **arXiv preprint** built on these (outline in `paper/outline.md`).

## Out of scope (for now)

- We do **not** yet claim real-hardware operation — that is the next phase
  (sim-to-real with a real AMG8833).
- We focus on a single PDE (heat diffusion); harder PDEs (Navier–Stokes) are
  future work.
- Speed-ups are measured on CPU against an explicit solver; this is stated
  explicitly.
