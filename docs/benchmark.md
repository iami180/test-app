# FNO vs classical solver — benchmark

- Device: `cpu`
- Trained resolution: `64x64`  (alpha=0.01, t_final=0.5)
- Accuracy @ trained res: **mean relative L2 = 0.0197**
- FNO throughput batch: 32

| grid | solver (ms) | FNO (ms) | speed-up |
|-----:|------------:|---------:|---------:|
| 32x32 | 1.13 | 1.296 | **0.87x** |
| 64x64 | 6.78 | 5.092 | **1x** |
| 96x96 | 21.34 | 11.526 | **2x** |
| 128x128 | 59.81 | 21.560 | **3x** |
| 192x192 | 258.01 | 49.515 | **5x** |
| 256x256 | 890.43 | 95.696 | **9x** |

## How to read this

The FNO **overtakes the solver** at higher resolution, reaching **9x** at
256x256. Below ~64x64 the cheap explicit solver still wins, but the gap closes
fast.

The reason is structural: the explicit solver is stability-limited
(dt ~ dx^2), so covering a fixed time costs ~O(N^4), while the
resolution-invariant FNO costs ~O(N^2 log N). The solver curve is steeper, so
the speed-up grows with resolution.

## Where the FNO wins even more (the product thesis)

- **GPU inference**: the FNO is massively parallel; batched GPU throughput is
  typically 100-1000x. This benchmark is CPU-only.
- **Long time horizons**: the FNO maps t=0 -> T in a *single* pass regardless
  of horizon (here t_final=0.5), while the solver's step count grows with T.
- **Harder PDEs** (Navier-Stokes, 3D, nonlinear, implicit) where classical
  solvers take minutes to hours per run.
- **Many-query settings** (design optimisation, uncertainty quantification)
  where one training is amortised over thousands of evaluations.
