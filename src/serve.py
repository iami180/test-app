"""Web demo: serve FNO heat-diffusion predictions over HTTP.

A small FastAPI service that loads a trained Fourier Neural Operator and exposes
it as an API plus a single-page UI. The page lets you generate a random heat
source and watch the FNO predict the field at a future time in milliseconds,
side by side with the classical solver (ground truth) and its timing — so the
speed-up is visible live.

Run:
    python src/serve.py            # then open http://127.0.0.1:8000
    # or: SERVE_CKPT=src/checkpoints/fno_heat.pt python src/serve.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from dataset import random_field
from model import FNO2d
from simulate import solve

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("SERVE_CKPT", os.path.join(_HERE, "checkpoints", "fno_heat_long.pt"))


def load_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    model = FNO2d(modes=saved.get("modes", 12), width=saved.get("width", 32))
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {
        "grid": saved.get("grid", 64),
        "alpha": saved.get("alpha", 1e-2),
        "t_final": saved.get("t_final", 0.02),
    }
    return model, meta


MODEL, META = load_model(CKPT)

# The FNO is resolution-invariant: trained at META["grid"], it still runs (and
# wins big) at higher resolution. The demo grid defaults higher so the speed-up
# is visible; override with SERVE_GRID.
DEMO_GRID = int(os.environ.get("SERVE_GRID", "256"))

# Warm up so served timings reflect steady-state, not one-off lazy-init cost.
with torch.no_grad():
    _warm = torch.zeros(1, 1, DEMO_GRID, DEMO_GRID)
    for _ in range(5):
        MODEL(_warm)


def fno_predict(field: np.ndarray) -> tuple[np.ndarray, float]:
    x = torch.from_numpy(field).float().unsqueeze(0).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = MODEL(x).squeeze().numpy()
    return pred, (time.perf_counter() - t0) * 1e3


def solver_predict(field: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    out = solve(field, META["alpha"], META["t_final"])
    return out.astype(np.float32), (time.perf_counter() - t0) * 1e3


app = FastAPI(title="Neural-operator heat-diffusion demo")


class PredictRequest(BaseModel):
    field: list[list[float]]


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "checkpoint": os.path.basename(CKPT),
            "demo_grid": DEMO_GRID, **META}


@app.get("/api/sample")
def sample() -> dict:
    rng = np.random.default_rng()
    u0 = random_field(DEMO_GRID, rng).astype(np.float32)
    return {"grid": DEMO_GRID, "field": u0.tolist()}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    field = np.asarray(req.field, dtype=np.float32)
    pred, fno_ms = fno_predict(field)
    truth, solver_ms = solver_predict(field)
    rel = float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-8))
    return {
        "grid": int(field.shape[0]),
        "prediction": pred.tolist(),
        "truth": truth.tolist(),
        "fno_ms": round(fno_ms, 3),
        "solver_ms": round(solver_ms, 3),
        "speedup": round(solver_ms / max(fno_ms, 1e-9), 1),
        "rel_l2": round(rel, 4),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>Neural-operator heat demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 880px; margin: 2rem auto;
         padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; } .sub { color:#666; margin-top:-.5rem; }
  button { font-size: 1rem; padding: .5rem 1rem; margin: .25rem .25rem .25rem 0;
           border:1px solid #ccc; border-radius:8px; background:#f6f6f6; cursor:pointer; }
  button:hover { background:#eee; } button:disabled { opacity:.5; cursor:default; }
  .row { display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:1rem; }
  .panel { text-align:center; } canvas { border:1px solid #ddd; border-radius:6px;
           image-rendering: pixelated; width:240px; height:240px; }
  .panel small { display:block; color:#666; margin-top:.3rem; }
  #stats { margin-top:1rem; font-size:1.05rem; }
  .big { font-weight:700; color:#0a7; }
</style></head>
<body>
  <h1>Neural operator vs solver — heat diffusion</h1>
  <p class="sub">Generate a heat source, then predict the field at a future time.</p>
  <button id="gen">1. Generate heat source</button>
  <button id="run" disabled>2. Predict</button>
  <div class="row">
    <div class="panel"><canvas id="c0" width="64" height="64"></canvas><small>initial u0</small></div>
    <div class="panel"><canvas id="c1" width="64" height="64"></canvas><small>FNO prediction</small></div>
    <div class="panel"><canvas id="c2" width="64" height="64"></canvas><small>solver (ground truth)</small></div>
  </div>
  <div id="stats"></div>
<script>
let current = null;
function color(t){ // simple blue->red "thermal" map
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255*Math.min(1, t*1.6));
  const g = Math.round(255*Math.max(0, Math.min(1, t*1.6-0.6)));
  const b = Math.round(255*Math.max(0, 1-t*1.6));
  return [r,g,b];
}
function draw(id, data){
  const n = data.length, cv = document.getElementById(id), ctx = cv.getContext('2d');
  cv.width = n; cv.height = n;  // match the field resolution (CSS scales display)
  let lo=Infinity, hi=-Infinity;
  for(const row of data) for(const v of row){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const img = ctx.createImageData(n, n);
  for(let y=0;y<n;y++) for(let x=0;x<n;x++){
    const t = (data[y][x]-lo)/((hi-lo)||1), [r,g,b]=color(t), k=4*(y*n+x);
    img.data[k]=r; img.data[k+1]=g; img.data[k+2]=b; img.data[k+3]=255;
  }
  ctx.putImageData(img, 0, 0);
}
async function gen(){
  const r = await (await fetch('/api/sample')).json();
  current = r.field; draw('c0', current);
  draw('c1', current.map(row=>row.map(()=>0)));
  draw('c2', current.map(row=>row.map(()=>0)));
  document.getElementById('run').disabled = false;
  document.getElementById('stats').innerHTML = '';
}
async function run(){
  if(!current) return;
  const btn = document.getElementById('run'); btn.disabled = true;
  const r = await (await fetch('/api/predict', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({field: current})})).json();
  draw('c1', r.prediction); draw('c2', r.truth);
  document.getElementById('stats').innerHTML =
    `FNO: <b>${r.fno_ms} ms</b> &nbsp; solver: <b>${r.solver_ms} ms</b> &nbsp; ` +
    `→ <span class="big">${r.speedup}× faster</span> &nbsp; (FNO vs solver rel L2: ${r.rel_l2})`;
  btn.disabled = false;
}
document.getElementById('gen').onclick = gen;
document.getElementById('run').onclick = run;
gen();
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn

    print(f"loaded {os.path.basename(CKPT)} | grid={META['grid']} "
          f"alpha={META['alpha']} t_final={META['t_final']}")
    print("open http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
