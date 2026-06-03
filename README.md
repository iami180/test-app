# Neural Operator találkozik a valósággal

*Gyors PDE-solverek tanulása valós mérésekből — hődiffúzió mint tesztrendszer.*

## A kutatási kérdés

> Be tudok-e tanítani egy **Fourier Neural Operatort** szimulált fizikán, ami
> aztán a valódi, zajos, ismeretlen-peremfeltételű garázs-kísérleten is
> működik — és ha nem, hogyan zárom be a **sim-to-real** szakadékot?

A Neural Operator irodalom szinte teljes egészében tiszta szimulált adaton
fut. A megkülönböztető érték itt egy **olcsó valós mérés** (AMG8833 8×8
hőszenzor egy fémlapon), amin a szimon tanult modellt validáljuk.

## Felépítés

```
.
├── requirements.txt
├── src/
│   ├── simulate.py     # 2D hődiffúzió finite-difference solver (ground truth)
│   ├── dataset.py      # (u0 -> uT) tanítópárok generálása
│   ├── model.py        # Fourier Neural Operator (2D FNO)
│   ├── train.py        # tréning loop + relatív L2 kiértékelés
│   └── predict.py      # vizualizáció: u0 | igazi uT | FNO jóslat | hibatérkép
└── sensor/
    └── read_amg8833.py # 8x8 hőszenzor kiolvasó (MicroPython / ESP32)
```

## Indulás

```bash
pip install -r requirements.txt

# Önálló smoke-tesztek
python src/simulate.py      # ellenőrzi az energiamegmaradást (Neumann perem)
python src/model.py         # paraméterszám + felbontás-invariancia
python src/dataset.py --samples 200 --out data/heat_dataset.npz

# Tréning (adat menet közben generálva, ha nincs --data)
python src/train.py --samples 1000 --epochs 50
# vagy előre generált adatból:
python src/train.py --data data/heat_dataset.npz --epochs 50

# Vizualizáció a betanított modellből (paper/pitch ábrák)
python src/predict.py --n-samples 4 --out outputs/predictions.png
```

**1. heti cél:** a validációs relatív L2 hiba < ~2% (a kanonikus FNO benchmark).
*Elért:* val relL2 ≈ 0.016 már 500 mintából / 20 epochból (CPU). ✅

## Roadmap

- [x] **1. hét** — szimulátor + adatgenerálás + FNO tréning + vizualizáció (szimon, val relL2 ≈ 0.016)
- [ ] **2. hét** — garázs kísérlet (AMG8833 + fémlap), α kalibráció
- [ ] **3. hét** — sim-to-real teszt: szimon tanult modell valós adaton (failure analysis)
- [ ] **4. hét** — fine-tune / fizikai loss → a valós hiba csökkentése + demó

## A fizika

2D hővezetési egyenlet a `[0,1]²` egységnégyzeten:

```
du/dt = alpha * laplace(u)
```

Az explicit séma stabil, amíg `alpha * dt * (1/dx² + 1/dy²) <= 0.5`; a solver
ezt automatikusan betartja (`cfl` paraméter). Alapból zero-Neumann (szigetelt)
peremek → a teljes hőmennyiség megmarad, ez jó ellenőrzési pont.
