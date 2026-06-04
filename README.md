# Neural Operator találkozik a valósággal

*Gyors PDE-solverek tanulása valós mérésekből — hődiffúzió mint tesztrendszer.*

## A kutatási kérdés

> Be tudok-e tanítani egy **Fourier Neural Operatort** szimulált fizikán, ami
> aztán a valódi, zajos, ismeretlen-peremfeltételű garázs-kísérleten is
> működik — és ha nem, hogyan zárom be a **sim-to-real** szakadékot?

A Neural Operator irodalom szinte teljes egészében tiszta szimulált adaton
fut. A megkülönböztető érték itt egy **olcsó valós mérés** (AMG8833 8×8
hőszenzor egy fémlapon), amin a szimon tanult modellt validáljuk.

## Eredmény: az FNO akár 9× gyorsabb a klasszikus solvernél

A `t=0.5` horizontra tanított FNO **azonos pontosság mellett** (val relL2 ≈
0.0197) felülmúlja a klasszikus finite-difference solvert — és a gyorsulás
**nő a felbontással**, mert a stabilitás-korlátos solver költsége ~O(N⁴),
az FNO-é ~O(N² log N). Mindez **CPU-n, GPU nélkül**.

![FNO vs solver gyorsulás](docs/benchmark_speedup.png)

| felbontás | solver | FNO | gyorsulás |
|---:|---:|---:|---:|
| 64×64 | 6.8 ms | 5.1 ms | **1× (break-even)** |
| 128×128 | 59.8 ms | 21.6 ms | **3×** |
| 192×192 | 258 ms | 49.5 ms | **5×** |
| 256×256 | 890 ms | 95.7 ms | **9×** |

> Őszinte megjegyzés: **rövid** horizonton (`t=0.02`) a triviális solver még
> gyorsabb (az FNO 0.04–0.48×). Az FNO ott nyer, ahol a feladat drága: nagy
> felbontás, hosszú horizont, nehéz PDE-k, GPU, vagy sok lekérdezés. Részletek:
> [docs/benchmark.md](docs/benchmark.md).

Az FNO jóslata szabad szemmel megkülönböztethetetlen a solvertől:

![FNO jóslat vs igazi megoldás](docs/predictions.png)

## Digital twin: olcsó szenzorból teljes mező

A `digital_twin.py` egy **működő digital twin prototípus**: egy valós hődiffundáló
objektum élő virtuális mása, ami három dolgot tud:

1. **Kalibráció** — visszafejti a fizikát (α diffúziós állandó) megfigyelésekből.
   Két út: (a) gyors zárt-alakú FD legkisebb-négyzetes becslés (**4.8% hiba**,
   tiszta adatra); (b) `pinn.py` egy **zaj-robusztus** kétfázisú módszer — előbb
   sima felületet illeszt az adatra, majd az α-t a háló deriváltjaiból olvassa ki
   a PDE-vel (**~1–2% hiba**, és kezeli a naiv inverz-PINN identifiability-driftjét).
2. **Assimiláció** — egy **8×8-as, zajos** szenzor-leolvasást (a jövőbeli AMG8833
   hardver szimulációja) beolvaszt a teljes mezőbe, megőrizve a fizikából jövő
   finomszerkezetet.
3. **Előrejelzés** — időben előre gördíti az állapotot (a gyors FNO vagy a solver).

Az élő hurokban a twin a **rejtett, teljes felbontású valóságot** követi, miközben
csak az olcsó 8×8 szenzort látja — **3–9% relatív L2 hibával**:

![Digital twin követés](docs/digital_twin.png)

> Ez köti össze a szoftvert a tervezett **hardveres** fázissal: a szimulált 8×8
> szenzor helyére egy valódi AMG8833 kerül, és ugyanez a twin követi a valós
> fémlapot. (Roadmap 2–4. hét.)

## Felépítés

```
.
├── requirements.txt
├── src/
│   ├── simulate.py     # 2D hődiffúzió finite-difference solver (ground truth)
│   ├── dataset.py      # (u0 -> uT) tanítópárok generálása
│   ├── model.py        # Fourier Neural Operator (2D FNO)
│   ├── train.py        # tréning loop + relatív L2 kiértékelés
│   ├── predict.py      # vizualizáció: u0 | igazi uT | FNO jóslat | hibatérkép
│   ├── compare_speed.py # benchmark: FNO vs klasszikus solver (idő + pontosság)
│   ├── pinn.py         # Physics-Informed NN: inverz kalibráció (α visszanyerése)
│   └── digital_twin.py # digital twin: kalibráció + assimiláció + előrejelzés
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

# Benchmark: FNO vs klasszikus solver (a 9× gyorsulás reprodukálása)
python src/train.py --samples 600 --epochs 25 --t-final 0.5 \
    --out checkpoints/fno_heat_long.pt
python src/compare_speed.py --ckpt checkpoints/fno_heat_long.pt

# PINN inverz kalibráció: rejtett alpha visszanyerése megfigyelésekből
python src/pinn.py --true-alpha 0.01

# Digital twin: 8x8 zajos szenzorból teljes mező követése (+ ábra)
python src/digital_twin.py --plot outputs/digital_twin.png
```

**1. heti cél:** a validációs relatív L2 hiba < ~2% (a kanonikus FNO benchmark).
*Elért:* val relL2 ≈ 0.016 már 500 mintából / 20 epochból (CPU). ✅

## Roadmap

**Szoftver (startup-versenyképes mag):**
- [x] FNO szimulátor + adatgenerálás + tréning + vizualizáció (val relL2 ≈ 0.016)
- [x] Benchmark: FNO vs klasszikus solver (akár **9× gyorsulás**, CPU-n)
- [x] PINN inverz kalibráció (α visszanyerése megfigyelésekből)
- [x] Digital twin: olcsó 8×8 szenzorból teljes mező követése (**3–9% hiba**)
- [ ] Nehezebb PDE (Navier–Stokes) — itt 100–1000× a gyorsulás
- [ ] Web API / SaaS demó + bizonytalanság-becslés

**Hardver (a megkülönböztető réteg):**
- [ ] **2. hét** — garázs kísérlet (AMG8833 + fémlap), valós α kalibráció
- [ ] **3. hét** — sim-to-real teszt: szimon tanult modell valós adaton
- [ ] **4. hét** — fine-tune / fizikai loss → valós hiba csökkentése + demó

## A fizika

2D hővezetési egyenlet a `[0,1]²` egységnégyzeten:

```
du/dt = alpha * laplace(u)
```

Az explicit séma stabil, amíg `alpha * dt * (1/dx² + 1/dy²) <= 0.5`; a solver
ezt automatikusan betartja (`cfl` paraméter). Alapból zero-Neumann (szigetelt)
peremek → a teljes hőmennyiség megmarad, ez jó ellenőrzési pont.
