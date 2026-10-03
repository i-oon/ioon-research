# Start here (resume prompt)

You are resuming a long research project mid-stream. Read in this order, then confirm the state back
before running anything.

1. **`doc/STATUS.md`** — what is done, what is running, what is next, open decisions. The living tracker:
   update it when something changes.
2. **Your memories** — locked conventions (timescale, Froude labels, resource limits, slide style). They
   override anything older.
3. **`doc/DATA.md`** — the only current data (`data/counterfactual_walks/`), its train / val / heldout
   splits and the scripts that made it. Anything under `_superseded/` or `_archive*` is not used.
4. **`ARCHITECTURE.md`** — the model next to LAC-WM and Egocentric VSM.
5. **`doc/FINDINGS.md`** — the evidence, numbered. Not append-only: read corrections and withdrawals.
   Results before 2026-10-01 are pre-fix (F293 / F295 / F297 / F300 / F301) and not quotable.
6. `doc/DATA_PLAN.md` (how the data was designed), `doc/PROGRESS.md` (dated log, Thai + English),
   `doc/SIM_GUIDE.md` (install and simulators).

**The project.** Cross-embodiment locomotion from egocentric video: a latent action world model (frozen
V-JEPA2, ITM, FTM) whose latent is grounded in Froude-scaled body motion (forward, lateral, yaw at the
centre of mass), shared by every body; the only per-body part is the action projector. Bodies:
hexapod c10f10t10 (pretraining, CoppeliaSim), hexapod c08f09t09 (test only), Unitree B1 (MuJoCo).

**Where things are.**

| | |
|---|---|
| model code | `wm/` (`train.py`, `config.py`, `data/`, `models/`, `policy/`) |
| data generation and checks | `scripts/dataset/`, `sim/collect/`, `sim/render/`, `scripts/diagnostics/dataset/` |
| training / evaluation | `scripts/run/round1_counterfactual.sh`, `scripts/run/eval_suite.sh` (every reported number) |
| runs | `wm/runs/<name>/` (old runs in `wm/runs/_archive_old_runs/`) |
| results | `results/eval/<name>/summary.txt`, logs in `results/wm/logs/` |
| caches | `data/_frame_cache/` (training frames), `results/wm/cache/` (eval embeddings, stamped) |
| old docs | `doc/_archive/` |

**Rules.** `.venv/bin/python3` from the repo root. Resource limits from memory (≤ 6 own CoppeliaSim
instances, never port 23000, one GPU job, estimate RAM before loading data). The user commits and pushes;
no attribution lines in commits. Never use the expert CSV. Don't touch `report/report.tex`.
