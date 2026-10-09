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

**Current state (2026-10-09). Read this before anything else.**
- The user is the researcher; they commit / push. Answer in Thai when they write Thai. Read STATUS, the memories and the
  relevant FINDINGS BEFORE acting; change one variable per experiment; never compare across weeks when the setup changed; name
  every run / data dir together with what it is (bodies, branches, rooms).
- Direction (advisor, week 19; memory `project_week19_direction`): prove the world model where it is needed -- predictive wall
  avoidance (rollout must beat direct / reactive), then maze; IROS paper. Babbling / no-controller bodies paused; candidates =
  tuned libraries. Original rooms (sized to the body), no random size; randomise the view (grayscale, brightness, texture,
  lighting) as Egocentric VSM.
- What stands: branches (F308); direct selection everywhere (F327); hexapod physics loop fixed (F323, `--cpg_clock`).
- Rollout's weak point: Froude reads shrunk toward the middle on every channel, worst in unseen rooms (z carries the room,
  F329 / F330); gait-phase mismatch at selection fixed for the hexapod (F330).
- Running / next: STATUS (grayscale + augmentation retrain; shared-room re-render `srbal_*`; then the wall test).
- Selection is judged only by achieved Froude per channel, never by which clip was picked (memory).

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
