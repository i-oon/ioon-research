# Pre-world-model results

Everything here predates the world model and is dated **2026-08-06**, except one preview folder
from 08-12. It is the evidence behind `PROGRESS.md` sections 13 and 15, kept because those
sections' conclusions rest on it and nothing regenerates it.

**Do not cite these for current claims.** Current results live under `results/wm/`. If a number
here and a number there disagree, the one under `results/wm/` is the live one.

| | what it is |
|---|---|
| `ik/render_lock_*` | the render-lock verification: same body and behaviour recorded twice must render identically. `PROGRESS.md` §15 -- void 0.00%, camera offset identical across three variants. The `emb.npz` files inside are encoded features and are regenerable by `scripts/dataset/render_lock_check.py` |
| `amp_failed/` | the AMP route before the IK pipeline replaced it. `PROGRESS.md` §13. Kept as the evidence that it was tried and produced worse, less coordinated behaviour -- which is an argument for the chosen approach, not an omission |
| `gait_ik_walk_3sec/`, `leg_loss/`, `original_uneven_morph_walk_aligned/` | early gait and leg-removal checks from the same week |
| `ik_4leg_middleloss_clean9_preview/` | 08-12, the preview pass on the first 4-leg build. Superseded by the held-out `c08f09t09` build; see F59 |

`results/wm/dataset/` is a different thing despite the similar name: it holds **current**
data-quality checks that belong to neither Stage 1 nor Stage 2.
