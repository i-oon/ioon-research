# data/

Not in git (except this file). Full map, splits and the scripts that made each set: **`doc/DATA.md`**. Design:
`doc/DATA_PLAN.md`. Copy to another machine with `rsync -aL` (`doc/SIM_GUIDE.md` section 10).

| dir | what |
|---|---|
| `counterfactual_walks/` | locomotion clips + branches, c10 / B1 (train / val / heldout), c08 (test only) |
| `wall_b1/`, `wall_c10/` | wall data: rendered `train` / `val` / `test`, physics of all 28 commands in `_physics/`, balanced across bodies; `_physics/_unused_balance/` = valid states left out by the balance; `train_branches/` = symlinks used by the wall training run |
| `_frame_cache/` | decoded-frame cache written by training (rebuilt automatically; large) |

Every frame is 0.05 s (20 Hz). The previous version of this file (old dataset naming, 2026-09) is in
`doc/_archive/` history via git.
