# Data — what is current, which split, made by which script

> Updated 2026-10-03. Plan and design: `doc/DATA_PLAN.md`. Facts and checks: FINDINGS F293-F306. Label conventions:
> memory `project_timescale_conventions` (0.05 s frames; 1 s Froude at the centre of mass; smoothing and differencing never
> cross a command switch; read-out window 21). Everything under `data/` is git-ignored; copy to the server with `rsync -aL`.

## Current data: `data/counterfactual_walks/` (use only this)

| dir | body | split | files | what | made by |
|---|---|---|---|---|---|
| `c10_clips_{train,val,heldout}` | hexapod c10f10t10 | train / val / heldout | 48 / 24 / 24 | main clips: 24 behaviours, 66 frames, windows of one long walk per behaviour | `scripts/dataset/collect_c10_walks_and_branches.py` |
| `c10_walks` | hexapod | — | 24 walks | source walks (+ `targets.npy`, `_work/centre_pose.npz`) | same |
| `c10_branches_{train,val,heldout}` | hexapod | train / val / heldout | 3456 / 1728 / 1728 | branches: 3 points per main clip x 24 commands, 31 frames, start bit-identical | same |
| `b1_clips_{train,val,heldout}` | B1 | train / val / heldout | 48 / 24 / 24 | main clips; commands tuned to the hexapod's CoM Froude (24/24 within 0.002) | `scripts/dataset/collect_b1_walks.py` + corrected-mount re-render (`build_branches.py b1_remount`) |
| `b1_walks` | B1 | — | 24 walks | source walks (+ `tuning.json`, `targets.npy`) | same |
| `b1_branches_{train,val,heldout}` | B1 | train / val / heldout | 3456 / 1728 / 1728 | branches (exact MuJoCo restore, walk command as run) | `scripts/dataset/build_branches.py` |
| `c08_clips_heldout` | hexapod c08f09t09 | test only | 24 | main clips (never trained on) | `scripts/dataset/collect_c08_test_set.py` |
| `c08_walks` | c08 | — | 24 walks | source walks | same |
| `c08_branches_heldout` | c08 | test only | 1728 | branches for the read-out | same |
| `branch_points.json` | all | — | — | branch points (frame, gait phase) per main clip | `build_branches.py` |

**Rooms** (seed per clip, identical for every body): train 0-47 (behaviour i, copy k -> 2i+k), val 100-123, heldout 200-223.
Branches use their source clip's room. **Training pairs per body** (stride 5, rollout 2): main 2688 / 1344 / 1344, branches
38016 / 19008 / 19008. **Fields:** loader fields + `com_pos`, `cam_pose`, `room_seed`, `cond_index`; branches also
`segment`, `first_pair` (= 10), `froude_height`, `cf_*`; hexapod deterministic runs `det_*`.

**Checks (run after any data change):** `scripts/dataset/check_branches.py` (and `--c08`),
`scripts/diagnostics/dataset/check_splits.py` (includes c08; `--no_c08` to skip), `tests/test_froude_labels.py` (`FROUDE_TEST_HEX=c08` for c08),
`scripts/dataset/collect_c10_walks_and_branches.py targets`, `scripts/dataset/collect_b1_walks.py check`.

**Use:** pretraining = `*_clips_train` (+ `*_branches_train` for counterfactual arms); checkpoint selection = `*_val`; every reported
number = `*_heldout` and `c08_*` via `scripts/run/eval_suite.sh`.

## Superseded (kept, do not use)

| dir | why |
|---|---|
| `counterfactual_walks/_superseded/c10_replay_noise/` | hexapod clips/branches made by replay after a scene reload: branch starts ~6 cm apart (F304) |
| `counterfactual_walks/_superseded/b1_full_precision_command/` | B1 branches run with the full-precision command instead of the walk's (F305 close) |
| `counterfactual_walks/_superseded/b1_tuned_to_replay_targets/` | B1 tuned to the replay-based hexapod targets (4 conditions; 960 branches with old commands) |
| `counterfactual_walks/_superseded/b1_camera_yawed/` | B1 main clips with the camera yawed off the body (F304) |
| `counterfactual_walks/_superseded/b1_head_reference_targets/` | B1 tuned to head-reference hexapod Froude (F301) |
| `_archive_old_datasets/egocentric_v3/` | earlier B1 data (one command per clip, matched rendering): F293-corrected; replaced by `counterfactual_walks/` |
| `_archive_old_datasets/egocentric/` | beh12 / beh24 data: hexapod duplicate motions in 4 rooms (F300), B1 poses buggy before F293 (frames not re-rendered), test sets overlapping training (F297) |
| `_archive_old_datasets/allocentric/`, `_archive_old_datasets/proprioceptive/` | older third-person / state-only collections (F293 re-posed, frames not re-rendered) |
