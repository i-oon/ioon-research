# Data — what is current, which split, made by which script

> Updated 2026-10-14 (wall data `data/wall_b1`, `data/wall_c10` added below). 2026-10-09: random-size-room data `rr_*`, `rrv2_*`, `rrv3_*` deleted; dropped direction). Plan and design: `doc/DATA_PLAN.md`. Facts and checks: FINDINGS F293-F306. Label conventions:
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
| `branch_points_current.json` | all | — | 216 | branch points (frame, gait phase) per current main clip (c08 = its c10 heldout counterpart's) | `build_branches.py rekey` (from `branch_points.json`) |
| `branch_points.json` | hex, B1 | — | 192 | same points keyed by the superseded stage-1/2 clip paths (kept; read only by `rekey`) | `build_branches.py points` (2026-10-01) |

**Rooms** (seed per clip, identical for every body): train 0-47 (behaviour i, copy k -> 2i+k), val 100-123, heldout 200-223.
Branches use their source clip's room. **Training pairs per body** (stride 5, rollout 2): main 2688 / 1344 / 1344, branches
38016 / 19008 / 19008. **Fields:** loader fields + `com_pos`, `cam_pose`, `room_seed`, `cond_index`; branches also
`segment`, `first_pair` (= 10), `froude_height`, `cf_*`; hexapod deterministic runs `det_*`.

**Checks (run after any data change):** `scripts/dataset/check_branches.py` (and `--c08`),
`scripts/diagnostics/dataset/check_splits.py` (includes c08; `--no_c08` to skip), `tests/test_froude_labels.py` (`FROUDE_TEST_HEX=c08` for c08),
`scripts/dataset/collect_c10_walks_and_branches.py targets`, `scripts/dataset/collect_b1_walks.py check`.

**Use:** pretraining = `*_clips_train` (+ `*_branches_train` for counterfactual arms); checkpoint selection = `*_val`; every reported
number = `*_heldout` and `c08_*` via `scripts/run/eval_suite.sh`.

## Added 2026-10-09

| dir | what | made by |
|---|---|---|
| ~~`srbal_{c10,b1}_{clips,branches}_train`~~ | DELETED 2026-10-12 (shared-room-look variant, not pursued, F331 note). Also deleted then: `rs_*` / `orig24_*` (room-size / shifted-room held-out test), `b1_babble_*`, `b1_sbabble_*` (babbling), `data/wall_predcheck` v1. Their numbers stay in FINDINGS; scripts in `scripts/_archive/`. | -- |
| `data/wall_pilot/b1/` | 24 windows of B1's own forward walks, rotated to face +x, front wall 1-6 m ahead at the last frame; rooms 400-423, original recipe; `wall_dist` per frame | `scripts/diagnostics/wall_pilot/b1_wall_pilot.py` |

Grayscale models (`cfg.grayscale`, e.g. `round1_branches_s0_gray`): the frozen encoder converts frames to luminance
(`scripts/vjepa2_encoder.py`, env `VJEPA_GRAY`, switched on automatically when such a checkpoint is loaded); embedding caches
for them live in separate `*_gray` files / dirs. Never load a colour and a grayscale model in one process (it refuses).

Deleted 2026-10-09: `rr_*` (random room size 8-26.5 m + random start, F314-F319), `rrv2_*` / `rrv3_*` (2 more random rooms per
training clip). Room draws kept in `rr_rooms.json`, `rr_v2_rooms.json`, `rr_v3_rooms.json`; re-render with
`scripts/dataset/render_random_room.py` / `render_multiversion.py` if ever needed.

## Wall data (2026-10-12 / 14): `data/wall_b1/`, `data/wall_c10/` -- NOT yet for training (gates + balance pending)

Design: `doc/DATA_PLAN.md` section 13 (fairness rule, gates). Facts: FINDINGS F332-F334, F338, F346.

| dir | body | what | made by |
|---|---|---|---|
| `wall_b1/_physics/{train,val,test}` | B1 | approach-escape episodes (`epNNNNN.npz`, 600 frames) + branches (`epNNNNN_bB_cCC.npz`: 10 shared frames + every one of the 28 commands held 3 s, far band 5 s); physics only, all 28 commands; `train/selection.json` = the stratified 14 of 28 per state that are rendered | `scripts/dataset/collect_b1_wall_data.py` (MuJoCo physics, CoppeliaSim render) |
| `wall_b1/{train,val,test}` | B1 | rendered branch files (frames 256 px, `cam_pose`); train = selected 14/28, val / test = all 28 | same, `render` |
| `wall_c10/_physics/{train,val,test}` | hexapod c10 | same layout; episodes = the live closed-loop run's replay (stored only when identical to the live run, F346); fields `room_c`, `room_phi` (virtual room per episode), `log` (controller decisions), `replay_equals_live`, `approach_group` (top-up) | `scripts/dataset/collect_c10_wall_data.py` (CoppeliaSim scene reuse) |
| `wall_c10/{train,val,test}` | c10 | rendered branch files | same, `render` |
| `wall_c10/_physics/_moved_val_number_collision/` | c10 | 2 val corner episodes whose numbers collided with test (F346); kept, not used | -- |
| `wall_c10/_physics/_moved_replay_mismatch/` | c10 | 11 top-up episodes stored in another determinism class than their live run (F346); kept, not used | -- |
| `wall_c10/_physics/{pilot,_pilot_v1,cornerfix,pilot_topup}`, `wall_c10/pilot` | c10 | pilots (controller, corner homing, approach-type placement); not used | -- |

Splits: episodes and room looks disjoint (B1 train 8000-, val 8500-, test 8800-; c10 numbers in DATA_PLAN 13). Checks:
`scripts/dataset/check_wall_data.py --body {b1,c10} [--render] [--ref_body b1]` (`results/check/wall_data_{b1,c10}/gate.txt`).
Accounting is by independent START STATES per distance band x approach group (not files); per-split reports
`collect_*_wall_data.py report`.

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
