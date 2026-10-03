# Status — current work tracker

> Living document: rewritten as work moves, not appended. History goes to `PROGRESS.md`, conclusions to
> `FINDINGS.md`, fixed conventions to the memory files `project_timescale_conventions` and `feedback_resource_limits`.
> Last update: 2026-10-03. Workspace cleaned: old data in `data/_archive_old_datasets/`, all pre-v4 runs in `wm/runs/_archive_old_runs/` (every one trained on pre-fix data), old caches deleted, superseded run scripts in `scripts/run/_archive/`. Data map: `doc/DATA.md`.

## Goal of this phase

Make rollout (the part that uses dynamics) work: train on counterfactual transitions (same state, different
action) for both bodies, built identically, measured with one standard evaluation on clean held-out data.
Slides are updated once, when results are in.

## Current data (v4, `data/counterfactual_walks/`; plan: `doc/DATA_PLAN.md`)

| set | hexapod | B1 | status |
|---|---|---|---|
| main clips 48 / 24 / 24 (24 behaviours; windows of one long walk per behaviour; rooms train 0-47 / val 100-123 / heldout 200-223, same for both bodies) | `c10_clips_*` | `b1_clips_*` (B1 commands tuned to the hexapod's CoM Froude, 24/24 within 0.002) | both exact (F305) |
| branches 3456 / 1728 / 1728 (3 branch points per main clip spread over gait phase x 24 commands; all 576 pairs per split) | `c10_branches_*` (bit-identical starts) | `b1_cf_*` (exact MuJoCo restore, walk command as run) | done |
| c08 test set (held-out body) | `c08_clips_heldout` (24), `c08_branches_heldout` (1728) | — | done (F306) |
| superseded, kept | `_superseded/b1_head_reference_targets/`, `_superseded/b1_camera_yawed/`, `_superseded/c10_replay_noise/`, `_superseded/b1_tuned_to_replay_targets/`, `_superseded/b1_full_precision_command/` | | |

Every file: `com_pos` (Froude at the centre of mass), `cam_pose`; branches also `segment` / `first_pair` / `froude_height`.

## Running

| job | where | notes |
|---|---|---|
| (none) | | |

## Done (2026-10-01 / 02)

| item | where |
|---|---|
| B1 pose bug (`_face_forward`) fixed in code + 569 files; v3 frames re-rendered | F293, F294 |
| Froude labels edge-correct + switch-aware (`smooth`, `segment`); timing verified (0.05 s frames); read-out window 21 | F295 |
| Velocity differencing also stops at command switches | F303 |
| Test sets overlapped training -> heldout split only; leave-goal-out | F297 |
| Hexapod turn sign resolved (`turn_sX` = left); recipe table corrected | F298 |
| B1 branch controller-gain switch fixed; all B1 restore/replay paths take settings from the recording | F299 |
| Hexapod beh24 clips were duplicate motions in 4 rooms -> v4 long walks, one room per clip | F300 |
| Froude reference = centre of mass for both bodies | F301, F302 |
| v4 branches built; B1 v4 camera yawed off the body -> corrected-mount re-render; hexapod replay noise measured | F304 |
| Review fixes: `first_pair` in every pair builder; stats over usable frames; `collect_beh24` keeps 24 conditions; eval suite B1 goals | F303 |
| `eval_suite.sh` on v4 data; read-out on the heldout physics branches (`counterfactual_readout.py`) | |
| Label tests `tests/test_froude_labels.py` 9/9 (missing data fails) | |
| Camera vs body motion checked against LAC-WM / Egocentric VSM; `cam_pose` recorded | open items |

## Next, in order

1. **Commit** (user runs it) + rsync `data/counterfactual_walks/` (current dirs only, see `doc/DATA.md`) to the server.
2. **Round 1 (server, 2 GPUs):** new script on v4 data — joint c10 + B1 pretraining on main clips only vs + both bodies' branches (equal step budget); evaluate both with `scripts/run/eval_suite.sh`.
3. **Render-shift test set** + **augmentation pilot** (update `aug_pilot*.sh` to v4 paths first).
4. **Round 2:** camera vs body options (frame-conditioned decoder / camera target), randomised or multi-version rendering, second seed, longer branches if post-switch rollout is weak.
5. **Physics closed loop** with heldout goals; then **slides once**.
6. Housekeeping: `build_branches.py` / `collect_b1_walks.py cut` / `add_cam_pose.py fill` still read their sources from `_superseded/` (branch points were keyed on those clip paths) -- re-key to the current clips before regenerating anything; `collect_beh24.py --verify` rewrite; delete `data/counterfactual_walks/_superseded/` (~24 GB) once round 1 passes.

## Open decisions / known issues

- Sideways hexapod heading drift (up to ±28°): accepted; revisit if side goals are worst in the baseline eval.
- **Camera vs body motion:** LAC-WM splits z into end-effector + camera targets and conditions its motion decoder on the current frame; Egocentric VSM supervises base motion only. Round 2 tries (1) frame-conditioned decoder, (2) extra camera target (fwd/lat only, CoM-height scaling), vs the z-only Froude head. Decide by measurement.
- **Multi-version rendering** (each clip in K rooms, one version per training sample) proposed for randomised rendering; lighting is fixed for both bodies today.
- **Branch length 20 frames (1 s) kept** (user). In 1 s the post-switch motion is still in transition (median fraction of steady target in the last 10 frames: hexapod fwd/lat/yaw 0.95 / 0.72 / 0.84, B1 0.90 / 0.91 / 0.54; direction and ordering correct, r 0.95-1.0). States 1-2 s after a switch are in no data set. Measure: multi-step rollout after the branch on heldout (up to 4 steps) and the physics closed loop for longer horizons; if weak, add 40-frame branches or multi-switch clips.
- Ego view of the bigger body changes ~2x slower at equal Froude (room scaled by height): a real cross-body difference.
- Leave-goal-out excludes only the goal clip; other windows of the same walk stay in the c10 library.
- Old scripts relying on the planner's former window default (0) will not reproduce earlier numbers.
- Every result before 2026-10-01 is pre-fix (F293 / F295 / F297 / F300 / F301): not quotable until re-measured.
- Later: gecko, continuous action sampling, temporal attention.
