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
| Lazy frame loading (`wm/data/frame_store.py`, memory-mapped cache in `data/_frame_cache/`, RAM guard 8 GB); arm B 0 GB frames in RAM, 0.8 s/step, bit-identical to the eager loader | |
| Eval embedding caches stamped with file size + mtime (`wm/data/emb_cache.py`); stale entries re-encoded; shared latent reads c08 heldout (`--c08`), not the archived beh12 dir | |

## Next, in order

1. **Commit** (user runs it) + rsync `data/counterfactual_walks/` (current dirs only, see `doc/DATA.md`) to the server.
2. **Round 1 (running 2026-10-03):** `scripts/run/round1_counterfactual.sh` — arm A `round1_clips_s0` (clips only, server GPU 0) vs arm B `round1_branches_s0` (clips + both bodies' branches, local); joint c10 + B1, ~30k steps each, seed 0. Evaluate both with `scripts/run/eval_suite.sh NAME joint wm/runs/NAME/best.pt`. **Decision rule, written before the results:**
   - *Training length (corrected 2026-10-03):* both arms are judged at their planned length (equal ~30k steps). The learning rate is a cosine over the planned epochs, so a finished run cannot be extended with `--resume` (`wm/train.py` refuses); val still dropping at the last epoch partly reflects the annealing. A longer arm B is a separate fresh run (new name), only if the read-out result is borderline. Arm B finished 21:41: val 9.13 / 8.78 / 8.52 over epochs 1-3.
   - *Primary (does the model see another action's future):* counterfactual read-out on the heldout branches, Pearson r mean over fwd / lat / yaw, real future and FTM-predicted future. Branches help if arm B beats arm A by ≥ 0.10 on the FTM-predicted future on both c10 and B1.
   - *Secondary (does it reach selection):* rollout from the robot's current frame, normalised score, B1 and c10; improves by ≥ 0.10 with direct not worse by > 0.05.
   - *Noise:* one seed per arm; seed-to-seed differences seen before were up to ~0.1 normalised score, so a difference < 0.10 is "no difference" until a second seed.
   - c08 (zero-shot) is reported, not used to decide.
   - **Result (2026-10-04, F308):** primary met on c10 (+0.22 / +0.19) and B1 (+0.15 / +0.10), both seeds of B vs A; secondary met on c10, not on B1 (+0.03 / +0.01). Real-future read 0.27-0.32 -> 0.72-0.83. Branches kept for all further training. Next: hexapod-only -> adapt B1; FTM yaw prediction is the weak point.
3. **Render-shift test set** + **augmentation pilot** (update `aug_pilot*.sh` to v4 paths first).
4. **Round 2:** camera vs body options (frame-conditioned Froude head / camera target), randomised or multi-version rendering, second seed, longer branches if post-switch rollout is weak.
5. **Physics closed loop** with heldout goals; then **slides once**.
6. Housekeeping: data scripts re-keyed to the current clips (2026-10-03: `branch_points_current.json`, walk plans copied to `c10_walks/_work/plans/`, shared constants in `scripts/dataset/beh24_conditions.py`); no current script reads `_superseded/`. Delete `data/counterfactual_walks/_superseded/` (~24 GB), `data/_archive_old_datasets/` and old runs only after round 1 is evaluated and the user agrees.

7. **4-leg bodies (new-body adaptation test, 12-D actions, own projector each):** feasibility done 2026-10-04 (`scripts/dataset/four_leg_feasibility.py`, `results/check/four_leg_feasibility/`). c10 commands do not reproduce c10 behaviours on any variant; 12-D CPG babbling (4 min each): hind_loss best (falls 0.84/upright-min, 90% usable, 17/24 targets within 0.05), middle_loss borderline (body drags, CoM 0.085 m), front_loss weakest. All miss backward speeds and the strongest turns. Next: babble design per the report (more backward, wider spin, both strafe signs, resets), label by measured CoM Froude; heldout test set built like c10. After hexapod-only -> B1.

8. **Claim = drive a new robot with no prior knowledge** (user 2026-10-04): c10 and joint-B1 rows are reference only. Claim rows: hexapod-only pretraining + babbling-adapted B1 (expert B1 library = upper bound, DATA_PLAN 10) and the 4-leg variants (DATA_PLAN 9, test commands from the body's own babble only).

## Open decisions / known issues

- **Truthfulness of the setup (user 2026-10-04, DEFERRED -- do not re-collect / re-render until the user decides):**
  (1) rooms scaled to each robot's camera height = resizing the world with knowledge of the robot (hides identity
  artificially, body-ID 0.55). Options: (a) keep; (b) one fixed world (identity visible via apparent scale); (c) room size
  random per clip from one wide range shared by all bodies (scene-level, rendered natively -- NOT image zoom, which would
  leak identity through resampling blur). (c) = rendering only via replay renderers, same clip count; may need more
  variety (multi-version rendering) -- measure. Small test first: a few clips per body in (c) + body-ID probe + single-frame
  speed probe (F307) + frame review with the user.
  (2)+(3) MEASURED (F309): labels + fixed lever arm explain camera planar motion R2 >= 0.99 at 1 s; sway (z / roll / pitch)
  is gait-specific, large on the hexapod per 5-frame step, ~0 after 1 s -> no camera target needed; claim scope = planar locomotion. (4) labels need the robot's height and true velocity (sim / proprioception);
  VSM labels a new body by visual odometry -- state as an assumption or test later.

- Sideways hexapod heading drift (up to ±28°): accepted; revisit if side goals are worst in the baseline eval.
- **Camera vs body motion:** LAC-WM splits z into end-effector + camera targets and conditions its motion decoder on the current frame; Egocentric VSM supervises base motion only. Round 2 tries (1) frame-conditioned Froude head (the same 3-d Froude target, one head for all bodies, reads the current frame + z; NOT the removed per-body joint-command decoder), (2) extra camera target (fwd/lat only, CoM-height scaling), vs the z-only Froude head. Decide by measurement.
- **Bounce / roll / pitch (2026-10-03, not added):** not targets of the shared Froude head — they are gait-specific (tripod vs trot), not the same behaviour across bodies; near zero when averaged over a step; the FTM's reconstruction already has to model the camera sway. Measure first after round 1: (a) single-frame frozen-encoder probe -> Froude, fit on train rooms, tested on heldout rooms (speed visible without the room?); (b) share of camera motion (`cam_pose`) left unexplained by fwd / lat / yaw. Only if both show a real gap: per-body auxiliary sway head (not shared), or the frame-conditioned option.
- **Multi-version rendering** (each clip in K rooms, one version per training sample) proposed for randomised rendering; lighting is fixed for both bodies today.
- **Branch length 20 frames (1 s) kept** (user). In 1 s the post-switch motion is still in transition (median fraction of steady target in the last 10 frames: hexapod fwd/lat/yaw 0.95 / 0.72 / 0.84, B1 0.90 / 0.91 / 0.54; direction and ordering correct, r 0.95-1.0). States 1-2 s after a switch are in no data set. Measure: multi-step rollout after the branch on heldout (up to 4 steps) and the physics closed loop for longer horizons; if weak, add 40-frame branches or multi-switch clips.
- Ego view of the bigger body changes ~2x slower at equal Froude (room scaled by height): a real cross-body difference.
- Leave-goal-out excludes only the goal clip; other windows of the same walk stay in the c10 library.
- Old scripts relying on the planner's former window default (0) will not reproduce earlier numbers.
- Every result before 2026-10-01 is pre-fix (F293 / F295 / F297 / F300 / F301): not quotable until re-measured.
- Later: gecko, continuous action sampling, temporal attention.
