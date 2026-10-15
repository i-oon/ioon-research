# Status — current work tracker

> Living document: rewritten as work moves, not appended. History goes to `PROGRESS.md` (previous versions archived
> there: 2026-10-09, 2026-10-14), conclusions to `FINDINGS.md`, data map in `doc/DATA.md`, fixed conventions in memory.
> Last update: 2026-10-15 09:00 (D1-D3 control and vision-goal physics done, F349; wall retrain running).

## Goal (advisor, week 19; memory `project_week19_direction`)

Show where the world model is needed: scene-dependent command selection -- a test where imagining each command's visual
consequence (rollout) must beat selecting without the view (direct), the current image alone and a direct risk
predictor. Task: walking near a visible VIRTUAL boundary (wall), for every body (B1, c10; c08 test only). Paper: IROS,
8 pages. Candidates: the discrete recorded library (24 motion commands + 4 turns on the spot); continuous search out of
scope. Rooms: original setup; view randomised (grayscale + brightness / contrast / blur / noise).

## Where we stand (2026-10-15)

| | result | finding |
|---|---|---|
| branches (same state, every command) | the model reads other actions' futures (r 0.1-0.5 -> 0.6-0.9); losses lower on every term | F308, loss table in the 15/10 deck |
| view randomisation + imagined-read term | rollout +0.82 / +0.81 / +0.87 (c10 / c08 / B1), direct unchanged; imagined yaw still ~1/3 of its size | F331, F333, F343 |
| physics closed loop | direct follows every goal on all bodies; rollout loses the strong turn on both hexapods | F335-F337, F342 |
| B1 as a NEW body (c10-only + 2.4 min) | direct transfers fully (+0.96); rollout +0.63, +0.76 with the imagined-read term in adaptation (joint +0.87) | F347 |
| shared z | well shared between hexapods; with B1 only forward in joint pretraining, more shared when B1 is adapted | F344, F347 |
| centring | read-out transfer better, selection not; not adopted | F345, F347 |
| wall data | B1 + c10, equal start states per cell and side (153 / 30 / 57), exactness gates pass | F346 |

## Exists vs planned (2026-10-15)

**Exists and runs:**
- base model `round1_branches_s0_gray_imread` (+ seed 1); c10-only runs `round1_hexonly_s0_gray_imread[_center]` and the
  adapted-with-imagination eval (other PC);
- wall data + tools: collectors, `balance_wall_data.py`, `check_wall_data.py`, `wall_predcheck.py` (D1-D3, protocol
  locked in `doc/WALL_SELECTION_PLAN.md` section 9);
- training: arm `W` (`round1_counterfactual.sh W`: arm B + wall branches, 3 epochs x 10,176 batches), step-level loss
  logging (`train_step/`, `val_step/`), `wm.adapt --lambda_imread`;
- physics loops with `--goal_source vision` (`GS=vision`), the 4-colour trace plot (`weekly_visuals.py vision_base=`);
- checks: frame / label time alignment (`check_frame_label_alignment.py`, PASS), loss before / after branches
  (`loss_before_after.py`), outcome check figure (`action_outcome_check.py`).

**Planned, not built:**
- D4 direct action-conditioned risk; planner safety term (cost = goal error + penalty on the minimum imagined clearance);
  closed-loop wall environment (B1 MuJoCo loop, hexapod CoppeliaSim loop); oracle one-step escape test; spin audit
  (4 spin commands were outside the projector's training);
- episodes-only ablation (episodes trimmed to equal frames per body); c08 wall test set.

## Running (2026-10-15 from 08:45, `results/wm/logs/queue_overnight2.sh`)

THIS PC: wall retrain `round3_wall_s0_gray_imread` (~10 h, until ~19:00) -> eval suite -> D1-D3 on the wall model (~22:00).
Done overnight: wall data final (all gates pass, cells equal); D1-D3 control (F349: B1 D1 pass / D3 fail; c10 D1 fail);
vision-goal physics (F349). OTHER PC: free.

## Next, in order

1. **Decide the c10 wall test** (F349: at 1.25 s the command alone ranks c10's outcomes): longer horizon / mirrored-corner
   subset / report as is and claim prediction on B1. Evaluation only; the retrain is unaffected.
2. Wall retrain -> D1-D3 before / after (one variable: wall data in training).
3. If D3 passes after the retrain: D4, oracle escape test, closed loop (blind direct / current-image / direct risk /
   rollout / true-future oracle; crossings and progress).
4. Separate experiments afterwards (one variable each): stride-matched Froude labels (hexapod yaw sway); blur strength
   (imagined yaw); `imread_k` 5 on wall data; room-per-behaviour test (48 rooms with mixed behaviours); longer training
   if the curves have not flattened; recurrent context (WMP / QWM-like) as future work.
5. Paper (IROS) once Gate 2 has a result.

## Later, if there is time (user 2026-10-14)

Move everything to MuJoCo (physics + rendering) with the `red_mirror` hexapod (`brain-web-mujoco/public/assets/red_mirror/`):
exact save / restore removes the determinism classes and the live-vs-replay re-queue. Pilot first (~1 week); until then
c10 / c08 (CoppeliaSim) stay the paper's bodies.

## Open issues

- Imagined yaw is the weakest channel on every body (F333, F347).
- Hexapod Froude yaw labels keep gait sway: c10 stride ~0.55 s vs the 1 s window (`BODY_WINDOW_S` assumes 0.95 s).
- The rollout window is centred ~0.6 s after the goal time it is compared with (F339 b): fine for steady goals, to fix
  before time-varying goals.
- c08 speed 7.1 physics goal fails the ego-view check (corr 0.968 < 0.97), so it is not run.
- Claim scope: planar locomotion; labels from simulation; B1 walks with its trained walking policy (commands, not joints).
- Every result before 2026-10-01 is pre-fix (F293 / F295 / F297 / F300 / F301).
