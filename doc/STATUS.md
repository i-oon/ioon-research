# Status — current work tracker

> Living document: rewritten as work moves, not appended. History goes to `PROGRESS.md` (the previous version of this
> file is archived there under 2026-10-09), conclusions to `FINDINGS.md`, data map in `doc/DATA.md`, fixed conventions in
> memory. Last update: 2026-10-12 (wall-data DESIGN approved; exists-vs-planned split below).

## Goal (advisor, week 19; memory `project_week19_direction`)

Show where the world model is needed: a test where rollout (imagining the future from the camera) must beat direct (scoring
a command without the view) -- predictive wall / obstacle avoidance, later maze navigation. Paper: IROS, 8 pages.
Babbling / no-controller bodies are paused; candidates = the tuned ("upper bound") libraries. Rooms: the ORIGINAL setup
(sized to the body, start at the room centre), no random size; randomise the view instead (grayscale, brightness, texture,
lighting, as Egocentric VSM).

## Where we stand (2026-10-09)

| | result | finding |
|---|---|---|
| branches (same state, 24 commands) | the model reads another action's real future (r 0.3 -> 0.8-0.9) | F308 |
| direct selection | follows the goal offline and in physics on all bodies (E ~0.04) | F327 |
| rollout selection | good on many goals, fails on strong turn / side L0, same offline and in physics | F327 |
| why rollout fails | the Froude reads are shrunk toward the middle on every channel; also on steady clips; worst in unseen rooms -> z carries the room; the FTM shrinks yaw further | F329, F330 |
| gait-phase mismatch at selection | rollout fed candidate commands at another gait phase; fixed in the hexapod loop + offline (rollout 0.086 -> 0.071) | F330 |
| hexapod physics loop | gait broke at every switch; fixed (`--cpg_clock`) | F323, F327 |
| wall pilot (B1) | wall distance readable from the frozen embedding (R2 0.96) and from FTM outputs; on straight walking not better than "copy the current frame" | F328 |

## Exists now vs planned (2026-10-12)

**Exists and runs:**
- Model `round1_branches_s0_gray_imread` (+ seed 1 on the other PC, reproduces within ~0.003): base model.
- Physics loops (`scripts/run/physics_orig_*.sh`) with the documented rollout setting `GA=imagined` (F342: small gain on
  all bodies; strong-turn onset not fixed). `--calib` = rejected (F341), off by default.
- B1 wall-data GENERATOR `scripts/dataset/collect_b1_wall_data.py` (physics, render, review, stats): the data DESIGN is
  approved by the user (pilot `escape_var`, 20 episodes physics + 10 rendered). The full collection has NOT been run.
- c10 spin recipes matched to B1's spins (pilot measurement only).

**Planned, NOT built -- wall training / evaluation is NOT runnable end to end:**
- wall-data loader into training (28 commands); clearance head + imagined-clearance loss (targets 0.25-1.25 s);
- c10 wall collector (geometry-rule controller, scene-reuse branches); far-band pilot; stratified 14-of-28 train
  selection + per-split start-state / episode / look report;
- baselines: current-image selector, direct risk predictor; planner score with wall risk (must allow a temporary
  retreat); prediction check on the new data; closed loop.
- UNTESTED: whether the model predicts wall clearance / crossings at all, and any closed-loop wall result.

## To revisit later (parked 2026-10-12 by the user; wall data + retraining first)

**Weekly-update storyline redo** from "Is anything in the data inflating these results?" on, as ONE storyline on the
current setting (grayscale + view randomisation, original rooms): drop the room-size test and all random-room rows, drop
the babbling section; inflation check -> view randomisation -> shrink -> imagination term -> z transfer -> physics ->
wall. Reusable now: `results/eval/round1_branches_s0_gray(_imread)`, physics `_gray_imread(_ga)`, prewalk summaries, F333
+ seed 1. To run then: (1) single-frame speed probe on grayscale frames, seen vs new rooms
(`scripts/diagnostics/egocentric_view/single_frame_speed_probe.py`, ~30 min GPU); (2) shrink table (real vs imagined
slopes) on the grayscale model before / after the imagination term (~1 h GPU); (3) decide: retrain "hexapod-only + B1
adapted" on grayscale (~1 day GPU, other PC) or drop that z-transfer row.

## Known issue to correct (2026-10-12)

`selection_eval.py` read frames before loading the model, so OFFLINE SELECTION rollout / vision-goal scores of the
grayscale models were computed on colour frames (fixed 2026-10-12; direct, read-outs, shrink tables, physics loops not
affected). Seed 0 (`_gray`, `_gray_imread`) rerun with the fix (`results/eval/<name>/selection.txt`; old kept as
`selection_colourbug.txt`). **Seed 1 (`round1_branches_s1_gray_imread`, other PC) NOT rerun -- skipped by the user
2026-10-12; its selection numbers in F342 carry the bug and are not cited until rerun.**

## Running

Nothing (2026-10-11 ~05:00). All own CoppeliaSim instances stopped.

## Results 2026-10-11

- **F341 de-shrink calibration of the rollout read: refuted** -- strong turn from straight walking 0.158 -> 0.107, but mild
  turn, side goals and B1 worse (c10 mean 0.062 -> 0.076, B1 0.048 -> 0.063). `--calib` not used.
- **F342 goal alignment** (`--goal_align imagined`: compare with the mean goal over the labels the imagined window was trained
  on, removes the ~0.6 s offset): small gain on all bodies (c10 0.062 -> 0.058, c08 0.088 -> 0.082, B1 0.048 -> 0.044); strong
  turn onset not fixed (~0.14 vs direct ~0.03). **Kept as the rollout setting.**
- Seed 1 of `_gray_imread` (other PC) reproduces seed 0 within ~0.003 (F342 note).
- Wall data (`scripts/dataset/collect_b1_wall_data.py`, physics-only checks in `results/check/wall_data_pilot/stats_*`):
  APPROACH-ESCAPE episodes (30 s): approach a wall / corner, branch states at ~1 m and ~0.6 m (all commands held 3 s;
  labelled "no crossing under a held command", not "escape"), then a complete escape (make room -> turn ON THE SPOT
  toward open space -> walk out 2-4 s) -> next wall. B1's 24 tuned commands cannot turn away from a wall (their turns
  walk forward 0.36 m/s at <= 14 deg/s), so the wall library adds 4 SPIN commands (vx 0, wz +-0.30 / +-0.45; Froude yaw
  +-0.075 / +-0.112); the hexapod CPG matches them (turn recipe, a0 0.05, 6 cycles, spin -0.67 / -0.95 / +0.50 / +0.80;
  `scripts/diagnostics/wall_pilot/hex_spin_pilot.py`). escape28 (20 episodes): 23/23 finished escapes complete (no
  crossing, facing open space = > 4 m free and within 60 deg of away from the wall, walked out > 0.3 m); 50 branch states,
  crossing within 3 s readable from the state at 1.25 s (AUC 0.99); forward-goal utility: oracle +0.25, best fixed
  command +0.08, always back up -0.31, blind forward -0.85 (45/50 cross). Next: render 10 for the user's review; NO full
  collection until approved. Planner check needed: does its objective allow a temporary retreat (Codex).
  **Pilot APPROVED by the user 2026-10-12** (`escape_var`, review `results/check/wall_data_pilot/review_escape_var_r10/`
  incl. `allocentric/` top-down cross-check): varied turn sizes, walk-out 1-2.5 s, next wall nearby, episodes = complete
  escapes only, per-frame label SAFE / NEAR (< 0.15 m = 0.13 body lengths) / CROSSED. Next: far-band pilot, c10 pilot.
  **Plan revised 2026-10-11: `doc/DATA_PLAN.md` section 13 (rooms, accounting by start states, two runs, risk baselines).**
  **Scope (user, 2026-10-11): the wall test is for ALL bodies; B1 is only the first.** Hexapod version later: room and
  clearance in body-length units (hexapod room 8 m), c10 spin recipes ready (c08 to check), branches need the
  deterministic CoppeliaSim scene-reuse procedure (expensive physics), c10 = seen body (reference), c08 = unseen
  (test set not built from c10). Order: B1 end-to-end first, then c10, then c08.

## Results 2026-10-10 (this session, details in FINDINGS)

- F331 grayscale + photometric aug (`_gray`): reading video less room-bound; imagined turning worse.
- `_srbal_gray` (shared room looks, other PC): slightly worse than `_gray` everywhere (fewer room looks); not pursued.
- **F333 imagined-read loss (`_gray_imread`): rollout c10 +0.69 -> +0.82, c08 +0.69 -> +0.81, B1 +0.76 -> +0.87; predicted yaw
  roughly doubled; the failing turn / side goals now near direct offline.** Base model from now on.
- F335 physics with it: B1 rollout 0.062 -> 0.048 (better); hexapods not (c10 0.065 -> 0.062, c08 0.066 -> 0.088): rollout
  scores are much less certain on live loop frames than on stored clip frames.
- **F336 root cause:** a turn command's first ~0.75 s reads (real and imagined) ~0.3x its yaw; from a standing / straight
  state rollout never imagines a turn, takes a forward command and locks on. Not render, not switching frequency (F336 test).
  Fix candidates: score against the goal's onset profile (planner, no training), onset-aware labels, k = 8 imagined-read
  training; start the loop from walking (benchmark fairness).
- F332 wall feasibility pilot: a footprint TTC rule (privileged distance) avoids all; oracle only keeps more progress.
- F334 wall prediction check: NOT passed -- imagined clearance does not beat command-only; the state set had no left/right
  mirror, so the commands' safe ordering hardly depended on the state. Needs a state set where the best command depends on
  the state before any wall claim.

## Next, in order

1. **Timing step 2 (Codex; separate ablation, needs training -- plan to the user first):** z sees a 0.25 s frame pair but
   is trained toward a 1 s stride-smoothed label (mean over 1.2 s). Caveat: the 1 s window exists to remove gait sway (F295,
   `body_motion` docstring: 5-frame window -> sway dominates), so simply narrowing the label is expected to hurt. Options: (a)
   widen what z sees (pair span 0.5-1 s) and keep the label; (b) keep both and add onset-weighted loss. Same checks as F341/F342.
2. Wall: user reviews the pilot; then full collection; then prediction check v2 on the new model (F338 protocol).
3. Paper (IROS 8 pages) once a rollout-beats-direct result exists.

## Older plan (2026-10-09, kept for reference)

1. **Evaluate `round1_branches_s0_gray`** against `round1_branches_s0`: read slopes in unseen vs trained rooms (atten check,
   F330), `readout_channels.py` on the held-out branches, offline per-goal selection (direct; rollout with `--phase_align`),
   Result-1 read-out (must not drop). Pass: unseen-room slopes approach trained-room ones, rollout on turn / side goals better,
   direct not worse.
2. **Train on `srbal_*`** with the same grayscale + augmentation (only the images differ: room tied to behaviour vs not).
3. **Physics** (c10, c08, B1) with the better model; hexapod rollout with the gait-phase fix; B1 gait phase from foot contacts.
4. **Wall test:** reactive controller (walk, turn when close) collecting wall data in the shared looks, branches near walls
   (several commands from one state); wall-distance read on imagined frames; baselines direct (blind), reactive rule, current
   frame without prediction, rollout; metrics collision rate, closest distance, progress (`doc/ref/lit_vision_wm_locomotion_baselines.md`).
5. Multi-step imagination (training with an imagined-read loss through the frozen ITM / head, F329 / F253 lessons) only if
   1-2 leave rollout's reads shrunk.

## Open issues

- Rollout reads candidates' commands at index t in the B1 loop too (gait-phase mismatch); B1 phase must come from foot contacts.
- Direct reads a centred window, rollout a forward one (default `window_align`); small for steady goals.
- c08 speed 7.1 physics goal fails the ego-view check (corr 0.968 < 0.97, start pose slightly off).
- Claim scope: planar locomotion; labels from sim / proprioception (F309).
- Every result before 2026-10-01 is pre-fix (F293 / F295 / F297 / F300 / F301).
