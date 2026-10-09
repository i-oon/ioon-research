# Status — current work tracker

> Living document: rewritten as work moves, not appended. History goes to `PROGRESS.md` (the previous version of this
> file is archived there under 2026-10-09), conclusions to `FINDINGS.md`, data map in `doc/DATA.md`, fixed conventions in
> memory. Last update: 2026-10-09 (clean-up: random-size-room data deleted, dropped scripts in `scripts/run/_archive/`).

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

## Running

| job | where | what / question |
|---|---|---|
| `round1_branches_s0_gray_imread` (gpu_guard) | local GPU, from 2026-10-09 ~14:00, ~10-11 h + eval | = `round1_branches_s0_gray` (control, F331) + imagined-read loss (`--lambda_imread 1.0 --imread_k 4`): FTM rolled 4 steps with real z, Froude read through the FROZEN ITM + head matched to each step's label, gradient to the FTM only; steps past the clip end masked (same 81,408 pairs). Smoke (2 short epochs, tiny data, no control): imagined steps 2-4 error 0.58 / 0.68 / 0.85 -> 0.43 / 0.54 / 0.64, slope fwd ~0.5 -> 0.85, yaw ~0.4 -> 0.6. Question: does the imagined yaw recover (F331 0.31 / 0.23) and rollout on turn goals improve? |
| `round1_branches_s0_srbal_gray` | OTHER PC, NOT STARTED yet (2026-10-09) | arm S of `round1_counterfactual.sh`: grayscale + aug on the shared-room renders `srbal_*` (room not tied to behaviour) vs `round1_branches_s0_gray`. Bring back `best.pt`, evaluate here. |

## Next, in order

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
