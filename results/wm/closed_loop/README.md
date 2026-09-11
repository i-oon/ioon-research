# Closed-loop runs

**STALE WARNING (2026-09-11): the "Current runs" table below documents directories that no
longer exist on disk** (`hex_trained`, `hex_unseen_zeroshot`, `b1_selfgoal_commit1`,
`b1_hexgoal_arm1_frozen`, etc. -- none of these are present). What IS actually on disk right now
and NOT described anywhere in this file:

- `direct_froude/` -- the A/B/C/D mode-comparison grid, cited directly in FINDINGS.md (F187/F188).
  **Do not touch.**
- `b1_hexgoal_mse_s0_c*_hexapod_ep*/`, `b1_hexgoal_nce_s0_c*_hexapod_ep*/`,
  `b1_mse_s0_c1_b1_ep*/`, `b1_nce_s0_c1_b1_ep*/` (60 directories, dated 2026-08-30) -- raw
  per-episode evidence for the MSE-vs-InfoNCE stage-3 adaptation comparison documented in
  `PROGRESS.md` (non-overlapping forward-progress ranges, MSE best 38% vs InfoNCE worst 71%;
  turning stays at/below chance for both -- F93's headline "the objective, not the architecture,
  is what makes cross-embodiment transfer work"). **Verified against PROGRESS.md/FINDINGS.md
  (F93/F107/F110/F134) 2026-09-11 -- the numbers are captured in prose there, so these raw files
  are no longer needed for the current pipeline. Moved to `_archive/` the same day.** InfoNCE
  itself is not gone -- it is `wm.adapt3` (stage 3) in the current `wm.finetune_new_body`
  pipeline, gated automatically by stage 2's rollout-gap ratio, and legitimately inactive for
  babble-CPG actions (built for a PPO policy's one-to-many collapse, which babble doesn't have --
  see F183, and F134 for why forcing it on cross-body would cost more than it buys).
- `b1_babble_v2/` -- the closed-loop render from the 2026-09-11 babble-margin experiment (see
  FINDINGS.md F194's v2 follow-up).
- A handful of loose root-level `.npz` files (`a1_diagnostic*`, `f142_*`, `f144_*`,
  `rl_loop_isolation_test*`, `ppo_p0_*`, `r0_regrounding_curve*`, `v5_twohot_closed_loop.npz`,
  `ondist_disentangle.npz`) and `plan_without_library/`, `f142_video/` -- likely evidence for the
  Dreamer/RL-wall investigation (F179 area) by naming convention (symlog, mc_check, hard_target,
  twohot are all Dreamer-critic terms), but **not verified against FINDINGS.md citations yet --
  treat as unconfirmed, do not archive or trust the table below for these.**
- `_archive/` -- 2026-09-11 confirmed-safe-to-remove-from-top-level outputs: bug-hunting dirs
  whose findings are already in F193, one abandoned render with a fabricated goal clip, and the
  60 MSE/NCE directories above (verified against PROGRESS.md/FINDINGS.md before moving, not
  guessed).

**The table below is kept for its per-run methodology notes (what `--demo_dir` means, how to read
S.R. vs median error) even though its specific directory names are gone -- read it for the
concepts, verify any specific path against `ls` before trusting it points to something real.**

Every run here is a `.npz` per episode holding the frames, the candidate chosen at each step, every
candidate's score, and the body trace. Score them with

```
.venv/bin/python3 scripts/diagnostics/score_closed_loop.py <dir>/*.npz --demo_dir <dataset>
```

**The `--demo_dir` matters and is not guessable from the run.** `full` is the trained body and takes
`data/allocentric/beh12_c10f10t10_flat`; every other hexapod run is the held-out body and takes
`data/allocentric/beh12_c08f09t09_flat`; B1 runs take `data/allocentric/beh12_b1_flat`.

**Read `S.R. speed` and the median error together.** The rate is a 15% threshold and hides whether a
miss was by a point or by a factor of three.

## Current runs

Scores below are with the corrected channel selection (F98); anything quoted from before
2026-08-28 in an older FINDINGS entry was graded on forward speed even when the behaviour was a turn.

| directory | what it is | behaviour | survival | speed | median error |
|---|---|---|---|---|---|
| `hex_trained` | hexapod, **trained** body, 9 runs | 100% | 100% | **56%** | **14.3%** |
| `hex_unseen_zeroshot` | hexapod, unseen body, projector as-is, 6 runs | 83% | 100% | 0% | 41.3% |
| `hex_unseen_fewshot` | hexapod, unseen body, projector refitted -- **F85's headline** | 100% | 100% | 0% | 23.0% |
| `hex_unseen_commit1` | hexapod unseen body, `--commit 1`, 5 repeats x 3 goals | 100% | 100% | 13% | 36.2% |
| `hex_unseen_commit3` | the same at `--commit 3`, 10 runs | 100% | 100% | 50% | **14.8%** |
| `hex_unseen_turn` | hexapod on the two goals that actually turn, 10 runs | 100% | 100% | 40% | 47.3% |
| `hex_unseen_nowarm` | the `hex_unseen_commit1` configuration with **no warm start** (F101) | 93% | 100% | 20% | 58.8% |
| `hex_unseen_turn_nowarm` | real turns, no warm start, `--commit 1` | 100% | 100% | 0% | 77.3% |
| `hex_unseen_turn_nowarm_commit3` | the same at `--commit 3` -- half the switching, faster turn entry | 100% | 100% | 10% | 75.4% |
| `hex_unseen_side` | hexapod sideways after the library correction, 5 runs | 100% | 100% | 0% | 18.0% |
| `hex_unseen_side_commit1/5/10/20` | sideways commitment sweep, 5 runs each | 100/60/60/100% | 100% | 0/0/0/40% | 46.7/62.1/48.7/22.2% |
| `b1_selfgoal_commit1` | B1 in MuJoCo physics, `--commit 1`, 3 goals | 67% | 100% | 33% | 25.3% |
| `b1_selfgoal_commit3` | the same at `--commit 3` -- **the B1 configuration reported** | 67% | 100% | 33% | **20.3%** |
| `b1_selfgoal_commit5` | the same at `--commit 5` | 67% | 100% | 33% | 24.5% |
| `b1_hexgoal_arm1_frozen` | **cross-embodiment ladder** (F103): frozen world model, projector fitted only | n/a | 100% | n/a | n/a |
| `b1_hexgoal_arm2_mse_separate` | ITM+FDM adapted separately, MSE | n/a | 100% | n/a | n/a |
| `b1_hexgoal_arm3_mse_joint` | projector+FDM adapted jointly, MSE | n/a | 100% | n/a | n/a |
| `b1_hexgoal_arm4_nce_joint` | the same **+ InfoNCE** -- the reported configuration | n/a | 100% | n/a | n/a |
| `b1_hexgoal_warmturn` | the warm-start control (F100): the same goals started with a **turning** clip | n/a | 100% | n/a | n/a |
| `b1_hexgoal_warmforward` | the same goals with a **forward** warm start -- the F100 control | n/a | 100% | n/a | n/a |
| `b1_hexgoal_speedrange` | seven forward goals over a 1.72x Froude range -- the speed-tracking test (F102) | n/a | 100% | n/a | n/a |

**The cross-embodiment directories cannot be scored for speed or behaviour by this script.** Its
reference is the run's own `demo`, which for those runs is the B1 clip that supplied the warm start
and not the hexapod clip that supplied the goal. Running it anyway measures how closely the robot
kept doing what the warm start did -- which is the F100 finding, but it is not a success rate.
Grade those on behaviour-family accuracy against chance instead: forward is 67 / 84 / 71%
across the three warm-start settings against 33% chance, turning clears chance in no arm, and
sideways is 2 / 0 / 0% against 17%. The four `arm*` directories are the adaptation ladder, four goal
clips per condition; `plot_adapt_objective.py` draws them. **Speed across embodiments is measurable via Froude, which is
dimensionless** -- compute it against the *goal* clip, not the run's `demo`; `b1_hexgoal_speedrange` does
this and finds no tracking at all (F102).

## Figures

`figures/adapt_objective.png` and `.pdf`, from `scripts/figures/plot_adapt_objective.py`, which
reads the `.npz` files directly and holds no measured value as a literal.

## Videos

`video_hex_unseen`, `video_b1_selfgoal_commit3`, `video_hex_unseen_fewshot`, `video_b1_hexgoal_warmturn`. Rendered
with `sim/render/render_closed_loop.py`, which shares `channel_for` with the scorer and takes
`--goal_dir` to show the goal clip beside the robot.

**A same-robot run's warm start is the goal clip's own first ten actions**, so its score window
starts from a body the correct behaviour already put in place. The `*_nowarm` directories are the
control (F101); read them beside their warm-started twins, never on their own.

**Watch them.** Three defects in this project were found from the video and none of them from the
table: the robot leaping at the start, the camera following the robot, and the loop turning the
wrong way. The tables were internally consistent in all three cases.

## `_superseded/`

Runs kept for provenance, several because they are the evidence for a defect. Its README lists what
replaced each. **Nothing in there belongs in a new comparison.**
