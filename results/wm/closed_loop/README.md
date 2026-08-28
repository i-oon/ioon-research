# Closed-loop runs

Every run here is a `.npz` per episode holding the frames, the candidate chosen at each step, every
candidate's score, and the body trace. Score them with

```
.venv/bin/python3 scripts/diagnostics/score_closed_loop.py <dir>/*.npz --demo_dir <dataset>
```

**The `--demo_dir` matters and is not guessable from the run.** `full` is the trained body and takes
`data/beh12_hex_flat`; every other hexapod run is the held-out body and takes
`data/beh12_c08f09t09_flat`; B1 runs take `data/beh12_b1_flat`.

**Read `S.R. speed` and the median error together.** The rate is a 15% threshold and hides whether a
miss was by a point or by a factor of three.

## Current runs

Scores below are with the corrected channel selection (F108); anything quoted from before
2026-08-28 in an older FINDINGS entry was graded on forward speed even when the behaviour was a turn.

| directory | what it is | behaviour | survival | speed | median error |
|---|---|---|---|---|---|
| `hex_trained` | hexapod, **trained** body, 9 runs | 100% | 100% | **56%** | **14.3%** |
| `hex_unseen_zeroshot` | hexapod, unseen body, projector as-is, 6 runs | 83% | 100% | 0% | 41.3% |
| `hex_unseen_fewshot` | hexapod, unseen body, projector refitted -- **F95's headline** | 100% | 100% | 0% | 23.0% |
| `hex_unseen_commit1` | hexapod unseen body, `--commit 1`, 5 repeats x 3 goals | 100% | 100% | 13% | 36.2% |
| `hex_unseen_commit3` | the same at `--commit 3`, 10 runs | 100% | 100% | 50% | **14.8%** |
| `hex_unseen_turn` | hexapod on the two goals that actually turn, 10 runs | 100% | 100% | 40% | 47.3% |
| `hex_unseen_nowarm` | the `hex_unseen_commit1` configuration with **no warm start** (F110) | 93% | 100% | 20% | 58.8% |
| `hex_unseen_turn_nowarm` | real turns, no warm start, `--commit 1` | 100% | 100% | 0% | 77.3% |
| `hex_unseen_turn_nowarm_commit3` | the same at `--commit 3` -- half the switching, faster turn entry | 100% | 100% | 10% | 75.4% |
| `hex_unseen_side` | hexapod sideways after the library correction, 5 runs | 100% | 100% | 0% | 18.0% |
| `hex_unseen_side_commit1/5/10/20` | sideways commitment sweep, 5 runs each | 100/60/60/100% | 100% | 0/0/0/40% | 46.7/62.1/48.7/22.2% |
| `b1_selfgoal_commit1` | B1 in MuJoCo physics, `--commit 1`, 3 goals | 67% | 100% | 33% | 25.3% |
| `b1_selfgoal_commit3` | the same at `--commit 3` -- **the B1 configuration reported** | 67% | 100% | 33% | **20.3%** |
| `b1_selfgoal_commit5` | the same at `--commit 5` | 67% | 100% | 33% | 24.5% |
| `b1_hexgoal_arm1_frozen` | **cross-embodiment ladder** (F112): frozen world model, projector fitted only | n/a | 100% | n/a | n/a |
| `b1_hexgoal_arm2_mse_separate` | ITM+FDM adapted separately, MSE | n/a | 100% | n/a | n/a |
| `b1_hexgoal_arm3_mse_joint` | projector+FDM adapted jointly, MSE | n/a | 100% | n/a | n/a |
| `b1_hexgoal_arm4_nce_joint` | the same **+ InfoNCE** -- the reported configuration | n/a | 100% | n/a | n/a |
| `b1_hexgoal_warmturn` | the warm-start control (F109): the same goals started with a **turning** clip | n/a | 100% | n/a | n/a |
| `b1_hexgoal_warmforward` | the same goals with a **forward** warm start -- the F109 control | n/a | 100% | n/a | n/a |
| `b1_hexgoal_speedrange` | seven forward goals over a 1.72x Froude range -- the speed-tracking test (F111) | n/a | 100% | n/a | n/a |

**The cross-embodiment directories cannot be scored for speed or behaviour by this script.** Its
reference is the run's own `demo`, which for those runs is the B1 clip that supplied the warm start
and not the hexapod clip that supplied the goal. Running it anyway measures how closely the robot
kept doing what the warm start did -- which is the F109 finding, but it is not a success rate.
Grade those on behaviour-family accuracy against chance instead: forward is 67 / 84 / 71%
across the three warm-start settings against 33% chance, turning clears chance in no arm, and
sideways is 2 / 0 / 0% against 17%. The four `arm*` directories are the adaptation ladder, four goal
clips per condition; `plot_adapt_objective.py` draws them. **Speed across embodiments is measurable via Froude, which is
dimensionless** -- compute it against the *goal* clip, not the run's `demo`; `b1_hexgoal_speedrange` does
this and finds no tracking at all (F111).

## Figures

`figures/adapt_objective.png` and `.pdf`, from `scripts/figures/plot_adapt_objective.py`, which
reads the `.npz` files directly and holds no measured value as a literal.

## Videos

`video_hex_unseen`, `video_b1_selfgoal_commit3`, `video_hex_unseen_fewshot`, `video_b1_hexgoal_warmturn`. Rendered
with `sim/render/render_closed_loop.py`, which shares `channel_for` with the scorer and takes
`--goal_dir` to show the goal clip beside the robot.

**A same-robot run's warm start is the goal clip's own first ten actions**, so its score window
starts from a body the correct behaviour already put in place. The `*_nowarm` directories are the
control (F110); read them beside their warm-started twins, never on their own.

**Watch them.** Three defects in this project were found from the video and none of them from the
table: the robot leaping at the start, the camera following the robot, and the loop turning the
wrong way. The tables were internally consistent in all three cases.

## `_superseded/`

Runs kept for provenance, several because they are the evidence for a defect. Its README lists what
replaced each. **Nothing in there belongs in a new comparison.**
