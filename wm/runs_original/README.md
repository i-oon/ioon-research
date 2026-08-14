# The original Stage 1 checkpoints, recovered

Recovered 2026-08-14 from the desktop trash, where they had been since `wm/runs` was deleted.
`best.pt` and `summary/` only; the periodic `epoch###.pt` snapshots were left behind to save disk.

**These are not the models to measure from.** They are what produced every number currently in
`report/update_slide.md`, and they trained on the contaminated data:

| run | trained on | held out | the problem |
|---|---|---|---|
| `m3d_cross` | 5 bodies in `ik_walk_8body` | `c08f09t09` | `c10f10t06` and `c06f10t06` fail 30/30 clips — 40% of training veering |
| `m3d_bracketed` | same 5 | `c08f09t09` | same |
| `tib_cross` | 4 bodies | `c10f10t06` | the held-out body does not walk straight |
| `bracket_cross` | 7 bodies in `ik_walk_bracket` | `c10f10t06` | same |

`tib_ctrl` is absent because it was never trained. Slide 3 names it as `tib_cross`'s control; that
claim only becomes true with the retrain.

The configurations read off these checkpoints match `results/wm/RUNS.md` exactly, which confirms
`FINDINGS.md` F42 from the weights rather than from the collection logs.

## Why keep them

The clean retrain (`scripts/retrain_stage1.sh`) is not a restoration — it changes the training set.
So old against new is a direct measurement of **what the contaminated data cost**, on the one
comparison where both sides exist. That is worth more than either number alone, and it is not
reproducible once these are gone.

Two things make the comparison inexact, and both have to be said whenever it is quoted:

- `m3d_cross` and `m3d_bracketed` here ran at the legacy `action_lag` 0, the setting where the
  collector's frame ordering leaks the answer into the decoder's input (F29). The retrain uses 1.
  So the m3d pair differs in two things, not one.
- `tib_cross` and `bracket_cross` here held out `c10f10t06`. The retrain holds out `c10f10t08`.
  Scores on different test bodies are not comparable; what can be compared is each model against
  the *same* held-out body, which means scoring these recovered checkpoints on `c10f10t08` too.

That second one costs nothing: changing a test body does not change the weights.

  .venv/bin/python3 scripts/diagnostics/score_body.py --ckpt wm/runs_original/tib_cross/best.pt \
      --data_dir data/ik_walk_decoupled --bodies c10f10t08
