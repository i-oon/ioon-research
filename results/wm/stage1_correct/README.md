# Stage 1, retrained on clean data

Landing zone for the retrain driven by `scripts/retrain_stage1.sh`. Kept apart from `stage1/`
because **the runs reuse the same names** -- `m3d_cross`, `m3d_bracketed`, `tib_cross`,
`tib_ctrl`, `bracket_cross` -- so every output filename collides exactly with its predecessor.
One shared directory would silently overwrite the old numbers, and those are needed (see below).

## Why the two are not comparable run-for-run

`retrain_stage1.sh` states both deviations. They are deliberate, and they mean a difference
between old and new is *not* attributable to one cause:

- **`action_lag 0` -> `1`.** The originals were trained before `action_lag` existed, at the
  setting where the collector's frame ordering leaks the answer into the decoder's own input
  (F29). The retrain uses the corrected semantics. So the `m3d` pair differs in two things.
- **Only bodies and clips that walk.** The originals trained on `c10f10t06` and `c06f10t06`,
  which fail 30/30 clips each -- 40% of the training data was a body veering 0.36-0.43 m off
  course (F42). `tib_*` and `bracket_*` also *held out* `c10f10t06`, so slides 8 and 9 measured
  extrapolation onto a body that does not walk straight. The retrain drops them and holds out
  `c10f10t08` instead.

## Why `stage1/` is kept rather than replaced

The old checkpoints survive in `wm/runs_original/` and their outputs in `stage1/`. Together they
are the only direct measurement of **what the contaminated data cost**, on the one comparison
where both sides exist. That is not reproducible once either side is gone.

`wm/runs_original/README.md` carries the two caveats that must be quoted whenever the comparison
is used, including that scores on *different* held-out bodies are not comparable -- what can be
compared is each model against the *same* body, which means scoring the recovered checkpoints on
`c10f10t08` too. Changing a test body costs nothing, since it does not change the weights.

## When this is verified

Point `FINDINGS.md` and `report/update_slide.md` at the figures here, then rename `stage1/` to
`stage1_superseded/` in the same pass so nothing keeps citing the old numbers by accident.
Until that happens the deck still cites `stage1/`, which is correct -- those are the numbers the
current slides were written from.
