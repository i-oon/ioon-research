# Stage 1, retrained on clean data

Landing zone for the retrain driven by `scripts/run/retrain_stage1.sh`. Kept apart from `stage1/`
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

## `stage1/` is not going to be retired, and should not be

An earlier version of this file said to rename `stage1/` to `stage1_superseded/` once the citations
moved. **That was wrong and is withdrawn.** The retrain regenerated five figures -- three
`action_trace_*`, `coverage_experiment.png`, `cross_loss_effect.png` -- and deliberately not the
rest. The other ~15 in `stage1/figures/` document findings on datasets that no longer exist: the
three-body leg-scale set behind the morphology axis, the 100-epoch framed runs behind F11 and F12.
Nothing here replaces them, so renaming would break eleven embeds and relabel a legitimate
historical record as an error.

**Which directory a citation should use:**

| the claim is about | cite |
|---|---|
| current Stage 1 numbers -- held-out scores, the pathway, coverage | `stage1_correct/` |
| what the contaminated data cost, old against new | both, as the pair |
| an early finding on a dataset since replaced (F4, F6, F11, F12) | `stage1/`, labelled as such |

The deck does this correctly today: its one `stage1/` figure carries the caption "from the earlier
three-body dataset" and illustrates a claim the table beside it measures on current data.
