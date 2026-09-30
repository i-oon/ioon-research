# Stage 1 findings: why cross-morphology transfer fails, and what fixes part of it

> **Role**: What is true, with the numbers.
>
> **Not append-only.** A later result that bounds or corrects an earlier finding is written INTO
> that finding, and a finding whose topic duplicates an existing one is merged rather than given a
> new number. Chronology lives in `PROGRESS.md`; open questions live in `OPEN_QUESTION.md`.

**The short version.** A frozen video encoder carries robot morphology in a linearly decodable
form that generalises to unseen bodies (F17). The world model trained on top of it ignores that
and identifies the body from an 11-percent component of its own latent instead (F16), which is a
lookup and does not extend past the bodies it saw. Four decoder-side interventions fail to change
this (F4, F18, F19) and more training bodies helps only inside their hull (F14, F15).

**What fixes it is changing the question, not the architecture.** Decoding one body's latent
against another body's frame, supervised by that body's command, makes the lookup wrong by
construction. Held-out error improves 23-26%, the decoder switches to reading the body from the
frame (frame ablation 10.7x-54x, crossed swap test follows the frame to within 0.01 deg), copying
stops, and performance no longer decays with training (F21).

Underneath: **the reconstruction loss, meant to make the latent an action, contributes 3-7% of the
forward model's accuracy while taking 99% of the gradient** (F20). The latent is shaped almost
entirely by the motion loss (1% of the training signal), and a body code is the cheapest thing
that satisfies it.

Two datasets: F1-F12 use `data/ik_walk_100_framed` (two training bodies, one parameter differs).
F13 onward use `data/allocentric/fwd_hex8body` (five training bodies, three parameters differ).
`data/ik_walk_100_framed`: 100 expert episodes x 3 leg-length bodies x 66 frames, 19,800 frames, 0
clipped. Architecture: frozen V-JEPA2 ViT-g/16 encoder, ITM -> `z` in R^64, FTM, Motion Decoder ->
18-D joint command. Training bodies `long` (leg scale 1.0) and `short` (0.5); `medium` (0.75) held
out unless stated.

## How to read the numbers

| Quantity | Unit | Reference point |
|---|---|---|
| joint error | degrees per joint, RMSE | the walking signal itself has 12.6 deg std on the held-out body |
| motion MSE | squared standardised action | 1.0 = predicting the training-set mean, on the training bodies only |
| axis position | dimensionless, 0 to 1 | 0 = identical to `long`, 1 = identical to `short` |
| leg scale | dimensionless | 1.0 = base body. Two-body set: `medium` 0.75, `short` 0.5. Eight-body set: per segment, e.g. `c08f09t09` is coxa 0.8, femur 0.9, tibia 0.9 |
| probe accuracy | fraction | chance is 1/(number of training bodies): 0.500 for two, 0.200 for five |
| gradient steps | steps | 1,543 per epoch at batch size 8 |

Joint names are leg-major: six legs (FL ML HL FR MR HR) x three joints.

| Joint | Anatomy | Function |
|---|---|---|
| TC | thorax-coxa | swings the leg fore and aft |
| CF | coxa-femur | lifts the leg |
| FT | femur-tibia | extends the leg |

Note: an F-number renumbering occurred 2026-09-10 (old F-numbers before that date do not match
this file) and a consolidation pass 2026-09-07 merged debugging-step-artefact clusters into single
home entries; both are historical and omitted here — see git history of FINDINGS.md if needed.

## What works

---

### F1. The model reads body identity from pixels and applies the right joint offsets, with no morphology label

Reconstruction RMSE in degrees per joint on bodies seen in training (`wm/runs/stage1_100ep_framed_runB/epoch020.pt`):

| Body | TC | CF | FT |
|---|---|---|---|
| long | 0.75 | 0.51 | 0.46 |
| short | 0.71 | 0.53 | 1.14 |

The two bodies' mean joint angles differ by 33.8 deg (CF) and 50.1 deg (FT); the model places each within 0.03-0.06 deg of the correct one, despite morphology appearing nowhere in the input or loss.

---

### F2. The latent action `z` is doing real work

Zeroing `z` costs a factor of 3-4x on the held-out body across training (`heldout/motion_zero_z` divided by `heldout/motion`, 2,600 pairs per point). The decoder does not read the joint command off the current frame alone.

---

### F3. Phase is recovered from video essentially exactly

TC on the held-out body scores 1.35 deg RMSE against a signal of 17.1 deg std. The model reliably tracks gait-cycle phase.

---

### F4. Transfer to an unseen body covers about half the required distance; the loss is stage-localised, not a capacity problem, and not fixed by a linear readout outside the training bracket

Merges the original investigative arc: where the morphology signal is lost, whether a lower-capacity readout recovers it, whether that survives physics replay, and whether it survives extrapolation.

**1. Where the signal is lost** (fold 1, held out `medium`, bracketed between `long`/`short`). Axis position (0=long, 1=short), 12 clips/body:

| Representation | Position |
|---|---|
| frozen V-JEPA2 embedding `e_t` | 0.499 |
| learned latent `z` | 0.335 |
| decoder output, CF/FT | 0.188 / 0.180 |
| correct answer | 0.357 (CF), 0.304 (FT) |

`e_t` matches the leg-scale prediction (0.5); the decoder recovers only 53% of the needed correction. Held-out errors: TC 1.35, CF 11.34, FT 15.14 deg. Quadrupling data (195->780 transitions/body) changes figures by <0.04 — not a small-sample artefact. The leg-scale-to-joint-offset map is not linear (a 0.75-scale body sits at 0.357, not 0.5); the ITM absorbs that curvature and the trained decoder overrides it, pulling 0.335 back down to 0.151.

**2. Lower-capacity readout (ridge probe on frozen `z`, fit on the 2 training bodies):**

| Predictor, RMSE deg/joint | long | short | medium (held out) |
|---|---|---|---|
| Motion Decoder | 0.63 | 0.92 | 11.04 |
| ridge probe on `z` | 3.56 | 3.79 | 5.13 |
| mean of two training bodies | -- | -- | 6.68 |

Decoder is 6.6x better in-distribution, 2.2x worse OOD. Since a linear map can only pass a latent's axis position through unchanged, this relocates credit to the ITM (places held-out body at 0.335 of correct 0.357, 94% of the way) — the trained decoder overrides a largely-correct signal. Caveat: bodies are uniformly scaled (near-affine task, favors linear readout), and training end-to-end with a smaller head (`head_linear`, 272,914->10,258 params) does NOT reproduce the probe's advantage — it's 1.4-2.1x worse than the mlp head in every epoch, so capacity reduction only helps post hoc on an already-trained frozen `z`, not as a training-time fix. Read narrowly: a high-capacity readout can distort a latent that was already close to correct, and a linear one cannot — "less capacity generalises better" is not established as a general rule here.

Separating capacity from input access, all fit on training bodies only:

| Input | Readout | trained, deg | medium, deg | degradation |
|---|---|---|---|---|
| `z` only | linear | 3.58 | 4.96 | 1.4x |
| `z` only | MLP 256-256 | 0.77 | 5.48 | 7.1x |
| `e_t` only | linear | 1.98 | 8.21 | 4.1x |
| `e_t`+`z` | linear | 0.89 | 4.88 | 5.5x |
| `e_t`+`z` | Motion Decoder | 0.71 | 10.94 | 15.4x |

Capacity is the dominant term (gap widens with capacity). `z` alone (4.96) and `e_t`+`z` (4.88) land in the same place, so the ITM already distilled what a well-conditioned readout needs.

**3. Closed-loop physics replay** (held-out `medium`, 3 clips, `sim/render/render_wm_prediction.py`):

| Driven by | mean forward, m | mean abs heading err | body height, m | tripod score |
|---|---|---|---|---|
| IK ground truth | 0.639 | 4.4 deg | 0.111 | -0.40/-0.35/-0.38 |
| Motion Decoder | 0.156 (24%) | 42.5 deg | 0.089 | +0.16/+0.16/+0.25 |
| ridge probe on `z` | 0.480 (75%) | 37.6 deg | 0.111 | -0.34/-0.17/-0.46 |

Tripod score is the correlation between the two tripod groups' contact counts: negative means the groups alternate correctly, positive means they've stopped coordinating. Neither predictor is a usable controller (both veer 38-43 deg vs ground truth's 4.4), but the decoder loses the alternating tripod gait (positive score) and sinks in body height; the probe keeps the gait (negative score, correct height). Decoder commands swing 1.9x/1.7x past the body's own joint range (up to 40 deg excursion — F5's `gain`=0.30 made physical: over-amplified distal joints drive the leg into a pose the body cannot hold); the probe stays closer to range. This range-excursion measure needs only the body's own joint limits, not matched expert data, so it carries over to bodies/embodiments with no ground truth. Single clips mislead — report all clips (e.g. decoder's heading errors across 3 clips are +25.2, -37.5, +64.8 deg; the "best" single clip is not representative).

Videos: `results/wm/replay/replay_*.mp4`. Reproduce: `.venv/bin/python3 scripts/diagnostics/morphology_axis.py --ckpt wm/runs/<run>/epoch020.pt`, `scripts/diagnostics/wm_gait_report.py`.

**4. Extrapolation (fold 2, held-out `short`, outside training range [long, medium]):**

| Stage | fold 1, medium (bracketed) | fold 2, short (outside), ep 6 | fold 2, ep 20 |
|---|---|---|---|
| frozen `e_t` | 0.499 | 1.507 | 1.507 |
| latent `z` | 0.335 | 1.202 | 1.098 |
| decoder output | 0.188 | 1.052 | 1.026 |
| correct | 0.357 | 2.803 | 2.803 |

Both ITM and decoder pull the unseen body back toward the nearest training body, harder with more training. Diagnosis differs by fold: bracketed, only the decoder fails (ITM is 94% correct); outside the range, the frozen encoder is already the largest gap (54% of required distance), and ITM/decoder make it worse. Linear readout only helps where the representation is already right: probe advantage is 2.2x (fold 1, medium) vs 1.1x (fold 2, short) over the decoder; neither fold-2 predictor is close to correct.

**Bottom line: generalisation comes from coverage, not from the model or the readout.** A bracketed body recovers to 94% at the latent; an out-of-range body is pulled to the nearest training body at every stage. No linear readout, capacity reduction, or physics replay substitutes for training sets that bracket target bodies.

---

### F5. Two training bodies cannot define a curve

The motion loss sees only two points ("looks like long -> long's offsets", "looks like short -> short's offsets"); every function through those two points has identical loss, so nothing constrains the space between them, and an in-between input falls toward the nearer plateau.

Evidence: standardising loss by within-body spread instead of pooled spread (F9) raises CF/FT weight from 0.12 to 0.95 of TC's, and the model then moves along the morphology axis but without control:

| Run | axis CF | axis FT | all-joint RMSE, deg |
|---|---|---|---|
| pooled std, epoch 6 | 0.11 | 0.10 | 8.99 |
| pooled std, epoch 20 | 0.15 | 0.14 | 10.95 |
| within-body std, epoch 6 | 0.25 | 0.24 | 9.87 |
| within-body std, epoch 18 | 0.61 | 0.59 | 17.30 |
| correct answer | 0.36 | 0.30 | 0 |

The corrected weighting swings from undershoot (0.11) to overshoot (0.61) as training proceeds — it drifts through the right answer rather than converging on it, since the objective doesn't mark where that answer is.

---

### F6. Trivial baselines beat the model on this task

Held-out `medium`, RMSE in degrees per joint:

| Predictor | TC | CF | FT | all |
|---|---|---|---|---|
| mean of the two training bodies' commands | 0.04 | 5.33 | 10.26 | 6.68 |
| model, epoch 6 | 2.02 | 9.64 | 12.05 | 8.99 |
| model, epoch 20 | 1.35 | 11.34 | 15.14 | 10.95 |

Held-out `short` (fold 2, `--train_morphs long medium`), motion MSE in run's own units:

| Predictor | MSE |
|---|---|
| linear extrapolation, `medium + (medium - long)` | 1.91 |
| copy `medium` ground truth | 6.96 |
| model, epochs 1-28 | 6.93-7.07 |
| mean of the two training bodies | 10.77 |
| predict the training mean | 12.54 |

On fold 2 the model is indistinguishable from copying the nearest training body; a linear extrapolation using only leg-length ordering beats it 3.7x. Held-out score does not move across 28 epochs while validation improves 8x.

Caveat: baselines use time-aligned ground-truth commands the model doesn't receive; F3 shows the model recovers phase to 1.35 deg so the comparison is meaningful but not fully like-for-like.

---

### F7. Between-body differences on this dataset are 92-99 percent constant offset

IK retargeting drives every body along one shared Cartesian foot trajectory, scaled to fit, so resulting joint commands are near-affine transforms of each other:

| Pair | RMSE, deg | after removing per-joint constant | fraction that is offset |
|---|---|---|---|
| long vs medium, CF | 11.86 | 1.10 | 99% |
| long vs medium, FT | 14.88 | 4.08 | 92% |
| short vs medium, CF | 22.08 | 5.88 | 93% |
| long vs medium, TC | 0.59 | 0.23 | 84% |

This is a property of data generation, not the world model, and explains why the F6 averaging baseline is strong. The relationship is nonlinear though: `medium` (leg scale 0.75) sits at 0.5 on the scale axis but 0.30-0.36 on the joint-offset axis, so no linear baseline can be right about it — a model reading appearance correctly would beat every F6 baseline, but two training bodies can only express a straight line.

---

### F8. Aggregate error hides which joints transferred

The 18-joint average on the held-out body reads 0.208 (standardised units), which looks healthy. Split by joint type, against each joint's own clip mean:

| Joint type | MSE | constant baseline | verdict |
|---|---|---|---|
| TC | 0.006 | 1.001 | 164x better than a constant |
| CF | 0.382 | 0.121 | 3x worse than a constant |
| FT | 0.236 | 0.211 | no better than a constant |

TC scores near zero and drags the mean down; report `motion_mse_per_joint_type` from `wm.evaluate`, not the average. TC is also the wrong joint to celebrate — the three bodies' TC commands differ by only 0.58-1.17 deg, so there's almost nothing to transfer there. Cross-morphology claims rest on CF and FT (the joints setting foot height/reach).

---

### F9. The standardisation scale silently reweights joints

Standardising the motion target by pooled spread across training bodies counts the posture gap between bodies as signal amplitude. Measured on `long` + `short`:

| Joint type | pooled std, deg | within-body std, deg | weight in the loss |
|---|---|---|---|
| TC | 17.1 | 17.1 | 1.00 |
| CF | 18.4 | 7.9 | 0.12 -> 0.95 |
| FT | 31.4 | 20.7 | 0.11 -> 0.84 |

TC kept full weight since all three bodies move it equally; the joints that actually differ between bodies (the point of the experiment) received an eighth of the gradient. Fixed by `within_body_std` in `wm/config.py`, on by default. Correcting this did not improve held-out RMSE (F5) — not the bottleneck — but is the correct scaling and changes model behavior on the morphology axis.

---

### F10. Validation on held-out episodes cannot detect cross-body failure, and held-out scores need error bars that in-distribution scores do not

Merges the original F10 x3. Motion MSE is standardised by training-body statistics, so "predicting the mean costs 1.0" holds only for those bodies: on held-out `medium`, the trivial predictor costs 0.495, and that body's own mean costs 0.444. Claims like "N times better than no skill" must use the held-out body's own baseline; `wm.evaluate` reports both as `predict_training_mean` and `predict_this_body_mean`.

Two runs, identical config/seed 0, different GPU, 3,086 to 30,860 gradient steps:

| | run A | run B |
|---|---|---|
| validation motion, unseen episodes of training bodies | 0.0118 -> 0.0011 (11x better) | 0.0125 -> 0.0012 (10x better) |
| held-out body | 0.295 -> 0.220 | 0.140 -> 0.203 |

90% of compute buys an order of magnitude on validation and nothing measurable on the held-out body — a new body is a different distribution, not a held-out sample. Fold 2 shows the same shape (validation 8x better, held-out flat across 28 epochs).

Run-to-run spread from floating point alone (same config/seed, different GPU), median/worst ratio between runs:

| Metric | median ratio | worst |
|---|---|---|
| reconstruction, train/val | 1.003 | 1.005 |
| motion, train/val | 1.14-1.18 | 1.38 |
| held-out body | 1.247 | 2.103 |

Floating-point rounding alone can move held-out performance 2x while models stay within 0.3% in-distribution — the numerical signature of F5 (extrapolation is underdetermined). Single-run held-out numbers are not interpretable.

---

### F11. More episodes of the same bodies does not help

| Run | Episodes | Steps | Held-out MSE |
|---|---|---|---|
| stage1_6ep_clipped | 6 | 9,750 | 0.166 |
| stage1_100ep_clean | 100 | 9,500 | 0.179 |
| stage1_100ep_clipped | 100 | 30,880 | 0.422 |

A 16x increase in episodes of the same two bodies changes nothing. Data budget spent on episodes is wasted; spend it on bodies.

---

### F12. Peak transfer arrives in the first tenth of training

Re-scoring every snapshot on 2,600 identical cached held-out pairs (`wm.sweep_checkpoints`), transfer peaks between 3,086 and 9,258 of 30,860 gradient steps and does not improve after. Selecting a checkpoint on this curve would leak the test body — it's for reporting compute cost, not choosing a model.

What more bodies fix (below): measured with two training bodies differing along one axis until now. This uses `data/allocentric/fwd_hex8body`: nine bodies scaling coxa/femur/tibia independently (`sim/scene/make_leg_morphology.py`), 30 clips each, 0% edge-clipped frames. Two bodies dropped (both femur 0.6 + tibia 1.0 stumble, head height 0.03m vs 0.111m for the rest). Five bodies train; `c08f09t09` (0.8,0.9,0.9) held out inside their convex hull; `c06f06t06` held out in a second run, outside it.

---

### F13. The morphology space is three parameters but two dimensions

Scaling the coxa barely moves joint commands:

| Change, other segments fixed | Command change |
|---|---|
| coxa 1.0 to 0.6 | 0.73 deg |
| tibia 1.0 to 0.6 | 28.63 deg |

SVD of how the five training bodies deviate from their mean: 82.4% to first direction (correlates with tibia, -0.93), 17.5% to second (correlates with femur, -1.00), 0.0% to remaining three. The coxa sits against the body so shortening it barely moves the foot, leaving little for IK to compensate.

Reconstructing the held-out body from a k-dimensional basis of the training bodies:

| k | RMSE deg |
|---|---|
| 0 (average body) | 11.489 |
| 1 | 0.483 |
| 2 | 0.203 |
| 5 | 0.174 |

Two numbers place the held-out body to 0.2 deg; the decoder emits 18 free numbers per timestep.

---

### F14. Five bodies cut held-out error 3.1x and beat the no-learning baseline

Held-out `c08f09t09`, RMSE deg per joint, `m3d_bracketed` at epoch 6:

| Predictor | TC | CF | FT | all |
|---|---|---|---|---|
| best possible linear mixture | -- | -- | -- | 0.18 |
| model | 2.25 | 3.59 | 4.53 | 3.57 |
| copy the nearest training body | 0.22 | 4.30 | 4.18 | 3.46 |
| mean of the five training bodies | 0.15 | 19.29 | 4.85 | 11.48 |
| predict this body's own mean | 18.24 | 6.30 | 11.83 | 13.07 |

3.1x better than the two-body setup's 11.04 deg, and clears the averaging baseline (3.57 vs 11.48). The z-ablation gap rises from 3-4x (two bodies) to 10-37x (five bodies) — the latent became far more load-bearing. Still 20x worse than a linear mixture of training bodies, and indistinguishable from copying the nearest one.

---

### F15. Bracketed against outside, with everything else held fixed

`m3d_bracketed` and `m3d_outside` share training bodies, data, normalisation and hyperparameters, differing only in which body is held out:

| | bracketed `c08f09t09` | outside `c06f06t06` | ratio |
|---|---|---|---|
| best over 12 epochs | 0.0764 | 0.9341 | 12x |
| mean, epochs 1-6 | 0.0902 | 1.9707 | 22x |
| mean, epochs 7-12 | 0.1034 | 1.5439 | 15x |
| z-ablation gap | 10-37x | 1.6-5.1x | |

Bracketing is worth 10-30x in error and ~7x in latent contribution. Reproduces F4 on an independent dataset with a properly matched control.

More bodies also help the unbracketed case, which two bodies did not:

| | trend, late epochs over early |
|---|---|
| `fold_short`, 2 training bodies, outside | 1.02x (flat across 41 epochs) |
| `m3d_outside`, 5 training bodies, outside | 0.78x (22% better) |

---

### F16. The decoder takes the body from the latent, not from the frame, keying off 11 percent of it while ignoring the whole frame

Merges the original F16 x2. Every body walks the same expert episodes, so at a given timestep two bodies share a gait-cycle point and the decoder's two inputs can be crossed. `m3d_bracketed` epoch 8, between two bodies whose commands differ by 28.63 deg (`scripts/diagnostics/swap_pathway.py`):

| frame from | latent from | RMSE vs c10f10t10 | RMSE vs c10f10t06 |
|---|---|---|---|
| c10f10t10 | c10f10t10 | 1.38 | 28.68 |
| c10f10t10 | c10f10t06 | 27.35 | 3.48 |
| c10f10t06 | c10f10t10 | 11.75 | 23.81 |
| c10f10t06 | c10f10t06 | 28.14 | 2.02 |

Given body A's frame and body B's latent, the decoder emits body B's commands to within 3.48 deg — the frame it looks at makes almost no difference. This strengthens through training: at epoch 20 the crossed case reaches 2.08 deg (vs 1.22 deg when both inputs agree), while the reverse crossing spreads from 4.6 deg apart (epoch 6) to 14.5 deg apart (epoch 20). This inverts the architecture's intent: `z` should be a body-independent action and `x_t` should say which body, but the model uses them the opposite way.

Decomposing `z`'s variance across the five training bodies at matched gait phases:

| Source | Share |
|---|---|
| where in the gait cycle | 64.1% |
| which body it is | 11.1% |
| interaction and residual | 24.8% |

Gait dominates `z`, but a linear probe still recovers the body from `z` at 0.724 (vs 0.200 chance) — and this small component is what the decoder uses. The frame carries leg lengths in full and the decoder ignores it; a lookup over five well-separated codes is cheaper than reading geometry off 256x1408 tokens, and has no entry for an unseen body.

Fitting which mixture of training bodies the model's output resembles (`scripts/diagnostics/morphology_mix.py`): 0.883 of the weight sits on one training body (best possible mixture spreads to 0.697); implied segment scales (0.980, 0.975, 0.973) vs actual (0.80, 0.90, 0.90). By epoch 20, concentration rises to 0.947 and switches which body it copies (from `c10f10t10` to `c06f10t10`, whose implied scales (0.615, 0.991, 0.984) closely match its actual (0.6, 1.0, 1.0)) — switching lookup entries, not converging. Held-out error over the same span goes 3.57 to 3.88 deg. Coverage helps because it puts a closer code in the table, not because the model learned to infer morphology.

---

### F17. The encoder carries morphology and generalises; the decoder does not use it

No world model involved: fit a regression from mean-pooled frozen embedding `e_t` to the three segment scales on the five training bodies, apply to an unseen body.

Predicted vs actual segment scale (selected rows):

| Body | | coxa | femur | tibia |
|---|---|---|---|---|
| c08f09t09 | held out, bracketed | 0.850 / 0.80 | 0.939 / 0.90 | 0.898 / 0.90 |
| c06f06t06 | held out, outside | 0.872 / 0.60 | 0.683 / 0.60 | 0.710 / 0.60 |

Errors on the bracketed body are 0.050, 0.039, 0.002, from ridge regression on a 1408-d average of patch tokens. Against this, the trained Motion Decoder's implied scales for the same body are (0.980, 0.975, 0.973) vs truth (0.80, 0.90, 0.90) — a 5.2M-parameter decoder with the frame in front of it is further from the answer than a linear map with 4,227 parameters. The decoder is the failing component, not information starvation.

MLP vs linear readout (mean abs error, training / held-out bodies):

| | training bodies | held-out bodies |
|---|---|---|
| linear | 0.020/0.003/0.001 | 0.161/0.061/0.056 |
| MLP | 0.005/0.006/0.011 | 0.130/0.115/0.121 |

MLP is better in-distribution, ~2x worse held-out (same capacity pattern as F4).

Condition on when the probe works: it predicts a new body to within 0.03 only if that body is a reachable non-negative mixture of the training bodies; otherwise error jumps to 0.16-0.17. `c08f09t09` is exactly reachable (distance 0); `c06f06t06` and `c10f10t06` are not (distance 0.283 each) because of correlated segment constraints in the training set. This doesn't settle whether the encoder carries the information for unreachable bodies — the readout can't be tested past its fitting range regardless.

Why the decoder can't do what the probe can: the probe sees the mean over all 256 patch tokens; the decoder sees those tokens via cross-attention with `z` as query, so it only retrieves what `z` asks for, and `z` is 64% gait phase (F16). Morphology is present in the tokens and never queried — also explains why removing the body code from `z` didn't help (F18).

---

### F18. Removing the body code from the latent moves the channel but does not help

`--lambda_adv 0.1` puts a gradient-reversal classifier on `z`. Against `m3d_bracketed` (differs only in this flag), averaged over ten epochs:

| | control | adversarial | change |
|---|---|---|---|
| held-out error | 0.097 | 0.118 | 1.21x worse |
| z-gap (`zero_z`/held-out) | 23.4x | 5.1x | latent used 4.6x less |
| x-gap (`zero_x`/held-out) | 10.7x | 21.8x | frame used 2.0x more |
| probe on `z` | 0.724 post hoc | plateau at 0.440 | body code partly removed |

The decoder moved off the latent onto the frame by 2x, and the shift grows (x-gap climbs from 11.3x at epoch 1 to 31.9x at epoch 10). Transfer never improved — pushing the decoder onto the frame just fails more clearly, since (per F17) the decoder cannot extract morphology from the frame even though a linear probe on the same embeddings can. The probe settles at 0.440 from epoch 7 on, above the 0.200 chance level, and stops falling (ITM/classifier reach a standoff).

Note: below-chance classifier accuracy (a 5-epoch smoke run drove the probe to 0.002 against 0.200 chance) means the code is rotating faster than the classifier tracks it, not that it's gone.

---

### F19. Giving the decoder direct access to the frame makes it use the frame less

F17: a ridge probe on mean-pooled `e_t` recovers a held-out body's segment scales to 0.05, but the decoder doesn't. `--md_head pooled` gives the decoder that exact view (mean over patch tokens, projected, added as a residual onto the action), initialised to zero so training starts bit-identical to the `mlp` decoder; a residual on the output can't be down-weighted away (unlike an earlier tried concatenation design, where the fusion layer suppressed the new path — frame-ablation gap fell from 12.5x to 7.1x on the smoke set).

Against `m3d_bracketed` (differs only in `md_head`), over eleven epochs:

| | control | pooled | change |
|---|---|---|---|
| held-out error | 0.098 | 0.099 | 1.01x, identical |
| z-gap | 21.1x | 29.6x | latent used 1.4x more |
| x-gap | 10.9x | 1.4x | frame used 7.6x less |

Handed the view that works, the decoder relies on the frame almost not at all, holding transfer exactly level by leaning harder on `z` instead. Direct residual read confirms: at epoch 6 (best held-out error), residual magnitude is 0.24-0.28 deg (full run, 9,425 pairs) vs 1.5-1.9 deg (smoke, 975 pairs) — varies ~3x more with leg pose than leg length, and is 0.9% of the 28.6 deg gap between two training bodies. More data made it smaller and the between/within ratio worse (0.32 vs 0.67), same direction as the frame ablation.

Cross-input test (epoch 6): pooled decoder follows the latent MORE faithfully than control does when frame/latent are crossed (RMS 3.42 vs control's 16.52) — a mismatched frame moves its answer only 12%.

Summary of all interventions tried:

| Intervention | Result |
|---|---|
| rescale the motion target (F9) | no change |
| shrink the decoder head (F4) | 1.4-2.1x worse |
| remove the body code from `z` (F18) | frame used 2x more, transfer 1.21x worse |
| give the decoder the pooled view (F19) | frame used 7.6x less, transfer level |
| more training bodies (F14, F15) | 3.1x better — the only one that worked |

`L_motion` is satisfiable by a lookup over five body codes in `z` at lower cost than reading leg geometry off pixels, regardless of what route to the pixels is provided; nothing in the loss requires the appearance-to-morphology mapping that transfer needs.

---

### F20. The reconstruction loss barely uses the latent, and it is 99 percent of the gradient, because cross-augmentation makes its target almost entirely noise

Merges the original F20 x2. `L_recon` is an MSE on unnormalised V-JEPA2 embeddings (needs baselines to interpret). Held-out body, `m3d_bracketed` epoch 20 and `stage1_100ep_framed_runB` epoch 20:

| horizon | FTM | copy `e_t` | FTM with `z` zeroed | `z` helps | FTM vs copy |
|---|---|---|---|---|---|
| 1 | 1.452 | 2.116 | 1.549 | 1.07x | 1.46x |
| 2 | 1.778 | 2.756 | 1.910 | 1.07x | 1.55x |
| 5 | 2.494 | 3.646 | 2.620 | 1.05x | 1.46x |
| 10 | 3.187 | 4.431 | 3.294 | 1.03x | 1.39x |

The forward model beats a static-frame baseline by 39-55%, but removing the latent costs it only 3-7% (vs 2,000-3,700% for the Motion Decoder) — the FTM is barely conditioned on `z`. Not specific to the five-body run (two-body gives 1.04x at horizon 1); contribution falls further at longer horizons since the frame becomes unpredictable enough that the model falls back on an average, needing no action.

With `lambda_recon = lambda_motion = 1.0`, recon sits at 1.6 and motion at 0.01: 99% of the gradient goes to a loss that doesn't need `z`, 1% to the loss that does. So `z` is shaped almost entirely by `L_motion`, which a lookup satisfies (F16).

Why `L_recon` barely uses the latent: the FTM predicts view 2's next frame from view 2's current frame and a latent from view 1; the two views have independently sampled crops/jitter, so part of the target is noise no latent could predict. Measured on 40 frames of one clip (same units as `L_recon`):

| | value |
|---|---|
| augmentation noise (one frame, two views) | 8.51 |
| signal (consecutive frames, no augmentation) | 1.97 |
| what FTM is asked to close | 8.43 |

No augmentation setting recovers the signal — jitter alone (no crop) still produces noise 2.10x the signal; crop 85->95% removes only 21% of the noise. Noise is 4.33x the signal, and augmentation accounts for 101% of the FTM's target; `z` could explain at most 23% of `L_recon`, and measured contribution is 3-7%. `z` was never under pressure to become an action.

Cross-augmentation is necessary (blocks the ITM from satisfying `L_recon` by copying `x_{t+1}` into `z`) and is confirmed load-bearing by F20 x F25: dropping it was considered because `z` only helps `L_recon` by 3-7%, but that was measured while `z` had no other job — under `action_lag 1` (F25) the decoder can only reach `a_{t+1}` through `z`, making compression of `e_{t+1}` into `z` the cheapest route to satisfy both `L_motion` and `L_recon` simultaneously (the degenerate solution cross-augmentation blocks). It stays on. A making-target-clean fix is ruled out (would make the copy shortcut more attractive, not less).

---

### F21. `lambda_cross` (cross-body motion decoding loss) improves transfer inside the range the training bodies span

**Scope (established by F24):** holds for a held-out body inside the training bodies' range. On `c06f06t06`, outside that range, the same flag makes transfer 1.35x worse than control.

**Corrected by F60:** an earlier claim that LAC-WM's shared latent "emerges from sharing modules across embodiments" was wrong — LAC-WM has an explicit alignment (motion-decoding) term too. What it lacks is a term for the cross-morphology regime specifically, since in its data body identity mostly doesn't determine the command (one robot does many manipulations), unlike here where each body does one behaviour.

`--lambda_cross` decodes body A's latent against body B's frame, supervised by body B's command (both bodies walk the same expert episodes at matching timesteps).

Against `m3d_bracketed` (control) over 25 epochs, on held-out body:

| | control | cross |
|---|---|---|
| held-out error, mean epochs 1-10 | 0.0992 | 0.0760 (23% better) |
| best (deg) | 3.57 | 2.91 |
| x-gap (frame reliance) | 10.7x | 40-69x |
| z-gap (latent reliance) | 21x | 2.2-3.2x |

Swap test (epoch 8, two bodies differing by 28.63 deg): given body A's frame + body B's latent, cross model answers within 1.18 deg of A's command (vs 1.17 when inputs agree) — the frame decides, not the latent. Control's equivalent crossing produced the latent's body to 2.74 deg.

Mixture concentration on one training body falls from ~0.88-0.95 (control) to 0.540, below the 0.697 best-possible mixture; beats copy-nearest-body (2.91 vs 3.47 deg). Does not degrade with more training (epochs 11-25 average better than 1-10), unlike prior runs which peaked by epoch 8 (F12).

Caveat: `z` becomes barely used (z-gap 2.2-3.2x vs control's 21x) — decoder reads most of body+command from the frame. Forward model unchanged (removing `z` costs 1.03x cross vs 1.07x control): F20's finding that `L_recon` doesn't constrain the latent still stands.

---

### F22. `lambda_cross` purifies the latent (removes body content) rather than emptying it, on training bodies

| | control ep 20 | cross ep 8 | cross ep 27 |
|---|---|---|---|
| foot-contact pattern decodable from `z` | 0.757 | 0.744 | 0.787 |
| body decodable from `z` | 0.707 | 0.638 | 0.665 |
| variance of `z`: gait phase | 64.5% | 88.7% | 83.4% |
| variance of `z`: body | 8.8% | 1.2% | 1.2% |
| variance of `z`: interaction | 26.8% | 10.1% | 15.4% |

(Eight contact patterns, majority class 0.144.)

Gait/behaviour decoding from `z` is as good or better than control; body's share of variance falls ~7x (8.8% to 1.2%). Achieves what the adversarial head (F18) was built for and failed at, as a by-product of a well-posed task rather than by fighting the latent — gait information survives intact. Explains the low-z ablation: with body now read from the frame, removing `z` only costs gait, not body code, so a 2.2x z-gap reflects `z` no longer carrying work that wasn't its own, not `z` being empty.

Scope: measured on training bodies only (see F30 for the held-out-pair boundary).

---

### F23. `lambda_cross` improves pose quality in physical replay, not distance walked

Physical (CoppeliaSim) replay of `m3d_cross` ep8 vs matched control `m3d_bracketed` ep6, held-out body `c08f09t09`, 3 clips, open loop.

| | control ep6 | cross ep8 |
|---|---|---|
| mean R2 over 18 joints | 0.832 | 0.868 |
| mean RMSE | 3.40 deg | 2.75 deg |
| duty-factor error vs IK | 0.076 | 0.044 |
| commands outside body's usable range | 7.7% | 5.4% |
| worst excursion outside range | 20.2 deg | 5.5 deg |
| forward distance as fraction of IK | 93% | 89% |

Distance did not improve (control 93% vs cross 89% of IK distance), despite better joint accuracy. Earlier claim that transfer covered "less than half" the required distance was from a two-body dataset; with five bodies both configs reach 84-96%, so coverage (not `lambda_cross`) fixed distance. What `lambda_cross` fixed is pose quality: control commands legs up to 20.2 deg outside reachable configuration (causes leg-folding once error accumulates in closed loop); cross model's worst excursion is 5.5 deg. Tripod index closer to IK ground truth for cross model.

3-4 point distance gap is within cross model's own clip-to-clip spread (84-93%), so not conclusive on its own; needs more than 3 clips to settle.

Figures: `results/wm/action_trace_*_c08f09t09.png`, `results/wm/gait/gait_*.png`, videos `results/wm/gait/replay_*.mp4`.

---

### F24. `c06f06t06` is not an extrapolation test; it bounds F21 — `lambda_cross` fails when scaling isn't relative

`c06f06t06` is `c10f10t10` uniformly scaled by 0.6 (collector scales IK foot targets by leg length), so its correct joint trajectories equal a training body's to 0.07 deg — the right answer is to copy a training body verbatim, not extrapolate.

| predictor | RMSE deg | mean R2 |
|---|---|---|
| copy `c10f10t10` (correct answer) | 0.07 | — |
| predict own mean | 12.73 | 0.00 |
| control `m3d_bracketed` ep6 | 13.92 | -2.01 |
| cross `m3d_cross` ep8 | 18.82 | -4.63 |

Both models worse than the trivial predictor (negative R2 on 12/18 joints). `lambda_cross` makes it worse: it reads apparent per-segment size well (femur/tibia scale read as 0.691/0.671 vs true 0.60) but wrongly applies a command change for what should be a uniform, commands-preserving scale (correct implied scale is 1.0/1.0/1.0 in command space). None of the 5 training bodies scale all three segments together, so uniform scaling is a direction the data never demonstrates — the model treats each segment's apparent size independently instead of reading the ratio between them, which is what actually determines commands.

Bounds F21: `lambda_cross` helps inside the training bodies' range but is an interpolation across seen bodies, not geometry-reading that holds outside it. Also means the earlier `m3d_outside` "extrapolation" result (held-out MSE 1.71-2.61) was actually the easiest possible case, not a real extrapolation test. Fix indicated is a data fix (add a uniform-scale body family), not a loss/architecture fix.

---

### F25. The transition target was mislabeled: `frames[t]` already contains the answer to `actions[t]`, so `z` was never forced to carry motion

Motion Decoder takes `(e_t, z)`, never sees `e_{t+1}`, so `z`'s only job should be to carry what the second frame adds. Substituting `e_{t+1}` (held-out body `c08f09t09`, 195 transitions):

| what's given as `e_{t+1}` | control ep6 | cross ep8 |
|---|---|---|
| real next frame | 3.57 | 2.91 |
| `e_t` again (no transition) | 3.96 (1.11x) | 3.47 (1.19x) |
| random other-time frame | 9.65 (2.70x) | 6.10 (2.10x) |
| `e_{t-1}` (backwards) | 5.13 (1.44x) | 4.18 (1.44x) |
| latent zeroed | 19.24 (5.39x) | 6.04 (2.08x) |

Deleting the transition costs only 11-19%; a wrong transition costs more (1.44x-2.70x) than none, so the latent is sensitive to it, but most of what's needed is already in `e_t`.

Re-measured on the corrected target (`lag1_ctrl` ep12, `lag1_cross` ep5): duplicate-frame cost rises to 1.36x/1.23x (from 1.11x/1.19x) — correction works, but wasn't what transfer was short of.

**Root cause (collector, not model):** `sim/collect/collect_ik.py` applies `cmds[t]`, steps sim, then captures `frames[t]` — so `frames[t]` is the result of `actions[t]`, and `actions[t]` (target for training on `(e_t,e_{t+1})`) is already visible in `e_t`. Nothing forced anything through `z`.

**Fix: `action_lag`, now defaults to 1** — decoder must predict the command that *caused* the transition, forcing the answer through `z` since it never sees `e_{t+1}`. Runs recorded before 2026-08-09 are read back with `action_lag 0` via `wm.config.from_checkpoint`, so their numbers are unchanged.

Note: moving the target alone isn't sufficient if the model sees both frames (a ridge probe on `[e_t,e_{t+1}]` gains equally for `a_t` or `a_{t+1}`); what matters is the decoder's input being `e_t` alone.

---

### F26. A single frame nearly determines the joint command at every prediction horizon, because the gait is periodic

Ridge regression from a single frame's pooled embedding to joint command at various offsets, `c10f10t10`, 6 clips (4 fit / 2 test), command spread 11.33 deg:

| target | RMSE deg |
|---|---|
| `a_t` | 4.61 |
| `a_{t+8}` | 5.23 |
| `a_{t+32}` | 4.45 |

Predicting 32 frames ahead is as accurate as predicting the current command — error never exceeds 5.33 at any horizon tested.

> **Superseded in its numbers by F46 (conclusion unchanged).** Re-measured on `fwd_m3d` (18 clips, more samples): errors uniformly lower (3.00/3.40/2.86 for `a_t`/`a_{t+8}`/`a_{t+32}`); measured gait cycle is 19 frames (not 22), same across all 5 bodies. 32 isn't a multiple of 19, so "wraps back to a similar phase" isn't the mechanism — open-loop IK plus fixed phase makes every horizon predictable regardless. Use F46's table for numbers.

Direction/change is also mostly readable from one frame: predicting `a_{t+1}-a_t` (the change, std 4.78 deg) from frame `t` alone gets RMSE 2.61 (one frame explains ~70% of variance); adding frame `t+1` only gains to 2.40 (1.09x). Single-leg swing direction is ambiguous from one frame, but the configuration of all 18 joints resolves it (which feet are swinging: 0.815 accuracy vs 0.5 chance).

Confirmed via `lag1_ctrl` vs `m3d_bracketed` (differ only in `action_lag`): by epoch 2, held-out error is identical to 3 decimals (0.1219 vs 0.1215) and z-gap is *lower* under `action_lag 1` (11.3x vs 24.9x) — opposite of what the fix was meant to produce.

**Bounds F25 and F21's action-decoding path specifically**: no target choice makes the transition necessary for action decoding on this data (single-speed forward-walking insect gait is nearly a closed loop in configuration space). Says nothing about the forward model's own competence (F46 finds that intact) — conclusion is "the action decoder was never the place to look for the world model," not "the world model is inert."

---

### F27. Forward-model signal-to-noise improves 3.7x by widening the frame gap and reducing augmentation together, independently of each other

F20 found FTM target signal=1.97 vs augmentation noise=8.4 (best augmentation setting still 2.10x noise), because consecutive 20 Hz frames barely differ. Widening the frame gap is an independent second dial.

Embedding distance vs frame gap (noise floor 8.39):

| frames apart | real change | signal/noise |
|---|---|---|
| 1 (current) | 2.01 | 0.24x |
| 5 (source paper's stride) | 3.58 | 0.43x |
| 16 | 4.89 | 0.58x |

Gain saturates past ~5 steps (periodic gait bounds the achievable distance).

Combined with weaker augmentation:

| gap + augmentation | signal | noise | ratio |
|---|---|---|---|
| 1 step, current aug | 2.01 | 8.42 | 0.24x |
| 5 steps, photometric jitter only | 3.58 | 4.02 | 0.89x |
| 10 steps, jitter only | 4.35 | 4.02 | 1.08x |
| 16 steps, jitter only | 4.89 | 4.02 | 1.22x |

3.7x improvement (0.24x to 0.89x); at 10 steps signal exceeds noise for the first time — first config where the FTM predicts something mostly real. Also explains why the source paper uses 5-step action chunking (not just downsampling, but signal preservation).

Risk: weaker augmentation reopens the frame-copying shortcut, which under `action_lag 1` could pay into both loss terms. Untested hypothesis: at 5-10 step gaps the latent must carry a frame 250-500ms away through a 64-D bottleneck, harder to copy than a near-identical neighboring frame — needs a probe from `z` to future frame content to confirm.

---

### F28. A proper (non-degenerate) extrapolation test: held-out tibia-short body fails, and the model's error shape matches the gap in training data

`c06f06t06` (F24) was degenerate; the proper test holds out the tibia-short family. Trained on `c10f10t10, c06f10t10, c10f06t06, c08f09t09`, held out `c10f10t06`. In every training body femur and tibia scale together (1.0/1.0, 1.0/1.0, 0.6/0.6, 0.9/0.9); the held-out body is the first with femur≠tibia (1.0/0.6).

Pre-registered thresholds (from baselines on commands): own-mean predictor 15.99 deg, best mixture 20.31, copy-nearest 20.34. Below 15.7 = geometry read; above 18 = mere interpolation.

| predictor | `c10f10t06` | `c06f10t06` |
|---|---|---|
| own mean | 16.01 | 15.75 |
| best mixture of training bodies | 19.58 | 18.43 |
| copy nearest | 20.37 | 19.12 |
| model, `lambda_cross 0.5`, best checkpoint | 27.68 | 25.60 |

Model is 1.36x worse than the mixture ceiling — not just failing to extrapolate but worse than optimal interpolation. Implied segment scale: truth (1.00, 1.00, 0.60), model says (0.93, 0.70, 0.70) — ties femur to tibia just like the best mixture does (0.99, 0.62, 0.62), because no combination of training bodies (where the two always move together) can separate them. All 18 joints score negative R2, including fore-aft swing joints that normally survive.

Matched control (no cross term) is 1.11x better (mean 9.41 vs 10.47) and flat across 10 epochs — repeats F24's pattern that outside training range, `lambda_cross` doesn't help and is slightly worse; failure belongs to the split, not the loss term. Control's probe on `z` climbs 0.519→0.762 over training (latent reverts to body code without the cross term opposing it).

Conclusion: compositional generalisation failure, points to a data fix (generate bodies where femur/tibia differ independently, via `sim/scene/make_leg_morphology.py`), not a loss/architecture fix.

---

### F29. A cheap frozen-encoder probe (F17), fit before any training, predicts which held-out bodies will transfer

Segment-scale probe error (distance from held-out body's true segment scales to nearest non-negative mixture of training bodies' scales — no encoder/model/run needed) correctly ranks all three held-out bodies tested:

| held out | reachable by mixing training bodies | probe error | trained model deg | baseline | outcome |
|---|---|---|---|---|---|
| `c08f09t09` | yes, exactly | 0.030 | 2.91 | copy-nearest 3.47 | beats it |
| `c06f06t06` | no, 0.283 away | 0.155 | 18.82 | own mean 12.73 | loses |
| `c10f10t06` | no, 0.283 away | 0.172 | 27.68 | own mean 16.01 | loses |
| `c06f10t06` | no, 0.283 away | 0.172 | 25.60 | own mean 15.75 | loses |

Separation factor of 5; no other measurement here orders the three correctly (the mixture ceiling calls `c06f06t06` trivially easy at 0.07 deg command-space distance, yet the model fails on it anyway).

Probe costs minutes of CPU with no training; discovering the same result by training costs ~4 GPU-hours per body. Useful for vetting a train/held-out split before running it. Caveat: three points establish an ordering, not a calibrated threshold; doesn't require resolving whether large probe error means the encoder lacks the info or the fitting bodies simply can't reach that far — predicts the outcome either way.

---

### F30. `lambda_cross`'s latent-purification effect (F22) does not survive when the held-out pair includes an out-of-range body

F22 measured variance decomposition on 5 training bodies. Same grid built on the 2 held-out bodies (`scripts/diagnostics/z_body_share.py`), with all 10 training-body pairs as a like-for-like reference at matching group size:

| body's share of latent variance | training bodies | training pairs | held out |
|---|---|---|---|
| old target, no cross (`m3d_bracketed` ep6) | 11.3% | 7.2% (0.0-10.8 range) | 6.8% |
| corrected target, no cross (`lag1_ctrl` ep12) | 10.4% | 6.7% (0.0-9.2) | 11.7% |
| old target, with cross (`m3d_cross` ep8) | 1.2% | 0.8% (0.0-1.3) | 10.6% |
| corrected target, with cross (`lag1_cross` ep5) | 0.8% | 0.5% (0.0-0.8) | 8.6% |

Corrected target alone moves body share <1pt; cross term moves it an order of magnitude; together they compound to 0.8%, the best held-out error (0.0698) and best reconstruction (2.82 deg). But the held-out column resists all 4 configs (6.8-11.7%, plain control best).

**Scope caveat:** the held-out decomposition needs ≥2 bodies, computed on `c08f09t09` (in-hull) + `c06f06t06` (0.283 out-of-range) together — the 8.6% could be driven entirely by the out-of-range body; this measurement can't separate them. Defensible claim: purification does not survive a held-out pair where one member is outside the training range — consistent with F24/F28/F29, not an independent new limit. Separating the two needs two in-hull held-out bodies (not available in current body set).

Mechanism: `lambda_cross` only constrains latent↔frame pairs present in training; a body outside that set produces an unconstrained latent free to carry apparent size etc.

Summary table (inside vs outside training range):

| measurement | inside | outside |
|---|---|---|
| decoder joint commands | 2.91 deg, beats copy-nearest | 25.60-27.68 deg, loses to constant |
| encoder segment-scale probe | 0.030 | 0.155-0.172 |
| latent body content | 1.2% | 10.6% |

Consequence: Stage 2 cross-embodiment transfer requires body-independence on unseen bodies, and this is direct evidence the mechanism doesn't provide it there.

---

### F31. The simulator has a hard feasibility floor on leg morphology: `|femur - tibia|` sets a dead zone the foot targets can fall into

A two-link leg cannot place its foot closer to the shoulder than `|femur - tibia|` (triangle inequality). Collector pulls foot targets to half the hip-to-foot distance; closest target across 30 episodes is 92.5mm (spread 92.5-93.7mm).

It's a step function, not a gradient:

| body | \|femur-tibia\| | targets inside dead zone | IK residual |
|---|---|---|---|
| `c10f10t08` | 11.8mm | 0.0% | 19mm |
| `c10f10t10` | 71.0mm | 0.0% | 32mm |
| `c10f10t06` | 94.6mm | 0.3% | 2mm |
| `c10f07t09` | 132.5mm | 24.2% | 809mm |
| `c10f08t10` | 139.6mm | 27.3% | 349mm |

A body 2mm past the limit loses 0.3% of targets and walks fine; one 40mm past loses a quarter and doesn't. No body is ever too-far-reach-limited (0.0% in every row); leg length alone doesn't sort failures.

Fix: `sim/scene/make_leg_morphology.py` now refuses to generate a violating body and prints the margin (`--force` to override). 3 of the first 6 attempted bodies were infeasible; the rule catches all 3 pre-collection.

Consequence: `c10f10t06` (the held-out body of the tibia-short split, F28) is itself 2mm past the limit, so no feasible body can bracket it exactly in segment-scale space (nearest feasible hull distance 0.0707 across all 182 permitted bodies on a 0.1 grid). This is a floor in command space and is unaffected by the split's design.

---

### F32. WITHDRAWN by F37 — Stage 2 shared-trunk embodiment-identity measurement was contaminated

> **Headline withdrawn by F37, not merely caveated.** The 33.0% embodiment-share number below was measured on `stage2_balanced`, which has a 10.5:1 class imbalance repaired only by repeating B1 data ~10x/epoch, manufacturing a spurious phase/embodiment correlation (75% of frames at one phase bin are B1). The clean measurement (F37) reverses the substantive claim: embodiment identity is fully decodable (0.99) but **not load-bearing** — removing it costs less than removing a random direction. Cite F37, not this. Kept only for how the mistake was found.

Stage 2 (first run): one ITM/FTM/decoder backbone shared across 18-DOF hexapod and 12-DOF quadruped, per-embodiment output head, no cross-embodiment term (matching the source method, which claims sharing emerges from weight sharing alone).

Latent variance split (stance fraction as phase label since embodiments share no episodes): gait phase 39.6%, which embodiment 33.0%, interaction 27.4%; linear probe separates embodiment at 1.000.

UMAP comparison (2,104 frames): frozen encoder `e_t` gives probe 1.000, silhouette +0.671, cluster separation 4.01x; learned latent `z` gives probe 1.000, silhouette +0.140, separation 0.77x — weight sharing narrows separation ~5x but the projection still shows two clean clusters despite the weak underlying separation (UMAP oversells separation at low true separation — a general lesson about UMAP + probe/silhouette needing to be reported together).

Caveats even before F37: validation metric was unusable (`val_fraction 0.1` on 14 B1 clips → 67 transitions, balanced sampling repeats them); LR schedule hit zero at epoch 6 while validation was still improving at epoch 12.

---

### F33. SUPERSEDED BY F37 — this "embodiment identity is load-bearing and smeared" measurement was a contamination artefact

> Measured on `stage2_balanced` (2 robots that collapse/rotate, 10.5:1 imbalance fixed only by repeating B1 data 10x/epoch, F36). On clean data the conclusion reverses: removing embodiment identity costs *less* than removing random directions on both seeds — identity is passive leakage, not load-bearing. See F37. Kept because the reasoning/method is sound and the reversal itself is informative (presence ≠ use; a contaminated dataset can make a passive quantity look functional).

As measured (superseded): tested whether identity (33.0% of latent variance, F32) is used downstream. Peeled identity-carrying directions out of the 64-D latent (`scripts/diagnostics/z_identity_ablation.py`, `stage2_balanced/best.pt` ep12, 2,104 latents), compared to removing equal count of random orthogonal directions:

| latent | B1 | hexapod | mean vs intact |
|---|---|---|---|
| intact | 3.42 | 3.39 | 1.00x |
| identity removed, 8 dirs | 4.03 | 7.45 | 1.69x |
| random 8 dirs removed | 3.79 | 4.13 | 1.16x |
| `z` zeroed | 29.34 | 18.82 | 7.07x |

1.69x vs 1.16x control read as "identity is used," cost mostly on hexapod side. Removing directions one at a time never cleared identity (1.000→0.806 after 8 removed, vs 0.500 chance) — identity distributed across the latent, not localized to a subspace (explains adversary's below-chance rotation behaviour rather than deletion).

Motivated `cfg.ftm_embodiment_channel` (side channel supplying identity directly to FTM so `z` is freed from carrying it) — but F37 found this channel relieves a pressure that doesn't exist on clean data.

---

### F34. The vision path runs at 10.5 Hz — closer to biological perception timescales than to robot control loop rates

V-JEPA2 ViT-g/16 (1B params, frozen), single-frame encoding with frame-duplication trick, on a 2080 Ti: 94.9 ms (10.5 Hz), averaged over 20 calls post-warmup. ITM/FTM/decoder combined are negligible by comparison.

| | sensors | loop time |
|---|---|---|
| biological | ~10^6 nerve endings | 200 ms |
| robot control | few | 20 ms (50 Hz) |
| this pipeline's vision path | one camera | 94.9 ms (10.5 Hz) |

Not framed as a defect: vision here is not for bandwidth/latency but because it describes an 18-DOF hexapod and 12-DOF quadruped in the same coordinates without being given a description of either (they share no joint space, sensor correspondence, or midpoint — F32's premise). That costs 95ms/frame.

Note: morphology-agnostic proprioceptive control does exist (joints as tokens over the kinematic graph), but those methods must be handed the kinematic tree explicitly, whereas a camera needs nothing handed to it — the defensible distinction.

Consequence: architecture is necessarily two-rate — perception plans at ~10Hz, control stabilizes at 50Hz (proprioception isn't removed, it does the job vision can't). Cross-embodiment stance-fraction probe (F35) shows within-embodiment contact reads at 0.82-0.89x target spread vs 1.04-1.16x across embodiments — near/past the point where the image is worth nothing there.

---

### F35. Cross-embodiment probe failure was mostly an artefact of mean-pooling and appearance offset, not the encoder; per-leg contact reveals training makes embodiments less comparable, not more

**Pooling artefact.** F31's cross-embodiment stance-fraction probe reported 4.72x/3.00x of target spread (mean-pooled patch tokens) — three reductions compared:

| fitted→tested | mean | bands | max |
|---|---|---|---|
| insect→insect | 0.88x | 0.84x | 0.92x |
| B1→B1 | 0.89x | 0.89x | 0.92x |
| insect→B1 | 4.72x | 2.74x | 1.32x |
| B1→insect | 3.00x | 2.35x | 1.06x |
| embodiment cluster separation | 3.94x | 3.58x | 1.83x |

Same feature count (1,408) in all cases — not a capacity issue. Mean-pooling absorbs a large constant embodiment offset into the ridge intercept, which then misfires cross-embodiment; max-pooling discards the offset and separation/error drop sharply. Every cross cell still ≥1.00x though — a cross-fitted readout is never better than ignoring the image; F32's premise stands, only the *magnitude* was overstated.

Lesson: pooling must match the quantity — mean-pooling works for frame-spread quantities (segment scale 0.050, embodiment identity probe 1.000) but not localized ones (which feet are loaded, ~6-12 of 256 patches).

**Appearance artefact.** Standardizing each embodiment by its own mean/spread (removes color + apparent-size offset, uses only dataset origin not target — standard unsupervised domain adaptation):

| cross cells | mean | bands | max |
|---|---|---|---|
| raw | 4.72x/3.00x | 2.74x/2.35x | 1.32x/1.06x |
| appearance controlled | 1.57x/1.07x | 1.16x/1.04x | 1.22x/1.02x |

Best reported figure: band-pooled + controlled, 1.16x/1.04x — with color, size and pooling all controlled, frozen encoder still gives nothing usable across embodiments. Limit: needs a batch of the new robot's frames (domain adaptation, not zero-shot) — already required anyway since a new embodiment needs its own output head. This correction does NOT apply to F32's 33% embodiment share in `z` (that quantity *is* the between-group mean, so centering would zero it by construction).

Reproduce: `scripts/diagnostics/cross_embodiment_probe.py --features {mean,bands,max} --normalize`.

**Per-leg contact (replaces stance-fraction, which is unfixable — B1 trots, insect wave-gaits, no shared phase; F37).** Binary, near-balanced, anatomically matched corner legs (`scripts/diagnostics/leg_contact_probe.py`, band-pooled, per-embodiment standardized):

| leg | insect→insect | B1→B1 | insect→B1 | B1→insect |
|---|---|---|---|---|
| mean | 0.806 | 0.941 | 0.531 | 0.547 |

Genuine diagonal ceiling (0.806/0.941), so 0.531/0.547 cross cells (barely above 0.5 chance) are real, not "nothing to predict." Front legs transfer *below* chance, hind legs above — tracks duty-cycle difference (insect front leg duty 0.309 vs B1 0.578; hind legs closer, 0.591 vs 0.500) — points at behavioral difference, not appearance.

**On the learned latent `z` (direct test of shared-latent claim):**

| | insect→insect | B1→B1 | insect→B1 | B1→insect |
|---|---|---|---|---|
| frozen encoder `e_t` | 0.806 | 0.941 | 0.531 | 0.547 |
| `z`, `stage2_clean` | 0.811 | 0.986 | 0.373 | 0.401 |
| `z`, `adv_warm10` (adversarial) | 0.798 | 0.969 | 0.490 | 0.500 |
| `z`, centred | 0.851 | 0.978 | 0.559 | 0.468 |

Training makes embodiments *less* comparable: cross cells fall below chance (0.373/0.401) for the base trained latent — a readout fitted on one embodiment is systematically wrong-signed on the other. Not a capacity issue (`z` is 64-D vs `e_t`'s 5,632-D but is *better* within-embodiment). The adversary repairs this specifically (0.373→0.490, 0.401→0.500) but nothing exceeds where the frozen encoder already was — no intervention makes the two embodiments more comparable than V-JEPA2 alone.

Solver note: use `RidgeClassifierCV` not `LogisticRegression` when n_features > n_samples (5,632 features vs ~1,900 samples) — logistic regression fails to converge in hours.

---

### F36. Two hexapod bodies in the training data do not walk and two more crab sideways — an unenforced exclusion convention contaminated every Stage 1/2 run. UNRESOLVED

Discovered chasing why the latent cleanly splits hexapod frames by femur-vs-tibia length. `data/allocentric/fwd_hex8body`, 9 bodies measured on recorded trajectories:

| body | f/t ratio | dead zone | forward (m) | sideways drift (m) | verdict |
|---|---|---|---|---|---|
| c06f10t06, c10f10t06 | 1.38 | 94.6mm | 0.37-0.48 | 0.35-0.38 | crabs sideways |
| c06f06t10, c10f06t10 | 0.50 | 208.2mm | ~0 to -0.37 | 0.28-0.33 | does not walk / walks backwards |
| (5 others, ratio 0.83) | 0.83 | 42.6-71.0mm | 0.37-0.66 | 0.06-0.17 | fine |

Cause is F31's dead zone (`|femur-tibia|`), not the ratio per se — Track A bracket bodies at ratios 1.04-1.10 (dead zones 11.8-26.0mm) walk normally. `c10f06t10` got in via a bug: walk check used unsigned displacement `norm(h[-1,:2]-h[0,:2])`, misreading a backwards-reversing body as walking (0.46m). Both bad bodies were already flagged in PROGRESS 18.2 (walked successfully in only 20-21/30 clips) and "cut" only by omission from `train_morphs` in Stage 1 — but the clips stayed on disk, and Stage 2 globs the whole directory, so the exclusion (living in callers, not data) silently broke.

Contamination by run: `m3d_cross`/`m3d_bracketed`/`lag1_*` train on 2 of 5 crab-walkers; `tib_cross`'s held-out body `c10f10t06` is itself a crab-walker; every Stage 2 run includes both non-walking bodies (~22% of hexapod clips). Does NOT invalidate comparisons *between* Stage 2 runs sharing the same (contaminated) data (`stage2_balanced`/`stage2_sidechannel`/`stage2_centered`).

Re-scoring `tib_cross`'s single checkpoint (not retraining) on 3 additional clean held-out bodies shows the femur/tibia extrapolation gap is real but the crab-walker `c10f10t06` overstates it 3-7x (27.76 deg / R2 -3.16 vs 11-13 deg / R2 -0.4 to -1.1 on clean bodies) — comparable to signal (11.7 deg spread), not "four times it." Caution: re-scoring one checkpoint isolates the test body; retraining with a different held-out body also changes what was learned (one bad attempt conflated the two and produced a spurious "-1.34x frame harm" signal).

Biological note: ratio 1.38 (femur > tibia) doesn't occur in stick insects (Phasmatodea); occurs in orthopteran jumping legs. Base insect ratio is 0.83.

**Admission rule now in code:** body usable iff `|femur-tibia|` < 92.5mm (closest commanded target). Ratio alone is not the criterion (1.04, 1.10 walk straight and fine; 1.38 veers; 0.50 collapses).

Contact labels checked and sound (stance-fraction 0.27N threshold sits in an empty valley, 1.8% of samples within ±0.07N). Expert gait is a real variable insect wave (not tripod) — expected, not a defect; `c10f10t06` uniquely shows negative tripod separation.

**Still outstanding (not done):** (1) drop the 2 non-walking bodies from Stage 2 sources — needs body filtering in `embodiment_split` or a curated directory; (2) rerun Stage 2 on clean bodies, re-measure the 33.0% (see F37, since done); (3) fix walk check to test forward/lateral displacement separately and signed; (4) decide whether the 94.6mm crab-walkers stay (walk, but outside animal proportions, and underlie the extrapolation claim); (5) get ratio diversity via coxa (doesn't enter the dead-zone formula) rather than rescaling the foot trajectory (which would break `lambda_cross`'s well-posedness).

---

### F37. Stage 2 on clean data: cross-embodiment transfer works and generalizes; embodiment identity is passive (reversing F33); behavioral replay shows metrics understate the gap

First defensible Stage 2 run: all training bodies walk, embodiments balanced by data not repetition, validation stratified, a hexapod body withheld. Two seeds, 60 epochs, converged.

    hexapod  4 bodies x 4 clips x 65 = 1,040 pairs
    b1       2 policies x 6 clips    = 1,003 pairs  (ratio 1.04:1)
    held out c08f09t09 (never trained on)
    excluded c06f06t10, c10f06t10 (collapse/rotate, F36)
    withheld c06f10t06, c10f10t06 (veer 0.35-0.40m, F36)

**Held-out hexapod body, both seeds positive R2 for the first time** (Stage 1 held-out bodies scored -0.42 to -3.16):

| | seed 0 | seed 1 |
|---|---|---|
| deg/joint | 3.85 | 3.43 |
| R2 vs body's own mean | +0.87 | +0.90 |
| latent zeroed | 0.365 | 0.444 |
| frame zeroed | 0.193 | 0.395 |

`c08f09t09` is inside the training range (interpolation, not F27/F28-style extrapolation) but is the same body Stage 1 held out, making stages comparable — Stage 1's `m3d_cross` scored 2.91 deg on it, so learning a quadruped alongside costs ~30% hexapod accuracy without breaking it. Caveat: `zero_x` figures differ 2x between seeds, so input-ablation ratios (incl. older z-gap/x-gap) deserve skepticism.

**Embodiment identity is passive, reversing F33:**

| | identity removed | random control |
|---|---|---|
| contaminated (F33) | 1.69x | 1.16x |
| clean, seed 0 | 1.03x | 1.18x |
| clean, seed 1 | 1.04x | 1.14x |

Removing identity costs *less* than removing random directions — nothing downstream reads it, even though `z` overall is heavily used (zeroing it costs 7.6-8.3x). Explains why the `ftm_embodiment_channel` side channel (F33) had no effect: it relieved a pressure that doesn't exist.

**Variance decomposition is not a usable number — withdraw F32's 33.0%.** Two seeds disagree by 2x (12.0% vs 6.7% embodiment share); probe stays 0.99 both seeds. Root cause: stance-fraction phase label has only 8 distinct values, quantile-bin edges collapse (e.g. requesting 4 bins yields only 2 occupied), giving a 2.7x swing (32.0% at 3 bins vs 12.0% at 6 bins) on the same checkpoint from a supposedly cosmetic parameter. Deeper fault: B1 sits at phase 0.5 86.6% of the time (trot) vs hexapod's spread wave gait, so B1 alone supplies 75% of all frames at phase 0.5 — phase and embodiment are half the same variable, not independent axes. Stage 1's `z_body_share` is unaffected (insect bodies share identical expert episodes so phase = timestep by construction); the general lesson is two robots with genuinely different gaits may have no shared phase label to measure at all.

Reliable claim: **embodiment is fully decodable from the latent at 0.99, and nothing uses it — removing it costs less than removing random directions.**

**Behavioral replay (requested by Ajan Blink: never report a number without the gait beside it) — CoppeliaSim, held-out body `c08f09t09`, clip ep101, 65 steps:**

| | forward (m) | heading | mean feet down | RMSE deg | out-of-range cmds |
|---|---|---|---|---|---|
| ground truth (IK) | +0.592 | +13.8 | 3.00/6 | — | 0.0% |
| intact | +0.374 | +14.0 | 2.74 | 3.98 | 7.9% |
| identity removed | +0.345 | +24.8 | 2.75 | 4.18 | 7.5% |
| frame zeroed | +0.297 | +26.8 | 3.09 | 7.26 | 3.2% |
| latent zeroed | +0.100 | +59.6 | 3.12 | 9.68 | 6.7% |

Zeroing latent stops the robot walking (0.100m vs 0.592m, 59.6 deg off course, drags a leg). Intact model walks only 63% as far as reference despite "small" 3.98 deg/R2+0.87 metrics — the metric and behavior tell different stories; recommend never reporting reconstruction R2 without replay. Identity removal is nearly behaviorally free for distance (0.345 vs 0.374, matches 1.03-1.05x from two independent methods) but heading degrades (24.8 vs 14.0 deg) — one clip, not conclusive, but "invisible" overstates it.

Two bugs found (both silently plausible-but-wrong): a `load_model` rewrite dropped `itm.load_state_dict` (ITM ran randomly initialized, made `zero_z` score better than intact); `load_clip` was hexapod-specific (B1 clips key commands as `action`, crashed the B1-side identity basis).

Method fix: `wm/evaluate.py:training_bodies(cfg)` now derives body list from the checkpoint's own config (previously 5 diagnostics hardcoded 3 different versions of `INSECT_BODIES`; 3 silently scored models on unseen bodies, e.g. `z_identity_ablation` read 15.99 deg where trained bodies read 1.45).

---

### F38. Stage 2 pretrained features transfer few-shot to a third embodiment (4-leg insect), 2.6-3x better than a random backbone; open-loop replay walks

**Body:** stick insect with middle legs removed (`ML,MR`), ghost-removed at runtime (no new scene file), driven by the unchanged 6-leg IK gait (no new policy trained) — 4 remaining legs are geometrically identical to the 6-leg versions, so existing commands apply directly (18-D -> 12-D). Only this leg-loss variant walks (front-loss tips over, hind-loss rears and collapses). Data: `data/ik_4leg_middleloss_clean9`, 9 clips (`--cam_dx -0.6 --spawn 0 0 --scale 0.5 --travel 0.8 --warmup 20`); some clips drift 0.19-0.20m laterally — a probe set, not training data.

**Test design (few-shot, not zero-shot — 12-D action spaces of B1 and 4-leg insect aren't semantically aligned):** freeze V-JEPA/ITM/FTM/decoder backbone, add new 12-D head, fit only that head on few clips, score held out. Control: identical procedure on a random (untrained) backbone.

| split | pretrained Stage 2 | random backbone | gain |
|---|---|---|---|
| mean | 1.75 +/- 0.10 deg, R2 +0.967 | 4.99 +/- 0.24 deg, R2 +0.743 | 2.86x |

Also sample-efficient: pretrained beats random by 2.6-2.9x at every clip budget (1,3,5,7); pretrained with **1 clip** (2.56 deg) beats random with **7 clips** (4.78 deg).

**Latent contributes, not just the frame** (F26 showed one frame nearly determines commands, raising the worry this is just that shortcut):

| | test deg | R2 |
|---|---|---|
| real `z` | 1.86 | +0.96 |
| zero `z` | 2.49 | +0.94 |
| shuffled `z` | 3.35 | +0.88 |
| random backbone | 5.06 | +0.74 |

zero-z beats random (frame+backbone carries a lot) but real-z beats zero/shuffled-z, so an aligned transition latent adds something beyond frame alone.

**Open-loop replay in CoppeliaSim walks**, closely matching IK reference (e.g. ep101: predicted +0.660/-0.233m vs IK +0.701/-0.239m, 1.8% out-of-range commands) across all 4 held-out split-A clips — still open-loop, not closed-loop control.

Does not repair bad demonstrations: on a veering 8-clip bad-demo set, pretrained still beats random (2.31 vs 6.75 deg) but replay veers where the IK veers — it learned the correspondence, not gait correction.

**Embodiment-identity interventions, all measured together:**

| | 4-leg few-shot split A | held-out hexapod R2 | probe after 8 dirs removed | z zeroed |
|---|---|---|---|---|
| `stage2_clean` | 1.86 | +0.87 | 0.738 | 7.63x |
| `stage2_clean_centered` | 1.88 | +0.89 | 0.697 | 9.96x |
| `stage2_clean_adv_warm10` (adversary) | 1.66 (best) | +0.88 | 0.598 (lowest) | 4.44x |

Centering does nothing (online probe relearns identity to 1.000 within 25 epochs — centering only removes the first moment; shape/silhouette/leg-count differences survive it; the F35 "offset wrecks linear readout" lesson applies to linear readouts, not a nonlinear model trained on both). Adversary is the only lever that moves both the 4-leg result and residual identity.

The apparent "weaker latent" signal (z-zeroed 7.63x->4.44x) does not survive scrutiny — that was measured on the decoder. Rolling the **forward model** itself (true latents, held-out body, 162 rollouts) is identical within 1% across clean/adversary/centred at every horizon (1-10 steps) — `z` still carries everything the FTM needs; what changed is only the decoder's `zero_z`/`zero_x` balance (clean 0.365/0.193 -> adversary 0.140/0.266): decoder relies less on latent, more on frame post-adversary.

> **Last paragraph corrected by F40 (2026-08-14).** The swap test on these same checkpoints shows the decoder still answers with the latent's body under the adversary. The zero_z/zero_x shift is real but overstates the effect — zeroing an input is out-of-distribution and compares runs rather than locating the pathway. "Moves the decoder toward reading geometry from pixels" holds directionally; "a candidate to replace the baseline" does not.

Measurement trap: `scripts/diagnostics/score_body.py` initially didn't apply stored `embedding_offsets`, so a centred checkpoint scored 15.10 deg/R2 -0.95 raw vs 3.61 deg/R2 +0.89 corrected — falsely read as "centring breaks transfer." Does not affect `fit_4leg_head` (new head absorbs constant offset into its bias).

**Setup guidance distilled from the project:**
1. Bodies, not episodes: 16x more episodes of the same two bodies changed nothing (F11); two to five bodies cut held-out error 3.1x (F14). ~30 episodes across 6-8 bodies costs the same to collect as 100 across 3.
2. Bracket the test body inside the training hull — 10-30x better than outside it with everything else identical (F15); state which regime a reported number comes from.
3. Check that morphology axes actually move commands before designing a held-out split around one: coxa moves commands 0.73 deg vs tibia's 28.63 deg (F13), so a three-parameter family can be two-dimensional in the space that matters.
4. Keep the shared foot trajectory — it is what makes "the same latent action across different bodies" well defined and what `lambda_cross` needs to be well-posed.
5. Report copy-nearest-body, training-body average, and best-linear-mixture baselines in every held-out table (the mixture reached 0.18 deg here, a ceiling as well as a baseline).
6. Report per joint type, against the held-out body's own baseline, over at least three runs (F8, F10 each independently show one aggregate number misleading).
7. Do not select a checkpoint on the held-out curve (F12) — fix the budget in advance or report the whole curve.

**What this enables — the mechanism, not just a score.** A frozen video encoder (V-JEPA2, never trained on robots) carries body morphology in a form that is linearly decodable and generalises to an unseen body: ridge regression on mean-pooled embeddings, fit on five bodies, recovers a sixth body's segment scales to 0.050/0.039/0.002 (F17). The trained world model does not use this — it identifies the body from an 11% component of its own latent instead of the frame (a lookup over seen bodies, F16), landing at (0.98, 0.98, 0.97) implied scales against a true (0.80, 0.90, 0.90), worse than the 4,227-parameter linear probe despite having 5.2M parameters. The gap is architectural and specific: the decoder reaches the frame only through cross-attention with the latent as query, and the latent is 64% gait phase, so morphology sits in the tokens unqueried — confirmed by three failed interventions (rescaling the target, F9; shrinking the decoder, F4; removing the body code from the latent, F18). What works is training-body coverage inside the hull (F14, F15), not architecture changes.

For the cross-embodiment thesis: the encoder result is load-bearing — vision carries body geometry in a usable, generalising form that proprioception cannot match here, because an 18-DOF hexapod and a 12-DOF quadruped share no joint space or midpoint for a baseline (copy-nearest, averaging, linear mixture) to exist in. That asymmetry — vision as a common space where proprioception is not — is what the 4-leg few-shot result above empirically supports.

---

### F39. No candidate label lets `lambda_cross` be ported to Stage 2 cross-embodiment pairing on current data

`lambda_cross` (F21) is the only Stage 1 intervention that improved transfer, but it's well-posed only because every insect body walks the *same* expert episodes — hexapod and B1 share none. Per-leg contact was the candidate pairing label (needs no shared gait period, corner legs anatomically correspond, F35). Measured before running anything (`scripts/diagnostics/pairing_feasibility.py`, 150 hexapod clips/5 sound bodies, 14 B1 clips):

| label | overlap | hexapod frames pairable | intent (hexapod) | intent (B1) |
|---|---|---|---|---|
| `n_feet_down` (0-4) | 0.572 | 98.9% | 0.913 | 0.998 |
| `diagonal` (which pair loaded) | 0.711 | 100% | 0.918 | 0.605 |
| `corner_pattern` (16-way) | 0.240 | 33.8% | 0.630 | 0.524 |

`intent` = matched-pair command distance / random-pair distance within one body (geometry held constant); 1.0 = label carries no command information. Since `L_cross` supervises with the partner's command, a label whose value doesn't imply a similar command on one robot is a wrong target on the other, not just noisy.

No label works: `n_feet_down`/`diagonal` cover almost all frames but are near-coin-flips of intent on one robot each (0.998 on B1, 0.918 on hexapod) — coarsening for coverage destroys the meaning. The 16-way pattern carries real intent on both (0.630/0.524) but leaves two-thirds of hexapod frames unpaired. Mechanism: B1 spends 84.6% of time in just 2 of 16 patterns (the two trot diagonals); hexapod spreads over all 16 with none above 15%, and 9/16 are hexapod-only — there is no single choice of "same phase" that is both covered and meaningful for a 6-leg wave vs 4-leg trot on this data.

Sensor check: B1's `foot_contact` already binary (0.5 threshold is a no-op); hexapod uses validated 0.27N threshold; overall duty matches closely (0.533 vs 0.515) — mismatch isn't a sensor calibration artefact.

Limits: only 14 B1 clips/1,143 frames; both datasets are behaviorally narrow (one IK wave gait, one RL trot at fixed velocity), so part of the non-overlap may be F26's constraint restated rather than a hexapod-vs-quadruped fact — a dataset with varied gaits/speeds is the change that would test this, and this measurement quantifies how much overlap would need to widen to matter.

Method: cheap (minutes of CPU, no encoder/training) pre-run feasibility check, analogous to F29's Stage 1 probe — here concluding a mechanism can't yet be built, saving a ~4-GPU-hour run.

---

### F40. Stage 2 has the same body-lookup pathology as Stage 1 (F16), within the hexapod head; the adversary only partially fixes it

The Stage 1 swap-test pathology (F16) had never been checked on any Stage 2 checkpoint. Two hexapod training bodies (commands differ by 21.13 deg), decoded through the hexapod head, 195 transitions (`scripts/diagnostics/swap_pathway.py --embodiment hexapod`):

| checkpoint | frame from | latent from | vs body A | vs body B | follows |
|---|---|---|---|---|---|
| `stage2_clean` | c10f10t10 | c10f06t06 | 21.03 | 6.98 | latent |
| `stage2_clean` | c10f06t06 | c10f10t10 | 5.26 | 20.19 | latent |
| `adv_warm10` | c10f10t10 | c10f06t06 | 18.76 | 8.04 | latent |
| `adv_warm10` | c10f06t06 | c10f10t10 | 6.87 | 18.04 | latent |

Diagonals near-identical across runs (own-body reconstruction unaffected), so crossed cells are comparable. `stage2_clean` answers with the latent's body within 7 deg despite a 21-deg-different frame — same shape as F16's Stage 1 control. First direct evidence a cross term is needed in Stage 2, not just inherited by assumption from Stage 1.

Adversary narrows the margin from 3.0-3.8x to 2.3-2.6x (~1/4 of the way) but never approaches 1.0 (where the decoder would start following the frame) — unlike `lambda_cross`, which reversed this test outright in Stage 1 (F21).

**Explains why F38 read the adversary more favorably:** F38 measured the pathway via zero_z/zero_x ratios (0.365/0.193 → 0.140/0.266), which looked like a reversal, but zeroing an input is out-of-distribution and compares runs rather than locating the pathway (standing caveat in `scripts/README.md`). The swap test uses real embeddings and disagrees — the measurement with the known confound was the optimistic one.

Scope: this pathology lives within the hexapod head (4 bodies sharing one 18-D output), not between embodiments (per-embodiment heads already select structurally, and identity in `z` is passive there per F37). F39 rules out cross-embodiment pairing specifically; says nothing about this within-hexapod-head case, where the 4 bodies do walk identical expert episodes so `lambda_cross` is as well-defined as in Stage 1. Currently `MultiEmbodimentPairs` builds no `partners` map and emits no `cross_x_t` — the cross term is simply never computed (a code gap, not a data gap; the needed `group` field is already loaded).

---

### F41. The 4-leg's latent lands on the base body (transfer is lookup, not composition), but rebuilt on held-out geometry the transfer margin survives unchanged

The 4-leg insect is the base stick insect with middle legs ghost-removed, driven by the unchanged six-leg IK gait; its 12 commands are the base body's corner-leg columns bit-identically (max diff 0.0000 deg).

Cross-decoding test (frame from `c10f06t06`, latent from 4-leg, hexapod head, corner columns, 195 transitions, `scripts/diagnostics/swap_embodiment.py`): the latent-matched RMSE (5.78/6.91) tracks the latent's body, looking like the thesis property (latent from an untrained embodiment still drives correct output). But normalized latent distances show body identity dominates gait phase (different bodies at same timestep: 1.103 apart vs same body at random timestep: 0.981 apart), and the 4-leg latent sits only 0.578 from the base body — closer than same-body-different-phase. So the ITM reads the 4-leg video as "the base body," and the result is lookup, not composition: only leg count was novel, not geometry.

Control: repeated the same middle-leg removal on `c08f09t09` (held out from Stage 2 training, so leg count AND geometry are both novel). Few-shot transfer margin: 1.91 ± 0.08 deg (pretrained) vs 5.45 ± 0.16 deg (random backbone) = 2.85x, versus base-geometry body's 2.86x (F38). Margin is unchanged — the few-shot result was not an artifact of in-distribution geometry.

Side-finding: removing middle legs flips lateral drift direction (6-leg bodies drift positive, both 4-leg builds drift negative, 30/30 and 9/9 respectively), a systematic consequence of missing middle legs, not instability; this causes ~2/3 of episodes to fail the 0.20m lateral gate on both 4-leg builds, but the F38/F41 comparison is not confounded since selection is the same operation on both (both are the low-drift tail).

Data: `data/ik_4leg_c08f09t09_clean10` (10 clips kept from 30-episode sweep, same construction as `clean9`).
Scripts: `scripts/diagnostics/swap_embodiment.py`

---

### F42. Retrained on clean data (all bodies walk, `action_lag 1`): the pathway mechanism strengthens, the accuracy gain shrinks, and coverage becomes the strongest Stage 1 result

All five Stage 1 deck runs retrained by `scripts/run/retrain_stage1.sh` on clean data (every training body walks, corrected `action_lag 1`), scored with `scripts/diagnostics/score_body.py`. Raw numbers: `results/wm/stage1_correct/measurements/heldout_scores.csv`.

| run | held out | deg | R^2 | zero_z | zero_x |
|---|---|---|---|---|---|
| m3d_cross (0.5) | c08f09t09 | 3.44 | +0.81 | 0.917 | 1.621 |
| m3d_bracketed (0.0) | c08f09t09 | 3.67 | +0.79 | 0.729 | 0.083 |
| tib_cross (0.5) | c10f10t08 | 12.67 | -0.78 | 2.349 | 1.991 |
| tib_ctrl (0.0) | c10f10t08 | 13.41 | -0.41 | 5.404 | 1.510 |
| bracket_cross (0.5, 6 bodies) | c10f10t08 | 3.27 | +0.89 | 0.699 | 1.037 |

Pathway result is stronger than on contaminated data: control's zero_x=0.083 (deleting frame costs nothing) vs zero_z=0.729; with cross term the order inverts to zero_x=1.621 vs zero_z=0.917 — a 16x swing in which input the decoder depends on, from one flag (cleanest statement of F16/F21).

Accuracy gain from `lambda_cross` shrank from 18% (contaminated: 2.91 vs 3.57) to 6% (clean: 3.44 vs 3.67) — mechanism strengthened, headline number weakened, contrary to Q13's prediction.

Coverage: 4 bodies tying femur/tibia score 12.67 deg, R^2 -0.78; 6 bodies decoupling them score 3.27 deg, R^2 +0.89 (matched 96 training clips both sides). Clean version crosses zero R^2 and beats every baseline (contaminated version only matched the constant-pose baseline).

Caveats: held-out body changed from `c10f10t06` (doesn't walk, F36) to `c10f10t08`; with 6-body set spanning 0.83-1.10, `c10f10t08` sits inside the hull, so bracket_cross is interpolation vs tib_cross's extrapolation — not a like-for-like comparison. m3d pair differs from deck in both data and action_lag. The three 10-epoch runs had not converged (still improving at epoch 10).

Metric disagreement: tib_cross beats tib_ctrl in degrees (12.67 vs 13.41) but loses on R^2 (-0.78 vs -0.41) — both orderings valid, answer different questions (raw error pooled vs standardized units weighting low-variance joints).

Full re-measurement of deck numbers on clean checkpoints (contaminated -> clean): latent-deletion cost reversed (3.97x/2.61x -> 2.88x/3.48x), replay distance/IK share reversed (93%/89% -> 85%/90%), replay heading deviation reversed (3.7/6.8 -> 11.8/5.5 deg), out-of-range commands advantage gone (7.7%/5.4% -> 6.1%/6.4%). Three deck claims did not survive and were dropped: cross term reduces latent dependence, cross term lowers out-of-range frequency, neither model veers more than IK reference (on one clean clip reference walks -0.8 deg while both models veer +11.5/+15.0 deg).

Latent-deletion reversal is informative: under contamination the cross term reduced dependence on z (z was largely a body code, contaminated by veering bodies); on clean data cross term increases dependence on z (z is 92.6% gait) — division of labor is frame=which body, latent=what movement, only visible on clean data.

Checked on `best_motion.pt` vs `best.pt` (selected on validation total, ~99% reconstruction) for all five runs: identical figures to two decimals, so checkpoint selection rule doesn't affect deck numbers.

---

### F43. Insect-only features transfer to the B1 quadruped, weakly but consistently, and entirely through the latent z (not the frame pathway); the forward model is actively harmful across embodiments

First test holding a genuinely different robot (B1) fully out of training: backbone `stage1_m3d_cross` (four insect bodies, never a quadruped) frozen, fresh 12-D B1 head fit few-shot, vs same head on a random backbone. Protocol from `fit_4leg_head` (`scripts/diagnostics/fit_b1_head.py`).

| split | train/test | pretrained | random backbone | margin |
|---|---|---|---|---|
| random | 5/9 | 20.49 deg, R^2 +0.35 | 23.80, +0.22 | 1.16x |
| random | 7/7 | 16.05 deg, R^2 +0.62 | 20.09, +0.48 | 1.25x |
| random | 9/5 | 15.62 deg, R^2 +0.68 | 20.48, +0.51 | 1.31x |
| velocity-stratified | 7/7 | 15.98 deg, R^2 +0.58 | 20.49, +0.39 | 1.28x |

Margin stable at 1.25-1.31x wherever both arms produce a usable head (5-clip split has both arms fail, R^2 -0.14/-0.12, not meaningful). Velocity-stratified split (controls for "new robot" vs "new speed" confound, B1 set = 2 policies x 7 speeds) gives 1.28x, matching the random 7/7 split's 1.25x — not a speed-generalization artefact.

Note: an earlier draft quoted "1.29x" for the random split, which was arithmetically inconsistent with the table (20.09/16.05 = 1.25x); corrected 2026-08-18, table values are authoritative.

Do not compare 1.28x (B1) to the 4-leg's 2.85x (F41) as "2.2x weaker transfer" — task difficulty differs independently of embodiment (B1 spans 2 policies x 7 speeds with unseen combos in test; every 4-leg clip is one behaviour at one speed).

Ablation on the velocity-stratified split isolates where transfer lives:

| arm | deg | R^2 | margin over random |
|---|---|---|---|
| pretrained, real z | 16.01 | +0.577 | 1.28x |
| pretrained, z zeroed | 20.86 | +0.342 | 0.98x |
| pretrained, z shuffled within clip | 22.04 | +0.283 | 0.93x |
| random backbone | 20.49 | +0.393 | -- |

Zeroing z removes the entire margin (0.98x, same as random within noise) — the decoder trunk's processing of the frame carries nothing across embodiments; all transfer is through z. Shuffling real latents within a clip scores worse than supplying none (22.04) — the latent's value is its alignment with the frame, not its distribution as such.

Forward model (FTM) rolled on B1 video with true latents, insect-trained (`stage1_m3d_cross`), vs held-out insect body:

| steps ahead | on B1 video | on held-out insect body |
|---|---|---|
| 1 | 0.63x | 1.52x |
| 3 | 0.57x | 1.72x |
| 5 | 0.63x | 1.69x |
| 10 | 0.71x | 1.46x |

Below 1.0 at every horizon on B1 — the FTM predicts worse than assuming nothing moves. Module summary: ITM latent z transfers (1.28x); decoder trunk's frame use does not (0.98x, same as random weights); FTM is harmful (0.57-0.71x). Since the deployed method rolls the FTM for planning against a goal, not the motion decoder, the current model could not be deployed on the B1 regardless of the few-shot head numbers.

---

### F44. Coverage (more bodies/embodiments in training) repairs the motion decoder but barely helps the forward model generalize to a new embodiment

Follow-up to F43's finding that the FTM is harmful across embodiments (B1).

Exposure, not architecture, is the limit: `stage2_clean` (trained on insects and B1) rolls on B1 video at 1.39-1.53x across 1/3/5/10 steps; `stage1_m3d_cross` (insects only) rolls at 0.57-0.71x on the same video. Same architecture/objective — the FTM can model a quadruped once it has seen one.

But within-family coverage barely helps FTM generalize: `tib_cross` (4 bodies, femur tied to tibia) vs `bracket_cross` (6 bodies, decoupled), rolled on the same held-out body at matched volume, improves only 5-8% at every horizon (e.g. 3-step: 1.27x to 1.38x). On the same checkpoint pair, the motion decoder improved 12.67 deg to 3.27 deg (R^2 -0.78 to +0.89), a 3.9x factor (F42) — the same intervention transforms the decoder but moves the FTM almost not at all.

Multi-embodiment training costs the FTM something on insects: `stage1_m3d_cross` (insects only) rolls at 1.46-1.72x on a held-out insect body, vs `stage2_clean` (added B1) at 1.30-1.52x — sharing the trunk with a quadruped slightly worsens the insect forward model.

Implication: adding bodies/embodiments to training is not a route to a forward model that works zero-shot on an unseen robot — the robot needs to be inside its training distribution. Since the source method's own transfer is a LoRA finetune on 7,265 target-robot trajectories (not zero-shot), the comparable question becomes sample-efficiency of adapting the FTM to a new robot, not zero-shot generalization.

---

### F45. The forward model adapts to the B1 from one clip; insect pretraining is worth roughly 7x fewer target clips

ITM/FTM from `stage1_m3d_cross` finetuned on N clips of B1 video vs same architecture from random init, encoder frozen both arms, scored by rolling FTM on its own output / holding frame still. Three splits per budget, 1,000 optimizer updates, four held-out clips.

| clips | model | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|
| 1 | pretrained | 1.02x | 1.00x | 0.94x | 0.83x |
| 1 | scratch | 0.89x | 0.68x | 0.64x | 0.66x |
| 5 | pretrained | 1.23x | 1.17x | 1.09x | 0.95x |
| 5 | scratch | 0.98x | 0.97x | 0.91x | 0.85x |
| 9 | pretrained | 1.37x | 1.36x | 1.26x | 1.05x |
| 9 | scratch | 1.01x | 1.10x | 1.06x | 0.96x |

Pretrained beats scratch in all 20 cells (5 budgets x 4 horizons), both rising monotonically with budget. Pretrained reaches 1.0x at h=1 with one clip; scratch needs seven — roughly 7x fewer target clips needed with insect pretraining. Curves separate rather than converge: from 5 clips on, scratch is flat at h=1 (~0.98-1.01x) while pretrained keeps climbing (1.23x to 1.37x at nine clips). At h=10, only pretrained clears break-even (1.05x at nine clips); scratch never crosses (max 0.96x).

Against F44's frozen-FTM result (0.57-0.71x, worse than no motion), one clip of target data already takes it to 1.02x — what fails to transfer zero-shot is calibration to the new body's dynamics, not the representation, and that calibration is cheap to acquire.

Budget cap: 14 clips total, 4 held out, so 10 is the largest clean training budget (train = order[:n], test = order[-4:] overlap once n+4>14; an n=11 run leaked into held-out and is not reportable — script now refuses overlapping budgets).

Scripts: `scripts/diagnostics/finetune_ftm.py --clips 1 3 5 7 9 --splits 3` (measured on machine com7). Note: batch device transfer, not the optimizer step, in `adapt()` — summing gradients over every span before stepping made `--steps` count epochs and blew up cost (374,400 passes, 13 hours) in an earlier run; see `scripts/README.md`.

---

### F46. F26 re-measured on the clean dataset (same conclusion, lower errors), gait cycle is 19 frames not 22, forward-model ablation shows it doesn't help action reconstruction, and rollout quality shows it does model dynamics

F26 was fitted on 4 clips (264 samples vs 1,408 encoder features), badly underdetermined. Re-fit on 18 clips (`fwd_m3d`, 26 clips of `c10f10t10`):

| predict from one frame | F26 (4 clips) | re-fit (18 clips) |
|---|---|---|
| command spread | 11.33 deg | 11.34 deg |
| a_t | 4.61 | 3.00 |
| a_{t+8} | 5.23 | 3.40 |
| a_{t+32} | 4.45 | 2.86 |
| second frame, on the change | 1.09x | 1.11x |

Errors fall (underdetermined fit was the cause) but spread and conclusion reproduce (0.01 deg agreement). Pooling all 5 bodies (140 clips) gives the same answer as a percentage of signal (26/30/24% vs 26/30/25% for single body).

Gait cycle is 19 frames, not 22 (autocorrelation, identical across all 5 bodies, range 19-19, as expected from replayed expert episodes). Since 32 is not a multiple of 19, F26's "distant offset wraps to similar phase" mechanism was wrong even though the conclusion held; the actual mechanism: commands are open-loop IK, one frame fixes phase and everything downstream is determined at any horizon.

Forward-model ablation (`lambda_recon 0` vs `m3d_bracketed` control, both `action_lag 0`, five-body dataset): FTM learns nothing (`recon` stays flat at initial 9.40 vs control's 1.6 falling), and action reconstruction is unchanged (held-out error 0.1025 vs 0.0992). Confirms the forward-prediction term contributes nothing to action reconstruction (follows from F25: target already visible in e_t). Side effect: removing it pushes the decoder onto z and off the frame (latent-ablation gap rises 21x to 24-62x; frame-ablation gap falls 10.7x to 2.5-6.8x) — L_recon was pushing toward pixels while taking ~99% of gradient for no accuracy gain. This does NOT mean the forward model learned nothing — only that it doesn't help action reconstruction.

What the forward model actually learned (correct task: rollout quality, not action reconstruction): FTM closed on its own output, real latents from ITM, un-augmented frames of held-out body, 162 rollouts across 3 clips:

| steps ahead | forward model | hold e_t still | constant velocity | vs hold |
|---|---|---|---|---|
| 1 | 1.53 | 2.11 | 5.78 | 1.38x |
| 3 | 2.07 | 3.05 | 27.6 | 1.47x |
| 5 | 2.54 | 3.57 | 66.0 | 1.41x |
| 10 | 3.63 | 4.36 | 236.5 | 1.20x |

Beats frozen world at every horizon to 10 steps, beats constant velocity by two orders of magnitude at the far end — the forward model can roll the world forward; it just doesn't make L_motion easier since the joint command was already recoverable from e_t. Caveats: holding e_t still is a weak baseline (1.2-1.5x margin is real but modest, decays with horizon); true latents were supplied (isolates FTM but easier than a rollout requiring latent selection too). Matches the source method's usage: Motion Decoder is an auxiliary regularizer, deployed system predicts future embeddings and rolls out 8 steps for action selection via subgoal comparison — measuring FTM by action-reconstruction improvement was the wrong task.

---

### F47. Insect pretraining transfers two separable things to the B1: transferable one-step dynamics (frozen) and a foothold in V-JEPA2's feature space (only the latter drives the 7x adaptation speedup)

Two ITM+FTM arms, identical except the ITM's second input frame: `real` uses e_{t+1} (actual next frame); `shuffled` uses e_s, random s in the same clip (s != t). Same 100 clips of `fwd_m3d`, same architecture, 15,000 minibatch steps, same seed, ITM+FTM only (no decoder/cross term, matches `finetune_ftm.py`).

After B1 finetuning, the two arms are indistinguishable (e.g. at 9 clips: real 1.31x/1.30x/1.21x/1.05x vs shuffled 1.31x/1.33x/1.24x/1.05x across h=1/3/5/10).

Frozen (pre-finetune) on B1, they differ and both are poor: real 0.54x/0.51x/0.53x/0.59x vs shuffled 0.39x/0.45x/0.49x/0.57x (h=1/3/5/10) — real's advantage decays with horizon (1.38x, 1.13x, 1.08x, 1.04x), consistent with real being trained on one-step pairs.

Frozen, in-domain (held-out insect clips), both arms are competent: real 1.38x/1.37x/1.24x/1.04x; shuffled 1.33x/1.46x/1.34x/1.10x (h=1/3/5/10) — real wins at h=1, shuffled (trained on stride-scale pairs averaging 21.9 frames apart) wins at h>=3 by 7-8%. So training the FTM on adjacent frames produces worse multi-step rollout than stride-scale pairs, in-domain — a design point since the FTM is used via multi-step rollout.

Embodiment gap on the same weights: 1.38x in-domain (insects) vs 0.54x on B1 — not undertraining, it's the change of robot.

Interpretation: pretraining supplies (1) transferable one-step dynamics (frozen comparison, real 1.38x at h=1, but both arms far below 1.0x on B1 so useless alone) and (2) a foothold in the shared V-JEPA2 feature space (post-finetune comparison — this produces the measured 7x adaptation speedup, F45). A thousand optimizer steps on B1 clips teach more B1 dynamics than insect pretraining carried, overwriting the frozen edge; pretraining still matters (9 clips: pretrained 1.31x vs scratch 1.01x, gap never closes) — what survives finetuning is familiarity with the shared representation, not learned dynamics per se.

`shuffled` does not remove motion, only adjacency: partners average 21.9 frames apart vs gait cycle of 19 (F46); pose distance is 3.44 deg at one frame, peaks 17.67 deg at half a cycle, falls to 5.81 deg at a full cycle; 14.6% of shuffled pairs land within one frame of same phase under uniform sampling.

Scale note: insect expert data runs at 20 Hz (`sim_time` in `expert_66k_aug3c_fcontact.csv`), clip = 3.30s, stride = 0.95s, t->t+1 = 50ms = 1/19 of a stride (19% of half-stride pose change). This reconciles: slide 11's second frame being worth only 1.11x, shuffled matching real after finetuning, and real's frozen edge living at h=1 and dying by h=10 — the informative window is stride-scale, not one timestep.

Caveat: both frozen comparisons are single evaluations, not repeated on a second pretraining seed.

---

### F48. Synthesis (no new data): the cross-embodiment failure chain has one named cause — no frame-pairing exists across embodiments, so lambda_cross cannot be trained, so the trunk partitions instead of sharing

Chain built from existing findings: F39 (no cross-embodiment frame pairing exists — contact-pattern labels either cover everything and mean nothing, or mean something and pair a third) -> lambda_cross cannot be applied (needs two frames showing the same intent; insect bodies share expert episodes, hexapod/B1 do not) -> nothing forces z to mean the same thing across robots (F37/F40: embodiment decodable from z at 0.994, yet deleting it costs only 1.03x, less than deleting random directions — label present but inert) -> trunk partitions rather than shares (per-leg contact probe: 0.986 within an embodiment, 0.373 across, below frozen encoder's 0.531 and below chance) -> learned dynamics do not travel (F47: what survives a robot change is the foothold in the encoder's shared space, not what the trunk learned, because the trunk learned to split).

Stage 1 is the positive control: there lambda_cross is definable (every body walks the same expert episode) and it reverses the swap test outright. In Stage 2 without it, an adversary narrows the swap test from 3.0-3.8x to 2.3-2.6x but never approaches 1.0, and moves cross-cell probe values only to 0.490/0.500 (chance) — an adversary removes decodability but does not install shared meaning.

Falsifiable prediction: if a cross-embodiment pairing can be defined and lambda_cross trained on it, the cross-embodiment contact probe should rise above 0.531, the swap test should move toward 1.0, and the frozen FTM should clear its current 0.57-0.71x on the B1. If pairing is fixed and none of those move, this chain is wrong.

Re-weights Q14: widening behavioral overlap across embodiments is the precondition for a pairing to exist at all, not simply "more data" — it is the one intervention aimed at the cause rather than a symptom.

---

### F49. The B1 and hexapod gaits differ in degrees of freedom (B1: 1 DOF trot, insect: ~6 loosely coupled legs), which structurally prevents a tight cross-embodiment frame pairing

Measures the reason F39 found no contact label both covered and meaningful. Anchoring phase at front-left touchdown and measuring circular concentration of other feet's landing phase (1.0 = perfectly repeatable, 0.0 = uniform):

| B1 leg | mean phase | concentration |
|---|---|---|
| FL | 0.00 | 1.00 |
| RR | 0.05 | 1.00 |
| FR | 0.49 | 0.99 |
| RL | 0.55 | 0.99 |

Hexapod: FL concentration 1.00, but ML/HL/FR/MR/HR all in the 0.07-0.24 range (near-uniform).

B1 is a textbook trot (all four feet determined by one phase); the insect's other five legs are near-uniform — it walks a variable wave (real animal recording), needing ~6 loosely coupled numbers where the B1 needs one. Consequence is structural: any low-dimensional label fully describing B1 gait state must underdetermine the insect's, so a pairing built on one supplies a wrong partner command, not a noisy one (F39 condition 3 failure).

Best label tested — phase anchored at touchdown crossed with Froude number v/sqrt(g*h):

| label | overlap | hexapod pairable | b1 pairable | intent hexapod | intent b1 |
|---|---|---|---|---|---|
| F39 feet-down 0-4 | 0.572 | 98.9% | -- | -- | 0.998 |
| F39 diagonal | 0.711 | 100% | -- | 0.918 | -- |
| F39 corner pattern | 0.240 | 33.8% | -- | 0.63 | 0.52 |
| phase x Froude | 0.578 | 100% | 100% | 0.647 | 0.278 |

Better coverage than F39's labels, better B1 intent ratio, but 0.647 on hexapod is loose — the 0.647/0.278 asymmetry reflects the leg-DOF difference. Froude is the part that works: hexapod averages 0.155, B1 averages 0.159 (hip heights 0.13m vs 0.56m) — near-identical Froude despite 4x size difference, so task space overlaps where contact does not, independent of phase.

Trap: an earlier phase measure (Hilbert phase of joint commands' first PC) gave intent ratios 0.362/0.265, but was circular — phase derived from the commands it was then tested against. Re-derived from contact independently, hexapod intent ratio moved 0.362 -> 0.647 (worse) and phase-alone moved 0.437 -> 0.895 (near meaningless). Any label computed from what it's scored on passes condition 3 for free — a general trap for future pairing labels.

Rules out finding a tighter frame-level pairing via a better label search. Two remaining routes: make the insect's gait regular (expensive, new foot-trajectory gen, retrain Stage 1), or drop frame-level pairing for a constraint not requiring a bijection.

---

### F50. The insect expert dataset walked at one speed, which made body-speed transfer measurement meaningless until retimed; behavioral diversity alone (5 speeds) does not fix cross-robot body-speed transfer

F49 established Froude number as the one overlapping body-level quantity (0.155 hexapod vs 0.159 B1). Reading body speed from z on the original single-speed data gave nonsense from the frozen encoder: insect->b1 -0.284, b1->insect -0.142 (worse than predicting the mean).

Cause: Froude variation wasn't separated between-clip (commanded speed) vs within-clip (body rocking):

| | between clips | within clip (rocking) | ratio |
|---|---|---|---|
| hexapod | 0.0188 | 0.0714 | 0.26 |
| B1 | 0.0255 | 0.0170 | 1.50 |

The insect expert data has essentially one speed (forward velocity sd 0.0086 m/s on mean 0.454 = 1.9%, across 1,000 episodes). A readout fit on B1 learns commanded speed; fit on insect it learns stride rocking — different quantities sharing a name, not a transferable question as originally posed.

Fix: retiming via `collect_ik.py --speed`, which resamples the shared foot path along time (same Cartesian path, fewer/more samples) — every leg resampled by the same time map, so inter-leg phase relationships (F49's variable wave) are preserved; this replays real animal coordination at different tempo rather than authoring a synthetic gait.

Data: `data/_archive/ik_walk_speed5` — five speeds, 0.72-1.10, 67 clips from 75 after `walk_check`.

| | insect, 5 speeds | B1 |
|---|---|---|
| mean Froude | 0.164 | 0.164 |
| range | 0.113-0.221 | 0.121-0.216 |
| sd between clips | 0.0284 | 0.0266 |
| signal/rocking | 1.45 | 7.28 |

Matched to B1 by design (7 commanded B1 speeds vs 5 insect retimings, same Froude band; second axis differs on purpose — B1 has 2 gait policies, insect has 5 morphologies).

On this retimed data, frozen encoder is positive cross-robot: +0.012 and +0.079 (vs -0.284/-0.142 on single-speed data) — V-JEPA2 carried shared body-level structure the whole time; the original measurement was broken, not the representation.

Training destroys it: trained on the five-speed data with no new term, z gives -4.163 and -5.595.

Behavioral diversity alone does not fix cross-robot body-speed transfer (this claim is corrected/narrowed by F54 — see that entry): widening insect from 1 to 5 speeds moved b1->insect from -24.359 to -5.595 (4x improvement) but sign is still wrong, two orders below the frozen encoder. First test of Q14's behavioral-diversity lever; on its own it does not close the gap.

Two measurement traps recorded: (1) window size matters — between-clip vs within-clip rocking ratio reads 0.63 at a 5-frame window vs 2.17 at stride-length window on the same clips; raw per-frame values hand a readout mostly rocking, not speed (same lesson as F47's 50ms t->t+1 issue). (2) A "diverse" dataset can still be a lookup table — five discrete speeds is five numbers to memorize, which is what happened in F51.

---

### F51. A shared body-motion term (L_body, decoding Froude number from z through one head shared across embodiments) produces the first cross-embodiment result that beats the frozen encoder

Motivated by F48's diagnosis that `L_motion` supervises z through per-embodiment heads with no cross-robot correspondence. `wm/models/body_motion.py`: decode forward Froude number from z through one head shared by every embodiment (a per-embodiment head would reintroduce the same freedom).

Matched pair on `data/_archive/ik_walk_speed5` + `data/allocentric/fwd_b1_50hz`, one flag apart, 60 epochs, one seed:

| | insect->insect | b1->b1 | insect->b1 | b1->insect |
|---|---|---|---|---|
| frozen encoder | 0.666 | 0.750 | +0.012 | +0.079 |
| z, no term (control) | 0.624 | 0.155 | -4.163 | -5.595 |
| z, + L_body | 0.676 | 0.879 | -1.931 | +0.407 |

b1->insect reaches +0.407 vs frozen encoder's +0.079 (5.2x) — first cross-embodiment measurement in the project to clear the frozen-encoder bar. Three of four cells beat the frozen encoder; b1->b1 went 0.155 -> 0.879, past the encoder's 0.750. Both the term and the 5-speed data (F50) were needed (neither alone works).

Cost: val motion 0.0245 -> 0.0367 (+50%), val recon unchanged (1.5990 -> 1.6040). Superseded by F58: at `lambda_body 0.1` cost is only -2% with transfer unchanged within seed spread — so this cost is the chosen weight's price, not the mechanism's.

Embodiment probe stops saturating: control reaches 1.000 by epoch 32 and stays; L_body run plateaus at 0.954-0.963. First Stage 2 intervention to hold it below 1.0 (the adversary, F32, never managed this). Caveat: per F37, decodability alone (0.994, costs 1.03x to delete) isn't evidence of shared meaning — but the body-motion probe result (+0.407) is.

insect->b1 is still negative (-1.931): the direction that fails is fitted on the noisier side (signal-to-rocking: B1 7.28, insect 1.45, F50) — readout fit on the clean side transfers, on the noisy side does not.

The jointly-trained head memorizes: train loss 0.077 vs held-out 0.855 (target standardized so 1.0 = "predict the mean") — 11x gap, no generalization. 12 distinct speed values across 32 clips is a lookup table. But a fresh ridge probe fit directly on z found +0.407 — z acquired transferable structure the jointly-trained head never exploited; an earlier read calling this "memorization with no shared meaning" was wrong.

Next step (not yet done): continuous speed target within each clip (ramp retiming rather than constant factor) to remove the lookup table and raise insect signal-to-rocking simultaneously.

One seed only; `stage2_clean` was run with two and this should be repeated before weighing it alone.

---

### F52. The adversary's apparent 4-leg few-shot benefit disappears on a properly held-out test body; the z-ablation result survives and strengthens

F41 re-measured the headline few-shot margin on a 4-leg build cut from held-out `c08f09t09` and it survived (2.85x vs 2.86x). This re-measures the rest of that slide's claims on the same held-out build. Protocol: frozen backbone, new 12-D head, 5 training clips (ep28,69,93,101,113), 5 held out, random-backbone control at 5.35 deg.

| | base geometry | held-out c08f09t09 |
|---|---|---|
| stage2_clean | 1.86 deg | 1.99 deg |
| stage2_clean_adv_warm10 | 1.66 deg (11% better) | 2.13 deg (7% worse) |

The adversary's apparent gain was an artefact of the target's geometry being in distribution — on a genuinely unseen body, adversarial identity removal costs rather than helps. Consistent with F32 (adversary repairs leg probe only to chance) and the Stage 1 result that it made transfer 1.2x worse: the adversary moves identity metrics but never moves transfer.

z-ablation survives and strengthens on held-out geometry:

| held-out 4-leg | real aligned z | zero z | shuffled z | random backbone |
|---|---|---|---|---|
| base geometry | 1.86 | 2.49 (1.34x) | 3.35 (1.80x) | 5.06 |
| held-out geometry | 1.99 | 3.02 (1.52x) | 4.14 (2.08x) | 5.35 |

Aligned latent matters more on never-trained geometry, supporting rather than undermining the claim. Few-shot curve reproduces: 2.63x/2.81x/2.97x/2.94x at budgets 1/3/5/7 (held-out) vs 2.61/2.72/2.91/2.80 (base build).

General rule established: any claim measured on a body cut from a training body is provisional until re-measured on a held-out one — here 2 of 3 re-measurements held and 1 reversed.

---

### F53. Ramping speed within each clip (rather than constant per-clip speed) removes the body-motion head's memorization and flips the one failing transfer cell to positive

F51 left insect->b1 failing (-1.93 seed 0, +0.20 seed 1, the only cell to change sign) while b1->insect reproduced (+0.407, +0.377). Diagnosis: the failing direction is fitted on the noisier side (between-clip/within-clip signal ratio: B1 7.28, insect 1.45, F50), and F51's shared head memorized (train loss 0.077 vs held-out 0.855, where 1.0 = predict the mean; 5 constant speeds = 12 discrete values across 32 clips = lookup table).

Fix: `collect_ik.py --speed_end` ramps the rate linearly across a clip (renormalized to keep path distance constant, all change goes into elapsed time); all legs share the time map so inter-leg phase is preserved (verified at 0.056 lag vs constant case's 0.061). Data: `data/allocentric/fwd_hex7speed` = 5 constant speeds + both ramp directions, 91 clips from 105 after `walk_check` (both directions needed so a readout can't learn clip-position instead of speed).

| | body loss, train | held out | gap |
|---|---|---|---|
| 5 constant speeds, seed 0 | 0.0775 | 0.855 | 11.0x |
| 5 constant speeds, seed 1 | 0.0862 | 1.060 | 12.3x |
| ramped | 0.1010 | 0.705 | 7.0x |

Ramped head generalizes (0.705, first generalization shown) vs constant set sitting at/above 1.0 (no better than ignoring z).

Failing cell flips, one seed on speed7, epoch 60:

| | insect->insect | b1->b1 | insect->b1 | b1->insect |
|---|---|---|---|---|
| frozen encoder | 0.676 | 0.753 | -0.046 | +0.131 |
| control, no term | 0.664 | 0.167 | -7.083 | -2.357 |
| + L_body | 0.798 | 0.879 | +0.544 | +0.435 |

insect->b1: -1.93 -> +0.20 -> +0.544; note frozen encoder is negative in this cell, so the model creates structure the encoder didn't have. b1->insect holds at +0.435 (F51 seed1 gave +0.377). Compare against the control (identical 2-frame access, -7.102), not the encoder (1-frame, loaded comparison).

Cost: val motion 0.0166 -> 0.0259 (+56%) at lambda_body 0.5; superseded by F58 — at lambda_body 0.1 cost is only -2%. lambda_body 0.5 was copied from lambda_cross (Stage 1 value), never swept here.

Caveats: one seed on speed7 (insect->b1 is the cell that has already flipped once, second seed running); head still memorizes, 7x train-to-val gap vs 11-12x for constant set (reduced, not solved).

---

### F54. The cross-embodiment swap-test "switch" (F48's chain) was a property of single-speed training data, not an inherent trunk partitioning; the per-leg contact probe was never valid evidence for it

F48's causal chain (trunk partitions by robot instead of sharing) rested partly on two measurements that do not survive on speed-varied data, one of which was never valid.

Swap test (body A's frame with body B's latent, `c10f10t10` vs `c10f06t06`, commands differ 21.1 deg) reverses on speed-varied training data with no loss term:

| trained on | reads identity from | strength |
|---|---|---|
| fwd_hex8body, one speed | latent | 3.1x/3.8x |
| ik_walk_speed5, five speeds, no term | frame | 2.9x/3.9x |
| fwd_hex7speed, seven conditions, no term | frame | 5.0x/3.7x |
| fwd_hex7speed + L_body | frame | 4.8x/4.5x |

The controls (lambda_body 0.0, no shared head built) did this — L_body adds nothing here. This is what lambda_cross achieves in Stage 1 and what the adversary never managed (F52); here achieved simply by making the insect walk more than one speed.

Confounds checked and ruled out: same four bodies, same 5 clips/body, same 60 epochs, same architecture across both comparisons; scoring stage2_clean on speed-varied clips still gives latent at 3.1x/3.8x (training data matters, not eval data). Isolated further (`s2_fwd_hex8-b1_ctrl`, old data + new split matching fwd_hex7speed's held-out-body count): split choice does nothing (latent still wins, 3.2x/4.3x) — ruling out split and clips-per-body; frame-edge clipping also ruled out (0% of frames touch image edge in both datasets). What remains between datasets is speed variation.

The per-leg contact probe (slide 14's headline: z reads a loaded leg at 0.377 across, below frozen encoder's 0.531) was never valid evidence about the latent:

| | insect->insect | b1->b1 | insect->b1 | b1->insect |
|---|---|---|---|---|
| frozen encoder | 0.806 | 0.941 | 0.531 | 0.547 |
| stage2_clean | 0.802 | 0.986 | 0.377 | 0.398 |
| speed7 control | 0.842 | 0.989 | 0.586 | 0.515 |
| speed7 + L_body | 0.808 | 0.937 | 0.536 | 0.513 |

The 0.586 "beats the encoder" result is one leg (right hind 0.931, other three average 0.471, below chance); in the L_body run a different leg carries it (0.404 vs other three at 0.580). Matched-data run (`s2_fwd_hex8-b1_ctrl` vs `stage2_clean`, identical training data, differ only in split) shows the leg probe swings 0.19 (0.377 -> 0.562) while the swap test and forward model barely move (3.1x/3.9x vs 3.2x/4.3x; FM differs by 0.014) — the probe doesn't track the representation. Per-leg breakdown on the matched run: 0.30/0.42/0.74/0.79 — two below chance, two well above, within one run. Consistent with F49: the quantity structurally cannot transfer well (B1's four legs are phase-locked to one, insect's are not), so a four-leg mean with one leg jumping is not a transfer result — a bad score on an ill-posed question is evidence about the question, not the model. The deck now leads with body speed (which both robots genuinely share) instead of this probe.

What still stands: body-speed transfer across robots fails on every control regardless of data (-4.60/-24.36 single-speed, -7.10/-2.33 seven-condition), and only L_body moves it (F51, F53) — that is the real failure the loss term addresses, and data variation alone does not fix it.

---

### F55. `best.pt` checkpoint selection on validation `total` broke matched-pair comparisons when only one arm had an extra loss term (L_body); fixed by selecting on `recon+motion` only

`best.pt` was saved on validation `total` improving, which includes whichever extra loss a run enables. Since L_body generalizes poorly (F53: 0.10 train vs 0.71 held out), it makes total noisy with an early minimum, so `L_body` runs checkpointed much earlier than their controls:

| run | best.pt epoch |
|---|---|
| speed7 control | 59 |
| speed7 + L_body | 28 |
| speed5 control | 60 |
| speed5 + L_body | 49 |

This produced one wrong conclusion: forward-model rollout on B1 video appeared to show L_body costing the FTM something (control 1.42x/1.33x vs body-arm 1.37x/1.15x at h=1/10). Re-run from `last.pt` (epoch 60 vs epoch 60): identical (1.42x/1.42x at h=1, 1.33x/1.31x at h=10) — the entire apparent gap was the epoch mismatch; L_body neither costs nor helps the FTM.

It also understated the body-motion probe result it was meant to support: at matched epoch 60 (last.pt) every cell improves over the mismatched epoch-28 best.pt reading (e.g. insect->b1: +0.432 -> +0.544; b1->insect: +0.377 -> +0.435).

Comparisons that survived: swap test and leg probe (F54) compared control-against-control at matched epochs already, so unaffected. Body-motion probe was conservative (mismatch worked against the winning arm) rather than wrong. Only the forward-model test had the mismatch aligned with the effect being measured, producing a false conclusion.

Fix: `compute_losses` now emits `selection = lambda_recon * recon + lambda_motion * motion` (the two terms every run has) and `best.pt` selects on that; `total` still reports everything.

---

### F56. Behavioral variety (5-7 insect speeds vs 1) improves the forward model's rollout quality by ~7%, the first intervention besides within-family coverage to move it at all

F44 found the FTM the least responsive module (coverage moved it only 5-8% vs decoder's 3.9x). Giving the insect multiple speeds instead of one moves it by a similar order, consistently.

Design controls for distribution-match confound: `stage2_clean` (trained on `fwd_hex8body`, 1 speed) vs `s2_fwd_hex7-b1_ctrl` (trained on `fwd_hex7speed`, multi-speed), each scored on BOTH evaluation sets, both checkpoints at epoch 60, 4 clips/cell, 2 bodies (leg lengths 0.77m and 0.47m).

Result: speed7 control wins at every horizon (h=1,3,5,10) on both bodies and both evaluation sets — 24 of 24 comparisons. Mean gain +7.0% (+5.6% short horizons, +7.8% long horizons — grows with rollout length, the relevant direction for a rolled module). Distribution-match explanation ruled out: on `fwd_hex8body` clips (which speed7 never trained on and stage2_clean did), speed7 still wins at every horizon — a model beats the other model's owner on the owner's own training distribution.

Running total of what behavioral variety (data) vs the loss term fixed: decoder reading body identity from frame not z (F54, data); body-level question being answerable at all (F50, data); forward model rollout, +7% (this entry, data); z carrying body speed cross-robot (F51/F53, the loss term — every data-only control stays at ~-7.1).

Isolated 2026-08-18 (same run as F54): `s2_fwd_hex8-b1_ctrl` (old data, new split) matches the old single-speed run almost exactly (differs by 0.014 average across 12 cells) — split choice does nothing. Against this matched control, speed variation gains +7.9% on 12 of 12 horizons — the gain is from speed variation itself, not newer data collection, different split, or the frame-edge fix.

Caveat: small sample (4 clips/cell, 2 bodies), but effect is consistent in sign across all 24 comparisons, which supports credibility at this sample size.

---

### F57. Porting LAC-WM's frame-conditioned motion decoder (MD(x_t, z)) to the shared body-motion head breaks it; a shared head only constrains the latent if it is blind to embodiment

F51's shared body-motion head read z alone, which deviates from LAC-WM's `MD(x_t, z_t)` (conditioned on the observation). Rebuilt properly to match: one head on `MotionDecoder`'s shared `features(x_t, z)`, no embodiment key, gradient reaching the trunk (`s2_fwd_hex7-b1_bodyframe0.5`, same data/weight/control, epoch 60).

| | insect->insect | b1->b1 | insect->b1 | b1->insect |
|---|---|---|---|---|
| frozen encoder | 0.676 | 0.753 | -0.046 | +0.131 |
| control, no term | 0.664 | 0.167 | -7.083 | -2.357 |
| z-only head (F51) | 0.798 | 0.879 | +0.544 | +0.435 |
| frame + z head | 0.680 | 0.347 | -10.475 | -57.170 |

The frame-conditioned version is worse than adding nothing. Val motion cost also foreshadowed this: z-only head costs +55%, frame version only +12% — the term had simply stopped doing anything.

Diagnosis (`scripts/diagnostics/body_head_ablation.py`, zeroing one input at a time on the trained frame-conditioned head): real z gives body loss 0.3425; z zeroed gives 0.7932 (2.32x worse); frame zeroed gives 0.6795 (1.98x worse). The head does use z, but as a robot-specific code — the frame tells the head which robot it's looking at, so it learns one mapping per robot and z never has to agree with itself across embodiments. This re-introduces the per-embodiment-head problem (F48) through the image instead of an explicit embodiment key.

Rule established: a shared decoding head constrains the latent only if it is blind to embodiment — any input identifying the robot lets it decode conditionally, defeating the term's purpose. LAC-WM avoids this because end-effector/camera pose isn't readable from a still frame; body speed is (frozen encoder scores R^2 0.676 within-embodiment from one frame), so conditioning on the frame there gives the head everything it needs and the bottleneck learns nothing. The formulation does not port; the underlying principle does.

z is still built from two frames, so the blind head reads vision through the bottleneck being shaped — nothing in the deployed system (ITM, FTM, joint heads) is denied the image; only this one auxiliary head must be blind by design.

Config: `cfg.body_sees_frame` defaults to False; set True to rebuild the failed variant (kept reproducible as a negative result).

F51's +0.544 result stands, now justified by measurement rather than assertion; what doesn't stand is describing it as LAC-WM's mechanism ported over — the port was tried and fails here for a reason specific to locomotion.

---

### F58. The 55% validation-motion cost of L_body (F51/F53) was the loss weight (lambda_body 0.5, copied unjustified from lambda_cross), not an inherent mechanism cost; at lambda_body 0.1 the cost is ~2% while transfer is unchanged within seed spread

Swept lambda_body 0.1 vs 0.5, same control (`s2_fwd_hex7-b1_ctrl`, epoch 60):

| | val recon | val motion | cost | body loss train | held out |
|---|---|---|---|---|---|
| control | 1.5608 | 0.0167 | -- | -- | -- |
| lambda=0.5 | 1.5655 | 0.0259 | +55% | 0.1010 | 0.705 |
| lambda=0.1 | 1.5603 | 0.0164 | -2% | 0.1032 | 0.668 |

A fifth of the weight reaches the same body loss at no measurable cost; val recon/motion land marginally better than control.

Transfer numbers are noisier than initially read — an early comparison claimed lambda=0.1 improved alignment (+0.675 vs lambda=0.5's +0.544), but a second seed of lambda=0.5 landed at +0.749:

| | insect->b1 | b1->insect |
|---|---|---|
| control | -7.083 | -2.357 |
| lambda=0.5 seed 0 | +0.544 | +0.435 |
| lambda=0.5 seed 1 | +0.749 | +0.704 |
| lambda=0.1 seed 0 | +0.675 | +0.624 |

lambda=0.1 sits inside lambda=0.5's own seed spread — one seed per weight cannot separate them on transfer.

Seed stability by metric (spread across two seeds): val total 0.7%, val recon 0.9%, val motion 14%, probe insect->b1 27% — training is stable to under a percent, but the probe (a downstream ridge fit on z, not directly optimized) is noisy since small latent-geometry differences produce large transferability differences.

Effect still dwarfs the noise: control-to-treatment gap on the probe is 7.6, seed spread is 0.20 — 38x factor. So "the term makes body speed transfer" is solid; "this weight beats that weight" is not and needs a second seed. Report the seed range (insect->b1: +0.54 to +0.75), and treat the near-zero cost of lambda=0.1 as the operating point (valid since it's a within-seed comparison against a shared control) — but note it is still only one seed at lambda=0.1 and not yet reproduced.

---

### F59. Correlating two robot-specific readouts' predictions ("agreement") is a stable metric for cross-embodiment transfer; correlating their fitted weights is not, and standardized z still leaks embodiment to nonlinear probes

F58 found the R^2 probe noisy (insect->b1 moved 27% across seeds while training metrics move under 1%) — R^2 is unbounded below and charges for scale/offset on top of direction. New metric in `body_motion_probe.py`: fit a readout per robot separately, run both over the same frames, correlate outputs ("agreement") — bounded, symmetric, blind to scale/offset.

| | insect->b1 R^2 | b1->insect R^2 | agreement |
|---|---|---|---|
| frozen encoder | -0.046 | +0.131 | 0.313 |
| control (lambda=0) | -7.083 | -2.357 | -0.014 |
| lambda=0.5 seed 0 | +0.544 | +0.435 | 0.845 |
| lambda=0.5 seed 1 | +0.749 | +0.704 | 0.915 |
| lambda=0.1 seed 0 | +0.675 | +0.624 | 0.898 |

Control's -7.083 becomes -0.014 (readouts are uncorrelated, not inverted); frozen encoder's -0.046 becomes 0.313 (V-JEPA2 already carries partial shared ordering a single linear readout can't exploit). Agreement is more stable than R^2 (8% seed spread vs 32%) but doesn't rescue the weight comparison — ordering across the three treated runs is consistent (0.845<0.898<0.915 vs 0.544<0.675<0.749), meaning part of the seed gap is real geometry difference, but lambda=0.1 still sits inside lambda=0.5's seed spread — F58's refusal to separate the weights stands.

Comparing fitted ridge weight vectors directly does not work (tried first): z is 64-D and correlated, so ridge coefficients are unidentified — every run reads near chance on |cos(w_A,w_B)| (0.014-0.085) including the best-transferring run (lambda=0.5 seed 1: R^2 +0.749 but cos 0.014). General lesson: to compare fitted models, compare predictions, not weights, in correlated feature spaces.

Two corrections from reporting Pearson r alongside R^2: (1) the insect->b1/b1->insect asymmetry is mostly calibration, not geometry — b1->insect R^2 swings 0.435->0.704 (62%) across seeds while its r moves only 0.852->0.863 (1.3%); direction is stable, gain/offset is not (insect->b1's r does move 0.743->0.879, so the "real geometry difference" claim only holds for that direction). (2) Embodiment identity is carried linearly by per-feature mean/scale: a clip-held-out classifier gets AUC 1.000 on raw z but only 0.441/0.459 once each embodiment is standardized — the transfer results aren't identity leaking through a linear channel. But nonlinear readers recover embodiment exactly even after standardizing (`scripts/diagnostics/identity_linearity.py`): linear logistic 0.460, random forest 0.999, MLP 1.000 on standardized features. So the probe's transfer numbers are trustworthy (a linear ridge can't exploit what a linear classifier can't find), but no claim of "the latent forgets the body" is available — a nonlinear reader recovers the robot exactly. (An earlier version of this check read 0.212 due to a frame-level fold split leaking near-duplicate frames; fixed by splitting folds by clip.)

Report agreement as the headline metric, R^2 alongside for the practical "can one robot's readout be used on the other" question, and r to separate direction from calibration. Spearman was measured and dropped (tracked Pearson within 0.013 on every run).

---

### F60. A careful re-read of the LAC-WM source paper corrects three earlier claims about it; the real cross-robot shared coordinate was never the geometry (foot position) but behavioral coverage — forward Froude is the only channel currently varying on both robots

Corrections to earlier (F21/F57-based) readings of LAC-WM (`doc/LATENT ACTION ROBOT FOUNDATION WORLD MODELS FOR CROSS-EMBODIMENT ADAPTATION.pdf`, ICLR 2026 submission):

1. LAC-WM does have an alignment term: `L = lambda_recon*L_recon + lambda_motion*L_motion`, same shape as ours. Its Figure 2 (IDM without motion decoder — Agibot/Egodex cluster together but separate from Droid) matches our own `s2_fwd_hex7-b1_ctrl` control (r = -0.048), found independently.
2. Per-embodiment output heads are not what separates us from LAC-WM (their MD does emit different-dimensional outputs per dataset: Droid 10-D, Agibot 20-D, EgoDex 138-D/147-D with camera pose — mechanism for this unstated in the paper). What actually separates us: the coordinate predicted. Theirs is wrist/fingertip/camera pose in a shared 3-D physical frame (same meaning across human hand and robot hand); ours is body-specific joint angles with no cross-robot referent. Their labels are also far richer (10-147 D vs our 1-D), so a 1-D aligned target isn't a method limitation, it's what we asked for.
3. The unified coordinate is not LAC-WM's main target — L_recon (next-frame prediction) carries the task; the motion decoder is auxiliary (stated purpose: mitigate shortcuts), alignment is a consequence not the goal.
4. F57's rule (shared head must be blind to embodiment) IS a real precondition the paper satisfies for free, not a deviation from it: LAC-WM's MD does see the frame, but its targets are deltas (a still frame cannot supply a delta), while ours (body speed) is a state recoverable from one frame at R^2 0.676. The paper never needs the blindness condition because it never faces it.

Withdrawn: the earlier "manipulation has a unified label, locomotion doesn't" sufficiency argument — a 7-DoF arm has a null space too; the asymmetry was an artefact of comparing their 147-D label to our 1-D one.

A follow-up proposal (foot position normalized by leg length, analogous to end-effector position, with z split into feet/body-twist halves) was considered but is superseded/withdrawn below.

Chunking test: does 5-step action chunking (as LAC-WM uses) turn our target into a delta, avoiding the single-frame-readable problem? Measured via `scripts/diagnostics/target_window_sweep.py` (fit readout from one frame's frozen embedding to forward speed averaged over window W) before committing to a retrain — premise was wrong. Shorter windows are not harder to read from a still frame (insect R^2: 0.627 at W=1, 0.670 at W=5); what matters is window direction not length — centered window (t-10:t+10) reads 0.676, forward-looking window (x[t+20]-x[t])/20dt reads only 0.246. Comparing frame vs z at various W is weak evidence since z is built from two frames (tautological advantage on forward-looking targets) and only one seed/checkpoint was used; also W=20 is 1.0s for insect but 0.4s for B1 (different frame rates), so cross-robot comparison at fixed W is invalid — trend within each robot stands.

Central finding: the coordinate choice was never the binding constraint; the foot-position proposal is contradicted by the project's own data. Three gates any shared auxiliary target must pass (varies? hides the robot? means the same thing on both?), assembled from prior measurements:

| candidate | varies? | hides robot? | same meaning both robots? | |
|---|---|---|---|---|
| duty factor | no (0.533 vs 0.515, F39) | yes | -- | fails |
| lateral speed | yes | no (AUC 0.788) | -- | fails |
| which leg is loaded | yes | yes | no (transfers at 0.373, below frozen encoder 0.531 and below chance, F35) | fails |
| forward Froude | yes | yes | yes (agreement 0.85-0.92, F59) | passes |

Foot target fails the same gates: it decomposes into stance-feet-move-at-body-speed/leg-length (= body speed rewritten, nothing beyond lambda_body) plus which-foot-is-in-stance-when (= the gait, the 0.373 non-transferring row). A per-leg readout fit on the B1's trot is systematically wrong on the insect's six-leg wave — coarsening a leg-level label enough to describe both robots destroys what made it meaningful (F39's structural argument recurring).

Why the shared head found only one axis: body twist is 6-D in principle but in this data forward speed is the only varying channel (lateral speed is zero in every B1 clip, yaw rate is constant per policy, acceleration only occurs at clip start/stop) — five of six channels are constants, so there was never more than one direction to align.

Conclusion: the real blocker is behavioral coverage, not coordinate choice. With one behavior, current state fixes the future (z holds nothing a single frame lacks), which is why F57 had to blind the head and why alignment left the FTM at 1.42x either way. Giving one state several possible futures would let the frame-conditioned decoder LAC-WM actually publishes run here as written, instead of the blinded variant currently required.

Withdrawn: the foot-coordinate proposal, and "pick a horizon by the z-minus-frame margin." The window sweep stands as a description of the target (short windows stay frame-readable, forward-looking ones less so) but is not a fix for anything.

---

### F61. Each body channel decomposes into a slow (behaviour) and fast (gait) component; only the slow component of forward speed transfers across robots

`scripts/diagnostics/channel_screen.py` scores each body-velocity channel raw and smoothed (~1 stride), on the same clips/checkpoint.

| channel | timescale | varies | robot AUC | insect->b1 | b1->insect |
|---|---|---|---|---|---|
| forward | per frame | 1.00 | 0.623 | -1.453 | -2.078 |
| forward | smoothed | 1.00 | 0.529 | +0.544 | +0.435 |
| lateral | per frame | 1.07 | 0.705 | -0.999 | -1.939 |
| lateral | smoothed | 1.05 | 0.834 | -2.647 | -0.250 |
| vertical | per frame | 0.78 | 0.505 | -1.333 | -3.883 |
| vertical | smoothed | 0.20 | 0.510 | -0.471 | -3.460 |

- Forward: varies, smoothing hides the robot (AUC 0.529) and cross-robot readout flips from -1.45 to +0.54. Only channel that transfers.
- Vertical: collapses when smoothed (variance 0.78->0.20) — the bob is almost entirely gait, nothing slow to align.
- Lateral: keeps variance when smoothed (1.05) but robot AUC rises to 0.834 — its slow component is embodiment identity (B1 never drifts sideways, insect does), not shared behaviour. This is the earlier 0.788 that forced `BODY_CHANNELS = (0,)`.
- Roll/pitch/yaw could not be screened yet: `collect_ik.py` recorded `head` as position only, no orientation (later fixed, see F63).
- Balance note: with `clips_per_body hexapod=7`, training set is 2,780 hexapod frames vs B1's 1,143 (2.43:1 frames), while `balance_embodiments` equalizes gradient steps (repeats B1 ~2.4x/epoch), not data. Probe/UMAP read the uncapped directory and see 5.9:1 — any quoted ratio must specify which.

Scripts: scripts/diagnostics/channel_screen.py

---

### F62. A CPG-based joint-space oscillator gives the hexapod a commandable gait (straight, turn, sideways) matched in Froude speed to the B1; `--turn`/`--turn_bias` don't steer and were removed

`--turn_bias` (measured 2026-08-20) steers at most 8 deg against 30 deg of natural wander — it doesn't work because it offsets joint position, not stride length.

Working approach: joint-space oscillator ported from `student_Locomotion_Control_olaf_6legs`, two sinusoids a quarter-cycle apart per leg, signs flipped between tripod groups. `--gait cpg` in `sim/collect/collect_ik.py`. Drive pattern is a tripod (FL HL MR vs ML FR HR) but contacts are not a clean tripod (within-group agreement 0.641±0.008, across-group 0.691±0.007 vs 1.0 for a clean tripod) — this matches the lab's own Olaf scene, so it's the pattern's behaviour, not a porting bug.

Corrections found by measuring, not tuning:
- `--turn` doesn't steer, it brakes (scales down one side's amplitude, shortening stride on that side); heading change is non-monotonic (+0.3 -> +2 deg, -0.3 -> +14 deg) vs `--spin` which gives -73 deg. `--turn`/`--turn_bias` removed from `collect_ik.py` 2026-08-22; use `--spin` for turning.
- Heading must be read from `/abdomen` quaternion (unwrapped), not `/head` orientation (sways 129 deg/stride) or Euler angles (gimbal lock near beta=-84deg). `collect_ik.py` now records `body_quat` per frame; nothing before 2026-08-21 has it.
- Sideways gait needs fore-aft swing turned off (splay-lift antiphase `--ft_phase 0.5`, not 0.125 inherited from forward walking, which lost 80% of stroke to slip). `--spin_amp` added since `--spin` previously multiplied by the fore-aft amplitude (0 in this gait) and had no effect.
- Left and right sideways gaits are not mirror images (animal's leg pairs are asymmetric: 0.771/0.489/0.638 long) — collect/measure each direction separately.
- Extend amplitude `A[2]` ceiling of 0.30 (three joints clip past limit at 0.35+) was an artifact of compressed standing pose at `--scale 0.5`; at `--scale 0.65` headroom nearly triples (0.29->0.51 rad).
- IK needs `--ik_iters 8` (not 1); one-shot solver reports false "unreachable" residuals on any re-timed path. Residual dropped from 0.28/36.97mm to 0.00/0.00. Every dataset before 2026-08-21 carries the old residual.
- Oscillate around the animal's actual walking-pose mean (from IK commands), not the scene's default spawn pose — otherwise body height varies by behaviour (0.284 vs 0.103–0.248m) and Froude (which divides by height) becomes incomparable.
- Mirror ALL three joints (not just fore-aft sweep) between body sides, or lift joints fight across the midline (heading error -89 -> -1 deg after fixing).
- `walk_check` (start/end displacement) cannot catch this kind of looping/arcing error; watch the rendered clip. Net heading change is now printed per clip.

Adopted settings: `--scale 0.65` (matches hexapod Froude 0.131 to B1's `--vx 0.30` Froude 0.135; body height 0.176 vs recorded 0.129 — deliberately matched to B1, not the source insect). Speed ladder uses `--cycles` (5.8 to 8.8) to span the B1's Froude range while keeping height flat (within 3%); hexapod runs ~5% slow throughout (systematic bias). Sideways gait cannot use `--cycles` above 6 (destroys it) so sideways sits at a different, unmatched speed (Froude 0.119).

Final sideways settings (`--scale 0.65`):
right: `--amps 0.00 0.20 0.30 --ft_phase 0.5 --strafe 0.8 --spin -0.24 --spin_amp 0.25` (0.52m±0.06, -1deg, purity 1.00)
left: `--amps 0.00 0.20 0.30 --ft_phase 0.5 --strafe -0.8 --spin 0.19 --spin_amp 0.25` (0.30m±0.01, -0deg, purity 0.98)
both with `--gait cpg --ik_iters 8 --scale 0.65 --symmetric`

Behaviour set: straight, `--spin` at four levels (turn), two sideways gaits, `--cycles` at six levels (speed), all at `--scale 0.65`.

Four failed routes removed from `collect_ik.py` on 2026-08-22 (code no longer exists): `--gait tripod`/`synth_tripod` (residual 365mm, ellipse foot-path box doesn't match true reachable shell), same with planted-frame-only sweep (343mm), re-timed recorded wave into tripod (travel -0.43m, body at 40deg — leg paths for one gait don't support another), levelled planted feet (345mm — abdomen frame pitches with body, flat ground looks sloped in it).

The tripod-capable CPG gait and the original recorded variable-phase wave gait (concentration 0.07-0.24 vs B1's 0.99-1.00) should be collected as separate conditions, not conflated.

---

### F63. Yaw rate, nondimensionalized as ŵ = ω·sqrt(h/g), can be matched between the hexapod and B1 across their full commandable range

Nondimensionalized turn rate (ŵ) lets the hexapod's `--spin` and B1's `--wz` be matched like Froude number matches speed.

Re-derived 2026-08-22 at `--scale 0.65` (F62; foot-path widening roughly triples turn rate per unit `--spin`):

| ŵ target | hexapod `--spin` | ŵ measured | B1 `--wz` | ŵ measured | gap |
|---|---|---|---|---|---|
| ~0.007 | 0.05 | 0.0081±0.0006 | 0.00 | 0.0067 | +21% (small denominator, abs gap inside gait wander) |
| ~0.021 | 0.15 | 0.0200±0.0007 | 0.08 | 0.0209 | -4% |
| ~0.041 | 0.29 | 0.0388±0.0009 | 0.19 | 0.0407 | -5% |
| ~0.077 | 0.56 | 0.0736±0.0004 | 0.40 | 0.0772 | -5% |

Hexapod holds Froude 0.132/0.131/0.122/0.121 across the yaw range vs B1's 0.120-0.135 — speed and yaw stay separable channels. (Table above supersedes an earlier version measured at `--scale 0.5`, same qualitative match.) At `--spin 1.2`/`--scale 0.65` Froude drops to 0.054 — independence holds only within the adopted range.

Speed gap between the two gaits (hexapod Froude ~0.10 vs B1 ~0.17) is closed by slowing the B1 (`--vx 0.22` gives Froude 0.101, matching hexapod default `--cycles 6`), not speeding the hexapod, since turn rate and speed are independent on the B1 (ŵ unchanged across `--vx` 0.22 vs 0.40) but the hexapod's sideways gait breaks above `--cycles 6`.

Final matched set: hexapod `--cycles 6`, `--spin` 0/0.4/0.8/1.2 vs B1 `--vx 0.22`, `--wz` 0.00/0.08/0.19/0.40 — matched at both ŵ and Froude (~0.10).

Measuring yaw required reading `/abdomen` quaternion (unwrapped trend), not `/head` orientation (off by -151deg due to 129deg/stride sway) or Euler angles (off by -99deg near gimbal lock at beta=-84deg). `collect_ik.py` now records `body_quat` per frame; data before 2026-08-21 lacks it, which is why roll/pitch/yaw could not be screened before (F61).

---

### F64. Equalizing per-leg foot heights in the body frame makes the hexapod gait worse, because the body frame itself rocks

Unequal leg lengths (0.771/0.489/0.638 for front/middle/hind pairs) cause front feet to lift 0.111 vs middle's 0.045 and reach deeper, so contact pattern follows leg length rather than commanded phase (contact bars short/broken for middle legs).

`scripts/diagnostics/tune_legs.py` was built to fix this (per-leg gain + offset solved against kinematics so every foot rises/bottoms equally) — it converges cleanly (lift spread 0.072->0.0000, depth spread 0.056->0.0001) but makes the gait measurably worse:

| | in-group | across-group | feet down | duty spread | forward |
|---|---|---|---|---|---|
| untuned | 0.641±0.008 | 0.691±0.007 | 3.21 | 0.227 | 0.37m |
| tuned | 0.451±0.010 | 0.511±0.003 | 2.90 | 0.465 | 0.22m |

Duty spread — the exact quantity targeted — doubles. Cause: correction is computed in the body frame, which itself rocks (body attitude swings 3.5x more under tuned settings, hip height 0.114m vs 0.132m). Equalizing feet against a rocking frame feeds the rocking rather than fixing it. General lesson (same failure mode as one of F62's failed routes): a gait is a closed loop through the body; per-leg geometry correction is open-loop and insufficient. Proper fix needs pose solved against the world frame or contact-adapted timing (as in the lab's actual CPG, Larsen et al. 2023), not implemented here.

`--legtune` is wired into `cpg_commands` but left off by default. Adopted settings unchanged: `--gait cpg --ik_iters 8 --amps 0.25 0.20 0.30 --ft_phase 0.125 --symmetric`.

Scripts: scripts/diagnostics/tune_legs.py

---

### F65. Hexapod and B1 clips were recorded at different effective frame rates (20Hz vs 50Hz), invalidating every prior cross-embodiment number computed on them

Insect collector records at 20Hz; `render_b1_replay.py` rendered one frame per MuJoCo step (50Hz). Neither clip format stored a rate, so the mismatch was silent.

| | frames | duration | per-transition dt |
|---|---|---|---|
| hexapod | 66 | 3.30s | 50ms |
| B1 (`data/allocentric/fwd_b1_50hz`, and prior B1 passes) | 99-126 | ~2.0-2.5s | 20ms |

Since the ITM operates on `(e_t, e_t+1)` transitions, a "transition" was 2.5x longer in wall-clock time on the hexapod than the B1. This affects all prior results computed across the pair: F37/F40 (below-chance cross-embodiment sharing), F44 (forward model worse than no-motion baseline), F51 (per-channel AUCs), F39 (pairing feasibility) — not wrong about their given data, but the data wasn't comparable. Also interacts with F61: stride-averaging (19 frames hexapod vs 48 frames B1 per stride) partly corrected for this rate mismatch without anyone knowing, explaining why it helped more than expected.

Fix: subsample frames (`--fps 20` on the replay), not change physics — rollout stays 50Hz, renders every 2.5th step. Two silent pitfalls in the first fix attempt:
- Proprioception must be indexed by frames actually rendered (`T[k][:n]` is wrong under subsampling — desyncs vision/proprioception with no error raised).
- `dt` must be derived from the requested rate, not the first frame gap (2.5-step stride rounds to alternating gaps of 2/3, giving a 20% error if read from `idx[1]-idx[0]`).

`--max_frames` added so all conditions yield equal clip length regardless of speed. B1 set is now 66 frames at 3.30s, matching the insect, Froude back on calibration (0.133 vs 0.128 expected, 0.217 vs 0.209).

`data/allocentric/fwd_b1_50hz` still carries the old (wrong) rate — kept for reproducibility of results that cite it, but nothing new should be measured against it.

---

### F66. F63's |ŵ| yaw match hid that the two robots turn in opposite signed directions; fixed by re-collecting hexapod turns at negative `--spin`

F63's dimensionless yaw match (ŵ) compared magnitudes only, so it could not detect sign. Signed yaw rate (deg/s):

| | speed conditions | sideways | turn conditions | sign consistent |
|---|---|---|---|---|
| hexapod | +0.49±2.66 | -0.51±0.76 | -14.89±10.41 | yes, negative |
| B1 | +2.16±0.29 | +1.28±0.61 | +8.74±6.16 | yes, positive |

Positive `--spin` yaws the hexapod negative, opposite the B1's `--wz` sign convention. Pooled over 12 conditions, signed yaw alone separates the robots at AUC 0.871 — a shared body target trained on unsigned-matched yaw would let the head learn embodiment identity from sign instead of turn rate (same shortcut as F37/F40).

Also found: B1 yaw is positive in every condition including straight-commanded ones (+2.16 deg/s forward, +1.28 sideways, sd 0.29) — a constant bias, not scatter (hexapod scatters about zero, +0.49±2.66). This bias is left in and declared, not corrected (see F69 for the actual controller fix).

Fix applied here: re-collected the hexapod's four turn conditions at negative `--spin`. Rule: a quantity matched across robots must be matched as a signed vector in a shared world frame, with direction convention checked against actual measured travel — `|ŵ|` agreeing to 5% is compatible with opposite turn directions.

---

### F67. On the frozen encoder, re-screening channels on matched (condition-held-out) behaviour shows forward speed transfers and yaw's apparent transfer was a train/test leak

`data/allocentric/beh12_*` adds 12 matched conditions/robot (speed, turn, sideways; 4/4/4; 48 clips/side) to escape F61's "other channels are constants" issue. Screened via `screen_behaviour_channels.py` on the frozen encoder (`s2_fwd_hex7-b1_body0.5`, which predates F62's wider foot path and F65's frame-rate fix — so this is not the final data/encoder state).

Key methodological finding: within a condition, the 4 clips agree to 2-10% of the between-condition spread (near-duplicates), so a clip-level 70/30 split leaks and must be replaced with a condition-level split. Results (5 seeds, smoothed, frozen encoder, held out by condition):

| channel | robot AUC | hex->b1 | b1->hex |
|---|---|---|---|
| forward | 0.66±0.11 | +0.36±0.10 | -1.08±1.34 |
| lateral | 0.68±0.04 | -0.16±0.44 | +0.04±0.45 |
| vertical | 0.63±0.03 | -1.72±0.57 | -1.94±0.49 |
| yaw | 0.72±0.10 | -0.82±0.23 | +0.10±0.19 |

Yaw's clip-split result (+0.31±0.06, looked like the strongest finding) was an artifact of the leak — by-condition it is +0.10±0.19, i.e. zero. Forward speed is the opposite: it strengthens under the harder split (+0.21±0.13 clip-split -> +0.36±0.10 condition-split) — a genuine cross-embodiment transfer, the same channel F61 found.

Scope: this measures the frozen (untrained) encoder only. F59 shows forward speed goes from 0.31 frozen to 0.85-0.92 trained, so frozen-zero for lateral/yaw/vertical does not mean they are unusable — see F68.

---

### F68. F67's screen only measured the untrained baseline; judging channels by frozen-encoder value alone would also have rejected forward speed, which works after training

F59 numbers for forward speed: frozen encoder correlation 0.31, trained without body term -0.01, trained with shared body head 0.85/0.90/0.92. Judging by the frozen value (0.31) would have wrongly condemned the one channel known to work; yaw's frozen value is 0.10 (comparable), and no experiment has yet trained on it.

Correct statement of F67's result: "no new channel is shareable without being taught to be" — not "no new channel is shareable."

Tested and ruled out: that the dimensionless group for yaw (ŵ = ω·sqrt(h/g), using hip height) is the wrong length scale. B1 is 3.19x taller; hexapod stance is 1.39x wider (mean stance radius 0.576m vs 0.414m) — height and stance-radius scales differ by 4.4x in their robot-ratio. But `transfer()` standardizes both target vectors before fitting, so any affine rescale (including a length-scale swap) cancels exactly:

| yaw, smoothed | hex->b1 | b1->hex | robot AUC |
|---|---|---|---|
| height scale | -0.630 | +0.270 | 0.637 |
| stance-radius scale | -0.640 | +0.269 | 0.571 |

Transfer numbers identical to 3 decimals; scale choice only affects the robot-identity gate (stance radius hides embodiment better: 0.571 vs 0.637), not whether yaw transfers. Comparing scales against already-collected conditions is circular (collection itself was tuned to the height scale) — the only valid test is to collect fresh data under each scale.

Open question (not yet run): train on the beh12 data and re-measure yaw, following F59's precedent that training closed a much larger gap than exists between any two frozen channels.

---

### F69. The B1's heading controller was proportional-only, producing a constant yaw bias; the integral-term fix initially produced matched averages hiding an oscillation, so verify against per-clip time course

B1 yaw was positive in every condition including straight-commanded ones (+2.16 deg/s forward, +1.28 sideways, sd 0.29 — a constant, not scatter), vs hexapod's straight-walk scatter about zero (+0.49±2.66). Pooled, this constant separates the robots and would let a shared yaw target learn embodiment identity instead of turn rate (same shortcut as F37/F40).

Cause: `rollout_b1_mujoco.py` commanded yaw as `HEAD_K * yaw_err`, `HEAD_K = 0.5` (P-only) — a P controller cannot reject a constant disturbance (the policy's inherent turn bias) below disturbance/gain. Affects every clip ever rolled out.

First fix attempt (`ki = 5.0`) matched the target mean (0.0732 vs 0.0736, 0.5% match) but was wrong: turn rate inside a clip oscillated (0.19, 0.017, 0.024, 0.15, 0.035 — ~3s period against a 162-step clip), i.e. controller ringing labelled as a steady turn. Rule: verify a matched quantity by its time course inside one clip, not only its mean.

Tuned against standing drift, turn-level accuracy, and ringing simultaneously:

| kp / ki | standing drift | turn level 3 | ringing (sd/mean) |
|---|---|---|---|
| 0.5 / 0 (shipped) | 0.0064 | 0.0604 | 0.06 |
| 0.5 / 5.0 | 0.0001 | 0.0616 | oscillates, ~3s period |
| 2.5 / 1.0 (adopted) | 0.0000 | 0.0743 | 0.04 |
| 4.0 / 1.5 | -0.0005 | 0.0720 | 0.02 |

Adopted: `--head_kp 2.5 --head_ki 1.0`. No standing drift, turn matched to +1%, ringing 0.04 (less than the hexapod's own gait, 0.10). Re-collected: target reads hexapod 0.0148/0.0353/0.0736 vs B1 0.0146/0.0346/0.0714 (within 3%); non-turning conditions sit at ±0.002 on both robots (was +0.007 on B1 only).

Remaining real asymmetry: at the hardest turn, hexapod forward speed falls to 0.028 while B1 holds 0.096 — a sharp turn costs the insect speed and the quadruped nothing. This is morphology, not a controller defect, but means the forward channel carries some embodiment information at that condition (relevant to the robot-AUC gate). Note: this finding is superseded/refined by F70, which shows the apparent forward-speed collapse under turning was actually a world-frame measurement artifact, not morphology.

---

### F70. "Forward speed" was measured in the world frame, so it partly encoded heading rotation rather than walking speed; fixed by projecting into the body frame

`body_motion` differenced world x/y. The apparent forward-speed collapse under hard turns (hexapod 0.132->0.026 across turn levels vs B1's 0.131->0.102, previously attributed to morphology in F69) was an artifact: world-x speed is walking speed times cos(heading offset from world x). Straight-walking datasets never exposed this since both robots start aligned with +x.

Projected onto each robot's own heading (body frame):

| | world-x hexapod | body-forward hexapod | world-x B1 | body-forward B1 |
|---|---|---|---|---|
| turn level 0 | 0.132 | 0.135 | 0.131 | 0.131 |
| turn level 3 | 0.026 | 0.128 | 0.102 | 0.132 |

Neither robot actually slows when turning — both hold constant speed across all turn levels. The world-frame version would have taught the shared body head that turning means slowing, by different (embodiment-specific) amounts — a free embodiment cue in the one channel that had worked.

Also fixes lateral: in the body frame, sideways conditions read -0.114 vs -0.114 and +0.162 vs +0.159 (hexapod vs B1, signs and magnitudes agreeing). In the world frame, lateral meant "world y" regardless of heading and could never match — this was a major cause of F51's AUC 0.788 finding that lateral was "an embodiment label in disguise"; that conclusion was drawn on a partly-orientation quantity, not a clean lateral-velocity measurement.

Separately: the hexapod's abdomen z-axis points aft, so forward-axis projection must be negated (raw projection gives -0.135 hexapod vs +0.131 B1 for the same behaviour). `forward_axis()` now carries this correction, checked against the direction straight walking actually travels.

Rule: body-relative quantities must be computed in the body frame with orientation verified against measured motion; world-frame differencing only works when every robot walks straight along the same axis (true of old datasets, not a property of the quantity itself).

---

### F71. Much of the B1's apparent "character" (sideways drift, lean-into-turns) is an artifact of the specific trained policy, not the robot body

`sim/assets/b1_policy/` has two trained policies: `base_gait3` (2.0 Hz, previously the only one used) and `base_1.7hz_sym` (1.7 Hz).

| at `--vx 0.30` | gait3 | sym | hexapod |
|---|---|---|---|
| lateral drift | -0.022 | +0.004 | -0.04 to +0.01 |
| standing yaw | 0.0049 | 0.0008 | 0.0029 |
| stride rate | 2.00 Hz | 1.67 Hz | -- |

The B1's constant sideways drift and its lean-into-turns (previously explained morphologically as narrow-stance leaning in vs. hexapod's wide-stance swinging out) are both properties of `gait3` specifically — `sym` shows flat lateral (+0.004-0.005) across all turn levels where `gait3` grows to -0.040 at the hardest turn. Two policies disagreeing on the same body proves the body isn't causing it. The hexapod's outward swing (opposite sign, growing to +0.044) is not reproduced by either B1 policy and stands as the one real difference. `sym` also tracks matched turn levels to 1.5% vs `gait3`'s +3 to +23% error.

Decision: both policies kept, two clips each per condition (not four of one), because within-condition clips were previously found to be near-duplicates (F67) — using two policies gives genuinely different within-condition dynamics (forward spread rises to 0.112-0.122) rather than four copies of one limit cycle. Cost: half the clips carry `gait3`'s lateral bias, acceptable only because lateral is excluded from the shared target — this makes that exclusion permanent rather than provisional.

---

### F72. The ITM cannot run at control time (needs the next frame); the correct closed-loop design is LAC-WM's action projector sampling in action space, and adaptation requires LoRA on the FDM too, not just a frozen projector

Original plan (`e_t -> selector -> z_t -> Motion Decoder -> a_t`, sampling in latent space) is structurally broken: our ITM computes `z_t = ITM(e_t, e_{t+1})`, which needs the next frame — the thing being decided at control time. Every transfer number and probe measured so far reads `z` off two ground-truth frames (reconstruction, not control), consistent with `predict_actions.py`'s framing.

LAC-WM's actual method (per its paper) trains an action projector mapping explicit actions into latent action space, so the world model consumes raw actions directly:

| | sample in latent space (rejected) | sample in action space (LAC-WM, adopted) |
|---|---|---|
| flow | z -> MD -> a | a -> projector -> z -> FDM |
| every candidate executable? | no | yes, by construction |
| decode z->a at run time? | required, MD must generalize to new body | not needed |
| Motion Decoder's role | the controller | auxiliary loss during pretraining only |

Sampling in action space removes dependence on the Motion Decoder generalizing to an untested body. New closed loop: sample candidate actions, project them, roll the FDM, score, execute the winner directly.

Corrected claim: "a new body needs only video" is stronger than LAC-WM's own claim (finetuning for unseen embodiments) — video lets the world model span incomparable bodies, but the projector still needs actions from the target robot (cheap, not free; F45 measured one B1 clip clears break-even, nine clears every horizon).

Correction to an earlier draft of this entry: adaptation also fine-tunes the FDM, not just the projector. LAC-WM Section 3.2, three stages, LoRA rank 2: (1) fine-tune IDM+FDM end-to-end with LoRA rank 2; (2) freeze FDM, train action projector from scratch; (3) jointly fine-tune projector+FDM with LoRA rank 2. At inference, only the projector+FDM run. So a new robot costs a projector AND LoRA adapters on the FDM, not a projector against a fully frozen world model. `fit_projector.py` currently freezes both (conservative version) — LoRA fine-tuning is the documented fallback if frozen doesn't work, not an untested invention.

---

### F73. The shared body head causes cross-embodiment transfer channel by channel: a channel transfers only when it is in the supervised target, and adding a second channel trades off against the first

Three arms on `data/allocentric/beh12_*`, identical except for the body loss term. Held out by condition, frozen `best.pt`, smoothed (single-seed initial readout):

| | forward hex->b1 | forward b1->hex | yaw hex->b1 | yaw b1->hex |
|---|---|---|---|---|
| control, no body term | -28.918 | -43.075 | -45.254 | -29.550 |
| body head, forward only | +0.761 | +0.641 | -8.872 | -9.791 |
| body head, forward+yaw | +0.482 | +0.323 | +0.606 | -0.031 |

Without a body term, nothing transfers (far worse than predicting a constant) even though both robots perform matched behaviours at matched speeds — matched behaviour alone does not produce a shared code. F59's forward-only result survives the F65 frame-rate fix and improves (published +0.54/+0.68/+0.75 vs. here +0.761/+0.641).

A channel transfers only when supervised: yaw is -8.9/-9.8 when absent from the target, +0.606 when present, identical data/architecture otherwise.

Training loss and the `probe` (embodiment-identity probe) cannot see any of this — val/motion loss nearly identical across arms, and probe saturates at 0.936-0.994 in all three arms (z is near-perfectly separable by embodiment regardless of body term). Identity in z and shared-readout direction are independent; only the post-hoc transfer screen measures the latter.

Side effect: the body term cuts per-embodiment action error 38% (val motion 0.3517 -> 0.2183) at no reconstruction cost (1.5580 -> 1.5400) — matches LAC-WM's claimed "mitigates shortcuts" mechanism.

Five-seed condition-level splits refine the single-seed numbers substantially:

| arm | channel | hex->b1 | b1->hex | seeds positive |
|---|---|---|---|---|
| forward-only | forward | +0.610±0.140 | +0.573±0.240 | 5/5, 5/5 |
| forward-only | yaw | -5.230±2.383 | -10.268±3.059 | 0/5, 0/5 |
| forward+yaw | forward | +0.196±0.278 | +0.400±0.107 | 3/5, 5/5 |
| forward+yaw | yaw | +0.367±0.274 | -0.415±0.556 | 4/5, 1/5 |

Causal claim (supervision causes transfer) holds with no overlap between arms. But yaw only reaches ~zero (not usable transfer) when supervised — the single-seed +0.606 overstated it (see F75 for why: yaw's noise floor differs by robot). Adding yaw costs 68% of forward's hex->b1 transfer (0.610->0.196, 5/5->3/5 seeds positive). Practical conclusion: use forward-only target.

Ruled out as the cause of the yaw/forward tradeoff: "yaw carries less signal" — both channels have equal signal share (~0.86, between-condition sd / total) when decomposed into between-condition vs. within-clip variance; yaw's between-condition spread is actually larger. The real issue is yaw's variance concentrates in ~6 of 12 conditions (14 conditions near zero), so a condition-holdout split removes most of yaw's test signal — an evaluation-leverage effect, not model instability.

Also ruled out: longer smoothing window recovering yaw. Tested 1.0s (default) vs 1.5s vs 2.5s — monotonically worse (2.5s: yaw +0.606/-0.031 -> -0.116/-0.465); a 2.5s window on a 3.30s clip averages signal away with noise. Default smoothing is already at the useful limit.

The forward/yaw competition itself remains unexplained (capacity/optimization candidates untested).

---

### F74. Transfer R^2 is not comparable across the old (fwd_hex7speed) and new (beh12) datasets; three attempts to reconcile F73's -28.9 control vs F59's published -7.083 each found a different confound, so within-dataset comparison replaces cross-dataset comparison

| attempt | result | why it doesn't compare |
|---|---|---|
| as published | control -28.9 vs -7.083 | F73 holds out whole behaviours (condition-level split); F59 held out clips of behaviours already trained on (clip-level split). F67 showed this gap directly: yaw read +0.31 by clip, +0.10 by condition on identical data |
| same split protocol | control -16.7 vs -7.083 | new data asks for forward speed while the robot turns/strafes in 8 of 12 conditions; old data was forward-only, so "predict forward speed" meant predicting the only thing that varied |
| same split, speed-only conditions | body head -0.98/+0.32 vs +0.54/+0.44 | restricting to speed conditions removes variance rather than isolating the comparison |

Variance comparison explaining the third attempt: forward sd is 0.066 (all 12 conditions, 48 clips), 0.040 (speed-only, 16 clips), 0.045 (old `fwd_hex7speed`, 91 clips). R^2 is variance explained — the speed-only subset has less variance and a third of the clips of the old set, so it tests on ~5 clips over a narrower range; same representation scores +0.70 on the full set vs -0.98 on this subset with nothing about the model changed.

Conclusion: the two datasets differ in composition, dynamic range, clip count, frame rate, and split protocol simultaneously — no slice holds them all fixed. Stop trying to compare across datasets. Valid comparisons must hold dataset/split/seeds fixed and vary only the loss term (as F73 does: -16.7 -> +0.70 within one dataset). General rule: transfer R^2 is a joint statement about a representation and the distribution it's scored on; quoting one across datasets quotes half a measurement (same trap as `motion` MSE being incomparable across datasets/`action_lag` settings, documented in `wm/README.md`).

---

### F75. Yaw's transfer ceiling (from F73) is set by a real, uncorrectable noise-floor asymmetry: the hexapod's wide-stance gait has 3x the B1's yaw wander when not turning

Yaw is matched (AUC 0.506, indistinguishable between robots) in the 4 conditions where both robots actually turn, but mismatched in the 8 non-turning conditions (AUC 0.588): hexapod yaw sd is ~3x the B1's there (speed conditions: hexapod ±0.0160 vs B1 ±0.0050), even after stride-window smoothing — i.e. between-stride wander, not gait rocking. This explains F73's direction asymmetry exactly: a readout fitted on the B1 (nearly-constant yaw when not turning) learns little and fails on the hexapod's wander (-0.415); fitted on the hexapod it partly learns real turn signal (+0.367).

Hypothesis tested and refuted: that this asymmetry comes from us giving the B1 a PI heading controller (F69) while leaving the hexapod's oscillator open-loop. Added `--head_kp/--head_ki` to the insect collector to test:

| gains | yaw sd | lateral | forward | wander |
|---|---|---|---|---|
| open loop | 0.0130 | 0.04 | +0.55 | 1.76 |
| kp 0.5 ki 0.2 | 0.0141 | 0.02 | +0.57 | 1.72 |
| kp 1.0 ki 0.4 | 0.0149 | 0.02 | +0.56 | 1.70 |
| kp 2.5 ki 1.0 | 0.0177 | 0.00 | +0.70 | 1.48 |
| B1 | 0.0050 | | | |

Every gain setting makes travel better and yaw variability worse (controller steers continuously, reducing net drift but increasing instantaneous rate variance). No setting closes the gap: floor is open-loop's 0.0130, still 2.6x the B1's 0.0050. Conclusion: the gap is the hexapod's gait (wide stance swings body more per stride than a compact trot), not a missing controller — a real, uncollectable-away difference between the robots.

Decision: the heading controller is kept in the codebase but defaulted off. It improves the walk (lateral 0.04->0.00, forward +27%, wander 1.76->1.48) but raises yaw noise and shifts Froude 0.126->0.162 (would require recalibrating the speed ladder) — net negative for this experiment.

Separate finding from the same test: forward speed separates the robots at AUC 0.869 in sideways conditions (hexapod creeps forward while strafing, 0.013-0.029; B1 doesn't, 0.002-0.009) — a real capability difference, meaning the forward channel identifies the robot in a third of conditions.

Broader conclusion: this collapses a planned two-arm "clean vs dirty" robustness-test design (no artifact to remove, so the two arms would be identical). Confirms the pipeline currently has no working mechanism for suppressing nuisance embodiment differences — F38's three invariance methods didn't move transfer, the adversary shifts `probe` without changing transfer, and the body head leaves identity fully decodable (probe 0.94-0.99 in every arm including control). Cleaner collection is currently the only lever available.

---

### F76. Within-condition diversity must be measured on the input (commands/embeddings), not the target value; the hexapod's 4 clips per condition were near-identical commands, and fixing this failed on two tried parameterisations

The B1 got a second policy for genuine within-condition gait diversity (F71); the hexapod's equivalent (`--episodes 6,926,521,625`, 4 expert episodes) did not, because F50 found the expert is one gait (1.9% speed variation across 1000 episodes) — the four episodes' standing poses differ by only 0.0007 rad.

Target-based measurement wrongly reported "no asymmetry": within-condition sd of forward target is similar for both robots (hexapod 0.0016-0.0062, B1 0.0006-0.0088). But input-based measurement (mean pairwise correlation of joint commands) shows they are not equivalent: hexapod 1.000 (identical command sequences), B1 0.127 (genuinely different gaits). Target spread doesn't measure input diversity — the same command sequence can vary in target only via the physics response, and matched-target/different-gait clips are actually the ideal, not a flaw.

Re-reads F71: B1's larger target spread was partly its two policies not being speed-matched (smearing the condition), not pure diversity.

Command correlation (0.0007 rad bias difference) is not truly negligible — legged contact dynamics amplify it (heading changes of +21, +21, -5 degrees across repeats of the "same" hexapod config). Measured properly via V-JEPA2 embedding within/between-condition distance ratio: hexapod 0.389 (6.430/16.518), B1 0.622 (16.888/27.151). B1's diversity is real (2.6x further apart absolute, 1.6x relative) but the hexapod is not degenerate (0.389 means meaningfully different clips, not repeats) — asymmetry is moderate, not severe.

This input-diversity asymmetry is NOT what limits yaw transfer (F75's noise-floor mismatch is) — b1->hex fails despite B1 having more varied input, so diversity and noise-floor are separate problems; conflating them cost three collection attempts.

Two new hexapod parameterisations were tried and both failed to add real diversity while keeping targets matched:
- `--ft_phase 0.125` vs `0.0` (speed/turn, 8 conditions): correlation only 1.000->0.935 (an eighth-cycle phase shift is a perturbation, not a second gait), and it doesn't survive the speed ladder (matched -7% at cycles 5.8, drifts to -39% at c8.8).
- lift 0.20 vs 0.24 (sideways, 4 conditions): pure amplitude scaling, correlation stayed exactly 1.000, no variation added.

Two rejected axes: stride rate (Froude matches exactly at two settings but the faster variant needs cycles>10, past where drift sets in); `--lead` (inert, Froude 0.124-0.127 across 0.20-0.35 range).

Root cause: the CPG is a single generator where every parameter couples to speed (can't change how the gait walks without changing how fast). A real fix needs a different phase pattern (e.g. metachronal wave vs. tripod ordering) — an implementation task, not a parameter sweep. Left undone. Failed collection sits in `data/allocentric/beh12_hex2/`, not merged.

---

### F77. Without a shared body-head target, the forward model (FTM) barely reads the latent z; the body loss term (not reconstruction) is what forces z to carry information, despite being the smallest loss term by value

`L_recon` predicts `e_{t+1}` from `e_t` and `z_t`; for periodic constant-speed gait, `e_{t+1}` is largely guessable from `e_t` alone, so nothing forces reliance on z. Measured with `ftm_uses_z.py` (sweep one input, hold the other fixed):

| control arm | sweep z | sweep e | frame dominates | z / one step |
|---|---|---|---|---|
| speed conditions | 1.376 | 38.567 | 28x | 0.027 |
| turn conditions | 1.654 | 37.985 | 23x | 0.033 |
| sideways conditions | 3.163 | 35.392 | 11x | 0.068 |

Latent accounts for ~2.7% of one-frame embedding movement; swapping in a latent from a different behaviour costs only 0.25% (one step) / 0.6% (8-step rollout) in the control arm. Sideways (hardest to predict from previous frame) forces more z-usage than forward (easiest) — variety in behaviour puts pressure on the objective to actually use z.

With the forward body head, z-usage triples (sweep z 1.376->4.257, frame dominance 28x->9.3x, wrong-behaviour cost 0.25%->3.5%), while `recon` loss itself barely moves (1.5580->1.5400, looks like nothing in the log). This explains why cross-embodiment rollout gaps look small (`cross_latent_rollout.py`: own/other/random dynamic range only 3.8% with body head vs. wildly inconsistent without, e.g. 0.222/-0.420) — a near-ignored input has little room to show a gap.

Gradient analysis (`loss_gradient_balance.py`, dL/dz for each term on the same z) shows loss VALUE and loss GRADIENT INTO Z diverge sharply:

| term | lambda | share of loss | |dL/dz|×lambda | share of gradient |
|---|---|---|---|---|
| recon | 1.00 | 79.3% | 0.0004 | 5.1% |
| motion | 1.00 | 0.6% | 0.0011 | 12.3% |
| body | 0.50 | 20.2% | 0.0071 | 82.5% |

(Note: the 82.5% figure is overstated due to `body_motion` being standardized on a single-robot batch rather than the pooled `body_stats` train.py actually uses; rescaled estimate is ~body 50%, motion 37%, recon 13% — but recon remains small either way, and the direction of the finding holds.)

Consequence: lowering `lambda_recon` would NOT help (it's already a small gradient contributor despite being most of the loss value) — ruled out. The real lever is either raising `lambda_body` (has the largest raw |dL/dz| of the three terms, 0.0019 vs motion's 0.0011 and recon's 0.0004; currently halved by lambda_body=0.5 to 38.8% share) or widening the ITM frame pair via `cfg.frame_stride` (implemented 2026-08-24) to make next-frame prediction genuinely harder — but see F78, this was tried and backfires.

Consequence for planner: if z moves the prediction by only 3-8%, candidate actions in a planner score nearly identically — not fatal (body-head arm is 3x better than control) but a planner built on the no-body-term control arm would have had no signal at all.

---

### F78. Widening the ITM frame pair (frame_stride/action_chunk) does make the FTM read z more, but turns z into a robot-specific clip identifier — transfer collapses to zero. Fix rejected; kept in code but off by default.

Following F77's prescription to widen the training pair, implemented as `cfg.frame_stride` then `cfg.action_chunk` (first attempt broke the decoder due to a stride/chunk mismatch).

| hexapod, speed conditions | stride 1 | stride 5 + chunk | stride 10 + chunk |
|---|---|---|---|
| sweep z (FTM latent usage) | 4.257 | 7.662 | 12.279 |
| z per step | 0.083 | 0.150 | 0.240 |
| cost of wrong-behaviour latent | 3.5% | 6.0% | 5.6% |
| joint decoder, val motion @1 step | 0.218 | 0.906 | 0.879 |
| transfer, forward smoothed, seed 0 | +0.826/+0.328 | -0.102/+0.050 | -0.447/+0.102 |

FTM's use of z nearly triples as intended (F77's prescription works for what it targeted), but the joint decoder degrades to predicting the mean, and cross-embodiment transfer drops to ~zero (not catastrophically negative like the no-body-term control — just absent).

Ruled out as the cause: mismatch between z summarizing k steps and L_motion scoring against one step — chunking the action target fixes this exactly but barely changes the decoder (0.911->0.906) while still doubling sweep z (3.728->7.662). The mismatch was real but not the cause of the collapse.

Actual cause (four consistent signals): a wider pair turns z into a clip identifier rather than a movement code — sweep z rises 2.9x (an identifier predicts its own clip's future well), decoder trains to 0.06 but validates at 1.28 (memorization, no generalization), transfer falls to zero (identifiers are robot-specific by construction), and `probe` (embodiment-identity probe) rises 0.94->0.997 (z becomes MORE separable by embodiment). At stride 1, pair difference is dominated by gait phase (F16: 64%, generic); at stride 5-10 it's dominated by which clip this is (robot-specific).

This confirms/explains F47's earlier unexplained result (long-baseline arm wins every in-domain horizon, loses every cross-robot one).

Key tension identified: z being read by the FTM (F77) and z transferring across robots (F73) pull in opposite directions under most interventions measured — widening the pair improves the former, destroys the latter. The body term is the only intervention found that moves both in the same direction (sweep z 1.376->4.257 AND transfer -28.9->+0.610), because it adds a shared target rather than making prediction harder.

Kept in code: `cfg.frame_stride` and `cfg.action_chunk`, defaulting to 1 / "follow frame_stride" (i.e., off). No published number uses them.

---

### F79. Few-shot adaptation of a hexapod-pretrained forward model to a new robot (B1) is indistinguishable whether the pretraining backbone had 12 behaviours on 1 body or 1 behaviour on 4 bodies

Two hexapod-only backbones matched on clips/transitions, differing only in the axis varied:

| | bodies | behaviours | clips | transitions |
|---|---|---|---|---|
| `beh12_hexonly` | 1 | 12 | 48 | 2,779 |
| `m3d_body` | 4 | 1 (forward) | 48 | 2,860 |

Both frozen, both with shared body head, both adapted to the same 48-clip B1 set via `finetune_ftm.py`, identical budgets/splits. Result (9 clips, gap-closed ratio):

| horizon | 1 body, 12 behaviours | 4 bodies, 1 behaviour |
|---|---|---|
| h=1 | 1.25x | 1.25x |
| h=3 | 1.12x | 1.11x |
| h=5 | 1.01x | 0.99x |
| h=10 | 0.86x | 0.81x |

Indistinguishable across all budgets (1-9 clips) and the full horizon curve; closer to each other than to their own split-to-split variance (h=10 ranges 0.76-0.96 and 0.76-0.87). Displacement profile also matches (predicted/actual at 9 clips: 0.54/0.82/0.97/1.17 vs 0.53/0.82/0.97/1.23).

Conclusion: what transfers to a genuinely new robot's forward model is generic, not tied to morphological variety (more bodies) or behavioural variety (more behaviours) in hexapod pretraining. This does NOT extend F11's earlier finding (within the hexapod family / motion decoder, adding bodies helps, adding episodes of same bodies doesn't) — for cross-robot FTM adaptation, neither axis moves the number. Consequence: collecting more hexapod bodies or behaviours will not improve few-shot adaptation to a new robot; a third embodiment is still needed for the scaling claim and body-head-helps-new-robot tests (F99, F73), but not justified on this basis.

Scope: compares two pretraining sets against each other, not against no pretraining — the gap to random init is large and grows with budget (h=1: 1.25x vs 0.92x; scratch never clears break-even at any budget).

Secondary finding: the pretrained model is conservative at one step (predicts half the actual displacement, correct direction, scores 1.25x) but overshoots when rolled out 10 steps on its own output (falls to 0.86x/1.17 predicted-vs-actual). Scratch holds ~0.9 displacement ratio throughout, staying near a hold-still baseline — but this is NOT scratch "predicting no motion" (that would score exactly 1.00x by construction); the hypothesis is refuted.

---

### F80. The forward model CAN discriminate/rank candidate actions despite F77/F78's low sensitivity ratios — those ratios measured the wrong quantity (magnitude, not directional consistency)

F77 concluded from `sweep z` sensitivity ratios (frame outweighs latent 28x control / 9.3x with body head) that "every candidate scores nearly the same." This inference is refuted by direct measurement.

`plan_discriminates.py` tests the actual deployment path (projector + FDM only, no inverse model): given K candidate action sequences (one true), project through the fitted action projector, roll the FDM, score against true future. `beh12_hexonly`, 19 held-out clips, 12 conditions, 200 trials, top-1 rate:

| distractors from | phase | chance | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|---|
| another behaviour | free | 12.5% | 74.5% | 82.5% | 80.0% | 63.5% |
| another behaviour | aligned | 12.5% | 69.5% | 68.5% | 68.5% | 64.5% |
| same behaviour, other level | free | 25% | 73.9% | 76.9% | 75.4% | 62.7% |
| same behaviour, other level | aligned | 25% | 57.8% | 55.1% | 57.8% | 53.1% |

Hardest case (same-behaviour, phase-aligned, other speed level — distinguishing e.g. vx0.30 from 0.38/0.40/0.50): 57.8% vs 25% chance over 147 trials, ~9 standard errors, top-2 76.2%. The forward model has real, usable ranking signal.

Why the sensitivity ratios misled: `sweep z` measures magnitude of prediction change when z changes; ranking only needs the change to be consistently in the right direction. A small but consistently-ordered displacement ranks perfectly; a large noisy one doesn't — the two quantities are near-independent.

Two design conclusions:
- Withdrawn: an apparent "sweet spot" at horizon h=3 (free-phase: 74.5->82.5) was an artifact of the model exploiting gait-phase mismatch in distractors, which a real planner cannot rely on since all its candidates share a starting frame. Phase-aligned results are flat across horizons (69.5/68.5/68.5/64.5) — horizon does not actually matter.
- The action projector costs ~15 points of accuracy: an oracle "latent" arm (candidate's true ITM z) scores 10-17 points above the projector arm in every condition (hardest row: 72.8% vs 57.8%). This is the first measured price tag for LAC-WM's stage-3 joint projector+FDM fine-tuning (35k of their 60k adaptation iterations).

Scope: single robot (the one trained on), in simulation, ranking pre-recorded action sequences (not sampled ones), scored against true future (not a goal). Establishes the forward model carries usable ranking signal (a necessary precondition) but does NOT establish that the closed control loop actually works end-to-end.

---

### F81. A planner over recorded behaviours picks the right speed/lateral condition but not the right turn rate

`plan_open_loop.py` runs the control-time path (action projector + forward model, no inverse model) over frames of a held-out demonstration, scoring twelve recorded behaviours against the demonstration's frame `h` ahead. `beh12_hexonly`, 24 demonstrations, 478 decisions, horizon 5, replanning every third frame.

| | rate | chance |
|---|---|---|
| exact condition | 58.2% | 8.3% |
| right behaviour family | 90.0% | 25.0% |

By family (exact match): speed 9/9, sideways 6/6, turn 2/9 (every miss is turn->turn at wrong rate).

Turn-level separation is 2-4x better than speed-level separation (e.g. `turn_s0.29`/`s0.56` at 6.6x own-noise vs `speed_c7.1`/`c8.15` at 1.7x), yet turn resolves worse (2/9 vs 9/9) — the failure is specific to yaw, not fine-grained distinctions. Cross-family confusions: 17/20 are the slowest walk (`speed_c5.8`) taken for a turn.

Consistent with F73 (yaw transfer never becomes usable under supervision) and F75 (yaw noise floor is a gait asymmetry between robots) — three independent methods, one channel (yaw) that fails.

Scope: open loop over recorded (not self-generated) frames, so no compounding error; necessary but not sufficient condition — closed loop is untested here (see F82).

Scripts: `plan_open_loop.py`

---

### F82. Closing the loop on the hexapod: planner survives and holds behaviour class, but speed only recovers with a warm start

`sim/control/close_loop_ik.py` drives the hexapod in CoppeliaSim from vision alone (camera -> V-JEPA2 -> 12 candidate behaviours projected through forward model -> execute winner -> step). No inverse model, kinematics, or human command. 15 runs, 3 commitment settings, 5 repeats each, 20 steps vs a `speed_c5.8` demonstration.

| | commit 1 | commit 5 | commit 10 |
|---|---|---|---|
| survival | 5/5 | 5/5 | 5/5 |
| behaviour class | 5/5 | 5/5 | 5/5 |
| speed (within 15%) | 1/5 | 0/5 | 0/5 |
| median speed error | 21.8% | 40.2% | 35.4% |

Committing to a choice for longer makes speed error worse, not better — it locks in the (usually wrong) cold-start pick. Cold start (steps 0-4) is turned nearly all the time (80-100%) because a static scene looks like a slow turn; this differs from F81's open-loop accuracy since here the planner sees its own actions' consequences (covariate shift, not visible in open-loop eval).

Warm start (`--warm_start N` replays demonstration's own commands for N steps before handover), scored only on planned steps:

| | warm 0 | warm 5 | warm 10 |
|---|---|---|---|
| S.R. speed | 1/5 | 0/5 | 5/5 |
| median speed error | 19.9% | 20.3% | 3.5% |

10 warm steps: 1/5 -> 5/5, because the state inherited (already walking) lets roughly-right picks maintain speed, not because picks themselves change.

Two scorer bugs found and fixed (in the flattering direction): warm-start steps had been counted as planner's own work; a 10-step window had been compared to the demonstration's full 66-frame settled walk (fix moved warm-10 error 7.4%->3.5%). A third bug (punishing direction): scoring always used forward Froude number regardless of behaviour, so sideways runs read 23-34% error despite tracking lateral speed within 7% — fixed by scoring on the channel the demonstration is actually about.

Full-clip result (66 steps, 10 warm, 56 planned, 3 demonstrations x 3 repeats): S.R. speed 78% (7/9), behaviour class 100%, survival 100%, median error 7.0%.

Not established: yaw rate under closed loop (F81's turn-rate limitation untested here — `turn_s0.05` demo is gentle enough that forward motion dominates scoring); performance from a standstill (fails without warm start); only one robot/demonstration-per-behaviour/3 repeats; frames not yet visually verified.

Scripts: `sim/control/close_loop_ik.py`

---

### F83. The recorded-action-sequence planner does not port to the B1 because B1 actions are policy responses to state, not open-loop replayable sequences

The hexapod closed loop (F82) selects among recorded action sequences — viable because no state is read during replay (IK + CPG from a clock). Replayed open loop on `data/b1_traj` at native 50 Hz:

| trajectory | steps | survived |
|---|---|---|
| `fwd_vx0.2` | 300 | 289 (fell) |
| `fwd_vx0.3` | 300 | 154 (fell) |
| `fwd_vx0.4` | 300 | 72 (fell) |
| `fwd_vx0.5` | 300 | 58 (fell) |

0/8, survival falls monotonically with commanded speed. Cause: hexapod actions come from IK+CPG reading no state (exactly replayable); B1 actions come from a PPO policy reading base orientation, joint state and a phase clock at 50Hz (a *response*, not replayable without the state that produced it).

> Superseded in part by F91: the correct reading is narrower — a recorded B1 sequence stops being re-issuable after about three seconds (matches the closed-loop's actual horizon), not that it is never usable. Run for 3 seconds, forward behaviour survives a full episode and the loop clears chance on two of three demonstrations.

This is a property of the target robot, not the method: everything upstream of execution (F80's ranking, F81's selection, F82 slide 15's 3-clip forward-model adaptation) transfers to B1 latents fine. Two options for a working B1 closed loop, neither free: (1) command the PPO policy's own `(vx,vy,wz)` action space — concedes the project's joint-space-action claim; (2) replace candidates with a per-robot closed-loop primitive — adds a per-robot component, the cost recorded-sequences were meant to avoid.

---

### F84. The twelve behaviours reproduce on a held-out body; the weak sideways gait's direction reverses there

`data/allocentric/beh12_c08f09t09_flat`: same 12 conditions collected on `c08f09t09` (the body every Stage 1 result holds out), via `scripts/dataset/collect_beh12.py`.

| | c10f10t10, forward | c08f09t09, forward |
|---|---|---|
| `speed_c5.8`->`c8.8` | 0.135 0.166 0.210 0.200 | 0.129 0.158 0.200 0.215 |
| `turn_s0.05`->`s0.56`, yaw | | -0.007 -0.024 -0.037 -0.088 |

Both ladders rise monotonically; 64/66 condition pairs are >2x own spread apart (same two closest pairs as the original set: `speed_c8.15`/`c8.8` at 1.2x, `turn_s0.05`/`s0.15` at 1.4x).

Weak sideways gait reverses direction on this body:

| | c10f10t10 | c08f09t09 |
|---|---|---|
| `side_L_lvl0` | +0.019 | -0.045 |
| `side_R_lvl0` | -0.022 | +0.017 |

At `--strafe ±0.8` both bodies crab as commanded; at `±0.4` the shorter-legged body crabs the opposite way. Gait direction depends on morphology, not just the command parameter (extends F62's amplitude-sweep reversal to a morphology dependence). Does not invalidate planning tests (demonstration and candidates come from the same body, so self-consistent) but mislabels condition names on this body.

Collection note: the collector's walk-check gate is forward-only, so `FAILS forward`/`BACKWARDS` flags fire meaninglessly on sideways/turn clips — except here `BACKWARDS` on the two weak strafes was real, so flags should be checked against measurements. Separability (`--separability`), not value-matching (`--verify`) to `c10f10t10`, is the correct standard for single-body planning tests.

Data: `data/allocentric/beh12_c08f09t09_flat`
Scripts: `scripts/dataset/collect_beh12.py`

---

### F85. Refitting only a two-layer action-to-latent MLP partially restores closed-loop control on an unseen body

F82's closed loop run on `c08f09t09` (held-out body), same backbone/candidates/demonstrations/scene/warm start. V-JEPA2, ITM, and forward model stay frozen throughout — only the action projector (2-layer MLP) changes. 3 behaviours, 2 repeats, 49 planned steps each.

| | S.R. speed | S.R. behaviour | S.R. survival | median error |
|---|---|---|---|---|
| trained body | 7/9 | 9/9 | 9/9 | 7.0% |
| held-out, projector from trained body | 1/6 | 5/6 | 6/6 | 37.1% |
| held-out, projector refitted on it | 2/6 | 6/6 | 6/6 | 19.2% |

Refitting (a few minutes, no gradient through anything else) halves error and restores behaviour class outright; selection becomes decisive (e.g. holds `side_R_lvl1` for 30-34/49 steps vs ~30 switches un-refitted). Speed is not restored (2/6 vs 7/9) — roughly half the degradation is the projector, the rest is the forward model's ignorance of the new body, matching LAC-WM's 3-stage adaptation design.

> F92 later measures per-step (not whole-run) behaviour accuracy for these same runs at only 47-71%, despite 100% whole-run S.R. behaviour here — both numbers are correct, this one is the more flattering aggregate.

> Repeated 15x (5 per demonstration) after F95 showed CoppeliaSim physics is non-deterministic: survival and behaviour class hold 15/15. Speed passes 47% of runs, median error 19.0%. Per-channel: forward 20%±12 (2/5 within 15%); turn `s0.05` 130%±105 (0/5); turn `s0.29` 13%±8 (4/5); turn `s0.56` 79%±6 (0/5); sideways 39%±24 (0/5).

> Turning had never actually been scored on the yaw channel until this recheck — the criterion picked whichever channel was largest in the demonstration, and forward speed exceeds yaw in every turn condition on this body. Graded correctly on yaw, run-level speed pass rate falls from 47% to 13%.

> Re-run again after F96 fixed two broken candidate-library conditions: survival/behaviour still 15/15, speed pass rate unchanged at 47%, median error 19.0%->17.7%. Sideways result is inconsistent across repeats (2%-84% range), not a single reliable number.

Failure mode is walking the wrong way/level, not falling: median body speed is 0.097 zero-shot / 0.114 refitted vs demonstrations' 0.13-0.14; survival 6/6 both arms. Zero-shot pick accuracy stays flat (~40%) over time (not tracking); refitted arm's pick accuracy decays 48%->21% but score improves anyway, because it settles into the wrong turn *level* of the right behaviour (F81's yaw limitation recurring).

Not established: only one held-out body, one family, 3 demonstrations, 2(-5) repeats; forward-model adaptation + projector refit combo not built (`finetune_ftm.py` adapts/scores but doesn't save a checkpoint to plan with).

---

### F86. Across embodiments the planner defaults to one candidate instead of selecting, and the success criteria pass anyway

The B1 cannot replay recorded actions (F83), so selection was tested alone: `sim/control/close_loop_kinematic.py` poses the body from per-step deltas of the chosen behaviour with no physics (survival is not evidence here). Backbone `beh12_hexonly`: ITM and forward model have never seen a quadruped; only the projector's B1 head is fitted (on `beh12_b1_flat`).

| demonstration | speed error | class verdict | most-chosen | family accuracy |
|---|---|---|---|---|
| `speed_vx0.30` | 8.6% | pass | `speed_vx0.50` 17/49 | 53% |
| `turn_wz0.40` | 3.5% | pass | `speed_vx0.50` 21/49 | 8% |
| `side_R_lvl1` | 83.1% | fail | `speed_vx0.50` 18/49 | 10% |

`speed_vx0.50` is top choice for every demonstration including the sideways one — the planner defaults to the fastest forward walk rather than selecting. The two "passes" are arithmetic coincidence (mixing `speed_vx0.50` and `turn_wz0.00` averages onto forward-walk demonstrations' true speed). Success criteria are insensitive whenever most candidates (8/12 on the B1 set) move the same way; the only demonstration whose answer isn't "walk forward" fails outright.

Since the projector was already fitted on this robot (unlike F85's held-out-body case where refitting the projector recovered half the degradation), selection still collapses — meaning across families the binding constraint is the forward model's ignorance of the robot, not the action-to-latent map. Matches slide 15: a frozen forward model scores 0.57-0.71x on the B1 (worse than predicting no motion), needing 3 adaptation clips to clear break-even.

Not built: adapting the forward model on B1 clips + refitting the projector against the adapted ITM (LAC-WM stages 1 and 3) — `finetune_ftm.py` adapts/scores but doesn't save a checkpoint to plan with.

Scripts: `sim/control/close_loop_kinematic.py`, `finetune_ftm.py`

---

### F87. The action projector is 2.8x harder to fit on the B1 than the hexapod, for a reason later corrected by F88

Building LAC-WM stage 1 (`wm/adapt.py`, adapts ITM+forward model and saves a checkpoint, unlike `finetune_ftm.py` which discarded it) made stage 2 measurable on the B1.

Stage 1 (forward model) works: rolling on held-out B1 clips vs holding the frame still, horizon 1 goes from 0.68 (frozen) to 1.16 (after 9 clips) — worse-than-nothing to better-than-nothing, reproducing slide 15's curve.

Stage 2 (action projector) does not: rollout gap vs mean-z baseline (below 1.0 is better than knowing nothing) is 1.301 fitted on the same 9 clips, 0.841 fitted on all 48 clips — still barely useful. Not a data-quantity issue: with the same script/clip-count/unadapted checkpoint, hexapod (48 clips) scores 0.230 vs B1 (48 clips) 0.640 — 2.8x harder on the same latent space before any adaptation.

> Corrected by F88: the explanation given here (B1's action is a policy response to state, carrying nothing usable) is wrong — a classifier reads behaviour off B1 actions alone at 85% family accuracy vs 28% chance. The real issue: `z_ITM` encodes a transition depending on gait phase/state as well as behaviour, so `a -> z` is one-to-many even where `a -> behaviour` is not; the projector isn't starved of signal, it's asked to resolve an underdetermined target.

Adapting the forward model does not rescue the kinematic loop — it swaps one default candidate for another rather than enabling selection: frozen defaults to `speed_vx0.50` (family accuracy 53%/8%/10% for speed/turn/side), adapted defaults to `side_R_lvl1` (18%/10%/39%).

Deployment-claim consequence: "record a few clips and control it" holds where the robot's own controller is open loop; where it's a learned feedback policy, the projector needs either far more data or LAC-WM stage 3 (joint projector+FDM fine-tuning) — built later as `wm/adapt3.py`, first result in F88.

---

### F88. The forward model discards the B1's action under MSE adaptation; adding an InfoNCE term fixes it, and closed-loop numbers are only valid within one simulator session

Corrects F87's explanation for B1 stage-2 failure. Three results, read in order:

**1. The action carries the behaviour.** A classifier trained on actions alone (held out by clip): hexapod joint targets reach 100% (5-frame) / 68% (1-frame); B1 policy actions reach 80% (5-frame) / 61% (1-frame), family accuracy 85% vs 28% chance. F87's claim that response-shaped actions carry nothing usable is false.

**2. Stage 3 (`wm/adapt3.py`, 15k steps, lr 1e-4, 24 clips) with plain MSE still fails.** Training loss falls 2.05->0.35 and held-out prediction improves (0.830->0.805 /hold), but the mean-z-normalized score never moves (1.005->0.993) and family selection stays at chance (~19-25%). The forward model learns B1 dynamics while discarding the action channel entirely.

**3. Adding an InfoNCE term (true action must match `e_t+1` better than actions from other behaviours, same-time-index negatives) fixes it, all else held constant:**

| | /mean-z | family |
|---|---|---|
| stage 2 | 1.005 | 25% |
| stage 3, MSE | 0.993 | 19% |
| stage 3, + InfoNCE | 0.62 | 50% |
| chance | -- | 28% |

Mechanism: the action-dependent part of `e_t+1` is a small fraction of its variance; MSE can bank the large unconditional "what does a quadruped look like" win without ever using the action. This shortcut doesn't exist during pretraining (why the hexapod's forward model learned to use z). MSE rewards prediction; planning needs discrimination — they coincide in pretraining, diverge under adaptation.

Not memorization: hexapod-only checkpoint tested on unseen hexapod body still discriminates at 62% family accuracy (vs 84% on trained body, vs B1's 35%) — and the B1 arm's projector had actually seen all its test clips while the unseen-hexapod arm's projector had not, so the disadvantaged arm still wins by 27 points.

LAC-WM's three stages are MSE throughout; the contrastive term is this project's addition, recovering about half the distance from B1's 35% to the within-family 62%. Not solved, no longer a wall.

**Reproducibility caveat, applies to every closed-loop number in this project:** the same loop/checkpoint/demonstration run twice inside one CoppeliaSim session is pixel-identical; run across two different sessions it is not (6% agreement on chosen candidates; `frames.sum()` differed 1.4473e9 vs 1.4649e9). Loop results are only comparable within one simulator session. A B1 number from an unreproducible session was pulled before being used; re-measured properly (one session, 3 demonstrations, run twice, 100% repeat agreement): family accuracy turning 59%, sideways 24%, forward 0% (chance 28%) — all three top choices were turns; amplitude tracks the demonstration but behaviour selection does not.

The 57% held-out-clip ranking did not convert to closed-loop success: the loop adds compounding error and self-driven states that clip-level scoring lacks. The contrastive fix addressed what the forward model attends to, not what happens when small errors accumulate under the planner's own control.

Scripts: `wm/adapt3.py`

---

### F89. Raw evidence tables from cut Stage-1 diagnostic slides, preserved outside the presentation

The week-13 deck was cut from 25 to 13 slides; conclusions are written up elsewhere in FINDINGS, but some raw table rows existed only in the deck. Reproduced here so the record doesn't depend on a presentation file. Source: `report/update_slide_full.md`.

Cut slide 5 (frame source vs latent source, cross-checked): frame from c10t10f10 + latent from c10f06t06 matches c10f10t10 at 4.79, c10f06t06 at 21.64 — result follows the frame, not the latent source (and symmetrically the reverse).

Cut slide 6 (variance in latent, without/with cross-body loss): gait-phase variance 81.9%->92.6%; which-body variance 12.4%->3.4%; foot-contact decodability 0.729->0.732; which-body decodability 0.764->0.694.

Cut slide 8 (femur/tibia coupling limit): decoder generalizes within trained coupling range (`c08f09t09`, femur 0.9/tibia 0.9: 3.44 deg, R²=+0.81) but fails once femur/tibia ratios diverge from training (`c10f10t08`, femur 1.0/tibia 0.8: 13.35 deg, R²=-0.34). True/decoder/probe/best-mixture femur,tibia values: truth (1.00,0.80), trained decoder (0.681,0.681), linear probe (0.843,0.843), best mixture (0.600,0.600) — everything ties femur to tibia because training data does. By joint: TC (fore-aft swing) R² +0.46 to +0.83 (still works); CF (lift) -0.53 to +0.05; FT (knee) -0.45 to -3.99 (where the failure sits). Held-out bodies c10f10t08/c10f09t07/c10f08t06 all show negative R² (-0.34/-0.14/-0.33).

Cut slide 9 (diagnosis test): adding decoupled bodies to training recovers the femur-tibia gap (probe: 0.182 vs true 0.200), vs 4 tied bodies' probe values (femur 0.819, tibia 0.819 — collapsed) vs 6-body probe (femur 0.954, tibia 0.772 — recovered).

Cut slide 10 (pre-training transfer predictor): "mixture gap" and "probe error" measured before training predict transfer outcome — 4 spanning bodies held out c08f09t09: mixture gap 0.000, probe error 0.021, 3.44 deg, R²=+0.81, beats copy-nearest baseline (3.47). 6 decoupled bodies held out c10f10t08: gap 0.063, error 0.034, 3.27 deg, R²=+0.89, beats every baseline. 4 bodies all tied at 0.83 held out c10f10t08: gap 0.141, error 0.082, 12.67 deg, R²=-0.78, loses to the body's own mean.

Cut slide 11 (what `e_{t+1}` needs to be): real next frame gives 3.71 deg (3.37 with cross term) control error; substituting a copy of `e_t` (no transition) costs 1.28-1.34x; wrong transition (`e_{t-1}`) costs 1.65-1.67x; random-time frame costs 3.44-3.54x; zeroed latent costs 2.88-3.48x. The transition itself is worth about a third of the total.

Cut slide 12 (forward model vs naive baselines at various horizons): forward model beats holding the frame still by 1.46-1.72x across horizons 1/3/5/10, and both vastly beat constant-velocity extrapolation (e.g. horizon 5: FM 2.12, hold-still 3.57, const-vel 66.0).

---

### F90. The forward model's rollout genuinely carries the planner's candidate selection, and one rollout step captures most of the benefit

Tested whether the planner is actually using the forward-model rollout or just pattern-matching to training-similar actions (planner spends 60 FDM calls per control step vs a policy's 1). Three scoring rules on identical candidates/held-out clips (`scripts/diagnostics/does_rollout_matter.py`): `rollout` (roll FDM h steps, score predicted frame vs goal), `direct` (score `proj(a)` against `ITM(e_t,e_goal)`, no forward model), `blind` (score against mean latent, goal unused).

B1 family accuracy (chance 28%):

| horizon | rollout | direct | blind |
|---|---|---|---|
| 1 | 62% | 38% | 33% |
| 3 | 65% | 36% | 32% |
| 5 | 65% | 38% | 32% |
| 10 | 67% | 37% | 34% |

Deleting the forward model costs 24 points and brings `direct` within 5 points of `blind` (which ignores the goal) — the world model predicts rather than pattern-matches. Going from 0 to 1 rollout step adds 24 points; going from 1 to 10 steps adds only 5 more — the planner can run at 12 FDM calls/control-step instead of 60 (5x cut) for 3 points of accuracy.

> On the held-out hexapod body at horizon 1, the order reverses: `rollout` 63%, `direct` 89%, `blind` 35% — deleting the forward model helps by 26 points. At horizon 5 order returns to normal (`rollout` 80% vs `direct` 59%). The rollout-matters claim holds for the B1 at every horizon and for the hexapod only beyond horizon 1; the inverse model's advantage at one step on one robot but not the other is unexplained.

These rules score 62-67% on recorded clips while the same checkpoint in closed loop follows only 1 of 3 demonstrations — the gap is compounding error under the planner's own control (self-driven states vs always-real clip frames offline), not the scoring rule. Argues for moving search into training (a policy distilled in imagination trains on states it actually reaches).

Scripts: `scripts/diagnostics/does_rollout_matter.py`

---

### F91. A real physics loop for the B1 does run, and F83's "impossible" was actually an episode-length limit (~3s, not 6s)

F83's 0/8 replay-failure result reflected episode length, not impossibility: survival was 289/154/72/58 steps at 50 Hz (6 sec at the slowest command), while the closed loop only needs 3 sec. An earlier check stepped clips at 20ms/decision (the policy's rate) instead of 50ms (the planner's actual decision rate), inflating apparent survival — correcting to 50ms, forward survives 66/66 steps but turning only 28/66 and sideways 27/66.

Loop built as `sim/control/close_loop_b1_physics.py`: MuJoCo holds physics, CoppeliaSim poses the body from MuJoCo's state and renders the camera image — split is required so the B1 isn't distinguishable from the insect by render style alone.

Bug found: starting the robot standing (rather than mid-stride, matching how clips were recorded/cropped) caused a body-height jump (0.435->0.665 in 6 steps) not seen in demonstrations. Fixed by seeding MuJoCo from the demonstration's first frame (joint angles/velocities, height, orientation).

| | seeded | standing start |
|---|---|---|
| survival, all three | 65/65 | 65/65, fell at 29, fell at 37 |
| peak body height | 0.57-0.60 | 0.67-0.70 |

> Second, larger defect found via F92's diagnostics: the camera was following the robot in both earlier versions, deleting the fixed-background motion cue the model trained on (one-step FDM error was 2.9x the recorded-clip error). `close_loop_kinematic.py`/`close_loop_ik.py` place the camera once and are unaffected. Fixing the camera:

| | camera fixed | camera following (discarded) |
|---|---|---|
| turning | family 100%, exact 95% | family 51% |
| forward | family 71% | family 58% |
| sideways | family 35% | family 38% |

Turning is now the strongest quadruped result: planner picks the exact condition `turn_wz0.40` on 52/55 planned steps (rotation is read from background motion, which a following camera destroys).

Overall: survival 3/3, behaviour class 2/3, speed 0/3 (errors 18.7%, 25.3%, 91.7%). Corrected reading of F83: a recorded B1 action sequence holds for about 3 seconds, not that it never holds.

Scripts: `sim/control/close_loop_b1_physics.py`

---

### F92. Closed loops are in the right behaviour on roughly half their steps; whole-run "behaviour 100%" never measured this, and switching costs travel

Four candidate explanations for right-behaviour/wrong-speed were tested from existing data and three refuted: candidate library too coarse (refuted — right-speed behaviour was in the list on 9/9 runs); score can't see speed (refuted — within-family score spread is 67% of between-family spread); score orders speed wrong (refuted where it matters — rank correlation +0.88 sideways, +0.25 forward, inside correct family); switching often costs speed (refuted — correlation of switch rate with achieved/demanded speed +0.14).

What remains: per-step behaviour-family accuracy is much lower than whole-run "S.R. behaviour":

| | per-step in-family |
|---|---|
| hexapod, held-out body, six runs | 47/55/61/61/67/71% |
| B1, physics, three runs | 35/71/100% (after camera fix; 38/51/58% before) |
| what runs report as "behaviour" | 100%, and 2/3 |

Both numbers are correct but answer different questions: whole-run asks whether the right channel dominates overall; per-step asks how often the decision itself was right (wrong picks scatter, right family stays plurality). When in the right family, amplitude is picked correctly too (mean speed ratio 0.90-1.35 in-family vs 0.31-1.08 when out-of-family steps included) — the earlier "planner picks slow candidates" reading was an artefact of averaging in wrong-direction candidates.

Replay fidelity (no planner, exact seeding): B1 alone reaches 0.84/0.76/0.99 of recorded speed (policy-response actions drift under open-loop replay, per F83). Hexapod (clock/IK actions, should replay exactly): forward 1.06, turning 0.96, sideways 0.82 — prediction holds for forward/turning, inverts for sideways (unresolved; hexapod recording was from a different simulator session, and F91 showed sessions aren't neutral, so 0.82 may be session variance rather than replay loss).

Obvious B1 fix (replay achieved joint positions instead of targets) fails: robot stands still, travels almost nothing (0.01/0.06/0.28 of recorded speed) — motors need a target/position gap to produce torque; commanding achieved position removes it. The table already used (`DEFAULT_IL + ACTION_SCALE x action`) is the correct target sequence; the B1's issue is these targets reference states replay can't recreate, not that it's a policy vs. a table.

Residual term (switching cost) measured directly (`scripts/diagnostics/what_stitching_costs.py`) by comparing stitched executed sequence vs single most-common clip, same seeding:

| B1 demo | switches | stitched | single clip | ratio | residual predicted |
|---|---|---|---|---|---|
| forward | 44 | 0.122 | 0.102 | 1.19 | 1.08 |
| turning | 36 | 0.059 | 0.118 | 0.50 | 0.59 |
| sideways | 38 | -0.020 | -0.109 | 0.18 | 0.20 |

Direct measurement matches the residual estimate in all three: switching between clips costs half of turning speed, four-fifths of lateral speed, and nothing for forward — mechanism is directional cancellation (every candidate goes forward to some degree so forward motion survives stitching; turning/strafing need direction held, and interleaving clips at different rates partially cancels).

Scripts: `scripts/diagnostics/why_speed_misses.py`, `does_score_see_speed.py`, `what_stitching_costs.py`

---

### F93. Committing to a behaviour for 3 steps is the first setting to hit the speed criterion on the B1; no reversal on the hexapod after repeats

Following F92's stitching-cost finding, `--commit` (default 1 = re-decide every step, never varied before) was swept on the B1 physics loop, 3 demonstrations:

| commit | speed within 15% | behaviour | survival | speed errors |
|---|---|---|---|---|
| 1 | 0/3 | 2/3 | 3/3 | 18.7%, 25.3%, 91.7% |
| 3 | 2/3 | 2/3 | 3/3 | 13.1%, 6.4%, 85.5% |
| 5 | 1/3 | 2/3 | 3/3 | 39.0%, 9.8%, 94.8% |

commit=3 is the first setting anywhere in the project to clear the speed criterion on a quadruped (2/3). Trade-off: per-step accuracy on turning drops as commit increases (family/exact: 100%/95% at commit 1, 95%/89% at commit 3, 91%/76% at commit 5) — re-deciding tracks behaviour better but executes worse (every switch interrupts stride); commit 5 overshoots the optimum, turning's error triples to 39%. Sideways doesn't improve with commit (91.7%->85.5%) because its problem is being out of the right family 65% of the time, not stitching — committing just holds the wrong choice longer.

Confirms F92's stitching-cost prediction directly: reducing switching returns speed.

Hexapod reversal claim withdrawn after repeats: originally reported as commit 1 forward error 0.4% vs commit 3 at 39.7% (one run each), read as a reversal. Re-run 5x per setting after F95 established CoppeliaSim's non-repeatability:

| | commit 1 | commit 3 |
|---|---|---|
| forward | 23% ± 14, range 1-37 | 15% ± 13, range 4-40 |
| turning | 11% ± 5, range 4-19 | 13% ± 16, range 2-45 |

Distributions overlap almost entirely (commit 3 if anything slightly better on forward) — no hexapod reversal; original comparison took bottom of one distribution vs top of the other. B1 half stands (MuJoCo repeats bit-for-bit per F95): commit helps the quadruped, is neutral on the hexapod.

---

### F94. Offline clip-ranking accuracy does not predict closed-loop behaviour accuracy, and the two robots fail sideways for opposite reasons

Same checkpoint/candidates, behaviour-family accuracy on recorded clips vs inside the physics loop:

| behaviour | on recorded clips | inside the physics loop |
|---|---|---|
| sideways | 97-100% | 35% |
| turning | 55-63% | 100% (exact condition 95%) |
| forward | 32-36% | 71% |

Sideways ranks best offline and worst in the loop — conclusions from clip-level scoring (including F90's 62% headline) describe a different problem than closed-loop behaviour.

Step sequences show the mechanism differs by robot: B1's sideways demo oscillates between side/turn choices rather than locking into a wrong answer; hexapod holds `side` on ~96% of steps yet still misses lateral speed by 73.8%.

| | choice | execution |
|---|---|---|
| B1, sideways | cannot hold it (35% in family) | — |
| hexapod, sideways | holds it (~96%) | cannot achieve it (73.8% speed error) |

B1's failure traces to F83: sideways candidates don't make it actually strafe (policy-response actions replay at 0.76-0.99 of own motion), so the resulting frame isn't sideways and the ranking (accurate for real sideways frames) no longer applies — a state-reachability problem, not a ranking error. Hexapod's commands replay at 1.06/0.96 so its frames stay sideways and choices stay correct, but hexapod's sideways execution failure (replay fidelity 0.82, loop reaches 0.26 of demonstrated speed) remains unexplained.

Eight hypotheses tested and refuted in one day (kept as reference list):

| hypothesis | refuted by |
|---|---|
| candidate library too coarse | right-speed behaviour in list on 9/9 runs |
| score can't see speed | within-family spread 67% of between-family |
| score orders amplitudes wrongly | rank correlation +0.88 inside correct family |
| switching costs speed | correlation with switch rate +0.14 |
| gait phase drift explains prediction error | phase-broken control scored 1.33x, below in-phase 1.43x |
| replaying achieved joint positions fixes B1 | robot stands still — no tracking error, no torque |
| sideways was already bad at ranking | ranks 97-100%, best of the four |
| loop locks into first choice | B1's sideways run oscillates, doesn't lock |

What survived: F83's replayability (decides frames the loop sees, and everything downstream) and the camera defect (F91).

---

### F95. CoppeliaSim physics is not reproducible run-to-run; MuJoCo is — determines which loop numbers need repeats

Repeating one closed-loop hexapod configuration 5x, changing nothing:

| | lateral speed error, five runs |
|---|---|
| commit 1 | 50% ± 12, range 37-71% |
| commit 5 | 58% ± 23, range 23-92% |
| commit 10 | 68% ± 35, range 27-111% |
| commit 20 | 39% ± 30, range 10-87% |

No commit setting is significantly better than commit 1 (under one SE of difference at n=5); longer commitment mainly widens spread (fewer decisions per episode). A single-run reading that commit 20 fixes sideways (24.1%) was the bottom of a range reaching 87%.

Cause: CoppeliaSim reloads the scene per run and its solver/contact state doesn't reproduce; MuJoCo (used for B1, seeded explicitly from the demonstration's first frame) reproduces choices/frames/body track bit-for-bit across runs.

Status of previously reported numbers:
- All B1 loop results (F91, F93's B1 half, F94's B1 half): safe — MuJoCo repeats exactly, one run is the answer.
- F85's hexapod loop (survival 100%, behaviour 100%, 19.2% median error): re-run 15x — survival/behaviour hold 15/15, speed passes 47% with median 19.0%; headline survives, per-behaviour spread is wide and now reported with it.
- F93's hexapod half (commit 1 at 0.4% forward vs commit 3 at 39.7%): one run each, both inside the measured spread — comparison does not stand as originally reported.
- F92's hexapod replay fidelity (1.06/0.96/0.82): one clip each, recording from a different session — the 0.82 outlier may be session variance.

Rule going forward: a CoppeliaSim-physics number needs repeats before it can support a comparison; a MuJoCo one does not. Not known during most of this project's earlier loop-result collection, where hexapod numbers were read as single-run answers.

---

### F96. Two of the held-out body's four lateral conditions were mislabeled (traveling the wrong way), but fixing this did not fix the loop's sideways failure

Offline ranking showed an asymmetry (`side_L` 96%, `side_R` 77% at horizon 5) that turned out not to be a camera or model issue but mislabeled data. Median lateral speed per condition:

| | `side_L_lvl0` | `side_L_lvl1` | `side_R_lvl0` | `side_R_lvl1` |
|---|---|---|---|---|
| `beh12_c10f10t10_flat` | +0.071 | +0.185 | -0.118 | -0.186 |
| `beh12_c08f09t09_flat` (held-out) | -0.045 | +0.148 | +0.017 | -0.131 |

On the held-out body, both `lvl0` conditions strafe opposite to their names (std 0.001 across clips — consistent, not noise). Cause: per `collect_beh12.py`'s own docstring, the lateral recipe (a twist on the middle joint scaled about each body's hip) is not portable across bodies; at `lvl1` amplitude survives the port, at `lvl0` it's small enough for the sign to flip. All three channels on `side_R_lvl0` (forward -0.009, lateral +0.017, yaw +0.021) show it's essentially motionless — the recipe under-drives the shorter legs, and the residual sign is noise, not a reverse strafe.

The existing `--separability` check (are conditions farther apart than their own spread) passed (2/66 pairs below 2x) because it doesn't check whether `side_L` actually travels left, `side_R` right, or `lvl1` exceeds `lvl0` — a semantic gap, not a statistical one.

Fix applied: `--lvl0_strafe 0.7` (base body value 0.4) gives +0.076/-0.069 (target: half of lvl1), replacing the eight affected clips; dataset now passes both checks.

This invalidates the earlier camera-viewpoint explanation for the ranking asymmetry, and invalidates `corr(error, left-picks)=+0.72` (it counted a right-traveling clip as a left pick). It also means every "in-family" figure for lateral behaviours on this body in F92/F94/F95 groups opposite-direction conditions under one label — forward/turning numbers are unaffected.

**Fixing it changed nothing in the loop:**

| | sideways speed error, five repeats |
|---|---|
| before fix | 34% ± 22, range 2-64, within 15% on 1/5 |
| after fix | 35% ± 23, range 15-67, within 15% on 0/5 |

The sideways closed-loop failure is not explained by the mislabeling and remains unexplained after every hypothesis tried (library, score, amplitude ordering, switch frequency, gait phase, lock-in, camera angle, condition labels).

Scripts: `collect_beh12.py` (`--separability`, `--lvl0_strafe`)

---

### F97. A B1 driven toward a hexapod's goal image walks forward correctly; the turning result is later withdrawn by F100

First cross-embodiment control demonstration: goal image is a hexapod clip, robot driven is the B1 (candidates stay B1 clips since only those are executable). Physics, MuJoCo, `--commit 3`, behaviour-family accuracy (chance 28%):

| behaviour asked for | goal is B1 clip | goal is hexapod clip |
|---|---|---|
| forward | 42% | 67% |
| sideways, lvl1 | 31% | 2% |

Forward crosses embodiments, even better than the B1's own video as goal. Turning showed a dose-response with true yaw magnitude (`turn_s0.05` yaw -0.007: 18%; `turn_s0.29` yaw -0.037: 38%; `turn_s0.56` yaw -0.088: 47%) — later found to be a warm-start artifact, not a real signal.

> Withdrawn by F100: all runs here were warm-started with the same turning B1 clip; re-running with a forward warm start moves both turn rows to 27%/27% (below chance, unordered). Forward and sideways results survive the change; turning does not.

`hexapod_ep1001` (`turn_s0.05`) is the same clip used as "the turn demonstration" throughout the project (F85, commitment sweeps) despite barely turning; its celebrated 7%±2 error is a forward-speed measurement, not a yaw measurement (see F98).

The clip-level retrieval proxy (`z_crosses_bodies.py`, nearest-B1-clip-shares-behaviour) predicted the opposite of the loop result: turning 100% retrieval vs 18-47% in loop; forward 19% retrieval vs 67% in loop. The proxy pools all four turn rates (its 100% is dominated by strong turns) and averages `z` over a clip via nearest-neighbor, which is not what the planner computes (`FDM(e_t, proj(a))` against the goal, conditioned on the B1's actual current state) — the cheap proxy would have pointed at the wrong behaviour to test.

Three script defects found and fixed: two differently-goaled runs overwrote the same output filename (one run lost); goal embedding was centered with the driven robot's offset; auto-printed "reachable"/"not reachable" verdicts were threshold comparisons of overlapping means (numbers were right, printed verdicts were not).

---

### F98. Turn rate was never actually scored in any closed-loop run — the success criterion graded turning demos on forward speed instead

`S.R. speed` grades a run on whichever channel is largest in the demonstration. Forward speed exceeds yaw in all four turn conditions on both bodies (0.136 vs 0.088 even at `turn_s0.56`), so every turning run was silently graded on forward speed and yaw was never measured in any loop.

Graded correctly on yaw, held-out hexapod, five repeats each:

| turn rate | demonstrated yaw | yaw error | within 15% |
|---|---|---|---|
| `s0.05` | -0.007 | 130% ± 105 | 0/5 |
| `s0.29` | -0.037 | 13% ± 8 | 4/5 |
| `s0.56` | -0.088 | 79% ± 6 | 0/5 |

Only the middle turn rate is actually tracked. `turn_s0.05` — the clip underlying nearly every turning claim in the project (F85's loop, commitment sweeps, F97's first cross-embodiment attempt) — has a celebrated 7%±2 "error" that is a forward-speed measurement on a clip that barely turns.

| | before (magnitude-based grading) | graded on named channel |
|---|---|---|
| S.R. speed, held-out hexapod, 15 runs | 47% | 13% |
| median error | 17.7% | 36.2% |

Survival and behaviour class are unaffected (15/15 each, channel-independent). What falls is the claim that the loop tracks commanded turn rate.

Consistent with F81 (open-loop planner resolves speed 9/9, sideways 6/6, turn 2/9 — "knows it's turning, can't say how hard") — same failure now confirmed in physics, previously hidden by the criterion looking at the wrong channel.

Fix: grade `turn` conditions on yaw, `side` on lateral, `speed` on forward (semantic, not magnitude-based) — same class of gap as F96's separability check (checks conditions differ, not that they differ in the way their name claims).

---

### F99. Positioning against LAC-WM: our action space is joint-space and incommensurable across robots by design, theirs is commensurable body-pose space

Per F60, the divergence from LAC-WM is the coordinate the heads decode into, not their number/size. LAC-WM's motion decoder targets wrist poses/fingertip positions/camera poses — physically commensurable across their embodiments (e.g. a fingertip xyz means the same thing for a hand or a gripper). This project decodes joint angles (18-D vs 12-D, no dimension correspondence between robots) because the target setting is a robot about which nothing is known (no kinematic tree, URDF, or action labels — only video); a shared body-motion term (Froude/yaw) is a separate supervisory head, not the action space itself.

Measured (control arm decoding joint commands):

| | within-robot joint error | cross-robot transfer |
|---|---|---|
| joint target, no body term | 0.3517 | -28.9 / -43.1 |
| joint target + body term | 0.2183 | +0.610 / +0.573 |

Joint-space targets work within a robot alone; they do not cross robots without the body term, and the body term also improves within-robot decoding by 38%. Conclusion: a joint-space action target transfers across incomparable embodiments only when a shared body-motion term is present.

Two weaker framings considered and rejected: (1) "we supervise only 1-2 dimensionless numbers vs their 6-DOF pose" is a data limitation (oscillator doesn't populate vertical/roll/pitch — vertical fails the variation gate at 0.13 by construction since `--scale` is held fixed), not a principled design choice; (2) "there's no command for a given Froude number" is false — B1's policy is velocity-commanded in m/s; only the hexapod's oscillator needed empirical stride-to-Froude calibration (F62).

The framing that holds: body velocity is commandable on both robots, which is exactly why it isn't chosen as the action space — commensurable units (m/s) don't mean equivalent behavior (0.3 m/s is near-limit for a 0.176m insect, strolling for a 0.561m quadruped; hence Froude/w_hat matching in `data/allocentric/beh12_*`, not raw units). If the motion decoder emitted body velocity directly, both robots would trivially share a 3-DOF action space and there'd be nothing for a latent action to bridge — decoding joint angles is deliberately choosing the disjoint space worth crossing.

Reproducibility vs LAC-WM's evaluation sections: have equivalents for 5.1 (UMAP, `scripts/figures/plot_z_umap.py`), 6.1 (imagined rollout, `scripts/diagnostics/latent_rollout.py`, F44), 6.2 (closed loop, F72/F82 etc). Worth building: 5.2, cross-robot action-latent transfer scored via `scripts/diagnostics/cross_latent_rollout.py` (embedding-space, with bracketing baselines since clips aren't phase-synchronized, F39). Cannot do 6.3 (scaling with number of embodiments) — only two embodiments; state as a limitation.

Correction: LAC-WM's FDM predicts embeddings (not video) exactly as ours does — `x_hat_{t+1}=FDM(x_t,z_t)`, `L_recon` is MSE on embeddings, frozen V-JEPA2 tokenizer, 64-D action embedding — all identical to this project's setup. Their image metrics come from a separate, custom V-JEPA2 RGB decoder (which this project lacks, needed only for visualization, not measurement).

Proposed locomotion-analogue success metrics (their manipulation metrics don't apply): S.R. speed (Froude error <15%), S.R. behaviour (correct class by dominant channel), S.R. survival (didn't fall). Two departures from their protocol: report graded error alongside binary rate (12 conditions give a coarse binary rate); survival must be scored (a legged robot that fails falls over, unlike a manipulator that just misses a grasp).

---

### F100. The B1's turning in cross-embodiment runs was determined by the warm-start clip, not the goal — F97's turn dose-response is withdrawn

Found by watching video: the B1's head drifted at the start of every run and, on a `turn_s0.29` goal, arced left while the goal-pane insect turned right — invisible to family-accuracy metrics, which only count which label was picked, not direction.

Cause: every F97 run used `--demo b1_ep1301` (`turn_wz0.40`) as warm start, which supplies both starting state and the first 10 actions, so the loop always opens by executing a turn regardless of the goal.

Control (only warm start changed: `b1_ep1301`/turning vs `b1_ep2`/`speed_vx0.30` forward; same checkpoint/goals/`--commit 3`):

| hexapod goal | turn warm: yaw / family | forward warm: yaw / family |
|---|---|---|
| speed (`ep1`) | -0.031 / 67% | -0.023 / 84% |
| turn_s0.29 (`ep1200`) | +0.004 / 38% | -0.016 / 27% |
| turn_s0.56 (`ep1300`) | -0.025 / 47% | +0.005 / 27% |
| side_R (`ep2301`) | +0.002 / 2% | -0.022 / 0% |

Achieved yaw tracks the warm start, not the goal — shifting warm start by 0.06 shifts every planned yaw; a 30-fold range of commanded goal yaw barely moves it. `ep1300`'s "correct" sign under turn warm-start was coincidental (reverses when warm start does). Family accuracy for both turn goals crosses the 33% chance line (47%/38% -> 27%/27%) when warm start changes — F97's turn dose-response was measuring the warm-start choice, not the goal.

| | turn warm | forward warm | verdict |
|---|---|---|---|
| forward | 67% | 84% | above chance both ways — stands |
| turning | 47%/38% | 27%/27% | crosses chance line — withdrawn |
| sideways | 2% | 0% | below chance both ways — fails |

> Corrected by F117 and reframed by F118 (2026-08-30) — read all three together. F117 found the forward/turning ordering inverts on same-robot goals; F118 showed the loop isn't conditioning on its target at all (scores 56-70% against the demonstration but 18-23%, below chance, against the shown goal), so "which behaviour transfers" is not measurable from these runs — the numbers actually rank identifiability of each behaviour from its own dynamics. Do not cite "forward crosses, turning does not" going forward.

Removing warm start entirely (`--warm_start 0`, seeds state from demonstration's first frame, standing start) does not rescue turning and doesn't hurt locomotion: all four goals survive 65/65 steps at 0.054-0.119 m/s.

| | warm=turn clip | warm=forward clip | no warm start | chance |
|---|---|---|---|---|
| forward | 67% | 84% | 71% | 33% |
| turning | 47%/38% | 27%/27% | 37%/34% | 33% |
| sideways | 2% | 0% | 0% | 17% |

Forward clears chance in all three settings; turning straddles chance in all three. Across all 13 cross-embodiment runs, goal yaw and achieved yaw correlate at -0.33 with 46% sign agreement (no relationship). The planner controls forward-vs-not-forward only — turn goals do slow the robot (0.054/0.084 vs 0.117 m/s for forward goal, read as "not straight ahead"), but nothing selects rotation direction.

General lessons: family accuracy cannot see direction/sign (a wrong-direction turn scores identically to a correct one) — any signed condition needs sign agreement reported separately from magnitude. A warm start is an intervention (10 steps at 50ms is half of a 3-second episode and the loop never escapes it) — future comparisons must hold warm start fixed-and-neutral or vary it as an explicit control.

Third defect in this project found by watching video rather than reading tables (after the standing-start jump and the following-camera bug) — tables were internally consistent in all three cases.

---

### F101. The warm start hides an entry transient, not an inability to turn

`--demo` gives same-robot runs the goal's own actions as warm-start, so the scorer excludes those steps but not their consequence (body already in the right state at step 11). Removing warm start (`--warm_start 0`, held-out body, 5 repeats/goal):

| goal | family picks | speed error |
|---|---|---|
| `ep1` forward | 56% -> 75% | 23.0% -> 12.9% |
| `ep1001` turn_s0.05 | 46% -> 23% | 86.8% -> 154.4% |
| `ep2301` side_R_lvl1 | 91% -> 69% | 26.2% -> 58.1% |
| all 15 | | 36.2% -> 58.8% (15/15 -> 14/15) |

Forward improves without the hint; everything else degrades: the real line is forward vs. every departure from straight walking, not turning vs. rest. F85's 15/15 becomes 14/15 without warm start.

On real turn goals (`ep1200`), the planner enters the turn unaided over ~2/3 of the episode; `--commit 3` vs `--commit 1` halves switching (42->18) and reaches 82% of commanded yaw vs 63%. 59-step episodes only show the entry transient, not whether it converges to the commanded rate (untested).

`ep1300`'s apparent turning without warm start is not control: last-third picks are 74 forward candidates (with positive yaw of their own, +0.003 to +0.009) vs 10 turn candidates, at 35 switches/59 steps — no candidate runs long enough to express its own behavior (F92 stitching cost).

The closed-loop checkpoint (`wm/runs/beh12_hexonly`, `body_dim: 1`, `sources: hexapod` only) has no shared body target across robots — `lambda_body` supervises forward within the insect alone, no quadruped present. Stage 3 (`wm/adapt3.py`) trains only `proj`/`ftm`, so the body head plays no part in the B1 path. `screen_behaviour_channels.py --split condition` on this checkpoint shows nothing transfers (forward hex->b1 -0.112, b1->hex -0.514; lateral/yaw/vertical all worse) even though forward crosses in the loop at 67/84/71%. So cross-embodiment forward transfer is carried by V-JEPA2 features + stage-3 adaptation, not by `lambda_body`. This does not contradict F73 (different checkpoint: stage2_* with two robots vs. stage1 beh12_hexonly) — F73's +0.761/+0.641 must not be quoted for the loop-closing model. No checkpoint yet is both correct (post F65 frame-rate fix) and cross-embodiment; proposed next run: stage 2 on `beh12_c10f10t10_flat` + `beh12_b1_flat`, `lambda_body 0.5` vs `0.0` control.

Linear-readout screens of pooled z (`z_crosses_bodies`, F97) are not predictive of loop behavior — fourth such failure. Initial-pose cue is not the explanation either: forward's starting pose is the *least* distinctive from the grand mean (0.159 rad vs turn_s0.56 0.331, side_R 0.361), the wrong direction for that to be the cause.

---

### F102. Across embodiments the loop transfers the kind of motion, not the amount

Using Froude number as a dimensionless cross-embodiment speed target, 7 hexapod forward goals (Froude 0.129-0.222, 1.72x range) driving the B1 (`--commit 3`, no warm start): B1 achieved Froude ranges 0.0526-0.1490 with mean picked `vx` staying flat at 0.354-0.375 regardless of goal. corr(goal, achieved) = +0.074; corr(goal, mean vx selected) = -0.167 — the planner doesn't even pick faster candidates for faster goals. The B1's own candidate library spans the needed range (vx0.30 at Froude 0.126 up to vx0.50 at 0.206), so it isn't a library limit.

Same robot vs. same robot (`hex_unseen_commit3`): 50% of runs within 15% band, median speed error 14.8%. Cross-embodiment: family selection 66-80% (vs 33% chance) but speed corr ~0 (no tracking).

Conclusion: the loop transfers behavior *category* (what crosses is what a large translation in the image can carry) but not magnitude, across embodiments.

> Read together with F118: the forward-amount-transfer failure measured here stands, but the implied forward-vs-turning contrast does not — F118 shows the planner doesn't condition on goal in either embodiment, so behavior ranking here reflects classifiability, not transfer. F117's cross-embodiment-metric-failure reading is withdrawn.

Readings (controlled / weakly tracked / not tracked) were pre-registered before runs finished.

---

### F103. The pretrained latent does not cross embodiments; the contrastive adaptation objective is what does

Four checkpoints differing only in adaptation amount/loss, same loop/goals, `--warm_start 0`, `--commit 3`, 4 clips per condition (B1 loop in MuJoCo repeats bit-for-bit per F95, so spread comes from goal clip, not reruns):

| adaptation | forward (speed_c5.8) | turning (turn_s0.56) |
|---|---|---|
| frozen world model, projector fitted only | 5% +/- 0 | 2% +/- 2 |
| ITM+FDM adapted separately, MSE | 28% +/- 5 | 12% +/- 2 |
| projector+FDM adapted jointly, MSE | 32% +/- 7 | 2% +/- 2 |
| projector+FDM adapted jointly, + InfoNCE | 74% +/- 3 | 32% +/- 5 |
| chance | 33% | 33% |

Frozen (pretrained-latent-only) is below chance — the pretrained latent does not transfer to this robot in any usable sense. MSE adaptation (either arm) stays near/below chance. Only adding the contrastive term (`--lambda_nce`, same file/clips/architecture/`adapt3.py` code path) raises forward selection to 74% (reaches 84% with forward warm start, per F100). Training budget can't explain it: the MSE arm (`stage3_b1_full`) ran 15,000 steps vs. the NCE arm's (`stage3_b1_nce`) 12,000 — the losing arm got more optimization.

Turning never clears chance in any arm (best 32% +/- 5, on the 33% line); both MSE arms are *below* chance on turning (2%, 12%), i.e. they avoid turn candidates rather than fail to find them.

Same mechanism as F88 (recorded clips: MSE 19%/28% chance -> +InfoNCE 50%) now shown to decide a physics loop: MSE 32%/33% chance -> +InfoNCE 71%.

Conclusion: cross-embodiment transfer is a property of the contrastive adaptation loss term specifically, not of stage/data/architecture/budget. Sideways fails at every rung; the frozen arm's 38% on sideways is not a result — it's the planner defaulting onto side_* candidates when it can't discriminate forward (F86 failure mode).

---

### F104. The B1 walked out of its own camera frame; re-rendered dataset fixes this

Found via preview video: B1 partially out of frame in sideways clips. Measured across all 96 clips (robot mask = diff from clip's own median background):

| | smallest visible area (frac of clip max) | frames touching image edge |
|---|---|---|
| B1 side_R_lvl1 | 42% | 100% |
| B1 other sideways | 85-90% | 100% |
| B1 forward/turning | 69-76% | 36-47% |
| hexapod, all 12 conditions | 83-94% | 0% |

Insect never touches an edge in 48 clips; B1 does in every sideways frame and 1/3-1/2 of the rest. In closed loop, B1 visibility drops to 43-66% on cross-embodiment goals, 47% on its own sideways goal, vs hexapod 89-93%. Not established as sole cause of sideways failure (forward clips also clip 69-76% yet forward works at 74%+/-3) — a real confound, not a proven explanation.

Floor texture ruled out as a factor: B1 background sd 5.53 vs insect 3.74, but nearly all of the gap is a smooth lighting gradient (5.30 vs 3.37), not floor texture (0.99 vs 1.02, effectively identical). Prior readings attributing this to floor pattern are withdrawn.

Fix: widened B1 camera FOV from 15 to 25 degrees (chosen by sweep; 24 deg clears all 48 clips, 25 deg gives 0% edge-touching on all 12 conditions with margin). Trade-off: B1 bounding box goes from 157px (33% larger than insect) to 94px (20% smaller) — clipping loses information, size difference does not, and the robots genuinely differ 4x in size. Also fixed: B1 camera was never pinned to a fixed world point (unlike insect's identical-every-clip background) — now uses `--spawn 0 0`.

Re-rendered and verified dataset: `data/allocentric/beh12_b1_fov25` (also `data/allocentric/beh12_b1_fixed` — final verified version), via `scripts/dataset/rerender_b1_framing.py --cam_fov 24 --spawn 0 0 --floor_scale 3`. `--cam_fov`/`--floor_scale` exist on both `render_b1_replay.py` and `close_loop_b1_physics.py` (defaults 24/3 on the latter) and must agree between rendering and adaptation/loop, else the loop measures a static framing difference rather than behavior.

Durable pitfalls found along the way:
- `sim.scaleObjects` scales a box without moving its center — floor_scale 3x previously sank the walking surface to z=+0.200 while the robot stood at z=0, clipping its feet; none of clipping/background-sd/edge metrics detect this (only pixel count >200 brightness, i.e. foot specular dots, caught it: 24 -> 0 -> 9 after fix). `--floor_scale` now prints `surface +0.000 -> +0.000` to verify.
- Moving the camera back (vs. widening FOV) does not work: shrinks the robot without adding frame room on the side it's leaving.
- Remaining B1-vs-insect gap is color/contrast, not framing (B1 grey-on-grey contrast 25.5 vs insect orange 39.8) — left unfixed to avoid confounding framing fix with appearance change.

Status: MuJoCo was never re-run (clips replayed from stored `base_pos`/`base_quat`/`joint_pos`); this is a re-render, not a re-rollout. Stage 3 still needs to be refitted on `beh12_b1_fov25`/`beh12_b1_fixed` — until then all B1 numbers in this project (including all of F103) stand on the old 15-deg unpinned-camera frames.

---

### F105. One of the four B1 turn levels (`wz0.00`) is the forward clip under another name

Spotted in preview video: `b1_turn_wz0.00` walks straight (commanded yaw rate 0). Measured identical to 4 decimals to the forward clip:

| condition | forward Froude | yaw |
|---|---|---|
| speed_vx0.30 | +0.1259 | +0.0008 |
| turn_wz0.00 | +0.1259 | +0.0008 |
| turn_wz0.08 | +0.1288 | +0.0146 |
| turn_wz0.19 | +0.1297 | +0.0359 |
| turn_wz0.40 | +0.1295 | +0.0760 |

The 12-condition set contains 11 behaviors. Insect's weakest turn (`turn_s0.05`, yaw -0.0072) is milder but not a duplicate.

This biases the two families oppositely: choosing wz0.00 is correct for a forward goal (scored as miss) and wrong for a turn goal (scored as hit). Recomputing F103's ladder scoring by actual behavior instead of label: forward 74%->83% (vs true chance 42%, since forward now holds 5/12 conditions), turning 32%->23% (vs true chance 25%). Both conclusions survive: forward still ~2x chance, turning still doesn't clear it. The contrastive arm gains most on forward and loses most on turning under relabeling — consistent with actually reading the goal.

The `--separability` check can't catch this: it only checks that each level exceeds the one below it on its own channel; it never checks a family's weakest level against a different family. Fix applied: collect a new `wz0.60` level rather than relabel `wz0.00`, restoring 4 levels to turning and extending B1's turn range (was capped at yaw 0.076 vs insect's 0.088). `rollout_b1_mujoco.py --wz` supports this.

---

### F106. F66's turn-direction sign flip was recorded as fixed but was not; it invalidated every cross-embodiment turning result until re-measured

`direction_plan.md` claimed "all four are fixed." Re-measured 2026-08-28 on the data every result in this project used:

| | beh12_b1_flat | beh12_c08f09t09_flat |
|---|---|---|
| weakest commanded turn | +0.0146 | -0.0241 |
| middle | +0.0359 | -0.0372 |
| strongest | +0.0760 | -0.0878 |

The two robots still turn opposite ways (F66 diagnosed this 2026-08-22; the fix was documented but never applied). This means "turning does not cross embodiments" was true by construction in F97, F100, F102, F103's 23-32%-at-chance turning numbers — none of those are evidence about the model, only that the dataset asked for a left turn and offered right ones. Forward and sideways are unaffected (sideways direction already corrected per F96).

Calibration gap also found: F63 matched robots on *commanded* turn rate (within 3%), but *achieved* rates differ — insect's 4 levels: 0.0072/0.0241/0.0372/0.0878; B1's: 0.0008/0.0146/0.0359/0.0760 (only the 3rd pair matches). Sweeping B1 at `--vx 0.30` gives a linear response; calibrated commands to match insect's achieved yaw: `--wz -0.064` (for -0.0072), `-0.153` (-0.0241), `-0.223` (-0.0372), `-0.491` (-0.0878). Forward Froude stays 0.120-0.129 across this range (speed match undisturbed). This also fixes F105 (weakest level becomes a real turn, not the forward clip).

Blocker for re-collection: `rollout_b1_mujoco.py` is deterministic from (0,0), yet the four existing clips per condition start at different points — meaning they were cut as windows from one longer rollout by a script not in the repo. Windowing choice (clip overlap) must be made deliberately when re-collecting.

Fix required: re-rollout twelve clips (not re-render) via `rollout_b1_mujoco.py --wz`, negating the three turn levels and adding F105's real fourth level. Nothing about turning could be claimed until this was done (see F107/F108 for the follow-up).

---

### F107. On the corrected dataset, forward selection survives an action-blind forward model; the contrastive term's real benefit is turn selection, not forward

Full pipeline refitted on corrected `data/allocentric/beh12_b1_flat` (post F104/F105/F106 fixes). Two stage-3 arms now differ only in `--lambda_nce` (same 24 clips, 12,000 steps, batch 8 both). Ten cross-embodiment runs/arm, `--warm_start 0 --commit 3`:

| | forward | turning | sideways | upright | turn sign correct | yaw last third |
|---|---|---|---|---|---|---|
| MSE | 53% +/- 5 | 22% +/- 3 | 19% +/- 1 | 10/10 | 4/4 | -0.0475 |
| + InfoNCE | 54% +/- 0 | 43% +/- 11 | 20% +/- 3 | 10/10 | 4/4 | -0.0353 |
| chance | 33% | 33% | 17% | | 50% | -0.0878 commanded |

Forward is now identical under both objectives (53% vs 54%) — F103's 32% vs 74% gap does not reproduce; likely cause is F105 (old set filed the forward clip as `turn_wz0.00`, penalizing correct forward selection on forward goals).

The contrastive term's real benefit is turn selection: 43% vs 22% (MSE below its own 33% chance), consistent with offline family accuracy (52% vs 23%). Both arms get turn sign correct on 4/4 goals (first time measurable at all, post F106 fix), but MSE achieves more raw rotation (-0.0475 vs -0.0353) by alternating between the fastest straight clip and the hardest turn candidate, while the contrastive arm picks the turn level nearest the goal rate — neither reaches commanded rate (54%/45% of it). On turn goals specifically: MSE picks `speed_vx0.50` (yaw +0.0007) 35% of the time vs InfoNCE's 22%, and `turn_w0.075` (yaw -0.0750) 15% vs 9%; InfoNCE instead picks `turn_w0.037` (yaw -0.0371, nearest the goal rate) 27% of the time, a level MSE never picks.

`/mean-z` ratio (from F88, checks if forward model ignores action; 1.0 = fully ignores): MSE = 0.977 (nearly ignores action) yet still selects forward at 53% — forward is separable from frame content alone, so picking it is not evidence the world model uses the action. +InfoNCE = 0.493 (uses action), and this is exactly what turning needs (can't be chosen without using the action): MSE turning sits below chance (22%), contrastive clears it (43%). On forward, where even a near-collapsed model beats chance, the contrastive term buys nothing. This reframes F90 (rollout deletion costing 30 points was measured on an action-using arm; an already-action-blind model loses far less).

Caveat: comparison confounds objective and budget/batch. Retraining MSE with different configs on the corrected data:

| MSE arm | batch | steps | data | forward | turning |
|---|---|---|---|---|---|
| the original | 16 | 15,000 | old | 32% | 2% |
| refit, original config | 16 | 15,000 | v2 | 58% | 36% |
| refit, matched to the contrastive arm | 8 | 12,000 | v2 | 53% | 22% |

Data alone (same objective, same optimizer settings, same clip count, old vs v2) moves forward +26pts, turning +34pts. But the batch/step configuration is worth 14 points to MSE on turning by itself (36% at batch16/15k vs 22% at batch8/12k) — so the "matched" ablation above gave MSE the weaker configuration. Against the stronger MSE baseline, the contrastive arm's turning advantage shrinks to 43% vs 36% (7 points, inside its own +/-11 spread) — the claim that the contrastive term buys turn selection is NOT yet established; both arms were being retrained at matched batch 8/15,000 steps to settle it.

Sideways stays at chance under both arms (19%/20% vs 17%) — fourth independent measurement of this failure, first on data with no known defect.

---

### F108. The two hexapod bodies (pretraining vs. goal source) turn in opposite directions; fixed by re-collecting to match the pretraining body

World model pretrained on `beh12_c10f10t10_flat`; cross-embodiment goals come from `beh12_c08f09t09_flat`. Their turn ladders have matching magnitudes and opposite signs on all 32 clips (e.g. turn_s0.56: +0.0762/+0.0759/+0.0790/+0.0788 vs -0.0857/-0.0899/-0.0857/-0.0899).

F84's comparison table had this data but left the training body's cell blank, so the reversal (unlike the sideways reversal, which was caught and reported) went unnoticed for a week.

Root cause confirmed via walked-path visualization (`results/wm/dataset/figures/turn_paths_three_sets.png`) and the canonical `yaw_rate()` helper (which correctly handles that hexapod `body_quat` is `(x,y,z,w)` off an aft-pointing abdomen z-axis, vs B1's MuJoCo `(w,x,y,z)` — documented in `wm/data/embodiment.py` docstring). Two ad-hoc quaternion checks written to cross-verify were themselves wrong (assumed `(w,x,y,z)` for both) and gave nonsense; use `yaw_rate()`, never hand-roll a quaternion helper for these two robots. Likely a real morphology effect (same CPG/`--spin` levels, same leg-ratio change that reverses the weak sideways gait per F84 plausibly reverses turning too), not a bug — but it means `turn_s0.56` doesn't denote the same behavior across bodies.

Consequences: (1) the pretrained model never saw the insect turn the way goal clips turn; (2) B1 was matched to the goal body (F106) — correct for the loop, but disagrees with the pretraining insect; (3) any hexapod-to-hexapod turning comparison across these two bodies (including F85's held-out-body loop) is sign-inconsistent. Does not invalidate F107 (there, goal source and candidate library are both negative — a coherent question; the pretrained model's opposite-sign exposure is a handicap shared by both arms).

`collect_beh12.py --separability` previously only checked turning for *size* (`abs(w) < 0.5*abs(turn_s0.56)`), not sign — same blind spot as F66. Now has `--turn_sign` to catch cross-set sign disagreement (each of the 3 sets — pretraining, goals, candidates — was internally sign-consistent, which is why nothing failed before; disagreement only shows up across sets, and no check compared sets until now).

Fix: re-collected held-out body and B1 to match the pretraining body's sign (re-pretraining `c10f10t10` is the far more expensive artefact, so the choice was economic). `collect_beh12.py` gained `--spin_sign`; verified `--spin -0.56` gives +0.0814 vs `+0.56`'s -0.0878 (same magnitude, flipped sign). All three sets now agree in sign across all 4 levels.

The direction was also confirmed in the image frame, which is the frame that matters: taking the principal axis of the robot's silhouette through a clip, all three sets rotate anticlockwise on screen (+140deg pretraining, +152 held-out, +54 B1) — the two scenes share identical cameras, so on-screen sense is a shared language, and being unable to name the turn direction from the picture alone would concede the project's own claim that vision carries shared meaning across bodies.

Lesson: an empty cell in a comparison table reads as "not applicable" but can mean "not checked" — fill every cell or state why it's missing.

---

### F109. F108's corrected turn ladders reproduce independently; the aggregation method matters and three data leaks were found and fixed before the rebuild

Independent re-measurement (2026-08-29, from a fresh session, before spending GPU on the rebuild — because F108 records the fix and F106 records a fix that had not yet happened) of F108's turn table matches to four decimals:

| | level 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| beh12_c10f10t10_flat, pretraining | +0.0032 | +0.0141 | +0.0363 | +0.0775 |
| beh12_c08f09t09_flat, held out | +0.0069 | +0.0215 | +0.0407 | +0.0863 |
| beh12_b1_flat, candidates | +0.0105 | +0.0268 | +0.0401 | +0.0807 |

Aggregation convention that must be used to reproduce: median over frames of a clip, then mean over clips of a condition (what `achieved()`/`_channels()` in `scripts/dataset/collect_beh12.py` do, `dt`=0.05 hard-coded). A naive mean-over-frames gives different numbers (+0.0029/+0.0148/+0.0353/+0.0736 on the pretraining set — same signs/ordering, wrong in the third decimal, 5% low at top level) because `yaw_rate`'s one-second convolution window (`same` mode) averages the first/last half-windows against zero padding, pulling the mean down; a median ignores them. Trimming those frames instead overshoots to +0.0813. A turn-ladder number is only comparable to another aggregated the same way.

Three data leaks found and fixed before the stage-3 rebuild ran (rebuild was 19 minutes into six stage-3 seeds when found, and was restarted rather than annotated):
1. `wm/adapt.py` permuted the full 48-clip directory for stage 1's 9 adaptation clips, causing overlap with the 12-clip candidate library and validation clips used by `adapt3`/loop (specifically `b1_ep100` and `b1_ep1300` were both candidates, plus three validation clips). Fixed: `wm/adapt.py` now takes `--train_clips`; both stage 1 and stage 3 source the same 24-clip list from `scripts/run/b1_stage3_clips.sh`; stage 1 draws 9 from those 24 (`b1_ep1002, 102, 1102, 1302, 2, 2201, 2302, 301, 302`) with zero overlap with candidates/validation. Whether the previously deleted `adapted_b1_v2.pt` had the same contamination is unknown (train_paths were recorded only inside the deleted checkpoint, which is gone).
2. Stage 1's 9-clip draw is now stratified (`--stratify` in `wm.adapt.select_clips`, walks families in turn for coverage) instead of a plain permutation: old pool covered only 6/12 conditions (turn_w0.075 x3, one sideways clip total); new pool covers 9/12 conditions, three per family (`b1_ep1002, 102, 1102, 1202, 2, 2001, 201, 2101, 2201`) — better by construction, not accident. Selection logic exposed as callable `wm.adapt.select_clips(paths, clips, test_clips, seed, stratify)` without needing to load the 383MB checkpoint/encoder — previously this lived inside `main()`, which is why nothing caught the contamination for the length of the original run.
3. Stage 2's `fit_projector` had fitted on all 48 B1 clips (candidates+validation included) — milder since stage 3 retrains the projector for 15,000 steps under a different objective at `lr_proj 1e-3` vs stage 1's forward model being only nudged at `lr_ftm 1e-5`, but how much of the memorised `(a, z)` survives 15k steps was never measured, so "mild" was an argument rather than a number. Fixed: both sheets now pass `--exclude $HOLDOUT` (the 24 non-training clips, full `.npz` names — matching is by prefix, note `b1_ep100` is a prefix of `b1_ep1000`-`b1_ep1003`).

Note: stage 1's pool changed from 9-of-48 to 9-of-24, so its rollout ratios are no longer comparable to F45 or F87 — quote only against runs under the new rule. None of the three leaks could change the MSE-vs-contrastive ordering (all six runs share one stage 1 checkpoint and one projector), but each would have been an uncontrolled caveat on every absolute number the run produces.

---

### F110. Three seeds confirm: the contrastive term makes the quadruped's actions selectable offline; MSE is below chance

Measurement requested by F107, on corrected data with F109's three leaks closed. Six stage-3 runs (3 seeds per arm), identical except `--lambda_nce`: same stage 1 checkpoint, same stage 2 projector, same 24 clips, 15,000 steps, batch 8. Averaged over last 21 evaluations (step 10,000+, window fixed before runs):

| arm | family | cond | /mean-z | /hold |
|---|---|---|---|---|
| contrastive (--lambda_nce 1) | 54.8% +/- 1.1 | 28.6% | 0.490 | 0.891 |
| MSE (--lambda_nce 0) | 21.6% +/- 0.3 | 7.8% | 0.985 | 0.802 |
| chance | 28% | 8% | -- | -- |

Per-seed spread is tight and non-overlapping: contrastive 53.7/55.0/55.9, MSE 21.3/21.7/21.8 — settles F107's ambiguity (there, MSE at one budget hit 36% vs contrastive's 43%, within spread, at mismatched budgets).

MSE is below chance (21.6% vs 28%; cond 7.8% vs 8% guessing rate) because `/mean-z` 0.985 means the forward model returns nearly the same prediction regardless of action — nothing to rank, so sub-chance score is ranking noise. Conversely MSE has the better training loss (0.69 vs 1.12) and better held-out prediction (/hold 0.802 vs 0.891) — selection ability and prediction accuracy dissociate cleanly, and the objective decides which one you get.

Reproduces F88's pattern on data now free of the four B1 defects, slightly better on both arms (F88: contrastive /mean-z 0.62, cond 29%, family 50%; MSE 0.993/6%/19%).

Scope: this is offline discrimination over recorded clips (ranking 12 candidate actions against a known next embedding), not closed-loop. F90's warning applies — forward selection has cleared chance before with /mean-z as high as 0.977.

Scripts: `scripts/diagnostics/summarise_stage3_seeds.py` (no log kept — rerun against seed checkpoints to reproduce). `adapt3` stores only the final step in its checkpoint; family metric wanders ~4 points between evaluations (nce seed 0 final step reads 57% vs its 53.7% window mean).

---

### F111. In closed-loop physics the objective's 33-point offline lead shrinks to 6 points, and neither arm actually turns

Closed loop on F110's checkpoints (MuJoCo physics, CoppeliaSim camera), 12 episodes: two stage-3 arms x six goal clips (2 per family, both sideways signs), goals are `...3` validation clips (not in candidate library or training set). Only `--ckpt` differs between arms.

Selection accuracy over the planner's 330 picks per arm:

| | offline (F110) | in the loop |
|---|---|---|
| contrastive | 54.8% | 41% |
| MSE | 21.6% | 35% |
| chance | 28% | 28% |

Gap collapses from 33 points to 6; MSE moves from below chance to above chance. Most of the offline discrimination signal does not survive being evaluated on self-driven frames.

Actual robot behavior (median over planned steps) vs target, both arms: survival 12/12, but the speed criterion (within band of target) is 0/6 in both arms. Turn and sideways magnitudes are only 5-20% of target (signs mostly right, one reversal: contrastive arm on sideways right) — the quadruped barely turns/strafes rather than turning the wrong way.

Conclusion: F110's offline contrast is real (3-seeded) but does not carry into control on this robot — selection scores and loop behavior have decoupled before (F90).

Bug found and partially fixed: `score_closed_loop.py`'s `S.R. behaviour` class is decided by largest channel, and forward speed exceeds yaw in every turn condition on both robots (F98), so a turning goal with a walking outcome is misclassified as "forward" and passes for free (reads 67% vs 50% here, neither meaningful). The speed column was fixed to use `channel_for`; `ok_class` was not — needs fixing, and every previously-quoted `S.R. behaviour` number needs re-reporting.

Data: runs in `results/wm/closed_loop/b1_{nce,mse}_s0_b1_ep*/`.

---

### F112. With turn signs finally agreeing across all sets, cross-embodiment (insect-goal) control is still at chance and the quadruped does not turn

First cross-embodiment loop run on data where all three sets turn the same way (F109). Goals from held-out insect `c08f09t09`; candidates are B1 clips (only executable ones); `--demo` fixed to one forward B1 validation clip so only the goal varies (per F100/F101). Twelve episodes, two stage-3 arms, six goals.

Selection over 330 picks per arm:

| | contrastive | MSE | chance |
|---|---|---|---|
| same-robot goals (F111) | 41% | 35% | 28% |
| insect goals | 32% | 33% | 28% |

Both arms at chance and indistinguishable — the objective that separates by 33 points offline and 6 points same-robot separates by -1 point here.

Robot behavior (median over planned steps): survival 12/12, speed criterion 0/6 both arms. The robot walks forward regardless of goal — yaw achieves only 2-10% of target (twice with wrong sign), lateral only 1-11% of target. On forward speed, MSE is actually the better arm (23%/53% error vs contrastive's 47%/69%).

Conclusion: the F108 sign correction was necessary but not sufficient — before it the turning question was unaskable (insect and B1 candidates turned opposite ways); now askable, and nothing crosses. Two caveats: (1) the fixed forward demonstration gives 10 warm-start forward commands, making "walks forward regardless" the easiest null to fall into — a neutral/standing demo control has not been run; (2) this is one seed per arm (F110's three seeds cover only the offline claim).

Bug fixed: `score_closed_loop.py` previously read the reference from `--demo` (the neutral B1 clip) rather than the actual goal in cross-embodiment runs, scoring against the wrong target. Now prefers the recorded `goal` when it differs from the demo, takes `--goal_dir`, and scores using the reference's condition channel. Same-robot numbers unaffected (verified unchanged, e.g. `b1_nce_s0_b1_ep3` still reads 20.9%).

Data: runs in `results/wm/closed_loop/b1_hexgoal_{nce,mse}_s0_*/`.

---

### F113. Committing three steps restores B1 yaw magnitude in cross-embodiment loops, but it is not tracking the goal's turn

36 cross-embodiment episodes, two stage-3 arms x six insect goals x three loop settings (commit 1/warm 10 [=F112], commit 3/warm 10, commit 3/warm 0). Survival 36/36.

Committing widens achieved yaw range (-0.005..+0.021 at commit 1 to +0.004..+0.046 at commit 3; contrastive arm's `turn_s0.29` lands within 0.9% of goal — first apparent "success" on a cross-embodiment run). But this is not turning-specific: a *forward* goal produces as much yaw as a turning goal in the same batch (e.g. commit 3/warm 10: turn goals +0.041/+0.046 vs forward goals +0.023/+0.037). The 0.9% match is just a robot yawing 0.02-0.05 regardless of goal, coincidentally near 0.041.

Goal-yaw vs achieved-yaw correlation across the three settings: contrastive r = -0.30, +0.76, +0.33; MSE r = -0.18, -0.07, -0.66 — sign flips between settings within one arm, i.e. no real relationship at n=6. Matches F100's -0.33/46% sign agreement on the (pre-fix) old data.

Selection stays at chance under every setting (contrastive 32/34/36%, MSE 33/35/29%, chance 28%).

Conclusion: the turn-sign defect (F108) was real and fixing it was necessary, but it was not what stood between this pipeline and cross-embodiment turning. With sign now correct: forward travel crosses embodiments, turning does not (F100's claim confirmed without the sign confound). Warm start is not load-bearing for survival either — at warm 0 the robot starts standing and still survives 65/65 in all twelve runs; its only effect is setting an initial yaw the robot then coasts on (per F100).

Data: runs in `results/wm/closed_loop/b1_hexgoal_{nce,mse}_s0_c{1,3w10,3w0}_*/`.

---

### F114. Correcting the cross-embodiment goal metric for embodiment offset recovers offline selection

> Superseded by F116: the reading below ("mean shift recovers cross-embodiment selection") is withdrawn. F116 ran the control this entry called for and found the planner isn't reading the goal at all — the recovered score is a readout of the current frame `e_t`, not goal-following. The raw measurements here are correct; the interpretive claim is not.

The planner scores raw MSE between predicted B1 embedding and goal embedding; in cross-embodiment runs the goal is a hexapod frame, and embodiment is strongly decodable from these embeddings (F37, F40), so distance is dominated by which robot is pictured. No correction existed (checkpoints carry no `embedding_offsets`, `center_embeddings` false).

Tested offline (no simulator), checkpoint `stage3_b1_nce_s0`, 8 demonstrations, 95 decisions. `plan_open_loop.py` gained `--goal_dir`/`--goal_embodiment` and `--center` (translates goal clip into driven robot's mean appearance; only the goal is shifted, not `e_t`):

| goal | exact condition | right behaviour |
|---|---|---|
| B1's own clips | 19.4% | 47.2% |
| insect, raw MSE | 4.2% | 34.7% |
| insect, goal mean-shifted | 23.2% | 55.8% |
| chance | 8.3% | 25.0% |

Uncorrected exact-condition selection was below chance; corrected, both metrics clear the same-robot baseline. This located the failure of F111-F113: those loops scored candidates against an insect goal in raw embedding space, so the planner was ranking mostly on "which robot is this," not behavior. Scope noted at the time: open loop only (recorded clips, not self-driven frames), one checkpoint/seed, and mean shift is only a first-order correction. See F115 (loop test) and F116 (mismatch control) for the actual resolution.

Scripts: `wm/policy/planner.py` (where the shift was to be applied), `plan_open_loop.py`.

---

### F115. The goal mean-shift that fixed offline selection (F114) does nothing in the closed loop

F114's correction applied in physics via new `--center_goal` flag on `close_loop_b1_physics.py` (B1 reference taken from its own demonstration clip; only goal shifted). Twelve episodes, both arms, otherwise identical to F113's commit 3/warm 0 setting.

| | offline (F114) | in the loop |
|---|---|---|
| raw MSE, behaviour | 34.7% (chance 25%) | 36%/29% (chance 28%) |
| goal shifted, behaviour | 55.8% | 36%/34% |

Offline the shift is worth +21 points; in the loop it's worth nothing. Survival stays 12/12, speed criterion stays 0/6, achieved forward speed range narrows slightly (0.072-0.093 vs goals 0.017-0.218 for contrastive, narrower than unshifted).

It does redistribute which family gets picked (not visible in the aggregate): contrastive raw sideways/forward/turning = 16/52/40%, shifted = 18/32/58% (turning rises, forward falls to near chance). Yaw still doesn't track the goal: r = -0.14 (contrastive), +0.13 (MSE); 83% sign agreement in both is uninformative since all goals and the robot's own drift are positive (F100).

Conclusion: the goal-metric defect (hypothesis a) is real but is not the explanation for loop failure — fixing it offline doesn't help in the loop, pointing to hypothesis (b), the 5-step rollout on self-driven off-manifold frames, as the remaining candidate (tested next in `does_rollout_matter.py`/`loop_frames_are_off_manifold.py`).

Flagged control (not yet run at time of writing, resolved in F116): shuffle which insect behavior is paired with which demonstration, to check whether the offline gain reflects goal content or just landing on the B1 manifold.

Data: runs in `results/wm/closed_loop/b1_hexgoal_{nce,mse}_s0_c3w0ctr_*/`.
Known bug: the shift-norm diagnostic print computes norm after the shift is applied, so it always prints 0.00 (cosmetic only; the shift itself is correct).

---

### F116. The planner does not read the goal at all; every offline selection number in this project shares this confound

Mismatch control (`plan_open_loop.py --mismatch`) pairs each B1 demonstration with an insect goal from a *different* behavior family, scoring picks both against the demonstration and against the goal actually shown. With a matched goal these numbers are identical by construction — which is why nothing before could distinguish them.

Contrastive checkpoint, 8 demonstrations, 95 decisions, goal mean-shifted as in F114:

| goal shown | scored vs demonstration | scored vs goal |
|---|---|---|
| matched (F114) | 55.8% | 55.8% (identical by construction) |
| a different behavior | 55.8% | 22.1% (below 25% chance) |

Showing a mismatched goal doesn't move picks at all — 55.8% vs the demonstration either way. The 55.8% is a readout of `e_t` (the quadruped's current frame), not goal-following. This explains F114/F115: the mean shift made the goal term nearly constant across candidates, so ranking fell back on `e_t`, which correlates with the demonstration's condition (55.8% offline) but in closed loop `e_t` is just whatever the robot is currently doing — a readout of it is not a controller, hence F115's null loop result.

This confound is not confined to this experiment: any offline selection score using a goal drawn from the demonstration's own future (every prior number in this project, including the same-robot 47.2% baseline and F90's 62%) cannot distinguish "reached the goal" from "named the behavior already visible in the current frame." The mismatch control should be attached to any such number before it is quoted.

Does not invalidate the forward model itself: F110's discrimination (given the true next embedding) still ranks at 54.8% vs 28% chance — a different question. What fails is the planning objective as implemented (minimize distance from rollout prediction to goal embedding) under cross-embodiment goals.

Full control matrix (contrastive checkpoint):

| metric | scored vs demonstration | scored vs goal shown |
|---|---|---|
| raw MSE | 33.7% | 38.9% |
| goal shifted by clip mean | 55.8% | 22.1% |
| goal shifted by dataset mean (avg over 12 clips/robot) | 49.5% | 25.3% |
| chance | 25% | 25% |

Raw MSE weakly follows the goal (38.9% vs 25% chance); every centering method tested destroys this and substitutes an `e_t` readout (dataset mean no better than clip mean — "behaviour lives in the clip mean" hypothesis refuted too). Honest number for cross-embodiment goal-following: 38.9% vs 25%, uncorrected. `--center` kept in script (default off) to keep the negative result reproducible.

Two silent pairing bugs found while building this control, both from cross-robot naming mismatches: matching by condition string kept only sideways clips (`speed_vx0.30` vs `speed_c5.8` share no prefix) and silently reported 0.0% from 12 decisions; a family rotation over `side_L`/`side_R` matched almost nothing since the recorded `behaviour` field only has three values (`speed`,`turn`,`side`), silently reporting a misleading 66.7% from 2/8 demonstrations. Both failed by skipping, not raising. Script now refuses to start if any demonstration is left unpaired.

Logs: `/tmp/ol_mismatch.log`, `/tmp/ol_mismatch_raw.log`, `/tmp/ol_dsmean_mismatch.log`.

Next: `does_rollout_matter.py`'s `blind` arm should run before `loop_frames_are_off_manifold.py` — check whether the goal term influences the argmin at all under any metric.

---

### F117. The forward model rollout works well same-robot, but this measured behaviour-classification not goal-following

> Superseded/withdrawn by F118 (and F116's confound applies): the goal-following reading below is wrong. Mismatch control (different-behavior goal, same robot) shows the rollout scores 56-70% against the demonstration but only 18-23% against the goal actually shown (below 28% chance) — the rollout names the behavior the robot is already in, not goal-directed selection. Treat all numbers below as "state classification accuracy," not planning.

`does_rollout_matter.py` on contrastive stage-3 checkpoint, 48 clips, 12 candidates, 2,340 transitions from held-out clips, same-robot goals:

| horizon | rollout | direct | blind | chance | per family (rollout) |
|---|---|---|---|---|---|
| 1 | 61% | 31% | 34% | 28% | side_L 100%, side_R 100%, speed 30%, turn 53% |
| 3 | 71% | 32% | 38% | 28% | side 98/100%, speed 46%, turn 67% |
| 5 | 73% | 35% | 36% | 28% | side 96/100%, speed 39%, turn 82% |
| 10 | 72% | 34% | 35% | 28% | side 86/100%, speed 31%, turn 89% |

Rollout beats `direct`/`blind` by 27-37 points; deleting rollout (`direct`) costs almost all of that gain — the world model is acting as a predictor here, not a similarity function. Horizon 5 is best, horizon 10 is no worse (rollout is not fragile to horizon length as assumed). Turning is the only family that improves monotonically with horizon (53%->89%) and sideways stays at 86-100%; forward is weakest throughout (30-46%).

This inverted F100/F102's "forward crosses, turning does not" framing — both of those were cross-embodiment measurements, and F117 originally localised the cross-embodiment goal comparison as the one broken component (with the metric sound, turning is mechanically the strongest family since it's a sustained low-frequency signal a longer rollout accumulates). That framing was later shown (F116) to be an artifact of the cross-embodiment goal-metric defect, not a property of the behaviors. Correction notes were added to F100 and F102.

Correction to this entry's own localization claim: F118 shows the planner fails to condition on goal even *within* one robot (no cross-embodiment metric involved), so "everything works except the cross-embodiment comparison" is false — F116's metric defect sits on top of a planner that wasn't using its goal at all, in either setting.

Comparison to F116 (same checkpoint/candidates/projector/rollout): goal from B1 itself scores 73% vs 36% blind baseline; goal from insect scores 38.9% vs 25% chance baseline — this gap was later shown to be about goal-conditioning generally, not specifically the cross-embodiment comparison.

Design-change proposal that followed from this (pursued in F119/F120): a raw embedding distance was never a shared coordinate between the two robots, only hoped to become one via the shared frozen encoder. Score candidates instead by the body-motion head that `lambda_body` supervises (F51, F58, F59) — a dimensionless coordinate both robots are measured in by construction — comparing predicted body motion against the goal clip's body motion, rather than raw embedding distance.

Caveat on the 73%/38.9% numbers: same-robot goals here come from clips the candidates did not come from, but the goal is still the demonstration's own future, so F116's confound applies to this table too (part of 73% may be a readout of `e_t` rather than goal-following) — this is exactly what the F118 mismatch control (not run in this script) was built to test.

---

### F118. Even within one robot, the planner's argmin does not follow its goal; the rollout is a state classifier, not a controller

The control F117 called for, run before its 73% was quoted anywhere. `does_rollout_matter.py` gained `--mismatch`, which takes each demonstration's goal frame from a clip of a different behavior family of the *same* robot, scoring the rollout's pick both against the demonstration and against the actual goal shown.

| horizon | rollout vs demonstration | direct | blind | rollout vs goal shown | chance |
|---|---|---|---|---|---|
| 1 | 56% (61% matched) | 36% | 34% | 18% | 28% |
| 3 | 64% (71%) | 26% | 38% | 18% | 28% |
| 5 | 68% (73%) | 30% | 36% | 23% | 28% |
| 10 | 70% (72%) | 24% | 35% | 21% | 28% |

Showing a mismatched goal barely changes agreement with the demonstration (drops 3-7 points) and leaves the rollout below chance against the actual goal. F117's 73% was never goal selection — it is the same confound as F114, now measured within one embodiment, where no appearance mismatch and no cross-robot metric can be blamed. This corrects F117's diagnosis: the goal never drives the argmin, in either same-robot or cross-embodiment settings; F116's metric defect merely sits on top of that deeper failure. This is the deepest correction of the session — it reframes the whole diagnosis from "everything works except the cross-embodiment comparison" to "nothing was conditioning on the goal at all."

The rollout is doing real work, just not planning: it beats `blind` by 22-35 points under mismatch (forward model contributes information about which behavior the robot is currently in and how it continues), while `direct` sits at or below `blind` throughout — a good predictor wired as a state classifier.

Turning ordering from F117 survives as a real, useful capability (39%->88% with horizon under mismatch; sideways stays 85-100%) — it's rollout-based behavior identification, not planning. F100/F102's "forward crosses, turning does not" remains corrected.

Consequence: any proposed fix must be judged on the `roll/goal` column (score vs. the goal actually shown), not the score against the demonstration, which is passable without reading the goal at all. Applies directly to F117's body-motion-head scoring proposal, carried forward into F119/F120: the predicted-body-motion-vs-goal score must clear 28% on a mismatched goal, or it has changed the coordinate without changing what the planner uses.

This also puts a number on the teacher-student argument (Q16): candidate scoring here is not failing to pick the best candidate — it is not conditioning on the request at all. Emitting a policy makes the request part of training rather than something a run-time argmin has to honour.

Log: `/tmp/rollout_mismatch.log`.

---

### F119. Scoring by predicted body motion (shared coordinate) restores goal-conditioning same-robot; cross-embodiment result was an artifact of an unadapted head (see F120)

The falsification F118 demanded, run with its own control in the same pass so no number could stand alone. `scripts/diagnostics/score_by_body_motion.py` scores candidates by `score(a) = |body_head(proj(a)) - forward speed of goal clip|` — a dimensionless physical quantity, no embedding distance, no rollout, no `e_t`, no forward model at all. Matched and mismatched goals are computed in the same run.

Same-robot goals — first rule in the project that conditions on its goal:

| horizon | matched | mismatched vs demonstration | mismatched vs goal shown | chance |
|---|---|---|---|---|
| 1 | 49% | 33% | 49% | 28% |
| 3 | 47% | 31% | 49% | 28% |
| 5 | 45% | 28% | 49% | 28% |
| 10 | 47% | 25% | 51% | 28% |

Compared to the embedding rule under the identical control (F118: 18-23% vs goal, 56-70% vs demonstration), the ordering inverts — picks follow the target, not the current frame.

Cross-embodiment, the same rule does NOT separate:

| horizon | matched | mismatched vs demonstration | mismatched vs goal |
|---|---|---|---|
| 1 | 48% | 48% | 44% |
| 5 | 48% | 47% | 47% |
| 10 | 50% | 49% | 49% |

Matched and mismatched columns agreeing (~44-50% throughout) is the signature of a near-constant pick, not goal-following.

Why (measured, not assumed): the two datasets' speed conditions are calibrated to each other (B1 `speed_vx0.30` walks at 0.126 vs insect's `speed_c5.8` at 0.129; `speed_vx0.50` at 0.206 vs 0.215; every condition pairs within a few percent). What fails is the head's reading of the candidates, e.g. true forward speed vs `body_head(proj(a))`: `side_L_lvl0` -0.008 vs 0.079, `side_R_lvl1` 0.013 vs 0.119, `speed_vx0.30` 0.132 vs 0.129, `speed_vx0.50` 0.215 vs 0.126, `turn_w0.075` 0.127 vs 0.102 — true spread 0.223, predicted spread 0.059 (3.8x compression, correlation +0.60), the four speed levels read as identical to three decimals. The head can say "sideways or not" and nothing finer, so the argmin saturates: any target above the achievable band selects the same candidate, which is why matched and mismatched agree.

> Corrected by F120: this was originally diagnosed as the shared coordinate being "too narrow" (`body_dim 1`, forward-only, same channel-competition problem F73 measured and did not solve). That diagnosis is wrong. The body head used here (`beh12_hexonly`) is hexapod-only pretrained and has never seen a B1 latent — neither `wm/adapt.py` nor `wm/adapt3.py` touches the motion decoder. On the hexapod, the body it was trained on, it reads forward speed at correlation +0.99, compression 1.0x (exact), matching every family to three decimals. The 3.8x compression measured here is the head being evaluated out-of-domain, not a width limitation. The tables above are correct; the conclusion drawn from them is not.

Revised implication (per F120): the same-robot 49% result is a real, valid demonstration that scoring in a shared coordinate restores goal-conditioning, and it is a lower bound, since it was measured with a head reading B1 as near-constant. It also corrects F118's stronger claim that run-time search cannot be made to honour a goal — it can, this is the demonstration. The concrete next experiment is a pretraining change, not a planner change: widen the body head to yaw and lateral (`body_channels 0,1,2`) and check its calibration against measured speed on both robots before scoring anything with it.

Log: the tables above are reproducible with `--goal_dir data/allocentric/beh12_c08f09t09_flat`.

---

### F120. The body-motion head is exact on the robot it was pretrained on and returns the dataset mean on the robot it never saw; adaptation degrades it further

Body head (2-layer MLP on `z`) evaluated directly on ITM latents from real frames, both robots, using existing checkpoint weights:

| robot | correlation | true spread | predicted spread | compression |
|---|---|---|---|---|
| hexapod (trained on) | +0.99 | 0.194 | 0.194 | 1.0x |
| B1 (never seen) | +0.20 | 0.197 | 0.061 | 3.2x |

Per family, hexapod predictions match true values to three decimals (e.g. forward 0.160->0.161); B1 predictions collapse to near the dataset mean (~0.109) regardless of true family value. This is correct behavior for a model queried outside its training domain, not a head defect.

Cause: `beh12_hexonly` is pretrained on `sources hexapod=...` only; `wm/adapt.py` explicitly does not adapt the motion decoder, `wm/adapt3.py` only fine-tunes projector and forward model. The shared body head has never seen a B1 latent at any pipeline stage. F119 scored candidates with this head and misattributed the resulting flatness to coordinate width rather than domain mismatch.

Implication: widening to `body_channels 0,1,2` (as F119 proposed) would still be evaluated on a hexapod-only pretrain that has never seen the target robot — a negative result there would say nothing about channel competition (F73). The pretrain must include both embodiments; F73's own run is forward-only and carries the frame-rate defect (F65), so it needs redoing on `beh12_*` regardless.

F119's same-robot 49% goal-conditioning result stands and is now understood as a lower bound (measured with a head reading B1 as a near-constant, yet still conditioned on the target).

Second, independent degradation found: adaptation itself further corrupts the shared coordinate. Same head, same B1 clips, two checkpoints:

| ITM latent source | correlation | compression |
|---|---|---|
| beh12_hexonly/best.pt, unadapted | +0.76 | 2.2x |
| stage3_b1_nce_s0.pt, after stages 1+3 | +0.23 | 3.2x |

Adapting the forward model to B1 makes the shared head read B1 worse, because stage 1 moves what `z` means (documented in `wm/adapt.py` docstring) and nothing re-aligns the head to that shift. Calibration must be measured on both the pretrain and the actual deployed checkpoint, not just one. Run sheet now runs `body_head_calibration.py` before stage 1 and again after stage 3.

Tool validated: automated `body_head_calibration.py` reading (+0.23/3.2x for B1, +0.99/1.0x hexapod) matches the hand measurement (+0.20/3.2x) — the earlier apparent disagreement was two different checkpoints being compared, not a tool error.

Prescribed next run (handed to com7):
```
--sources hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat \
    --lambda_body 0.5 --body_dim 3 --body_channels 0 1 2
```
then stages 1-3, then per-channel calibration on both robots. Pass bar set in advance: all three channels within ~1.5x compression on both robots, and `roll/goal` above 28% on mismatched cross-embodiment goals. If forward calibrates but yaw does not, that reproduces F73's channel competition on corrected data, meaning the bottleneck is the pretraining objective, not head width — stop and report rather than tune further.

---

### F121. A shared head can serve both robots once it is fitted on both

`wm/fit_body_head.py` freezes everything and trains the body head alone (~8k params, `z_dim -> body_hidden -> body_dim`) on latents from real consecutive frames via the checkpoint's ITM. Split by clip, 20% of B1 held out.

Held-out correlation / compression vs 1.5x bar:

| head | B1 held-out | hexapod |
|---|---|---|
| as shipped, never fitted on B1 | +0.23, 3.2x | +0.99, 1.0x |
| fitted on B1 alone | +0.81, 1.5x | +0.74, 0.5x |
| fitted on both robots at once | +0.81, 1.7x | +0.97, 1.0x |

Held-out MSE vs predicting target mean, B1: 0.967 before, 0.514 after.

Fitting on both robots recovers both (hexapod +0.97, B1 +0.81, no trade), with per-family numbers matching physics (hexapod sideways 0.020->0.017, forward 0.160->0.155, turning 0.131->0.130). Fitting on the B1 alone flips the sign of the hexapod's sideways prediction (true +0.020 read as -0.067) because B1 strafes near zero forward speed while the insect doesn't; a head shown only one body learns a spurious correlation. The shared coordinate only holds if the fit sees both robots (as in `scripts/run/com7_pretrain_body3.sh`). A pretraining run with both embodiments is not the only route to a calibrated head — post-hoc fitting recovers most of the gap in minutes.

Superseded: fitting against the ITM's latent (as measured here) is the wrong target — see F122. Fitting against the ITM latent lifted the ITM path from +0.20 to +0.79 but left the projector path (what the planner uses) at +0.44 with 2.5x-too-wide range; same-robot goal-following fell from 49-51% to 19-34%. The head must be fitted on the latent it is actually shown.

Durable bug note: an early version of the fit script built the motion decoder with a placeholder action width and saved the whole decoder, overwriting the checkpoint's real 12-D B1 output head with a 1-D one. Fixed by updating only the `body_head.*` keys inside the checkpoint's decoder state.

Checkpoints: `stage3_b1_nce_s0_bodyfit.pt` (B1 only), `stage3_b1_nce_s0_bodyfit_both.pt` (both).
Logs: `/tmp/bodyfit*.log`, `/tmp/bodycal_{fit,both}.log`.

---

### F122. Fitted on the projector latent, the shared coordinate follows cross-embodiment goals above chance

Fitting the body head against the ITM's latent (F121) hurt selection because `body_head(proj(a))` is what the planner actually evaluates, and `a -> z` is one-to-many (F87) so projector latents occupy a different region than ITM latents.

| head, on B1 | ITM latent | projector latent |
|---|---|---|
| as shipped | +0.20, 3.2x | +0.44, 2.8x |
| fitted on ITM latents | +0.79, 1.5x | +0.44, 0.4x (range 2.5x too wide) |

`wm/fit_body_head --latent projector` fits the same 8k params against planner-supplied latents. Held-out MSE vs mean: 0.854 before, 0.132 after.

Selection accuracy vs mismatch control, chance 28%:

| | h1 | h3 | h5 | h10 |
|---|---|---|---|---|
| same robot, vs goal shown | 76% | 79% | 82% | 86% |
| same robot, vs demonstration | 18% | 16% | 14% | 7% |
| insect goals, vs goal shown | 36% | 37% | 35% | 38% |
| insect goals, vs demonstration | 26% | 26% | 26% | 25% |

Same-robot selection tracks the goal (86%) not the demonstration (7%, below chance) — rules out the frame-reading confound that affected F114/F117/F119. Cross-embodiment (insect goals) clears chance at every horizon (35-38% vs 28%), the first cross-embodiment selection number to survive this control. What's shared is not the latent (head is fitted per-robot on that robot's own projector latents) but the target quantity: dimensionless body motion, measurable from outside, same meaning across a 6-legged insect and 12-DOF quadruped.

Scope: one channel (forward speed only), one checkpoint, one seed, offline, no simulator. Head is fitted on the target robot's own actions (same cost as the action projector, F45).

Checkpoint: `stage3_b1_nce_s0_bodyfit_proj.pt`.
Logs: `/tmp/bodyfit_proj.log`, `/tmp/score_proj_{same,cross}.log`.

---

### F123. Using only frames and world-model rollout, quadruped selection follows an insect's video above chance

F122's scoring read the goal's measured trajectory value, not a video — no frame consumed at scoring, forward model never called. Three conditions isolate components, cross-embodiment goals, mismatch control throughout:

| | candidate side | target source |
|---|---|---|
| A | `body_head(proj(a))`, no frames/rollout | goal clip's measured trajectory |
| B | roll FDM h steps from `e_t`, read transition via ITM | goal clip's measured trajectory |
| C | same rollout | goal robot's own frames, `body_head(ITM(g_t, g_t+h))` |

Selection vs goal shown, mismatch control, chance 28%:

| horizon | A | B | C |
|---|---|---|---|
| 1 | 36% | 34% | 28% |
| 3 | 37% | 32% | 35% |
| 5 | 35% | 45% | 39% |
| 10 | 38% | 39% | 42% |
| C tracking demonstration instead | -- | -- | 25-27% |

C is the condition matching the project's claim (nothing but pixels and world model involved) and clears chance from 3 steps, rising to 42% at horizon 10 while its demonstration-tracking stays at chance. B and C both improve with horizon (B peaks 45% at h5, C at 42% at h10); A (never calls the rollout) is flat at 35-38%.

Withdrawn paragraph (see F126): the horizon-dependence argument above conflated two differences between A and C (A also gets a measured trajectory value where C reads target from frames). Mode D (C's vision-only target, no rollout) reaches 41% at horizon 3, overlapping C's 42% best — rollout's contribution is not established by this comparison. Rest of entry stands.

Scope: offline selection among 12 recorded B1 behaviours, one checkpoint, one seed, forward speed only, held-out clips. Not closed-loop, not a controller.

Logs: `/tmp/f132_{A,B,C}.log`; `scripts/diagnostics/score_by_body_motion.py --mode {A,B,C}`.

---

### F124. The contrastive objective, not mode A, is what makes the rollout carry usable information

Comparison of independent stage-3 checkpoints differing in one flag (contrastive vs MSE), same fitting and scoring, cross-embodiment goals, mismatch control:

| | mode A (no rollout, no frames) | mode C (rollout and frames) |
|---|---|---|
| contrastive | 36/37/35/38% | 28/35/39/42% |
| MSE | 36/38/36/37% | 28/24/22/26% |
| chance | 28% | 28% |

Mode A is identical under both objectives (expected — it never calls the forward model). Mode C works only for the contrastive arm; the MSE arm is at/below chance and gets worse with more rollout, consistent with F110 (MSE adaptation leaves `/mean-z` at 0.985 — forward model gives same answer for real vs mean action, so rollout accumulates nothing).

Links F110 (training-objective statement) and F123 (cross-embodiment selection statement) as the same fact: the contrastive term makes the FDM rollout carry action-specific information, and mode C is the only rule that uses it. A checkpoint differing in one flag failing exactly where that flag predicts is stronger evidence than a same-config second seed would be. Second-seed replication (`stage3_b1_nce_s{1,2}`, on com7) still outstanding.

Logs: `/tmp/f133_{A,C}_mse.log`, `/tmp/f132_{A,C}.log`.

---

### F125. Three shared channels (forward/lateral/yaw) calibrate on both robots with no channel trade-off; F73's channel competition does not reproduce

Run `beh12_hex-b1_body3`: pretrained on both embodiments with `body_channels 0 1 2` (forward, lateral, yaw, dimensionless, observed externally). Stages 1-3, with `body_head_calibration.py` before and after adaptation.

Straight off pretrain, every channel clears the 1.5x bar:

| | forward | lateral | yaw |
|---|---|---|---|
| hexapod | +0.99, 1.0x | +0.98, 1.2x | +0.98, 1.0x |
| B1 | +0.99, 1.0x | +0.97, 1.2x | +0.97, 1.0x |

Per-family signs/values match physics (hexapod side_L 0.116->0.111 vs side_R -0.151->-0.119; B1 turning yaw 0.038->0.038).

F73 (yaw costing forward 68%, buying yaw at +0.37+/-0.27) does not reproduce and is superseded: on corrected data with both embodiments in pretrain, forward stays +0.99 with yaw present. F73's result came from a two-embodiment pretrain on forward-walking-only data with the frame-rate defect (F65) and only one channel of variation.

Adaptation degrades calibration as F120 predicted:

| after stages 1+3 | forward | lateral | yaw |
|---|---|---|---|
| hexapod | +0.88, 1.5x | +0.81, 2.7x | +0.83, 1.1x |
| B1 | +0.82, 1.7x | +0.60, 3.8x | +0.80, 1.4x |

Correlations fall to +0.60-0.88, three of six cells fail the bar — second independent confirmation that stage 1 moves the latent without moving the head; F122's fix (refit head on planner-supplied latents, MSE 0.854->0.132) applies.

Open question: whether 3 channels beat F123's 42% forward-only bar under mode C with mismatch control — needs checkpoints, still on com7.

Run: `beh12_hex-b1_body3`, commit ba12c71. No log kept (scratch file was reused/removed).

---

### F126. Deleting the rollout costs nothing: F123's result is from the shared coordinate, not the world model

Reconfirmed independently by F188 (properly-adapted B1 checkpoint via F183/LAC-WM staged adaptation, live rendered closed loop, same-robot control). Same conclusion, different era of codebase: A=D, rollout buys nothing measurable.

F123 compared mode A (no frames/rollout, vs measured trajectory) with mode C (rollout, target from frames) and attributed the difference to the world model — but the two conditions differ in two things at once. Mode D isolates it: C's vision-only target, A's rollout-free candidate score.

Cross-embodiment goals, mismatch control, vs goal shown, chance 28%:

| horizon | A | C | D |
|---|---|---|---|
| 1 | 36% | 28% | 33% |
| 3 | 37% | 35% | 41% |
| 5 | 35% | 39% | 39% |
| 10 | 38% | 42% | 33% |

D matches C (best 41% vs 42%, equal at h5, D better at h1/h3). Rolling the forward model buys nothing measurable; F123's horizon-signature argument doesn't survive (D's target window also widens with horizon).

What stands: cross-embodiment selection clears chance with vision on both sides and no recorded trajectory value anywhere (33-41% vs 28%), demonstration-tracking at chance. The result is the shared coordinate, not the world model. "A quadruped selects its behaviour from an insect's video" is measured; "a world model plans the selection" is not — the best rule found reads a target from the goal video, maps each candidate to the same quantity, and takes the nearest, no future prediction required.

Mode D on the MSE arm is at chance too (33/24/26/28%) — unexpected under F124's story since A and D share the same candidate side and only A works on MSE. Hypothesis (untested): contrastive stage 3 leaves the projector's latents in the ITM's region so a head fitted on one reads the other; MSE's does not.

Per-family mode D baseline (chance: speed/turn 33%, side_L/side_R 17%):

| horizon | pooled | side_L | side_R | speed | turn |
|---|---|---|---|---|---|
| 1 | 33% | 13% | 31% | 23% | 62% |
| 3 | 41% | 20% | 39% | 52% | 62% |
| 5 | 39% | 25% | 38% | 45% | 54% |
| 10 | 33% | 17% | 33% | 18% | 57% |

Turning is best-identified on a forward-speed-only coordinate (54-62%) since turn clips occupy a narrow forward-speed band; side_L stays at chance (invisible to one channel).

Durable bug: first mode-D run reported 0% on every column with healthy sample count because the goal-encoding block still tested `mode == "C"`, silently skipping every sample. Now raises instead of skipping.

Log: `/tmp/f135_D.log`.

---

### F127. With three shared channels, cross-embodiment selection reaches 70% and strafing goes from invisible to near-perfect

Uses the 3-channel pretrain (F125), head refitted (per F120/F122's repair) on `proj(a)` and ITM latents actually shown to the planner, B1 + hexapod, 20% B1 held out.

Calibration after refit, held-out clips:

| | forward | lateral | yaw |
|---|---|---|---|
| B1, held out | +0.95, 1.1x | +0.83, 1.3x | +0.91, 1.6x |
| hexapod | +0.97, 1.1x | +0.96, 1.2x | +0.95, 1.1x |

5 of 6 cells clear the 1.5x bar (B1 yaw misses by 0.1).

Selection, cross-embodiment, mismatch control, per family (chance: pooled 28%, speed/turn 33%, side_L/side_R 17%):

Mode D (target from insect frames, no rollout):

| horizon | pooled | side_L | side_R | speed | turn |
|---|---|---|---|---|---|
| 1 | 70% | 86% | 79% | 50% | 54% |
| 3 | 68% | 92% | 65% | 62% | 45% |
| 5 | 70% | 87% | 69% | 41% | 64% |
| 10 | 70% | 100% | 84% | 43% | 39% |
| demonstration-tracking | 18-21% | | | | |

Mode C (same target, forward model rolled on candidate side):

| horizon | pooled | side_L | side_R | speed | turn |
|---|---|---|---|---|---|
| 1 | 33% | 21% | 62% | 43% | 24% |
| 3 | 43% | 42% | 57% | 55% | 30% |
| 5 | 44% | 43% | 64% | 55% | 28% |
| 10 | 37% | 32% | 40% | 86% | 21% |

Key results: (1) all channels clear their chance rate under mode D at every horizon — the channels transfer. (2) Pooled mode D goes from 33-41% (one channel) to 68-70% (three channels), while demonstration-tracking falls to 18-21% — the largest change measured in this line of work, from widening the target quantity, not the planner. (3) side_L was never hard, just invisible on one channel: 13-25% (at/below 17% chance) on forward-only vs 86-100% with lateral speed included — corrects a claim repeated since F92 that sideways "fails on every measurement." Mode C remains worse than D (33-44% vs 68-70%) and destroys the turning signal (21-30%, below chance) except forward speed at h10 (86%) — F126's reading hardens: the result is the shared coordinate, and rollout currently subtracts from it.

Scope: one checkpoint, one seed, offline selection among 12 recorded behaviours, held-out clips, not closed-loop. B1 head fitted on B1's own actions/latents (same cost as action projector).

Checkpoint: `wm/runs/beh12_hex-b1_body3/stage3_b1_nce_s0_bodyfit_proj.pt`.
Logs: `/tmp/f136_{fit,cal,C,D}.log`.

---

### F128. Without a demonstration library, naive action-space search never finds locomotion

F126/F127 measured selection over 12 recorded clips; the library supplies the "how" (a recorded clip already knows which joint sequence produces a body motion). `sim/control/plan_without_library.py` tests planning over sampled, unrecorded action sequences, goal read from insect frames via shared coordinate:

- condition 1: `body_head(ITM(e_t, FDM rolled h steps))` vs goal (planning)
- condition 2: `body_head(proj(a))` vs goal (no prediction)
- random: uniform pick from same bank

Winner of each executed in MuJoCo, actual body motion measured (not model's own prediction):

| | distance to goal | forward | lateral | yaw |
|---|---|---|---|---|
| world model | 0.331 | 0.186 | 0.082 | 0.248 |
| no rollout | 0.313 | 0.193 | 0.091 | 0.214 |
| random | 0.347 | 0.187 | 0.075 | 0.273 |

All three indistinguishable — every sampled action produces roughly the same motion, so no scoring rule can separate them; the bank contains no solutions.

Confirmed independent of noise scale, 24 samples/setting executed in physics: robot stays upright and travels backwards while rotating at every scale (0.10-1.00 sd of joint sd), essentially the dataset's mean pose held constant — vs recorded clips' forward -0.003..0.206, yaw 0.010..0.076. Low-passed random-walk noise around a mean posture is not a gait; locomotion needs periodic joint trajectories, which uniform/smoothed sampling of an 18-D continuous action space does not produce (`wm/policy/planner.py` docstring already noted this; now measured on the quadruped).

This is a search-space limitation, not a scoring-rule failure. With a library, the coordinate beats the rollout (F126); without one, neither rule has anything to choose between — this experiment could not test whether the world model "earns its place."

Implications: hand-authored CPG parameters would supply gaits but reintroduce per-robot knowledge; optimizing action sequences through the world model (CEM/gradients rather than uniform sampling) is the next honest test (F129); learning a policy (teacher-student) is the option this result argues for most directly.

Log: `/tmp/f137.log`; results in `results/wm/closed_loop/plan_without_library/`.

---

### F129. The forward model's imagined state barely depends on the action, so teacher-student distillation cannot proceed as-is

`scripts/diagnostics/rollout_fidelity.py`, 3-channel two-embodiment pretrain `beh12_hex-b1_body3/best.pt`, rolled on 48 held-out insect clips, actions teacher-forced from the ITM (isolates the forward model).

| horizon | error/holding-still ratio | predicted/actual displacement | latent perturbed 1sd | real latent from another state |
|---|---|---|---|---|
| 1 | 0.732 | 0.41 | 0.741 | 0.775 |
| 3 | 0.702 | 0.63 | 0.708 | 0.767 |
| 5 | 0.764 | 0.76 | 0.768 | 0.826 |
| 10 | 0.978 | 1.02 | 0.984 | 1.013 |

One-step ratio 0.732, growth +0.027/step, by 10 steps 0.978 (indistinguishable from predicting no motion). Usable imagination horizon is ~5 steps (0.25s), an upper bound with true actions supplied. The model is mediocre at step 1 (not a compounding failure) — a statement about the pretraining objective (reconstruction + auxiliary readouts, never rollout fidelity).

Sharper finding: state prediction hardly uses the action. Perturbing latent by 1 sd: error moves 0.732->0.741 (1%). Substituting a real latent from another state: costs 6%. This is the `/mean-z` pathology of F88/F110 present in the pretrain itself (held-out body), not just MSE adaptation; nothing repairs it during pretraining under the current objective.

Verdict: teacher-student distillation does not proceed as-is — if imagined futures move 1-6% with the action, a distilled policy gets almost no action signal. A capped horizon doesn't fix it (defect present at step 1). Points toward a pretraining objective combining a rollout-fidelity term with an action-conditioning term (the contrastive term already measured to restore action sensitivity during adaptation, F110).

Scope: one pretrain, one seed, hexapod held-out clips, teacher-forced actions. Not yet run on B1 or contrastive-adapted checkpoints.

Log: `/tmp/f138_hex.log`.

---

### F130. Contrastive adaptation's action-conditioning and MSE's state fidelity were measured on the wrong (ITM) latent path

Verdict of this entry is withdrawn by F131: all numbers below roll the forward model on the ITM's latents, but F130 itself shows the forward model is action-sensitive only in the projector's region. Re-measured on the projector path, the contrastive arm has state fidelity 0.710 (not 1.370) with `/mean-z` 0.476 — it has both action-conditioning and state fidelity at once. "No cheap fix because no checkpoint has both" is false. Tables below remain correct as ITM-path measurements; the conclusion from them is not.

State-rollout diagnostic (as F129), B1 clips, three checkpoints differing only in post-pretrain step, actions teacher-forced from ITM. Ratio of rolled-state error to error of holding `e_t` still (below 1.0 beats no motion):

| checkpoint | h=1 | 3 | 5 | 10 |
|---|---|---|---|---|
| pretrain, unadapted | 1.573 | 1.712 | 1.750 | 1.781 |
| contrastive (`--lambda_nce 1`) | 1.370 | 1.848 | 2.071 | 2.221 |
| MSE (`--lambda_nce 0`) | 0.592 | 0.862 | 1.008 | -- |

Unadapted pretrain worse than holding still at every horizon (hexapod-only forward model doesn't predict a quadruped, cf. F44). MSE arm predicts state far better (0.592 vs 1.370 at h1) and stays near break-even to h5 where contrastive is at 2.07+ — same trade F110 found from the loss side, larger at the state level.

Reconciliation: F110's `/mean-z` 0.49 (contrastive arm) and F129's 1-6% perturbation effect are both correct but use different latents:

| action sensitivity, contrastive arm, h=1 | |
|---|---|
| `/mean-z`, projector latents (F110) | 0.49 |
| `/mean-z`, ITM latents (here) | 0.965 |
| 1sd perturbation of ITM latent | 1.370->1.366 |
| real ITM latent from another state | 1.370->1.425 |

The forward model is action-sensitive only inside the region the projector produces (stage 3 trains it jointly with the projector on `proj(a)` inputs only); ITM latents (what F129 fed) are outside that region.

Original (withdrawn) conclusion: neither arm has both action-conditioning and state fidelity, so no cheap fix exists and the objective needs a rollout-fidelity term and an action-conditioning term together. This is now overturned by F131's projector-path remeasurement.

Scope: one seed per arm, B1 clips, teacher-forced ITM actions, one-channel checkpoints. Projector-latent `/mean-z` quoted from F110, not re-measured at every horizon here.

Logs: `/tmp/f139_*.log`.

---

### F131. Measured on the projector path (the one a policy would drive), the contrastive arm already has both state fidelity and action-sensitivity

F130 measured action-sensitivity only exists in the projector's region, then judged state fidelity on the ITM's latents. `rollout_fidelity.py` gained `--latent projector`; result:

| B1 clips, one step | state fidelity | `/mean-z` across clips | `/mean-z` within a clip |
|---|---|---|---|
| MSE (`--lambda_nce 0`) | 0.585 | 0.969 | 0.975 |
| contrastive (`--lambda_nce 1`) | 0.710 | 0.476 | 0.951 |

(same contrastive checkpoint on the ITM path reads 1.370 / 0.965 — F130's numbers)

The contrastive arm beats holding still and is action-sensitive simultaneously, on the path that matters. Trade is mild: state fidelity 0.585->0.710 (~1/5) buys `/mean-z` 0.969->0.476. This already clears the bar the planned lambda sweep was meant to find (state under ~0.8 with meaningful action-sensitivity) at lambda 1, no sweep or re-pretrain needed.

`/mean-z` reconciliation: F110's 0.49 and F129/F130's 1-6% both correct, averaging different things — mean across the whole dataset (0.476, matches F110/wm/adapt3) vs mean within the clip being predicted (0.951).

Correction from F138: the within-clip 0.951 number is measured at fixed magnitude (periodic gait, frame fixes phase, action redundant by construction). Letting magnitude vary within the same behaviour family gives 0.485 (vs 0.476 across all behaviours) — the model is not blind within a behaviour, only where there is nothing to see. Quote 0.951 as a property of the task, not the model discarding information. Same shape as F102 (kind of motion transfers, amount doesn't), now visible in the forward model's own predictions — a real limit on fine control.

Consequences: teacher-student is not blocked by the objective (F130's verdict does not survive). What remains true from F129: horizon — contrastive arm degrades from 0.710 at h1 to worse-than-hold-still by h5, so imagined rollouts must stay short. The lambda sweep (`scripts/run/com7_lambda_sweep.sh`) becomes a refinement, not a decision, since lambda 1 already gives both properties.

Logs: `/tmp/f140_*.log`.

---

### F132. Teacher-student design, constrained by prior measurements (design only, not run)

Q16 committed to teacher-student distillation (2026-08-28): scoring candidates needs a target-robot policy that already performs behaviours; learning one needs only the ability to actuate. F131 removed the objection that the objective trades state fidelity against action-sensitivity.

Measured constraints shaping the design:

| constraint | measured | consequence |
|---|---|---|
| horizon | contrastive arm 0.710 at h1, worse than frozen by h5 (F129, F131) | imagine h<=3 only |
| magnitude | `/mean-z` 0.476 across behaviours, 0.951 within one (F131) | model reads which movement, not how much |
| coordinate | 3 channels calibrate on both robots, 70% cross-embodiment selection (F125, F127) | goal is a body-motion vector, not a frame/joint target |
| search | sampled joint commands never produce locomotion (F128) | policy must be fitted, not searched at run time |

Design: goal g is a body-motion vector (forward/lateral/yaw) read from video via shared head, cross-embodiment by construction. Teacher: from state e_t, roll FDM h<=3 steps per candidate action on proj(a), read `body_head(ITM(e_t, rolled))`, label = action nearest g. Student: `pi(e_t, g) -> joint targets`, trained on those labels on states the student itself reaches; one forward pass at run time, no library, no rollout. Trained on the insect (where the FDM is worth rolling); deployed on the B1 (per Q16's ordering).

Candidate set at training time is not a library — it's the student's own current output plus local exploration, ranked by the teacher; this escapes F128 (which searched once, blind, at run time) only if the student starts somewhere that moves.

Success criteria fixed in advance: primary — student produces the right behaviour family (forward/turning/strafing, correct sign) above chance, on a body with no candidate library, goal from another robot's video (magnitude limit permits only this). Secondary, expected to fail — achieved magnitude within a family tracks the goal's (F102: 0.074 correlation for cross-embodiment speed; F131 explains why — forward model barely distinguishes actions within a behaviour). Report correlation, expect near zero. Framed explicitly as a partial result (behaviour-type control, not full locomotion control).

Hole found before building: at initialization the student is random, so the teacher ranks random joint targets — exactly F128's failure mode. Bootstrap options considered:

| bootstrap | cost to claim |
|---|---|
| recorded clips of target robot | claim collapses to candidate scoring with extra steps |
| recorded clips of source robot (insect) | acceptable — insect data is the premise, not a concession |
| random exploration + survival/motion reward | honest but is RL; project's AMP branch was abandoned for poor gaits (PROGRESS.md sec13) |

Chosen: clone student on insect's recorded actions to bootstrap motion, then improve via world-model teacher with goals in shared coordinate on the target robot. Since action spaces are disjoint (insect 18-D vs B1 12-D), the clone cannot transfer directly across robots — only the goal (shared coordinate) crosses. The student must be bootstrapped per-robot, so the open question is where a target robot's first motion comes from if not a library or RL. Until answered, teacher-student on the B1 inherits F128's result; on the insect it is buildable today (clips exist, replay exact, F92). First experiment: teacher-student within the insect, testing the mechanism without claiming transfer.

Needs a simulator in the loop for the insect (CoppeliaSim, one instance, GUI, F88). Nothing measured yet; cite as design only.

---

### F133. Teacher-student on the insect (same-robot engine test): reference, bar, and cloning-control results, teacher stage pending

Engine test: same robot both sides, no transfer claimed. Question: can short imagined rollouts train a walking policy at all.

Step 1 — reference: `hexapod_ep100` (`speed_c7.1`) replayed through insect physics, 66 steps/3s. Recorded displacement 0.6454 m; replayed D_real = 0.6566 m (replay ratio 1.017, cf. F92's 1.06). Bar fixed at 50% of D_real = 0.3283 m.

Step 2 — pre-registered criterion (both required): upright throughout (head height never below 0.6 of settled value) AND >=0.3283 m travelled.

Cloning control/bootstrap: student cloned on insect's own forward-walking clips (F128 showed noise-init never walks). Held-out cloning error 0.065 after 2000 epochs. Result: travelled 0.2349 m = 36% of D_real; upright full window (min head height 0.1400 vs 0.1501 settled). Verdict: FAIL on distance (gait works, under-travels by 2/3 — not a collapse). This establishes cloning-alone fails, so any improvement from the teacher stage will be attributable.

Video: `results/wm/closed_loop/f142_video/f142_bc_vs_real.mp4` (recorded vs clone), `f142_bc_student.mp4`.

Teacher stage not yet built: insect-side forward model measured on projector path, held-out clips — state fidelity 0.757 (h1), 0.727 (h2), inside bar, but `/mean-z` across clips 0.966 — rollout is good but deaf to action, so it would rank noise. `scripts/run/com7_stage3_hexapod.sh` applies F110's contrastive term (as used for the quadruped, 0.476, F131) with a gate written in before running.

Logs: `/tmp/f142_*.log`; runs in `results/wm/closed_loop/f142_*`.

---

### F134. The insect teacher passes its gate on its own training body, fails on an unseen body (cross-morphology limit)

`scripts/run/com7_stage3_hexapod.sh` applied F110's contrastive term on the insect, 15,000 steps, 24/48 `c10f10t10` clips (other 24, including `hexapod_ep100` used by F133, held out). Gate fixed before the run.

On the trained body (F133's body):

| | before stage 3 | after |
|---|---|---|
| state fidelity, h=1 | 0.705 | 0.739 |
| `/mean-z` across clips | 0.955 | 0.583 |
| exact condition picked | 53% | 85-90% |
| behaviour family (chance 28%) | 84% | 95% |

Re-measured over all 48 clips (optimistic, 24 were training data): 0.592/0.715/0.796 at h1/2/3, `/mean-z` 0.534/0.578/0.621, wrong-latent cost rises from 3% to 64% at h1.

Gate passed: sensitivity gained, fidelity stays under 0.8 out to h3 (matching F132's horizon cap), same trade shape/size as B1 (F131: 0.585->0.710 buying 0.969->0.476) — F131's refutation of F130 holds on a second robot. F133's teacher stage can now be built.

On an untrained body (`c08f09t09` held out entirely):

| | before | after |
|---|---|---|
| state fidelity, h=1 | 0.757 | 1.052 (worse than a frozen frame) |
| `/mean-z` across clips | 0.966 | 0.774 |
| wrong latent costs | 3% | 19% |

The contrastive term buys action-sensitivity everywhere but the fidelity cost is small within the adapted body (5%) and total across an unadapted body — a cross-morphology limit (Stage 1's territory), distinct from the cross-embodiment result. A teacher meant to generalize to an unseen body needs something this adaptation doesn't provide; whether the lambda term can be tuned for both is the lambda sweep's open question (`scripts/run/com7_lambda_sweep.sh`), now with a reason to run it.

Checkpoint/run: `beh12_hex-b1_body3/stage3_hex_nce_s0.pt`, commit 3923cfa. Log in `results/wm/`.

---

### F135. The teacher-student engine test fails: the taught policy travels less far than cloning alone

Judged against F133's pre-registered bar, on `c10f10t10` (body F134 validated the teacher for). Goal/reference from `hexapod_ep100`, held out of both teacher training and cloning split.

| | distance | % of D_real (0.6566 m) | upright full 3s | verdict |
|---|---|---|---|---|
| cloning only (control) | 0.2349 m | 36% | yes | FAIL |
| teacher-student, 10 DAgger rounds | 0.2042 m | 31% | yes | FAIL |
| bar | 0.3283 m | 50% | required | |

Clone reproduced its earlier number exactly (36%, same simulator session) — not a session artefact. Pre-registered reading for 31%: the teacher adds nothing (made the policy slightly worse). Both arms stay upright throughout; failure is a gait that doesn't travel, not a collapse.

Training details: 10 episodes, 66 teacher labels each, 32 candidates/step ranked by rolling FDM 3 steps. Distance during training wandered 0.09-0.35 m with no trend; fitting loss fell 0.067->0.055 (student fit its labels; labels weren't worth fitting).

Video: `results/wm/closed_loop/f142_video/f144_real_clone_taught.mp4` (diagnosis only, does not overturn the numbers).

Hypothesis (not measured): candidates are Gaussian perturbations of the student's own action (0.5 sd/joint); if the teacher's ranking of these is near-arbitrary, DAgger trains on noise around itself, matching the observed small degradation. The settling measurement (execute labelled vs. student's own action from same state, compare body motion) was not run; F134's `/mean-z` 0.534 does not substitute for it (it shows the model separates a real action from an average one, not that it orders small perturbations correctly).

Scope: this teacher is validated on `c10f10t10` only — on `c08f09t09` its state fidelity is 1.052 (worse than a frozen frame, F134). Do not reuse off `c10f10t10`.

Conclusion: teacher-student as specified in F132 does not train a walking policy on the easiest designed case. Does not show distillation from world models cannot work in general — shows this teacher, ranking local perturbations at this horizon, produced no usable signal.

Runs: `results/wm/closed_loop/f144_*`; logs `/tmp/f144_*.log`; students in `wm/runs/students/`.

---

### F136. The teacher can rank behaviours but not perturbations within one — the mechanism behind F135's failure

Characterization of F135's failed engine test, using the same teacher and scoring rule the labeller used.

Local (perturbations F135's teacher actually ranked, judged in simulator): at 12 branch points on a held-out clip, student's own action and teacher's pick (of 32 Gaussian perturbations) each executed from the same state for 3 steps, body motion compared to goal.

- teacher's pick closer to goal: 4/12 = 33% (coin = 50%)
- teacher kept the student's own action: 0/12
- mean distance to goal: student 0.1299, teacher 0.1304

Note (read with F138): candidates are perturbations of one action at one magnitude, and physics barely separates them (0.1304 vs 0.1299) — partly a task property, not only a teacher failure. The teacher's ranking is worse than a coin and its labels are indistinguishable in outcome from the student's own — this is the mechanism of F135's 36%->31% degradation (DAgger trained on noise around itself).

Coarse (12 recorded conditions as candidates, same rule/states): pick shares goal's behaviour family in 55% of 120 states (chance 33%). Above chance, far below F134's 95% — different questions: F134 (via `adapt3`) is handed the true next embedding; this is handed only a goal body motion and must reach it (harder, more deployment-like).

Conclusion: the teacher is coarse (can distinguish behaviour families, 55% vs 33%) but not fine (cannot refine within a behaviour) — same wall as F102's 0.074 speed correlation and F131's within-clip `/mean-z` 0.951, now measured on the labelling path via physics. A scheme needing only behaviour-choice has signal; one needing within-behaviour refinement (gait improvement, F132's design) does not. Not a tuning failure — no amount of DAgger rounds/candidates/horizon fixes it, since the ordering isn't there to sharpen. F135's bar stands.

Runs/logs: `/tmp/f145_*.log`; `scripts/diagnostics/teacher_label_quality.py`.

---

### F137. A revised pretraining objective was pre-registered to fix Context Collapse — it was run and failed (see F141)

Plan (not run at time of writing, later executed — see status below): three terms to replace the current pretraining objective, targeting the fact that the forward model orders behaviours but not perturbations (F129, F136).

1. Rollout-level action-sensitivity hinge: penalize similarity between a rollout on real actions vs. a null action, accumulated over K steps (not one) since a one-step penalty can't see the failure (F129: 0.732 at h1 decaying to 0.978 at h10). Null = each embodiment's standing stance (per-embodiment, identically defined; F139 establishes the zero vector and dataset-mean pose both fail as nulls — one or both robots fall/drift).
2. A frozen, randomly-initialized, never-trained action-readout head (a NEW module, not the ITM) scoring rollout separation; sensitivity loss backprops through latents only, not the readout. Design decision: do not freeze the ITM (it produces the `z` the action projector is fitted to imitate; freezing it at random weights breaks the projector and every control-time path).
3. Keep the prediction loss (prediction + hinge + frozen readout together). F131 established the fidelity/sensitivity trade is a tunable knob, not a wall.

Pre-registered expectations: usable imagination horizon past h=3 is the primary target (clean, confirmed Context Collapse). Magnitude ranking is left open, pre-declared neither way (prediction already reacts to magnitude, `/mean-z` 0.485 across sizes, per F138, but ranks poorly — turning 39-64%, forward 41-62% vs 33% chance, F127). Action-sensitivity across disjoint embodiments is the open question/contribution — this is narrower than first read: corrected by F197, F134 never built or tested a cross-embodiment correspondence (`wm/adapt3` fits one embodiment's own condition labels; no second body enters that file). F134 shows single-body contrastive fine-tuning can cost a different body fidelity through weight interference, not that a cross-body Froude-correspondence InfoNCE was tried and failed.

Not addressed by this plan: where a target robot's first motion comes from without a library or RL (F128, F132); whether a repaired forward model would rescue candidate scoring (F126 showed scoring doesn't need a rollout at all).

Status: RUN, and it FAILED (F141). The pre-registered 50-epoch from-scratch rebuild on com7 diverged past the frozen-frame baseline by 4x starting at horizon 2 — worse than the pre-registered failure case. A lag-3 frameskip follow-up also failed. Mechanism: a one-step prediction loss cannot anchor a multi-step hinge, so steps 2+ are unopposed and diverge. F141's reading supersedes this entry: the limit is the representation, not the objective — no frameskip reaches it; next move is the encoder or prediction target, not another objective term.

---

### F138. Within-behaviour-at-fixed-magnitude blindness is a task property, not a model failure — corrects F131 and F136's framing

Slide 11 (task property): at one speed the gait is periodic, a single frame fixes the phase; removing the transition costs only 28-34%, a second frame is worth 1.11x, predicting 32 frames ahead is as accurate as predicting the present. This redundancy should not be removed by any objective. F129/F131/F136 measured a separate model failure, but the number conflating them (`/mean-z` 0.951 within a clip) was measured at fixed magnitude, where the task itself is redundant.

Separating measurement — third baseline, mean latent of the same behaviour family at other magnitudes:

| `/mean-z`, h=1, projector path | within one clip | within the family | across all behaviours |
|---|---|---|---|
| B1, contrastive stage 3 | 0.951 | 0.485 | 0.476 |
| insect, contrastive stage 3 | 0.697 | 0.597 | 0.534 |

On the B1: holding speed fixed, action changes prediction by only 5%; letting speed vary, changes it by >50% (0.485, close to across-everything 0.476). The model is not blind within a behaviour — only blind within a behaviour at fixed magnitude, where there is nothing to see.

Corrections: F131's within-clip 0.951 is not evidence of collapse — it's Slide 11's task property measured on the forward model; should be quoted as "action near-redundant at fixed magnitude," not "model ignores action within a behaviour." F136's local-ranking failure (33% vs coin's 50%) is at least partly the same effect — physics itself barely distinguished the candidates (teacher's pick vs student's own: 0.1304 vs 0.1299 mean distance to goal); the world, not just the model, didn't separate those actions.

What is unchanged: Context Collapse is still a real model failure — on the pretrain, held-out insect clips, rolled state goes 0.732 (h1) to 0.978 (h10), full-sd action perturbation changes the answer by only 1% (F129), at every magnitude, horizon decay not task redundancy. The ActSWM rebuild (F137) still targets this.

Refined three-level picture:

| question | answer |
|---|---|
| within one magnitude: does action matter at all | no, correctly so (task property) |
| across magnitudes: does prediction react | yes, 0.485 — not collapsed |
| across magnitudes: does it rank correctly | not well, open — F127's weakest families (turning, forward) |

F137 should be judged primarily on horizon (0.732->0.978), not magnitude ranking, which is open and not pre-declared either way. Whether a model could ever rank actions within a behaviour at fixed magnitude remains open; 0.1304 vs 0.1299 (12 states, one robot) suggests the outcome genuinely may not depend on them — a control-precision limit no objective fixes.

Logs: `/tmp/f147_{hex,b1}.log`; `rollout_fidelity.py --family_mean`.

---

### F139. The correct "null action" for the sensitivity hinge is each embodiment's standing stance; zero vector and dataset-mean pose both fail

Prerequisite for F137's hinge, which contrasts a rollout on real actions vs. a null action — if the null causes a fall, the hinge just separates walking from falling, teaching nothing about the action channel. Four candidates tested, held constant 3s, both robots:

B1 (MuJoCo):
| candidate | travel | min height | verdict |
|---|---|---|---|
| settled pose | 0.008 m | 0.462 | still |
| standing stance (clip start) | 0.008 m | 0.502 | still |
| dataset-mean pose | 0.628 m | 0.126 | FALLS |
| zero vector | 1.077 m | 0.088 | FALLS (lateral speed -0.104, faster than any recorded strafe) |

Hexapod (CoppeliaSim):
| candidate | travel | min height | verdict |
|---|---|---|---|
| settled pose | 0.0002 m | 0.151 | still |
| standing stance | 0.0002 m | 0.151 | still |
| dataset-mean pose | 0.063 m | 0.150 | drifts (150x the stance) |
| zero vector | 0.034 m | 0.019 | FALLS |

Zero vector fails on both robots because these action spaces are joint targets, not torques — zero commands every joint to angle zero, an unsupportable posture (B1 collapses/slides 1.08m; insect folds to 1/5 standing height). Dataset-mean pose also fails: averaging over a gait cycle (swing vs stance) produces a posture in no real frame that holds nothing up (consistent with F128's finding that sampling near the mean pose at 0.1 sd sent the quadruped backwards while rotating).

Chosen null for F137: the standing stance of each embodiment (the pose its clips start in and settle into) — per-embodiment by necessity (18-D vs 12-D vectors) but identically defined on both (the pose that body stands still in), which is what a cross-embodiment sensitivity comparison requires (otherwise it would measure the difference between nulls rather than between models). On the insect the two stance variants coincide by construction (warm-up holds the clip's first command).

Note: this null is a pose (actively held), not absence of actuation — the correct contrast for a joint-target action space is "commanded to hold still" vs "commanded to move," not "actuated" vs "unactuated."

Script: `scripts/diagnostics/null_action.py`; log `/tmp/f148.log`.

---

### F140. On the three-channel pretrain, reconstruction is 96-99% of loss but only 22-41% of gradient into the latent

Pre-rebuild baseline for F137. `loss_gradient_balance` now uses checkpoint-stored statistics pooled across both embodiments (correcting F77's over-standardization on an insect-only batch). Re-measured on `beh12_hex-b1_body3/best.pt`, `body_dim 3`, 24 transitions, gradient of each term taken w.r.t. the same `z`:

Hexapod:
| term | lambda | loss | share of loss | share of gradient |
|---|---|---|---|---|
| recon | 1.00 | 1.5266 | 98.8% | 40.6% |
| motion | 1.00 | 0.0046 | 0.3% | 22.8% |
| body | 0.50 | 0.0266 | 0.9% | 36.6% |

B1:
| term | lambda | loss | share of loss | share of gradient |
|---|---|---|---|---|
| recon | 1.00 | 1.3131 | 95.6% | 22.3% |
| motion | 1.00 | 0.0304 | 2.2% | 46.3% |
| body | 0.50 | 0.0589 | 2.1% | 31.4% |

Reconstruction is 96-99% of loss but only 22-41% of gradient — loss share was never the right thing to read; F20's "99% of gradient goes to reconstruction" was inferred from loss magnitudes, not measured, and should not be quoted. The body term has the largest raw `|dL/dz|` on both robots (0.0052, 0.0079 vs reconstruction's 0.0029, 0.0028) despite being halved by `lambda_body 0.5` — the smallest-weighted term pulls hardest on the latent.

Compared to F77's one-channel run (recon ~13%, motion ~37%, body ~50% of gradient after rescaling), the three-channel pretrain is more balanced (recon share up to 22-41%, body share down to 31-37%) — widening the shared target spread the gradient rather than concentrating it.

Purpose: this table is the baseline the ActSWM rebuild (F137) must be measured against — the claim is that gradient into the latent increases via the new term, not that the loss curve moves. Also rules out lowering `lambda_recon` as an intervention (same as F77 found on a different checkpoint) since reconstruction was not dominating gradient to begin with.

Script: `scripts/diagnostics/loss_gradient_balance.py`; logs `/tmp/f149_{hex,b1}.log`. Batch 24, one robot at a time (two concurrent runs OOM an 11GB card).

---

### F141. ActSWM rebuild arc: wiring checks pass, but the full rebuild fails its pre-registered criterion; frameskip does not fix it

Merges the original chain of F141 sub-entries (chronological). F142 and F143 are separate entries, not folded here.

**Wiring checks** (`scripts/diagnostics/check_actswm_wiring.py`, no training): (1) null-action contrast (standing stance vs real action) is real but small: hexapod 0.078→0.236, B1 0.089→0.146 (h=1-5). (2) Frozen readout (`[e_t,e_t+1]->action`, 725,778 params, `requires_grad=False`) correctly passes 0 gradient to itself, 312 tensors / norm 27.56 to the forward model. (3) Starting action-sensitivity (real vs null rolled error): hexapod 0.956→0.863, B1 0.936→0.925 (h=1-5) — action buys only 4-14%.

**Short calibration runs** (400 steps, continuing the 3-channel checkpoint, `wm/actswm_short.py`): found pretraining needs a different null than F139's stance null, since pretraining has no action projector yet. Correct null: `ITM(e_t, e_t)` ("latent of nothing happened"); F139's stance null still applies post-projector. With margin 0.3, K=5: hinge gradient enters but prediction degrades toward frozen-frame baseline (hexapod 0.658→0.798, B1 0.714→0.904) and separation is unstable (overshoots 0.3, collapses to 0.008). With margin 0.1, K=3 (two variants, alpha_pred 1.0 vs 3.0): separation now rises and holds (no collapse), but prediction still slips in both (less with alpha_pred 3.0). Verdict: margin 0.1 fixes the instability but doesn't fully pass all 3 acceptance criteria (hinge alive, separation stable, prediction not degrading).

**Pre-registered full rebuild settings**: margin 0.1, K=3, H=1, `lambda_recon=3.0`, `lambda_hinge=0.5`, `lambda_readout=1.0`, null=`ITM(e_t,e_t)`. Script: `scripts/run/com7_pretrain_actswm.sh`. Two reading rules fixed in advance: read per body, never pooled (insect sensitivity improves with horizon, B1's is flat and degrades fastest in every short run); read `/mean-z` only against the null it was trained on (pretraining null reads ~1.0, projector-stage null reads 0.86-0.96 — the two are not interchangeable as a baseline).

**Result: rebuild fails.** 50 epochs from scratch, both embodiments, com7, commit 5306889, run `wm/runs/beh12_actswm/`. Rollout prediction vs frozen frame: divergence horizon collapsed from >10 (old pretrain) to **2** on both bodies; by step 5, hexapod 3.139x worse than frozen frame, B1 4.007x worse (vs old pretrain 0.764 at h=5). Step 1 alone stays fine (0.743/0.707, matches old 0.732) because it's anchored by `L_recon`; steps 2-3 have no prediction anchor, only the hinge pushing away from null, so the rollout diverges rather than collapsing. Conclusion: **need a multi-step prediction anchor over the same K the hinge spans**, not a smaller lambda_hinge — this is the finding, not a hyperparameter to retune.

What survived: shared body-head coordinate unaffected (hexapod/B1 all channels +0.95 to +0.99 correlation) — F125's result stands on this checkpoint regardless of rollout failure. `/mean-z` cannot be read from this run (measuring ratio of two broken quantities). Monitoring gap: run had no separation-curve logging (print added only after commit 5306889); must commit the monitoring print before any future run.

**Lag-3 frameskip follow-up (also negative).** F143 found lag 3 the best target spacing but only as an off-distribution lower bound (both ITM and FTM fitted at lag 1). Tested training directly at lag 3, no hinge: `scripts/run/com7_pretrain_lag3.sh`, run `wm/runs/beh12_lag3_nohinge`, `frame_stride=3`, `action_chunk=0`, `lambda_hinge=0`, `lambda_readout=0`, `lambda_recon=1.0`, 10 epochs, commit 9fecfb3. Result: `null/real` at lag 3, trained at lag 3 = **1.032** (down from F143's off-distribution 1.078; real beats null 82.9% vs 94.6%). Comparing each model at its own lag: lag-1 model 1.028 (70.2%), lag-3 model 1.032 (82.9%) — action worth 2.8% at lag 1 vs 3.2% at lag 3, same ~30% motion-error reduction either way. Conclusion: **frameskip does not create the missing signal; F143's lag sweep was measuring model mismatch (off-distribution penalty), not task structure** — same artifact reproduces with sign reversed when the lag-3 model is tested off its own distribution. Limit is the representation, not the objective; no hinge, no full pretrain, no lambda tuning follows from this. Body-head coordinate again survived (probe 0.974, body 0.1436 at 10 epochs).

Alternative explanation considered and weighed against: this run used only 10 epochs vs. the lag-1 model's 50, so action-sensitivity could in principle still be growing — but the run's own losses had largely flattened by epoch 10 (val 2.7527→2.7373) and the 50-epoch ActSWM run above did not produce action-sensitivity either, so more epochs was not treated as the likely fix.

Scripts: `scripts/diagnostics/check_actswm_wiring.py`, `wm/actswm_short.py`, `scripts/run/com7_pretrain_actswm.sh`, `scripts/run/com7_pretrain_lag3.sh`. Runs: `wm/runs/beh12_actswm/`, `wm/runs/beh12_lag3_nohinge`.

---

### F142. One-step prediction does not need the action; no lambda weighting can fix the ActSWM hinge

Tested whether F141's failure was a lambda imbalance or whether one-step prediction is simply solvable without reading `z`. `scripts/diagnostics/action_necessity.py` swaps the forward-model drive at a single step: real latent, null `ITM(e_t,e_t)`, a shuffled real action from elsewhere in the same clip, the clip mean, and no motion.

Key metric `null/real` (both rows are checkpoint `beh12_hex-b1_body3`, pre-rebuild, the only one locally available):
- insect, all: real 1.5511, null 1.5949, null/real **1.028**, real beats null 70.2%; turning family worst at 1.010 (51.4%, coin flip)
- B1, all: real 1.3645, null 1.3994, null/real **1.026**, real beats null 82.3%

Knowing the true action is worth under 3% on both robots; an unrelated action from the same clip costs only 4-7% more; predicting no motion at all costs 36-42% — the model predicts mostly from state, not action.

Conclusion: there is no lambda balance point between the tested settings (1.0 vs 3.0 in F141) — at one step the real and null rollouts are already nearly identical, so any separation the hinge creates is manufactured where the prediction loss cannot see it (matches F141's h>=2 divergence). The objective's target must change (wider lag / multi-step target) before any hinge weight is meaningful. Recommends measuring `null/real` on any candidate change before training with a hinge.

Scope: measured only on pre-rebuild checkpoint `beh12_hex-b1_body3`; `beh12_actswm` (the rebuild, on com7) not measured this way as of writing — `scripts/run/com7_action_necessity.sh` runs both there.

---

### F143. Lag 3 is the best target spacing on both robots, but the effect is real and small (later shown to be partly an off-distribution artifact, see F141)

Swept target lag `k` (action = `ITM(e_t,e_t+k)`, target = `e_t+k`, forward model applied once) on checkpoint `beh12_hex-b1_body3` (pre-rebuild), held-out body for insect.

| lag | null/real, insect | real beats null | real/hold | null/real, B1 | real beats null | real/hold |
|---|---|---|---|---|---|---|
| 1 | 1.028 | 70.2% | 0.737 | 1.026 | 82.3% | 0.702 |
| 2 | 1.066 | 93.1% | 0.677 | 1.043 | 91.1% | 0.666 |
| **3** | **1.078** | **94.6%** | 0.676 | **1.053** | **93.4%** | 0.655 |
| 5 | 1.072 | 91.5% | 0.693 | 1.049 | 93.1% | 0.661 |

Lag 3 peaks on both robots; prediction never breaks down (`real/hold` stays 0.65-0.74 at every lag — no divergence-horizon issue in this one-shot measurement). The sign becomes far more reliable: insect turning goes from 51.4% (coin flip, F142) to 92.9% at lag 3. But magnitude stays small — best null/real is 1.078, i.e. action still explains <8% (insect) / <6% (B1) of prediction error; `real/hold` is flat across lags, so widening the gap scales all quantities proportionally rather than increasing action's share — consistent with a representation-level ceiling.

Caveat (confirmed correct by F141's follow-up): both ITM and FTM were fitted at lag 1, so this sweep is off-distribution for lag>1 and is only a lower/upper bound, not a trained result. F141's direct lag-3 training run later showed the 1.078 value was inflated by this off-distribution effect (trained-at-lag-3 null/real measured only 1.032).

Scripts: `scripts/diagnostics/` lag sweep tool referenced as extended by `scripts/run/com7_action_necessity.sh` (sweeps lags on `beh12_actswm` too).

---

### F144. The residual left by action-blind prediction is mostly not the missing action, on either robot

Tested whether `r = e_t+k - FTM(e_t, ITM(e_t,e_t))` (the action-blind prediction residual) is action-structured signal or noise. `scripts/diagnostics/residual_structure.py`, checkpoint `beh12_hex-b1_body3` at lag 1, ridge in the dual on the full 360,448-dim embedding, split by clip.

| | action R2, insect | action R2, B1 |
|---|---|---|
| `r` (null residual) | 0.786 | 0.274 |
| `e_t` alone (control) | 0.777 | 0.161 |
| `e_t+k - e_t` (raw diff) | 0.641 | 0.063 |

Insect: `r` adds ~0.009 R2 over the bare frame (essentially nothing) — worse than the frame on speed/turning families, better only on sideways. B1: `r` (0.274) exceeds the frame (0.161), notably on turning (-0.166 → 0.114), but 0.27 R2 is too small to build a direction on.

Pair test (matched-action nearest-neighbour pairs vs random pairs, ratio of residual distance): insect 0.707, B1 0.937 — B1 residuals barely differ whether the action matches or not, meaning `r` is not action-determined; insect ratio less extreme but test is conservative (nearest-neighbour in continuous action space biases ratio toward 1).

Caveats: family-accuracy separation (`r`: insect 1.000/B1 0.869) is not real evidence since `e_t` alone gets 1.000/0.893 — the frame alone identifies behavior/clip identity. `r` contains `e_t` implicitly since FTM is a function of `e_t`, so the control row (not `r` alone) is the real measurement.

Conclusion: predicting `r` directly is not a viable next direction — it reproduces the same problem (residual is 94% not-action-determined on B1, +1% over raw frame on insect). Converges with F141/F143: V-JEPA2 embeddings of this scene are dominated by what does not move; no added term reaches around that. Measured on lag-1 checkpoint only; `beh12_lag3_nohinge` (com7) not yet re-run with `--lag 3`.

---

### F145. A single frame recovers most of the insect's commanded action but not the B1's

Compared against Yeom et al. (V-JEPA inverse-dynamics R2: 0.40 frozen, 0.85 with ID head, on CALVIN's static tabletop). Ridge in the dual on full 360,448-dim embedding, split by clip, `beh12_c08f09t09_flat` held out for insect.

| features | insect | B1 |
|---|---|---|
| `e_t`, one frame | 0.779 | 0.161 |
| `[e_t, e_t+1]`, pair | 0.867 (+0.088) | 0.342 (+0.182) |
| `[e_t, e_t+3]`, wider pair | 0.887 (+0.108) | 0.328 (+0.167) |

Insect: one frame recovers 88% of what a pair recovers (turning: 0.931 single vs 0.957 pair — pose is nearly the whole command). B1: one frame recovers only 47% of the pair's value (turning: -0.166 single, worse than predicting the mean, only 0.120 even with a pair).

Interpretation: inverse-recoverable does not imply forward-necessary — on the insect the action is recoverable at R2 0.887 yet contributes <3% of one-step forward prediction error (F142). The "pose already says the command" framing is insect-specific; B1 fails differently (not much recoverable at all, not a periodicity story). `direction_plan.md`'s contribution-statement sentence has been scoped to the insect to avoid overclaiming.

Cross-paper comparison (our 0.779 vs their 0.40 frozen) is not controlled — different data/action space/head/split — worth one framing sentence only, not a claim of "twice as easy."

Scope: measurement is nearly checkpoint-independent (frozen encoder embeddings; only action normalization is checkpoint-config-dependent). Confirming on `beh12_lag3_nohinge` at its own stride (pair_lags 3) not yet run.
Scripts: `scripts/diagnostics/inverse_dynamics_r2.py --ckpt wm/runs/beh12_lag3_nohinge/best.pt --data data/allocentric/beh12_c08f09t09_flat --embodiment hexapod --pair_lags 3`

---

### F146. The lambda_body=0 control: the body-coordinate objective did not cause the action-insensitivity, and it also makes `z` body-identifiable

Question: did the body-coordinate objective (`lambda_body`) cause the action-insensitivity found in F141-145, by squeezing joint-level detail out of `z`? Note this is narrower than F145's encoder-level claim, which `lambda_body` never touches.

Setup: `scripts/run/com7_lambda_body0_control.sh`, run `wm/runs/beh12_lag3_nobody`, held identical to `beh12_lag3_nohinge` (frame_stride 3, action_chunk 0, lambda_hinge/readout 0, lambda_recon 1.0, 10 epochs) except `lambda_body=0` while `body_dim`/`body_channels` are kept (architecture unchanged, only the gradient removed). Measurement path pinned in advance: `action_necessity.py` never touches body_head/MotionDecoder; null=`ITM(e_t,e_t)`; lag=3.

Result (commit 3d86ebc): removing the body term made the action matter *less*, not more.

| at lag 3 | insect null/real | real beats null | B1 null/real | real beats null |
|---|---|---|---|---|
| baseline, lambda_body 0.5 | 1.032 | 82.9% | 1.009 | 86.1% |
| control, lambda_body 0 | 1.015 | 74.4% | 1.008 | 83.2% |

Conclusion: the coordinate objective was a small positive contributor to action-sensitivity, not a suppressor — the cause of the insensitivity is upstream in the encoder (F145), not this objective. Also completes F141's B1 lag-3 baseline table (was missing): B1 null/real goes from 1.026 (lag1) to 1.009 (lag3, trained there) — training at wider spacing made the action matter less on B1 too, reinforcing F141's "frameskip doesn't help" conclusion.

Morphology probe result: body identity decodable from `z` at 0.974 with the body term, 0.732 without (chance=0.5) — `z` is substantially body-identifiable. Validation motion also slightly worse without the term (0.6134 vs 0.5360).

**Scoping correction (applied 2026-08-31, not a retraction):** `z` is not body-blind — every document claiming "morphology-agnostic z" was wrong and has been corrected. The agnosticism lives in the shared body-motion coordinate (forward/lateral/yaw, F127), not in `z` itself; wording is now "`z` maps to a shared body-motion coordinate." Changed in `doc/direction_plan.md` (3 places), `doc/START_HERE.md`, `report/presentation_proposal.md` (3 places). `report/update_slide.md` needed no change. F97's related claim is superseded by this section; earlier findings left as-written.

Note: `probe` = `MorphProbe(z)` classifying which body (not the 3-channel body-motion coordinate). The lambda_body=0 control has no trained body head, so no body-motion coordinate number exists for it.

---

### F147. Novelty positioning against three named neighbours: the scoped contribution claim

Canonical scoped contribution (narrowed 2026-08-31):

> In periodic visual locomotion, the joint action is inverse-recoverable from a SINGLE frame (F145: insect R2 0.78, turning 0.93) because gait phase makes the pose encode the command — so the action is forward-redundant (F144: <3% of prediction error). This closes the loop between two prior observations — adjacent-frame redundancy (AHA-WAM) and the action-invariant teacher-forcing solution (UWM-JEPA) — with a measured mechanism, and shows the objective-level fix-family fails on it (ActSWM hinge, F141) and that the residual-target route is closed (F144) — because there is no action-dependent forward signal to recover, only a redundant one.

The unscoped version ("visual world models cannot be action-conditioned in locomotion") must never be written — AHA-WAM and UWM-JEPA both state parts of it already.

Positioning vs neighbours:
- AHA-WAM (2606.09811): assumes adjacent-frame redundancy as design premise; we measure it and attribute it to gait periodicity.
- UWM-JEPA (2605.25313, Sec 4): names the action-invariant-solution structure behind F141 (teacher-forced targets admit an action-invariant solution); their fix is counterfactual targets. We show this action-invariant solution is near-optimal in locomotion because the pose already encodes the command (F145) — a target fix has nothing better to converge to.
- Yeom et al. (2606.07687): shows V-JEPA carries inverse-recoverable action signal, with CALVIN's static scene letting per-frame appearance substitute for temporal context. We show inverse-recoverable is not forward-necessary (measured); periodicity is the severe structural form of their CALVIN exception.

This is a measurement-and-mechanism contribution positioned between named neighbours, not a discovery of the phenomenon.

**Correction note:** the target-level clause was narrowed from an earlier version that said "target-level (counterfactual/residual targets, F144)" — overstated, since F144 only tested the residual of the action that was taken, not a genuine counterfactual target (UWM-JEPA's actual fix, constructed for an action not taken). Narrowing chosen over running the counterfactual arm; if that arm is ever trained, this sentence widens.

Measured (both bodies): action contributes <3% of one-step prediction error (F142); no weighting recovers it (F141); no frameskip creates it (F141); action-blind residual doesn't carry it (F144); insect single frame R2 0.779 vs pair 0.887 (F145). Not measured: a counterfactual-target arm, or a non-V-JEPA2 encoder.

---

### F148. Direction B (motion representation) killed quickly: temporal differencing does not break redundancy and destroys cross-body transfer

De-risking test before any encoder rebuild: cheapest possible motion representation, `m_t = e_t+1 - e_t` (no training, no new encoder). `scripts/diagnostics/motion_rep_check.py --ckpt wm/runs/beh12_hex-b1_body3/best.pt`. Two pre-registered failure conditions both fired.

Part 1 — redundancy does not break (action R2, split by clip, held-out insect body):

| body | family | `e_t` appearance | `m_t` motion | `[m_t,m_t+1]` pair |
|---|---|---|---|---|
| insect | all | 0.779 | 0.646 | 0.650 |
| insect | turning | 0.931 | 0.783 | 0.826 |
| B1 | all | 0.161 | 0.063 | 0.100 |

Insect command still reads at R2 0.646 from a single motion snapshot (drop of only 0.133 from appearance); adding a second snapshot buys only +0.004. Differencing does not remove pose information — an appearance-organized space differenced is still appearance-organized.

Part 2 — transfer destroyed (body-motion coordinate fitted on insect, applied unrefitted to B1, raw ridge from frozen embeddings — not F127's trained body-head numbers, not comparable to them):

| representation | test | forward | lateral | yaw |
|---|---|---|---|---|
| `e_t` appearance | insect held-out | 0.98 | 0.95 | 0.96 |
| `e_t` appearance | B1 unrefitted | 0.63 | 0.43 | 0.07 |
| `m_t` motion | insect held-out | 0.60 | 0.52 | 0.84 |
| `m_t` motion | B1 unrefitted | -0.05 | 0.07 | 0.00 |

Differencing halves the within-insect coordinate and annihilates cross-body transfer (forward transfer 0.63 → -0.05).

Decision: Direction B killed on the cheapest candidate, failing both criteria (action-necessity, body-transfer). Does not prove no encoder could work (a learned motion encoder differs from a difference of an appearance encoder), but the cheap evidence points against it — proceed with Direction A (write up what's measured) instead.

---

### F149. The single-frame action redundancy is a data property (steady gait), not an encoder property — confirmed by collecting off-rhythm data

Hypothesis: on a steady gait the pose fixes the phase and the phase predicts the next frame, so the action may be redundant by rhythm rather than by anything about V-JEPA2. Tests the data, not the representation (unlike F148).

Built `--cmd_noise` (`--noise_tau`, `--noise_seed`) in `sim/collect/collect_ik.py`: temporally correlated noise (tau=5 steps) added to the final joint command after heading/oscillator branches, before `actions.append`, so the logged `a_t` is the actual perturbed command sent (white noise would be filtered by the joint controller and never reach the pose). `scripts/dataset/collect_offrhythm.sh` collects clean (cmd_noise=0) and noisy (cmd_noise=0.05) arms in one sitting via `collect_beh12.py` (avoids cross-sitting confounds).

Pre-registered reading: single-frame R2 drops AND pair-minus-single gap opens → redundancy is a data property (Direction B = train on exploratory data, cheap). Stays high → encoder-level, only route left is encoder rebuild.

Free first-pass check on already-collected speed ramps (`data/allocentric/fwd_hex7speed`, 24 ramped/24 constant, matched): pair-minus-single gap is +0.024 in BOTH arms — a speed ramp (retimes the whole foot path, preserves inter-leg phase per F53, lag 0.056 vs 0.061) does not move the number. Lowers prior on data hypothesis but doesn't settle it, since ramping varies rhythm rate rather than breaking it.

Main result (cmd_noise 0 vs 0.05, c10f10t10, 12 conditions, 2 repeats, 24 clips each):

| | clean (0) | noisy (0.05) |
|---|---|---|
| `e_t` single frame | 0.764 | 0.196 |
| `[e_t,e_t+1]` pair | 0.839 | 0.351 |
| `[e_t,e_t+3]` pair | 0.848 | 0.369 |
| pair - single | +0.084 | +0.173 |

Both pre-registered conditions fired: single-frame R2 collapses (0.764→0.196) and gap doubles (+0.084→+0.173) — opposite of the speed ramp result, matching F53's distinction (noise breaks inter-leg phase, ramp doesn't). Sideways family: 0.763→-0.047 single frame (worse than mean), pair recovers to 0.305.

Caveat: injected noise is exogenous/random by construction — no single frame could predict it regardless, so part of the gap-opening is tautological; this establishes the mechanism is data-side, not that a world model trained on it would be useful (the readable component is noise).

**But 0.05 rad is too high to train on**: separability check fails on noisy arm (24/66 condition pairs closer than 2x their own spread, closest pair 0.3x) vs clean (0/66, closest 2.4x) — the twelve conditions are no longer distinguishable behaviors at this noise level. Next step is a noise-level sweep, not a rebuild (see F150).

Bugs fixed in passing: `separability()` crashed (referenced unpassed `args`) so clean-arm results were invisible; `collect_offrhythm.sh` called it with wrong argument form; `wm/adapt3.gather` gained a `condition` fallback to `behavior` for Stage-1 `fwd_*` sets that never went through 12-condition flattening.

---

### F150. Noise-level sweep: 0.02 rad is the best trade-off between breaking action-redundancy and preserving separability

F149 confirmed the redundancy is data-side but 0.05 rad noise destroyed dataset separability. `scripts/dataset/noise_sweep.sh` tested levels 0.0, 0.02, 0.03 (correlated, tau=5), plus F149's 0.05 as reference, all collected 2026-08-31. Design: separability is checked first and gates whether the R2 gap number means anything (a level that dissolves the 12 behaviors has "collected one noisy condition twelve times").

Result:

| cmd_noise | pairs < 2x spread | semantic check | single-frame R2 | pair R2 | gap |
|---|---|---|---|---|---|
| 0.0 | 1 of 66 | FAILS (speed_c8.8 < speed_c8.15) | 0.729 | 0.832 | +0.102 |
| **0.02** | 11 of 66 | **passes** | **0.383** | 0.581 | **+0.198** |
| 0.03 | 19 of 66 | FAILS (side_R_lvl0 wrong sign) | 0.344 | 0.516 | +0.172 |
| 0.05 (F149) | 24 of 66 | FAILS | 0.196 | 0.369 | +0.173 |

0.02 is best on every pre-registered axis: only level that passes the semantic check (even the clean arm fails it — a 5% inversion exists with zero noise, so this body's recipe is marginal regardless), and has the largest gap in the sweep (+0.198) while single-frame R2 halves (0.729→0.383). Beyond 0.02 the gap stops growing while separability keeps degrading (trade-off has an optimum, not monotone).

Honest caveat: 0.02 is not a clean pass — unresolved condition pairs go from 1/66 to 11/66 (tenfold degradation), closest turn levels at 0.9x combined spread. The separability check was designed for planner candidate-library use, not pretraining data; for pretraining, varied-but-imperfectly-separable data is likely acceptable (argument for using 0.02 as pretraining data, not as a candidate library).

Clean arm reproducibility check: F149's clean arm read 0.764/+0.084; this sitting's clean arm reads 0.729/+0.102 — sitting-to-sitting variance is ~0.035 (R2) / ~0.018 (gap), so 0.02's +0.198 (vs null ~+0.09) is roughly double and well outside noise.

**What this confirms:** mechanism is data-side and breakable while behaviors remain individually correct in direction/ordering — Direction B's cheap form (perturbed collection, not encoder rebuild) is alive.

**What it does not confirm, must stay in any writeup verbatim:** that a world model trained on this would be useful. The noise is random by construction, so a model learns to read jitter, not intent; the gap-opening is partly tautological (only the transition can carry an exogenous, unpredictable component). Write up as "phase-breaking is possible without destroying the behaviors," never as "the world model can now be action-conditioned."

Standing next question (not yet measured): does *meaningful* phase-breaking motion (real turns/transitions/speed-breaks carrying intent, not jitter) produce the same gap-opening? This sweep is only the cheap proxy that it's possible in principle.

Data: `data/allocentric/beh12_c10f10t10_sweepn{00,002,003}_flat`. No log kept.

---

### F151. Meaningful phase-breaking collection: VOID — `--schedule` was silently discarded by `--gait cpg`, only 4/12 conditions were actually perturbed

Goal: test whether *meaningful* (non-random, intent-carrying) phase-breaking motion opens the single-frame action-redundancy gap, unlike F149's noise (jitter, useless) or F149's speed ramp (phase-preserving, no effect). Built `--spin_schedule` in `sim/collect/collect_ik.py` (turn-onset scheduling, `value@fraction` grammar) and used existing `--schedule` for speed/side conditions. `scripts/dataset/collect_intent.sh`: 12 conditions (4 speed, 4 turn, 4 side), 24 clips.

**Result: VOID.** `--gait cpg` silently discards `--schedule` — `schedule_path` retimes the recorded foot path, but the CPG branch only keeps `cmds.mean(0)` as a bias pose and regenerates the stroke from `--cycles`, so the 8 `speed_*`/`side_*` conditions had no within-clip change at all, despite command lines and logs saying they did. Only `--spin_schedule` worked (applied inside `cpg_commands`) — turn conditions (4/12, 8 clips) were genuinely intentful (within-clip yaw sd roughly doubled vs clean: 0.030-0.035 vs 0.013-0.020); speed/side conditions showed no change (sd 0.036-0.038 vs clean's 0.033-0.047).

Numbers as measured (not to be used as the answer — collection was broken):

| target | intent arm gap | clean arm gap |
|---|---|---|
| `a_t` instantaneous | +0.103 | +0.102 |
| `da` command change | +0.011 | +0.017 |
| family label | +0.005 | +0.001 |

Nothing moved (expected, since 2/3 of conditions were unperturbed); the turn-only subset showed `da` gap +0.020 vs clean +0.031, but n=8 clips is too few to conclude anything.

**Fix**: `cpg_commands` now takes a per-frame `pace`, phase advances via `cumsum(rate) - rate` instead of `arange(frames)` — verified bit-identical to old behavior at rate=1, and verified freezing phase correctly on a stop schedule. Re-run is F152.

**Durable lesson**: a flag accepted, echoed to the log, and then silently ignored is worse than one that errors — neither the separability gate nor the R2 tables caught this; only checking within-clip variance against the clean arm before write-up caught it. `--schedule` removed from CPG condition lines going forward.

---

### F152. Meaningful phase-breaking, re-run correctly: the action-redundancy gap does NOT open — only exogenous noise (F150) opens it, not intent-carrying motion

Re-run of F151 (void) with the collector fix (`cpg_commands` per-frame `pace` via `cumsum(rate)-rate`) and a new required physical gate, `scripts/dataset/check_within_clip_intent.py`: measures sd of the *smoothed* (stride-averaged) channel per family (speed→forward, turn→yaw, side→lateral) against the clean arm's own max on that channel — catches exactly what F151 missed. Validated against F151's void set: correctly fails the 8 broken `--schedule` conditions (ratio 0.50-0.90) and passes the 4 working `--spin_schedule` ones (ratio 2.13-2.37). Now runs before measurement under `set -e` in `collect_intent.sh`.

Gate also caught two more real issues before any number was read: (1) uncommanded start/end transients (every clip accelerates from rest, slows at end) were inflating the "clean" envelope — fixed by trimming 15% off each end, applied identically to both arms; (2) `side_L_stopmid`/`side_R_stopmid` genuinely fail the corrected gate (0.34x, 1.21x) because sideways motion under-drives this body (Froude 0.016-0.019) — a recipe limitation, not a scheduler bug; these 2 conditions excluded. Remaining 10 clear the gate at 2.1-5.6x, measured as `data/allocentric/beh10_c10f10t10_intent2_flat` (20 clips).

**Result (gate-verified this time):**

| target | intent, gated | clean null |
|---|---|---|
| `a_t` instantaneous | +0.061 | +0.102 |
| `da` command change | +0.028 | +0.017 |
| family label | +0.002 (acc 1.000) | +0.001 (acc 1.000) |

The action gap did not open (it's lower than clean); the intent (`da`) gap is only +0.011 above clean, inside F150's measured sitting-to-sitting variation of ±0.018 — nothing moved. `da` reads 0.743 from a single frame even on intentful data (stops, turn reversals mid-clip).

**Conclusion, across all interventions tried:** clean +0.102; speed ramp (phase-preserved) +0.024; random cmd_noise (F150, exogenous) +0.198; meaningful intent (this, verified landed) +0.061. Random jitter opens the gap, meaningful intent does not — **what opens the gap is not phase-breaking, it's the command being unpredictable from the pose.** A command a controller would actually issue stays pose-readable however non-periodic it is, because the body's configuration reflects the intent. This refines (not overturns) F145: periodicity is *sufficient* but not *necessary* for the redundancy — the deeper mechanism is that joint command is a function of visible pose. Direction B's cheap data-perturbation route is now measured negative for the useful case (only useless jitter opens the gap).

Separability: 3/66 pairs below 2x, all among turn-onset conditions differing in *when* the turn happens — expected, not a result either way.

**Appended scoping note (not a finding, no experiment run):** checked whether a counterfactual-outcome measurement (two measured futures from the same state under different actions) is testable on existing data. Constructing `(e_t, alternative_action)` pairs is trivial but unfalsifiable without a recorded alternative future. Ground truth is effectively absent: on the insect, only 4/765 transitions have a cross-family pose as close as a same-family one (no ground truth); on the B1, 166/768 (22%) look close, but a mirage — matched-pose pairs' actions differ by only 0.25 sd/joint vs 1.41 sd/joint for random pairs, i.e. where poses match, actions already match too (restates F145's finding in the data's own geometry). A true counterfactual measurement would differ meaningfully from F144 (doesn't depend on the trained model) but collapses back into F144-like territory without recorded alternative futures — not testable on `data/allocentric/beh12_*`; would require a resettable simulator collecting the same state under two different commands (see F153/F154 for whether that's viable).

---

### F153. `z` is more single-frame-readable than the action itself — not "forced" redundancy, closed as a direction; plus counterfactual-reset feasibility findings

**Main finding:** tested whether `z = ITM(e_t,e_t+1)` (built from two frames) is itself single-frame readable, to check whether the pipeline "forced" `z` toward the action (which would make it equally readable) or whether the transition itself is pose-determined. Same probe as F145, checkpoint `beh12_hex-b1_body3`, held-out body for insect.

| body | target | single frame | pair | gap |
|---|---|---|---|---|
| insect | `a_t` action | 0.779 | 0.867 | +0.088 |
| insect | `z` (width 64) | 0.856 | 0.895 | +0.039 |
| B1 | `a_t` action | 0.161 | 0.342 | +0.182 |
| B1 | `z` (width 64) | 0.543 | 0.649 | +0.106 |

On both bodies `z` is read from one frame *better* than the action is (B1: 0.543 vs 0.161, over 3x). Since `z` is more readable than `a_t`, not equally readable, "we forced z to equal the action" is not the explanation — `z` behaves substantially like a behavior label (family recoverable at accuracy 1.000, F152; body identity at 0.974, F146), which is trivially single-frame readable. Direction closed: no hidden dynamics quantity in `z` that a single pose fails to determine — but whether that's because the transition is genuinely pose-determined or because `z` collapsed toward identity+behavior is not separated by this test. Checkpoint-dependent (trained at lambda_body 0.5, which F146 showed increases body-identifiability); not claimed to generalize to a differently-trained `z`.

**Appended note — counterfactual-reset feasibility (no experiment run, scoping only):**
- B1/MuJoCo exact reset: `mjSTATE_FULLPHYSICS` (qpos/qvel/act) is NOT enough — max pose diff 2.83e-2 rad over 5 steps due to solver warmstart not being saved, comparable to the 5-step counterfactual signal itself. `mjSTATE_INTEGRATION` (adds warmstart/ctrl/forces) gives exact bit-identical (0.0) reset.
- Insect/CoppeliaSim: reset is NOT exact (F95: 37-71% reproducibility across repeats, no full dynamics-state snapshot available) — counterfactual experiment is B1-only.
- Counterfactual divergence (turn wz=0.6 vs sideways vy=0.4, same start state, zero noise floor since exact reset): at h=1, difference is only 0.04-0.08 px (sub-pixel, unusable); grows to 1.07-2.96 px by h=10-15, 5.8-18.6 px by h=30 for sideways; turning stays near 1 px even at h=30 (rotates without translating). Horizon needed: at least 10-15 steps, turning may need more than 30. Joint-angle column is non-monotone (gaits drift in/out of phase) — use base displacement, not joint angle.
- Throughput: physics reset+branch is 1.5ms (300 pairs <1s); rendering is the cost (~0.17s/frame; 300 pairs at 15-step horizon ≈ 9000 frames, ~25 min) — comfortably affordable. B1 physics (exact) is rendered separately via CoppeliaSim kinematic replay from stored states, so renderer non-determinism doesn't corrupt the physics.
- Not yet settled: whether the divergence is visible in the V-JEPA2 embedding (this is answered negatively in F154).

**Appended note — insect viability check:** repeat-vs-repeat noise floor (bit-identical commands) is 16-22x below cross-behavior signal at every horizon (e.g. h=1: signal 7.39mm vs noise 0.35mm, 21.2x; already 1.08px at h=1, vs B1's 0.04px). F95's earlier 37-71% reproducibility concern applied to closed-loop episodes (planner re-deciding on diverged states, compounding error) — open-loop scripted commands drift only millimeters. Correction: **both robots are viable for counterfactual work; the insect is actually the stronger candidate**, not excluded as previously implied. Caveat: these numbers compare different behaviors from the same spawn (diverge at frame 0, no shared momentum) — the harder, more realistic design (shared-prefix branch from identical pose/velocity/contact) needed confirmation (see below).

**Appended note — confirmation run**, `scripts/dataset/confirm_counterfactual.sh`, pre-registered pass mark: signal >3x noise on position AND heading, every horizon, per arm. Heading matters separately because turning barely displaces the robot (position alone would call it a failure).

B1: PASSED. Noise floor exactly 0 at every horizon (position and heading). Turn arm: h=25 position only 9.29mm (<1px) but heading 13.3deg — passes only because heading was measured. Sideways arm: mirror image (180mm displacement, 0.7deg heading). Neither channel alone would pass both arms — must report both. An off-by-one bug (heading referenced to first divergent frame instead of last shared frame) was caught by the render (compasses showed +1.4/+1.2 deg on a bit-identical prefix) and fixed.

Insect: PASSES from h=10 on both channels; turning passes on heading from h=5 (12.4x) while position is only 0.4x at h=5 — the pre-registered edge case, confirmed. Insect noise floor is nonzero but bounded (gait-phase offset ~26mm at frame 35, back to 12mm by frame 60, not accumulating). Two corrections found during this run: position must be measured relative to the last shared frame (not absolute) — fixed a large systematic bias (turn h=10 went from 1.6x to 3.3x, side h=10 from 1.5x to 7.8x); the `faster` (speed-schedule) arm does not actually share its prefix (`--schedule` shifts the whole CPG bias pose — a real defect in the F152 collection route, noted not silently fixed) and was excluded.

Clips: `results/cf_confirm/insect_forward-vs-turn.mp4`, `insect_forward-vs-side.mp4`. Render tool: `scripts/render/merge_counterfactual.py` (side-by-side, branch marked, prints shared-prefix pixel diff as a sanity check).

---

### F154. The counterfactual direction dies at the encoder: physically real, embedding-invisible

Final de-risk gate before any counterfactual-outcome collection: F153 confirmed the counterfactual divergence is physically real and cleanly measurable (world coordinates). This measures it in V-JEPA2 embedding space, since the forward model only ever sees embeddings, not millimeters. Insect, shared prefix to frame 33, displacement-since-last-shared-frame, noise floor = two identical-command runs through the same encoder.

| arm | h | signal | noise | ratio |
|---|---|---|---|---|
| turn | 5 | 1168 | 1236 | 0.9x |
| turn | 25 | 1432 | 1338 | 1.1x |
| side | 10 | 1298 | 1234 | 1.1x |
| side | 25 | 1498 | 1338 | 1.1x |

Nothing clears 3x (the pre-registered bar) or even 1.3x. Two futures 134mm and 30 degrees apart in the world (F153) are, to the encoder, as far apart as two runs of the identical command. Scale check: consecutive frames of one clip differ by 763-1010 in these units, essentially equal to the noise floor (~1000-1236) — one frame of gait motion moves the embedding as far as 25 steps of behavioral divergence. Same fact as F16/F22 (64-89% gait phase in `z`) and F145 ("pose determines command"), seen from the embedding-metric side directly.

3x bar justified because L2 in embedding space is literally the training loss (`L_recon` scores `FTM(e_t,z)` vs `e_t+1` by MSE) — a divergence invisible in L2 is absent from the gradient, not merely hard to detect.

**Conclusion: the counterfactual-outcome direction dies here** — everything upstream (exact reset, shared prefixes, physical divergence clearing 3x from h=10 on both robots per F153) passed; only the encoder fails. Measured on insect only — B1 frames need CoppeliaSim rendering; B1's physical divergence at h=25 (9.29mm) is an order of magnitude smaller than the insect's (134mm), so there's no reading where B1 would pass a gate the insect failed.

Left open, not proposed as a plan: cosine similarity shows a small real separation (0.767 vs 0.831 at h=25, ~0.06, against phase variance 0.17) that a *learned* metric might exploit — different project. Phase-aligned comparison is ill-posed post-branch (gait phases drift apart, "same phase" undefined).

Scripts: `scripts/diagnostics/embedding_divergence.py`. Data: `data/allocentric/cf_confirm/`, `results/cf_confirm/`.

---

### F155. Egocentric view breaks the single-frame action redundancy on the insect and preserves cross-body transfer — both pre-registered gates pass

Following F154 (allocentric counterfactuals invisible to the encoder), tests whether a head-mounted camera (which can't show the robot its own pose) breaks the redundancy. Minimal build: `sim/scene/ego_camera.py` mounts `vjepa_cam` on the head with a **measured** forward direction (insect: head-abdomen vector; B1: actual travel direction) — not an assumed axis convention (an earlier version mounted on `/abdomen` with a wrong assumed direction, caught only by looking at a rendered frame). Four colored/textured walls around spawn provide optical flow (blank walls give none, r=-0.20; high-contrast repeating edges alias, r=-0.16, per prior 2D-pan texture test — noted as not directly transferable to 3D rendering with parallax/depth/shadows). `--ego`, `--ego_forward`, `--ego_offset`, `--ego_box` flags added to both `collect_ik.py` and `render_b1_replay.py`.

Two pre-registered guards against confounds already seen elsewhere in the project: (1) matched seeds across bodies (`--ego_seed` on both collectors) so Q2 cross-body comparison isn't confounded by differently-looking rooms (F146-shaped risk); (2) `scripts/diagnostics/check_appearance_leak.py` verifies room-color doesn't leak heading information (crude color/hue summary vs heading, nearest-centroid, split by clip) — exits non-zero on a leak, gating collection (F151-shaped risk: a silently-broken setup passing every downstream check). Wall colors/textures/ground randomized per episode/repeat from `--ego_seed` so no color correlates with heading across the dataset.

Guard 2 check passed: room color predicts heading at 0.34x chance (insect), 0.50x chance (B1) — below chance, no landmark leak.

**Q1 result — redundancy breaks, decisively on insect:**

| insect | single frame | pair | gap |
|---|---|---|---|
| allocentric (F145) | 0.779 | 0.887 | +0.108 |
| egocentric | 0.293 | 0.578 | +0.285 |

| B1 | single frame | pair | gap |
|---|---|---|---|
| allocentric | 0.161 | 0.328 | +0.167 |
| egocentric | 0.097 | 0.259 | +0.162 |

Insect single-frame readability drops 0.486, gap nearly triples — the first intervention in the entire F141-F154 chain to move this quantity. B1 barely changes (was already low). Per family (insect): sideways drops to -0.008 (from 0.609), speed to 0.421 (from 0.812), turn to 0.483 (from 0.931).

**Q2 result — shared coordinate still crosses bodies, yaw improves substantially:**

Fitted on insect egocentric embeddings, tested on B1 without refitting:

| | forward | lateral | yaw |
|---|---|---|---|
| allocentric, B1 unrefitted (F148) | 0.63 | 0.43 | 0.07 |
| egocentric, B1 unrefitted | 0.50 | 0.39 | 0.64 |

Yaw transfer jumps 0.07→0.64 (turning is the weakest channel throughout the project, per F127/F154; head camera sees rotation as global image flow regardless of body). Forward/lateral transfer decline modestly (0.63→0.50, 0.43→0.39), and within-insect fit for forward also falls (0.98→0.77) — an honest trade-off, not an unqualified win.

**Pre-registered branch taken: Q1 pass + Q2 pass → direction alive, full environment worth building.**

Caveats: this is not a trained-model result — Q1/Q2 show the signal exists and transfers, not that a world model trained on this data will use it (F142-F154 repeatedly found signals a trained predictor then ignored); training pre-registration not yet written.

Data: `data/egocentric/beh12_c10f10t10_ego_flat`, `data/egocentric/beh12_b1_ego_flat`.

**Appended note — coordinate's supervision requirements vs Demo-JEPA (scoping only, nothing run):** the shared body-motion coordinate needs no paired cross-body data (`lambda_cross=0.0` in the measured checkpoint), no temporal alignment (no DTW/GTCC), and no hand labels (forward/lateral from differenced position, yaw from quaternion, Froude-normalized) — strictly less supervision than Demo-JEPA's end-effector retargeting + GTCC approach, because the target (forward/lateral/yaw) is a physical quantity both bodies already possess rather than a constructed correspondence. Must not be claimed supervision-free: it requires privileged simulator state (position/orientation ground truth — on hardware this is odometry/mocap, not vision) and a locomotion-specific stride-smoothing window. Scope limit: buys exactly 3 channels; anything not shared (joint spaces, gaits, contact patterns) is not carried, consistent with F99/F73.

**Appended note — gait shake vs world motion are separable and removing gait shake improves cross-body transfer (candidate method, not yet a finding to build on):** decomposing camera yaw into linear trend (net turn) + residual (gait oscillation) shows gait sits at 6 cycles/clip on both bodies, comparable in magnitude to the turn signal (gait/turn sd ratio 0.79 insect, 0.96 B1) — separable by frequency. Subtracting sin/cos at measured stride frequency (+2 harmonics) per clip, then refitting the coordinate (insect→B1 unrefitted): forward 0.45→0.47, lateral 0.38→0.46, yaw 0.57→0.61 (all improve cross-body; within-insect fit doesn't change much) — improvement is specifically cross-body, consistent with removing a body-specific nuisance component. Effect is modest and the method crude (linear per-clip harmonic subtraction); viable as a research direction, not validated as an improvement to ship. Note baseline numbers here differ slightly from Q2's own table due to different frame striding/split — only compare the two rows of this sub-table to each other.

Script: `scripts/diagnostics/degait_coordinate.py`.

---

### F156. Which wall texture V-JEPA2 reads motion from, measured directly; plus ten scene defects, a data reorg, a grayscale test, and egocentric-training scoping

**Main finding:** measured (no simulator — large texture panned past a 256px viewport, encoded by the project's own encoder) correlation between true pan distance and embedding distance, to pick a wall texture for the egocentric room (walls are the whole signal there, unlike a background floor).

| surface | r |
|---|---|
| blank | -- (no change) |
| checkerboard | -0.149 (reproduces `set_floor_texture.py`'s own -0.16, validating the method) |
| white noise | 0.199 |
| value noise, 5 octaves (old floor recipe) | 0.478 |
| **value noise, octaves 4/8/16 (adopted)** | **0.723** |
| value noise, octaves 8/16/32 | 0.613 |
| octaves 4/8/16, tiled 4x4 | 0.474 (repeating costs ~1/3) |

Adopted octaves 4/8/16 for egocentric walls. Best surface reaches r=0.72; this is a 2D-pan proxy (no depth/parallax/shadows) and doesn't establish sufficiency in the rendered 3D scene — that's answered by F155's Q1. Script: `scripts/diagnostics/texture_for_vjepa.py`.

**Ten scene defects found while building the egocentric room** (`sim/scene/ego_camera.py`, `--view egocentric`), each with what caught it: (1) camera mounted on abdomen with guessed axis — caught by looking at a frame; (2) `sim.createTexture` return values mis-unpacked — walls flat color; (3) shape color and texture both tinted, multiplied — near-black walls, caught by looking; (4) texture PNGs cached by filename, fix reused broken files — caught by `/tmp` timestamps; (5) 6m tile on 8m wall, visible seam — user looking; (6) plane-mapping fix smeared texture — caught by 4-way mapping render; (7) FOV set before `startSimulation` reverts to scene file's value (15deg not 90) — caught by junction angle mismatch (19.4 vs predicted 6.0); (8) room built before respawn, robot 2.8m off-center; (9) floor 5m inside 8m room; (10) insect head carried +7.53deg up while walking, camera parented to it — caught by 13.6 vs 6.0deg junction mismatch. Plus 3 cross-body asymmetries (B1 camera aimed along travel direction not body axis; B1 clips started at different headings; `--floor_scale` applied after texture, stretching it 3x) and 2 of the measurement instruments themselves were buggy (an argmax junction detector picked ceiling/floor edge inconsistently; a "floor detail" figure compared mismatched regions across bodies). Rule: when a measurement disagrees with geometry, check the measurement before changing the scene.

**Data reorg:** `data/` split into `data/allocentric/` (all pre-2026-09-01 sets) and `data/egocentric/` (head-camera sets) — a number measured under one view cannot be compared with the other (F155's whole subject). 278 path references rewritten across 117 files in 3 passes; 34 broken symlinks repaired (not re-collected). Two smoke tests (`check_within_clip_intent`, `branch_divergence`) confirm the move is inert (numbers reproduce exactly).

**`report/NUMBERS.md` and `report/retrain_and_remeasure.md` deleted** (superseded/obsolete). One orphaned number preserved: `NUMBERS.md` flagged a long-body distance figure (documented 4.125±0.434m) that does not reproduce (recomputed 4.404±0.187m) — still quoted uncorrected in `doc/PROGRESS.md:255` and `sim/SOURCES.md:76` (which itself calls it "bimodal, lands on 4.479 or 3.593"). It's a Step -1 morphology-gap number, nothing current depends on it, but it should carry a note. Standing rule kept: every number in a document should be regenerable by a command; unregenerable numbers get marked ORPHANED.

**Grayscale input test** (motivated by wanting to remove color-landmark leaks, weighed against V-JEPA2 being RGB-pretrained): BT.601 luma replicated to 3 channels. Q1 (redundancy) barely changes (insect 0.293→0.289 single-frame, B1 0.097→0.111 — no meaningful degradation). But grayscale makes the appearance-leak guard *worse*: leak ratio roughly doubles (insect 0.34x→0.87x chance, B1 0.50x→1.04x) because different wall tints map to different luminances, giving a cleaner brightness cue than the hue-mixed-with-noise RGB case. Verdict: do not switch to grayscale; RGB+color-randomization passes the leak guard by a wider margin. Nothing changed in the pipeline.

**Egocentric-training scoping (no run yet):** confirms training-set scale is right (48+48 clips egocentric, matching existing allocentric pretrain's 5,402 pairs) but the held-out evaluation body `c08f09t09` (every reported number in F145/F142 etc. is measured on it) has no egocentric collection yet — needed before any gate comparison is meaningful (~48 clips, ~10 min). Null `ITM(e_t,e_t)` carries over conceptually and is better-grounded egocentrically (stationary = a real state, vs allocentric's "pose frozen mid-swing"), but a preliminary check found feeding egocentric embeddings through an *allocentric*-trained ITM is out-of-distribution (cos(null,real) 0.952/0.982 vs allocentric reference 0.903) — can't be resolved before training an egocentric ITM, so it becomes a mid-run gate (`scripts/diagnostics/null_separability.py`, threshold 0.94, calibrated against the allocentric reference's 0.903/0.922) rather than a precondition. Architecture needs only `--sources` swapped (no camera-specific code elsewhere); `close_loop_b1_physics.py` still captures from the fixed camera and must be switched before any closed-loop run against an egocentric-trained model. Sequence defined as `scripts/run/step1_egocentric.sh {collect|train|measure}` with GATE A (appearance leak), GATE B (null separability), GATE C (null/real vs allocentric 1.03, pass if >~1.10).

---

### F157. Egocentric training: the world model now uses the action (GATE C passes) — but sideways calibration data was reconstructed and unreliable, and the action decoder fails to generalize

50 epochs, `wm/runs/beh12_ego` (com7), `--sources` swapped to egocentric, otherwise identical to the allocentric pretrain.

**GATE B (null must be a distinct latent, not an out-of-distribution artifact):** passes with margin — cos(null z, real z): allocentric reference 0.903, egocentric insect 0.756, egocentric B1 0.812 (lower = more distinct = better). Confirms the earlier 0.952/0.982 readings (F156) were indeed an OOD artifact of feeding egocentric embeddings through an allocentric ITM.

**GATE C (null/real, pre-registered pass bar >1.10):**

| lag | insect | B1 | allocentric baseline |
|---|---|---|---|
| 1 | 1.161 | 1.079 | 1.026-1.032 |
| 2 | 1.190 | 1.134 | -- |
| 3 | 1.171 | 1.135 | 1.032 |
| 5 | 1.156 | 1.121 | -- |

First time null/real moves meaningfully in the entire F141-F154 chain. Real beats null on 86.8-96.1% of samples (vs allocentric's ~70%); prediction/frozen-frame reads 0.515 vs allocentric's 0.737. Insect clears 1.10 at every lag; B1 is marginal — below bar at lag 1 (1.079), clears from lag 2. Sideways family remains the exception: insect speed 1.229, turn 1.208, side 1.047 (still at allocentric's level) — matches Q1's earlier finding that sideways didn't move.

**Body-motion coordinate:** forward and yaw hold (insect +0.98/+0.96, B1 similar, ~1.0-1.1x predicted-vs-measured ratio) but insect **lateral fails**: +0.30 correlation, 1.7x ratio, sign wrong on `side_R` (-0.074 measured vs +0.009 predicted). Traced to a data problem, not the camera: a fresh allocentric collection today reproduces the same weak lateral value (-0.103 vs stored -0.176), and sideways commands differ from the stored dataset by up to 1.07 rad (worst on the FT joint) while `speed_c7.1` reproduces bit-for-bit.

**Root cause (bisect note):** `beh12_c10f10t10_flat`'s sideways conditions were hand-collected before `scripts/dataset/collect_beh12.py` existed, and the exact commands were never recorded. Ten of twelve conditions are recoverable from condition names (e.g. `speed_c7.1` = `--cycles 7.1`); the two sideways levels per direction are not — `collect_beh12.py`'s sideways parameters (`SIDE_BASE`, `LVL0_STRAFE=0.4`, per-direction yaw-cancelling `--spin`) are a documented reconstruction from F62 fit to recorded lateral speeds, and it lands weaker than the original (`side_R_lvl1`: -0.120 reconstructed vs -0.176 stored; the generator's own `--verify` already fails on both left strafes). No commit is broken/regressed — the original commands simply never existed as data. **Consequence: sideways numbers are not comparable across allocentric/egocentric (different, both-weak versions of "sideways") and must not be quoted as a viewpoint effect.** Forward and turning are unaffected (commands recoverable from names) — GATE C and downstream gates stand for those. Fix path: re-derive sideways magnitudes from achieved lateral speed and re-collect, or drop sideways from the claim.

**Action decoder does not generalize from egocentric:** `MotionDecoder` train motion 0.83→0.076 over 50 epochs, but validation motion stays at 1.53 (worse than predicting the training mean, F78 reference 0.928) and never improves. This is the flip side of Q1/GATE C passing — the decoder reads `(e_t, z)` and egocentric `e_t` no longer supplies the command (as intended), but nothing recovered it from `z` either in this run. Body-coordinate head (reads `z` alone) trains normally (0.77→0.078, morphology probe 1.000) — this failure is specific to action-decoding, not the ITM/FTM gates.

**GATE D reference arm (measured first, dual ridge on `MotionDecoder`'s own input, architecture/optimizer removed):** `beh12_hex-b1_body3`, held-out `c08f09t09`.

| features | insect | B1 |
|---|---|---|
| `e_t` alone | 0.773 (reproduces F145's 0.779) | 0.166 |
| `z` alone | 0.903 | 0.790 |
| `[e_t, z]` | 0.938 | 0.789 |

**Unexpected finding: the B1's action was never recoverable from its pose** (0.166, essentially F145's insect-only mechanism) — pose-redundancy is an insect-specific route to the ignored-action problem, not universal; the B1's null/real still sat at allocentric ~1.03 without any pose-based explanation. Egocentric fixing the B1's null/real (to 1.13, per GATE C) cannot be explained as "removing the pose" since there was no pose-based redundancy to remove on the B1. This narrows F145's central mechanism to the insect and is a real gap in the account, flagged for correction in reporting (deck's mechanism slide currently overstated).

Scripts: `scripts/diagnostics/motion_decoder_ceiling.py`, `scripts/run/step2_gate_projector.sh`. No log saved for the main training run; egocentric arms tracked at `wm/runs/beh12_ego` (com7).

---

### F158. GATE D: egocentric action-command information drops on both `e_t` and `z` (not just the frame); the projector is unaffected; "ceiling" language corrected

**D1 — command readability, dual ridge, split by clip:**

| features | insect allo | insect ego | B1 allo | B1 ego |
|---|---|---|---|---|
| `e_t` alone | 0.773 | 0.329 | 0.164 (reproduces F157's 0.166) | 0.117 |
| `z` alone | 0.903 | 0.614 | 0.790 | 0.345 |
| `[e_t,z]` | 0.938 | 0.608 | 0.789 | 0.334 |

`e_t` falling was expected (that's Q1). Unexpectedly, **`z` fell too** — insect 0.903→0.614, B1 0.790→0.345 (less than half). The pre-registered PASS condition (burden shifting onto `z`, `z` holding near its allocentric level) is not met on either body. Nor is FAIL met (neither input carrying the command) — both still carry meaningful signal, confirmed by refit headroom: trained head scored -0.530 (broken/untrained), ridge on same input scored +0.608 (insect) / +0.334 (B1) — a +1.14/+0.86 headroom gap, meaning F157's val-motion-1.53 failure was a **training failure, not a representation failure** (the information was present but the trained decoder didn't extract it). Note: "z carries the command less" (this finding) is separate from "the forward model uses z more" (GATE C, F157, 1.03→1.16) — both true simultaneously; egocentric does NOT make z carry the action better.

**D2 — projector unaffected:** rollout-gap ratio (vs predicting mean z): insect allo 0.352 vs ego 0.355 (identical); B1 allo 0.235 vs ego 0.169 (better egocentrically). `a→z` learnability is unaffected by viewpoint.

**Verdict (revised from pre-registration):** teacher-student is neither blocked nor cleanly cleared. Projector passes; decoder failure is repairable (training issue, not representational); but the achievable ceiling for a student dropped from 0.938→0.608 (insect) and 0.789→0.334 (B1) — a B1 student can account for at most ~1/3 of variance even with a perfect head, and this number must be the reference for any teacher-student result, not discovered after the fact.

**Correction issued within the hour: "0.608/0.334 ceiling" was the wrong word — it's a linear reference, not an upper bound.** Refitting the decoder (with cross-attention, not linear ridge) on the *allocentric* checkpoint lands ABOVE the ridge on both bodies: insect 0.938→0.982 (105%), B1 0.789→0.910 (115%). The ridge is linear; the decoder is not, so the ridge is a lower bound on what a cross-attention head can extract, not an upper bound. Corrected reading rule: (1) read every refit/student number against its own body's reference (0.608 insect, 0.334 B1), never against 1.0; (2) expect a refit to land slightly ABOVE the reference (roughly 0.64-0.70 insect, 0.35-0.38 B1 estimated at 105-115%) — landing at the reference is a pass, landing far under is the stop; (3) "0.334 bounds the controller" is too strong — restate as "0.334 is what a linear readout reaches; the nonlinear head beat that by 15% allocentrically."

Unchanged: egocentric still helps the world model (GATE C) and costs the decoder — both facts should always be stated together.

Scripts: `scripts/run/step2_gate_projector.sh` (com7, no log saved), `wm/refit_decoder.py` (control arms, `beh12_hex-b1_body3`, 60 epochs each body). Checkpoint: `wm/runs/beh12_ego/projector_ego.pt`.

---

### F159. The refit decoder clears the reference on both bodies — egocentric's true cost to the decoder is ~14%, not 35-58% as the linear ridge implied; teacher-student unblocked

`wm/refit_decoder.py` on `beh12_ego`: ITM frozen (same `z` GATE C was measured on), decoder heads reinitialized, backbone kept, same clip-split-by-family as F158's ridge.

| | best test R2 | floor/reference (F158) | ratio |
|---|---|---|---|
| insect | 0.847 | 0.608 | 139% |
| B1 | 0.778 | 0.334 | 233% |

B1 at 0.778 vs egocentric linear readout's 0.334 settles F157's open question in the strongest form: the val-motion 1.53 was purely a training failure, information was present all along.

**Correction to F158's framing:** "egocentric costs the decoder 35%/58%" (measured via linear ridge) understates it — with a real (cross-attention) head, cost is closer to 14%:

| ratio ego/allo | insect | B1 |
|---|---|---|
| linear ridge | 0.65 | 0.42 |
| refitted decoder | 0.86 | 0.86 |

0.863 and 0.855 — same number on two bodies with disjoint action spaces, unlikely to be coincidence. Hypothesized mechanism (not measured, a hypothesis): the ridge flattens 256×1408 patch tokens into one vector and fits linearly; the decoder does cross-attention over tokens. Allocentric command info (limb configuration) survives flattening; egocentric command info (where things sit in frame, spatial) needs attention to read, so linear methods understate it more egocentrically (measured gap: 2.3x ridge-vs-decoder gap on egocentric B1 vs 1.15x allocentric).

Caveat: head memorizes somewhat — train R2 hits 0.99+ by epoch 20 while test plateaus at 0.84/0.77 by epoch 50 and doesn't improve over next 250 epochs; `--wd 1e-2` didn't remove it. Generalizes "well enough," not a solved head.

**Decision: teacher-student opens.** Pre-registered stop condition (refit landing far under the reference) didn't fire on either body. Refitted decoder numbers (0.847 insect, 0.778 B1) become the new tighter reference — no student handed its own `z` can be expected to beat a head given the true `z`. Checkpoint: `wm/runs/beh12_ego/md_refit.pt` (300 epochs, best ~epoch 50).

**Appended note — P3 allocentric control for the pooled student, run before the egocentric arm** (`scripts/diagnostics/pooled_student_check.py`, `beh12_hex-b1_body3`, same clip split):

| | insect | B1 |
|---|---|---|
| ridge, every token | 0.773 | 0.166 |
| ridge, pooled | 0.634 | 0.216 |
| ridge, `[pooled,z]` | 0.921 | 0.761 |
| MLP, `[pooled,z]` | 0.954 | 0.788 |
| **MLP, pooled alone** | **0.800** | **0.250** |

Given `z`, pooling costs almost nothing allocentrically (0.954 vs cross-attention decoder's 0.982; 0.788 vs 0.910) — pooling isn't the bottleneck here. **The actually load-bearing measurement**: the Student model is never handed `z`, only `(pooled(e_t), goal)`, with goal constant within an episode — so within an episode the command must come from the pooled frame alone. That row: **0.800 insect / 0.250 B1, allocentric**, with the student's own architecture. This is a pure architectural bound that applied (unmeasured) to F135/F136 all along. Corrected reading frame: F159's 0.847/0.778 references a head handed the true `z` (right reference for the decoder) but is NOT the student's bar, since the student doesn't receive `z`. Egocentric arms of this check are outstanding (F160 follows up) and needed to know whether pooling costs more egocentrically (F159's spatial hypothesis).

---

### F160. P3: the pooled student architecture is dead on egocentric B1 (0.081); losing the token grid (not pooling itself) is the cause; causal-z injection doesn't help; plus teacher scope, and a corrected student bound

P1 (teacher assembly) done first: `teacher_ego.pt` = `best.pt` (itm, ftm) + `projector_ego.pt` + `md_refit.pt`, no stage-3 adaptation — all downstream gates measured on these exact weights.

**Student architectural bound** (MLP on `pooled(e_t)` alone, since goal is constant within an episode):

| | allocentric | egocentric |
|---|---|---|
| insect | 0.800 | 0.263 |
| B1 | 0.250 | **0.081** |

B1 egocentric pooled student cannot represent the command at all (0.081); allocentric B1 (0.250) was already near-nothing. This is an architectural bound reached before any teacher exists, and a candidate (partial) explanation for F135 that F136 never separated from teacher quality.

**Pooling itself is cheap** — like-for-like linear comparison shows pooling costs only 2-3% either allocentric or egocentric (`[pooled,z]` as % of `[e_t,z]`: 98%/96% allo, 97%/98% ego). F159's "spatial hypothesis" (pooling costs more egocentrically) is NOT confirmed at the linear level. It only appears when comparing the pooled MLP against the full cross-attention-over-tokens decoder: allocentric 97%/87% (insect/B1) vs egocentric 75%/54% — the gap roughly doubles egocentrically. **Conclusion: it's losing attention over the token grid that hurts, not pooling per se** — these are different claims, and earlier framing conflated them.

**Causal-z injection tested and rejected:** since `z=ITM(e_t,e_t+1)` needs the future, tested giving the student `proj(a_t-1)` as a causal substitute.

| Student MLP | insect | B1 |
|---|---|---|
| pooled alone | 0.263 | 0.081 |
| + `proj(a_t-1)` | 0.785 | 0.482 |
| + `a_t-1` raw (control) | 0.807 | 0.496 |

Raw previous command beats its projection on both bodies — the lift is pure autoregression (periodic locomotion: `a_t-1` nearly states `a_t`, restating F145), not the latent; `proj(a_t-1)` adds nothing over `a_t-1`. Caution: this lift is not competence — an autoregressive student that continues its own pattern produces a gait it can't be steered out of (same coarse-not-fine failure as F136); it would score well on this metric yet fail closed-loop. This row is a diagnosis, not a design recommendation.

**Decision:** the pooled student is not the right architecture; the fix indicated is restoring the token grid (attention), not injecting a causal z — attention-over-tokens reaches 46% of reference on egocentric B1 vs pooled MLP's 54%-of-nothing. The 20Hz throughput budget that motivated pooling needs revisiting. F135 is only half-reattributed to this bound — a student this bounded produces F136's exact symptoms, but disentangling teacher-quality vs architecture-bound explanations requires "Q1" (not yet run); F136's conclusion should not be treated as settled until then. Script: `scripts/diagnostics/pooled_student_check.py` (com7, no log saved).

**Appended note — teacher scope check:** F135 had bounded its (different, stage-3-adapted) teacher's usable body to `c10f10t10` (state fidelity 1.052 on held-out `c08f09t09`, worse than freezing the frame). Checked whether `teacher_ego.pt` inherits this limit: `rollout_fidelity.py` on held-out `c08f09t09` egocentric gives ratio 0.510 at h=1, rising only to 0.596 by h=10 (moves 0.52-0.63, ruling out input-copying collapse) — does NOT inherit the limit. Confirms Q1 should run on `c08f09t09` (the truly held-out body used for GATE B/C/F158/F160), correcting an earlier lean toward `c10f10t10`.

Recorded prediction (pre-registered before Q1 runs): shuffled-z costs only 22% (0.622 vs 0.510) and a 1-sd off-manifold perturbation costs only 3% — since F135's planner candidates are 0.5-sd Gaussian perturbations (much smaller than a full shuffle), Q1 is predicted to come back near chance, for reason (b) candidates being near-identical through model+physics, not (a) pose-redundancy (already ruled out by GATE C).

Flag correction: `teacher_label_quality.py` must score through `proj(a)` (labeling path), not the raw ITM path (F130: forward model is action-sensitive only inside the projector's region). Re-run on projector path: ratio 0.533 (h=1, close to ITM path's 0.510, confirming the labeling path is sound). `/mean-z` within-clip/within-family/across-all all collapsed to ~0.88 egocentrically — allocentrically these were far apart (F110's 0.49 vs F131's 0.951) but **that comparison is invalid**: F110/F131 were measured on stage-3-adapted checkpoints, `teacher_ego.pt` is stage-1-only — comparing conflates viewpoint and adaptation stage (a mistake this project has had to withdraw before). No claim about egocentric changing coarse/fine sensitivity can be made from these numbers until a same-stage allocentric control is measured (needs a projector fit against `beh12_hex-b1_body3/best.pt`, not yet done). Only within-run reading stands: real action beats mean action by ~12% uniformly at every scale. Confirms the near-chance Q1 prediction from a second angle (whole shuffle worth 17% via projector path, 1-sd perturbation worth only 1%).

**Appended note — Q2 clone beats P3's stated bound, because the bound excluded the goal input:**

| | clips | best R2 | epoch |
|---|---|---|---|
| `--forward_only` | 13 (3 held out) | 0.396 | 400 |
| all 12 conditions | 39 (9 held out) | 0.433 | 200 |

0.433 exceeds P3's pooled-alone bound of 0.263 (insect) — impossible unless the bound was measured wrong. **It was**: P3 measured `pooled(e_t)` alone, but `Student` actually receives `(pooled(e_t), goal)`, and goal (a 3D body-motion target differing per condition) explains most between-condition variance. The correct comparison row (`[pooled(e_t), goal]`) was never measured — correcting the locked reading frame, not the data. Implication: a student scoring 0.433 by reading its goal is mostly "selecting a behavior" (which F136 already showed the pipeline can do, 55% vs 33% chance), not producing the right *amount* of that behavior (steering) — the number that would actually predict closed-loop performance is R2 computed *within condition* with goal partialled out (same coarse-vs-fine distinction as F136/F102), not yet measured. Minor notes: clone runs overfit early (best at epoch 200-400 of 2000, best-weights-kept fix applied but not a healthy fit); `--forward_only`'s 0.396 is not comparable to the 12-condition 0.433 (narrower target) — script now warns about this itself.

---

### F161. The goal is redundant egocentrically too; the closed-loop bar is 0.294 / 0.059

Prediction (goal helps a lot egocentrically, within-cond drops below pooled) was wrong on both counts.

| Student MLP, `[pooled, goal]` minus `pooled` alone | allocentric | egocentric |
|---|---|---|
| insect | +0.001 | +0.001 |
| B1 | -0.003 | +0.043 |

Goal adds ~nothing (insect identical allo/ego; B1 gain is 0.081->0.124, small numbers). Does not explain clone (0.433) vs P3 (0.263) gap.

Two unmeasured, plausible mundane biases for that gap (not established): (1) clone early-stops on the reported set (best of 10 noisy held-out evals, ~0.04 MSE spread) vs P3's CV done only inside train half; (2) clone trains on more/holds out less (39 clips, 9 held out) vs P3's split-by-family halving.

Closed-loop bar (`within cond`, goal held constant):

| | overall | within cond |
|---|---|---|
| insect, `[pooled, goal]` | 0.322 | 0.294 |
| B1, `[pooled, goal]` | 0.093 | 0.059 |

B1 student explains only 6% of within-condition command variance; a perfect teacher cannot fix this (policy can't represent the magnitudes). within-cond sits 0.01-0.04 below overall everywhere on both bodies/viewpoints -- F136's coarse-vs-fine wall does not reappear at the student; it is bad at both selection and steering.

Q1 must be read against 0.294/0.059 per body, not 0.263 or 1.0. Insect is the only body where Q1 can mean anything; B1 at 0.059 is too close to zero to separate a teacher's ranking from noise.

Scripts: `scripts/diagnostics/pooled_student_check.py` (run on com7, no log saved).

Note: this bar (0.294/0.059) is later superseded by F163 (0.297/0.127 pooled, 0.388/0.205 attention).

---

### F162. Pooling never bought the budget it was adopted for: encoder is 95.5 ms, every head is under 0.3 ms

`Student` pools the token grid to hold 20 Hz; the budget was never actually measured. Allocentric checkpoint, held-out `c08f09t09`, 2080 Ti:

| head | params | R2 | within cond | ms/frame | % of 50ms step |
|---|---|---|---|---|---|
| pooled (current Student) | 1.0M | 0.748 | 0.738 | 0.09 | 0.2% |
| conv | 0.7M | 0.767 | 0.757 | 0.23 | 0.5% |
| attention (F160's path) | 2.0M | 0.804 | 0.796 | 0.29 | 0.6% |

Frozen V-JEPA2 encoder alone: 95.5 ms/frame = 191% of a 50 ms step. Encoder is 329x the cost of the attention head and already overruns the step by itself; pooling saves only 0.2 ms out of 95.8 ms. The architecture's original speed rationale doesn't exist -- 20 Hz was never reachable with this encoder in the loop, so head choice should be made on accuracy alone. Attention costs 0.06% of a step more than pooling for +0.056 R2; literature (Hu et al. 2207.03386, Xiao et al., 2605.14106) also reads spatial grid rather than pooling it, though those encoders are trained end-to-end vs. frozen here.

Caveat: measured on one GPU (2080 Ti); com7 is faster so absolute ms differs, but a 329x ratio won't invert. Encoder cost measured per-frame with no batching/caching.

Open question (resolved in F163): whether attention recovers the egocentric gap, where pooled reads 0.263/0.081 vs decoder-with-true-z's 0.847/0.778. Allocentrically pooled-to-attention gap is only 0.056 since pose is already in-frame regardless of read method.

---

### F163. Attention is the right head, buys +0.08 (not the tenfold hoped for); pooled baseline was also underfit

| egocentric, within condition | pooled | conv | attention |
|---|---|---|---|
| insect | 0.297 | 0.278 | 0.388 |
| B1 | 0.127 | 0.054 | 0.205 |

| overall R2 | pooled | conv | attention |
|---|---|---|---|
| insect | 0.325 | 0.306 | 0.412 |
| B1 | 0.158 | 0.088 | 0.234 |

Attention wins on both bodies by ~0.08; conv loses to pooling on both (rejected by measurement -- CNN's spatial bias doesn't transfer to a frozen 16x16 grid of 1408-dim semantic tokens).

Hoped-for reading (B1 "solved" via head choice, reaching near 0.778 decoder-with-true-z ceiling, or corrected estimate ~0.27) did not happen: came back at 0.234 overall / 0.205 within-cond. B1 is not solved, just less bad (accounts for ~1/5 of within-cond variance).

Pooled baseline itself rose once properly fit (minibatch AdamW, 200 epochs, vs earlier full-batch fit):

| pooled, within condition | F160/F161 fit | this fit |
|---|---|---|
| insect | 0.263 / 0.294 | 0.297 |
| B1 | 0.081 / 0.059 | 0.127 |

So "pooling architecturally fatal" claim is withdrawn -- the earlier B1 low number was substantially a fitting-procedure artifact. Honest gap is 0.127 (pooled) to 0.205 (attention): architecture worth changing but not a rescue.

Closed-loop bar superseded: 0.297/0.127 pooled, 0.388/0.205 attention (replaces F161's 0.294/0.059).

F162 confirmed on com7: encoder 51.3 ms/frame (103% of 50ms step) vs this machine's 95.5 ms; every head 0.05-0.17ms (0.1-0.3% of step). Ratio 340x on faster card -- 20 Hz still unreachable with frozen V-JEPA2 in the loop regardless of head. Encoder cost, not head choice, is the real budget problem.

---

### F164. Q1: physics separates candidates cleanly but the teacher still can't rank them (egocentric, Gaussian perturbations)

`teacher_label_quality.py` on `teacher_ego.pt` and egocentric clone, held-out `c08f09t09`, goal clip `hexapod_ep100` (`speed_c7.1`, repeat 0, in clone's `val_paths`), head camera fov 90, 8m room, scene `medauroidea_c08f09t09.ttt`, 15 branch points.

| | egocentric | F136 allocentric |
|---|---|---|
| teacher's pick closer to goal | 7/15 = 47% | 4/12 = 33% |
| coin | 50% | 50% |
| teacher kept student's own action | 1/15 | 0/12 |
| mean distance student/teacher | 0.1360/0.1358 | 0.1299/0.1304 |

47% vs 50% coin -- teacher changed the action 14/15 times and gained nothing measurable.

Control F136 lacked: same action executed twice (4 states) gives mean |teacher-student| distance floor of 0.0000 (simulator repeats exactly), vs 0.0034 measured signal -- 180x signal over floor. So outcomes are distinguishable/reproducible; F136's "physics barely separates them" explanation is ruled out here -- the teacher simply cannot order them. (Pre-registered prediction of near-chance was right, but for the wrong reason.)

GATE C (forward model uses the action, 1.03->1.16, F157) did not convert into ranking ability -- these are separate capabilities. Student's own architecture bound is real (0.297 pooled, 0.388 attention, per F163) but independent of teacher's labeling failure (a coin-flip label can't be fixed by student architecture).

Coarse arm (12 recorded conditions instead of perturbations), re-run after fixing a cache bug (cache built for `c10f10t10`; every `c08f09t09` clip silently skipped via `if p_ not in cache: continue`, printing "0% of 0 states" as if measured -- script now raises instead of silently skipping):

| egocentric, 120 states | 52% |
| F136 allocentric | 55% |
| chance | 33% |

Coarse survives (teacher tells walking/turning/strafing apart), fine does not (can't tell a good walk from a slightly different one) -- reproduces F136's coarse-vs-fine wall under the new viewpoint at both ends; barely moved from allocentric (55%->52%, 33%->47% vs coin).

Scripts: `scripts/diagnostics/teacher_label_quality.py`, CoppeliaSim port 23000. No log saved.

Note (four operational traps, carried over before `OVERNIGHT.md` was removed; still relevant):
1. Encoder cache has no locking -- concurrent writers to one cache path clobber each other, causing silent full re-encodes (made overnight jobs look hung). Use per-script cache paths. Still live: egocentric session added scripts writing `results/wm/cache/ego_hex.pt` and `fid_*.pt`.
2. `--encode_device cuda` cached tensors on GPU while trained modules stayed on CPU (fixed in four scripts).
3. `--checkpoint_every 2` at 60 epochs writes 30 snapshots / 11GB, filling disk to 100%; use 10 for long runs.
4. `pkill -f <pattern>` matches its own command line and kills the issuing shell (hit twice this session, exit 144); pattern must exclude the caller or address by PID.

---

### F165. Teacher ranks 83% on recorded conditions vs chance on perturbations -- F164's failure was separation, not sensitivity

> Note: the reading below (limit = separation magnitude) is withdrawn by F166 -- a sigma sweep reaches 10.0% separation and still ranks at 42%. The 83% here is real, but the cause is that library candidates stay inside the recorded command range and carry gait structure, not that their outcomes are further apart.

De-risk run before an ActSWM-on-egocentric retrain. Same instrument/body/goal clip/physics judge as F164; only candidate set changed from Gaussian perturbations (0.5 sd) to the 12 recorded conditions.

| | F164 perturbations | F165 conditions |
|---|---|---|
| separation between executed outcomes | 2.5% | 12.8% |
| teacher's pick closer to goal | 7/15 = 47% | 10/12 = 83% |
| coin | 50% | 50% |
| mean distance student/teacher | 0.1360/0.1358 | 0.1359/0.1261 |
| teacher kept student's own action | 1/15 | 0/12 |
| simulator noise floor (same action twice) | 0.0000 | 0.0009 |
| signal over floor | 180x | 20x |

83% vs coin, teacher's picks land 7% closer to goal on average -- world model can rank; F164's failure was candidate separation, not model sensitivity.

Implication (per note above, later revised by F166): ActSWM-on-egocentric retrain not justified by this evidence alone -- an objective sharpening action-sensitivity wouldn't address the binding cause. F135/F136's candidate generator (Gaussian perturbation at 0.5 sd) is what needs fixing, not the teacher/objective.

Unresolved: exact threshold between 47%@2.5% and 83%@12.8% is unmeasured (a sigma sweep would give the curve -- see F166). Ranking among 12 recorded conditions is close to F122's existing behaviour-library selection setting, so 83% here isn't fully a new capability. Open question was whether candidates can be generated (not just recorded) to differ by ~10% -- relates to F128 (sampled action sequences producing same motion).

Noise floor moved 0.0000->0.0009 (F164's exact zero was fortunate, not a simulator property); both runs still far above floor.

Scripts: `scripts/diagnostics/teacher_label_quality.py --candidates conditions --states 12`, CoppeliaSim port 23000, held-out `c08f09t09`, egocentric.

---

### F166. Sweep corrects F165: separation is not the limit -- being on-manifold is

F165 drew a conclusion from two points; a sigma sweep with four more points overturns it.

| candidates | separation | teacher closer | one-sided p | joints outside recorded range | upright |
|---|---|---|---|---|---|
| perturb, sigma 0.5 (F164) | 2.5% | 7/15 = 47% | 0.70 | not measured | not measured |
| perturb, sigma 1.0 | 4.4% | 6/12 = 50% | 0.61 | 14.8% | 12/12 |
| perturb, sigma 2.0 | 10.0% | 5/12 = 42% | 0.81 | 33.3% | 12/12 |
| perturb, sigma 4.0 | 9.3% | 7/12 = 58% | 0.39 | 40.3% | 12/12 |
| twelve recorded conditions | 12.8% | 10/12 = 83% | 0.019 | -- | -- |

Sigma 2.0 reaches 10.0% separation (near the library's 12.8%) but ranks at 42%, below coin. Every perturbation arm sits at chance regardless of separation; only the library arm beats chance. F165's conclusion ("limit is separation magnitude") is withdrawn.

What actually separates library candidates: whether the action is one the model was fitted on, not how far outcomes land. Off-range joint fraction climbs with sigma (14.8%, 33.3%, 40.3%) exactly as ranking fails. Recorded conditions are in-range by construction and structured (periodic gaits, not pose noise). This restates F130 (forward model is action-sensitive only inside the region the projector produces) now measured in physics: magnitude was never the variable, kind was.

Realizability was not the failure mode: all four sigmas stayed upright 12/12 -- robot doesn't fall, it's driven by commands the model can't predict consequences of.

Consequence for candidate generation: Gaussian perturbation is disqualified as a generator entirely -- no sigma works (small = too little separation, large = off-manifold). Generator requirement (measured): produce actions inside the recorded command range with gait structure, that still differ enough in outcome. Only the recorded library meets this; nothing generated has (ties to F128's result that sampled action sequences all produce the same motion).

Caveat on n: 12 branch points give SE ~14 points, so 42/47/50/58% are statistically one number; only 10/12 (83%, p=0.019) clears. Sweep establishes no perturbation arm beats chance; does not resolve them against each other.

Scripts: `scripts/diagnostics/teacher_label_quality.py --sigma {1,2,4} --states 12`, egocentric, held-out `c08f09t09`, CoppeliaSim port 23000.

---

### F167. World model gradient is not usable for planning -- fails even at one step, all excuses ruled out by controls

De-risk before an actor-critic in imagination. Backprop imagined F127 body-motion reward through `body head <- ITM <- FTM^K <- projector` into the action sequence, take one normalized step, execute in simulator, check if real motion moved toward goal. Six branch points, held-out `c08f09t09`, egocentric, step 0.5 sd.

| K | grad better | random better | mean base | mean grad | mean random | mean \|grad\| | joints out of range |
|---|---|---|---|---|---|---|---|
| 1 | 3/6 | 3/6 | 0.1366 | 0.1370 | 0.1366 | 1.97 | 0.9% |
| 3 | 3/6 | 4/6 | 0.1325 | 0.1327 | 0.1330 | 3.71 | 1.9% |
| 5 | 3/6 | 4/6 | 0.1314 | 0.1311 | 0.1308 | 3.67 | 1.7% |

Gradient never beats its own random control at any horizon; at K=1 following it is worse than not moving; at K=5 a same-norm random step does better.

Controls ruling out alternative explanations: gradient flows fine (norms 2-9, finite, nonzero, no vanishing/exploding through 5 FTM applications). Steps stay on-manifold (0.9-1.9% joints out of range, vs the 33% that broke ranking at sigma 2.0 in F166) -- gradient isn't being read off an unseen region. Fails already at K=1, so this is not compounding-horizon error (F129, F141) -- a clean, in-range, one-step gradient points nowhere better than chance. Per pre-registered read: failure at K=1-2 means gradient misleads; an actor trained through it would be pushed the wrong way. Dreamer-style imagination planning is not viable with this world model.

Limits: n=6 per horizon (3/6 is exact chance, cannot separate 50% from 65%); mean distances are the sharper statistic and agree to the third decimal (gradient/random/no-move all ~0.3% apart on 0.13).

Unifying statement (covers F164, F166, F167): step is 0.5 sd; F164 measured 0.5 sd separates outcomes by only 2.5%, F166 measured sigma 2.0 puts 33% of joints out of range. No step size makes the model's own signal both effective and trustworthy -- too small changes nothing, too large and the model is guessing.

Scripts: `scripts/diagnostics/dreamer_gradient.py --states 6 --horizons 1 3 5`, CoppeliaSim port 23000.

---

### F168. LDAD tested and rejected: it makes the world model worse -- reconstruction gain traded away action-use

Delta-JEPA's Latent Difference Action Decoding (LDAD) trains the world model so consecutive state-latent differences carry the action, weight 10-50. Mapping: Delta-JEPA's `z` is a state latent, ours is the action latent; here `dz = e_t+1 - e_t` (the V-JEPA2 embedding difference), a different/easier quantity than decoding from a difference of action codes.

Untrained baseline (held-out `c08f09t09`, egocentric, split by clip 499/499):

| features | action R2 | within cond |
|---|---|---|
| dz = e_t+1 - e_t (LDAD's input) | 0.199 | 0.167 |
| e_t alone | 0.329 | 0.302 |
| [e_t, dz] (F153's setting) | 0.490 | 0.469 |

Response-separation diagnostic (Delta-JEPA Fig 6 style: hold e_t, vary action, measure predicted-response magnitude): action perturbed 0.5 sd (F164's 47%-ranked candidates) -> response 0.147 of magnitude; action from a wholly different transition -> 0.461; ratio 3.1x. In physics the two classes separate by 12.8% vs 2.5% = 5.1x. Model's ratio (3.1x) is below physics' (5.1x) -- suggestive that fine perturbations are relatively over-weighted vs coarse ones (mis-calibration hypothesis, not blindness), computed in different spaces so only the ratio-of-ratios is comparable.

Pre-registered LDAD run: `wm/models/ldad.py` (3 transformer layers, per-embodiment heads), `lambda_ldad`/`ldad_layers` in `wm/config.py`, term applied to prediction `FTM(e_t,z)-e_t` (not two true frames), run sheet `scripts/run/f183_ldad.sh`, lambda {10,50}, 15 epochs vs baseline's 50.

Result:

| | dz_pred R2 | dz_pred within cond | response ratio | fraction of physics 5.1x |
|---|---|---|---|---|
| baseline, lambda 0 | 0.363 | 0.338 | 3.1x | 0.61 |
| lambda 10 | 0.537 | 0.518 | 4.0x | 0.78 |
| lambda 50 | 0.555 | 0.537 | 4.3x | 0.84 |

Both moved monotonically; within-cond (pre-registered deciding metric) rose 0.338->0.537 (59% relative), response ratio moved toward physics (3.1x->4.3x). Note: `dz=e_t+1-e_t` itself (a frozen-encoder property) cannot move with any objective and read identically (0.199/0.167) at all three lambdas -- the correct baseline is `dz_pred`'s lambda-0 value (0.338), not the frozen dz row.

Both fine and coarse responses shrank ~5-fold in absolute terms (fine 0.147->0.030, coarse 0.461->0.119) even as the ratio improved -- flagged as needing a GATE 1 check before trusting the ratio gain.

GATE 1 (`null/real` on held-out insect, vs F157's 1.16) -- FAILED, inverted:

| lag | no LDAD (F157) | lambda 10 | lambda 50 |
|---|---|---|---|
| 1 | 1.161 | 0.999 | 0.993 |
| 2 | 1.190 | 1.001 | 0.986 |
| 3 | 1.171 | 1.004 | 0.986 |
| 5 | 1.156 | 1.006 | 0.987 |
| real<null, lag1 | 86.8% | 43.7% | 34.7% |
| real/hold, lag1 | 0.515 | 0.668 | 0.734 |

At lambda 50, model predicts better given no action than given the true action (real beats null only 34.7% of the time at lag1, worse than coin). Effect scales with lambda -- the term is the cause, not noise. GATE 2 (ranking test) was not run per pre-registration (a failed GATE 1 stops the sequence).

Mechanism: LDAD's objective is satisfiable by writing `z` into the prediction residual in a linearly-readable form, without requiring the prediction itself to be accurate -- model stamps the action into output while getting the future more wrong. Named "the LDAD shortcut," inverse of Context Collapse (ActSWM predicts well from frame alone while failing to use the action; LDAD makes the action legible while failing to predict well). Conclusion: displacement-reconstruction (as used by Delta-JEPA) cannot by itself establish that a world model uses the action -- it separates "action is legible" from "model uses it," and in periodic locomotion this shortcut is reachable (untested for manipulation).

Verdict: LDAD made the world model worse; gains were bought by sacrificing absolute action-sensitivity. Not claimed: whether a longer run than 15 epochs would recover `null/real` while keeping the reconstruction gain (untested). The Delta-JEPA/LDAD path is closed -- rejected at both lambdas in the authors' own recommended range.

Runs: `wm/runs/beh12_ego_ldad{10,50}`, projectors fitted per arm. Logs: `results/f183/`.

---

### F169. Behavioural test: egocentric nearly destroys the policy; allocentric plain cloning already passes the bar

F135 only measured the allocentric clone on a different body; the missing same-body, same-scene, same-goal-clip, same-code comparison (only camera differs) was run here.

| | held-out cloning error | replayed reference | clone travelled | fraction | result |
|---|---|---|---|---|---|
| allocentric, `c08f09t09` | 0.0116 | 0.6927 m | 0.3756 m | 54% | PASS |
| egocentric, `c08f09t09` | 0.6040 | 0.7151 m | 0.0426 m | 6% | FAIL |
| F135, allocentric, base body | 0.065 | 0.6566 m | 0.2349 m | 36% | FAIL |

Both stayed upright the full three seconds; egocentric failure is a policy that barely travels, not a collapse. Denominator (replayed reference distance) was re-measured per body/arm rather than reusing F133's base-body number.

Egocentric costs the policy heavily: 54%->6% travel, clone's own fit 52x worse (0.0116 vs 0.6040). Consistent with Q1: a single egocentric frame states the command at 0.329 vs allocentric's 0.779, so the clone has far less to fit and drifts faster. Internal gains from the egocentric direction were real but bought at this behavioural price.

Allocentric clone clears the F133 pass bar (54% > required 50%, upright throughout) -- first policy in the project to pass it, using no world model at all. F135's premise (cloning fails, leaving room for teacher-student) does not hold on this body with cloning done properly. Any future claim that a world model contributes must now beat 54% (not F135's 36%).

Unresolved: why the two allocentric clones differ (54% here vs 36% in F135) -- confounds body and training set, not separable from these runs.

Scripts: `scripts/diagnostics/clone_walk_test.py`, CoppeliaSim port 23000, students `wm/runs/students/insect_bc_{ego,allo_c08}.pt`.

---

### F170. Delta-state target is learnable (R2 0.81-0.85) but still cannot rank candidates -- the wall was never the representation target

`target_action_share.py` showed the reconstruction target buries the action (z reads body motion at ridge R2 0.359, embedding at 0.005). Fix tested: `StateHead` reading `FTM(e_t,z) - e_t`, predicting body motion directly, trained end-to-end (`scripts/run/com7_state_head.sh`, warm-started from `teacher_ego.pt`, `lambda_state 1.5`, 50 epochs). Offline this holds: R2 0.81-0.85 against retrained checkpoint (both frozen and joint FTM), matching the closed-form ridge oracle. Target-representation problem is solved offline.

Closed-loop test (F164's protocol unmodified): held-out `c08f09t09`, goal clip `hexapod_ep100` (`speed_c7.1`), n=40 branch points, `--repeat_control 4`, CoppeliaSim port 23000, four scorers on same branch points/perturbation draws/realizability checks.

| scorer | run1 | run2 | run3 |
|---|---|---|---|
| f179 (embedding rollout, baseline) | -0.40x | -0.37x | +0.31x |
| direct (state-blind control) | -0.03x | -0.68x | +0.48x |
| ridge (Delta-state lower bound) | +1.90x | -0.28x | +0.89x |
| state (trained network) | not run | -2.06x (wiring bug: fed 3-step-rolled delta, out of training distribution) | -0.44x |

No scorer clears noise floor reproducibly; `ridge` looked real once (1.90x) but didn't repeat. `state`, correctly wired, sits at ~coin-flip (50% win rate), small negative gap -- not a win, not a pathological failure. This matches the pre-registered pass-bar prediction: "state-head scorer ~= f179/coin even with R2 0.81-0.85 offline -> wall is between reconstruction accuracy and ranking, not the target."

Why: F166 already showed the wall is candidate separation being off-manifold (sigma 2.0 reaches 10.0% separation, near library's 12.8%, but ranks at 42%, below coin, because 33.3% of joints are outside dataset range). A target fix cannot rank candidates that don't separate on-manifold -- it doesn't address candidate generation.

Summary: target representation -- solved (Delta-state learnable, R2 0.81-0.85, holds under retrain). Candidate separation -- standing, untouched by this fix (a generation problem, not representation).

Scope note: does not reopen F166's sigma sweep (already tested, fails off-manifold not under-separation); candidate-generation problem remains open, not attempted here.

Checkpoints: `wm/runs/beh12_state/{best_state.pt,teacher_state.pt}`. Scripts: `scripts/diagnostics/planning/rank_fine_three_ways.py` (extended with `state` arm), CoppeliaSim port 23000, held-out `c08f09t09`.

---

### F171. Ceiling test resolves F170's fork: Delta-state scorer ranks well once candidates separate on-manifold

F170 left open whether `state`'s offline R2 (0.81-0.85) or its closed-loop coin-flip was the "real" number. `rank_fine_three_ways.py --candidates conditions` (12 recorded conditions from `--train_data`, in-distribution, instead of 32 Gaussian nudges) tests this; same four scorers, same 40 branch points, same goal, same protocol as F170, only candidate pool changed.

| scorer | perturb (F170) signal/floor | conditions (this run) signal/floor | win rate on conditions |
|---|---|---|---|
| f179 (embedding rollout) | -0.40x/-0.37x/+0.31x | 0.73x | 60% |
| direct (state-blind) | -0.03x/-0.68x/+0.48x | 4.64x | 70% |
| ridge (Delta-state lower bound) | +1.90x/-0.28x/+0.89x | 4.46x | 62% |
| state (trained network) | -0.44x (n=1, corrected) | 3.95x | 68% |

`state` clears the ceiling decisively (3.95x, same order as `direct`/`ridge`, none near noise floor). Offline R2 does convert to real, physically-verified ranking ability when candidates separate on-manifold. F170's fork resolved: the wall was candidate separation, not the scorer or the target.

Anomaly named, not smoothed: `f179` stays weak even here (0.73x, only scorer below 1x) -- its problem (the one underlying F164's original 47%) is not fully explained by candidate separation alone, since the other three scorers rank the same candidates cleanly.

`direct` (state-blind) is nominally best of the four here -- at this scale of separation (twelve whole, physically distinct behaviours) action alone determines outcome closely enough that reading current state barely helps. Not a contradiction of the Delta-state story; state-awareness should matter most where separation is small (the `perturb` regime), not where it's large (`conditions`).

Caveat: this is not comparable to F134's 85-90% (different chance baseline/protocol -- F134 scores family-match at chance 33%, this scores win-rate vs student's own action at chance 50%). The valid comparison is `perturb` vs `conditions` in the table above (same protocol, only candidate source changed).

Direction: approach (2) (world model as rollout predictor rather than candidate selector) not triggered -- selector paradigm works given separating candidates. Open problem unchanged from F170: need a candidate generator that is both fine-grained (unlike the 12-behaviour library) and on-manifold (unlike Gaussian noise past sigma 0.5); does not yet exist.

Scripts: `scripts/diagnostics/planning/rank_fine_three_ways.py --candidates conditions`. Checkpoints: `wm/runs/beh12_state/teacher_state.pt`. CoppeliaSim port 23000. Data: `data/egocentric/beh12_c10f10t10_ego_flat`, goal held-out `c08f09t09`.

---

### F172. F171 corrected: two error layers, not one -- the scorer's own error is magnitude discrimination, not direction

F171's 68% win rate is far below the ~100% a scorer with no error of its own should get once candidates are 4x above noise floor apart. `perturb` conflates candidate-similarity and scorer-inaccuracy; `conditions` isolates scorer-inaccuracy alone. Ground truth for "which condition is best" is each condition's own recorded whole-clip-average motion (same quantity `body_goal` uses), so no new simulator rollouts needed.

| scorer | exact accuracy | mean true-rank of pick (0=best,11=worst) | wrong picks in same family |
|---|---|---|---|
| f179 | 2% | 6.40 (indistinguishable from random) | 11/39 = 28% |
| direct | 20% | 2.67 | 31/32 = 97% |
| ridge | 22% | 2.48 | 29/31 = 94% |
| state | 28% | 2.33 | 25/29 = 86% |

`direct`/`ridge`/`state` almost never cross families (86-97% of mistakes stay within e.g. "speed"); when wrong, they still point near the true direction (cosine 0.86-0.87 vs true best), error is almost entirely magnitude (mean gap 0.71-0.76 standardised units). `f179` fails differently and worse: cosine 0.17 (near-orthogonal), no real signal, consistent with 6.40 mean-rank.

`state`'s state-awareness buys nothing measurable on the hard pairs: cosine/magnitude-gap (0.867/0.714) statistically indistinguishable from `direct` (0.860/0.755) and `ridge` (0.866/0.728) on the same pairs.

Reframing: Gaussian perturbation (F164/F166/F170) also only varies magnitude within one behaviour, never crosses families. The wall hit by perturbation testing and the wall isolated here may be the same underlying limitation: this model family discriminates behaviour categories well but discriminates magnitude within a category poorly. Candidate generation (Workstream 2) and scorer accuracy (Workstream 1) may not be separable fixes if both are downstream of the same magnitude-discrimination limit.

Scripts: `scripts/diagnostics/planning/condition_confusion.py`, checkpoint `wm/runs/beh12_state/teacher_state.pt`, CoppeliaSim port 23000, library/ground truth `data/egocentric/beh12_c10f10t10_ego_flat`, goal held-out `c08f09t09`. Diagnosis only, no training.

---

### F173. Two architecture fixes for the action lever both land flat; rollout supervision helps prediction but not action-use

> Correction (F175): the "frozen-encoder ceiling" inference below is overturned, not just weakened. F175 tested the claim directly (untrained kNN probe on raw frozen embedding delta) and found the fine speed signal clearly present (49-51% four-way accuracy vs 25% chance, R2 +0.403). The two architecture fixes below did land flat, but not because of an encoder-geometry ceiling -- the wall is downstream (calibration/data-sparsity, per F176 onward). Do not cite the ceiling framing below as explanation.

Action-lever metric (real-z vs mean-z cosine on FTM's predicted direction) stayed flat-to-decaying with horizon: 0.054/0.062/0.042/0.026 at k=1/2/5/10 (`action_lever_vs_horizon.py` on `teacher_ego.pt`). Two cheap 5-epoch warm-started de-risks tried before a full pretrain:

Fix 1: auto-regressive 2-step rollout supervision (`K=2`, one teacher-forced + one self-fed step, unweighted sum with one-step loss, per Demo-JEPA's own recipe read from code). Implemented as `lambda_rollout` in `wm/train.py`, needs third view (`MultiEmbodimentPairs.rollout_k`).

Fix 2: wider injection surface -- `z_tokens` 1->4, `ftm_blocks` 8->12 ("deep injection"), scaled-down literal version of Demo-JEPA's 24-layer design. Needed `--allow_shape_mismatch` to warm-start everything except the resized `ftm.latent_proj`.

| k | baseline | rollout-trained | deep-injection |
|---|---|---|---|
| 1 | 0.054 | 0.060 | 0.059 |
| 2 | 0.062 | 0.060 | 0.061 |
| 5 | 0.042 | 0.042 | 0.044 |
| 10 | 0.026 | 0.026 | 0.024 |

Both flat within noise at every horizon -- neither recipe nor architecture moves the lever.

Rollout term did help something else: auto-regressive rollout accuracy (model-MSE-over-copy-forward ratio) improved at targeted horizons: 0.517->0.511 at k=1 (flat, as expected/untargeted), 0.496->0.465 at k=3, 0.518->0.471 at k=5 (`rollout_horizon_accuracy.py`). Model got better at predicting what happens next, not at using the action to do so -- separable questions.

Scripts: `wm/runs/derisk_rollout5`, `wm/runs/derisk_deepinject5`, `scripts/diagnostics/objective_experiments/{action_lever_vs_horizon,rollout_horizon_accuracy}.py`, held-out `c08f09t09`. Both training runs local, 5 epochs (de-risk, not full pretrain).

---

### F174. Reference body's turn sign was not reproducible from scratch on the current scene; caught before training on wrong data

While collecting 5x more clips of the training body's 12 conditions (`c10f10t10`), `speed` and `side` families verified clean; `turn` did not -- yaw sign consistently negative (-0.007 to -0.081) against the established positive reference (+0.007 to +0.077, F108/F109), confirmed still positive on the existing 48-clip dataset via the same measurement code re-run.

Ruled out: (1) comparison-path artifact -- same `--separability` function reads positive on old data, negative on new, same session; (2) session-level physics noise (F95) -- fresh `CoppeliaSim_b` instance reproduced identical negative sign to two decimals (-0.0808 vs -0.081), so deterministic not noise; (3) egocentric-specific artifact -- fresh allocentric collection (no `--view egocentric`) also reads negative (-0.086).

Fix: `--spin_sign -1` (F108's mechanism, previously thought needed only for non-reference bodies `c08f09t09`/`b1` to match `c10f10t10`), applied to `c10f10t10` itself, reproduces positive reference almost exactly (+0.070 vs +0.073 to +0.077). Root cause not identified -- something in the scene file or CPG code changed since F108/F109 so the identical nominal command now produces the opposite sign, regardless of session or camera mode.

Consequence: `c10f10t10` (the reference body) now needs `--spin_sign` re-verified before being trusted, same as non-reference bodies already require via `--verify`/`--separability`. Corrected 240-clip set (`data/egocentric/beh12_c10f10t10_more_ego_flat`) passes `--separability` cleanly on all 12 conditions, with only the pre-existing `speed_c8.8` vs `speed_c8.15` quirk (already present/accepted in the original 48-clip set).

Note: this data-sparsity fix does not carry over to B1 the same way (see project memory `b1-mujoco-deterministic`).

Scripts: `scripts/dataset/collect_beh12.py --spin_sign -1`, `CoppeliaSim_b` port 23001, scene `medauroidea_c10f10t10.ttt`.

---

### F175. Fine speed signal IS present in the raw frozen latent -- F173's encoder-ceiling framing was wrong

Direct test (vs F173's indirect inference): raw frozen embedding delta `e_t+1 - e_t`, no ITM/FTM/trained head, for four speed conditions (forward 0.124/0.152/0.186/0.192, `--separability`-verified monotonic). Off-the-shelf probes:

Four-way classification (chance 25%): kNN (k=5, k=15) on raw 1408-D and 50-component PCA, plus MLP classifier -- all land at 49-51% held-out, twice chance, consistent across variants. Continuous regression on achieved forward speed: kNN k=15 gives R2 +0.403, Spearman rho +0.560. Confusion concentrates on adjacent speeds (`c5.8`<->`c7.1`: 206 errors) vs distant ones (`c5.8`<->`c8.8`: 80 errors) -- same shape F172 found in the trained pipeline, reproduced here by an untrained kNN.

MLP regressor was noisy (R2 -0.932, unstable with ~3800 points in 1408-D, no HP search) but rank correlation still positive (+0.213); other three probes already agree without it.

Ceiling check: an oracle given the exact TRUE per-transition forward speed (no embedding/model) classifies at only 39.0%, R2 ceiling 0.289 -- both below the embedding kNN's 50.3%/0.403. Instantaneous speed genuinely overlaps between adjacent conditions per 50ms step (gait-phase oscillation), so no probe reaches 100% on a per-transition read. The embedding-based probe beating the exact-value oracle means the raw latent carries extra condition-specific structure (plausibly gait posture/rhythm) beyond instantaneous speed.

Averaging over a whole clip (not one transition) with the same true-speed oracle: accuracy 39.0%->83.8%, R2 0.289->0.987. `speed_c5.8`/`speed_c7.1` become perfectly separable (20/20); every remaining error is `speed_c8.15` vs `speed_c8.8` (true clip-averaged means 0.1842 vs 0.1862, gap smaller than clip-to-clip spread) -- not fixable by averaging/more data/better probe, those two conditions are genuinely near-identical achieved speed.

Recalibrates F171's 68%: ranking test operates in the per-transition noise regime (39% oracle), not clip-averaged (83.8%); 68% is well above the 39% ceiling that perfect instantaneous-physics knowledge alone would give.

Confirms F173's measurements stand but not its inference: encoder has the signal (found by an untrained 5-line kNN). Wall is the trained pipeline not extracting/calibrating what's already present -- consistent with F172's data-sparsity diagnosis, not an encoder-level fix. The com7 retrain on 5x more clips (F174's corrected dataset) is the right lever.

Scripts: `scripts/diagnostics/objective_experiments/embedding_speed_ceiling.py`, data `data/egocentric/beh12_c10f10t10_more_ego_flat`, held-out by clip (5 of 20 per condition), raw frozen `VJEPA2FrameEncoder`, no checkpoint. Diagnosis only.

---

### F176. Neither the ITM bottleneck nor the projector loses signal -- wall fully relocated to calibration in the state head

Same probes/data as F175, checking the next two links: `real z = ITM(e_t, e_t+1)` from true transitions, and `projected z = proj(action)` from the recorded action alone (what ranking actually consumes at test time).

| stage | best accuracy (chance 25%) | R2 | rho |
|---|---|---|---|
| raw embedding delta (F175) | 50.3% | 0.403 | 0.560 |
| real z, ITM on true transitions | 80.0% | 0.781 | 0.836 |
| projected z, action alone | 97.7% | 0.791 | 0.879 |

All three carry the signal clearly. `real_z` scoring well above raw delta rules out the ITM's 64-D bottleneck as an information loss (it concentrates signal, doesn't discard it). `proj_z`'s 97.7% is near-tautological, not a strong result -- it's classified from the recorded action command itself, and different speed conditions use systematically different `--cycles` values, making this an easier task than reading a noisy observed outcome; do not read it as "projector preserves more signal than ITM." What it does support: `proj_z` doesn't collapse distinct actions into indistinguishable representations.

Not tested here (the honest remaining gap): calibration -- whether `proj(action)` fed through the FTM/state-head path produces predictions correctly ORDERED against physical reality (that's F171/F172's ranking test, 68%, now read against the 39% per-transition noise ceiling as real but imperfect).

Candidate list closed out: projector and ITM bottleneck downgraded from candidate bottlenecks to ruled-out. Remaining: state head's own readout/calibration, and data sparsity (F172) -- same question from two ends (readout may be imprecisely calibrated because fit on 4-20 clips/condition). The com7 recollection (F174's 240-clip set) tests both at once: if it lifts the 68%, calibration was data-limited; if not, the state head's own readout (architecture/loss) is the next, narrower target.

Scripts: `scripts/diagnostics/objective_experiments/pipeline_speed_ceiling.py`, checkpoint `wm/runs/beh12_ego/teacher_ego.pt`, same data/split as F175. Diagnosis only.

---

### F177. Combining pooled delta into the state head's readout makes fine-magnitude prediction worse, not better -- but the real retrain does not confirm the fix

State head forward pass is `head(pool(delta - offset) + z_proj(z))` (`wm/models/state_head.py`). Offline kNN probe (same held-out-by-clip split as F175/F176), predicting forward speed:

| feature | R2 | rho |
|---|---|---|
| z alone | +0.781 | +0.836 |
| mean-pooled delta, raw | +0.403 | +0.560 |
| mean-pooled delta, offset-subtracted | +0.403 | +0.560 |
| std-pooled delta | +0.275 | +0.365 |
| z + mean-pooled delta | +0.625 | +0.733 |
| z + std-pooled delta | +0.556 | +0.627 |

`z` alone (0.781) beats `z` + pooled delta (0.625) by a wide margin -- adding delta dilutes what `z` already has. Explains why trained state head (F172: 68% win rate, low exact-pair accuracy) underperforms what `z` alone supports (F176: 80%, R2 0.78).

Offset-subtraction confirmed a true no-op for fine-grained regression too (raw vs offset-subtracted identical to 3 decimals, 0.403 both) -- matches F172's "cosmetic" finding for coarse accuracy. `std_pool` (spatial variance) does not rescue it: worse than mean-pool alone (0.275 vs 0.403) and worse combined with z (0.556 vs 0.625) -- extra signal isn't in cross-token variance (doesn't rule out learned/attention pooling).

Confirmed on the deployment-relevant `proj_z` (not just oracle `real_z`): `proj_z` alone R2 +0.791; combined with offset-corrected mean-pooled delta, R2 drops to +0.537 -- larger relative drop than oracle's (0.781->0.625). Fix (state head input = z alone, not z+pool(delta)) confirmed on the actually-deployed path.

Real retrain result -- reported as null: `state_use_delta=False` built into `StateHead`/`wm.train`, full 50-epoch retrain on identical `beh12_state` dataset, same warm start. `condition_confusion.py`, same protocol as F172:

| metric | z+delta (control, F172) | z-only (this retrain) |
|---|---|---|
| exact accuracy | 28% | 22% |
| mean true-rank (0=best) | 2.33 | 2.60 |
| cosine(picked, true-best) | 0.867 | 0.864 |
| mean magnitude gap | 0.714 | 0.770 |

Slightly worse on every metric. Gaps plausibly noise at n=40 (same order as `ridge`'s run-to-run bounce: 60%/50%/62% across F170/F171/F176), but not the decisive win the offline kNN predicted (R2 0.781 vs 0.625) -- a static kNN on frozen features can't adapt like a trained MLP, which may already partially down-weight the noisy delta term during backprop. Conclusion: the real retrain does not confirm the fix works. Not confirmed-worse either -- genuinely inconclusive.

Scripts: `scripts/diagnostics/objective_experiments/state_head_readout_ablation.py`, `scripts/diagnostics/objective_experiments/proj_z_delta_ablation.py`. Checkpoints: `wm/runs/beh12_ego/teacher_ego.pt`, offset from `wm/runs/beh12_state/best_state.pt`, retrain outputs `wm/runs/beh12_state_zonly/{best_state.pt,teacher_zonly.pt}` trained on `data/egocentric/beh12_c10f10t10_ego_flat` (not the enlarged "more" set). CoppeliaSim port 23000. Diagnosis + one real retrain.

---

### F178. Six independent fixes for fine-grained ranking all null, including an exact counterfactual-target test on B1 -- structural wall confirmed

Merges four steps of one diagnostic arc converging on the same ceiling.

**1. Data-sparsity retrain (5x hexapod data), mixed result, not a confirmation.** `beh12_state_more` (5x data, same recipe) finished with val loss statistically same as control (4.4992/1.3516 vs 4.5110/1.3596). `condition_confusion.py` ranking test:

| scorer | acc (48-clip control) | acc (5x data) | true-rank (control) | true-rank (5x) |
|---|---|---|---|---|
| f179 | 2% | 18% | 6.40 | 4.70 |
| direct | 20% | 10% | 2.67 | 1.65 |
| ridge | 22% | 15% | 2.48 | 2.15 |
| state | 28% | 8% | 2.33 | 1.62 |

`state` (the scorer being tested) got worse on exact accuracy (28%->8%) while mean true-rank improved for every scorer -- not a clean win. Confound: ground truth's representative clip per condition differs between the 48-clip and 240-clip dirs (`cand[c]` picks first clip found), so true-distance values shifted (e.g. `speed_c5.8`: 0.390 vs 0.3613). Does not confirm F176's data-sparsity branch either way.
Scripts: `scripts/diagnostics/planning/condition_confusion.py`, checkpoint `wm/runs/beh12_state_more/teacher_more.pt`, `--train_data data/egocentric/beh12_c10f10t10_more_ego_flat`, CoppeliaSim port 23000.

**2. Smoothing-window length and clip-average aux loss for state-head readout, both null.** Longer smoothing window makes kNN regression (e_t+1-e_t -> forward speed) worse, not better (scope mismatch: single-transition feature vs increasingly clip-level target):

| window | R2 | rho |
|---|---|---|
| 1.0s (current) | +0.362 | +0.525 |
| 2.0s | +0.270 | +0.486 |
| 3.0s | +0.246 | +0.482 |
| 3.3s (~whole clip) | +0.254 | +0.493 |

Auxiliary loss (match trained state_model's per-timestep predictions averaged over a clip to the clip's true average speed, added on top of per-timestep loss, itm/ftm/projector frozen), `condition_confusion.py`:

| | accuracy | true-rank (0=best) |
|---|---|---|
| control (z+delta, no aux) | 28% | 2.33 |
| + aux clip-average loss | 22% | 2.67 |

Null, leaning negative; direction/magnitude breakdown (cosine 0.868/gap 0.761) matches control (0.867/0.714) -- no gain in fine-magnitude discrimination.
Scripts: `scripts/diagnostics/shared_body_target/screen_behaviour_channels.py --features frozen --window {1,2,3}`; aux-loss checkpoint `wm/runs/beh12_state/teacher_state_auxavg_derisk.pt`, scored on `data/egocentric/beh12_c10f10t10_ego_flat`.

**3. Approximate counterfactual-target fine-tune (UWM-JEPA-motivated), trustworthy null.** Correction of an earlier overclaim: UWM-JEPA (arXiv 2605.25313)'s binary hidden-velocity task on toy oscillator/CartPole explicitly does not claim to extend to unstructured visual prediction. Its load-bearing result (Sec 4): under teacher-forced training, action term collapses near-zero (||H1||/||H0||~0.03) regardless of latent geometry; fix is training against counterfactual targets (a different sampled action rolled through the real simulator), not a unitary predictor (Sec 6 showed no significant gain there). Our pipeline is textbook teacher-forced (`z=itm(view1_t,view1_next)` always from the true observed pair) -- exactly what Sec 4 diagnoses.

Cheap approximate test: paired clip A's early frame with clip B's (different condition) action and B's own next frame as approximate counterfactual target, 1,440 pairs, FTM fine-tuned against these targets alone (replacing teacher-forced loss). Action-lever metric (`action_lever_vs_horizon.py`, k=1):

| | real z | mean z | lever |
|---|---|---|---|
| baseline (`teacher_ego.pt`) | 0.687 | 0.633 | 0.054 |
| approximate counterfactual fine-tune | 0.550 | 0.543 | 0.007 |

Real null (both real-z and mean-z fell, not just the gap) -- fine-tune degraded general directional accuracy rather than just failing to add signal. Pre-registered: this null does not license further tuning of the approximate version, only a properly-controlled exact test.
Checkpoint: `wm/runs/beh12_ego/teacher_ego_cf_derisk.pt`, scored via `action_lever_vs_horizon.py --ks 1` on `data/egocentric/beh12_c08f09t09_ego_flat`.

**4. Exact counterfactual-target test on B1/MuJoCo (deterministic state-resume), null, below pre-registered bar.** B1/MuJoCo policy is fast/deterministic, enabling exact state-resume (save/restore full physics state mid-rollout, continue under a different command) -- built via `--save_state_at`/`--load_state` in `sim/collect/rollout_b1_mujoco.py`. Initial validation showed large apparent divergence (0.19 rad joint) between resumed and original continuation under the same command; traced to an off-by-one in the comparison script (not save/restore code) -- correctly aligned, residual is small and decelerating (~0.007m over 36 steps), consistent with solver reconvergence, not a bug. State-resume validated.

Pre-registered bar: action-lever metric is exactly deterministic on B1 (bit-identical across 3 seeds), so a magnitude bar was locked instead of a noise threshold: baseline L_b1=0.043 (`teacher_ego.pt`, `--embodiment b1`, 48-clip `beh12_b1_ego_flat`, k=1); success = counterfactual-trained lever > 0.086 (2x baseline).

24 exact counterfactual examples: all 12 B1 conditions branched at a validated settled-gait step, each continued under two counterfactual commands from a different behaviour family than the source; z from `proj()` on the real recorded action actually taken, rendered via CoppeliaSim as passive deterministic replay.

| | real z | mean z | lever |
|---|---|---|---|
| L_b1 baseline (teacher-forced) | 0.683 | 0.640 | 0.043 |
| exact counterfactual fine-tune | 0.511 | 0.486 | 0.025 |

Below the pre-registered bar and below baseline. Per locked discipline, this is the conclusion (not a prompt for more data; 24 was the agreed scope). Stage 2 (CoppeliaSim exact counterfactuals for hexapod) is gated on this and does not proceed.
Runs: `sim/collect/rollout_b1_mujoco.py` (new flags), CoppeliaSim port 23000, `sim/env/b1_flat.ttt`; checkpoint `wm/runs/beh12_ego/teacher_ego_b1_exact_cf.pt`, scored by `action_lever_vs_horizon.py --embodiment b1 --ks 1` on `data/egocentric/beh12_b1_ego_flat`.

**Overall conclusion.** Across this session's arc, six independent, mechanistically distinct fixes -- training recipe (F173 rollout supervision), architecture (F173 deeper injection), data volume (this entry, 5x hexapod), readout design (F177 z-alone, this entry's aux clip-average loss), and the counterfactual-target mechanism fix at both approximate and exact strength (this entry) -- all null. This is a characterized structural limitation, not "nothing worked": teacher-forced training removes the incentive for a video world model to depend on the action at all (confirmed here to apply beyond UWM-JEPA's toy benchmark), and the obvious remedy (counterfactual targets) does not transfer to video-scale locomotion prediction at this data budget. Coarse behaviour-family discrimination works; fine within-family magnitude discrimination does not, and every cheap-to-moderate lever tried has been ruled out.

---

### F179. Dreamer-style imagination-RL on the frozen FTM for B1: original arc confounded by a non-converging critic; full critic-stabilization ladder and PPO both fail on an FTM-side long-horizon error; escalation ladder exhausted

**Original attempt (retracted).** Actor-critic-in-imagination (Hafner Dreamer mechanics) against the frozen FTM, goal: beat a B1 behaviour-cloned baseline (92% of D_real, itself FAIL by F133's rule) in real closed loop. Four steps tried in sequence (frozen FTM no correction; BRAC/BCQ action regularization; MOReL-style joint (state,action) uncertainty penalty; FTM re-grounding on real B1 transitions at 6/24/48 cycles) each appeared to show a distinct FTM-side cause (exploitation, wrong regularization target, weak gradient, insufficient re-grounding cadence). All four readings are retracted: an isolation test (actor-critic alone vs the plain pretrained FTM, frozen, never re-grounded, 5,000 iterations -- 15-16x any prior budget) showed no real convergence: realized_return first-quarter -37.3 -> last-quarter -39.6, one transient plateau (~-22 to -25, iter ~1000-1800) that destabilized (critic_loss >3000, return crashed to -94) before settling at a no-better fixed point. Critic's value0 diverged (-82 -> -421, 5x) via unresolved bootstrap-target divergence (same failure Stage A0 first showed, never actually fixed by the EMA target-critic). All prior interpretations (steps 1-4) used shorter budgets (300-2,400 iterations) than this isolation test and are confounded by the same broken optimizer, not confirmed findings.

One finding survives independent of the confound: the coarse controller-signal result (FTM is a usable coarse-level control signal -- rollout accuracy 0.90-0.97 correlation to k=10, ranking rho 0.84-0.93 both bodies, gradient-usefulness 3/3 on B1) was measured offline with no actor-critic training and holds.

**Critic-stabilization escalation ladder (pre-registered order), same isolation test / 5,000-iteration budget / pass bar throughout: relative difference `|value0 - MC_discounted_return(800-step)| / |MC_return| < 0.25`.**

| rung | mechanism | relative difference | verdict |
|---|---|---|---|
| (1) | EMA target critic alone | unbounded value drift, no MC check needed | fail |
| (2) | + symlog critic regression + percentile return normalization | 0.464 | fail (short-horizon realized_return converges cleanly, -37.5->-9.0/-9.2 both replicate runs, but critic underestimates long-horizon badness) |
| (3) | (2) with EMA replaced by periodic hard target updates (every 100 iters) | 0.624 | fail, worse than (2) |
| (4) | (2)'s target mechanism + two-hot distributional critic (DreamerV3's actual key mechanism, NUM_BINS=255 symlog bins, cross-entropy vs two-hot target) | 0.272 | fail, best of the four but still over 0.25 |

Rung (4)'s policy tested in real closed loop (MuJoCo+CoppeliaSim, same B1 clone baseline): stayed upright the full 66-step window (never fell) but covered only 19% of D_real (0.227m vs 1.21m reference) vs clone's 92% and the 50% pass bar. Imagination/reality gap +4.75 (smaller than original +16.6) but correlation ~zero (-0.071). Distinct failure mode from the original (confident-but-wrong exploitation): here the policy is safe but frozen, barely committing to gait. Confirms the 0.272 MC-check gap is a real blocker, not proxy artifact. Per pre-registration: proceed to PPO.

**PPO isolation test (Stage P0), same protocol/bar, no Dreamer tricks (plain scalar critic, standard rollout+GAE(lambda)+clipped surrogate, tanh-squashed Gaussian with Jacobian correction), trained purely in imagination against the same frozen FTM (no backprop through FTM, 300 updates x 100-step rollouts, 16 envs, 10 PPO epochs/update).** realized_return improved (-42.2->-35.4) but did not cleanly converge (destabilization event, update 210-270, spike to -56.5). MC check: value0=-747.4 vs MC return=-583.6, relative difference 0.281 -- fails, nearly identical to rung (4)'s 0.272. Two structurally different algorithms (4 Dreamer variants + PPO) converge to ~0.27-0.28 relative discrepancy -> evidence the wall is FTM-side (compounding prediction error over imagined horizon), not algorithm-side, matching the documented Koopman Dreamer (2607.19719) failure mode for legged locomotion specifically. Per pre-registered rule: P0 failing means Stage P1 (real closed loop) is not run without revisiting the diagnosis.

**GAMMA-shortening test (direct test of the horizon-compounding prediction), rules out the hopeful reading.** GAMMA 0.99->0.95 (effective horizon ~100->~20 steps), same PPO mechanics, T_MC scaled to 200, plus an action-variation myopia diagnostic. Result: gap did NOT shrink -- relative difference 0.306, slightly worse than GAMMA=0.99's 0.281, despite 5x shorter horizon. realized_return converged cleanly this run (-38.6->-18.6, std 7.2->0.8) but action variation collapsed 0.110->0.0092 (~92% drop, past the pre-registered myopia threshold) -- the policy froze into a static, trivially-predictable stance rather than genuinely sustaining locomotion; the clean convergence was a degenerate solution, not evidence the underlying problem got easier.

**Final reading.** The FTM's error is not purely horizon-compounding -- it is wrong even at a much shorter horizon. Rules out "shorten the horizon" as a real fix. Per the pre-registered ladder, the required fix is now FTM-side: a Koopman-style spectral constraint on the FTM's latent dynamics (2607.19719's proposed fix for this exact locomotion failure mode), or direct retraining/re-grounding of short-horizon prediction quality -- not further horizon/algorithm/critic-stabilization attempts. Boundary set for this project: the FTM is confirmed strong at coarse/short-horizon control (rollout accuracy to k=10, ranking rho 0.85+) but not accurate enough, even short-horizon, to support imagination-based long-horizon value estimation for control. This is a mapped limitation of this specific FTM, not of any RL algorithm tried against it.

Runs/checkpoints (representative): `wm/runs/beh12_state/stage_a0_actor_critic*.pt`, `isolation_test_v{3,4,5}_final.pt`, `ppo_p0_final.pt`, `ppo_p0_gamma0.95_final.pt`; logs `results/wm/closed_loop/{a1_diagnostic*,r0_regrounding_curve*,rl_loop_isolation_test*,v5_twohot_closed_loop,ppo_p0_isolation,ppo_p0_gamma0.95}.npz`. MuJoCo+CoppeliaSim B1 harness, port 23000, `sim/env/b1_flat.ttt`, demo `b1_ep0.npz`.

---

### F180. Long diagnostic arc on FTM/state-head action-insensitivity: eight causes ruled out (loss weight, prediction target, four architectures, input representation), root cause isolated as joint-training gradient competition on z, stop-gradient fixes the action-lever (~9x) but ranking parity is not reached

**1. Gradient-share check: rules out "reconstruction starves dynamics."** F140 already found reconstruction's real gradient share was only 22-41% (not 99%), but never included `lambda_state`. Extended `loss_gradient_balance.py` to add the `state` term, measured on `wm/runs/beh12_state/teacher_state.pt`:

| term | lambda | hexapod share | B1 share |
|---|---|---|---|
| recon | 1.00 | 25.5% | 8.6% |
| motion | 1.00 | 17.7% | 29.7% |
| body | 0.50 | 11.7% | 13.9% |
| state | 1.50 | 45.1% | 47.8% |

State/Froude term already has the largest gradient share on both bodies. No starved-gradient problem exists; a loss-reweighting retrain would not help. Converges with the independently-closed fine-magnitude calibration wall (F175-F177): more signal into the representation was never the missing piece.
Script: `scripts/diagnostics/forward_model/loss_gradient_balance_state.py --ckpt wm/runs/beh12_state/teacher_state.pt --embodiment {hexapod,b1} --batch 8`.

**2. Froude-forecasting FTM premise check: fails.** Idea: predict Froude directly instead of next embedding, since F173's action-lever (+0.055 embedding-space) might be an artifact of an appearance-dominated target. Tested via the existing trained state head, no retrain: embedding-space gap replicates (+0.054 hexapod, +0.034 B1, confirming measurement is sound) but Froude-space gap is worse -- hexapod **-0.051** (real action predicts direction *worse* than action-blind), B1 +0.042 (still below embedding-space reference). Premise fails; do not build.

**3. Collapse z (64-D) to ground-truth Froude (3-D) entirely: fails harder.** Fit two `MotionDecoder` copies (clean-split checkpoint, frozen ITM), one on real z, one on ground-truth `body_motion[t]`:

| | hexapod held-out R2 | B1 held-out R2 | hexapod train R2 | B1 train R2 |
|---|---|---|---|---|
| decoder(x_t, real z) | +0.655 | +0.274 | +0.767 | +0.998 |
| decoder(x_t, ground-truth Froude) | +0.023 | +0.006 | +0.942 | +0.998 |

Froude-conditioned decoder memorizes train (only 24 clips/body) but collapses to mean-baseline held-out (R2~0) -- a 3-D net-outcome summary can't disambiguate different gaits sharing one Froude value. Ruled out.
Script: `froude_bottleneck_action_ceiling.py`. Checkpoint: `wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt`.

**4. Augment FTM with ground-truth Froude alongside z: no effect.** froze ITM, trained two fresh FTMs (z-only 64-D vs z+ground-truth-Froude 67-D) on identical next-embedding-MSE objective/data:

| | real | mean (action-blind) | gap |
|---|---|---|---|
| baseline (z only) | 0.5441 | 0.5014 | +0.0428 |
| augmented (z + Froude) | 0.5441 | 0.5019 | +0.0422 |

Gap difference -0.0006 = noise. Rules out "extraction burden" reading -- three different ways of exposing Froude to the forward model (retarget, replace, augment) all null.
Script: `ftm_froude_conditioning_check.py`. Checkpoint: `wm/runs/beh12_hinge_cleansplit/best.pt`.

**5. Architecture category (never tried before this): stateless-vs-2-frame-context proxy, no change.** FTM is confirmed a stateless single-step predictor (no recurrence). Cheap proxy: fed two concatenated frames (512 tokens instead of 256, no retraining):

| | single-frame | two-frame context |
|---|---|---|
| hexapod | -0.051 | -0.056 |
| B1 | +0.042 | +0.044 |

No meaningful change (approximate/OOD proxy, doesn't rule out a real trained RSSM). Real RSSM build scoped as comparable in size to the ActSWM rebuild (F137-157) -- not undertaken without a cheap kill-gate first.

**6. RSSM kill-gate (Stage R0): GRU with real recurrent state, fails.** `h_t=GRUCell([pool(e_t-1),action_t-1],h_t-1)`, MLP decoder to Froude delta, teacher-forced, 2000 iterations, B1 only, 48-clip set. Pre-registered bar: gap > 2x stateless reference (0.055), i.e. >0.110. A methodology catch: initial sanity check (+0.339 vs +0.042 reference, 8x off) traced to computing on only 9 held-out clips instead of the full 48; recomputed correctly and reproduced exactly (+0.042 vs +0.042).

Gate result: real-action median cosine 0.689, mean-action 0.652, **gap +0.036** -- below even the raw +0.055 stateless reference, well under the 0.110 bar. FAIL -- full RSSM (posterior/prior/KL, multi-body pipeline) killed here, not after multi-day build. Evidence against this minimal deterministic construction specifically, not proof no recurrent architecture could help.
Checkpoint: `wm/runs/beh12_state/rssm_r0_killgate.pt`.

**7. Positive-control audit of the measurement path: healthy.** Given multiple nulls in a row, audited for measurement bugs: (a) mean_z is not an artificially easy baseline (comparable scale to inter-sample distance); (b) proj(a) doesn't collapse different actions (between-family >> within-family distance); (c) readout (state head) responds substantially to z swaps across behaviour families; (d) positive control -- matched z vs maximally mismatched z (different behaviour family): gap +0.121, clears even the 0.110 RSSM bar. Pipeline discriminates strongly between categorically different actions but specifically not between real-recorded and generic/averaged actions of the same family -- the nulls are real, not measurement artifacts. Reconciles with the independently-confirmed coarse controller signal (rollout accuracy 0.90-0.97, ranking rho 0.85+).
Script: `positive_control_audit.py`.

**8. MC-check critic-vs-policy-drift disentangling (RL arc, F179 follow-up): critic-learning failure confirmed genuine, unconfounded.** Prior MC-checks compared critic against FTM rollout under the trained (possibly off-manifold) policy, conflating critic quality with policy drift. Tested on-distribution (real recorded actions, never policy's own choices) at GAMMA=0.9 (effective horizon ~10, the validated coarse-good zone):

| evaluation | relative difference |
|---|---|
| actor's own (possibly off-manifold) rollout | 0.318 |
| on-distribution, clip's own matched goal (later found confounded: low-regret pairing) | 8.655 |
| on-distribution, random goal matching training distribution | **0.576** |

0.576 still fails the 0.25 bar and is worse than the actor's own rollout -- rules out both policy-drift and goal-pairing-artifact explanations. Confirms (with #7's audit) the RL critic-learning failure is genuine: TD-bootstrapped value learning does not converge on this reward structure even under best-case on-distribution, coarse-good-resolution conditions.
Scripts: `ondist_critic_disentangle.py`, `ondist_randomgoal_check.py`. Checkpoint: `wm/runs/beh12_state/ondist_disentangle_final.pt`.

**9. Ground-truth flatness test (Test 1): real signal exists in the data, not a flat task.** Pure ground-truth check (no model): within each of B1's 12 conditions, does a clip's deviation from the condition's mean action predict its deviation from the condition's mean Froude outcome? corr = +0.432, permutation p=0.0020 (n=20000), ridge leave-one-condition-out R2 = 0.204 (unregularized 12-D OLS overfit: in-sample R2=0.797, LOO R2=-0.621, discarded). Verdict: the model's +0.042 single-step action-lever gap is a model failure to capture something present in the data. Clears the gate for sequence-context architecture testing.

Part B: Froude is not a clean per-condition constant -- within-clip CV on forward speed 0.131±0.258, start-vs-end drift 0.184±0.373 (comparable to within-clip noise, non-negligible vs across-condition CV 0.719). Favors delta-Froude or history-conditioned Froude as target over absolute regression.
Script: `scripts/diagnostics/objective_experiments/ground_truth_action_flatness.py`. Data: `data/egocentric/beh12_b1_ego_flat`.

**10. Sequence-context kill-gates (Test 2): three architectures, all fail; spatial+recurrent design comes closest.** First two attempts confounded (attempt 1: windowed self-attention over 4 frames, spatial but non-recurrent, doesn't test recurrence; attempt 2: GRU with learned attention-pool, still pools before temporal integration, same confound as R0). Third design (ConvGRU: 16x16xC spatial hidden state via 3x3 convs, pooled only at final readout) genuinely combines spatial-preservation + recurrence:

| kill-gate | spatial? | recurrent? | gap |
|---|---|---|---|
| R0: pooled mean -> GRU | no | yes | +0.036 |
| attempt 2: attention-pool -> GRU | no | yes | +0.058 |
| attempt 1: full grid, self-attention, no recurrence | yes | no | +0.048 |
| ConvGRU: spatial hidden state | yes | yes | +0.069 |
| stateless single-step FTM (reference) | -- | -- | +0.042 |

All fail the 0.110 bar. ConvGRU is closest (0.041 short) and the only variant beating the stateless reference by a clear margin (1.6x) -- corrected from an earlier inverted reading that called it the worst. Honest reading: "not demonstrated at this budget" (2000-iteration probes), not "ruled out." Open: whether a genuinely full-scale trained sequence model (not a short probe) would do better -- untested.
Scripts: `sequence_context_killgate.py`, `spatial_recurrent_killgate.py`. Checkpoints: `wm/runs/beh12_state/{seq_spatial_gru_killgate,spatial_recurrent_killgate}.pt`.

**11. Input-representation sweep: signal is present in the encoder and in z; delta representation discards most of it.** Raw frozen `concat(e_t,e_next)` regressed against real delta-Froude: rho +0.35-0.45; the `delta=e_next-e_t` representation the state head is built on: rho only +0.02-0.22 -- encoder has the signal, delta discards most of it. Six-equation representation sweep (median rho): `z=ITM(e_t,e_next)` **+0.535** (winner) > bilinear +0.413 > concat+delta +0.406 > concat +0.405 > trained cross-attention probe +0.392 > delta baseline +0.215. State head's `pool(delta)+z_proj(z)` treats z as additive, not primary, despite z being the strongest signal.

Reconstruction-to-control substitution: `proj(action)` (control-time, no future frame needed) retains 80% of ITM-z's rho (+0.429 vs +0.535); per-channel lateral 97%, yaw 80%, forward only 39% (pre-registered caveat).

Does a z-primary head work when trained end to end? F177's `state_use_delta=false` checkpoint (`teacher_zonly.pt`, already existed) tested on the action-lever (not previously measured):

| z source | action-lever gap | bar 0.110 |
|---|---|---|
| z=ITM(e_t,e_next) | +0.045 | FAIL |
| z=proj(action) | +0.099 | FAIL |

Both fail; per-channel win rate (chance 50%): lateral 51-54% (neutral), forward 3-5% and yaw 2-15% (far below chance, actively worse than random). Conclusion: signal is present at every stage checked offline but does not survive being trained end-to-end into a predictor -- a joint-training problem, confirming F177's original finding on the metric that matters (action-lever, not just ranking accuracy).
Scripts: `embedding_transition_ceiling.py`, `embedding_representation_sweep.py`, `proj_action_ceiling_check.py`, `zonly_action_lever_check.py`. Checkpoint: `wm/runs/beh12_state_zonly/teacher_zonly.pt`.

**12. Isolated-head test: root cause confirmed as joint-training gradient competition; stop-gradient fix scoped.** Froze encoder/ITM/FTM/projector exactly as `teacher_state.pt` left them (z shaped under full joint loss), trained a fresh small head (LayerNorm->Linear->GELU->Linear) on Froude alone:

Delta-Froude target (bm_next - bm_t): all 4 combinations (z source x loss shape) pass by 3.5-6.7x the 0.110 bar (best: ITM+cosine +0.733). MSE matched or beat cosine throughout (rejects "MSE kills direction" hypothesis).

Gate check on `L_body`'s actual target (`wm/train.py`: absolute `batch["body_motion"][t]`, not a delta): re-ran with correct target -- transfers, more strongly (gaps ~10x the bar, e.g. ITM+MSE +1.048).

Root cause: z from a jointly-trained checkpoint still carries the full signal when read by a head that never competed with L_recon/L_motion/L_body/L_state for its gradient. The failure is a joint-training competition problem: every loss shares undetached gradient into z via one `z=itm(...)` computation.

Fix chosen: stop-gradient (`z.detach()` before body_head in `wm/train.py`'s `forward_step`); `L_state`/state_head retired entirely (redundant with `L_body` once the harmful delta ingredient, F177, is removed -- both target `batch["body_motion"]`). Retrain scoped (~9 hours on original machine) but deferred to a faster machine.
Scripts: `isolated_z_head_probe.py`, `isolated_z_head_probe_bodytarget.py`.

**13. Cheap proxy before the 9-hour retrain: L_body's gradient does real positive work too -- stop-gradient is a tradeoff, not free.** No existing `lambda_body=0` egocentric checkpoint existed to test cheaply; built an 8-epoch from-scratch proxy (`lambda_body=0.0`, recon+motion only). Retention vs the WITH-L_body reference (teacher_state.pt):

| target | forward | lateral | yaw |
|---|---|---|---|
| delta-Froude | 32% (0.170/0.529) | 71% (0.424/0.598) | 49% (0.265/0.535) |
| absolute body_motion[t] | 71% (0.470/0.660) | 64% (0.480/0.755) | 73% (0.683/0.938) |

Every cell nonzero (recon+motion alone develops real signal) but every cell weaker than WITH-L_body (32-73% retention) -- L_body's gradient contributes positively, not just competes. Forward is weakest (consistent with forward being the hardest channel project-wide: F119, F172, proj_action_ceiling_check's 39%). Confirmed on a second seed (delta retention 49/69/47%, absolute 73/61/76% -- consistent, not a fluke).

Decision: proceed with full stop-gradient retrain anyway, since even the weakest isolated-head condition (proj(a)+MSE, rho 0.405) cleared the bar by 3.5-10x. Pre-registered outcome branches: all channels clear -> win outright; forward fails, others clear -> partial/scaled stop-gradient as follow-up; all fail -> re-open diagnosis.
Script: `proxy_z_signal_check.py`. Checkpoint: `wm/runs/beh12_proxy_reconmotion/best.pt` (+seed1 replicate).

**14. Full stop-gradient retrain: action-lever clears by ~9x -- the first working fix in the arc.** 50 epochs, `z.detach()` on L_body, L_state retired, `wm/runs/beh12_body_stopgrad/best.pt`.

| | median cos |
|---|---|
| real z | 0.693 |
| mean z | -0.315 |
| gap | +1.008 (bar 0.110) |

Per-channel sign-agreement gap: forward +0.238 (52.1% vs 28.4%), lateral +0.031 (weakest, still positive, 81.5% vs 78.5%), yaw +0.243 (77.3% vs 53.0%). All three channels clear, exceeding the proxy's pessimistic prediction for forward (proxy predicted forward would likely fail at 32-49% retention; it didn't) -- the full 50-epoch run develops a stronger z than the 8-epoch proxy indicated, reported as a positive surprise against a stated prior.
Script: `stopgrad_action_lever_check.py`.

**15. Ranking test on the fixed checkpoint: lever cleared, ranking did not improve.** `condition_confusion.py` (F172's protocol, live CoppeliaSim, 40 branch points) on the assembled stop-gradient checkpoint (fresh projector fit needed):

| scorer | new (stop-grad) accuracy | new mean true-rank | F172 old (delta-head) accuracy | F172 mean true-rank |
|---|---|---|---|---|
| direct (body_head(proj(a)), no rollout) | 28% | 3.25 | 20% | -- |
| state (retired, not runnable) | -- | -- | 28% | 2.33 |
| ridge | 12% | 3.60 | 22% | -- |
| f179 (rollout) | 18% | 4.12 | 2% | -- |

`direct` ties the old delta-head on exact accuracy (28%) but is worse on mean true-rank (3.25 vs 2.33) -- ranking did not improve and may be marginally worse. `direct` itself (same scorer type, pre- vs post-fix) improved substantially (20%->28%), confirming the fix helped the z-readout specifically; it just didn't surpass the retired delta-head. Third confirmation this session that "readable," "rankable," and "controllable" are three separate bars that don't transfer.

**16. Oracle check: confirms exact-accuracy is saturated by per-transition noise; mean-rank is where the fix falls short.** Added a true physical oracle (each candidate's own recorded body_motion at the exact branch-point timestep, no model):

| scorer | accuracy | mean true-rank |
|---|---|---|
| f179 | 18% | 4.80 |
| direct (stop-gradient) | 28% | 3.25 |
| ridge | 12% | 4.03 |
| oracle (true instantaneous telemetry) | 20% | 2.15 |
| state (old, retired) | 28% | 2.33 |

Oracle itself only reaches 20% accuracy (below direct's 28%) -- per-transition gait-phase noise caps exact-match this low regardless of scorer quality; 28% was never a ceiling problem. Mean-rank has real headroom: oracle=2.15 is clearly best; `direct` (3.25) is worse than both oracle and the retired delta-head (2.33) -- a real regression on the metric that isn't saturated. Confusion pattern: oracle's errors are mostly cross-family (23/32, pure noise signature); `direct`'s errors are almost entirely within-family (24/29) -- direct nails coarse discrimination but misses fine within-family rank, the same coarse/fine split seen since F119/F127/F172.

**17. Rollout-horizon sweep on f179: no horizon closes the mean-rank gap.** k in {1,2,3,5,10}:

| k | f179 mean-rank |
|---|---|
| 1 | 5.20 |
| 2 | 4.38 |
| 3 | 4.47 |
| 5 | 4.40 |
| 10 | 4.65 |

Best (k=2, 4.38) still a full point worse than direct's 3.25 (no rollout); no monotonic trend toward oracle's 2.15; k=1 is worst, not best. Converges with F126/F127: rolling the FTM into a selection score makes ranking worse than skipping rollout, now confirmed across horizons on the new checkpoint too. The rollout mechanism itself is the problem, not its length.

**18. Real posterior/prior/KL RSSM (DreamerV3-style, tested properly): fails worst of all scorers tried.** Built `h_t=GRUCell(h_{t-1},z_{t-1},a_{t-1})`, posterior q(z_t|h_t,e_t), prior p(zhat_t|h_t), KL loss, Froude head from (h_t,z_t); hexapod only, pooled input. First attempt: posterior collapse (no KL warmup, KL fell to 0.003-0.015, held-out MSE ratio 1.120, worse than mean) -- caught by sanity check, fixed with linear KL warmup (0->full over 500 iters, 3000 total), KL stayed healthy (0.09-0.28), sanity ratio 0.900.

Integrated as fifth scorer in `condition_confusion.py`:

| scorer | accuracy | mean true-rank |
|---|---|---|
| direct | 28% | 3.25 |
| oracle | 20% | 2.15 |
| f179 (FTM rollout) | 18% | 4.33 |
| ridge | 2% | 4.67 |
| rssm (posterior/prior/KL) | 10% | 5.95 |

Worst scorer measured in the entire arc -- worse than f179 and ridge, both already-known nulls. Wrong picks are also least direction-sensible (mean cosine on errors 0.213, below f179's 0.468). Third architecturally distinct rollout mechanism to fail at improving ranking (stateless FTM, deterministic ConvGRU, now stochastic RSSM). Not evidence the field-standard architecture is wrong in general (one seed, one short run, one embodiment, pooled input) but no version of "add rollout architecture" has helped this pipeline's ranking task -- an overnight full-scale RSSM commitment is not justified by this evidence; recommendation is not to launch it.
Scripts: `rssm_stage1_gate.py`, `condition_confusion.py --rssm wm/runs/rssm_stage1_gate.pt`. Checkpoint: `wm/runs/rssm_stage1_gate.pt`.

**Overall shape of the arc.** Systematic elimination (loss weight, prediction target, 4 architecture variants spanning pooled/spatial x recurrent/non-recurrent, input representation) localized the cause to joint-training gradient competition on z; stop-gradient fixed the action-lever by ~9x, the first working fix in this arc -- but this did not convert to ranking improvement (exact accuracy tied, mean-rank regressed slightly), confirming "readable," "rankable," and "controllable" are three separate bars. The rollout mechanism itself (not horizon length, not architecture sophistication up to and including a real RSSM) is the unresolved obstacle to closing the ranking gap.

---

### F181. Zero-shot babble-grounding of a genuinely unseen body (gecko) fails; the diagnosis points to fine-tuning, not the babble fit, as the missing step

Gecko (16-D action, absent from any pretrain source) was grounded into the shared Froude coordinate using only motor babble (36 clips, `data/gecko/babble/`), via a new `nets.gecko` entry added to `beh12_body_stopgrad`'s `ActionProjector` with zero shared trunk. The fit itself passes (rollout gap ratio 0.0678, 0.252 vs mean-z).

Zero-shot transfer of the existing hexapod-trained `CleanFroudeHead` (frozen, no retraining) fails on gecko:

| body | forward rho | lateral rho | yaw rho | median rho |
|---|---|---|---|---|
| hexapod (head trained here) | 0.600 | 0.395 | 0.454 | 0.454 |
| B1 (zero-shot, in pretrain) | 0.181 | 0.427 | 0.628 | 0.427 |
| gecko (zero-shot, absent from pretrain) | 0.113 | -0.331 | 0.151 | 0.113 |
| gecko, ground-truth z (no babble-fit) | 0.189 | -0.030 | 0.184 | 0.184 |

Ground-truth z is equally weak (median 0.184), ruling out the babble fit as the weak link — the failure is the frozen WM's (ITM/FTM/head) domain shift to gecko, not the projector.

Superseded/corrected by F189: the full fine-tune pipeline does NOT recover gecko (median rho 0.032 projector path, 0.171 ITM path) — deeper than domain shift alone. F189 also found gecko's Froude numbers here were measured through a broken body-frame axis and should be re-derived; F189's per-channel check found all three channels uniformly weak (not a lateral-specific defect).

Scripts: `wm/fit_gecko_projector.py`, `scripts/diagnostics/objective_experiments/gecko_froude_transfer.py`.
Checkpoints: `wm/runs/beh12_body_stopgrad/projector_stopgrad_gecko.pt`, `wm/runs/ftm_froude_stopgrad_head_hexapod.pt`. Data: `data/gecko/babble/babble_{1..36}.npz`.

---

### F182. Dead end: `wm.train --init_ckpt` is not adaptation

Warm-starting `wm.train` (`--init_ckpt`) with a new body as a source looks like fine-tuning for a new embodiment but isn't: it jointly retrains ITM+FTM+full MotionDecoder+body_head+adversarial probe under the full multi-task pretrain loss — closer to resuming pretraining with a new source mixed in. Two rounds on B1 (plain, then with a diagnosed `body_stats`-recompute confound fixed) both made B1's transfer correlation worse than doing nothing, forward going negative in both.

The scale confound found and fixed is a real fact about this mechanism but is not a finding about claim (3): the right adaptation pipeline already existed (`wm/adapt.py`, `wm/fit_projector.py`, `wm/adapt3.py`, `wm/fit_body_head.py`, validated at F122/F123) and was not used here. See F183, where B1 grounds decisively via the correct pipeline, including on the forward channel that looked broken here.

Checkpoints/flags from this dead end were removed; use `wm.finetune_new_body` (`doc/FINETUNE_GUIDE.md`) for any future body.

---

### F183. LAC-WM's actual staged adaptation recovers B1 decisively (forward included), correcting F182's mechanism

F182's `wm.train --init_ckpt` "fine-tune" was never LAC-WM's adaptation procedure. The correct one, merged into `wm.finetune_new_body` and documented in `doc/FINETUNE_GUIDE.md`:

    stage 1  wm.adapt          fine-tune ONLY ITM+FTM on new body's clips, else frozen
    stage 2  (new, generic)    fit action projector against the adapted ITM's z
    stage 3  wm.adapt3         optional joint projector+FTM fine-tune (skipped: stage 2 rollout-gap 0.393, below threshold)
    stage 4  wm.fit_body_head  refit only the shared body_head (~8.8k params)

Run on B1 (absent from pretrain), stage 4 (`--latent both`, 400 epochs) took held-out MSE ratio from 1.330x (worse than mean) to 0.751x. Rho comparison, 585 held-out B1 transitions:

| B1, z=proj | forward | lateral | yaw | median |
|---|---|---|---|---|
| zero-shot | +0.057 | +0.264 | +0.526 | 0.264 |
| `wm.train` fine-tune (F182) | -0.068 | +0.231 | +0.326 | 0.231 |
| `wm.train` fine-tune, fixed stats (F182) | -0.061 | +0.234 | +0.424 | 0.234 |
| LAC-WM staged pipeline (this entry) | +0.572 | +0.449 | +0.670 | +0.572 |

Forward — never above-noise in any `wm.train`-derived checkpoint — is now clearly real (+0.572 proj / +0.517 itm). F182's negative results stand for `wm.train`-based joint retraining specifically; they don't generalize to claim (3).

Note (from F189): the stage-1 rollout ratio quoted here (1.55x down to 1.23-1.47x) has the sign backwards — `rollout()` returns higher-is-better, so stage 1 actually made B1's forward model worse, yet stage 4 still succeeded; stage-1 rollout ratio does not predict stage-4 success.

Selection/control on this checkpoint: see F184-F188 (mode A/D clear, rollout does not). Gecko's version of this pipeline: see F189 (fails at stage 4, 1.010x).

Scripts: `wm.finetune_new_body`, `wm.adapt`, `wm.fit_projector`, `wm.assemble_teacher`, `wm.fit_body_head`.
Checkpoints: `wm/runs/b1_adapt/adapted_b1.pt` (stage 1), `wm/runs/b1_adapt/projector_b1.pt` (stage 2), `wm/runs/b1_adapt/teacher_b1.pt` (merged), `wm/runs/b1_adapt/body_head_b1.pt` (stage 4). Manual: `doc/FINETUNE_GUIDE.md`.

---

### F184. Goal transfer on the correctly-adapted B1 checkpoint: mode A (direct) clears chance, mode C (rollout) does not

Tested via `scripts/diagnostics/planning/score_by_body_motion.py` against `wm/runs/b1_adapt/body_head_b1.pt`, with mandatory mismatch control.

Mode A (action-to-speed regression, no rollout) clears chance (28%):

| horizon | matched | mismatched vs demo | mismatched vs goal |
|---|---|---|---|
| 1 | 49% | 24% | 46% |
| 3 | 41% | 26% | 41% |
| 5 | 45% | 33% | 38% |

Per-family: speed/side_R transfer well (53-79%), turn weak (15-36%).

Mode C (goal read from source body's frames via ITM, candidates scored by rolling the FTM forward) does not clear chance at any horizon (1/3/5/10), and shows no rising trend through horizon 10 (unlike F123's mode-C, which cleared at horizons 5-10) — a clean negative.

Consistent with F126/F127 (rolling the FTM into a score makes ranking worse, not better): grounding a body absent from pretrain now works; the rollout mechanism's contribution to cross-body selection is the piece that still fails, independent of target-body adaptation quality.

Script: `scripts/diagnostics/planning/score_by_body_motion.py --mode {A,C}`.
Checkpoint: `wm/runs/b1_adapt/body_head_b1.pt`. Goal source: `data/egocentric/beh12_c10f10t10_ego_flat`.

---

### F185. Live rendered cross-embodiment closed loop on adapted B1: direct beats rollout again; both share a weak spot traced to categorical yaw-data absence

> F256 (2026-09-25): the `--mechanism rollout` half of this entry is invalid. Its rendered closed loop fed the planner a 24-deg egocentric view (training clips: 90) from a body whose pitch/roll/height drifted across candidate switches. Rollout result files moved to `results/_invalid_F256/`. `--mechanism direct` numbers stand (direct never reads the loop's camera).

First body_head-driven closed loop (`sim/control/close_loop_direct_froude.py`, `DirectFroudePlanner`/`RolloutFroudePlanner`), vs. prior `LatentPlanner`-based loops (F82-F115) which read "which robot is this" rather than the goal. Metrics reused verbatim from F85/F91. Setup: `wm/runs/b1_adapt/body_head_b1.pt`, B1's 12-condition library as candidates, hexapod goal clips, kinematic (not physically simulated) loop.

| mechanism | S.R. survival | S.R. behaviour class | S.R. speed within 15% | median speed error |
|---|---|---|---|---|
| direct | 3/3 | 2/3 | 1/3 | 15.6% |
| rollout | 3/3 | 2/3 | 0/3 | 43.7% |

Direct's median error (15.6%) beats F85 (19.2%) and F91 (0/3) baselines despite cross-embodiment being harder. A camera confound (allocentric scene feeding rollout's `e_t`) was found and fixed by mounting a second egocentric camera; fixing it did not change the qualitative result — rollout stayed scattered (25-36% best candidate) across all three goals.

Direct's one class miss traced to data: hexapod's own turn library never has yaw exceed forward even at max turn rate, so the "turn" goal used was never a clean single-axis test. At the decision step, the model's predicted Froude badly understates both candidates' true dominant channel (e.g. real yaw 0.088 read as 0.019). Full-dataset check: yaw is the dominant channel in 0% of B1's 3,168 training transitions (forward 66.7%, lateral 33.3%, yaw 0%) — categorical absence, not partial imbalance, explains yaw compression (ratio 0.77). Lateral is compressed too (ratio 0.54) despite being 33.3% of data, likely from majority-class (forward) capacity pull.

Proposed fix (not yet run): refit `body_head_b1.pt` with `--latent projector` only, matching what's used at control time (same fix pattern as F122).

Scripts: `sim/control/close_loop_direct_froude.py`, `wm/policy/planner.py`. Checkpoint: `wm/runs/b1_adapt/body_head_b1.pt`. Runs: `results/wm/closed_loop/direct_froude/{direct,rollout}_*.npz`.

---

### F186. Calibrating body_head on hexapod too fixes F185's diagnosed cause and exposes a confound between "direct vs rollout" and "goal source"

> F256 (2026-09-25): the `--mechanism rollout` half of this entry is invalid (24-deg egocentric view vs 90-deg training, pitch/roll/height drift). Rollout files moved to `results/_invalid_F256/`. `--mechanism direct` numbers stand.

F185's proposed fix run: `body_head_b1.pt` was fit `--latent both` but embodiment b1 only, never shown hexapod's own z — hence reading a forward-moving hexapod clip as negative forward. Refit stage 4 only with `--also hexapod=...` (`wm.fit_body_head --also`), keeping stages 1-2 untouched. Held-out B1 MSE ratio moved 0.751 → 0.803 (small expected cost). New checkpoint: `wm/runs/b1_adapt/body_head_b1_hex.pt`.

Sign fixed exactly as predicted (hexapod_ep0 goal forward reading): B1-only head -1.137 (wrong sign) → B1+hex head +0.491 (correct). F185's mixed-goal class miss also fixed:

| head | S.R. behaviour class | S.R. speed within 15% | median speed error |
|---|---|---|---|
| B1-only | 2/3 | 1/3 | 15.6% |
| B1+hexapod | 3/3 | 1/3 | 19.1% |

Real trade-off: direction correctness improves, but the easy forward goal's speed error worsens (7.5% → 19.1%) since the fixed-capacity head now splits across two bodies.

Rollout's sign error is also fixed (-1.721/-1.137 → +0.491) but its core indecisiveness is not (best candidate 22%, dominant-channel error 54.9%) — the third of three independent confounds (camera, goal noise, body_head domain mismatch) fixed without moving rollout's core failure, pointing at the mechanism itself.

Methodological fix: F185 confounded `--mechanism` (direct/rollout) with `--goal_source` (physics/vision) — direct always read goals from physics, rollout always from vision. Now independent flags on `close_loop_direct_froude.py`, giving a full A/B/C/D grid. Mode D (direct + vision goal) was the untested cell.

Scripts: `wm/fit_body_head.py --also`, `sim/control/close_loop_direct_froude.py --goal_source`, `wm/policy/planner.py` (`vision_goal`). Checkpoint: `wm/runs/b1_adapt/body_head_b1_hex.pt`.

---

### F187. Mode D: a vision-only goal costs direct-Froude selection nothing once body_head is properly calibrated

Mode D (direct candidate scoring, goal read from source robot's frames via ITM, no rollout), same three goals and hex-calibrated head (`body_head_b1_hex.pt`) as F186:

| condition | goal source | S.R. behaviour class | S.R. speed within 15% | median speed error |
|---|---|---|---|---|
| mode A, B1-only head | physics | 2/3 | 1/3 | 15.6% |
| mode A, hex-calibrated head | physics | 3/3 | 1/3 | 19.1% |
| mode D, hex-calibrated head | vision | 3/3 | 1/3 | 17.4% |

A vision-only goal costs nothing (median error slightly better than mode A). Lateral result is the best single number in this arc (1.9% error); the mixed-goal pick is more decisive under mode D (69%) than mode A (33%).

What this settles: candidate scoring that never touches vision, combined with a goal read only from vision, performs on par with using privileged physics state for the goal. What still fails regardless is the rollout mechanism on the candidate side (mode C) — the open problem is localized to candidate scoring, not to whether vision alone can specify a goal.

Scripts: `sim/control/close_loop_direct_froude.py --mechanism direct --goal_source vision`. Checkpoint: `wm/runs/b1_adapt/body_head_b1_hex.pt`. Runs: `results/wm/closed_loop/direct_froude/direct-vision_*.npz`.

---

### F188. Completed 2x2 ablation: goal source explains none of the gap, candidate mechanism explains all of it

> F256 (2026-09-25): the `--mechanism rollout` half of this entry is invalid (24-deg egocentric view vs 90-deg training, pitch/roll/height drift). Rollout files moved to `results/_invalid_F256/`. `--mechanism direct` numbers stand.

Mode B (rollout + physics goal) run to complete the grid. Two bugs fixed along the way: `RolloutFroudePlanner.from_checkpoint` never set `.channels`/`.standardize`; the script only built the VJEPA2 encoder in the vision-goal branch though rollout needs it regardless.

Full grid, same three goals, hex-calibrated head:

| mode | candidates | goal source | S.R. behaviour class | S.R. speed within 15% | median speed error |
|---|---|---|---|---|---|
| A | direct | physics | 3/3 | 1/3 | 17.4% |
| D | direct | vision | 3/3 | 1/3 | 17.4% |
| B | rollout | physics | 2/3 | 0/3 | 56.0% |
| C | rollout | vision | 2/3 | 0/3 | 54.9% |

Goal source explains none of the gap (A=D exactly, B=C within noise); candidate mechanism explains all of it. A same-robot control (B1 goal vs B1 candidates, zero cross-embodiment) also fails for rollout (best 20%, correct pick only 15%) — cross-embodiment was never the cause.

This reconfirms F126/F118 on a new checkpoint: rollout functions as a state classifier (identifies what the robot is already doing and continues it), a structural property of routing a rolled-forward prediction through ITM+body_head, not a bug in this session's setup. No further debugging of rollout itself is warranted.

Re-run 2026-09-10 with corrected egocentric camera (collector uses `--floor_scale 3.0`, loop had applied none, causing a black background band, 4.7% near-black vs training's 0.000, and hardcoded `fwd=[1,0,0]` instead of `heading()`-derived). Both fixed; re-run confirms the same conclusion on in-distribution frames:

| mode | goal read error | top pick | % steps in goal's family | \|top pick - true goal\| |
|---|---|---|---|---|
| A | 0.0000 | turn_w0.008 | 78% | 0.038 |
| D | 0.0293 | turn_w0.008 | 80% | 0.038 |
| B | 0.0000 | side_R_lvl1 | 47% | 0.290 |
| C | 0.0293 | side_R_lvl1 | 40% | 0.290 |

Rollout prefers the single worst candidate (side_R_lvl1, farthest of all 12 from goal at 0.290) even with a perfect goal read (mode B).

> WITHDRAWN 2026-09-14: the entry's original "candidate-spacing threshold ~0.033" reasoning for why A=D was arithmetically wrong (confused per-candidate distances-to-goal with the gap between the two nearest candidates, which is actually 0.0079, making the 0.029 read error 3.7x larger than the gap, not smaller). Void wherever it appears (F208, `report/proposal.tex`, `report/update_slide.md`); no code computed it. The A=D claim itself survives on other evidence — see F210's multi-clip table (12 goals, 96 decisions, mode D 92%/dist 0.0433 vs mode A 90-92%/0.0616).

Caveat: "3/3 behaviour class" counts the dominant pick per episode; 20-22% of individual steps are in the wrong family under A/D. Kept for comparability with F95/F101, not because it's the better measure.

Scripts: `sim/control/close_loop_direct_froude.py --mechanism rollout --goal_source physics`, `wm/policy/planner.py`. Checkpoint: `wm/runs/b1_adapt/body_head_b1_hex.pt`. Runs: `results/wm/closed_loop/direct_froude/rollout-physics_*.npz`, `rollout-vision_b1_ep2301_b1_ep2.npz`. See also F116-F118, F126.

---

### F189. Gecko: two measurement bugs, not a claim failure — body frame built on a near-vertical axis, and the projector fed an input that cannot contain the answer

Supersedes this entry's own first version ("babble narrowness" diagnosis) — both the cause and the direction of the stage-1 result were wrong.

**Correction 1 — stage-1 rollout ratio read upside down (here and in F183):** `finetune_ftm.rollout` returns `hold_error/model_error`, higher is better.

| | frozen base h=1 | after stage 1 | direction | stage-4 outcome |
|---|---|---|---|---|
| B1 | 3.627 | 3.698 | worse | succeeds (0.751) |
| gecko | 3.994 | 3.878 | better | fails (1.010) |

Stage-1 rollout quality is anti-correlated with downstream success across the only two cases tested — not a valid gate. Also: frozen, never-adapted model already beats holding-frame-still on both robots (1.55x, 1.57x), contradicting `doc/FINETUNE_GUIDE.md`'s stated stage-1 premise.

**Correction 2 — gecko's body frame used the wrong axis.** `forward_axis`/`heading` read gecko's forward from body x-axis, whose horizontal projection has mean length 0.065 (reaches 0.000) across all 36 babble clips — 99.9% of frames have meaningless `arctan2`. Heading jumped up to 358 deg between consecutive 50Hz frames (16.8% of steps >90 deg; B1's worst step: 2 deg), scrambling forward/lateral (not just yaw) and the reported Froude calibration (gecko never actually walked at the believed 0.133-0.138). Correct axis is -(body y): horizontal length 0.998, 0% unstable frames, zero per-step jumps over 2,340 steps. Effect on frozen-embedding ridge ceilings: forward +0.244→+0.303, lateral +0.606(inflated)→+0.285, yaw -0.042(dead)→+0.362-0.501 (now the strongest channel).

**Correction 3 — projector's single-frame input cannot contain the answer.** Ground-truth ridge (actions→body motion, held out by clip): single frame (what `ActionProjector` gets) gecko +0.215 vs B1 +0.459; window of 20 frames gecko +0.736 vs B1 +0.770 — with 20 frames gecko's actions are as informative as B1's. Explains why stage-2 MSE looked fine (0.230) while Froude signal was absent: MSE on z can't distinguish pose/phase from the Froude-relevant component.

Full pipeline re-run on corrected frame (`wm/runs/gecko_fixed`): stage 1 improves at every horizon, stage-4 held-out lands at 1.004 — the frame fix alone does not rescue stage 4 (shared `body_stats` come from hexapod pretrain; gecko's corrected motion is ~6x smaller). With both fixes and widened projector window:

| K | proj z-MSE ratio | stage-4 held-out | median rho (projector path) |
|---|---|---|---|
| 1 (current pipeline) | 0.574 | 0.988 | +0.186 |
| 10 | 0.164 | 0.975 | +0.210 |
| 20 | 0.132 | 0.970 | +0.249 |

Stage 4 finally drops below 1.0; not B1's 0.751/0.572, but the gap is now localized.

What remains is a property of the robot: actions carry 0.736, egocentric video carries only 0.374, pipeline delivers 0.249 (67% of what video allows) — video is the binding constraint because gecko is physically slow (forward Froude flat at 0.038-0.048 across gait frequencies 2-6Hz, vs B1's 0.126; frequency moves it only 27%, not the 4x the broken frame suggested).

Two earlier conclusions retracted: the "babble ~2x too narrow" diagnosis (measured through the broken frame), and `data/gecko/babble_v3` (54 clips, frequency sweep) — re-measured on the corrected frame it is the WORSE dataset (ceiling +0.146 vs original's +0.736): per-step babble noise, not across-clip parameter coverage, was doing the work.

Every gecko number recorded before this entry (including F181's transfer table and `collect_gecko_cpg.py`'s docstring) was measured through the broken frame and should be re-derived before citing.

Scripts: `wm/data/embodiment.py` (`forward_axis`/`heading` gecko fix), `sim/collect/collect_gecko_dataset.py --explore/--lift`.
Data: `data/gecko/babble` (36 clips, better), `data/gecko/babble_v3` (54 clips, superseded).
Checkpoints: `wm/runs/gecko_adapt/`, `wm/runs/gecko_adapt_s3/`, `wm/runs/gecko_v3_stage1/`.

---

### F190. A genuine 3-family B1 babble CPG built; the first two turn-primitive attempts were mathematically degenerate, not just mistuned

`sim/collect/collect_b1_cpg_babble.py`. Forward/lateral straightforward once `b1_flat_real.xml` replaced `b1_flat.xml` (the placeholder-physics file could not be walked forward by any hand-tuned sign/phase combination).

Yaw attempts 1 (sign-flip LEFT legs' swing) and 2 (reverse LEFT legs' phase clock) are both exactly zero net yaw BY CONSTRUCTION, not mistuning: the diagonal trot's phase groups (FL,RR)/(FR,RL) each already contain one LEFT and one RIGHT leg a half-cycle apart, so negating a LEFT leg's own-phase signal makes its waveform land exactly on its RIGHT diagonal partner's original — relabeling groups, not creating asymmetry. Measured yaw ~0 (0.0001-0.016) at every amplitude tried. Generalizes: any 2-way leg split where each group straddles both diagonal phase-groups is subject to this degeneracy if implemented as an own-phase sign/clock flip.

Working fix: differential on the HIP channel, front-vs-rear, on ONE side only, keyed to a GLOBAL clock (not per-leg phase): `hip += pivot_amp * thigh_amp * (+1 if front else -1) * (-sin(2*pi*freq*t_elapsed))`, applied only to {FL, RL}. Verified: `thigh_amp=0.6, pivot_amp=1.5, calf_amp=0.6` gives real monotonic heading change (+50 deg over 10s), froude yaw=0.043 dominant over fwd=-0.025/lat=0.029, no fall.

`calf_amp` (lift height), not gait shape, was the main lever on excess vertical bounce: lowering from 1.0 to 0.4-0.6 halved z_std (~0.045→~0.02) with no cost to stability, across all three families (not tuned to match B1's measured Froude).

Collected: `results/wm/dataset/b1_babble/batch2/`, 36 clips (12 fwd/12 lat/12 yaw). Forward/lateral 12/12 correctly dominant; yaw only 5/12 (real, acknowledged residual — see F191).

Scripts: `sim/collect/collect_b1_cpg_babble.py` (degeneracy proof documented in `--pivot_amp` help text).

---

### F191. Full 4-stage babble refit with hexapod rehearsal: forward fixed, lateral partially fixed, yaw untestable with this goal set

Ran `wm.finetune_new_body` stages 1-4 on F190's 36-clip batch (`wm/runs/b1_babble_adapt2/`), base `beh12_hexonly_stopgrad/best.pt`, then re-ran stage 4 with `--also hexapod=...` (`body_head_b1_hex.pt`) — the F186 rehearsal fix, confirmed necessary a second time on an unrelated body/data combo. Stage 2 rollout-gap 0.740, stage-4 held-out 0.837 (no rehearsal) / 0.895 (with) — pipeline is healthy; residual is a data/signal limitation.

Mode A (physics goal), whole-clip-mean, both pools scored through the same checkpoint:

| pool | accuracy | fwd | lat | yaw |
|---|---|---|---|---|
| expert (12, beh12_b1_ego_flat) | 25% (3/12) | 0/8 | 3/4 | untestable* |
| babble (36, F190) | 83% (10/12) | 8/8 | 2/4 | untestable* |

*hexapod's 12-condition goal set has zero yaw-dominant clips (see F192). Expert's drop from F184's 42% is not a regression — the checkpoint's projector is fit to babble's action distribution, not the trained-policy distribution expert clips come from; irrelevant to real deployment (an unseen body has no expert clips).

Lateral goal-reading (mode D, vision-only) per hexapod side_* clip: 4/4 broken (pre-rehearsal) → 3/4 broken (post-rehearsal); only side_R_lvl1 fixed. Real but partial progress; the 8 fwd/turn conditions stayed 8/8 throughout. See F192 — not a horizon/averaging artifact.

Checkpoints: `wm/runs/b1_babble_adapt2/{adapted_b1,projector_b1,teacher_b1,body_head_b1,body_head_b1_hex}.pt`. Data: `results/wm/dataset/b1_babble/batch2_rendered/`.

---

### F192. Hexapod's own lateral visual signal is genuinely weak, not a windowing artifact

Ruled out two alternative explanations for F191's lateral goal-misread. (1) Whole-clip averaging hiding a good local window: per-window (h=5) majority vote gives the same wrong answer as whole-clip mean for 3 of 4 lateral goal clips; for the 4th (side_R_lvl1, correct), whole-clip averaging is actually better since it implicitly weights by signal magnitude. (2) Wrong horizon: swept h=1/3/5/10/15/20 — lateral per-window accuracy sits at or below chance (~33%) at every horizon, no monotonic trend; h=1 marginally least-bad, h=5 (current default) one of the worst.

Consistent with a pre-existing limitation: insect-to-B1 unrefitted linear-probe transfer already showed lateral correlation (0.43 allo/0.39 ego) below forward's (0.63/0.50). Raw (non-argmax) channel check: side_R_lvl1's true lateral (-0.123) and vision-read lateral (-0.031) track correctly (~4x compressed) once magnitude clears the noise floor; the two weakest clips (true lateral 0.015/0.068) lose sign entirely. Weak-but-real, magnitude-compressed signal that degrades further as true magnitude shrinks — not fixable by horizon choice.

Rules out: sweeping `--horizon` on `vision_goal()` or averaging-window tweaks as the next move. Does not rule out: whether more/better hexapod-lateral rehearsal data at stage 4 helps (untested), or whether this predates babble adaptation entirely (untested).

Scripts: horizon-sweep/per-window-vote checks were one-off, not yet saved as a standalone script.

---

### F193. `free_offset`: two real bugs found and fixed; corrected mechanism gives only a small, honest gain over locked

`free_offset=True` on `DirectFroudePlanner` (and `close_loop_direct_froude.py --free_offset`) lets the planner pick any offset `tau` within a candidate, since `DirectFroudePlanner` never reads a candidate's frames.

Bug 1: `score_offsets` doesn't depend on `t`, so calling it fresh each step under an unchanging goal returns the identical `(candidate, tau)` every time — total freeze (100% of 56 steps on one candidate). First fix attempt (replan every `horizon` steps, advance `tau` between replans) looked right but was incomplete.

Bug 2 (found from video, not a table): re-searching on any fixed schedule still snaps back to the identical `(candidate, tau0)` since `score_offsets` is deterministic. The body replayed the same 5-frame window ~11 times; that window's own net dpos/dquat isn't zero, and compounding it caused a full tip-over (up.z 1.0→0.40, height +0.43→-0.09, below ground). Invisible to every prior accuracy/distance number, which never checked orientation plausibility.

Real fix: replan once (at warm start/first step), then let `tau` advance continuously for the rest of the episode; re-search only when the window would run past the candidate's recorded length. Verified: up.z stays 0.993-1.0, height stays 0.385-0.475 post-fix. Applied to `close_loop_direct_froude.py`, `final_2x2x2_test.py`, `free_offset_candidate_test.py` (all had the same re-search-on-schedule pattern).

Every free_offset number reported before this fix (this entry's and F194's original versions) is wrong. Corrected numbers:

| pool | goal | locked | free_offset, buggy (WRONG) | free_offset, corrected |
|---|---|---|---|---|
| expert | A | 96%, dist 0.0735 | 100%, dist 0.0745 | 100%, dist 0.0769 (~unchanged) |
| expert | D | 100%, dist 0.0877 | 100%, dist 0.1072 | 100%, dist 0.1071 (~unchanged) |
| babble | A | 69%, dist 0.1133 | 92%, dist 0.1412 | 72%, dist 0.1285 |
| babble | D | 60%, dist 0.0871 | 83%, dist 0.0830 | 70%, dist 0.0909 |

Honest conclusion: `free_offset` gives babble a small real family-accuracy gain (+3 to +10 pts, not +14 to +23) but is still worse than locked on continuous Froude distance in both goal modes. Expert is essentially unaffected. F194's directional conclusion (family accuracy overstates the benefit vs. genuine closeness) survives; its magnitudes do not and must not be quoted.

A second defect in bug 1's original fix (kinematic pose step read motion at literal `t` even when action was scored from a different `tau`) was also fixed by threading `tau` through to the motion index.

Unaffected by either bug: `wm/fit_projector.py`'s `gather()` uses real, consecutive, index-aligned transitions (not shuffled), and the projector/body_head fit is order-independent, so free_offset needs no re-fit — revisit only if the pipeline becomes recurrent.

Scripts: `wm/policy/planner.py` (`DirectFroudePlanner.free_offset`, `score_offsets`), `sim/control/close_loop_direct_froude.py --free_offset`, `scripts/diagnostics/objective_experiments/{final_2x2x2_test,free_offset_candidate_test}.py`.

---

### F194. Q21's clean 2x2x2: babble substitutes for the teacher library on family accuracy, but not on the more trustworthy continuous distance metric — genuinely open

Pre-registered (`doc/OPEN_QUESTION.md` Q21): babble substitutes if it clears its own chance AND `capture_ratio = lift_babble/lift_expert >= 0.5`. One script/run/log: `scripts/run/b1_babble_clean_redo.sh`, `results/wm/dataset/b1_babble/clean_redo_log.txt`. Checkpoints: `wm/runs/b1_babble_clean/{adapted_b1,projector_b1,teacher_b1,body_head_b1,body_head_b1_hex}.pt`. Data: `data/egocentric/b1_babble_ego_flat/`.

Headline cell (goal=A physics, free_offset=False — untouched by F193's bugs):

| metric | babble | expert | capture_ratio | verdict |
|---|---|---|---|---|
| family accuracy | 69% (chance 45%, lift +24) | 96% (chance 56%, lift +40) | 0.60 | clears 0.5 |
| Froude distance (lower better) | 0.1133 (chance 0.1518, lift 0.0385) | 0.0735 (chance 0.1584, lift 0.0849) | 0.45 | fails 0.5 |

Verdict flips by metric; Froude distance is the more trustworthy one since family accuracy is a coarse same-family-counts-as-correct proxy. Per-family: fwd 78% (50/64), lat 50% (16/32) — lateral weaker, consistent with F192. Step 1 (does babble substitute for the teacher library) remains genuinely open (`doc/OPEN_QUESTION.md` Q21).

Mechanism behind babble's weaker D-mode robustness: expert candidates have 0% chance their nearest neighbor (predicted-Froude space) is a different family (perfectly separated); babble has 19% of candidates at a zero-margin family boundary. A v2 fix attempt (bigger lateral strafe amplitude) failed — margin checked in TRUE Froude space doesn't transfer to PREDICTED Froude space (F110: direction right, extent wrong); cross-family risk stayed 19%. Corrected rule (memory `babble-generation-rule-target-pretrain-range.md`): margin must be checked through a fitted checkpoint's prediction, never raw motion.

`free_offset`: see F193 for corrected numbers — small real family-accuracy gain over locked but worse on continuous distance in both goal modes; every number this entry originally reported was superseded.

Goal source (A vs D, both free_offset=False): babble worsens under vision goal (69%→60%), consistent with F192. Expert's D≥A (96%→100%) is not a goal-redistribution artifact — exact top-1 picks change for 11/12 goals under vision noise, but 8/12 expert candidates are already forward-family so same-family reshuffles still read as correct. The margin property (0% vs 19% cross-family risk) is the real explanation.

Full 8-cell table in `results/wm/dataset/b1_babble/clean_redo_log.txt` and `corrected_free_offset_full_log.txt`.

---

### F195. F136's reward-quality wall generalizes to B1: current checkpoint's scoring function is at/below chance on local action perturbations, in real MuJoCo physics

Rebuilt F136's test for B1's own MuJoCo harness (`scripts/diagnostics/objective_experiments/reward_quality_gate_b1.py`), reusing `collect_b1_cpg_babble.py`'s control convention. Baseline is a recorded expert clip's action (`data/egocentric/beh12_b1_ego_flat/b1_ep0.npz`) replayed to a real branch state (real qvel/contacts), snapshotted/restored between trials — babble deliberately kept out to avoid conflating this test with Q21's open babble-quality question.

Method: 8 branch points, 16 Gaussian perturbations at sigma 0.1/0.3/0.5, executed in real physics for 3 steps, true resulting Froude compared against a forward goal, separately scored via `body_head(proj(action))`. Hit = model's argmin matches true-physics argmin.

| | value |
|---|---|
| candidates per trial | 17 |
| chance | 5.9% |
| overall hit rate | 4.2% (1/24) |
| sigma=0.10 | 12.5% (1/8) |
| sigma=0.30 | 0.0% (0/8) |
| sigma=0.50 | 0.0% (0/8) |

At or below chance at every perturbation size, matching F136's insect-side shape on a different body/checkpoint/physics engine. Building an RL controller on this checkpoint would likely reproduce F179's collapse — the reward signal has no local gradient for PPO-style exploration. Does not test whether a different checkpoint/objective/fitting procedure could produce a locally-discriminative reward.

Scripts: `scripts/diagnostics/objective_experiments/reward_quality_gate_b1.py`. Log: `results/wm/dataset/b1_babble/reward_quality_gate_log.txt`.

---

### F196. B1 now stands and moves under native CoppeliaSim-Bullet dynamics; prior collapse was an initialization bug

Root cause: the imported convex scene stores all 12 joints at 0 rad. The standing probe set only `setJointTargetPosition(DEFAULT_IL)` (unlike the MuJoCo collector, which initializes both qpos and control), so Bullet began with straight 0.7m legs intersecting the floor — trunk z jumped 0.600→0.496 on the first step and the body collapsed. Fix: `setJointPosition(DEFAULT_IL)` before starting dynamics.

100-step Bullet verification: final z 0.5358m, min up.z 0.9998, xy drift 0.0296m, final worst joint error 0.0182 rad — exceeds the pre-registered 60-100-step guardrail. Peak force reached the configured 93 Nm only transiently.

Two suspected causes closed: a force sensor is initially a rigid link with no compliance setting (per CoppeliaSim manual); CoppeliaSim PID values are not MuJoCo stiffness/damping — its dynamic position controller generates a constrained motor command with `targetForce` applied separately.

First CoppeliaSim-native controller (hand-designed diagonal-trot CPG, no trained policy): 160 steps/8.0s, z≥0.535m, up.z≥0.998, advanced 0.177m, lateral drift 0.015m, worst joint error 0.046 rad. A larger 1Hz/0.6-action gait rolled over (0.867 rad error). Exact repeat from fresh scene load: 0.178m forward, 0.010m lateral, same z/up.z bounds, 0.049 rad error — repeatable. Verified conservative default: 0.5Hz / 0.2 thigh / 0.2 calf.

Scripts: `scripts/diagnostics/objective_experiments/b1_coppelia_dynamics_probe.py`, `scripts/diagnostics/objective_experiments/b1_coppelia_cpg_controller.py`. Scene: `sim/env/b1_flat_convex.ttt` (Bullet).

---

### F197. Stage 3's InfoNCE has never used Froude as its positive-pair signal and never touched a second embodiment — correcting an overstated prior claim

Prompted by F134 (stage-3 contrastive adaptation on one body cost state fidelity on another, 0.757→1.052) being mis-cited as "stage-3 InfoNCE has a specific history of not transferring across bodies." Reading `wm/adapt3.py` end to end (lines 70-98, 262-328) shows this is not supported: one `gather()` call loads clips from one `--data` directory under one `--embodiment` name — no second embodiment anywhere. The positive pair is `(e_t, true recorded action a_t) → e_t+1`; negatives are drawn from a different condition label at the same timestep, within the same body's own clips. Froude values are never read/computed/compared in this file — they only exist downstream in stage 4's `body_head`.

So the correspondence actually built is "same condition label, same timestep, single body" vs "different condition label" — not "same Froude value, any body." The cross-embodiment Froude-correspondence design (hexapod Froude 0.1 ↔ B1 Froude 0.1 as an InfoNCE positive pair) has never been implemented. F134's cross-body fidelity drop is a different phenomenon — single-body contrastive fine-tuning perturbing shared ITM/FTM weights (interference/forgetting), not a correspondence-design failure.

Literature review (`doc/ref/literature_review3_infonce_modality_gap.md`) confirms this is open, not settled-negative: vanilla InfoNCE with independent encoders and naive negatives tends to produce modality gaps in multimodal contrastive learning generally; InfoNCE with real correspondence signal, reconstruction terms, and controlled negative sampling has documented cross-embodiment successes in robotics (e.g. arXiv:2506.14608, 25.3% improvement from pairwise InfoNCE across retargeted action embeddings). `wm/adapt3.py` is the first kind (single-embodiment, condition-label negatives); the second has not been built.

What this opens, not yet run: a stage-3 variant with genuinely cross-embodiment positive pairs (hexapod condition paired with a B1 condition at matching Froude family/value) — would be the first real test of the correspondence hypothesis, and the first place Q23's hexapod-CoppeliaSim-vs-B1-MuJoCo Froude-scale question becomes load-bearing during training. No script exists yet.

Scripts read: `wm/adapt3.py` (full). Reference: `doc/ref/literature_review3_infonce_modality_gap.md`.

---

### F198. ConvGRU kill-gate at full training budget still fails, closing the recurrent-architecture line

F180's ConvGRU kill-gate (spatial-preserving recurrence, gap +0.069, best of four architecture variants) was only tested on a 2000-iteration probe; every other fix (R0, two non-spatial attempts, pooled full-stochastic RSSM, F141's hinge rebuild, F178's counterfactual targets) had been run at full strength and failed. This was the one lead with headroom left.

Method: `SpatialRecurrentModel` (`wm/models/convgru_ftm.py`, unchanged from the probe) trained from scratch for 20,000 iterations (10x) on hexapod only, full 48-clip set — matching every prior probe's one-body-at-a-time convention. Not the full `wm.train`-integrated rebuild (only warranted if this gate clears).

Result: gap -0.006, FAIL — real median cosine 0.863, generic median cosine 0.869, worse than the probe's +0.069 and further from the 0.110 bar.

Collapse check run because training loss reached ~0: `moves` ratio came back 0.4504 with per-channel prediction std at 40-57% of true std — genuine non-degenerate variation, just under-shooting magnitude. Near-zero loss is ordinary overfitting on 48 clips over 20,000 iterations, not a collapse artifact; the FAIL is real.

Closes the recurrent-architecture line: R0 (+0.036), ConvGRU probe (+0.069, best short-budget), full RSSM (worst scorer in project history), full-budget ConvGRU (-0.006) — recurrence has never cleared the bar at any budget, on either body, in any of four architecturally distinct forms. Combined with F141 (hinge) and F178 (counterfactual targets), both objective-level and architecture-level fix families for fine action-discrimination are exhausted (five independent mechanisms, all null). Next move per project convention: write up as a fully characterized negative result, not a sixth mechanism.

Not closed: whether a from-scratch, full-pretrain-scale ConvGRU-in-the-real-FTM-slot rebuild (joint ITM/reconstruction/body losses, as F180 scoped) would behave differently — untested, not warranted by this evidence.

Scripts: `wm/models/convgru_ftm.py`, `scripts/diagnostics/objective_experiments/{convgru_full_retrain,convgru_collapse_check}.py`. Checkpoint: `wm/runs/convgru_full/convgru_full.pt` (hexapod, 20,000 iterations, BIAS-2).

---

### F199. F141's missing multi-step reconstruction anchor passes all three pre-registered criteria; a follow-up B1 fix makes it the best B1 reward-quality result in project history

F141's ActSWM rebuild diverged because `recon` only anchors step 1 while the hinge acts at steps 1-3, leaving steps 2-3 unopposed. F141's own proposed fix (a K-step prediction loss matching the hinge's horizon) was never run until now. `wm.train`'s existing `lambda_rollout` (2-step auto-regressive consistency, already wired but never combined with the hinge) supplies it with zero new code; `hinge_K` reduced 3→2 to match. 50 epochs, hexapod only, BIAS-2 (`wm/runs/beh12_hinge_multistep_anchor/`).

| criterion | F141 (no anchor) | this run |
|---|---|---|
| prediction healthy | fails (diverges 3-4x by horizon 2) | passes (ratio 0.52-0.58 flat through horizon 10) |
| /mean-z down | not reached | passes (0.938/0.886/0.892/0.888/0.893 at h=1/2/3/5/10, flat) |
| separation holding | collapses | passes (0.0007→0.347 over 50 epochs, holds) |

Weaker in raw /mean-z terms than F133/F134's contrastive fine-tuning (0.53-0.58 there vs 0.89-0.94 here), but achieved without contrastive's cross-body fidelity cost (F134: 0.757→1.052) and without divergence.

Follow-up ceiling check (`family_z_ceiling.py`, real transitions only, no FTM): within-condition MSE vs across-condition-same-family MSE ratio is ~1.0 on two independently-trained checkpoints (`beh12_hexonly_stopgrad` 0.983, this run ~1.001) — the family's real z's are barely more separated than repeats of one condition. The hinge rebuild's 0.886-0.938 is close to what the data allows, not evidence of a weak model. Extends F142 (action's causal weight on e_{t+1} under 3%): even the ITM's own z summary doesn't spread out much across a family.

Coda (candidate selection, `hexapod_pretrain_discriminate.py`, n=300 held-out, real z candidates not projector actions): exact condition (chance 9.1%) 33.3%→41.0%, family (chance 29.7%) 80.7%→86.3%, margin +51.0%→+56.7% — modest, consistent, real improvement.

**Correction found during B1 adaptation attempt:** this checkpoint was trained with `lambda_body=0` (default), so it has no `body_head` module at all — the launch command omitted `--lambda_body/--body_dim/--body_channels`. /mean-z and discrimination results are unaffected (read off z/FTM rollout directly, per the project's stop-gradient convention), but B1 adaptation is blocked until retrained (`beh12_hinge_multistep_anchor_v2`). Process note: the 1-epoch smoke test already showed the missing `body` loss term but wasn't checked against the reference config before the full run launched — verify every loss term against a reference checkpoint's config before a real run.

**Unblocked 2026-09-12:** v2 retrained clean with `lambda_body=0.5` etc.; re-verified more action-sensitive than the original at every horizon (0.757/0.736/0.550/0.526/0.631 vs 0.938/0.886/0.892/0.888/0.893).

First B1 adaptation (plain `wm.adapt`, no hinge) scored 8.3% (2/24) on F195's gate — worse than F195's own expert-fit baseline (16.7%), no real margin over 5.9% chance. Diagnosed: `wm.adapt`'s stage 1 uses only one-step MSE, dropping the pretrain's hinge/readout/rollout terms entirely; a direct check (`b1_adaptation_sep_check.py`) confirmed this erodes pretrain's hinge-built separation by 38-62% over 1000 steps — F142's MSE-dominance mechanism recurring inside B1 adaptation.

Fix: optional K=1 hinge term added to `wm.adapt` (`--lambda_hinge`, `--hinge_margin`, off by default). K=1 (not the pretrain's K=2) since `wm.adapt`'s loss is already one-step. Re-running with `--lambda_hinge 0.5 --hinge_margin 0.1`: separation retention improved 38%/62%→49%/77% at steps 1/2.

Full pipeline result on F195's gate:

| | hit rate | chance |
|---|---|---|
| F195, expert-fit (old pretrain) | 16.7% (4/24) | 5.9% |
| F195, babble-fit (old pretrain) | 4.2% (1/24) | 5.9% |
| this checkpoint, expert-fit, stage 1 WITHOUT hinge fix | 8.3% (2/24) | 5.9% |
| this checkpoint, expert-fit, stage 1 WITH hinge fix | 20.8% (5/24) | 5.9% |

Best B1 reward-quality-gate result in project history. F199's hexapod-level improvement does propagate to B1, provided B1's adaptation stage doesn't silently discard it via plain-MSE fine-tuning — the bottleneck was in B1's adaptation procedure, not a fundamental ceiling (a distinction F201, itself corrected, initially failed to draw).

Scripts: `scripts/diagnostics/objective_experiments/family_z_ceiling.py`, `scripts/diagnostics/objective_experiments/b1_adaptation_sep_check.py`, `wm/adapt.py --lambda_hinge/--hinge_margin`. Training flags: `--lambda_hinge 0.5 --lambda_readout 1.0 --lambda_rollout 1.0 --hinge_K 2 --hinge_margin 0.1 --lambda_recon 1.0`. Checkpoints: `wm/runs/beh12_hinge_multistep_anchor/best.pt`, `wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/`.

---

### F200. Current native-Coppelia B1 babble: claim-honest and visibly walkable as a preview, but still below the pretraining Froude band

Since F196, native CoppeliaSim-Bullet B1 now has a usable generic motor-babble preview (not just a dynamics probe). Safe seed: fixed generic diagonal duty-cycle CPG — 65% planted stance, 35% raised return, 2.0 Hz, amplitude 0.24, calf ratio 2.5, per-step Gaussian motor noise 0.03. No trained policy, retargeting, demonstrations, body-feedback, or command input — keeps the claim line as generic CPG + exploration noise.

Best safe rendered run: `results/wm/dataset/b1_babble/coppelia_fast_duty_candidate/f2.0_a0.24_s9_ego.{npz,mp4,yaml}` — 160 steps/8s upright, egocentric, moved +0.596m, mean body-frame Froude [+0.0328, +0.0026, -0.0062] (forward-dominant, low lateral/yaw leakage). A harder edge setting (`f2.0_a0.30_s10_allo`) reached [+0.0366, +0.0029, -0.0115] allocentric, but the paired egocentric rerun fell — 0.24 is the current safe setting.

Ruled out: replaying old MuJoCo-scale sine amplitudes in Coppelia generates larger instantaneous motion but mostly dumps energy into lateral roll and falls. Diagnostic phase/sign-convention knobs (`--generic-trot-pairing`, `--generic-thigh-sign-layout`, `--generic-calf-sign-layout`) show some signed sine variants briefly reach forward Froude ~0.03-0.05 but lose height before 8s completes. The support-biased duty-cycle waveform remains the robust open-loop option.

Current status: physically usable and honest as a visual preview, but not the final babble dataset — stable forward Froude is still ~3-6x below the useful pretraining/expert region (~0.10-0.20). Final collection still needs a frozen parameter distribution with every sampled rollout retained, including falls. Clean/expert Coppelia controller remains undone.

Scripts: `sim/collect/collect_b1_coppelia_babble.py`, `sim/collect/collect_b1_coppelia_fast_duty_preview.py`. Handoff: `results/wm/dataset/b1_babble/q22_handoff_prompt.md`.

---

### F201. B1's reward-quality gate: six fixes tried, the sixth (carrying hexapod pretrain's hinge separation through B1 adaptation) initially appeared to work

**WITHDRAWN by F228 (2026-09-18).** Fix #6's headline 20.8% (5/24) does not reproduce on the same checkpoint: 5 reseeds at original sample size landed 4.2-8.3%, 2 further seeds at 4x sample size landed 1.0-2.1% vs a 3.0% chance rate. Treat this entry's table/numbers as historical; the real result is closer to F195's at-or-below-chance finding. See F228 for the re-test.

Fixes tried and outcomes:
| # | fix | result |
|---|---|---|
| 1 | stage-3 contrastive (`wm.adapt3`) | `family` 30% vs chance 27% -- not selection |
| 2 | F141 hinge (no anchor) | diverged 3-4x past frozen-frame baseline |
| 3 | F178 counterfactual targets | null |
| 4 | recurrence (4 variants) | all fail (F198) |
| 5 | F199 hinge + multi-step anchor, pretraining only | real hexapod improvement, but B1 `wm.adapt` (plain one-step MSE) erodes 38-62% of it |
| 6 | F199's fix carried through B1 adaptation (`wm.adapt --lambda_hinge`, K=1) | claimed 20.8% (5/24) -- now withdrawn, see above |

Mechanism: B1 `wm.adapt` stage 1 originally used a single one-step MSE loss with no separation term, silently discarding action-sensitivity from pretraining (measured via `b1_adaptation_sep_check.py`, 38-62% loss of `sep`). Adding K=1 hinge to stage 1 cut that loss to 23-51%, but the resulting gate improvement did not reproduce (see withdrawal above).

Scripts: `scripts/diagnostics/objective_experiments/b1_adaptation_sep_check.py`, `wm/adapt.py --lambda_hinge/--hinge_margin`. Checkpoint: `wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/`.

---

### F202. F201's hinge fix does not transfer to babble-fit data; failure isolates to babble data quality, not the fine-tuning method

On `data/egocentric/b1_babble_ego_flat` (the actual target scenario, no expert library), held-out rollout ratio (before/after `wm.adapt`) went 1.27/1.27/1.25/1.38 -> 1.82/1.88/1.88/1.86 at horizons 1/3/5/10 with `--lambda_hinge` -- worse than doing nothing at every horizon. With `--lambda_hinge 0` (plain MSE) on identical babble data: 1.82/1.87/1.88/1.88 -- nearly identical, ruling out the hinge term as the cause.

Consistent with F194's "visually unstable" babble observation: a small (9-clip) adaptation fit overfits to erratic motion rather than learning generalizable dynamics, regardless of loss method.

Implication: neither the candidate-scoring path nor the RL-controller path (Q21 step 3) can be adapted to a genuinely unseen body until babble motion quality improves (MuJoCo CPG source F190-F194, or CoppeliaSim-native replacement F200, not yet at usable Froude range) -- only to B1, which has an expert library and isn't the target scenario.

Scripts: same as F201, run on `data/egocentric/b1_babble_ego_flat`. Checkpoints: `wm/runs/beh12_hinge_multistep_anchor_v2/b1_babble_adapt_hinge/`, `.../b1_babble_adapt_nohinge/`.

---

### F203. The Q21 step 3 RL controller's action space could not reach high-scoring joint-command magnitudes, and `body_head(proj(action))` is blind to time

Building `wm/policy/b1_mujoco_env.py`/`b1_real_ppo.py` surfaced two bugs.

Bug 1 -- action-space reachability: PPO actor's tanh output is `[-1,1]` per joint, but `proj`/`body_head` were fit on `beh12_b1_ego_flat`'s recorded babble actions, which are asymmetric per-joint (hip `[-1.21,0.32]`, thigh `[-1.67,1.21]`, calf `[0.85,3.59]`, one-sided). Random actions confined to `[-1,1]` reached max tracking score 0.007, same as standing still (0.0068). Fixed with per-joint affine `scale_action()` (`ACTION_LO`/`ACTION_HI` in `b1_mujoco_env.py`/`b1_coppelia_env.py`); after fix, random actions reach tracking up to 0.79, higher than any real expert-clip step (0.61).

Bug 2 (a limit, not fixable) -- reward is memoryless: shuffling a real high-scoring expert action sequence's time order and rescoring gives an identical score (0.2118 vs 0.2118). Real expert actions do score far higher than near-still ones (0.21 vs 0.01-0.03), so the reward isn't degenerate, but it can't distinguish coherent walking from the same actions scrambled -- consistent with F142's <3% action-causality finding. `body_head(proj(action))` reads only instantaneous joint-command shape, not temporal pattern.

Scripts: `scripts/diagnostics/objective_experiments/reward_quality_gate_b1.py` convention, re-run manually on `data/egocentric/beh12_b1_ego_flat/b1_ep*.npz`. Fix: `wm/policy/b1_mujoco_env.py`, `b1_coppelia_env.py`.

---

### F204. Four fixes to the from-scratch B1 PPO controller, including a ground-truth velocity reward, all give the identical null result

| # | Fix | Result |
|---|---|---|
| 1 | F203's action-space reachability fix | tracking flat ~0.09 across 50 updates |
| 2 | `TRACKING_SCALE` recalibrated for true-Froude units | tracking flat ~0.09 |
| 3 | fall no longer ends episode (penalized, pose reset, continues) | `died_frac`->0%, tracking flat ~0.09 |
| 4 | AR(1)-correlated exploration noise (~0.4s) | tracking flat ~0.09; entropy grew unboundedly instead (reverted) |

Every configuration converges to a deterministic frozen single joint pose; measured true Froude (`info["true_froude"]`, from real MuJoCo physics, independent of reward) is within noise of exactly zero in all four.

Fixes #2-4 tested with `reward_mode="true_froude"` (real measured base velocity/yaw_rate, not the WM proxy) gave the identical failure -- rules out WM reward quality/temporal blindness (F203) and fall-risk aversion as sole blockers.

Reference (`sim/assets/b1_policy/base_gait3/`, Unitree/Isaac-Lab-trained) is the same feedforward-MLP architecture but adds: sin/cos gait-phase clock in observation, `gait_phase_tracking` reward term, actuator domain randomization, and 4096 parallel envs x 800 iterations (~10^8 steps) vs this project's ~10^5 steps single-env (>1000x fewer samples).

Sample-budget test: `VecB1MuJoCoEnv` at `--n_envs 16` (~16x prior throughput) still gave tracking 0.090-0.093 through 18 updates, flat -- rules out modest parallelism increase alone as a fix (16 envs is still ~250x short of reference's 4096); suggests something structural, not just scale.

Scripts: `wm/policy/b1_mujoco_env.py`, `b1_coppelia_env.py`, `b1_real_ppo.py`, `b1_ppo_eval.py`. Runs: `wm/runs/beh12_hinge_multistep_anchor_v2/b1_ppo_mujoco_scalefix.pt`, `b1_ppo_mujoco_truefroude.pt`, `b1_ppo_mujoco_truefroude_nofallend.pt`, `b1_ppo_mujoco_corrnoise.pt`.

---

### F205. Command curriculum and feet-air-time reward are also null, closing out the RL-controller mechanism search

Command curriculum (`--curriculum_updates`, `goal_scale` in `wm/policy/b1_mujoco_env.py`, mirroring `velocity_env_cfg.py`'s `lin_vel_cmd_levels`): observed decay ratios across 16 updates (0.954...0.944) cluster tightly around the frozen-policy prediction (~0.9416), meaning the declining curve is fully explained by the target getting harder, not by any policy learning.

Feet-air-time reward (`AIR_TIME_WEIGHT=0.1`, `AIR_TIME_THRESHOLD=0.5s`, `_feet_air_time_reward`), matched to `velocity_env_cfg.py`'s exact weight/threshold: confirmed alive under random actions (74/200 steps triggered nonzero event). Trained 51 updates (`--n_envs 16`, `reward_mode=true_froude`): tracking stayed 0.0908-0.0913 throughout -- same flat null.

Summary: six independent fixes (action-space reachability F203, reward-scale calibration, no-fall-termination, correlated exploration noise, command curriculum, air-time shaping) all produce the identical null -- deterministic frozen joint pose, true Froude ~0 -- tested on B1's best-case real expert-fit checkpoint (`body_head_b1_hex.pt`). Compute scale tested (~10^6 total steps) vs reference's ~10^8. Two live options remain: (a) much larger compute budget (unverified), or (b) return to candidate-scoring line -- though F201's B1-only success is withdrawn (F228), leaving babble data quality (F202) as the only remaining open item on that line.

Scripts: `wm/policy/b1_real_ppo.py` (`--curriculum_updates`), `wm/policy/b1_mujoco_env.py` (`goal_scale`/`set_goal_scale`, `_feet_air_time_reward`). Runs: `wm/runs/beh12_hinge_multistep_anchor_v2/b1_ppo_mujoco_curriculum.pt` (not saved), `b1_ppo_mujoco_airtime.pt`.

---

### F206. F203's own action-space fix was itself a bug, plus an independent per-step velocity measurement bug -- explaining every null in F204/F205

F203's `scale_action()` remapped policy output onto `beh12_b1_ego_flat`'s recorded actions, which are Isaac Lab's unbounded (no-tanh) expert policy outputs, not a physical requirement. A real CPG gait applied via the standard `DEFAULT_IL + ACTION_SCALE * action` mapping (values in `[-1,1]`) gives 2.12 m forward travel in 5s (Froude 0.196); the same action through `scale_action()` gives Froude ~0.005. Reverted: `ACTION_LO`/`ACTION_HI`/`scale_action()` removed from `b1_mujoco_env.py`/`b1_coppelia_env.py`.

Second bug: `body_velocity`/`yaw_rate` (`wm/data/embodiment.py`) smooth over one stride (`BODY_WINDOW_S=1.0s`, ~50-sample kernel), but `b1_mujoco_env.py`'s per-step `true_froude` called them with only 2 samples per step, diluting the result (correct gait measured Froude 0.0025 vs actual 0.196). Fixed with a rolling `pos_hist`/`quat_hist` buffer recomputed each step.

Combined fix verified: real CPG gait measures Froude 0.155, mean tracking_reward 0.336, vs the ~0.09 ceiling every F204/F205 config was stuck at. Both bugs, not the six tested mechanisms, explain F204/F205's nulls.

CORRECTION after watching the render: a policy reported here as "a real walking policy" (forward Froude 0.113 matching goal 0.105, "0.446 m in 4s", "upright throughout") does NOT walk -- it falls twice per 4s episode, holds a collapsed posture (mean height 0.405 vs nominal 0.56), and lunges forward at 0.56 m/s (~2x normal). The reported travel was measured across fall-reset teleports; "100% survival" came from an evaluator checking `fell` only on the final step. Actual accumulated travel is 2.25 m interrupted by two falls -- a fast unstable lunge, not locomotion, caused by removing episode termination on falls (making falling cheap; a forward dive maximizes forward-Froude reward). `b1_ppo_eval.py` now counts fall events, mean height, and teleport-excluded travel.

Scripts: `wm/policy/b1_mujoco_env.py`, `b1_coppelia_env.py`.

---

### F207. 2x2 re-run on the anchored-hinge checkpoint: rollout's failure confirmed structural; vision-goal "regression" withdrawn

> **F256 (2026-09-25): the `--mechanism rollout` half of this entry is invalid.** Its rendered closed loop fed the planner a 24-deg egocentric view (training clips: 90) from a body whose pitch/roll/height drifted across candidate switches. Rollout result files moved to `results/_invalid_F256/`. `--mechanism direct` numbers stand (direct never reads the loop's camera).

> The vision-goal "regression" half of this entry is WITHDRAWN, see F208: numbers below were measured at an untrained frame spacing (5, deployed) vs the trained spacing (1), which depressed all vision-goal numbers and hit the newest checkpoint hardest. Read at the trained spacing with a properly fitted head, the vision goal matches a measured one exactly (100% vs 100%).

Re-ran F188's 2x2 (direct/rollout x physics/vision goal) on the anchored-hinge checkpoint (`wm/runs/b1_adapt/` old checkpoint no longer exists; old column is F188's recorded values):

| mode | F188 (old ckpt) | anchored-hinge ckpt |
|---|---|---|
| A direct + physics | 78% family, dist 0.038 | 71% family, dist 0.029 |
| D direct + vision | 80% family, dist 0.038 | 33% family, dist 0.179 (spacing artifact, see withdrawal) |
| B rollout + physics | 47% family, dist 0.290 | 35% family, dist 0.233 |
| C rollout + vision | 40% family, dist 0.290 | 38% family, dist 0.233 |

Rollout does not revive with the better checkpoint: 35-38% vs direct's 71%, still picks a sideways clip against a forward-dominant goal, unaffected by a perfect goal (B vs C nearly identical) -- reproduces F118's mechanism that the rollout path ignores the goal.

Rollout re-tested at the inverse model's own training spacing (adjacent-pair, matching how it's trained) with a perfect goal:
| planner horizon | % right family | distance |
|---|---|---|
| 1 (matches training) | 20% | 0.233 |
| 2 | 27% | 0.233 |
| 5 (original) | 35-40% | 0.233 |

Shorter horizon is worse, opposite of what a spacing-artifact hypothesis predicts -- rollout fails structurally across three horizons with a perfect goal and best head, picking the same farthest-from-goal candidate every time.

Scripts: `sim/control/close_loop_direct_froude.py` (4 modes). Runs: `results/wm/closed_loop/direct_froude_hinge_v2/`, `results/wm/closed_loop/rollout_h1/`, `rollout_h2/`.

---

### F208. One flag served two unrelated jobs and depressed every vision-goal number in the project; fixed, vision goal matches measured goal exactly (100% vs 100%)

Every fitting stage (inverse model, `wm.adapt` stage 1, `wm.fit_projector`, `wm.fit_body_head`) trains on adjacent (spacing-1) frame pairs, except the forward-model rollout anchor (`rollout_k=2`, an auxiliary term, not on the reading path). But the deployed goal read used spacing 5, because `close_loop_direct_froude.py`'s `--horizon` set both the planner's rollout depth (5, deliberately, per F131/F141) and the goal-read spacing, which should have been 1. Same default existed in `read_vision_goal` in both RL env files. Fixed: `--goal_horizon`/`GOAL_HORIZON` now separate; goal cache key includes spacing.

Second defect: `wm.fit_body_head` only held out validation data for the adapted body, not the `--also` body -- so the source body's read quality (the one goals are actually read from) was never measured. Fixed: `--also` bodies now get their own clip-level split.

With both fixed (head refit 6000 epochs, source body validated: held-out ratio B1 0.662, hexapod 0.608), goal-read error at trained spacing (1) vs deployed spacing (5): best head 0.0170 median (77% of clips under threshold) at spacing 1 vs 0.0889 (4%) at spacing 5.

Closed-loop result: measured (privileged) goal and vision goal (read from other body's video) both give top pick `turn_w0.008`, 100% family accuracy, distance 0.029 (the optimal choice among 12 candidates) -- indistinguishable from each other, both beating F188's original 78%/80%.

Implication: F188's 0.0293 error was this bug plus a favorable clip; F207's claimed regression is withdrawn (newest checkpoint was measured at an untrained spacing). F207's rollout finding stands: with near-perfect goal (0.0097) rollout still selects at 40%, 3x farther from goal than optimum.

Lesson: where a spacing/horizon/window appears in more than one place in the pipeline, it needs its own name; metrics must be computed in the units the system actually scores in.

Scripts: `sim/control/close_loop_direct_froude.py` (`--goal_horizon`), `wm/fit_body_head.py` (`--also` validation split), `wm/policy/b1_mujoco_env.py`, `b1_coppelia_env.py` (`GOAL_HORIZON`). Runs: `results/wm/closed_loop/direct_froude_v2head_gh1/`. Checkpoint: `wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1_hex_v2.pt`.

---

### F209. CoppeliaSim/Bullet rollouts are not reproducible (invalidating a day of n=1 screening); the real cap on B1 babble speed was Bullet's missing joint damping

CoppeliaSim/Bullet is not deterministic like MuJoCo: five identical runs (same seed/flags/scene) gave forward Froude 0.1206-0.1322 with yaw changing sign between runs, and near a stability boundary the upright/fell verdict is close to a coin flip. All n=1 conclusions from one session had to be discarded (e.g. "wave gait caps at 0.097" was actually 0.163-0.183 at n=3-8). Protocol going forward: minimum 3 repeats per config, report upright fraction and spread, never a single number.

Real cause of the low speed cap: Bullet exposes no `*_joint_damping`/`*_joint_frictionloss` parameter (only `bullet_joint_pospid1/2/3`, `normalcfm`, `stopcfm`, `stoperp`); only MuJoCo/Vortex expose joint damping/friction. Same open-loop CPG on MuJoCo: `b1_flat_real.xml` (system-identified damping/friction) gives +1.997 m over 160 steps; `b1_flat.xml` (uniform placeholder) gives -0.234 m (walks backward). Joint friction is what makes an open-loop CPG walk; the Coppelia scene had no equivalent.

Fix: use `jointdynctrl_spring` (Coppelia's spring mode runs `tau = K(q*-q) - C*qdot` in-engine), with K/C set from `b1_flat_real.xml`'s own numbers (K = kp 550/700/970, C = kv+damping = 2.745/4.700/5.201). Took forward Froude from 0.140 to 0.220 on the same gait, 8/8 upright, spread 0.214-0.221 -- no scene/engine change needed.

Ruled out (each at n>=3): ground/foot friction, link masses, joint velocity ceilings, joint ranges, PID gains 600-900. Hip-abduction stance widening fixed roll instability but only via unrealistic 36-degree splay (dropped trunk to 0.30m vs 0.52m nominal) -- not adopted.

Frozen config for data collection: diagonal trot, 5.0 Hz, amplitude 0.30, duty 0.55, spring mode -- 8/8 upright, Froude 0.214-0.221, cleanest attitude (up.z 0.985-0.988); this is B1's shipped gait.

Not yet safe to collect on: parameters were searched against the hexapod goal band, risking fitting the generator to the answer. Before collection: declare sampled ranges up front, justify by uprightness alone, randomize, retain all rollouts including falls, report Froude coverage as an outcome.

Scripts: `sim/collect/collect_b1_coppelia_babble.py` (`--joint-control spring`, `--spring-k/-c`, `--contact-friction`, `--max-joint-vel`). Data: `results/wm/dataset/b1_babble/coppelia_spring_preview/`.

---

### F210. Vision-read goal matches privileged recorded goal across the full goal set; the "threshold" argument that used to justify F188's single-clip result was arithmetically wrong

F188's claim (read error 0.029 "costs nothing" because smaller than "the gap between the two closest candidates," 0.033-0.038) misread the numbers: 0.033/0.038 were two candidates' distances to the goal, not the gap between them. The actual gap is 0.0079 -- a 0.029 read error is 3.7x larger.

Measured directly (perturb goal by fixed magnitude in 2000 random directions, count top-pick changes):
| read error | top pick changes |
|---|---|
| 0.0293 (F188's clip) | 65% |
| 0.0170 (median, 48 clips) | 42% |
| 0.0079 | 6.5% |
| 0.0040 | 0% |

So F188's single-clip agreement was luck; no code ever computed the "~0.033 threshold" inherited by F208/proposal.tex/update_slide.md -- it was prose only, now void.

Replacement, measured over the full goal set (`final_2x2x2_test.py`, 12 hexapod goal conditions x 8 steps = 96 decisions, `body_head_b1_hex_v2.pt`, `--goal_horizon 1`):
| goal source | free_offset | family accuracy | chance | froude dist (median) |
|---|---|---|---|---|
| A physics (privileged) | no | 90% | 56% | 0.0616 |
| A physics | yes | 92% | 56% | 0.0450 |
| D vision | no | 92% | 56% | 0.0433 |
| D vision | yes | 92% | 56% | 0.0452 |

Vision goal matches or beats the privileged recorded goal on every metric across the whole set (forward goals 64/64, lateral 24/32). Downstream failures (candidate scoring, rollout, RL reward, controller) are not caused by goal reading. Chance is 56% (8/12 goals forward-family) so 92% is a smaller margin than it looks; no yaw-family goals exist in this set (hexapod `turn_*` clips are forward-dominant), so turning is untested here.

One bug fixed to run this: `final_2x2x2_test.py` still passed `--horizon` (planner window, default 5) into `vision_goal` as frame spacing (F208's bug), missed in the earlier fix elsewhere. Split into `--goal_horizon` (default 1). Every mode C/D number this script produced before 2026-09-14 is not comparable to the table above.

Scripts: `scripts/diagnostics/objective_experiments/final_2x2x2_test.py`.

---

### F211. F209's new spring-mode babble adapts better than expert data on a matched protocol, but a new-vs-old-babble comparison stays confounded by composition, not physics

Ablation (`wm.adapt`, `beh12_hex-b1_body3/best.pt`, B1, 1000 steps, lr 1e-4), matched clip count/selection between new babble and expert:

| babble source | clips (train/test) | selection | h1 before/after | h3 | h5 | h10 |
|---|---|---|---|---|---|---|
| new, spring-mode (F209), 0 falls/40, fwd 0.124-0.216 | 30/10 | stratified | 1.17 / **1.76** | 1.04 / **1.74** | 0.97 / **1.66** | 0.92 / **1.47** |
| expert, `beh12_b1_ego_flat`, matched | 30/10 | stratified (matched to new-babble run) | 1.09 / **1.60** | 0.94 / **1.62** | 0.85 / **1.53** | 0.77 / **1.30** |
| old, `b1_babble_v2_ego_flat` (position-control CPG, pre-F209) | 26/10 | plain permutation (v2 files carry no `condition` field, so `--stratify` errors) | 1.19 / **1.82** | 1.10 / **1.83** | 1.04 / **1.78** | 0.99 / **1.70** |

With the confound removed (expert row matched to new-babble row in clip count and selection method), new babble beats expert at every horizon (1.76 vs 1.60, 1.74 vs 1.62, 1.66 vs 1.53, 1.47 vs 1.30). That comparison is clean and stands. The old-babble row does not: different clip count and selection method, so "old >= new" cannot be read off it.

The "old babble ties/beats new babble" reading is WITHDRAWN: the two sets are not comparable in content, read directly from each set's own metadata:

| set | n | behavioural composition |
|---|---|---|
| expert (`beh12_b1_ego_flat`) | 48 | 3 distinct behaviours -- forward-speed, turn, sideways -- 4 sub-levels x 4 seeds each |
| new babble (spring, F209) | 40 | forward-trot only; varies solely by random seed (per-step noise + amplitude draw in [0.20, 0.32]); zero lateral or yaw variety, because F209's sampling-range declaration only covered the forward-trot stability point |
| old babble (`b1_babble_v2_ego_flat`) | 36 | filenames (`fwd_f*`, `lat_f*`, `yaw_f*`) imply fwd/lat/yaw families exist, but the npz files carry no `condition` field at all -- unverifiable without re-deriving from filenames, and the reason `--stratify` errored on it |

So old babble very plausibly ties or beats new babble because it has direction variety and the new babble has none, not because F209's Coppelia physics fix failed to help anything. F209's fix has only ever been validated at the forward-trot point -- no lateral or turning babble has been collected under spring-mode control yet, so comparing "new babble" against sets that include turning and sideways motion compares a narrower thing to a broader one by construction. The rollout-ratio numbers are kept for the record, but the conclusion drawn from them (old >= new) does not follow and should not be cited.

What actually needs to happen, cheaply, before spending more adapt-job time: extend F209's declared-range methodology to lateral and turning gaits under spring-mode control, collect a matched-composition babble set (3 behaviours, comparable clip counts, a real `condition` field), and only then re-run this ablation. Checking composition first costs a `glob` and a dict read; running an unneeded `wm.adapt` costs ~5-10 minutes of VJEPA2 encoding each -- do the former first.

Checkpoints: `wm/runs/beh12_hex-b1_body3/b1_adapt_spring/adapted_b1.pt` (new babble), `wm/runs/beh12_hex-b1_body3/b1_adapt_expert/adapted_b1.pt` (expert), `wm/runs/beh12_hex-b1_body3/b1_adapt_oldbabble/adapted_b1.pt` (old babble).

---

### F212. B1's `teacher_student_insect.py` port: two bugs fixed before they could confound the result; a severe train/eval physics gap flips which student fails how; neither clears the pre-registered bar

Ported the insect's cloning-only stage (not `improve()` -- local-perturbation DAgger ranking already shown to fail for a task-level reason, fixed-magnitude action perturbations barely change real physics, F135/F136/F138, not insect-specific so not re-tested). Input is proprioceptive (34-d: 12 joint pos + 12 joint vel + 4 base quat + 6 base lin/ang vel, matching `B1MuJoCoEnv._obs()`'s own convention exactly) -- no CoppeliaSim or VJEPA2 anywhere in this pipeline. Base velocity isn't recorded in the clips, so it's finite-differenced (central difference: 2.1% mean-abs-error vs live MuJoCo `qvel` on linear velocity, 18.1% on angular, validated by re-simulating a clip and comparing before trusting it as training data; a one-step forward difference measured worse, 8.5%/39.8%, confirmed not assumed). Data: `data/egocentric/beh12_b1_ego_flat`, confirmed balanced (16/16/16 across speed/turn/side) before training. Teacher checkpoint: `wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/teacher_b1.pt` -- correction (F214): F210 actually used `body_head_b1_hex_v2.pt`, not this checkpoint. `teacher_b1.pt` is still the right choice here, but for a narrower reason: `clone_b1()` reads only `cfg.body_channels` from it, never `body_head` (F214 confirms this precisely), and `cfg` is identical across the whole `b1_adapt_hinge` family, so which family member is named makes no difference to this stage.

Two real bugs found and fixed before they could produce a false result, both caught by watching video rather than trusting a number: a first pass reported "246% of D_real, PASS" on a 20-epoch smoke test, implausibly good, and watching it showed the robot walking backward at an odd cadence.
1. `D_real` was computed from the wrong initial condition: `rollout_b1_mujoco.py` runs `--policy_warmup 45` steps of real walking before recording starts, so a clip's own `actions[0]` was applied to an already-moving body mid-gait, never a freshly reset standing one. Replaying it from `env.reset()`'s static pose gave -0.23 m (wrong direction) against the clip's own recorded +1.25 m, same action sequence, same model file -- not a sign bug, a mismatched initial state. Fixed by seeding MuJoCo's `qpos`/`qvel` from the clip's own recorded frame-0 state (`seed_from_clip()`).
2. The recorded expert actions are unbounded (min/max -1.42/3.50, 32% of all values exceed |1|) -- an unclipped Isaac Lab actor output, the same class of value F203/F206 already documented for this dataset. `B1MuJoCoEnv.step()` clips the raw action to [-1, 1] before scaling, correct for a tanh-squashed RL actor but wrong here: it silently truncated a third of the commanded targets. Fixed by reproducing `rollout_b1_mujoco.py`'s own convention exactly (`apply_action_unclipped()`: clamp only the final scaled target at the physical actuator range).

With both fixed, the expert's own recorded actions replay at 95.5% fidelity (1.216 m replayed vs 1.273 m recorded, correct direction) under `b1_flat_real.xml` -- comparable to F133's insect replay ratio of 1.017, and now trustworthy as `D_real`.

The train/eval physics gap is real and severe, not just theoretical. `beh12_b1_ego_flat` was collected on `sim/assets/b1_mujoco/b1_flat.xml` (uniform placeholder joint physics, `rollout_b1_mujoco.py:21`); `B1MuJoCoEnv` defaults to `b1_flat_real.xml` (system-identified damping/friction -- F209's own finding that this is what makes an open-loop gait viable at all). Evaluated on both, per student:

| student | `b1_flat_real.xml` (eval default) | `b1_flat.xml` (the collection physics) |
|---|---|---|
| forward-only (R2 0.999 offline) | 52% of `D_real`, FALLS (min height 0.13 vs 0.56 settled) | 35%, upright (min height 0.53, barely dips) |
| all 3 behaviours (R2 0.995 offline) | 30%, upright (min height 0.41) | 7%, upright (min height 0.53, near-stationary) |

No cell clears the pre-registered bar (upright the whole window AND >= 50% of `D_real`). But the failure mode flips entirely with the physics, not just the magnitude: under the higher-damping `b1_flat_real.xml`, the student that travels farthest (52%) is the one that falls hardest; under `b1_flat.xml` (matching what it was actually cloned on), both students are stable but barely move. Read together with the offline R2s (0.995-0.999, matching F135's own warning that held-out regression fit does not predict closed-loop rollout quality): the student has learned *a* action pattern that reproduces recorded joint targets well in isolation, but that pattern's real-world behaviour depends heavily on which joint damping/friction it is executed against -- a policy fit on one physics model's dynamics does not transfer its stability margin to another's, even when both are B1 in name.

Reading against the two precedents this project already has: F133's insect clone stayed up and undertravelled (36%); the informally-found `wm/runs/students/b1_bc_ego_forward.pt` (vision-embedding input, not reproduced here, cited only from F179) travelled far and fell (92%). This B1 proprioceptive clone reproduces BOTH patterns depending on which physics it's dropped into -- upright-but-slow on the matched physics, fast-but-falling on the mismatched one -- suggesting the two "modes" seen across this project's B1/insect attempts are not properties of the input representation (vision vs. proprioceptive) or the specific checkpoint, but of how well the evaluation physics matches whatever implicit dynamics assumption the cloned action pattern carries.

This does not settle whether B1 behaviour cloning can pass the bar -- neither run was tuned, and the physics-mismatch confound was uncontrolled going in. It does rule out reading either FAIL as evidence about the cloning mechanism itself, since the SAME weights produce a qualitatively different failure depending only on which physics they're asked to act in. The clean next test, not yet run: re-collect (or re-render) B1 clips on `b1_flat_real.xml` so training and evaluation share one physics model, removing this confound before judging the mechanism.

Scripts: `sim/control/clone_b1.py` (`bc`/`eval` stages -- see F216 for other mechanisms tried against this baseline afterward, none survived, so the file holds only this control). Checkpoints: `wm/runs/students/b1_bc_forward.pt`, `wm/runs/students/b1_bc_all.pt`. Videos: `results/wm/students/eval_{forward,all}{,_flat}.mp4`.

---

### F214. `teacher_b1.pt` was wrongly used as "the teacher" for two z-sharing diagnostics despite never having a fitted `body_head`; correcting to `body_head_b1_hex_v2.pt` reverses both results

`wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/teacher_b1.pt` has no `body_head_fit` key -- its `body_head` is whatever the original hexapod-only pretrain left it at (`lambda_body=0.5` during joint training, never touched by `wm.adapt`'s stage-1 loss, which is pure next-embedding MSE with no body term). `body_head_b1_hex_v2.pt` is the checkpoint that actually went through `wm.fit_body_head` (6000 epochs, B1 data) -- the one F210 used for 92% family accuracy. Both scripts tonight (`body_head_hidden_share.py`, `linear_vs_itm_froude.py`) defaulted to `teacher_b1.pt` instead. ITM and the projector are identical between the two checkpoints (`body_head_b1_hex_v2.pt`'s own metadata: `body_head_fit.source = teacher_b1.pt`, body_head was fit on top of it, everything else inherited unchanged) -- confirmed directly, since raw-`z` numbers are bit-identical across both checkpoints.

Result 1 (linear-pair-mapper vs `body_head(ITM(.))`, 48 hexapod clips, 36/12 held-out split): with wrong checkpoint, linear ridge held-out median err 0.0202, `body_head(ITM(.))` 0.0669 (loses to linear), R2 +0.292. With correct checkpoint, linear unchanged (baseline doesn't use the checkpoint), `body_head(ITM(.))` 0.0168 (beats linear), R2 +0.736.

Result 2 (cross-embodiment same-behaviour clustering, cosine gap, 18 clips, hexapod+B1): raw z (64-D) gap 0.080 identical in both (same ITM). body_head hidden (128-D) gap 0.027 -> 0.007. body_head Froude output (3-D) gap 0.041 (25% of within-body signal, wrong checkpoint) -> 0.672 (89% of within-body signal, correct checkpoint), held out 3-fold: 0.75/0.89/0.91, all three folds strongly positive.

With the properly-fit checkpoint, `body_head`'s Froude output clusters strongly by behaviour across the two bodies -- 89% of the within-one-body behaviour signal survives crossing embodiments, validated with a proper train/test split (identity direction fit on 2 seeds per cell, tested on the third, never seen together). This directly supports the thesis's actual claim (`report/proposal.tex`'s "shared body-motion coordinate", not a shared `z`) with geometric clustering evidence, not just F210's candidate-selection accuracy. The linear baseline, which briefly looked like it beat `body_head(ITM(.))` on the wrong checkpoint, does not beat it on the checkpoint actually fit for this task.

Unaffected: F212's BC results. `sim/control/clone_b1.py`'s `clone_b1()` reads only `cfg.body_channels` from its `--base_ckpt` argument, never `body_head` itself, and `evaluate_b1()` constructs `B1MuJoCoEnv` with `goal_std=np.zeros(3)`, never querying `body_head`'s output in the rollout loop.

Rule going forward, stated so it isn't tripped over again: `teacher_b1.pt` is valid whenever only `itm`/`projector`/`cfg` are needed (identical in both checkpoints). `body_head_b1_hex_v2.pt` is required whenever `body_head`'s actual fitted output matters -- check a checkpoint's own `body_head_fit` key before trusting its `body_head`, rather than assuming "the teacher checkpoint" is interchangeable with "the one with a properly fit body_head".

NOTE: F215 later found this linear-vs-ITM comparison (Result 1) suffered a data-split leak and, on a clean split, retracted "ITM beats linear/MLP." Result 2 (clustering) is a separate measurement, not touched by that retraction, though not yet re-run on the clean checkpoint either.

Scripts (both fixed in place, no new files): `scripts/diagnostics/cross_embodiment/body_head_hidden_share.py`, `scripts/diagnostics/cross_embodiment/linear_vs_itm_froude.py`.

---

### F215. ITM's Froude read vs linear/MLP baselines on the raw frame pair: initial "ITM wins" result had a data leak; on a clean split, ITM loses to both baselines, and the loss traces to `z`'s lossy compression, not to the fitting

F214's corrected result (linear ridge 0.486 R2, `body_head(ITM(.))` 0.736 R2, held out) left one gap unclosed: does ITM's own transition-model structure matter, or would *any* nonlinear function of the same `concat(pooled(e_t), pooled(e_next))` pair close most of that gap? Added a second baseline to `linear_vs_itm_froude.py`: an MLP with `body_head`'s exact shape (LayerNorm, Linear->128, GELU, Linear->3), same capacity as `body_head` but applied directly to the raw pair instead of to ITM's `z`.

Held out (12 hexapod clips, same split as F214):

| method | median\|err\| | mean\|err\| | R2 |
|---|---|---|---|
| linear ridge (no ITM, no z) | 0.0202 | 0.0595 | +0.486 |
| MLP, same capacity as body_head, no ITM | 0.0181 | 0.0471 | +0.667 |
| existing `body_head(ITM(.))` | 0.0168 | 0.0397 | **+0.736** |

Train split: linear 0.983, MLP 0.997, ITM 0.964 -- the MLP fits training data almost perfectly (0.997) yet generalizes worse than ITM (0.667 vs 0.736 held out), a larger train/held-out gap than ITM's own (0.964 -> 0.736): evidence the MLP is overfitting on raw pixel-embedding noise that ITM's transition structure does not, not just "under-capacity." Reading at this point: most of the linear-to-ITM gap (0.486 -> 0.736) is recovered by adding *any* nonlinearity of matching capacity (0.486 -> 0.667), but a real, smaller remainder (0.667 -> 0.736) survives when the nonlinearity is specifically ITM's two-frame transition structure rather than a generic MLP on the raw pair, and it generalizes better while doing so.

**Correction, clean split: the ranking reverses once the leak is fixed.** The numbers above came from this script's own ad-hoc 25%-split, seed=0, of the full (unsplit) hexapod directory -- the same leaked-split bug class F222 found and fixed -- and `CKPT` still pointed at a checkpoint superseded by this session's clean retrain. Re-run against the stratified `_cleantrain`/`_cleanheldout` split (seed=42) and `body_head_b1_hex_clean.pt`, with an overlap-check guard added (same pattern as `eval_body_head_true_heldout.py`):

| method | held-out median\|err\| | held-out mean\|err\| | held-out R2 |
|---|---|---|---|
| linear ridge (no ITM, no z) | 0.0357 | 0.0412 | **+0.862** |
| MLP, same capacity as body_head, no ITM | 0.0280 | 0.0356 | **+0.884** |
| existing body_head(ITM(.)) | 0.0596 | 0.0585 | +0.650 |

Train split (24 clips): linear +0.961, MLP +0.984, ITM +0.734 -- same ranking as held-out at every split, the opposite ranking from the leaked result.

**Objection raised and closed: was this just the cost of sharing?** `body_head` here is the multi-embodiment fit -- one set of weights serving hexapod AND B1 jointly (`wm/models/motion_decoder.py`: one function, no embodiment key) -- while the linear/MLP baselines were fit on hexapod alone. A hexapod-only specialist beating a shared head at a hexapod-only task would prove nothing about ITM itself. Controlled for directly: refit linear/MLP on the exact same union of both bodies' `_cleantrain` clips `body_head` was fit on, scored per body against each body's own `_cleanheldout` set:

| method | hexapod held-out R2 | B1 held-out R2 | both pooled held-out R2 |
|---|---|---|---|
| linear ridge (no ITM, no z) | +0.832 | +0.779 | +0.801 |
| MLP, same capacity as body_head, no ITM | **+0.881** | **+0.885** | **+0.884** |
| existing body_head(ITM(.)) | +0.650 | +0.302 | +0.446 |

The sharing tax is not the explanation -- ITM+body_head still loses on both bodies, by a wider margin on B1 than on hexapod (ITM's B1 R2 dropped to +0.302, the weakest cell in either table). F214/F215's original claim, "ITM's transition structure earns its keep over linear/MLP," does not survive a leak-free, sharing-controlled measurement and is retracted.

F214's separate clustering result (89% of within-body behaviour signal in `body_head`'s Froude output surviving the cross-embodiment jump, a geometric measurement, not a regression fit) is different and not touched by this correction -- `body_head_hidden_share.py`'s `CKPT` was updated to the clean checkpoint for consistency but its own numbers have not been re-run, so that claim's status under the clean checkpoint is still open separately.

**One real confound still uncontrolled, so the retraction isn't overstated: the comparison also lets baselines see strictly more information than ITM does.** `z` is a fixed-size bottleneck (`cfg.z_dim`) shaped jointly by every loss in the pipeline (reconstruction, next-embedding prediction, hinge, rollout, body), not optimized only for Froude. The linear/MLP baselines see the full, uncompressed `concat(pooled(e_t), pooled(e_next))` pair and are fit for this one task alone. A baseline with strictly more task-specific freedom beating a general-purpose bottleneck is a real result, but "the bottleneck loses information useful for reading Froude" and "ITM's transition modeling adds nothing" are different claims -- this establishes the first, not the second.

**That control, run: once the input is held constant at `z`, `body_head` is roughly on par with a freshly-refit head, not clearly beaten.** Same union-fit recipe, input changed from the raw pair to `z` itself:

| method | hexapod held-out R2 | B1 held-out R2 | both pooled held-out R2 |
|---|---|---|---|
| linear(z) | +0.610 | **+0.366** | +0.467 |
| MLP(z) | **+0.723** | +0.243 | +0.441 |
| existing body_head(ITM(z)) | +0.650 | +0.302 | +0.446 |

No consistent winner: MLP(z) beats `body_head` on hexapod but loses on B1; linear(z) beats `body_head` on B1 but loses on hexapod; pooled, all three sit within 0.44-0.47. This settles the confound: the loss to the pair-baselines is specifically about `z`'s compression throwing away information, not about `body_head`'s own fit being poor given what `z` retains. `body_head` is tied with a fresh head trained on the identical input. Whether ITM's cross-attention transition structure specifically (vs simple pooling of two frames) adds anything is still untested in the strict sense.

Caveat on trust: the MLP baselines (pair and z alike) are fit on only 48 clips total (24 hexapod + 24 B1, `--epochs 3000`, 128 hidden units) and repeatedly reach train R2 = +1.000, a memorization signature at this sample size, not evidence the representation makes Froude trivial to read. The held-out MLP numbers are noisier for this reason than the linear ones; the linear(z) vs `body_head` comparison (both closer to +0.5, neither near a perfect train fit) is the more trustworthy half of this table.

Script: `scripts/diagnostics/cross_embodiment/linear_vs_itm_froude.py` (extended in place with `PairMLP`/`mlp_fit`/`mlp_predict`, the per-body union-fit, and the `z`-input baselines, no new file).

---

### F216. Four mechanisms to make the B1 world-model-assisted clone walk were tried; none clear the pre-registered bar, but the reasons are now specific

| # | mechanism | travelled/D_real | closed-loop behaviour |
|---|---|---|---|
| 0 | plain BC (baseline) | 52% | falls near the end |
| 1 | DAgger, fine local grading against `body_head` | 31% | falls at ~halfway, worse than baseline |
| 2 | coarse-candidate distillation (relabel with model's best-matching library clip) | 58% | fully collapses on its side by ~frame 35 |
| 3 | auxiliary model-consistency loss (backprop through frozen ITM/FTM/body_head) | 85% | violent slip-and-tilt from ~step 30, never recovers |
| 4 | frame-stacking, history=8 / history=3 | 38% / 43% | history=8 collapses by ~frame 50; history=3 stays upright but wobbly gait |

None pass the bar (upright whole window AND >=50% D_real).

Reasons: mechanisms 1 and 2 are candidate-scoring and inherit `body_head`'s inability to discriminate close actions/recordings (already known from insect, F135/F136/F138; confirmed directly on B1 here, e.g. only 6/48 clips scored best against their own recording in mechanism 2) -- grading with a non-discriminating signal injects near-random labels, worse than no correction. Mechanism 3's loss did move (~30% drop) so the gradient carries more info than discrete ranking, but the resulting policy still doesn't walk cleanly, and never trips the height-only "upright" check despite slipping/tilting violently -- showing that check alone is insufficient to certify a working gait. Mechanisms 1-3 can't fix gait-level instability in principle since `body_head` scores only one Froude number averaged per clip, no notion of instantaneous leg state. Mechanism 4 tests a CPG-shaped hypothesis (short temporal context lets a policy fall back to a stable limit cycle): ambiguous result -- history=8 overfits (train R2 +0.999) and falls sooner than baseline; history=3 never falls but doesn't reach distance bar either. Frame-stacking is not the same test as giving the network real temporal processing (hidden state).

Not yet tried: a small recurrent (GRU) student.

All four mechanisms' code removed from `sim/control/clone_b1.py` after logging (recoverable from git history); only `bc`/`eval` (mechanism 0) remain. Their checkpoints/videos removed with the code.

---

### F217. F216's baseline itself had four bugs; fixed one at a time, plain BC clears the pre-registered bar

Four independent bugs in F216's baseline, found and fixed sequentially (each verified before the next), on the same held-out clip:

| # | bug | fix | effect |
|---|---|---|---|
| 1 | trained on `b1_flat.xml` (placeholder physics), evaluated on `b1_flat_real.xml` (system-identified), never actually checked | re-collected training data on `b1_flat_real.xml` | falling stopped; travel dropped to 20% (stable but slow) |
| 2 | recorded actions are 20Hz but replayed at env's native 50Hz (2.5x too fast, 40% decision resolution) | hold each action for `DATA_DT`=0.05s (10 physics substeps) | 20% -> 29% |
| 3 | goal is whole-clip mean Froude, ~12% below real cruising speed (dragged down by accel/decel edges) | `--trim_goal`: average over steady-state window only | 29% -> 37% |
| 4 | expert relies on an external heading-hold controller (`head_kp`) whose input the student never saw | `--yaw_err_input`: append live heading error as 35th state dim | drift halved (53.8deg->25.7deg) but distance dropped (37%->23%) -- real speed/correction trade-off, not a bug (mean action effort unchanged) |

Fifth fix: training data never demonstrated recovery (every clip starts at zero heading error). Added `--yaw0` to `rollout_b1_mujoco.py` (spawn rotated off target) and collected 6 recovery clips at +-10/20/30 deg (confirmed real convergence, e.g. +25.6 -> +1.8 deg); the `sym` policy fell at every offset and was excluded.

Retrained on combined 22-clip set (16 original + 6 recovery):
| student | travelled/D_real | upright | verdict |
|---|---|---|---|
| original bugged baseline (F212/F216) | 52% | falls | FAIL |
| all 4 bugs fixed, no recovery data | 23-37% | stable | FAIL |
| + recovery data | 65% | stable (min head z 0.55 vs 0.58) | PASS |

Watched side by side against the expert's own rendered trajectory; student's gait judged comparable to the expert's own (non-perfectly-clean) messiness.

One unresolved anomaly flagged, not chased: same checkpoint scored 0% travel (frozen) against a different goal clip that was a training example, not held out.

Settles that a properly-measured, properly-informed plain clone CAN clear the bar for this goal -- none of F216's grading/gradient mechanisms were necessary; the blocker was measurement/missing-input bugs. Does not settle whether the model earns its keep on top of a working baseline (CPG/residual design still to be run).

Scripts: `sim/control/clone_b1.py` (`body_goal_trimmed`, `--trim_goal`, `--yaw_err_input`, `DATA_DT` fix). `sim/collect/rollout_b1_mujoco.py` (`--yaw0`). `scripts/dataset/recollect_b1_flatreal.py` (`collect_recovery()`/`--yaw0_list`). Data: `data/proprioceptive/beh12_b1_flatreal` (54 clips: 48 + 6 recovery). Checkpoint: `wm/runs/students/b1_bc_flatreal_forward_trim12_yaw_recovery.pt`. Videos: `results/wm/b1_bc_flatreal_forward_trim12_yaw_recovery_heldout2.mp4`, `results/wm/expert_b1_ep2100_goal.mp4`.

---

### F218. The goal-conditioning gate (does a different Froude goal produce genuinely different B1 behaviour, on ground-truth data) is not yet passed for turn/side; speed regressed from F217

Testing across all three behaviour families (speed/turn/side), not just F217's forward-only case.

All-behaviour student with `yaw_err_input` on (F217's recipe, 54 mixed clips): speed/turn converge to a static frozen pose (stable standing equilibrium, not a gait) within 10-20 steps; side reaches 133% of D_real but falls.

Ruled out: fall-contaminated training data (checked directly, all 32 turn/side clips' `base_pos[:,2]` in normal 0.56-0.61m range); action-normalization skew from mixing behaviours (per-behaviour action stats nearly identical); undertraining (held-out MSE converged at 2000 epochs, 5000 epochs no better and noisier closed-loop).

Removing `yaw_err_input` eliminates the static collapse (all three produce real motion) but all three then fall (75%/93%/120% of D_real, min head z ~0.09 vs ~0.58). So `yaw_err_input` is a real contributing cause of the multi-behaviour freeze (plausibly because it means different things for speed vs turn), though not proven as the exact mechanism.

Decisive check: even a training goal (offline R2 0.99) still falls in closed loop -- rules out "can't generalize to unseen goals"; this is the same compounding-drift instability F216 diagnosed for forward-only, never fixed for turn/side.

Built `--wz` extension to `collect_recovery()` for turn (target advances at commanded turn rate while correcting initial offset), collected 6 turn-recovery clips (yaw0 in +-10/20/30 deg, wz=0.169), confirmed real convergence (e.g. -14.3 -> +13.8 deg vs target +17.2 deg).

Retrained on 60-clip set (54 + 6 turn-recovery), yaw_err_input on: still fails, a third pattern -- speed 13% of D_real (slow, stable), turn 132% (falls). Side never given recovery data, not re-tested.

Status: goal-conditioning gate not passed for turn; speed regressed from F217's clean 65% PASS once trained alongside turn/side. The same fix (recovery data) that worked for forward-only does not transfer cleanly to multi-behaviour. Paused at user's request to redirect to writing.

Scripts: `sim/control/clone_b1.py` (unchanged). `scripts/dataset/recollect_b1_flatreal.py` (`collect_recovery()` extended with `--wz`/`--recovery_ep0`). `sim/collect/rollout_b1_mujoco.py` (`--yaw0`, reused). Data: `data/proprioceptive/beh12_b1_flatreal`, 60 clips. Checkpoints (all FAIL): `b1_bc_flatreal_all_trim12_yaw.pt`, `b1_bc_flatreal_all_trim12_noyaw.pt`, `b1_bc_flatreal_all_trim12_noyaw_5k.pt`, `b1_bc_flatreal_all_trim12_yaw_v2.pt`.

---

### F219. A fourth instance of hexapod turn-sign instability; this one saturates rather than inverts

Extending the hexapod recipe to 24 conditions needed a mirrored turn family (`turn_conditions(sign=-1.0)`). Backward speed and B1's mirrored turn both verified clean (real sign reversal). Hexapod turn did not: `--spin -0.56` measured yaw +0.071, indistinguishable from the unmodified positive table's +0.077 at the same level.

Fourth documented instance of hex turn-sign non-robustness (F66, F106/F108, F174), and the first with both signs present in the same session/scene. Ruled out staleness: fresh CoppeliaSim restart, re-ran identical condition, got +0.069 (reproduces to two decimals) -- deterministic given current scene/CPG code, not a launch flake.

Different symptom from F174 (which was a full-table inversion fixed by `--spin_sign -1`): here the positive table is untouched and still correct (+0.073 vs +0.077 reference); the new negative condition saturates instead of inverting -- negative `--spin` at any magnitude (-0.56 to -2.0) never produces negative yaw. Root cause not identified (consistent with 0/4 prior instances finding one).

Consequence: negating `--spin` cannot be trusted to mirror hexapod turn direction without a fresh per-instance check. `turn_neg` dropped from the 24-condition set for hexapod (user's call). B1's mirrored turn (direct `wz` command through a trained policy) is unaffected and stays in.

Scripts: `scripts/dataset/collect_beh24.py` (`TURN_NEG`, added, not merged into any dataset). Diagnostic data not retained.

---

### F220. The 2x2 ablation's whole-clip-mean goal is the wrong resolution for the real planner; a time-varying per-timestep re-measurement replaces it, and direct-beats-rollout survives

`wm/policy/planner.py`'s `score_offsets` picks a horizon-window from any candidate clip at any offset to match a goal, every timestep -- it never selects a whole clip as an atomic unit. But `final_2x2x2_test.py`'s goal reading (`body_motion.mean(0)`) uses one constant vector for the whole clip -- a resolution mismatch, separate from any data-leakage issue.

Built `scripts/diagnostics/objective_experiments/froude_match_timevarying.py` (checkpoint `body_head_b1_hex_v2.pt`, B1 candidates from `beh12_b1_ego_flat`, goal = hexapod `turn_s0.05`): reads goal fresh every timestep, reports continuous L2 distance (forward/lateral/yaw) rather than family-accuracy rate.

| comparison | mean |achieved-goal| |
|---|---|
| goal reading alone (vision vs physics) | 0.024 |
| direct scoring, physics goal | 0.038 |
| direct scoring, vision goal | 0.034 (no worse than physics, consistent with F210) |
| rollout scoring, physics goal (fixed `e_t`, see caveat) | 0.082 |

Direct beats rollout survives at corrected resolution (same direction as F118/F126/F184/F185/F188): rollout error ~2.2x direct's; direct locks onto right family by step 10-15 and stays close, rollout swings above/below the goal throughout rather than tracking, picking across all 12 B1 conditions over the run (not one frozen wrong pick as in the whole-clip-mean version).

Caveat: rollout's `e_t` is one fixed B1 frame held constant for the whole trial (no live closed loop) -- read as the scoring mechanism in isolation, not a live control episode.

Methodology bug caught before trusting the number: an earlier version misaligned goal/achieved frames by indexing consecutive frames instead of the strided decision timesteps, inflating rollout error to 0.090 (corrected: 0.082). Fixed by indexing both off the identical `steps` array.

Not covered: one goal clip, one body pair, one horizon (2); mode C (vision goal + rollout) not run.

Scripts: `scripts/diagnostics/objective_experiments/froude_match_timevarying.py` (checked in, supersedes earlier scratch versions). Outputs: `results/deck/froude_{1,2,3}_*.png`, `results/deck/froude_match_2x2_*.png`.

---

### F221. The same whole-clip-mean-goal resolution bug F220 fixed also affects the entire F135/F136/F164/F165/F166 fine-ranking arc -- diagnosed by code reading, not yet re-run

`sim/control/teacher_student_insect.py:body_goal()` (used by `scripts/diagnostics/planning/teacher_label_quality.py`, the whole coarse-vs-fine action-ranking arc) does the identical thing F220 fixed in `final_2x2x2_test.py`: `motion.mean(0)` -- one constant goal vector for the whole clip, computed once and used as a fixed target across every branch point tested (`branch = np.linspace(6, min(len(seed)-horizon-1, 50), states)`, ~15 points), never re-read at each branch point's local time.

Hypothesis, not yet tested: F136's near-identical student/teacher scores (0.1299 vs 0.1304, read as "physics barely separates candidates") could instead reflect both arms being scored against a smeared wrong-resolution target, masking a real ranking signal. F166's later correction (synthetic separation to 10% via sigma sweep, ranking still only 42%) is not obviously explained by this bug alone, but was never tested with a corrected per-branch-point goal either.

What would settle it: replace the single `body_goal()` call with a goal read fresh at each branch point's own local window, re-score student/teacher against that, compare before/after at the same branch points and checkpoint. Everything else in `teacher_label_quality.py` (physics-judged execution, `repeat_control` noise floor, realizability check) is independent of this bug.

Not yet run: needs a live CoppeliaSim instance + GPU encoding, held pending the standing no-GPU-without-permission rule (hardware fault, aria-desktop, 2026-09-18).

Scripts: none run. `sim/control/teacher_student_insect.py:body_goal()` (bug, unfixed), `scripts/diagnostics/planning/teacher_label_quality.py` (caller, unfixed).

---

### F222. First clean, leak-free measurement of Section 9's held-out fit claim: both bodies generalize, but numbers differ from the original (leaked) report

Section 9's original table (0.264 -> 0.572) was measured against a random 25%-split, seed=0, that turned out to have zero overlap with the model's real, deterministic held-out set — every number in it was train-set/memorized performance, not generalization. Re-run end to end (pretrain, adapt to B1 with 9 clips stratified, fit projector, fit body_head with hexapod rehearsal) on `scripts/dataset/make_clean_split.py`'s stratified split (seed=42, 1 clip per condition held out, verified zero overlap with train). Checkpoint chain: `wm/runs/beh12_hinge_cleansplit/` on BIAS-2.

Fixed two real code bugs to get this to run at all, neither specific to this data. `wm.train --val_fraction 0` (wanted: use every train clip, since the real held-out check is the separate clean-split directory) crashed on an empty validation set (`MultiEmbodimentPairs.__init__`, `EmbodimentBatchSampler._batches_per_group()`, both now guard the empty case), and more fundamentally cannot work at all — `wm.train`'s checkpoint-selection picks `best.pt` by comparing `val_metrics` every epoch, which needs real validation data. Fixed properly: added `--val_sources` (`wm/config.py`, `wm/train.py`) so a pre-built stratified validation directory can be given directly instead of only auto-splitting one shared pool by fraction. Final split: 24 train / 12 val / 12 test per body, all three disjoint. Separately, stage 4 (`wm.fit_body_head --ckpt`) was pointed at the wrong upstream checkpoint (stage 2's projector output instead of stage 1's adapt output, which the script's own help text says it needs — "not the pretrain: stage 1 moves what z means"); wrong in the first draft of this command, never caught until it ran.

A third bug, this time in how the result was read, not in running the pipeline — caught by the user asking a clarifying question about the data pipeline, not by this entry's own review. `wm.fit_body_head`'s printed "held out" column is NOT the external `_cleanheldout` set — it's `--val_frac` (default 0.2, never overridden) applied as a random split of whichever directory `--data`/`--also` was pointed at. Stage 4 was run with `--data ..._cleantrain` (all 24 clips available for fitting, on purpose), so the "held out" it printed was actually an internal, non-stratified ~4-5-clip random subset of the TRAIN pool, not the careful stratified 12-clip test set. The first version of this entry reported that number (1.054 for B1) as the real held-out result; it measured something else.

Corrected with a standalone script that never splits anything itself, `scripts/diagnostics/objective_experiments/eval_body_head_true_heldout.py`: every clip in `_cleantrain` scored as train, every clip in the real `_cleanheldout` (12 clips) scored as held out, with a hard refusal if any filename appears in both. Same MSE/mean/ratio metric as `fit_body_head.py`'s own `report()`, so numbers are directly comparable.

Corrected result, body_head fit ratio = MSE / predicting-the-mean-MSE (below 1.0 is genuine signal):

| | train (fit_body_head's own pool) | TRUE held-out (`_cleanheldout`, never seen at any stage) |
|---|---|---|
| B1 | 0.665 | **0.730** |
| hexapod (rehearsal) | 0.597 | **0.659** |

Both bodies generalize for real, and the gap between train and true held-out is small for both — the opposite conclusion from the first (wrong) version of this entry. B1's 0.730 is comfortably better than the mean-baseline and close to its own train number (0.665), the signature of a fit that transfers rather than memorizes 9+24 specific clips. Still weaker than hexapod's own 0.659 (more rehearsal data, same body pretrained on), but not the "barely-better-than-the-mean" result first reported.

Read together with F220/F221's own resolution-mismatch caveat: this MSE-ratio number is still a whole-clip-level metric, the same coarser resolution those findings argue is the wrong shape for what the real planner does — worth eventually re-reading this same clean checkpoint with a per-timestep check, not just accepting these ratios as the final word on B1's cross-embodiment fit.

Not yet done: re-measuring Slides 22 and 23 against this same clean checkpoint and split (only Section 9's own claim was re-run here). Both are affected by the same original leakage and neither has been re-measured yet.

Scripts: `scripts/dataset/make_clean_split.py`, `scripts/run/clean_retrain.sh`, `scripts/diagnostics/objective_experiments/eval_body_head_true_heldout.py` (the correction), fixes in `wm/data/dataset.py`, `wm/config.py`, `wm/train.py`. Run on BIAS-2. Checkpoint: `wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt`.

---

### F223. beh24 (24-condition behaviour library) assembled for both bodies; cross-body Froude calibration mostly matches; clean-retrain shows a real accuracy regression from beh12 due to interference, not just a harder task

beh12 (12 conditions) doubled to beh24: original 8 forward-speed/positive-turn conditions kept unchanged, plus 16 new/recalibrated conditions (side walking at all 4 levels, backward speed, negative turn). Which raw directory was canonical was not documented anywhere, so it was determined by correlating each raw directory's file mtimes against `results/beh24_final_v2/`'s already-reviewed QC videos (each video's mtime lands 3-13 minutes after its true source clip's, 12+ hours after superseded ones): hexapod canonical = `beh24_hex_side_raw` (8), `beh24_hex_speedbwd_raw` (4), `beh24_hex_turnneg_raw` (4) — NOT `beh24_hex_new12_raw`, the superseded first attempt. B1 canonical = `beh24_b1_new16_fovfix_raw` (all 16, FOV-fixed) — NOT `new12_raw`, `new12_raw_v2`, or `side01_recal_raw`.

Existing `merge_behaviour_dirs.py` couldn't combine four separate source locations (needs one balanced `--src` tree, refuses a non-empty `--out`), so two new scripts do the combine directly: `build_beh24_hex_ego_flat.py` (copies the 8 unchanged conditions byte-identical, tags and adds the 16 new ones with fresh episode numbers) and `build_beh24_b1_ego_flat.py` (simpler: the B1 raw dir was already tagged and rendered — confirmed directly, real frame pixel statistics plus `condition`/`behaviour`/`level`/`expert_episode`/`embodiment` fields already present, so no simulator run was needed, just a copy). Both original beh12 `side_L/R_lvl0/lvl1` conditions were dropped in favour of their recalibrated replacements, not kept alongside them. Output: `data/egocentric/beh24_c10f10t10_ego_flat`, `data/egocentric/beh24_b1_ego_flat`, 96 clips each (24 conditions x 4), verified byte-for-byte identical to the frames that produced the already-reviewed QC videos.

Cross-body Froude match (`scripts/dataset/check_beh24_froude_match.py`, rank-matching speed/turn by achieved-magnitude order since command units differ per body, matching side by literal name):

| group | worst pair | relative diff |
|---|---|---|
| speed fwd/bwd | speed_c8.15 vs speed_vx0.40 | 13.4% |
| side L/R | side_R_lvl3 vs side_R_lvl3 | 15.1% |
| turn pos/neg, smallest 2 levels | turn_s0.05 vs turn_w0.008 | **29.1%** |
| turn pos/neg, smallest 2 levels | turn_s0.15 vs turn_w0.024 | **32.8%** |

Everything else (10 of 12 turn pairs, all speed pairs, all side pairs) is under 16%. Accepted as-is, not recalibrated: the smallest two turn levels are ~30% off between bodies, and hexapod's own `side_L_lvl2`/`lvl3` achieved Froude is mildly rank-inverted (lvl2 measures 0.1407, lvl3 0.1362, should be reversed, ~3% apart, likely noise rather than real miscalibration; `side_R` and both of B1's side directions are correctly monotonic). Consistent with this session's standing call (F219's neighbourhood): imperfect calibration is acceptable diversity, not a blocker, given the planner matches Froude per-timestep rather than depending on any one condition being clean.

Addendum (2026-09-19): beh24 clean retrain ran end to end (`scripts/dataset/make_clean_split_beh24.py`, seed=42, same one-test-one-val-per-condition methodology, 48/24/24 per body; `clean_retrain_beh24.sh`, identical hyperparameters to beh12's; checkpoint `wm/runs/beh24_hinge_cleansplit/`), scored with the same `eval_body_head_true_heldout.py` used to catch F222's measurement bug, so leak-free from the start:

| | beh12 held-out ratio (F222) | beh24 held-out ratio |
|---|---|---|
| B1 | 0.730 | **0.761** |
| hexapod | 0.659 | **0.695** |

Both got worse by 0.03-0.04, but this is not a same-difficulty comparison: beh12 was one-sided on every Froude axis (forward speed only, one turn direction only, side levels 0-1 only, roughly half the achievable lateral range); beh24 completes all three axes with genuine +/- coverage. A model scoring 0.695/0.761 on the FULL bidirectional task is not doing worse than one scoring 0.659/0.730 on HALF of it in any way the raw ratio captures.

Controlled comparison, run: it is interference, not just a harder task. Scored both checkpoints on ONLY the 8 conditions beh12 and beh24 share byte-identically (forward speed x4, positive turn x4; side excluded, recalibrated not shared). The standardized ratio blew up far past the full-set gap — beh24's checkpoint scored WORSE than predicting the mean on these shared conditions (B1 1.499, hexapod 1.468) against beh12's checkpoint solidly beating the mean (B1 0.769, hexapod 0.864) — large enough to ask whether it was real or an artifact of each checkpoint standardizing against a different-width target distribution (beh24's std, fit over a much wider range including large side/turn/speed-backward magnitudes, shrinks these 8 conditions' own variance in standardized units, which alone inflates a ratio with no real accuracy change). Checked directly with a raw-unit (actual Froude, not standardized) MSE, which is checkpoint-independent:

| | beh12 checkpoint raw MSE | beh24 checkpoint raw MSE | increase |
|---|---|---|---|
| B1 held-out | 0.00112 | 0.00234 | **2.1x** |
| hexapod held-out | 0.00120 | 0.00201 | **1.7x** |

Real, not an artifact. Absolute prediction error on the exact same 8 conditions roughly doubled after training on the fuller 24-condition set — completing direction coverage cost real accuracy on conditions the model used to handle well, the signature of interference/capacity competition between new and old conditions under the same fixed model size and 50-epoch budget, not of a uniformly-harder-but-equally-well-learned task.

Not yet done: diagnosing which of the candidate explanations accounts for the regression; no training run has used `beh24_*_ego_flat` for anything besides this one clean-split retrain.

Scripts: `scripts/dataset/build_beh24_hex_ego_flat.py`, `build_beh24_b1_ego_flat.py`, `check_beh24_froude_match.py`, `scripts/diagnostics/objective_experiments/eval_body_head_true_heldout.py` (extended with `--conditions` and raw-unit MSE reporting, no new file).

---

### F224. Gait periodicity measured directly for the first time: dominant period is ~6.6 frames, not the ~19 previously cited

`report/update_slide.md` Slide 6 cited "period ≈19 frames" with no autocorrelation curve behind it. `scripts/diagnostics/objective_experiments/periodicity_curve.py` existed (single-frame ridge-regression command-recoverability error vs. offset, matching F26/F46's metric) but had only been run at 7 sparse offsets.

Found and fixed a confound: the per-offset window `range(1, T-h-1)` shrank/shifted earlier as `h` grew, so larger offsets scored on a smaller, earlier-biased subset than smaller offsets (n_test fell 624->247 over the sweep). Fixed by holding the starting-frame range fixed across all `h` (n_test constant at 78, max_h=58).

Result, confound removed, h=0..58 (26 clips, `beh12_hexonly/best.pt`, held out by clip): corrected curve is roughly flat ~2.0-2.5° (confirms F26/F46's flat-at-7-points was correct, just under-sampled). FFT: one dominant spectral component at **period ≈6.6 frames** (~3x power of any other), plus a secondary component at **period ≈19.7 frames** (close to the old "≈19" figure, but not dominant — plausibly a beat/subharmonic, 6.6×3≈19.8).

Not yet investigated: what recurs every ~6.6 frames vs. every ~19-20 (would need per-leg phase data).

Scripts: `scripts/diagnostics/objective_experiments/periodicity_curve.py` (confound fix). Plots: `results/deck/periodicity_curve_fixed_58.png` (reported result), `results/deck/periodicity_curve_fixed.png` (h=0-45, superseded). `results/deck/periodicity_curve.png` is the pre-fix confounded run — do not cite.

---

### F225. F127's selection-accuracy test rechecked with a continuous regret metric: discrete "misses" are small misses, not far ones

`score_by_body_motion.py` (behind F127/Slide 18's 68-70% pooled cross-embodiment selection-accuracy numbers) already reads the goal at true per-timestep resolution (not the whole-clip-mean bug fixed elsewhere), but still grades by discrete family-match hit/miss.

Added `regret`: true distance-to-goal of the picked candidate minus the best real candidate available, in real Froude units, from each clip's own recorded trajectory (`full_body_motion()`, new). Additive; existing hit/miss columns unchanged.

Rerun on nearest available sibling checkpoint (`beh12_hex-b1_body3/stage3_b1_nce_s0.pt`; F127's exact checkpoint not available locally). Reproduces F127's shape:

| horizon | pooled | side_L | side_R | speed | turn | mean regret |
|---|---|---|---|---|---|---|
| 1 | 62% | 78% | 83% | 20% | 47% | 0.041 |
| 3 | 64% | 76% | 80% | 24% | 55% | 0.032 |
| 5 | 67% | 73% | 93% | 14% | 64% | 0.034 |
| 10 | 73% | 92% | 84% | 11% | 70% | 0.026 |

Regret is small and shrinks with horizon even on the weakest-by-discrete-grade family (`speed`, 11-24% hit rate but regret 0.026-0.041, falling with horizon). The two grades ask different questions (named-right-family vs. real-unit closeness); a coarse selector can look weak on the first while fine on the second.

Scripts: `scripts/diagnostics/planning/score_by_body_motion.py` (`full_body_motion()`, regret computation added). Cache: `results/wm/cache/b1_body3_scoretest.pt`, `results/wm/cache/bodycal_hexapod.pt`.

---

### F226. Replacement for the withdrawn "92% across all 12 goal conditions" claim: direct beats rollout on 12/12 conditions; vision-read goal costs nothing on average

Slide 21's corrected per-timestep methodology (`froude_match_timevarying.py`) had only been run on one goal clip (`turn_s0.05`). Extended to all 12 hexapod goal conditions via new `froude_match_all12.py` (thin loop over the same functions, one model/encoder load).

Checkpoint substitution (as in F225): exact original checkpoint unavailable; used `beh12_hex-b1_body3/stage3_b1_nce_s0.pt`. Sanity check on `turn_s0.05` alone confirms the substitution doesn't distort the shape of the result.

Result, all 12 conditions, real Froude units:

| comparison | mean error | range |
|---|---|---|
| direct, physics goal (privileged) | 0.0495 | 0.0280-0.0935 |
| direct, vision-read goal | 0.0438 | 0.0238-0.0729 |
| rollout, physics goal | 0.1044 | 0.0695-0.2062 |

Direct beats rollout on 12/12 conditions (mean error less than half). Vision-read goal costs nothing on average and beats the privileged number (0.0438 vs 0.0495). This replaces the withdrawn 92% claim (same scope, corrected per-timestep methodology, continuous real-unit distance instead of family-match percentage) — the two numbers are not directly comparable; 92% should not be requoted alongside this.

Scripts: `scripts/diagnostics/objective_experiments/froude_match_all12.py` (new), reuses `froude_match_timevarying.py`'s functions verbatim.

---

### F227. `collect_b1_cpg_babble.py`'s archive move had silently broken three more live imports; fixed by inlining (per established project precedent), not by re-pointing the import path

`collect_b1_cpg_babble.py` moved to `sim/collect/_archive/`; `wm/policy/b1_coppelia_env.py` already documents this broke imports once before ("7 dependent scripts") and the established fix is to inline constants deliberately (accepted drift risk) rather than re-import.

That fix had not been applied everywhere. Two more scripts were still live-importing the moved file, silently broken: `b1_coppelia_live_policy.py` and `b1_coppelia_cpg_controller.py`. A third, `reward_quality_gate_b1.py`, hit the error during this session's run. `collect_b1_coppelia_babble.py` was already correctly pointed at `_archive/`.

Fixed all consistently: inlined constants in the three broken scripts; also switched `collect_b1_coppelia_babble.py` to import the same constants from `b1_coppelia_cpg_controller.py` instead of `_archive/` — nothing in the active codebase now imports the archive.

Durable fact: the archived file's `MODEL` path computation (`os.path.join(ROOT, ...)`) counts directories up from its own location, which shifted by one level after the archive move, silently pointing `MODEL` at a nonexistent path (`sim/sim/assets/...`) for any script still importing it. Worked around by inlining the correct value directly, not fixed in the archive itself.

Scripts touched: `scripts/diagnostics/objective_experiments/reward_quality_gate_b1.py`, `b1_coppelia_live_policy.py`, `b1_coppelia_cpg_controller.py`, `sim/collect/collect_b1_coppelia_babble.py`. No behavioural change — same constants/values, verified against archived source before inlining.

---

### F228. F201's "20.8%" reward-quality-gate result does not reproduce on the confirmed-correct checkpoint — withdrawn; gate is still at chance

Re-running `reward_quality_gate_b1.py` (F225's regret fix applied) against the exact checkpoint F201 names (`wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1_hex_v2.pt`, confirmed correct — matches F201/F214's own description: `adapted` shows 9 B1 clips/1000 steps sourced from `beh12_hinge_multistep_anchor_v2/best.pt`, `body_head_fit` shows 6000 epochs on `beh12_b1_ego_flat`) gave **4.2% (1/24)**, not 20.8% (5/24). No exact command line for the original 20.8% run was ever recorded in this file, only its output log — so before assuming the checkpoint or script was wrong, checked whether the discrepancy could just be sampling noise at n=24.

It is not. 5 seeds at the original sample size (8 branch points x 3 sigmas = 24 trials):

| seed | hit rate | Spearman rho | regret |
|---|---|---|---|
| 0 | 4.2% (1/24) | 0.128 | 0.0047 |
| 1 | 8.3% (2/24) | 0.216 | 0.0044 |
| 2 | 8.3% (2/24) | 0.104 | 0.0046 |
| 3 | 8.3% (2/24) | 0.180 | 0.0048 |
| 4 | 4.2% (1/24) | 0.052 | 0.0046 |

Hit rate swings 4.2-8.3% across seeds, exactly the size of noise expected from moving 1 trial out of 24, but never once lands near 20.8% in 5 tries — if 20.8% were the true rate with this much natural variance it should appear occasionally, and it doesn't. Regret, by contrast, is remarkably stable across all 5 seeds (0.0044-0.0048), confirming the continuous grade (F225's fix) is already trustworthy at this sample size while the discrete hit-rate never was.

Scaled up 4x (32 branch points, 32 samples/sigma = 96 trials, chance drops to 3.0% with the wider candidate pool) for a properly-powered read, 2 seeds:

| seed | hit rate | Spearman rho | regret |
|---|---|---|---|
| 0 | 2.1% (2/96) | 0.064 | 0.0085 |
| 1 | 1.0% (1/96) | 0.035 | 0.0087 |

At or below the 3.0% chance rate, consistent between seeds, and consistent with F195's original finding (the checkpoint *before* the hinge fix, "at or below chance") — not with F201's claimed improvement. The honest reading: fix #6 (the hinge term carried through B1 adaptation) either does not work the way F201 concluded, or the 20.8% figure came from a genuinely different run configuration that was never recorded — either way, 20.8% cannot be trusted as this project's current best reward-quality-gate result.

Consequence for Q21 step 3 (RL controller): F201's own text says this result is what unblocked treating the reward-quality gate as "a live, open, promising question" rather than an exhausted wall. That basis is gone. The gate should be treated as still failing, at or below chance, the same place F195 left it, until a new fix is found and re-verified at a large enough sample size to trust (n=96, not n=24, going forward — this entry's own comparison is the reason why).

Command: `reward_quality_gate_b1.py --ckpt wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1_hex_v2.pt --branch_points 32 --samples 32 --seed {0,1}` for the large-sample rows; `--seed {0..4}` at defaults for the small-sample rows.

Addendum, same day — a second, real bug the user caught, checked, fixed, and it does not change the verdict. `goal_fr` was computed once as `body_motion.mean(0)` over the ENTIRE goal clip and reused identically at every branch point, regardless of where in the expert clip's own episode that branch point sat — the same whole-clip-mean anti-pattern this project already found and fixed in `final_2x2x2_test.py`/`froude_match_timevarying.py`, just not caught in this script until now. Fixed: the goal is now read from a short local window in the goal clip, matched to the same fractional progress through the episode as the current branch point (`goal_at()`, new). Rerun at the large sample size (32 branch points x 32 samples, seed 0) after the fix: hit rate 1.0% (1/96), Spearman rho 0.022, regret 0.0093 — statistically indistinguishable from the pre-fix numbers above. The goal-averaging bug was real and worth fixing, but it was not the explanation for F201's non-reproduction — the gate sits at chance either way.

---

### F229. `_cleantrain`/`_cleanheldout` directories were broken symlinks on this machine — fixed locally with relative paths; F222's held-out result reproduced exactly

After merging F222 (checkpoint `wm/runs/beh12_hinge_cleansplit/`), reproduction failed: all 72 files across the four `beh12_{b1,c10f10t10}_ego_flat_clean{train,heldout}` directories were symlinks pointing at an absolute aria-desktop path (`/home/aria/ioon-research/...`) not present locally — committed as symlinks by `make_clean_split.py`'s disk-saving mechanism, contradicting `doc/START_HERE.md`'s claim that the B1 equivalent was committed as real files.

Not a data problem: every symlink target exists locally under `data/egocentric/beh12_{b1,c10f10t10}_ego_flat/` (the non-clean source dirs).

First fix attempt (absolute local path) reproduced the same bug with a different machine's path — would break the next clone. Corrected to **relative symlinks** (`../<base>/<file>`) — portable across machines/clones.

Reproduced F222 exactly after the fix: B1 train 0.665/held-out 0.730, hexapod train 0.597/held-out 0.659 (matches to three decimals).

This relative-symlink fix should be committed so it doesn't silently re-break on the next clone/machine.

Scripts: none changed. Data fix: 72 symlinks across the four `_clean{train,heldout}` directories, re-pointed from absolute aria-desktop path to relative sibling-directory path.

---

### F230. Section 10's per-channel rho re-measured on the clean checkpoint: mixed by channel, not simply better than the leak-affected claim

Extended `eval_body_head_true_heldout.py` to compute per-channel Spearman rho (forward/lateral/yaw) on the same TRUE held-out set validated by F222/F229. Additive to the existing MSE-ratio line.

Result, `beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt`, true held-out:

| | forward rho | lateral rho | yaw rho | median rho |
|---|---|---|---|---|
| B1 | +0.261 | +0.578 | +0.474 | +0.474 |
| hexapod (rehearsal) | +0.714 | +0.403 | +0.558 | +0.558 |

Not simply an improvement on the original leak-affected claim (0.572/0.449/0.670/0.572): lateral improved (0.449->0.578), forward and yaw are weaker (0.572->0.261, 0.670->0.474). The leak inflated some channels, not uniformly. Median rho (+0.474) still clears the old zero-shot baseline (+0.264), but "every channel more than doubles" does not hold. The zero-shot baseline itself was not yet re-measured on a clean split at this point.

Scripts: `scripts/diagnostics/objective_experiments/eval_body_head_true_heldout.py` (per-channel Spearman rho added to `score()`).

---

### F231. Zero-shot baseline measured cleanly: B1 fails outright before adaptation (ratio 1.061, worse than the mean), on the same held-out set F230 used

Closed F230's flagged gap: ran `eval_body_head_true_heldout.py` against `wm/runs/beh12_hinge_cleansplit/best.pt` (hexapod-only pretrain, before `wm.adapt` touches B1), on the identical clean held-out set used by F222/F229/F230.

| B1 | held-out ratio | forward rho | lateral rho | yaw rho | median rho |
|---|---|---|---|---|---|
| zero-shot (this entry) | **1.061** | +0.178 | +0.125 | +0.187 | +0.178 |
| staged adaptation (F230) | 0.730 | +0.261 | +0.578 | +0.474 | +0.474 |

Zero-shot is worse than predicting the mean (ratio >1.0) — genuinely failing, not just weak — while staged adaptation clears the mean by a real margin on the same held-out clips. Fully clean, apples-to-apples comparison: same split, same held-out set, same metric, both points measured this session.

Scripts: `scripts/diagnostics/objective_experiments/eval_body_head_true_heldout.py` (run against the pretrain checkpoint instead of the adapted one, no code changes).

---

### F232. On the z.detach() checkpoint, a body_head fit separately per body does not transfer across bodies at all; a jointly-fit head is merely mediocre, not genuinely cross-body

Section 9's open item: re-measure Section 8's 4-way insect/B1 R2 table on the `z.detach()` checkpoint. Section 8's table used a head co-trained jointly on both bodies, leaving open whether cross-body sharing was real or gradient leakage between bodies in the same run. Ran a stricter test: fit a Cross-Body Head on `beh12_body_stopgrad/best.pt`'s `z` using only hexapod data and separately using only B1 data (`cross_body_head_4way.py`, new), scored cross-body.

| this checkpoint's z, head fit separately per body | insect->insect | b1->b1 | insect->b1 | b1->insect |
|---|---|---|---|---|
| held-out R2 | +0.274 | +0.157 | **-0.397** | **-0.594** |

Same-body readout works but weaker than co-trained (Section 8: 0.798/0.879 -> 0.274/0.157 here); cross-body transfer does not survive at all when heads are fit separately — both cross directions negative.

Original co-trained protocol replicated on this exact checkpoint (one head fit jointly on both bodies, `head_joint.pt`):

| held-out R2 | insect->insect | b1->b1 | insect->b1 | b1->insect |
|---|---|---|---|---|
| separately-fit (above) | +0.274 | +0.157 | -0.397 | -0.594 |
| jointly-fit, one shared head | 0.133 | 0.139 | 0.139 | 0.133 |
| Section 8, co-trained, NOT detached | 0.798 | 0.879 | +0.544 | +0.435 |

A joint head has no real "cross" cell (insect->b1 = b1->b1, b1->insect = insect->insect by construction) — the honest read is 2 numbers, ~0.13-0.14 on both, one mediocre generalist function, not recovered cross-body sharing. Neither protocol reaches Section 8's numbers on the non-detached checkpoint. Whatever produced Section 8's genuine cross-body result needs z's gradient un-detached — `z.detach()` is a same-body fix, not a cross-body one, on this evidence. (Superseded/refined by F233: the detach turns out unnecessary entirely.)

Scripts: `scripts/diagnostics/objective_experiments/cross_body_head_4way.py` (new, `--hex_ckpt`/`--b1_ckpt`/`--hex_dir`/`--b1_dir` args). Checkpoints: `wm/runs/beh12_body_stopgrad/head_hexonly.pt`, `head_b1only.pt`, `head_joint.pt`.

---

### F233. z.detach() (F199-201, F232) was never necessary — the hinge + multi-step-anchor fix alone already clears the action-lever bar with cross-body R2 intact

F232 found z.detach() trades cross-body sharing for same-body action-lever sensitivity. Checked whether `wm/runs/beh12_hinge_cleansplit`'s recipe (hinge + multi-step reconstruction anchor, no detach) already avoids that trade — its config confirms `lambda_hinge=0.5, hinge_margin=0.1, hinge_K=2, lambda_rollout=1.0`, no detach flag. This is the checkpoint underlying every Section 10 number (F222, F229-F231), never tested for action-lever gap until now. Extended `eval_body_head_true_heldout.py` with `action_lever_gap()` (real-z-vs-mean-z cosine gap, F232's metric, tests the checkpoint's own embedded body_head directly).

| | R2 held-out ratio | action-lever gap (bar 0.110) |
|---|---|---|
| B1 | 0.730 | **+0.978** |
| hexapod | 0.659 | **+0.686** |

Both clear the action-lever bar by 6-9x, with cross-body R2 (0.730/0.659) fully intact — contrast with z.detach() (F232): comparable gaps (0.544-1.124) but cross-body R2 collapses to negative or uniform-mediocre. z.detach() was solving a problem the hinge + multi-step-anchor fix had already solved; detaching costs cross-body sharing for no action-lever gain. Section 9's description of z.detach() as "the fix" is superseded by this entry.

Not settled: whether this conclusion holds on beh24 or on non-clean-split data — only established on the checkpoint already in continuous use for Section 10.

Scripts: `scripts/diagnostics/objective_experiments/eval_body_head_true_heldout.py` (`action_lever_gap()` added, additive). Checkpoint: `wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt` (pre-existing).

**Correction (2026-09-28): F233's premise is false -- `beh12_hinge_cleansplit` WAS trained with
`z.detach()`.** The stop-gradient on the body loss had no config field: it was hard-coded in
`wm/train.py` (`md.body(view, z.detach())`) from commit c1e1bb9 (2026-09-08) onward, and before that
(commit 5ab9b19, 2026-08-18) the body loss read `z` undetached. F233 read "no detach flag" in the
run's config and concluded the run was undetached; the run (config written 2026-09-18) postdates the
hard-coded detach, so it is a detached run like `beh12_body_stopgrad`. F233 therefore compares two
detached checkpoints and says nothing about whether the detach is necessary. It also compares a
held-out MSE ratio (0.730 / 0.659) against F232's cross-body R2, which are different metrics.

What the record supports instead:
- Every run after 2026-09-08 (beh12_hinge, beh24, stride 1/5/10, A/A2/B/C, babble arms) trained with
  the body loss unable to shape `z`.
- Section 8's cross-body R2 (+0.544 / +0.435) came from runs before the detach, pretrained on hexapod
  AND B1 together with one shared body head whose gradient reached `z`, read with a co-trained head.
  After the detach, heads fit on one body do not transfer (F232: -0.397 / -0.594; F272: c10 -> B1
  negative). Detach on vs off has never been compared with the same data, bodies and evaluation.
- F177's argument for the detach (the undetached `z`-only head failed the action-lever test) was
  measured at stride 1 on one-behaviour-per-clip data, both since found to hide the action (F259:
  1-step action visibility r ~0.2; F252/F266: start-state shortcut, which persists with the detach on).

Fix: `Config.detach_body_z` (default True, reproduces every run since 2026-09-08) makes the choice a
recorded config field. The controlled comparison is `scripts/run/detach_2x2_step1.sh` (hexapod only,
beh24 vs switch babble x detach on/off) and `scripts/run/detach_joint_step2.sh` (hexapod + B1 jointly
pretrained, detach on/off, fit-on-one-body / test-on-the-other read-out).

---

### F234. beh12 vs beh24, same clean-split eval, both bodies: real accuracy cost confirmed on hexapod (~2x), smaller on B1, with a genuine per-channel gain on B1's forward

`wm/runs/beh24_hinge_cleansplit/` hit the same broken-symlink bug F229 fixed (72 files there; 192 here across `beh24_*_ego_flat_clean{train,heldout,val}`), fixed identically with relative symlinks. Ran the same `eval_body_head_true_heldout.py` protocol used for beh12 (F230/F231), with the merged `--conditions`/raw-unit-MSE extension for fair comparison:

| | B1 ratio | B1 raw MSE | hexapod ratio | hexapod raw MSE |
|---|---|---|---|---|
| beh12 | 0.730 | 0.00308 | 0.659 | 0.00173 |
| beh24 | 0.761 | 0.00381 | 0.695 | 0.00356 |
| beh24/beh12 | 1.10x | **1.24x worse** | 1.05x | **2.06x worse** |

Confirms the earlier ~2x estimate almost exactly, but only on hexapod (16 new conditions added to its original 8, spreading fixed model capacity thinner). B1's cost is smaller (1.24x) and not uniformly worse per channel:

| B1 rho | forward | lateral | yaw | median |
|---|---|---|---|---|
| beh12 | +0.261 | +0.578 | +0.474 | +0.474 |
| beh24 | **+0.613** | +0.491 | +0.405 | +0.491 |

Forward nearly triples on B1 with beh24, plausibly because beh24 adds real backward-motion coverage for B1 that beh12 lacked entirely.

Read as a real mixed trade-off: beh24 costs measurable per-condition accuracy (~2x on hexapod) but buys real new capability (backward motion, opposite turn direction) beh12 could not represent. Neither dataset is strictly better.

Scripts: none changed beyond F233's `eval_body_head_true_heldout.py` extension. Data fix: 192 symlinks across `beh24_{b1,c10f10t10}_ego_flat_clean{train,heldout,val}`, same relative-path fix as F229.

---

### F235. Slide 22's two remaining claims (ITM-beats-raw-pair, 89% cross-embodiment clustering), re-run on the clean checkpoint: both collapse

F215's clean-split correction already reversed the R2 table but left the clustering half (`body_head_hidden_share.py`, F214's 89% figure) explicitly open. Both re-run against `beh12_hinge_cleansplit`'s `body_head_b1_hex_clean.pt`.

R2 table (reproduces F215's correction, independent run):

| method | hexapod held-out R2 | B1 held-out R2 | both pooled held-out R2 |
|---|---|---|---|
| linear(pair), no ITM | +0.832 | +0.779 | +0.801 |
| MLP(pair), no ITM | +0.855 | +0.867 | +0.862 |
| existing body_head(ITM(.)) | +0.650 | +0.302 | +0.446 |

Raw frame pair, never touching ITM/z, generalizes clearly better than the actual pipeline (not new, restated since Slide 22 still had pre-correction numbers).

Clustering, re-run for the first time on the clean checkpoint:

| representation | cross-embodiment gap | within-embodiment gap | ratio | 3-fold held-out gaps |
|---|---|---|---|---|
| raw z (64-D) | 0.184 | 0.224 | 82% | -- |
| body_head hidden (128-D) | 0.065 | 0.109 | 60% | -- |
| body_head Froude output (3-D) | 0.231 | 0.684 | **34%** (was 89%) | **+0.556 / -0.013 / -0.061** (was 0.75/0.89/0.91, all positive) |

89% does not survive: ratio drops to 34%, and the train/test-split check now gives one positive fold and two indistinguishable from zero — F214's "all three folds strongly positive" was a leaked-split artifact.

Reading: both headline numbers measured the leaked, non-stratified split, not genuine cross-embodiment structure. Consistent with Section 8's action-lever result and Slide 15's context-collapse fix: z is a lossy bottleneck relative to the raw frame pair; what clears the bar happens downstream of z through body_head's own fit, not through z carrying more information than raw pixels.

Scripts: `scripts/diagnostics/cross_embodiment/linear_vs_itm_froude.py`, `scripts/diagnostics/cross_embodiment/body_head_hidden_share.py` (already pointed at the clean checkpoint; no code changes, only execution).

---

### F236. Where the FTM's auto-regressive rollout stops earning its keep, on the clean B1 checkpoint: a concrete horizon number for any future imagination-RL redo

F179's arc traced the imagination-RL wall to the FTM's rollout, using gamma=0.99 (~100-step effective horizon) without checking how far this FTM's predictions stay trustworthy. Measured directly with `rollout_horizon_accuracy.py` (unmodified): `beh12_hinge_cleansplit`'s `body_head_b1_hex_clean.pt`, B1 data (48 clips, 66 frames, `--stride 4`), k=1 to 60:

| k | 1 | 2 | 3 | 5 | 8 | 12 | 16 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ratio (model MSE / copy-forward MSE) | 0.622 | 0.614 | 0.633 | 0.678 | 0.738 | 0.804 | 0.779 | 0.830 | 0.837 | 0.852 | 0.851 | 0.874 |

Ratio stays below 1.0 everywhere (model never worse than "predict no change," even at k=60) but climbs steadily from 0.62 at k=1 to 0.87 by k=60 — most of the model's edge is gone by k=20-30 (already 0.80-0.84 there).

Reading: does not contradict F179, gives it a number. gamma=0.99's ~100-step horizon asks the critic to trust predictions far beyond where fidelity was even measurable (clips cap at 66 frames) and well past where the edge over baseline is mostly gone. A future imagination-RL redo should pick horizon H from this curve directly (e.g. H in 15-25, ratio ~0.78-0.83) rather than from an unrelated discount-factor convention.

Scope: measures rollout fidelity only, not whether a critic trained inside k<=25 would converge to a working policy — open question for a future redo.

Script: `scripts/diagnostics/objective_experiments/rollout_horizon_accuracy.py` (no changes). Cache: `results/wm/cache/ego_b1.pt`.

---

### F237. Redoing F179's isolation test at F236's recommended horizon: actor's realized return improves, but plain EMA critic still diverges — fixed only by periodic Monte-Carlo anchoring, and that fix does not generalize past a single fixed (start, goal) pair

F179's critic (bootstrapping over gamma=0.99's ~100-step horizon) never converged (drifted to -421, true return near -40). F236 showed this checkpoint's FTM rollout stays meaningfully better than "no change" only out to k~20-30. Tested whether truncating the horizon alone (before reaching for F179's stabilization ladder: symlog/return-norm/two-hot) fixes divergence.

Rebuilt from scratch (`rl_imagination_isolation_v2.py`, new — F179's original script was never committed to git). Plain EMA target critic, no symlog/return-norm/two-hot, H=12 (shorter than F236's 15-25 suggestion; batch forced to 4 after OOMs at 64/16), 2000 iterations, frozen FTM, one fixed real branch state, one fixed goal.

| iter | eval_realized | eval_value0 | gap |
|---|---|---|---|
| 0 | -23.9 | -0.1 | +23.8 |
| 200 | -11.0 | -21.6 | -10.6 |
| 1000 | -9.8 | -37.5 | -27.7 |
| 1999 | -8.3 | -50.7 | -42.3 |

Actor's real return genuinely improves and holds (~-9 to -11 from iter 200 on). Critic's value belief drifts monotonically negative (-0.1 -> -50.7) — same unbounded-divergence signature F179 showed at H~100, now at roughly a tenth the horizon. Horizon length alone does not fix the plain EMA critic; it's orthogonal to the stabilization ladder.

Same-day follow-up: combined F236's short horizon with F179's rung-2 stabilization (symlog critic + DreamerV3-style return normalization, `--stabilize`), same H=12, 2000 iters. Critic still diverges by essentially the same final magnitude (-45.3 vs -50.7 unstabilized) even though `critic_loss` collapses to ~0.002 — the critic fits its own recursively-bootstrapped (drifting) target almost perfectly, which is consistent with unbounded divergence from ground truth. Actor's behaviour improved further (first-quarter -12.2 -> last-quarter -7.1). Points away from "horizon too long" or "critic scale miscalibrated" toward the bootstrap target's circularity (EMA target network with no periodic anchor to real value) as the root cause.

Same-day, second follow-up: periodic hard-reset to a real Monte Carlo anchor. Added `--anchor_every` (every 200 iterations, roll the current deterministic actor 20 real steps under `torch.no_grad()`, regress critic onto that real number for 10 steps, hard-copy target network), same H=12, symlog+return-norm, 2000 iters.

| iter | eval_realized | eval_value0 | gap | MC-return at nearest anchor |
|---|---|---|---|---|
| 0 | -15.4 | -19.6 | -4.1 | -22.2 |
| 200 | -6.5 | -5.0 | +1.5 | -10.4 |
| 600 | -2.7 | -1.9 | +0.8 | -4.3 |
| 1999 | -2.7 | -16.8 | -14.0 | (150 iters past last anchor) |

Both eval_realized and independently-measured MC-return stabilize and hold (first result in this arc where actor behaviour converges and holds rather than drifting). Critic's raw value between anchors still drifts the same way (gap widens to -7 to -17 by ~150-200 iters after each anchor) — the fix is periodic correction, not a stable critic.

Same-day, third follow-up: the anchored recipe does NOT generalize past the single fixed (start, goal) pair. Rebuilt goal-conditioned (`rl_imagination_isolation_v3.py`, new): actor/critic take `concat(pooled(e_t), goal)`, trained on 12 tasks (`_cleantrain`), evaluated on train pool and a genuinely held-out pool (`_cleanheldout`), same H=12/stabilization/anchoring (3 random tasks anchored per point).

| | first-quarter mean | last-quarter mean |
|---|---|---|
| train pool (12 tasks, seen) | -30.6 | -29.1 |
| held-out pool (12 tasks, never trained/anchored) | -30.6 | -30.2 |

Essentially flat, vs. the single-pair version's -15.4 -> -2.7 (>5x improvement). The single-pair result is consistent with memorizing one state/goal pair rather than generalizing. Filed as open: 2000 iterations at 4 tasks/iteration was likely insufficient budget for the harder goal-conditioned problem, not tried with more iterations/larger network/more tasks per anchor.

Scripts: `scripts/diagnostics/objective_experiments/rl_imagination_isolation_v2.py` (new, `--stabilize`, `--anchor_every`/`--mc_horizon`/`--anchor_steps` added), `rl_imagination_isolation_v3.py` (new). Projector: `wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/projector_clean.pt` (fit fresh, F236's checkpoint had none). Runs: `results/wm/closed_loop/rl_imagination_isolation_v2_h12.npz`, `..._h12_stab.npz`, `..._h12_anchor.npz`, `rl_imagination_isolation_v3_multi.npz`.

---

### F238. How many B1 clips does stage 4 (`fit_body_head`) actually need — a real clip-count sweep

Section 10's pipeline uses close to the full 24-clip B1 training pool for stages 2/4 despite staged adaptation's premise that less should be needed. Never measured directly until now.

Setup: fresh B1-adapted checkpoint (`wm.adapt`, 9 clips, 1000 steps, `lambda_hinge=0.5`), then `fit_body_head` run independently at four B1 clip-count budgets (3, 6, 12, 24, stratified round-robin across 12 conditions, hexapod rehearsal unchanged), each scored on the same disjoint 12-clip `_cleanheldout` set:

| B1 clips used | held-out ratio | forward rho | lateral rho | yaw rho | median rho |
|---|---|---|---|---|---|
| 3 | 1.530 | +0.169 | +0.191 | +0.359 | +0.191 |
| 6 | 0.865 | +0.218 | +0.512 | +0.391 | +0.391 |
| 12 | 0.736 | +0.326 | +0.551 | +0.466 | +0.466 |
| 24 (full) | 0.694 | +0.428 | +0.537 | +0.466 | +0.466 |

Real cliff, not a smooth curve: 3 clips fails outright (ratio > 1); 6 already clears the bar (0.865); 12 (half the pool) reaches 0.736, within a few points of the full-24 result (0.694) — most of the value is captured at half. Forward rho keeps improving to 24 (+0.326 -> +0.428); lateral/yaw are flat by 12.

Scope: sweeps stage 4 only, using curated behaviour clips, not random motor babble — babble data-efficiency is a separate open question.

Scripts: `wm.adapt` (checkpoint `wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/b1_adapted_prebody.pt`), `wm.fit_body_head` (4x, `--data` pointed at stratified subsets), `eval_body_head_true_heldout.py` (unchanged). Subset dirs: `data/egocentric/beh12_b1_ego_flat_subset{3,6,12,24}` (relative symlinks into `_cleantrain`).

---

### F239. Direct null-vs-real action test run for the first time on Stage 1's own single-embodiment setup: action is redundant, confirming F142 across horizons 1-16

Section 6 motivated an "action might be redundant" concern from indirect evidence, but the direct null-vs-real test (F142) had only been run on the later two-body Stage 2 setup. Run here on the one surviving Stage-1-era checkpoint (`wm/runs/beh12_hexonly/best.pt`, `data/allocentric/beh12_c10f10t10_flat`, `action_necessity.py`, unmodified):

| lag | family | real | null | shuffled | mean | hold-still | null/real | real beats null |
|---|---|---|---|---|---|---|---|---|
| 1 | all | 1.5706 | 1.6238 | 1.6863 | 1.6491 | 2.3087 | **1.034** | 87.2% |
| 1 | side | 1.5100 | 1.5984 | 1.6412 | 1.6223 | 2.0999 | 1.059 | 93.9% |
| 1 | speed | 1.6144 | 1.6532 | 1.7047 | 1.6680 | 2.4785 | 1.024 | 85.3% |
| 1 | turn | 1.5880 | 1.6202 | 1.7131 | 1.6573 | 2.3502 | 1.020 | 82.4% |

null/real ≈ 1.034 pooled — real action changes one-step error by ~3%, matching F142's <3% figure on the setup Section 6 actually uses. real/hold = 0.680 — model reads real structure but does not use the action to do it, matching F142's reading exactly.

Same-day extension: sweep `--lags 1 2 3 5 8 16` confirms null/real stays in a narrow 1.03-1.09 band at every lag tested — no horizon where the action starts to matter; the flat result is not an artifact of a too-short window.

Script: `scripts/diagnostics/objective_experiments/action_necessity.py` (unmodified, run on a previously-untested Stage-1 checkpoint).

---

### F240. null/real still holds on the fully combined checkpoint (egocentric + hinge + rollout), not just egocentric alone, at every lag tested

F157's GATE C measured null/real on `wm/runs/beh12_ego` (egocentric only, no hinge/rollout) and found real sustained improvement over the 1.03 allocentric baseline (1.16-1.19, lags 1-5). Untested whether that survives once the hinge/rollout fix is layered on, on `beh12_hinge_cleansplit` (the checkpoint every clean number in the deck runs on).

Result (`action_necessity.py`, unmodified, held-out hexapod `_cleanheldout` clips):

| lag | null/real | real/hold | real beats null | reading |
|---|---|---|---|---|
| 1 | 1.062 | 0.577 | 87.4% | action buys something |
| 2 | 1.106 | 0.538 | 94.9% | viable |
| 3 | 1.115 | 0.534 | 95.9% | viable |
| 5 | 1.109 | 0.536 | 96.9% | viable |
| 8 | 1.093 | 0.541 | 96.1% | action buys nothing |

null/real stays 1.06-1.12 at every lag, clear of the 1.03 allocentric floor and of the ~1.0 collapse seen in the LDAD arm. Weaker than egocentric-alone's best lags (F157: 1.16-1.19), but this is a held-out, stratified, leak-free split F157 lacked, and the fix stacks on top of egocentric rather than trading it away.

Script: `scripts/diagnostics/objective_experiments/action_necessity.py` (unmodified). Checkpoint: `wm/runs/beh12_hinge_cleansplit/best.pt`. Data: `data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout`.

---

### F241. Section 10's held-out B1 table re-measured under z = proj(action): weaker overall than the ITM path, not catastrophically, and better on forward

Section 10's table used `z = ITM(e_t, e_{t+1})`, which needs the future frame. At selection time only `z = proj(action)` exists, and that path had never been scored on held-out data.

Result (`scripts/diagnostics/objective_experiments/proj_path_eval.py`; `beh12_hinge_cleansplit` B1 head as-is, fresh `projector_clean.pt` fit on `_cleantrain`, 12 disjoint `_cleanheldout` clips, 780 transitions, same head/clips for both rows):

| z source | held-out ratio | forward rho | lateral rho | yaw rho | median rho |
|---|---|---|---|---|---|
| ITM(e_t,e_t+1) | 0.730 | +0.261 | +0.578 | +0.474 | +0.474 |
| proj(action) | 0.860 | **+0.438** | +0.393 | +0.376 | +0.393 |

ITM row reproduces Section 10 exactly (sanity check). Projector path still beats the mean (ratio < 1.0, enough for ordering candidates from different behaviours) but is worse on lateral/yaw and overall ratio; forward is better (+0.438 vs +0.261).

Caveat: the head was fit on ITM z (stage 4 default), not refit on projector z; a head fit with `--latent projector/both` may close part of the gap — not run.

Reading: the projector is not the single bottleneck — the ITM route itself is mediocre (ratio 0.73, forward rho 0.26); fine-ranking failures appear through both, but the projector route does cost lateral/yaw signal.

---

### F242. Direct action selection over a B1 motor-babble library works once every stage (not just the projector) is fit on the babble

Test: `scripts/diagnostics/objective_experiments/babble_library_eval.py`, drivers `scripts/run/babble_library_compare.sh`, `babble_library_full_adapt.sh`, logs `results/wm/dataset/b1_babble/library_*_log.txt`. Planner picks the candidate whose `body_head(proj(a))` is closest to goal_t (12 hexapod-clip goals, physics-read every horizon=2 steps), graded by true local Froude at the same offset.

| library | protocol | selected | oracle | random | % of random-to-oracle gap recovered |
|---|---|---|---|---|---|
| expert (24) | clean pipeline | 0.0733 | 0.0315 | 0.1216 | 54% |
| babble v2 (36) | projector only on babble | 0.1273 | 0.0437 | 0.1272 | 0% |
| babble spring (40) | projector only on babble | 0.1097 | 0.0739 | 0.1109 | 3% |
| babble v2 (36) | stages 1,2,4 all on babble | 0.0928 | 0.0437 | 0.1272 | 41% |
| babble spring (40) | stages 1,2,4 all on babble | 0.0930 | 0.0739 | 0.1109 | 48% |

Fitting only the projector on babble leaves the expert-fit head misreading babble z (near-chance results); fitting every stage on babble raises both libraries to 41-48%, close to expert's 54%. v2 covers goal space well (oracle 0.044) but has a worse scorer (loss 0.049); spring has a better scorer (loss 0.019) but is forward-only, so even its oracle misses lateral goals (side_* oracle 0.12-0.16).

Scope: 12 goals, one run, no seed variance; libraries differ in size/composition (24/36/40); projector and head fit in-sample per library; goals are hexapod clips read by physics, not vision.

---

### F243. The `data/egocentric/` "emergency" was tracked-but-deleted files, not real data loss; a second, unrelated symlink bug and a family-grouping bug were found alongside it

A machine restart left `data/egocentric/` looking damaged (621 files apparently missing across beh12/beh24 x hexapod/B1). `git status` showed all 621 as `D` (tracked, deleted from disk), not `??`; blobs were intact in `HEAD`. `git checkout -- data/egocentric/` restored all 621, no data lost. Anyone finding `data/egocentric/` looking damaged should check `git status --porcelain <path> | awk '$1=="D"'` before assuming re-collection is needed.

Second, unrelated bug found after the checkout: 213 more symlinks broken by wrong relative-path depth (pointing one directory too deep) plus a few with stale absolute `/home/aria/...` targets (same pattern as the 2026-09-23 fix). Fixed by indexing real `.npz` files by basename and relinking by basename + family name.

Third bug, in code: `family_z_ceiling.py` and `froude_ceiling.py`'s `FAMILY` grouping used `cond.rsplit("_", 1)[0]`, correct for beh12's single-suffix names but wrong for beh24's two-suffix names (e.g. `speed_c5.8_bwd`, `turn_s0.05_neg`) — it silently excluded every `_bwd`/`_neg` condition from ceiling checks. Both scripts now parse the numeric token specifically and keep the mode suffix, falling back to old behavior for beh12-style names.

Also removed: 12 dead nested clean-split duplicate directories (untracked, symlink-only, unreferenced except in 3 diagnostic-script docstrings, repointed to canonical paths) and an untracked, unreferenced `data/egocentric/fine_tune/` directory.

Scripts: `scripts/diagnostics/objective_experiments/family_z_ceiling.py`, `froude_ceiling.py`. Docs: `data/README.md` (new egocentric section), `data/README_LAYOUT.md` (both bugs logged in symlink-incident history).

---

### F244. `z`'s raw geometry is dominated by gait phase (83%), not condition identity (17%) -- measured directly for the single-body, cross-condition case, not inherited from F16

F16's "z is 64% gait phase" measured a different axis (body identity across five training bodies at matched phase, `m3d_bracketed`), not condition-vs-phase within one body; applying it here was importing the wrong number.

Method: nested variance decomposition on `beh12_hinge_cleansplit` (hexapod `beh12_c10f10t10_ego_flat_cleantrain`, 24 clips, 1548 transitions, all 64 raw z dims): between-condition 16.85%, between-clip-within-condition 0.05%, within-clip residual 83.10%. Room/clip-instance identity is negligible. PCA on the within-clip residual (PC1 60%, PC2 26%) plus per-clip autocorrelation shows a periodic signature (trough at lag 4-5, secondary peak at lag 11-12), consistent with a shared base gait cycle common to all conditions.

This reconciles the raw-ceiling-ratio (~0.98, `family_z_ceiling.py`) vs. RidgeCV probe R2 discrepancy: raw ceiling ratio is dominated by the 83% gait-phase nuisance direction across all 64 dims, while a linear probe can isolate the 17% condition-carrying subspace. Both readings are correct, answering different questions.

Scripts: session scratch (variance decomposition + PCA + autocorrelation against `wm/runs/beh12_hinge_cleansplit/best.pt`, run on CPU to avoid GPU contention with a concurrent run, see F246), not preserved on disk past `/tmp`; not yet a real script under `scripts/diagnostics/objective_experiments/`.

---

### F245. Three candidate fixes for F244 (reweighting, margin loss, ActionProjector-in-place-of-ITM) were tested and ruled out on held-out probe R2

| fix | mechanism | raw ceiling ratio | held-out probe R2 (RidgeCV) |
|---|---|---|---|
| `lambda_body` 0.5->2.0 (`beh12_lambdabody2_cleansplit`) | reweight existing Froude-supervision loss | 0.990 (baseline 0.983, no change) | 0.458 (baseline 0.492, worse on every channel: fwd 0.775->0.743, lat 0.343->0.317, yaw 0.357->0.316) |
| margin/hinge loss on raw z (`beh12_margin_itm_earlystop`) | contrastive push on within/across family pairs | 0.983->0.276 (looks like a win) | 0.4816->0.415 (worse, held-out generalization degrades even as raw separation improves) |
| `ActionProjector(action)` in place of `ITM(e_t,e_t+1)` | hypothesized action input is phase-invariant | 0.980 (baseline 0.983, no change) | not measured, raw ratio alone was decisive |

`lambda_body` reweight was previously judged a dead end on raw ceiling ratio alone; re-run on RidgeCV R2 shows it actively hurts (confirmed twice, closed for real). Margin loss's failure is now understood: it optimizes separation across all 64 raw dims, 83% of which is the gait-phase nuisance direction per F244, disturbing the 17% subspace a linear probe already uses well. The ActionProjector premise was wrong: `fit_projector.py`'s target is `a_{t+1}`, the raw per-timestep joint command (`wm/fit_projector.py:56`), itself just as phase-correlated as z.

Scripts: `scripts/diagnostics/objective_experiments/margin_heldout_check.py` (rerun with `--baseline_ckpt wm/runs/beh12_hinge_cleansplit/best.pt --margin_ckpt wm/runs/beh12_lambdabody2_cleansplit/best.pt`); ActionProjector ceiling check was session scratch, not preserved.

---

### F246. More behavioral diversity (beh24, 24 conditions vs beh12's 12) improves held-out probe R2 on shared conditions and zero-shot rollout on a held-out body -- the one lever that has worked in this arc

Training: `beh24_hinge_cleansplit`, same recipe as `beh12_hinge_cleansplit`, on `beh24_c10f10t10_ego_flat_cleantrain`/`_cleanval` (3057 train pairs). Training survived a GPU OOM and a mid-run reboot via `wm.train --resume auto`.

Raw ceiling ratio: unchanged/marginally worse (1.025 vs beh12's 0.983) -- expected per F244/F245, wrong metric for this question.

Held-out probe R2 on beh12's own 12 original conditions (both checkpoints scored on `beh12_c10f10t10_ego_flat_cleanheldout`, deconfounded, apples-to-apples):

| channel | beh12_hinge_cleansplit | beh24_hinge_cleansplit |
|---|---|---|
| forward | 0.775 | 0.804 |
| lateral | 0.343 | 0.395 |
| yaw | 0.357 | 0.392 |
| overall | 0.492 | 0.530 |

Zero-shot FTM rollout on a held-out body (`beh12_c08f09t09_ego_flat`, different leg-segment ratios, no adaptation step):

| checkpoint | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| beh12_hinge_cleansplit | 1.75x | 1.93x | 1.92x | 1.81x |
| beh24_hinge_cleansplit | 1.81x | 2.01x | 2.00x | 1.87x |

Both well above 1.0x (beats a frozen frame), `moves` 0.49-0.56. beh24 wins at every horizon; three independent metrics agree that more behavioral diversity, with no other change, is a real working lever, unlike the loss-engineering fixes in F245.

Checkpoints: `wm/runs/beh24_hinge_cleansplit/{best.pt,last.pt}`.

---

### F247. Architecture exactly clones LAC-WM's hyperparameters; cross-augmentation and the 3-stage adaptation pattern were already present; two identity-leak mitigations exist but are never enabled; FTM temporal-history levers test null; rollout degrades past ~35 steps but that's not a live blocker

Verified exact hyperparameter match to LAC-WM (`doc/ref/notes_lac_wm.md`): hidden/heads 512/16, IDM/ITM blocks 4 (2 self + 2 cross), FDM/FTM blocks 8, z_dim 64 -- same on both sides, deliberate per `wm/models/ftm.py`'s own docstring.

Corrected two prior misconceptions: (1) cross-augmentation (`wm/train.py:181-182`, `cross_augment=True` by default) was already active in every checkpoint diagnosed this session; it does not fix F244's phase-dominance since gait phase is identical content in both augmented views (defends against pixel-matching, not phase). (2) The 3-stage LAC-WM adaptation recipe was already modeled deliberately (`wm/adapt.py` docstring: "This is LAC-WM's stage 1"); stage 2 is `wm.fit_projector`; stage 3 (joint projector+FDM finetune) is not a general pipeline component, only run once for gecko (`wm/runs/gecko_adapt_s3/`). No LoRA anywhere -- all adaptation is full-parameter finetuning, unlike LAC-WM's rank-2 LoRA.

Two mechanisms built for F16/F33's "33% of z is embodiment identity" finding -- `ftm_embodiment_channel` and `center_embeddings` -- are never enabled in any run (checked every `wm/runs/*/config.yaml`, zero matches). Cheap to test but not yet run since no pretrain run mixes embodiments yet.

Diagnosis: LAC-WM proved its unified-latent claim across 3 visually unrelated domains (150k trajectories, batch 512); this project trains single-embodiment/single-domain, ~24-96 clips, batch 8 -- roughly 3 orders of magnitude smaller. Gait phase is the cheapest shortcut available since it's present in 100% of clips regardless of condition; LAC-WM's domain diversity never gives phase-equivalent content that chance. F246's diversity result (real data diversity, not loss engineering, fixing the problem) is read as confirming this diagnosis.

FTM temporal-context levers, both null: frame-history (`e_{t-1},e_t` into frozen FTM): hexapod -0.051->-0.056, B1 +0.042->+0.044, no meaningful change. Action-history (`z_{t-1},z_t` concatenated as FTM condition, `scripts/diagnostics/objective_experiments/ftm_action_history_lever_check.py`): baseline gap +0.0443 vs history gap +0.0429, no meaningful change.

Egocentric VSM's claimed LSTM temporal fusion does not do what the paper describes: 5 frames are fused as 5 input channels into one non-recurrent Conv2d/ResNet50 pass (`ResNet_RNN.py:105-129`), producing one fused feature replicated 5x into an LSTM whose "sequence" axis is really the feature dims -- no real cross-frame aggregation happens there. The real temporal mechanism is on the action side: `input_pre_a=True` concatenates current+previous action (`ResNet_RNN.py:95-97`) -- exactly what the action-history FTM lever tested, and found null for this project. Other paper-vs-code mismatches: default backbone is ResNet-50 not ResNet18; Table 6's batch 128 belongs to a separate VO-model call (VSM itself uses batch 16); "LR decay patience 20" is actually the early-stopping counter, real `ReduceLROnPlateau` patience is 5.

FTM rollout usable horizon (`scripts/figures/plot_direct_vs_rollout_c08f09t09.py`, `beh24_hinge_cleansplit` + hexapod-only projector merged into `full_c08f09t09.pt`, held-out body `beh12_c08f09t09_ego_flat`, condition `turn_s0.29`): rollout competitive early (0.0576 vs 0.0636 mean error, t<35), degrades after (0.0765, t>=35, 1.33x rise) while direct does not (0.90x). Confirmed as genuine closed-loop error accumulation, not a clip-length artifact. Not a live blocker: Egocentric VSM re-grounds every step, LAC-WM every 6-20 steps -- keep planner re-grounding cadence well under ~35 steps rather than extending rollout accuracy.

Bug fixed: `final_2x2x2_test.py`'s `true_local_froude(cand, spec, offset, horizon)` returned NaN when `offset` exceeded a candidate clip's length (general risk with variable-length pools, e.g. c08f09t09: 57-66 frames); now clamps offset to the clip's last valid index.

Scripts: `scripts/figures/plot_direct_vs_rollout_c08f09t09.py`, `scripts/diagnostics/objective_experiments/zero_shot_rollout_c08f09t09.py`, `scripts/diagnostics/objective_experiments/ftm_action_history_lever_check.py`. Doc: `ARCHITECTURE.md` (new). Checkpoints: `wm/runs/beh24_hinge_cleansplit/{best.pt,full_c08f09t09.pt,projector_c08f09t09.pt}`. Reference notes: `doc/ref/notes_lac_wm.md`, `doc/ref/notes_egocentric_vsm.md`.

---

### F248. `wm.adapt`'s full-parameter Stage 1 finetune catastrophically forgets the original embodiment; LoRA rank-2 (newly implemented) recovers most of it with no rehearsal data, now the default

Adapting `beh24_hinge_cleansplit` to B1 (Stage 1 `wm.adapt`, Stage 2 `wm.fit_projector`, `--clips 9 --stratify --lambda_hinge 0.5`), checking BOTH embodiments' rollout-gap ratio after Stage 2 (Stage 1's own printout only checks the target robot):

| embodiment | full-parameter Stage 1 | LoRA rank-2 Stage 1 |
|---|---|---|
| hexapod (pretrained on) | 0.918 (near-total forgetting) | 0.786 (meaningfully recovered) |
| b1 (adaptation target) | 0.307 (good) | 0.319 (same, within noise) |

Full finetune's target-robot `after` is also worse than `before` at every horizon (1.58->1.47 at h=1, 1.43->1.18 at h=10) despite decreasing training loss -- an overfitting/forgetting signature the existing script didn't flag.

LoRA had never been implemented in this codebase before, despite `wm/adapt.py`'s docstring naming LAC-WM's staged recipe as the template. New: `wm/models/lora.py` (`LoRALinear`, `apply_lora`, `merge_and_unwrap_lora`), wired into `wm/adapt.py` via a flag.

Scope limitation: `nn.MultiheadAttention`'s fused forward reads `out_proj.weight`/`.bias` as raw tensors, so a generic wrapper breaks it, and `in_proj_weight` is a single fused raw Parameter -- LoRA here covers only each block's `Mlp` (two plain `nn.Linear`s), not every weight matrix. After training, `merge_and_unwrap_lora` folds the delta into plain `nn.Linear` weights so the checkpoint loads unchanged in every other script.

Decision: LoRA rank-2 is now `wm.adapt`'s default (`--no_lora` restores old full-finetune behavior). Based on one clean comparison, not a sweep; rank taken directly from LAC-WM's reported value, not independently tuned; only hexapod->B1 tested so far.

Unrelated bug fixed: `--lambda_hinge`'s help string contained a bare `%` ("38-62% drop"), crashing `argparse.print_help()` since argparse treats help strings as `%`-format templates; escaped to `%%`.

Scripts/modules: `wm/models/lora.py` (new), `wm/adapt.py` (`--no_lora`/`--lora_rank` flags). Checkpoints: `wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24/` (full-finetune, comparison only, not recommended), `wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora/` (current best cross-embodiment checkpoint).

---

### F249. LoRA rank-2 Stage 1 is reproducible and flat across clip budget -- 3 clips is enough, the prior default of 9 bought nothing

Sweep (`beh24_hinge_cleansplit` -> B1, LoRA rank-2, `--stratify`, Stage 1 + Stage 2, same recipe as F248): `--clips` in {3, 6, 9, 15}.

| clips | hexapod rollout gap (vs mean-z) | b1 rollout gap (vs mean-z) |
|---|---|---|
| 3 | 0.764 | 0.315 |
| 6 | 0.768 | 0.303 |
| 9 | 0.788 | 0.302 |
| 15 | 0.766 | 0.307 |

`clips=9` (0.788/0.302) reproduces F248's original run (0.786/0.319) on a fresh draw, confirming F248's LoRA-vs-full-finetune result wasn't a one-off favorable draw. All four points sit in a tight band; 3->15 clips (5x data) moves neither number meaningfully, extending the prior full-finetune "3 clips is enough" finding to LoRA. `wm/adapt.py`'s default changed from `--clips 9` to `--clips 3` to match.

Scope: one embodiment pair (hexapod->B1), one sweep, no seed variance beyond the incidental clips=9 reproduction; not claimed to generalize to a larger domain gap (e.g. gecko).

Checkpoints: `wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c{3,6,9,15}/`.

---

### F250. With all four pipeline stages run, the beh24 pipeline (LoRA, 3 clips) and the beh12 pipeline select about equally well; the earlier apparent 3x regression came from a skipped Stage 4

Same goal clips (`beh12_c10f10t10_ego_flat_cleanheldout`), same candidate library
(`beh12_b1_ego_flat_cleantrain`), both pipelines with Stage 4 (`wm.fit_body_head`, hexapod
rehearsal), six conditions, mean L2 error of the picked candidate's true local Froude to the goal:

| condition | OLD direct | NEW direct | OLD rollout | NEW rollout |
|---|---|---|---|---|
| turn_s0.29 | 0.053 | 0.066 | 0.149 | 0.133 |
| turn_s0.56 | 0.058 | 0.065 | 0.159 | 0.145 |
| side_L_lvl0 | 0.084 | 0.098 | 0.151 | 0.122 |
| side_R_lvl1 | 0.034 | 0.045 | 0.179 | 0.153 |
| speed_c7.1 | 0.069 | 0.069 | 0.119 | 0.153 |
| speed_c8.8 | 0.061 | 0.086 | 0.136 | 0.190 |
| **mean** | **0.060** | **0.072** | **0.149** | **0.149** |

OLD = `beh12_hinge_cleansplit` + full finetune; NEW = `beh24_hinge_cleansplit` + LoRA rank 2, 3 clips.
Without Stage 4, NEW's body head has never seen a B1 latent: direct on turn_s0.29 was 0.161, 0.067
with it. A partial pipeline (Stages 1-2 only) produces a checkpoint that loads everywhere and
selects badly. The remaining direct gap is the in-sample fit of OLD's Stages 2/4 (F254).

F246's gains for beh24 (probe R2, zero-shot rollout) do not carry into selection.

Checkpoints: `wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c3/ckpt_lib_s4.pt`,
`wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/ckpt_lib_beh12_b1_ego_flat_cleantrain.pt`.

---

### F251. Single-step `z` carries per-step body motion, not the 1-second Froude that selection is graded on; reading over ~11 steps (one gait cycle) improves selection

Probe (`timescale_probe.py`, `beh24_hinge_cleansplit/best.pt`, hexapod train -> held-out, RidgeCV),
`z` moving-averaged over W steps, against the smoothed Froude (`body_motion`, ~20-step centred
average) and the same quantities unsmoothed:

| z window W | R2 vs smoothed Froude | R2 vs per-step Froude |
|---|---|---|
| 1 | 0.441 | 0.656 |
| 5 | 0.546 | **0.740** |
| 11 | 0.622 | 0.408 |
| 21 | **0.682** | 0.390 |

The 83% within-clip periodic variance of `z` (F244) is real within-stride motion.

Selection with a read-out window (`window` on both planners, default 0 = original scoring), decision
every 2 steps, same setup as F250:

| pipeline | path | w=0 | w=11 | w=21 | improved at w=11 |
|---|---|---|---|---|---|
| OLD | direct | 0.060 | 0.053 | 0.052 | 5/6 |
| OLD | rollout | 0.149 | 0.097 | 0.113 | 6/6 |
| NEW | direct | 0.072 | 0.063 | 0.067 | 4/6 |
| NEW | rollout | 0.149 | 0.130 | 0.136 | 3/6 |

11 steps (close to the 11-12-step gait period) is the best selection window. Scope: six goals, one run
each, rollout from one fixed start frame.

Scripts: `timescale_probe.py`, `wm/policy/planner.py` (`window`), `plot_direct_vs_rollout.py --window`.

---

### F252. Rollout selection fails because the FTM's prediction is driven by the frame it starts from, not by the action -- with the true `z` as well

Library bounds (`selection_eval.py`; oracle = best candidate per step, random = mean over
candidates; gap = share of random-to-oracle recovered): oracle 0.031, random 0.126.

| pipeline | w | direct | roll_fixed | roll_live |
|---|---|---|---|---|
| OLD | 0 | 0.060 (+0.70) | 0.148 (-0.23) | 0.139 (-0.13) |
| OLD | 11 | 0.053 (+0.77) | 0.098 (+0.30) | 0.117 (+0.10) |
| NEW | 0 | 0.072 (+0.57) | 0.149 (-0.24) | 0.139 (-0.13) |
| NEW | 11 | 0.063 (+0.66) | 0.130 (-0.04) | 0.150 (-0.25) |

Across-candidate correlation of predicted with true Froude, per step (fwd / lat / yaw):

| read-out | OLD | NEW |
|---|---|---|
| direct `body(proj(a_i))` | 0.64 / 0.56 / 0.69 | 0.54 / 0.67 / 0.33 |
| `body(ITM(e_i[t], e_i[t+1]))` (real next frame) | 0.43 / 0.56 / 0.76 | 0.59 / 0.56 / 0.51 |
| rollout from the candidate's own frame | 0.32 / 0.46 / 0.75 | 0.54 / 0.37 / 0.39 |
| rollout from one shared frame (selection) | 0.01 / 0.21 / -0.17 | 0.07 / 0.14 / 0.00 |

Full 24 x 24 grid of start frame x action, `P[s, a] = body(ITM(e_s, FTM(e_s, z_a)))`:

| checkpoint, z source | state share | action share | r_row fwd / lat / yaw |
|---|---|---|---|
| OLD, proj(a) | 0.63-0.87 | 0.02-0.11 | 0.04 / 0.15 / 0.01 |
| OLD, true z | 0.59-0.85 | 0.02-0.13 | 0.06 / 0.17 / 0.00 |
| NEW, proj(a) | 0.49-0.83 | 0.02-0.16 | 0.04 / 0.14 / 0.02 |
| NEW, true z | 0.40-0.79 | 0.03-0.17 | 0.04 / 0.18 / 0.05 |
| beh24 pretrained, hexapod val, true z | 0.26-0.66 | 0.15-0.23 | 0.23 / 0.20 / 0.00 |

(r_row = correlation across actions with their true Froude, from a fixed start frame.) The round trip
ITM(e, FTM(e, z)) keeps 16-60% of z's across-action variance.

- Rollout's Froude is mostly the motion already in the start frame (a V-JEPA2 frame spans two video
  frames). Rolling a candidate from its own frame tracks truth for that reason.
- The projector is not the cause: the true z gives the same result.
- r_row is invariant to any per-frame offset, so no read-out change can recover it.
- Present on the pretraining body (hexapod), weaker there.

Scripts: `selection_eval.py`, `selection_readout_diag.py`, `rollout_state_action_anova.py`.

---

### F253. A counterfactual cycle loss makes the FTM follow `z`; with the ITM trainable it also degrades `z`'s Froude read-out, and selection gets worse

`lambda_cycle` (`wm/config.py`, default 0): z shuffled across the batch; the ITM (weights detached for
this term) must recover it from ITM(e_t, FTM(e_t, z_shuffled)). 6-epoch fine-tunes from
`beh24_hinge_cleansplit/best.pt`, cycle vs identical control, same four-stage B1 pipeline.

| | action share (fwd) | r_row fwd / lat / yaw | cycle cos |
|---|---|---|---|
| hexapod val, true z, control | 0.12 | 0.26 / 0.23 / 0.01 | 0.30 |
| hexapod val, true z, cycle | 0.94 | 0.73 / 0.53 / 0.20 | 0.98 |
| B1 library, proj(a), control | 0.06 | 0.05 / 0.17 / 0.02 | 0.34 |
| B1 library, proj(a), cycle | 0.19 | 0.33 / 0.50 / 0.10 | 0.84 |

Selection at w=11: control direct 0.062 / roll_live 0.127; cycle 0.086 / 0.154. `body(proj(a))` R2
forward on the B1 library: control 0.35, cycle 0.12. Val body loss 0.67 (cycle) vs 0.55 (control).

Runs: `wm/runs/beh24_ft_{cycle,ctrl}`.

---

### F254. With the ITM frozen, the cycle loss improves rollout selection in 6/6 conditions; OLD's direct advantage came from fitting Stages 2 and 4 on the candidate library itself

`--freeze_modules itm` (`wm/config.py`, default empty). Pair `wm/runs/beh24_ft_{cycle,ctrl}_fz`; val
body loss identical (0.534), recon 3.71 vs 3.60. OLD's projector and body head were fit on the
candidate library (`scripts/run/clean_retrain.sh`); "in" rows below are fit the same way, "out"
rows on `beh24_b1_ego_flat_cleantrain`.

| arm | fit | w | direct | roll_fixed | roll_live |
|---|---|---|---|---|---|
| OLD | in | 11 | 0.053 (+0.77) | 0.098 (+0.30) | 0.117 (+0.10) |
| control | out | 11 | 0.064 (+0.65) | 0.140 (-0.14) | 0.142 (-0.17) |
| cycle | out | 11 | 0.071 (+0.59) | 0.131 (-0.05) | 0.123 (+0.04) |
| control | in | 11 | **0.056 (+0.74)** | 0.136 (-0.10) | 0.133 (-0.07) |
| cycle | in | 11 | 0.064 (+0.66) | **0.097 (+0.31)** | **0.102 (+0.25)** |

Cycle beats control in 6/6 conditions for roll_live and 5/6 for roll_fixed (in-sample, w=11). Its
direct is ~0.008 worse in both regimes. Yaw read from proj(a) on B1 is below the mean for every beh24
pipeline (R2 -0.27 control, -0.45 cycle, out-of-sample).

Scope: six goals, one seed, 6-epoch fine-tunes, lambda_cycle 1.0.

---

### F255. Withdrawn -- rendered closed-loop numbers measured on an incorrect ego view (F256); the valid result is F257

---

### F256. Rendered closed-loop rollout results before 2026-09-25 used an out-of-distribution ego view, and four B1 ego datasets were rendered with the wrong lens

- `sim/control/close_loop_direct_froude.py` created its ego sensor at 24 deg; every training clip is
  90 deg (`render_b1_replay.py --ego`). It also composed full 3-D pose deltas across candidate
  switches, accumulating pitch, roll and height.
- Row-luminance-profile correlation of first frames with the correct B1 ego clips: correct renders
  0.980-0.995; that closed loop 0.67-0.70; `b1_babble_ego_flat` and `b1_babble_v2_ego_flat` 0.81;
  `b1_babble_coppelia_spring_ego_flat` 0.92-0.94; `beh12_b1_more_ego_flat` 0.43-0.55. All beh12/beh24
  B1 ego sets are correct.
- The babble collector and the CPG controller set the FOV before `startSimulation`, which restores
  the scene sensor's authored 15 deg.
- Affected: the `--mechanism rollout` halves of F185, F186, F188, F207 (marked); their result files
  are in `results/_invalid_F256/`. `--mechanism direct` never reads the loop's camera and is unaffected.
- The code now sets the ego FOV in one place (`EGO_FOV_DEG`, set and read back by `attach_ego` /
  `set_ego_fov`), the closed loop refuses a first frame with profile correlation < 0.97, and
  `--level_body` (planar position and heading composed, attitude from the candidate) is its default.

---

### F257. In a rendered closed loop with the correct view, the cycle loss's rollout beats control in 6/6 conditions (gap +0.37); direct remains best on average

CoppeliaSim, 90-deg ego camera re-rendered every step -> V-JEPA2 -> planner every 2 steps, kinematic
posing, F254's in-sample checkpoints, same six goals. Mean L2 error (gap):

| run | turn .29 | turn .56 | side L0 | side R1 | speed 7.1 | speed 8.8 | mean |
|---|---|---|---|---|---|---|---|
| rollout control, w0 | 0.112 | 0.142 | 0.115 | 0.156 | 0.133 | 0.150 | 0.135 (-0.09) |
| rollout control, w11 | 0.131 | 0.143 | 0.171 | 0.121 | 0.158 | 0.185 | 0.151 (-0.26) |
| rollout cycle, w0 | 0.112 | 0.105 | 0.132 | 0.114 | 0.121 | 0.152 | 0.123 (+0.04) |
| rollout cycle, w11 | 0.087 | 0.102 | 0.093 | 0.085 | 0.086 | 0.097 | **0.092 (+0.37)** |
| direct cycle, w11 | 0.044 | 0.056 | 0.101 | 0.040 | 0.045 | 0.109 | 0.066 (+0.63) |

Rollout + cycle beats direct on side_L_lvl0 and speed_c8.8. Scope: kinematic (cannot fall), one seed,
six goals, in-sample Stage 2/4. Clips and plots: `results/deck/cycle_closed_loop/`.

---

### F258. The cycle loss's gain is a pass-through of `z`, not better prediction: against rendered counterfactual outcomes it predicts no better than control

8 library states x 4 steps; from each, all 24 candidates' one-step motion applied and rendered
(e*(s, a)); compared with FTM(e_s, proj(a)), F254's in-sample checkpoints:

| | control | cycle |
|---|---|---|
| raw-embedding nearest outcome, top1 (chance 0.042) | 0.061 | 0.053 |
| action-specific direction, top1 (chance 0.042) | 0.137 | 0.146 |
| action-specific direction, cos | +0.101 | +0.084 |
| read of the REAL outcome vs truth, r fwd / lat / yaw | 0.19 / 0.17 / 0.14 | 0.14 / 0.22 / 0.19 |
| read of the PREDICTION vs truth | 0.11 / 0.17 / -0.04 | 0.63 / 0.42 / 0.10 |

The cycle FTM's prediction is far more readable than the real future: it carries the action into a
form the frozen ITM decodes. Its recon on real pairs is 3% worse than control. Readers trained on
real frames cannot judge counterfactuals (real frame change -> Froude R2 ~0; single frame -> Froude
identifies the clip, within-clip R2 -2.0).

Scripts: `counterfactual_truth_check.py`, `independent_reader_check.py`.

---

### F259. The action becomes visible in the real future only over several steps

From 8 library states x 3 steps, all 24 candidates' motion applied for 11 steps and rendered (real
counterfactual trajectories, no model prediction):

| k | w=k read on real frames vs truth, r fwd / lat / yaw | model-free probe R2, yaw |
|---|---|---|
| 1 | 0.20 / 0.15 / 0.15 | 0.16 |
| 2 | 0.28 / 0.26 / 0.19 | -0.21 |
| 5 | 0.53 / 0.62 / 0.44 | 0.49 |
| 11 | 0.60 / 0.77 / 0.77 | 0.80 |

Over one 50 ms step the action is nearly invisible even in reality, so a one-step FTM ignoring `z`
(F252, F77) is close to optimal for its objective. The linear probe cannot read forward/lateral at
any k.

Script: `counterfactual_horizon_check.py` (cache `results/wm/cache/counterfactual_horizon.pt`).

---

### F260. The pipeline supports a world-model stride k; stride 1 is numerically unchanged

`wm/data/strided.py` defines a transition at `t`, stride k: frames `e_t -> e_{t+k}`, commands
`actions[t+lag : t+lag+k]`, target Froude averaged over `[t, t+k)`. Used by pretraining
(`wm/data/dataset.py`), Stage 1 (`wm/adapt.py`), the projector (input `(k, dim)`), Stage 2
(`wm/fit_projector.py`), Stage 4 (`wm/fit_body_head.py`), both planners (stride read from the
projector) and `rollout_state_action_anova.py`.

- Stride 1 reproduces `selection_eval` on OLD (0.0599 / 0.1482 / 0.1391 at w0, 0.0534 / 0.0976 /
  0.1172 at w11) and F252's grid exactly.
- Stride 5: frame gaps, action windows and targets checked with dummy models; planner scores match
  hand computation to 1e-6; the full pipeline runs end to end.
- `fit_body_head.py --latent projector|both` used actions one step early before this change (the
  pipeline's default `--latent itm` was unaffected).

---

### F261. A world model that steps 5 frames at a time selects actions better on every path and predicts the real counterfactual future better -- the first change that improves prediction and selection together

`beh24_stride5_cleansplit`: identical to `beh24_hinge_cleansplit` (same data, losses, 50 epochs from
scratch) except `--frame_stride 5` (5-command action chunk, F260). Both through the identical
four-stage B1 pipeline; "out" = Stages 2/4 fit on `beh24_b1_ego_flat_cleantrain`, "in" = fit on the
candidate library (how OLD was built). Selection (`selection_eval.py`, oracle 0.031, random 0.126):

| model | fit | w | direct | roll_fixed | roll_live |
|---|---|---|---|---|---|
| stride 1 | out | 0 | 0.072 (+0.57) | 0.149 (-0.24) | 0.139 (-0.13) |
| stride 1 | out | 11 | 0.063 (+0.66) | 0.130 (-0.04) | 0.150 (-0.25) |
| **stride 5** | out | 0 | 0.056 (+0.74) | 0.127 (-0.01) | 0.114 (+0.13) |
| **stride 5** | out | 11 | **0.048 (+0.82)** | **0.084 (+0.45)** | **0.089 (+0.40)** |
| stride 1 | in | 11 | 0.060 (+0.69) | 0.125 (+0.01) | 0.119 (+0.08) |
| **stride 5** | in | 11 | **0.040 (+0.91)** | **0.083 (+0.45)** | 0.094 (+0.34) |
| OLD (beh12) | in | 11 | 0.053 (+0.77) | 0.098 (+0.30) | 0.117 (+0.10) |

Out-of-sample at w=11, stride 5 beats stride 1 in 6/6 conditions for direct and 6/6 for roll_live
(e.g. speed_c8.8 roll_live 0.111 vs 0.193; turn_s0.56 0.069 vs 0.145). Direct in-sample reaches 91% of
the library's oracle gap, the best selection measured in this project.

Prediction against rendered counterfactual 5-step futures (F259's cache; `counterfactual_prediction_k.py`;
stride 5 = one FTM step with a 5-command chunk, stride 1 = five one-step FTM steps; out-of-sample
checkpoints):

| model | direction top1 (chance 0.042), ITM-free | read of prediction r fwd/lat/yaw | read of real outcome r |
|---|---|---|---|
| stride 1 | 0.161 | 0.04 / 0.22 / 0.03 | 0.32 / 0.27 / 0.24 |
| stride 1 + cycle (F254) | 0.158 | 0.17 / 0.24 / 0.05 | 0.34 / 0.31 / 0.25 |
| **stride 5** | **0.233** | 0.10 / 0.23 / 0.21 | 0.24 / 0.52 / 0.24 |

**Reading.**
- Stepping at the timescale where the action shows in the real future (F259) makes the FTM's
  action-specific prediction point at the right real outcome more often (0.233 vs 0.161), with no
  auxiliary loss; the cycle loss (F258) does not.
- Unlike the cycle loss, stride 5's prediction is not more readable than reality (pred r below real r
  on every channel): the selection gain is not a pass-through of z.
- Direct improves too, although it never uses the FTM: z = ITM(e_t, e_{t+5}) and its target (Froude
  over 5 steps) sit closer to the 1-second Froude selection is graded on (F251), and the projector
  maps a 5-command chunk, which carries the gait rather than one joint snapshot.
- Rollout remains behind direct (+0.40 vs +0.82 out-of-sample, w=11).

**Scope.** One seed per model, six goals, recorded-frame evaluation (`roll_live` replays the chosen
candidate's recorded frame); the rendered closed loop (F257's setup) has not been run for stride 5.
Val losses at epoch 50 are not comparable across strides (different target spacing).

Checkpoints: `wm/runs/beh24_stride5_cleansplit/{best.pt, b1_lora_c3/ckpt_lib_s4.pt, b1_lora_c3/ckpt_lib_insample.pt}`.
Raw: `results/deck/window_sweep/stride5_selection.txt`.

---

### F262. At stride 5 the pretrained body head reads the new body's latent without refitting (direct gap +0.59); at stride 1 it cannot (-0.32) -- the stride-5 latent is far more shared across bodies

> **Reading corrected by F272:** a direct test finds B1's latents are NOT shared with the hexapods' at either stride; what holds is the selection result through proj(a), not a shared latent.

Stage 4 (`wm.fit_body_head`) refits the shared body head on the new body's latents and needs that body's
Froude labels, a requirement LAC-WM does not have (it drops the motion decoder after pretraining).
Skipping it -- the body head is the pretrain's, which has seen only the hexapod -- with Stages 1-2 as
in F261 (`ckpt_lib_nostage4.pt`), same selection test:

| model | Stage 4 | direct w0 | direct w11 | roll_live w11 |
|---|---|---|---|---|
| stride 1 | no | 0.144 (-0.18) | 0.157 (-0.32) | 0.198 (-0.75) |
| stride 1 | yes | 0.072 (+0.57) | 0.063 (+0.66) | 0.150 (-0.25) |
| stride 5 | no | 0.073 (+0.56) | **0.070 (+0.59)** | 0.111 (+0.16) |
| stride 5 | yes | 0.056 (+0.74) | 0.048 (+0.82) | 0.089 (+0.40) |

**Reading.** At stride 1 the B1's latents land where the hexapod-trained head reads nonsense, so the
head must be refit on B1 labels. At stride 5 the same unadapted head reads B1 latents well enough to
beat the stride-1 pipeline *with* Stage 4 (0.070 vs 0.063 is within noise; both far above random) --
a latent spanning a gait-scale interval is shared across the two bodies, which is the property the
project's cross-embodiment claim needs. Stage 4 still adds (+0.59 -> +0.82) but is no longer
necessary, so the new body can be adapted from frames and actions only (Stages 1-2), as in LAC-WM.

**Also established** (`doc/ref/notes_lac_wm.md` 5.3-5.5): LAC-WM adapts with one finetuning dataset
(7,265 trajectories) used by all three stages, split by iterations (20k / 5k / 35k); it makes no
few-shot claim. In this project Stage 1 used 3 B1 clips while Stages 2 and 4 used ~38
(`beh24_b1_ego_flat_cleantrain`, 80% split), so the earlier "3 clips suffice" (F249) describes Stage 1
alone, not the adaptation as a whole.

Scope: one seed, six goals, physics goal (the hexapod side never passes through the body head here).

---

### F263. The stride-5 gain holds in the rendered closed loop: direct 0.050 vs 0.069, rollout 0.092 vs 0.140 (w=11), stride 5 better in 6/6 and 5/6 conditions

CoppeliaSim, 90-deg ego camera re-rendered every step (view check passed on every run, row-profile
corr 0.973) -> V-JEPA2 -> planner every 2 steps, kinematic posing with `--level_body`, out-of-sample
Stage 2/4 fits (F261), same six hexapod goals and grading as F257. Mean L2 error (gap; oracle 0.031,
random 0.126):

| run | turn .29 | turn .56 | side L0 | side R1 | speed 7.1 | speed 8.8 | mean |
|---|---|---|---|---|---|---|---|
| rollout stride 1, w0 | 0.119 | 0.129 | 0.122 | 0.157 | 0.123 | 0.161 | 0.135 (-0.09) |
| rollout stride 1, w11 | 0.120 | 0.145 | 0.115 | 0.128 | 0.158 | 0.175 | 0.140 (-0.14) |
| direct stride 1, w11 | 0.072 | 0.071 | 0.087 | 0.040 | 0.060 | 0.085 | 0.069 (+0.60) |
| rollout stride 5, w0 | 0.076 | 0.090 | 0.129 | 0.157 | 0.106 | 0.141 | 0.117 (+0.10) |
| **rollout stride 5, w11** | 0.075 | 0.070 | 0.107 | 0.130 | 0.067 | 0.102 | **0.092 (+0.36)** |
| **direct stride 5, w11** | 0.044 | 0.057 | 0.050 | 0.038 | 0.043 | 0.070 | **0.050 (+0.80)** |

Stride 5 beats stride 1 in 6/6 conditions for direct and 5/6 for rollout at w=11 (side_R_lvl1 is the
exception, both at random level). The closed-loop numbers agree with the recorded-frame evaluation
(F261: direct +0.82, roll_live +0.40), so the gain survives real rendered observations. Rollout still
trails direct. Rollout stride 5 matches the cycle arm's closed-loop score (F257, 0.092) without any
auxiliary loss and without the pass-through (F258, F261).

Scope: kinematic (cannot fall), one seed, six goals. Driver `scripts/run/closed_loop_stride_grid.sh`;
raw `results/wm/closed_loop/stride_live/grid.txt`; plots and 4-panel clips
`results/deck/stride_closed_loop/` (`scripts/figures/closed_loop_report.py --preset stride`).

---

### F264. Stride 5 is the best of 1 / 5 / 10 on both test bodies (B1 adapted, c08f09t09 zero-shot); stride 10 falls back on B1

`beh24_stride10_cleansplit`: identical to the stride-1 and stride-5 pretrains (F261) except
`--frame_stride 10` (2,193 training pairs vs 2,673 at stride 5, 3,057 at stride 1). Selection at w=11,
gap in brackets (0 = random, 1 = library oracle):

**B1 (cross-embodiment; four-stage pipeline, Stages 2/4 out-of-sample; oracle 0.031, random 0.126)**

| stride | direct | roll_fixed | roll_live |
|---|---|---|---|
| 1 | 0.063 (+0.66) | 0.130 (-0.04) | 0.150 (-0.25) |
| **5** | **0.048 (+0.82)** | **0.084 (+0.45)** | **0.089 (+0.40)** |
| 10 | 0.063 (+0.67) | 0.103 (+0.25) | 0.107 (+0.20) |

In-sample: stride 1 / 5 / 10 direct +0.69 / +0.91 / +0.81, roll_live +0.08 / +0.34 / +0.30.

**c08f09t09 (held-out hexapod morphology, zero-shot: no c08 data anywhere -- hexapod projector fit on
the pretraining body c10f10t10, pretrain body head; goals: the same six c10f10t10 held-out clips;
candidates: c08's 48 clips; oracle 0.020, random 0.121)**

| stride | direct | roll_fixed | roll_live |
|---|---|---|---|
| 1 | 0.081 (+0.39) | 0.085 (+0.35) | 0.095 (+0.26) |
| **5** | 0.059 (+0.61) | **0.082 (+0.38)** | **0.082 (+0.39)** |
| 10 | **0.056 (+0.64)** | 0.097 (+0.24) | 0.094 (+0.27) |

**Reading.**
- Stride 5 is the best setting overall on both bodies; stride 10 does not continue the trend. On B1
  it loses most of stride 5's gain (direct back to stride-1 level, rollout +0.20 vs +0.40); on c08
  direct is marginally better than stride 5 (+0.64 vs +0.61) but rollout is worse (+0.27 vs +0.39).
- F259 found the real future more action-separable at 11 steps than at 5, yet the model does not
  profit from it. Candidate reasons, not separated: 18% fewer training pairs with clips of 66 frames;
  one FTM step of 10 frames is harder to predict (val recon 4.00 vs 3.93 at stride 5, epoch 50);
  coarser decisions -- with w=11 the rollout read-out is only two FTM steps.
- On the same-embodiment held-out body, stride 1 rollout is already above random (+0.26) where on
  B1 it is below (-0.25): part of B1's rollout failure is the cross-embodiment gap, not only the
  one-step FTM. Stride 5 improves direct on both bodies (+0.39 -> +0.61 on c08, zero-shot).

Scope: one seed per stride, six goals; c08 has only beh12 data (12 conditions x 4 clips), so its
library is 48 clips; no rendered closed loop for c08 (the closed-loop script supports B1 only).

Raw: `results/deck/window_sweep/{stride10_selection,c08_zeroshot_selection}.txt`. Checkpoints:
`wm/runs/beh24_stride10_cleansplit/`, `wm/runs/*/c08_zeroshot/ckpt_lib_zeroshot.pt`.
Scripts: `scripts/run/{stride10_full,c08_zeroshot_eval}.sh`; `selection_eval.py --embodiment`.

---

### F265. LAC-WM's Stage 3 (joint projector + FTM fine-tune, LoRA rank 2) improves the stride-5 model's prediction of the real future but not its selection; without a latent anchor it degrades the direct path

`wm/adapt_joint.py` (new): projector (fully trainable) and FTM (LoRA rank 2) fine-tuned end-to-end on
MSE(FTM(e_t, proj(a chunk)), e_{t+5}), ITM frozen, 2,000 steps, batch 8, lr 1e-4, on the Stage-2 B1
clips (`beh24_b1_ego_flat_cleantrain` minus the projector's 9 held-out clips). Optional
`--lambda_z` x MSE(proj(a), ITM z) keeps the projector on the ITM's latent (LAC-WM has no such term).
Applied to F261's stride-5 out-of-sample checkpoint; body head unchanged (Stage 4 not rerun).

| | held-out next-frame MSE, FTM(e, proj(a)) | proj(a) vs ITM z MSE | direction top1 vs real 5-step outcome (chance 0.042) | direct w11 | roll_live w11 |
|---|---|---|---|---|---|
| before (Stages 1, 2, 4) | 3.931 | 1.21 | 0.233 | 0.048 (+0.82) | 0.089 (+0.40) |
| + Stage 3, lambda_z 0 | 3.742 | 1.98 | **0.276** | 0.066 (+0.63) | 0.085 (+0.44) |
| + Stage 3, lambda_z 1 | 3.756 | 1.27 | 0.267 | 0.049 (+0.81) | 0.092 (+0.36) |

roll_fixed w11: +0.45 / +0.47 / +0.42. Read of the prediction vs truth (fwd/lat/yaw) stays below the
read of the real outcome (0.24 / 0.52 / 0.24), so the prediction gain is not a pass-through (F258).

**Reading.**
- Stage 3 makes the FTM a better predictor of the real future under the projector's latent: held-out
  error -5%, action-specific direction toward the true counterfactual outcome 0.233 -> 0.27.
- That does not turn into better selection: rollout moves +0.40 -> +0.44 (lambda_z 0) or +0.36
  (lambda_z 1), within the spread of single-seed runs.
- With no anchor the projector's latent drifts from the ITM's (1.21 -> 1.98) and the direct path,
  whose body head was fit on ITM latents, drops +0.82 -> +0.63. The anchor removes the drift and the
  loss.
- The rollout's remaining gap to direct is therefore not closed by making the one-step prediction
  more accurate on average; the read-out of the imagined trajectory (ITM + body head on predicted
  frames) is a candidate bottleneck, not tested here.

Scope: one seed, 2,000 steps (LAC-WM: 35k), six goals, recorded-frame evaluation.
Checkpoints: `wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_s3_lz{0,1}.pt`.
Raw: `results/deck/window_sweep/stage3_selection.txt`.

---

### F266. The rollout's ceiling is set by the read-out, not the forward model: reading even the REAL counterfactual future through ITM + body head tracks the actions' Froude at r 0.24 / 0.52 / 0.24, against 0.83 / 0.91 / 0.84 for direct

`readout_bottleneck_check.py` (new). F259's rendered counterfactual trajectories (8 library states x
3 steps; every candidate's own motion applied from a shared start state and rendered). Correlation
across the 24 actions with their true Froude over [t, t+k), fwd / lat / yaw:

| reading | stride 1 | stride 5 | stride 5 + Stage 3 (F265) |
|---|---|---|---|
| direct `body(proj(a))` | 0.40 / 0.61 / 0.59 | **0.83 / 0.91 / 0.84** | 0.83 / 0.91 / 0.85 |
| `body(ITM(e_a[t], e_a[t+k]))`, each candidate's own recorded frames | 0.62 / 0.58 / 0.74 | 0.77 / 0.85 / 0.81 | same |
| `body(ITM(e_s, real outcome))` from a shared start state = a perfect FTM | 0.12 / 0.17 / 0.13 | **0.24 / 0.52 / 0.24** | same |
| `body(ITM(e_s, FTM(e_s, proj(a))))` = rollout | 0.06 / 0.10 / -0.01 | 0.10 / 0.23 / 0.21 | 0.27 / 0.27 / 0.09 |

Variance of the stride-5 reading of the real outcomes over the 8 states x 24 actions grid: start-state
share 0.09-0.52, action share 0.16-0.57 per channel and step, the remainder (0.3-0.6) neither.

**Reading.**
- The ITM + body head read motion well when the start frame belongs to the same movement (0.77-0.85)
  and poorly when an action is applied from a state it did not come from -- the situation selection
  always creates. The ITM, like the FTM in F252, is substantially driven by the start frame.
- A perfect forward model would therefore lift rollout only to ~0.24 / 0.52 / 0.24, far below direct.
  This is why making the FTM a better predictor (Stage 3, F265) barely moves selection, and why the
  cycle loss's gain had to come from a pass-through (F258).
- To make rollout competitive the read-out must learn to read counterfactual transitions: e.g. train
  the ITM/read-out on (state s, action a) -> outcome pairs, which the sim can render.

Caveat: the rendered counterfactuals are kinematic -- at the first step the body switches to action
a's recorded pose and height -- a discontinuity absent from the ITM's training pairs, which may inflate
the start-state effect. The own-state row has no such switch.

Scope: stride 1 and 5 B1 checkpoints (out-of-sample Stages 2/4), 24 state-steps, one seed.

---

### F267. A read-out fitted on counterfactual transitions reads the rollout 2-3x better on unseen start states (r 0.48 / 0.63 / 0.72 vs 0.10 / 0.23 / 0.21) -- the ITM latent carries the action; the body head was not reading it

`counterfactual_readout_fit.py` (new). Stride-5 B1 checkpoint (F261, out-of-sample). F259's rendered
counterfactuals: 8 library start states x 3 steps x 24 actions. Ridge read-outs (64-d z -> Froude)
fitted leave-one-start-state-out; score = correlation across the 24 actions with their true Froude
over [t, t+5), mean over held-out state-steps, fwd / lat / yaw:

| read-out | on z = ITM(e_s, real outcome) | on z = ITM(e_s, FTM(e_s, proj(a))) (rollout) |
|---|---|---|
| checkpoint's body head (fit on matched real pairs) | 0.24 / 0.52 / 0.24 | 0.10 / 0.23 / 0.21 |
| ridge fit on real counterfactual z | **0.58 / 0.60 / 0.56** | 0.07 / 0.55 / 0.35 |
| ridge fit on rollout z | -- | **0.48 / 0.63 / 0.72** |
| direct `body(proj(a))`, for reference (F266) | 0.83 / 0.91 / 0.84 | |

**Reading.**
- The ITM's latent of a counterfactual transition carries much more of the action than the body head
  extracts: a head that has seen counterfactual pairs reads it at ~0.56-0.60 on unseen start states,
  against 0.24-0.52. F266's ceiling is a read-out-training problem, not missing information.
- A head fitted on the rollout's own latents does best on the rollout (0.48 / 0.63 / 0.72, 2-3x the
  current read-out). A head fitted on real outcomes transfers poorly to imagined ones (fwd 0.07), so
  the read-out should be trained on what the world model actually produces.
- Still below direct (0.83 / 0.91 / 0.84).

**Caveats.** The label of each counterfactual is the executed action's own recorded Froude -- exact
under kinematic execution, not under physics. The rollout-fitted head sees z_roll, which depends on
proj(a); part of what it reads may be the action passing through the FTM rather than predicted motion
(F258) -- not separated here. Small data: 576 samples, 8 states, one checkpoint.

Scope: selection with such a head not yet measured.

---

### F268. Plugging the counterfactual-trained read-out into rollout selection does not improve it (roll_live w11 +0.40 -> +0.33; w0 +0.13 -> +0.17)

The F267 ridge read-out, refit on all 24 counterfactual state-steps (rollout z, alpha 0.32), replaces
the body head in the rollout planner only (`selection_eval.py --rollout_readout`; direct unchanged).
Stride-5 checkpoint, out-of-sample Stages 2/4:

| read-out | roll_fixed w0 | roll_live w0 | roll_fixed w11 | roll_live w11 |
|---|---|---|---|---|
| body head | 0.127 (-0.01) | 0.114 (+0.13) | 0.084 (+0.45) | 0.089 (+0.40) |
| counterfactual ridge | 0.121 (+0.06) | 0.111 (+0.17) | 0.091 (+0.37) | 0.095 (+0.33) |

**Reading.** F267's gain in across-action correlation on held-out rendered states does not carry into
selection. Differences between the two settings that plausibly explain it (not separated):
- **Scene shift:** the counterfactual states were rendered in one room (seed 0); selection reads the
  library's recorded frames, each clip in its own randomised room.
- **Read-out shift under w=11:** the head was fit on one FTM step from a real frame; the windowed
  rollout reads consecutive imagined pairs whose start frames are themselves predictions.
- **Coverage:** 24 state-steps at t = 10/20/30 only, 576 samples, against selection at every second
  step over 0-62.
- Selection also needs calibrated absolute Froude, not only the right ranking across actions.

A read-out for counterfactual transitions needs training data drawn from the same scenes, times and
read-out mode that selection uses.

Raw: `results/deck/window_sweep/cf_readout_selection.txt`; head
`wm/runs/beh24_stride5_cleansplit/b1_lora_c3/readout_cf_rollout.npz`.

---

### F269. Adapting the stride-5 model to c08f09t09 with switching babble (one dataset for all stages) does not beat zero-shot: rollout unchanged, direct worse

**Data.** `data/egocentric/babble_c08f09t09` (48 clips x 66 frames, `scripts/dataset/collect_babble_hex.py`):
CPG drives re-drawn every 15-25 steps (forward/backward walk with pace and spin, or sideways walk),
ramped over 4 frames; each clip in its own randomised room (`--ego_seed` 1000+i). Froude coverage
(2-98 pct) vs beh24: forward -0.27..+0.20 (beh24 -0.21..+0.22), lateral -0.29..+0.24 (-0.19..+0.16),
yaw -0.10..+0.11 (-0.08..+0.07), spread across the range rather than at a few fixed speeds. On c08
`--spin` turns the opposite way to c10 (yaw ~ -0.1 x spin - 0.02). Review renders:
`results/deck/babble_review/`.

**Adaptation** (`scripts/run/c08_babble_adapt.sh`), `beh24_stride5_cleansplit/best.pt`, one dataset
for every stage: Stage 1 LoRA (38 clips, 10 held out; held-out rollout ratio 1.93-1.95 vs hold-still),
Stage 2 projector, Stage 3 joint (lambda_z 1); body head = the pretrain's (no Stage 4, F262).

Selection (goals: c10f10t10 held-out; candidates: c08's 48 beh12 clips; oracle 0.020, random 0.121):

| model | direct w0 | direct w11 | roll_fixed w11 | roll_live w11 |
|---|---|---|---|---|
| zero-shot (F264) | 0.066 (+0.55) | **0.059 (+0.61)** | 0.082 (+0.38) | **0.082 (+0.39)** |
| babble, Stages 1+2 | 0.086 (+0.35) | 0.081 (+0.40) | 0.078 (+0.43) | 0.083 (+0.37) |
| babble, Stages 1+2+3 | 0.086 (+0.34) | 0.077 (+0.43) | **0.074 (+0.46)** | 0.087 (+0.34) |

**Reading.**
- Rollout moves within noise (roll_live +0.39 -> +0.37 / +0.34; roll_fixed +0.38 -> +0.43 / +0.46).
- Direct drops (+0.61 -> +0.40). The babble-fitted projector maps actions to z less well for the
  library's steady gaits than the zero-shot projector fitted on the pretraining body's steady gaits
  (Stage 3 held-out proj-vs-ITM z error 4.3); with no Stage 4, the pretrain body head reads this z.
- 38 clips of switching behaviour at adaptation time do not remove the start-state shortcut learned
  in pretraining (F266) as far as selection shows; whether pretraining itself on switching data would
  is untested.

Scope: one seed, one babble set, six goals; the counterfactual read-out check (F266) was not repeated
on c08 (its renders are B1-only).

---

### F270. On held-out switching babble the ITM + body-head read-out barely reads motion at all, before or after babble adaptation; the switch-vs-steady question is therefore not answerable with this read-out

`switch_readout_check.py` (new): on the 10 babble clips not used in Stage 1 (F269), R2 of
`body(ITM(e_t, e_{t+5}))` against the true Froude of the same interval ("now") and of the preceding
one ("prev"), on switch samples (top 30% of |now - prev|, 164) and steady ones (383), fwd / lat / yaw:

| read-out | switch: vs now | switch: vs prev | steady: vs now |
|---|---|---|---|
| zero-shot (pretrain ITM + head) | -0.03 / -0.01 / +0.03 | -0.12 / -0.01 / -0.39 | +0.29 / -0.02 / -0.01 |
| babble Stages 1+2 (pretrain head) | -0.08 / -0.02 / +0.05 | -0.22 / +0.01 / -0.28 | +0.27 / -0.05 / -0.05 |
| + body head fit on the 38 Stage-1 babble clips | +0.29 / +0.01 / +0.21 | +0.20 / -0.00 / -0.13 | +0.15 / -0.22 / -0.20 |

(The head's own held-out ratio on those 38 clips: 0.91 vs predicting the mean.)

**Reading.**
- The pretrain read-out cannot read lateral or yaw on c08 babble at all, even on steady stretches;
  the babble-adapted ITM does not change that, since nothing in Stages 1-3 trains the read-out.
- A head fit on babble reads switch samples slightly closer to the new motion than to the old
  (fwd 0.29 vs 0.20, yaw 0.21 vs -0.13), but generalises poorly across clips (steady lat/yaw < 0).
- So F266's start-state shortcut cannot be confirmed or excluded on this data with this read-out.

**Method note.** `wm.fit_body_head` draws its own 20% clip split from `--data`; evaluating it on a
separately defined held-out set requires restricting `--data` to that set's complement. Without it,
the same comparison read +0.58 / +0.46 / +0.61 (switch vs now), entirely leakage.

Checkpoints: `wm/runs/beh24_stride5_cleansplit/c08_babble/{ckpt_s12,ckpt_s124_clean}.pt`.

---

### F271. The direct planner's centred read-out window is not what separates it from rollout: reading forward-only like the rollout changes direct by <= 0.002

The windowed direct read-out (F251) centres its w-step window on the decision, so it can use a
candidate's commands before t; the rollout can only read forward from its start frame. With
`--direct_window_align forward` (new; planner `window_align`, default `centre`) direct reads the
same forward window as the rollout. B1, w=11, out-of-sample Stages 2/4:

| model | direct, centred | direct, forward |
|---|---|---|
| stride 1 (beh24) | 0.063 (+0.66) | 0.062 (+0.68) |
| stride 5 | 0.048 (+0.82) | 0.049 (+0.81) |
| OLD (beh12, in-sample) | 0.053 (+0.77) | 0.055 (+0.75) |

The direct-vs-rollout gaps reported in F251-F263 are therefore not an artefact of the two planners
reading different windows.

Also from a code review of the stride and babble changes (no effect on any reported number):
`stride_of` now raises when `action_chunk` != `frame_stride` (a mismatched projector would otherwise be
fitted silently wrong); `collect_ik.py --plan` refuses the heading loop; the closed loop refuses
`--free_offset` with `--replan_every > 1` and non-B1 embodiments. Open, not needed by any run so far:
at stride k with decision horizon h > k the direct and rollout paths score different spans.

---

### F272. Measured directly, z is shared between the two hexapod morphologies but NOT with the B1, at stride 1 or stride 5 -- F262's "shared across bodies" reading was wrong

Pre-registered expectation (written before running): z separates by body at stride 1 and groups by
behaviour across bodies at stride 5. `scripts/figures/shared_latent_figure.py`: z = ITM(e_t, e_{t+k})
from each model's B1-adapted ITM (`ckpt_lib_s4.pt`), on c10 (beh24 val, 24 clips), c08 (beh12, 48),
B1 (beh12 library, 24); every 2nd step.

| model | body-ID accuracy (chance 0.33) | Froude read-out fit on c10 -> c08 R2 fwd/lat/yaw | -> B1 R2 fwd/lat/yaw | kNN mixing (1 = random) |
|---|---|---|---|---|
| stride 1 | 0.67 | +0.64 / -0.03 / -0.41 | -1.07 / -0.05 / -0.17 | 0.56 |
| stride 5 | 0.67 | +0.62 / +0.10 / -0.36 | -0.91 / -0.29 / -0.18 | 0.48 |

(body-ID: logistic regression, 5-fold grouped by clip, bodies subsampled to equal size; read-out:
ridge; kNN: share of 10 nearest neighbours from another body over its random expectation.) PCA and
UMAP (seeds 0, 1, 2, all shown; `results/deck/shared_latent/latent_stride{1,5}.png`) agree: the two
hexapods intermix, the B1 occupies its own region in every projection; within each body the latent is
ordered by behaviour (side / turn / forward regions).

**Reading.**
- z transfers across a leg-length change of the same embodiment (c10 -> c08 forward R2 ~0.6) and does
  not transfer to the quadruped: a hexapod-fit read-out predicts B1 forward speed worse than the mean.
  Stride 5 does not change this.
- **Correction to F262's reading.** F262 showed that at stride 5 the pretrain body head, applied to
  the B1 projector's latents, ranks B1 candidates well enough to select (+0.59). It was read as "the
  stride-5 latent is shared across bodies"; this direct test says it is not. What stride 5 improved
  there is selection through proj(a), not the location of B1's ITM latents. Why the unadapted head
  ranks B1 candidates at stride 5 but not at stride 1 is not explained.
- Cross-embodiment selection (F261, F263) works through per-body fitting (projector, and at best the
  body head), not through a shared ITM latent.

Scope: one checkpoint per stride, recorded clips, a linear and a kNN probe.

---

### F273. Seed-to-seed spread of the stride-5 pipeline is 0.1-0.45 gap on B1; the A/B/C differences (switching vs steady babble) are inside it

Four stride-5 pretrains, identical except data or seed (`scripts/run/switch_pretrain_arms.sh`):
A = beh24 (F261), A2 = beh24 seed 1, B = beh24 + 48 c10 switching-babble clips, C = beh24 + 48 c10
steady-babble clips (same drive distribution, one drive per clip; `collect_babble_hex.py --steady`).
Same B1 pipeline (Stages 1, 2, 4 out-of-sample) and c08 zero-shot. Selection, w=11 (gap):

| model | B1 direct | B1 roll_fixed | B1 roll_live | c08 direct | c08 roll_fixed | c08 roll_live |
|---|---|---|---|---|---|---|
| A | +0.82 | +0.45 | +0.40 | +0.61 | +0.38 | +0.39 |
| **A2 (seed only)** | **+0.59** | **+0.00** | +0.21 | +0.55 | +0.26 | +0.30 |
| B (switching) | +0.78 | +0.56 | +0.30 | +0.67 | +0.43 | +0.35 |
| C (steady) | +0.71 | +0.53 | +0.32 | +0.67 | +0.39 | +0.43 |

Mechanism on B1 (r across actions vs true Froude, fwd/lat/yaw; `readout_bottleneck_check.py`):
reading the rollout's imagined future A 0.10/0.23/0.21, B 0.07/0.43/0.54, C 0.11/0.46/0.34; reading the
real counterfactual future A 0.24/0.52/0.24, B 0.23/0.49/0.54, C 0.36/0.58/0.64.

**Reading.**
- A seed change alone moves B1 selection by 0.19-0.45 gap; every A/B/C difference is within that.
  No selection effect of switching or of extra babble is established.
- Both babble arms read lateral/yaw counterfactuals better than A; C matches or exceeds B, so what
  helps the read-out is broader behaviour coverage, not switching (single seeds, same caveat).
- F261's stride-5 numbers came from seed A, which is at the favourable end; the stride-1 vs stride-5
  comparison rests on one seed each and needs replication before it is claimed.

---

### F274. Physics closed loop: rollout does not beat direct on either body, for any of the four models; direct beats random clearly, rollout on the B1 is at random level

The week's decisive test (goal 2026-09-26): with outcomes that depend on state, does the state-aware
selector (rollout) beat the state-blind one (direct)? `scripts/run/physics_loops.sh`,
`physics_random.sh`; six hexapod goals; w=11, decide every 2 steps; graded by the simulated body's
actual body_motion. B1: the chosen candidate's command executed by the B1's own policy in MuJoCo
(`close_loop_b1_physics_froude.py`), CoppeliaSim ego render (view check 0.995). c08: candidate joint
commands on the CoppeliaSim-physics CPG body (`close_loop_hexapod_froude.py`), plain or phase-matched
switching. No falls in any run. Mean L2 error over the six goals:

| body | model | direct | rollout | random |
|---|---|---|---|---|
| B1 | A / A2 / B / C | 0.070 / 0.077 / 0.075 / 0.078 | 0.083 / 0.103 / 0.090 / 0.086 | 0.089 |
| c08 plain | A / A2 / B / C | 0.072 / 0.078 / 0.059 / 0.071 | 0.077 / 0.104 / 0.069 / 0.080 | 0.126 |
| c08 phase-matched | A / A2 / B / C | 0.065 / 0.073 / 0.060 / 0.065 | 0.100 / 0.104 / 0.093 / 0.087 | 0.111 |

Per-condition values: `results/wm/closed_loop/physics/summary.txt`; figures
`results/deck/physics_closed_loop/`.

**Reading.**
- Direct < rollout in all 12 model x body x execution cells; the hypothesis that state-awareness lets
  rollout win once outcomes depend on state is not supported by these runs.
- Consistent with F266: the read-out cannot read counterfactual transitions, so the state rollout
  conditions on does not reach the score.
- The state dependence of these tasks is modest: the B1's policy absorbs behaviour switches (no
  falls, direct only ~0.02 worse than in the kinematic loop), and phase matching changes c08 direct by
  <= 0.013. A task where a state-blind choice fails more severely has not been built.
- Phase matching helps direct slightly and makes rollout worse in every model; not explained.
- What is established: selection in the world model's latent space (direct) is clearly better than
  random under real physics on both bodies (B1 0.070-0.078 vs 0.089; c08 0.059-0.078 vs 0.111-0.126).

---

### F275. A single egocentric frame identifies hexapod vs B1 almost perfectly (0.998) but not the two hexapod leg lengths (0.453): any head that reads the frame can solve a per-body target without a shared z

Question: F57 found that a Froude head reading frame + z (LAC-WM's motion-decoder form) destroyed
cross-body transfer (insect->b1 R2 -10.48, b1->insect -57.17, vs +0.544 / +0.435 for a z-only head),
attributed to the frame revealing the robot. F57 used allocentric frames; does the egocentric view
used since then still reveal the body?

Probe: frozen V-JEPA2 embeddings of single egocentric frames (every 4th frame, 24 clips per body,
mean over the 256 tokens), standardised, PCA 64, logistic regression, 5-fold CV grouped by clip.
Caches: `anova_hex_beh24val.pt` (c10), `selection_eval_c08.pt` (c08), `selection_eval_cands.pt` (B1).

| task | accuracy | chance |
|---|---|---|
| c10 vs B1 | 0.998 | 0.5 |
| c10 vs c08 | 0.453 | 0.5 |
| 3 bodies | 0.651 | 0.33 |

**Reading.**
- Egocentric frames separate the hexapod from the B1 as completely as allocentric ones could; the
  frame-reveals-the-body mechanism behind F57 applies to the current setup. Any module that sees the
  frame (motion decoder, frame-reading body head, FTM) can fit a per-body mapping without z being shared.
- The two hexapods are indistinguishable from one frame, consistent with F272 (z shared c10 <-> c08).
- Confound not separated: B1 clips are rendered by a different pipeline (camera height, room
  generation), so the probe shows the frame identifies the body, not whether it does so from the body
  itself or from rendering differences.
- Same pattern as arXiv 2609.19846's wrist-camera ablation (body-specific views: transfer 61.3% -> 46.0%).

Consequence for design: a cross-body grounding must act on z alone (z-only shared head, or a
similarity loss on z), never through the frame. Notes: `doc/ref/notes_prior_work_cross_body.md`.

**Addendum 2026-09-30: the separation is a rendering difference, not the body.** Same clips, same
grouped CV:
- **Simple image features** barely separate c10 from B1:
  - floor/wall edge row: 0.51 (median row 139 / 141, identical);
  - mean RGB: 0.63;
  - row-mean luminance profile: 0.76.
- **The V-JEPA2 embedding separates them from any image region alone:**
  - top rows, ceiling / upper wall: 0.990;
  - wall: 0.984;
  - floor/wall edge: 1.000;
  - floor: 0.995.

  So the cue is spread over the whole frame, not an object in view.
- **Image statistics differ by render pipeline:**
  - Laplacian variance (sharpness): B1 256 vs c10 178 / c08 163;
  - high-frequency energy: 55.1 vs 49.9 / 48.6;
  - ceiling texture std: 2.68 vs 3.71 / 3.43.

  B1 clips are rendered by `render_b1_replay.py` (MuJoCo replay, room scaled by camera-mount height),
  hexapod clips by `collect_ik.py`.

The egocentric frames of the two pipelines therefore differ in sharpness / texture statistics, which
V-JEPA2 detects everywhere in the image. This is a confound in the data, not evidence that a frame
reveals the body. It can also carry into z: part of what separates the B1's z from the hexapod's may be
render style. Before further cross-body claims, render both bodies through matched settings and require
this probe to be near chance.

**Sources found (2026-09-30).** Same sensor on both scenes (256x256, near 0.01, render mode 0), same
capture code (no resize), same room code, same four scene lights.
- **Lights: not the cause.** Scaling them with the room (x2.21 on the B1) changes no pixel (max
  difference 0), so the scene lights do not affect these renders.
- **Source 1: the B1 ego datasets predate the current room-texture recipe.** Re-rendering B1 clip
  `b1_ep0` from its stored MuJoCo states with the current `render_b1_replay.py --ego` differs from the
  dataset frame by up to 136 grey levels, and the ceiling texture std moves 2.58 -> 3.59, i.e. onto the
  hexapod's 3.43-3.71.
- **Source 2: remaining detail difference in the current renders.** Laplacian variance by region,
  B1 re-render vs c10:

  | region | B1 | c10 |
  |---|---|---|
  | ceiling | 59.4 | 53.9 |
  | wall | 8.3 | 5.4 |
  | floor | 129.4 | 95.7 |

  The floor and walls carry finer detail on the B1, plausibly the height-scaled ground UV / wall
  texture density.

Fix path: re-render the B1 ego datasets from their stored states with the current code, match the
floor / wall texture density to the hexapod's by these statistics, then require the frame-level
body-identity probe to be near chance before re-measuring B1 sharing.

**Re-render done (`scripts/dataset/rerender_b1_ego_matched.py` -> `data/egocentric_v2/`).**
- **Floor.** The B1 floor was stretched by a fixed x3 (`render_b1_replay.py --floor_scale`); the
  hexapod's grows with the room (`ego_camera.scale_floor`, x4.59 for the B1's 17.6 m room). With
  `--match_floor`, the dashed line at the wall base disappears.
- **Floor texture density.** Matched by `--ground_uv_mult 1.5`, chosen against c10 on Laplacian
  variance:

  | | whole frame | floor | wall base |
  |---|---|---|---|
  | c10 | 179 | 84 | 791 |
  | B1, old renders | 294 | 129 | 1522 |
  | B1, re-rendered | 187 | 93 | 918 |

- **Determinism.** Kinematic replay: rendering a clip twice gives identical frames (max difference 0).
- **Metadata.** Only `frames` is replaced; the other fields are copied from each original clip.

Body-identity probe, c10 vs B1 (`scripts/diagnostics/egocentric_view/body_id_render_check.py`, V-JEPA2 on
the CPU, float32):

| B1 frames | whole frame | ceiling rows | wall | floor/wall edge | floor |
|---|---|---|---|---|---|
| old renders | 0.998 | 0.990 | 0.984 | 1.000 | 0.995 |
| re-rendered | 0.904 | 0.877 | 0.932 | 0.907 | 0.943 |

**What remains is largely body-caused.** The horizon (floor/wall edge) fitted per frame:

| body | tilt, median abs | tilt, p90 | frame-to-frame tilt change | horizon row std |
|---|---|---|---|---|
| c10 | 4.33 deg | 9.45 deg | 3.40 deg | 5.6 |
| c08 | 5.07 deg | 9.71 deg | 3.88 deg | 7.2 |
| B1 (re-rendered) | 1.47 deg | 6.30 deg | 1.37 deg | 4.0 |

- The hexapod camera rolls and sways with the tripod gait; the B1's stays level.
- These two numbers alone (tilt, horizon row) separate c10 from B1 at 0.701.

The re-rendered frames still identify the body mostly through how the body carries the camera. That is
information the body genuinely causes, not rendering, and it is kept. Whether any rendering difference
remains beyond it is not separated.

**Correction (same day): the x1.5 floor setting was itself a rendering difference; pure scaling matches.**
`scripts/diagnostics/egocentric_view/static_scene_render_check.py` isolates the scene from the body. It
renders both scenes from the SAME camera poses:
- 16 rooms x 12 poses, positions and headings in room-scaled units;
- camera level at mount height x 1.551, 90 deg FOV;
- no body in view, no walking;
- each scene built as its collector builds it, same room seeds.

V-JEPA2 frame probe, hexapod scene vs B1 scene:

| B1 floor texture | whole frame | ceiling rows | wall | floor/wall edge | floor |
|---|---|---|---|---|---|
| `--ground_uv_mult 1.5` (v2) | 0.952 | 0.895 | 0.904 | 0.925 | 0.937 |
| `--ground_uv_mult 1.0`, pure scaling | **0.552** | 0.477 | 0.680 | 0.627 | 0.581 |

- **Pure scaling renders the two scenes alike.** Room, floor tiles (`--match_floor`) and floor texture
  all scale with mount height.
- **The x1.5 was tuned against walking data,** whose image-detail differences come from camera motion,
  and so introduced a scene difference. The v2 re-render is superseded by v3
  (`rerender_b1_ego_matched.py`, x1.0, `data/egocentric_v3/`).
- **The earlier explanation of v2's residual 0.904 as gait sway was wrong.** The static test shows the
  scene itself separated at 0.952.
- **Method note.** Region-restricted probes on V-JEPA2 tokens do not localise a cue: every token attends
  to the whole frame, which is why every region separated equally.

---

### F276. Pretraining on babble only: switching babble (drive held 15-25 frames) improves the read-out of the rollout on both seeds but not physics selection; rapid babble (drive every 5 frames) fails zero-shot c08 on both seeds

`scripts/run/babble_only_arms.sh`. Stride-5 pretrains on 48 c10 clips each, differing only in the
data, 2 seeds per arm, identical evaluation to F273/F274:
- beh24: one behaviour per clip. Runs A, A2.
- switch: babble only, new drive every 15-25 frames, ramp 4. Runs SW0, SW1.
- rapid: babble only, new drive every 5 frames (= one model step), ramp 2. Runs RP0, RP1.
  The rapid pilot showed the body's motion lagging the drive by about one segment (5-10 frames), and a
  2-98% forward Froude range of -0.14..+0.13 against beh24's -0.21..+0.22.

Kinematic selection, NS at w=11 (B1 oracle 0.031 / random 0.126; c08 0.020 / 0.121):

| run | B1 direct | B1 rollout (current start) | c08 direct | c08 rollout (current start) | read of rollout r fwd/lat/yaw | top-1 retrieval |
|---|---|---|---|---|---|---|
| A | +0.82 | +0.40 | +0.61 | +0.39 | 0.10 / 0.23 / 0.21 | 0.233 |
| A2 | +0.59 | +0.21 | +0.55 | +0.30 | 0.05 / 0.34 / 0.04 | 0.184 |
| SW0 | +0.76 | +0.33 | +0.54 | +0.45 | 0.13 / 0.48 / 0.68 | 0.262 |
| SW1 | +0.72 | +0.36 | +0.64 | +0.26 | 0.19 / 0.47 / 0.63 | 0.262 |
| RP0 | +0.63 | +0.11 | -0.19 | +0.11 | -0.10 / 0.37 / 0.11 | 0.222 |
| RP1 | +0.65 | +0.20 | -0.21 | -0.13 | -0.01 / 0.37 / -0.05 | 0.205 |

(read of rollout = r across 24 actions of body(ITM(e_s, FTM(e_s, proj(a)))) vs true Froude, B1;
top-1 retrieval = cosine match of the action-specific predicted change to the real one, chance 0.042.)

eta^2 state / eta^2 action on the pretrained model, hexapod val (fwd/lat/yaw): A2 0.63/0.41/0.43 vs
0.23/0.33/0.20; SW0 0.37/0.32/0.22 vs 0.31/0.49/0.47; SW1 0.51/0.41/0.32 vs 0.22/0.43/0.29; RP0
0.55/0.36/0.38 vs 0.29/0.41/0.44; RP1 0.44/0.29/0.23 vs 0.37/0.50/0.57. Measured on beh24 val, which
the babble runs never trained on.

Physics closed loop (F274 protocol), mean L2 error over six goals; random 0.089 (B1), 0.126 (c08 plain),
0.111 (c08 phase-matched):

| run | B1 direct | B1 rollout | c08 plain direct | c08 plain rollout | c08 phase direct | c08 phase rollout |
|---|---|---|---|---|---|---|
| A (F274) | 0.070 | 0.083 | 0.072 | 0.077 | 0.065 | 0.100 |
| A2 (F274) | 0.077 | 0.103 | 0.078 | 0.104 | 0.073 | 0.104 |
| SW0 | 0.077 | 0.088 | 0.082 | 0.082 | 0.084 | 0.091 |
| SW1 | 0.088 | 0.090 | 0.083 | 0.097 | 0.079 | 0.102 |
| RP0 | 0.078 | 0.096 | 0.139 | 0.154 | 0.143 | 0.129 |
| RP1 | 0.058 | 0.115 | 0.136 | 0.147 | 0.148 | 0.133 |

No B1 run fell. c08 falls are not detected by the loop (not checked here).

**Reading.**
- Switching babble, both seeds: the rollout's read-out of its own prediction (lat 0.47-0.48, yaw
  0.63-0.68) and top-1 retrieval (0.262 both) exceed both beh24 seeds. Kinematic direct stays in the
  beh24 range. The physics loop does not follow: B1 rollout 0.088-0.090 is at the random level (0.089),
  c08 errors sit at or slightly above beh24's. The mechanism gain is not reaching closed-loop selection.
- Rapid babble, both seeds: zero-shot c08 fails, kinematically (NS -0.19, -0.21) and in physics (every
  c08 cell 0.129-0.154, worse than random 0.111-0.126). Consistent with the pilot's drive-to-motion lag:
  with a new drive every model step, the action chunk and the motion that follows it are misaligned.
  RP1's B1 direct (0.058) is the lowest B1 direct error measured, on one seed, next to RP0's 0.078; not
  interpreted.
- Direct < rollout in 21 of 24 model x body x execution cells here and in F274 combined. Exceptions:
  SW0 c08 plain (0.0819 vs 0.0815, a tie) and RP0 / RP1 c08 phase-matched, where both mechanisms are
  worse than random. None is a rollout win over a direct that beats random.
- Babble drive duration matters: held 15-25 frames is usable, every 5 frames is not, for this body's
  response time. The detach experiment's babble cell uses switching babble for this reason.

Scope: one c10 body pretrained; B1 via the four-stage pipeline; c08 zero-shot; 2 seeds per arm, 6 goals.

---

### F277. Letting the Froude head shape z (detach off) improves selection on every body, both seeds, direct and rollout; and a same-body test shows rollout < direct with no body change at all

**Same-body test (new).** `scripts/run/c10_samebody_eval.sh`: goals = the six c10f10t10 held-out goal
clips, candidates = c10f10t10's own beh24 val library (24 clips, never trained on; the hexapod projector
from `c08_zeroshot/` was fit on beh24 cleantrain). No change of body or leg length. Library bounds:
oracle 0.0135, random 0.1535. Caveat: the library is fixed-behaviour clips, in-distribution for beh24
models and not for babble-only models (F276).

**Detach off.** `dt0_beh24_s0/_s1`: identical to A / A2 (beh24 cleantrain, stride 5, same losses, 50
epochs, seeds 0 / 1) except `--detach_body_z False`, so the shared z-only Froude head's gradient reaches
the ITM (as before 2026-09-08). Trained on the lab server, evaluated here through
`scripts/run/detach_eval_local.sh` (identical to every other arm's evaluation).

NS at w=11:

| test | A (detach on) | A2 (detach on) | dt0 s0 (off) | dt0 s1 (off) | SW0 | SW1 | RP0 | RP1 |
|---|---|---|---|---|---|---|---|---|
| c10 same body, direct | +0.80 | +0.73 | +0.91 | +0.90 | +0.60 | +0.68 | +0.01 | -0.09 |
| c10, rollout (library start) | +0.60 | +0.42 | +0.72 | +0.73 | +0.42 | +0.31 | +0.16 | +0.07 |
| c10, rollout (current start) | +0.49 | +0.52 | +0.64 | +0.59 | +0.42 | +0.14 | +0.14 | +0.01 |
| c08 zero-shot, direct | +0.61 | +0.55 | +0.80 | +0.79 | | | | |
| c08, rollout (library start) | +0.38 | +0.26 | +0.47 | +0.54 | | | | |
| c08, rollout (current start) | +0.39 | +0.30 | +0.52 | +0.43 | | | | |
| B1, direct | +0.82 | +0.59 | +0.82 | +0.82 | | | | |
| B1, rollout (library start) | +0.45 | +0.00 | +0.62 | +0.44 | | | | |
| B1, rollout (current start) | +0.40 | +0.21 | +0.46 | +0.28 | | | | |

(c08 / B1 values for the babble runs: F276.) Mechanism, B1: read of the rollout's prediction r
fwd/lat/yaw dt0 0.27/0.28/0.62 and 0.24/0.32/0.41 (A 0.10/0.23/0.21, A2 0.05/0.34/0.04); read of the
real counterfactual 0.20/0.42/0.59 and 0.22/0.45/0.48; top-1 retrieval 0.247 / 0.273 (A 0.233, A2 0.184).
eta^2 on the pretrained model, hexapod val, state / action: dt0 s0 0.61/0.40/0.43 vs 0.19/0.25/0.30,
s1 0.58/0.51/0.53 vs 0.20/0.17/0.21 (A2 0.63/0.41/0.43 vs 0.23/0.33/0.20).

**Reading.**
- Same body, every model: rollout < direct. The rollout's deficit exists without any change of body,
  so it is not caused by cross-embodiment transfer.
- Detach off beats detach on in 32 of 36 seed-pair comparisons above (each dt0 seed vs each of A, A2,
  9 rows): 2 ties (B1 direct, dt0 vs A, +0.82 each) and 2 losses (dt0 s1 vs A on B1 rollout library
  start, +0.44 vs +0.45, and current start, +0.28 vs +0.40). The largest gains are on the bodies whose z is shared with the pretraining body
  (c08 direct +0.55-0.61 -> +0.79-0.80; c10 rollout library start +0.42-0.60 -> +0.72-0.73), beyond the
  seed spread measured in F273.
- The gain is not an increase in the FTM's action share: eta^2 action is equal or slightly lower. It
  comes with a z whose Froude read-out is better (read of prediction and top-1 up), i.e. z is more
  organised by body motion.
- Consequence: the hard-coded detach (2026-09-08, see F233 correction) cost selection quality on every
  body tested. `detach_body_z False` is the better setting for single-body pretraining; the cross-body
  question (does it make the B1 share z) needs joint pretraining.
- On the same body, babble-only pretraining is below beh24 (library in-distribution for beh24 only).

Physics closed loop (`scripts/run/detach_physics.sh`, F274 protocol, six goals, mean L2 error; random
B1 0.089, c08 plain 0.126, c08 phase-matched 0.111):

| run | B1 direct | B1 rollout | c08 plain direct | c08 plain rollout | c08 phase direct | c08 phase rollout |
|---|---|---|---|---|---|---|
| A (detach on) | 0.070 | 0.083 | 0.072 | 0.077 | 0.065 | 0.100 |
| A2 (detach on) | 0.077 | 0.103 | 0.078 | 0.104 | 0.073 | 0.104 |
| dt0 s0 (off) | 0.072 | 0.073 | 0.053 | 0.071 | 0.057 | 0.077 |
| dt0 s1 (off) | 0.073 | 0.098 | 0.050 | 0.078 | 0.059 | 0.089 |

- The kinematic gain holds under physics on c08: direct 0.050-0.053 (detach on 0.072-0.078), the lowest
  c08 error measured; rollout 0.071-0.078 (0.077-0.104); phase-matched direct 0.057-0.059 (0.065-0.073).
- B1 direct unchanged (0.072-0.073 vs 0.070-0.077). B1 rollout: dt0 s0 0.073, equal to its direct
  (0.072) -- the first model whose rollout matches direct on the B1 in physics; s1 0.098.
- Direct <= rollout still holds in every cell.

---

### F278. Existing joint hexapod+B1 pretrains (egocentric, stride 1, Froude head NOT detached) do not share z with the B1 under the strict test either

Search of `wm/runs/` (incl. archives) for runs pretrained on both bodies: all date from before the
hard-coded detach (2026-09-08), so all trained with the Froude head's gradient reaching z:
`beh12_hex-b1_body3` (08-30, allocentric), `beh12_ego` (09-02), `beh12_state` (09-04), `beh12_state_more`
(09-05), all stride 1 (`beh12_lag3_*` stride 3, allocentric). None is clean-split; none has hinge or
read-out loss (`beh12_ego`: lambda_hinge 0, lambda_readout 0).

`scripts/figures/shared_latent_figure.py --model ...` (F272 protocol: pretrained ITM, recorded clips of
c10 / c08 / B1, ridge Froude read-out fit on c10 only), `results/deck/shared_latent_joint_check/`:

| model | body-ID acc (chance 0.33) | c10 -> c08 R2 fwd/lat/yaw | c10 -> B1 R2 fwd/lat/yaw | kNN mixing |
|---|---|---|---|---|
| joint beh12_ego (stride 1, not detached) | 0.73 | +0.45 / +0.09 / -0.33 | -0.91 / +0.07 / -0.18 | 0.39 |
| joint beh12_state (stride 1, not detached) | 0.74 | +0.38 / +0.00 / -0.22 | -0.89 / +0.08 / -0.90 | 0.39 |
| single A (stride 5, detached) | 0.67 | +0.63 / +0.09 / -0.35 | -1.01 / -0.34 / -0.09 | 0.48 |
| single dt0 s0 (stride 5, not detached) | 0.69 | +0.78 / +0.36 / +0.18 | -1.30 / -0.02 / +0.26 | 0.44 |

**Reading.**
- Pretraining both bodies together with an undetached shared Froude head did not, in these egocentric
  stride-1 runs, place the B1's z where a c10-fit read-out can read it (forward R2 -0.9), and body
  identity is at least as decodable as in single-body models (0.73-0.74 vs 0.67-0.69).
- F57's cross-body R2 (+0.544 / +0.435) was allocentric, stride 1, scored with the co-trained head
  (fit on both bodies), which F232 already flagged as not a strict transfer test; the strict
  fit-on-one-body test has now been applied to joint egocentric runs and fails.
- Detach off in single-body stride 5 (dt0) organises z better by motion within the hexapod family
  (c10 -> c08 R2 +0.78 / +0.36 / +0.18, the best measured), consistent with F277.
- Not separated: stride 1 (action barely visible per step, F259), beh12 data, no hinge / read-out
  terms, not clean-split, R2's sensitivity to a per-body offset in the read-out.
- Consequence: joint pretraining plus a regression Froude head is not by itself sufficient evidence
  for a shared z; the grounding may need a relational constraint on z across bodies (e.g. the
  similarity loss of arXiv 2609.19846) and stride 5.

**Addendum: cross-body retrieval** (added to `shared_latent_figure.py`, offset-insensitive unlike R2):
for each c10 point, its nearest neighbour in standardised z among body b's points; score = 1 -
L2(Froude, neighbour's Froude) / mean L2(Froude, random point of b); 1 = neighbour moves identically,
0 = no better than random.

| model | retrieval c10 -> c08 | retrieval c10 -> B1 |
|---|---|---|
| joint beh12_ego (stride 1, not detached) | 0.32 | 0.21 |
| single A (stride 5, detached) | 0.36 | 0.04 |
| single dt0 s0 (stride 5, not detached) | 0.44 | 0.05 |

Joint pretraining does give partial alignment with the B1 (0.21 vs ~0.05 without the B1 in
pretraining), which the R2 test (offset-sensitive) did not show. It is well below within-hexapod
alignment (0.32-0.44). This is the baseline the similarity loss has to improve on.

**Implementation, 2026-09-29:** `Config.lambda_sim` (+ `sim_queue`, `sim_kind`, `sim_sigma`,
`sim_cross_only`), `wm/train.py:froude_similarity_loss`. Batches are single-embodiment
(`EmbodimentBatchSampler`), so cross-body pairs come from a FIFO queue of recent detached (z,
standardised Froude) of every body. Body targets are standardised with pooled statistics across bodies
(`MultiEmbodimentPairs.body_stats`), so Froude similarity is comparable across bodies. Checks: CPU toy
(two bodies with offset z: nearest-other-body Froude cosine 0.90 after optimisation, from ~0); GPU
smoke run, joint hexapod + B1, 2 epochs: sim 0.646 -> 0.368, ~1000 cross-body pairs per step,
`lambda_sim` recorded in config.yaml.

---

### F279. Froude as the only motion target (joint-command motion decoder off) matches or slightly exceeds FS on every body; the per-body joint decoder is not needed

`scripts/run/froude_md_server.sh`: `fmd_beh24_s0/_s1`, identical to FS (`dt0_beh24`, F277: beh24, stride 5,
Froude head not detached, hinge / read-out / rollout) except `--lambda_motion 0`, so the per-body
joint-command motion decoder gives no gradient and the shared z-only Froude head is the only motion
target. Trained on the lab server; evaluated with `detach_eval_local.sh` and the c10 same-body test.
Note: best.pt selection uses recon + lambda_motion * motion, i.e. recon only here.

NS at w=11:

| test | FS s0 | FS s1 | D s0 | D s1 |
|---|---|---|---|---|
| c10 same body, direct | +0.91 | +0.90 | +0.91 | +0.91 |
| c10, rollout (library start) | +0.72 | +0.73 | +0.74 | +0.76 |
| c10, rollout (current start) | +0.64 | +0.59 | +0.61 | +0.66 |
| c08, direct | +0.80 | +0.79 | +0.78 | +0.79 |
| c08, rollout (library start) | +0.47 | +0.54 | +0.56 | +0.63 |
| c08, rollout (current start) | +0.52 | +0.43 | +0.53 | +0.54 |
| B1, direct | +0.82 | +0.82 | +0.85 | +0.85 |
| B1, rollout (library start) | +0.62 | +0.44 | +0.47 | +0.59 |
| B1, rollout (current start) | +0.46 | +0.28 | +0.46 | +0.35 |

Mechanism, B1: top-1 retrieval D 0.271 / 0.266 (FS 0.247 / 0.273); read of the rollout's prediction r
D 0.34/0.02/0.57 and 0.12/0.24/0.61 (FS 0.27/0.28/0.62 and 0.24/0.32/0.41). eta^2 on the pretrained
model, state / action: D s0 0.53/0.37/0.44 vs 0.24/0.25/0.28; s1 0.55/0.48/0.54 vs 0.23/0.23/0.26.

**Reading.** Dropping the per-body joint-command decoder costs nothing measured and is equal or slightly
better in most cells (c08 rollout library start +0.47-0.54 -> +0.56-0.63; B1 direct +0.82 -> +0.85),
within seed spread. The shared Froude head alone is a sufficient motion target for pretraining; the
per-body joint level can live in the projector only.

Physics closed loop: the first attempt ran while the joint pretraining (`joint_sim_beh24_s0`) held the
GPU; loading the V-JEPA2 encoder for rollout ran out of memory, and every later run returned nothing
(3 valid cells, one goal: D s0 c08 direct 0.028, c08 phase-matched 0.067, B1 direct 0.068). The next
re-run returned nothing because CoppeliaSim was left mid-simulation ("sim.loadScene: simulation is not
stopped"); after `sim.stopSimulation()` the re-run completed (72 / 72).

Physics closed loop, mean L2 error over six goals (random B1 0.089, c08 plain 0.126, c08 phase 0.111):

| run | B1 direct | B1 rollout | c08 plain direct | c08 plain rollout | c08 phase direct | c08 phase rollout |
|---|---|---|---|---|---|---|
| FS s0 / s1 | 0.072 / 0.073 | 0.073 / 0.098 | 0.053 / 0.050 | 0.071 / 0.078 | 0.057 / 0.059 | 0.077 / 0.089 |
| D s0 / s1 | 0.061 / 0.063 | 0.081 / 0.108 | 0.047 / 0.045 | 0.085 / 0.055 | 0.062 / 0.057 | 0.076 / 0.075 |

D's direct is the lowest measured on both bodies (B1 0.061-0.063, c08 0.045-0.047); rollout is mixed
across seeds (c08 plain 0.055-0.085, B1 0.081-0.108).

---

### F280. Joint hexapod + B1 pretraining with the Froude-similarity loss: the B1's video latents move toward the hexapod's (retrieval 0.05 -> 0.34) and B1 selection is the best measured

`scripts/run/joint_sim_local.sh`: `joint_sim_beh24_s0`, beh24 clean-train for both bodies (48 + 48 clips,
2673 / 2688 pairs, balanced batches, one body per batch), stride 5, Froude head shapes z, hinge /
read-out / rollout as FS, joint-command motion decoder on, `--lambda_sim 0.05` (queue 256, both bodies'
recent z). One seed. **No matched control** (same run with `lambda_sim 0`) yet: the comparison rows
differ in more than the similarity loss.

Shared latent (pretrained ITM; `results/deck/shared_latent_joint_sim/`):

| model | body-ID probe | cross-body R2 c10 -> c08 | cross-body R2 c10 -> B1 | k-NN mixing | retrieval c10 -> c08 | retrieval c10 -> B1 |
|---|---|---|---|---|---|---|
| joint + similarity (stride 5) | 0.73 | +0.72 / +0.36 / +0.17 | +0.13 / +0.30 / -0.34 | 0.37 | 0.43 | 0.34 |
| joint, beh12 (stride 1, no similarity, older losses) | 0.73 | +0.45 / +0.09 / -0.33 | -0.91 / +0.07 / -0.18 | 0.39 | 0.32 | 0.21 |
| single hexapod, FS | 0.69 | +0.78 / +0.36 / +0.18 | -1.30 / -0.02 / +0.26 | 0.44 | 0.44 | 0.05 |

Selection (standard evaluation: B1 through Stages 1, 2, 4 on top of the joint pretrain; NS w=11):
B1 direct +0.95, rollout library start +0.71, current start +0.54 (FS: +0.82, +0.44-0.62, +0.28-0.46);
c08 direct +0.79, rollouts +0.54 / +0.45; c10 same body direct +0.91, rollouts +0.76 / +0.53.
Mechanism, B1: top-1 retrieval 0.370 (FS 0.247-0.273), read of the rollout's prediction r
0.38 / 0.47 / 0.73. The in-training body probe (MorphProbe on z) stays at 0.999 throughout.

**Reading.** First model with a positive cross-body R2 to the B1 on two channels and B1 retrieval near
the within-hexapod level (0.34 vs 0.43); B1 selection and mechanism numbers are the best measured;
hexapod-side numbers unchanged. z still encodes body identity (probe), alongside the shared motion
structure. Which ingredient does it (similarity loss vs joint stride-5 pretraining itself) is not
separated.

---

### F281. Anchoring Stage 1 adaptation (frozen pretrained Froude head, or similarity to a fixed bank of hexapod z) moves the B1's latents toward the hexapod's only a little at 3 clips / 1000 LoRA steps

`scripts/run/anchor_stage1.sh`, `wm/adapt.py --anchor_froude / --anchor_sim` (new; the new body's
standardised Froude, checkpoint body statistics; the similarity bank = 12 hexapod beh24 clips encoded by
the unadapted ITM, held fixed). Base `fmd_beh24_s0`; Stage 1 as the standard recipe (LoRA rank 2 on the
Mlp layers only, 3 stratified B1 clips, hinge 0.5, 1000 steps); measured on the adapted ITM.

| Stage 1 anchor | retrieval c10 -> B1 | cross-body R2 c10 -> B1 | Stage 1 rollout ratio h=1 / h=10 (after) |
|---|---|---|---|
| none | 0.03 | -1.36 / +0.09 / +0.30 | 1.67 / 1.67 |
| Froude head | 0.12 | -0.44 / +0.12 / +0.42 | 1.74 / 1.69 |
| similarity | 0.11 | -0.45 / -0.01 / +0.33 | 1.74 / 1.70 |
| both | 0.11 | -0.48 / +0.06 / +0.38 | 1.74 / 1.70 |

c10 -> c08 retrieval unchanged (0.42-0.43). Each anchor moves the B1 toward the hexapod (0.03 -> 0.11-0.12,
forward R2 -1.36 -> -0.45) at a small cost in B1 prediction (ratio 1.67 -> 1.74 at h=1), far short of
joint pretraining (0.34, F280). Not separated: the adaptation budget (3 clips, 1000 steps, rank-2 LoRA on
Mlp layers only), and whether Stages 2-4 (without Stage 4's head refit) preserve or use the alignment.

---

### F282. The jointly pretrained model needs only a B1 projector (best B1 rollout measured); adaptation onto a hexapod-only pretrain, even with equal data and an anchor, stays below it

`scripts/run/equal_data_anchor.sh`. B1 selection (library beh12 B1 clean-train, held-out hexapod goals, NS
at w=11; oracle 0.031, random 0.126):
- J: `joint_sim_beh24_s0` (F280) with Stage 2 only -- the B1 projector fit on beh24 B1 clean-train;
  ITM, FTM and Froude head straight from pretraining (no Stage 1 re-adaptation, no Stage 4 head refit).
- E*: `fmd_beh24_s0` (hexapod only, F279) adapted with the same B1 data at every stage (beh24 B1
  clean-train: Stage 1 on 44 clips + 4 for its report, Stages 2 and 4 on all 48); Stage 1 with or
  without the Froude-head anchor (`--anchor_froude 1.0`), with or without the Stage 4 head refit.

| model | direct | rollout (library start) | rollout (current start) | retrieval c10 -> B1 |
|---|---|---|---|---|
| J: joint + similarity, Stage 2 only | **+0.94** | **+0.76** | **+0.65** | 0.34 (F280) |
| J with the standard Stages 1, 2, 4 (F280) | +0.95 | +0.71 | +0.54 | -- |
| E0: no anchor, no Stage 4 | +0.71 | +0.20 | +0.08 | 0.04 |
| E1: no anchor, Stage 4 | +0.84 | +0.57 | +0.44 | 0.04 |
| E2: anchor, no Stage 4 | +0.68 | +0.56 | +0.32 | 0.19 |
| E3: anchor, Stage 4 | +0.85 | +0.59 | +0.36 | 0.19 |

Stage 1 rollout ratio at h=10 (after): 1.80 without anchor, 1.79 with. Retrieval c10 -> c08 0.41-0.42
for both adapted ITMs.

**Reading.**
- The joint model is usable as pretrained: a B1 projector alone gives the best B1 rollout measured
  (+0.76 / +0.65); re-running Stages 1 and 4 on it lowers rollout (current start +0.65 -> +0.54).
- On a hexapod-only pretrain, the anchor with equal data raises retrieval to 0.19 (3 clips: 0.12,
  F281) and makes the unrefit pretrained head usable for rollout (library start +0.20 -> +0.56,
  current start +0.08 -> +0.32), but direct is not better (+0.68 vs +0.71) and, with the head refit,
  anchor and no anchor are equal (+0.84-0.85). No adapted variant reaches the joint model.
- Current best route to a shared, usable B1 latent: pretrain the bodies together (with the similarity
  loss); adapt only the projector. Single seed each; the matched similarity on/off pair is training.

---

### F283. A stronger anchored adaptation (LoRA rank 8, 3000 steps) makes the hexapod's own Froude head read the B1 without any head refit; retrieval stays at ~0.19

`scripts/run/local_overnight.sh` part A. Base `fmd_beh24_s0`; Stage 1 with LoRA rank 8 (was 2), 3000 steps
(was 1000), same B1 data at every stage (44 + 4 / 48 / 48 clips), anchor = frozen Froude head (A1) or
Froude head + similarity bank (A2). NS at w=11; retrieval and cross-body R2 on the adapted ITM.

| model | direct | rollout (library start) | rollout (current start) | retrieval c10 -> B1 | cross-body R2 c10 -> B1 |
|---|---|---|---|---|---|
| A1 Froude anchor, no Stage 4 | +0.87 | +0.66 | +0.41 | 0.19 | +0.14 / +0.24 / +0.45 |
| A1 Froude anchor, Stage 4 | +0.88 | +0.59 | +0.41 | | |
| A2 both anchors, no Stage 4 | +0.78 | +0.67 | +0.41 | 0.18 | +0.19 / +0.23 / +0.46 |
| A2 both anchors, Stage 4 | +0.88 | +0.56 | +0.31 | | |
| rank 2, 1000 steps, anchor, no Stage 4 (F282 E2) | +0.68 | +0.56 | +0.32 | 0.19 | -0.72 / +0.16 / +0.42 |
| usual recipe, no anchor, Stage 4 (F282 E1) | +0.84 | +0.57 | +0.44 | 0.04 | |
| joint + similarity, Stage 2 only (F282 J) | +0.94 | +0.76 | +0.65 | 0.34 | +0.13 / +0.30 / -0.34 |

Stage 1 rollout ratio h=10: 1.72 (rank 2: 1.79-1.80), i.e. the B1 prediction also improved.

**Reading.**
- With more adaptation capacity the anchor places the B1 where the hexapod-trained Froude head reads it:
  cross-body R2 to the B1 is positive on every channel (first time for an adapted model), and selection
  without the head refit (+0.87 / +0.66 / +0.41) is as good as or better than the usual recipe with it.
  The Stage 4 refit adds nothing (A1) or lowers rollout (A2).
- Nearest-neighbour retrieval stays at 0.18-0.19: the Froude-relevant part of z is aligned, the rest is not.
- Still below joint pretraining on every selection column.

---

### F284. Matched pair: the similarity loss makes a c10-fit Froude read-out transfer to the B1 but does not change nearest-neighbour retrieval or selection; joint pretraining itself gives the retrieval

Joint hexapod + B1 pretraining on the current pipeline (beh24 48 + 48 clips, stride 5, Froude head shapes z,
joint-command decoder off). The only difference between S0 arms is `lambda_sim`. `scripts/run/jointD_server.sh`
(S0 pair, lab server), `scripts/run/local_overnight.sh` part B (S1). Evaluated as joint models:
`scripts/run/eval_joint_models.sh`, Stage 2 only (projectors fit on beh24 clean-train with the pretrained
ITM), NS at w=11; shared latent on pretrained ITMs. **All on the old B1 renders** (F275 addendum).

| model | B1 direct | B1 roll lib | B1 roll cur | c08 direct | c08 roll lib | c08 roll cur | c10 direct | c10 roll lib | c10 roll cur |
|---|---|---|---|---|---|---|---|---|---|
| similarity 0.05, S0 | +0.94 | +0.75 | +0.65 | +0.78 | +0.61 | +0.57 | +0.92 | +0.77 | +0.68 |
| **no similarity, S0** | +0.94 | +0.74 | +0.65 | +0.79 | +0.56 | +0.48 | +0.91 | +0.68 | +0.69 |
| similarity 0.05, S1 | +0.93 | +0.75 | +0.60 | +0.78 | +0.55 | +0.45 | +0.90 | +0.79 | +0.72 |
| earlier: similarity + joint-command decoder (F280) | +0.94 | +0.76 | +0.65 | +0.78 | +0.53 | +0.51 | +0.91 | +0.75 | +0.57 |

| model | body-ID probe | cross-body R2 c10 -> c08 | cross-body R2 c10 -> B1 | k-NN mixing | retrieval c10 -> c08 | retrieval c10 -> B1 |
|---|---|---|---|---|---|---|
| similarity, S0 | 0.73 | +0.72 / +0.36 / +0.37 | **+0.48 / +0.07 / +0.17** | 0.35 | 0.45 | 0.34 |
| **no similarity, S0** | 0.74 | +0.70 / +0.36 / +0.23 | **-0.85 / +0.17 / -0.22** | 0.35 | 0.41 | 0.33 |
| similarity, S1 | 0.74 | +0.75 / +0.35 / +0.32 | **+0.70 / +0.30 / -0.07** | 0.36 | 0.44 | 0.42 |
| earlier, with decoder | 0.73 | +0.72 / +0.36 / +0.17 | +0.13 / +0.30 / -0.34 | 0.37 | 0.43 | 0.34 |

**Reading.**
- Joint stride-5 pretraining with the Froude head shaping z gives B1 retrieval about 0.33 without any
  similarity loss (hexapod-only pretraining: 0.04-0.05). The retrieval gain in F280 came from joint
  pretraining, not from the similarity loss.
- The similarity loss changes the cross-body read-out: a Froude read-out fit on c10 predicts the B1's forward
  motion (R2 +0.48 / +0.70 on two seeds vs -0.85 without it). It aligns the Froude-relevant direction of z
  across bodies, not the nearest-neighbour mixing.
- Selection on the B1 is identical with or without it (+0.94 / +0.74-0.75 / +0.65). The B1 projector is fit
  in the pretrained space either way. On c08 the similarity arm's rollout is higher at S0 (+0.61 / +0.57 vs
  +0.56 / +0.48) but not at S1 (+0.55 / +0.45): within seed spread.
- Joint pretraining gives the best B1 rollout measured (+0.74-0.76 / +0.60-0.65) in every variant.
- Pending: the same comparison on B1 data with matched rendering (v3).

---

### F285. The joint models' B1 representation depends on the old B1 rendering: fed matched-render (v3) frames, B1 selection and sharing collapse

`scripts/run/eval_on_v3.sh`, no retraining. The F284 models (pretrained on the old B1 renders) get:
- the B1 projector refit on v3 B1 clean-train frames (Stage 2);
- B1 selection on the v3 library;
- shared latent with the B1 on v3.

v3 = `rerender_b1_ego_matched.py`, pure scaling. Its scene is shown to render like the hexapod's by the
static test, 0.552.

| model | B1 direct | B1 roll lib | B1 roll cur | retrieval c10 -> B1 | cross-body R2 c10 -> B1 | body-ID probe | k-NN mixing |
|---|---|---|---|---|---|---|---|
| similarity, S0: old B1 -> v3 | +0.94 -> **+0.24** | +0.75 -> +0.38 | +0.65 -> +0.40 | 0.34 -> 0.22 | +0.48 / +0.07 / +0.17 -> -0.71 / -0.27 / -1.20 | 0.73 -> 0.65 | 0.35 -> 0.45 |
| no similarity, S0 | +0.94 -> **-0.08** | +0.74 -> +0.21 | +0.65 -> +0.27 | 0.33 -> 0.15 | -0.85 / +0.17 / -0.22 -> -1.66 / +0.09 / -1.27 | 0.74 -> 0.68 | 0.35 -> 0.45 |
| similarity, S1 | +0.93 -> **+0.39** | +0.75 -> +0.31 | +0.60 -> +0.42 | 0.42 -> 0.20 | +0.70 / +0.30 / -0.07 -> -0.61 / -0.09 / -0.94 | 0.74 -> 0.65 | 0.36 -> 0.47 |

**Reading.**
- The models encode the B1 through features specific to its old rendering. On matched-render frames the
  B1's z no longer lands where the pretrained Froude head reads it, even though:
  - the frames look more like the hexapod's (body-ID down, k-NN mixing up);
  - the projector is refit.
- Direct selection collapses, so the effect is in the B1's z / read-out, not only in rollout.
- The similarity arms degrade less than the control (direct +0.24 / +0.39 vs -0.08).
- This is a train/test render shift, not a measurement of a model trained on v3. It shows the old-render
  B1 results (F280-F284) rest partly on render-specific features, so they cannot be carried over.
  Cross-body results must be retrained on v3 to be clean.
