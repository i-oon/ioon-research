# Stage 1 findings: why cross-morphology transfer fails, and what fixes part of it

> **Role**: What is true, with the numbers.
>
> Append only. A later finding may bound or correct an earlier one -- it says so in place rather than the earlier one being edited, so the record of what was believed when stays intact. Open questions live in `OPEN_QUESTION.md`; the chronology lives in `PROGRESS.md`.

**The short version.** A frozen video encoder carries robot morphology in a linearly decodable
form that generalises to unseen bodies (F20). The world model trained on top of it ignores that
and identifies the body from an 11-percent component of its own latent instead (F18, F19), which
is a lookup and does not extend past the bodies it saw. Four decoder-side interventions fail to
change this (F4b, F21, F22) and more training bodies helps only inside their hull (F16, F17).

**What fixes it is changing the question, not the architecture.** Decoding one body's latent
against another body's frame, supervised by that body's command, makes the lookup wrong by
construction. Held-out error improves 23 to 26 percent, the decoder switches to reading the body
from the frame (frame ablation 10.7x to 54x, and a crossed swap test now follows the frame to
within 0.01 deg), copying stops, and for the first time performance does not decay with training
(F24).

Underneath all of it: **the reconstruction loss, which is supposed to make the latent an action,
contributes 3 to 7 percent of the forward model's accuracy while taking 99 percent of the
gradient** (F23). The latent is therefore shaped almost entirely by the motion loss, on one
percent of the training signal, and a body code is the cheapest thing that satisfies it. No
change to the decoder can repair a latent that nothing constrained.

Two datasets. F1 to F14 use `data/ik_walk_100_framed`, two training bodies differing in one
parameter. F15 onward use `data/ik_walk_8body`, five training bodies differing in three.

Measured on `data/ik_walk_100_framed` (100 expert episodes x 3 leg-length bodies x 66 frames,
19,800 frames, 0 clipped). Architecture: frozen V-JEPA2 ViT-g/16 encoder, ITM -> `z` in R^64,
FTM, Motion Decoder -> 18-D joint command. Training bodies `long` (leg scale 1.0) and `short`
(0.5); `medium` (0.75) held out unless stated.

Everything below is a measurement from this repository, with the command that produces it.
Numbers are quoted with the body and checkpoint they came from, because they differ by both.

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

## What works

### F1. The model reads body identity from pixels and applies the right joint offsets, with no morphology label

Reconstruction on bodies seen in training, RMSE in degrees per joint
(`wm/runs/stage1_100ep_framed_runB/epoch020.pt`):

| Body | TC | CF | FT |
|---|---|---|---|
| long | 0.75 | 0.51 | 0.46 |
| short | 0.71 | 0.53 | 1.14 |

The mean joint angles of these two bodies differ by 33.8 deg (CF) and 50.1 deg (FT), and the
model places each within 0.03 to 0.06 deg of the correct one. It is never told which body it
is looking at, and morphology appears nowhere in the input or the loss.

### F2. The latent action is doing work

Zeroing `z` costs a factor of 3 to 4 on the held-out body across training
(`heldout/motion_zero_z` divided by `heldout/motion`, 2,600 pairs per point). The decoder is
not reading the joint command off the current frame alone.

### F3. Phase is recovered from video essentially exactly

TC on the held-out body scores 1.35 deg RMSE against a signal of 17.1 deg std. Whatever else
fails, the model knows where in the gait cycle it is.

## What does not work

### F4. Transfer to an unseen body covers about half the required distance

![Morphology signal through the pipeline](results/wm/stage1/figures/morphology_axis.png)

*Left: the body's position between the two training bodies survives the encoder (0.465) and is
lost in the ITM and decoder. Right: correcting the loss weighting makes the model move along the
axis but not converge on the answer. Both panels: 0 = identical to `long`, 1 = identical to
`short`, green band = correct answer for the held-out `medium` body.*

Where the model places the held-out `medium` body on the `long` to `short` axis
(0 = long, 1 = short):

| Representation | Position, 12 clips per body | Position, 3 clips per body |
|---|---|---|
| frozen V-JEPA2 embedding `e_t` | **0.499** | 0.465 |
| learned latent `z` | 0.335 | 0.301 |
| decoder output, CF | **0.188** | 0.152 |
| decoder output, FT | 0.180 | 0.145 |
| correct answer | **0.357** (CF), 0.304 (FT) | same |

`e_t` sits at 0.499 against the 0.5 expected from leg scale alone, so the body's position
between the two training bodies survives the encoder intact. The signal is then lost
downstream: the decoder reaches 0.188 where 0.357 is correct, 53 percent of the way. Resulting
error on the held-out body: TC 1.35, CF 11.34, FT 15.14 deg.

Quadrupling the data (195 to 780 transitions per body) moves every figure by less than 0.04 and
changes no conclusion, so these are not small-sample artefacts.

Position is a scalar projection onto the axis between the two training bodies' means,
`t = <q - a, b - a> / <b - a, b - a>`, so `t = 0` at `long` and `t = 1` at `short`. In joint
space it is computed per joint and averaged over joints the two training bodies separate by
more than 2.0 deg; TC fails that filter entirely, which is F8 restated.

Read the rows as separate statements rather than one decaying quantity: the axis in embedding
space and the axis in joint space are different axes, so 0.465 in one is not "more" than 0.301
in the other. What is directly comparable is the last two rows, both in joint space.

Reproduce: `.venv/bin/python3 scripts/diagnostics/morphology_axis.py --ckpt wm/runs/<run>/epoch020.pt`

### F4b. A linear probe on the same latent generalises better than the trained decoder

Ridge regression from `z` to the joint command, fitted on the two training bodies only, then
evaluated on the held-out body. It reads exactly the same latent the Motion Decoder reads.

| Predictor, RMSE deg per joint | long (trained) | short (trained) | medium (held out) |
|---|---|---|---|
| Motion Decoder | **0.63** | **0.92** | 11.04 |
| ridge probe on `z` | 3.56 | 3.79 | **5.13** |
| mean of the two training bodies | -- | -- | 6.68 |

Measured on 12 clips per body; at 3 clips the same figures are 0.58 / 0.83 / 10.94 and
3.33 / 3.82 / 4.96.

The decoder is 6.6x better in-distribution and 2.2x worse out-of-distribution. The probe
degrades by 1.4x going to an unseen body; the decoder degrades by 15x. The probe also lands at
axis position 0.335 against a correct 0.357, where the decoder reaches 0.188.

The probe is not solving anything. An affine map sends a point at parameter `t` along the
segment between the two training bodies to the point at parameter `t` on the image segment, so
a linear readout can only pass the representation's axis position through unchanged. Measured:
`z` places the held-out body at 0.3352 and the ridge output lands at 0.3354.

That relocates the credit. The useful work is done by the **ITM**, which places the held-out
body at 0.335 against a correct 0.357 -- 94 percent of the way -- while the frozen encoder
placed it at 0.499, the value leg scale alone would predict. The map from leg scale to joint
offset is not linear (a 0.75-scale body sits at 0.357, not 0.5) and the ITM absorbs that
curvature. The linear probe then preserves it and the trained decoder overrides it, pulling
0.335 down to 0.151. The decoder is not failing to find the answer; it is discarding an answer
its input already contained.

Stated safely: on this dataset a high-capacity readout can distort a latent that was already
close to correct, and a linear one cannot. "Less capacity generalises better" is not
established as a general rule and should not be written as one.

It also matters for F6: the probe beats the averaging baseline (5.13 against 6.68 deg), so the
pipeline does contain something better than a predictor that does no learning. Constraining the
decoder is a second lever alongside adding bodies.

Two limits on how far this generalises:

**The bodies are uniformly scaled, so the task is close to affine.** `sim/scene/make_leg_morphology.py`
takes one factor and scales coxa, femur and tibia together, and the resulting joint commands
differ between bodies by 92 to 99 percent constant offset (F7). A linear readout is well matched
to a near-affine problem, so its win here may not survive a morphology space with independent
segment scaling. Scaling the three segments separately and repeating this table is the
experiment that settles it, and both outcomes are reportable.

**Every readout in these tables was fitted post hoc on a frozen `z`, and training end to end
with a smaller head does not reproduce the result.** `head_linear` keeps the cross-attention
backbone and replaces the two-layer output head with a single projection, 272,914 parameters
down to 10,258. Against `fix_norm`, which differs only in that head, over the first ten epochs
on held-out `medium`:

| | mlp head | linear head |
|---|---|---|
| best held-out | **0.522** | 1.121 |
| mean held-out, epochs 1-10 | **1.031** | 1.649 |
| mean held-out, epochs 6-10 | **1.201** | 1.735 |
| mean validation, epochs 6-10 | **0.0110** | 0.0131 |
| mean z-ablation gap | 4.2x | 4.2x |

The linear head is 1.4 to 2.1 times worse and is worse in all ten epochs, which is outside the
run-to-run spread. The likely reason is that it removes only 5 percent of the decoder and
leaves the cross-attention block (3.15M of 5.23M) untouched, while forcing the backbone to make
its features linearly separable on its own.

This closes the capacity hypothesis as a fix. What made the ridge probe look good was the
latent the original training produced, not the shape of the readout; change the readout during
training and the latent changes with it. Coverage remains the only intervention that has
improved anything.

The probe reads only `z` while the Motion Decoder also reads `e_t`, so capacity and input access
are confounded in that table. Fitting both model classes on all three input sets separates them,
every one fitted on the training bodies alone:

| Input | Readout | trained bodies, deg | medium, deg | degradation |
|---|---|---|---|---|
| `z` only (64-d) | linear | 3.58 | **4.96** | **1.4x** |
| `z` only | MLP 256-256 | 0.77 | 5.48 | 7.1x |
| `e_t` only (1408-d) | linear | 1.98 | 8.21 | 4.1x |
| `e_t` only | MLP 256-256 | 0.16 | 13.16 | 82.2x |
| `e_t` + `z` (the decoder's inputs) | linear | 0.89 | **4.88** | 5.5x |
| `e_t` + `z` | MLP 256-256 | 0.17 | 13.71 | 79.3x |
| `e_t` + `z` | Motion Decoder (5.2M) | 0.71 | 10.94 | 15.4x |

Capacity is the dominant term. Within every input set the nonlinear readout is better on the
training bodies and worse on the held-out one, and the gap widens with capacity: 1.4x for a
linear map on `z`, 15.4x for the trained decoder, 79x for a plain MLP on the same inputs.
Given the decoder's own inputs, a linear readout reaches **4.88 deg on the held-out body against
the decoder's 10.94**, so the decoder is not failing for want of information.

Access to `e_t` contributes a second, smaller effect: linear on `e_t` alone degrades 4.1x against
1.4x on `z` alone, which is consistent with `e_t` carrying body identity in a form that is easy
to key on. But `z` alone (4.96) and `e_t` + `z` (4.88) land in the same place, so the ITM has
already distilled what a well-conditioned readout needs.

### F4c. Replaying the commands in physics shows the same split

Both command streams driven open-loop through the same scene and physics as the collector used,
held-out `medium`, all three clips (`sim/render/render_wm_prediction.py`, `scripts/diagnostics/wm_gait_report.py`):

| Driven by | forward x, m (clip 0 / 1 / 2) | mean forward | mean absolute heading error | body height, m | tripod score (0 / 1 / 2) |
|---|---|---|---|---|---|
| IK ground truth | 0.653 / 0.625 / 0.639 | **0.639** | **4.4 deg** | 0.111 | -0.40 / -0.35 / -0.38 |
| Motion Decoder | 0.312 / 0.045 / 0.111 | 0.156 (24%) | 42.5 deg | **0.089** | **+0.16 / +0.16 / +0.25** |
| ridge probe on `z` | 0.358 / 0.514 / 0.568 | 0.480 (75%) | 37.6 deg | 0.111 | -0.34 / -0.17 / -0.46 |

Tripod score is the correlation between the two tripod groups' contact counts: negative means
the groups alternate as a stick insect's gait should, zero means they have stopped coordinating.
Heading error is the angle of net displacement away from the walking direction.

Neither is a usable controller: both veer, by 38 to 43 degrees on average against the ground
truth's 4.4. The difference is in what survives. The **decoder** sinks from 0.111 m to 0.089 m
of body height in every clip, covers 24 percent of the ground-truth distance, and its tripod
score is positive in all three clips, so the alternating gait is gone. The **probe** holds body
height exactly, covers 75 percent of the distance, and keeps a negative tripod score in all
three clips, so the gait survives.

The visible failure in the videos is legs folding into the abdomen, and it is measurable
without any per-frame ground truth: what fraction of commanded angles fall outside the range
that body reaches across all 100 of its episodes.

| Driven by | commands out of range (clip 0 / 1 / 2) | worst excursion | TC | CF | FT | commanded peak-to-peak, CF / FT |
|---|---|---|---|---|---|---|
| IK ground truth | 0.0% / 0.0% / 0.0% | 0.0 deg | 0% | 0% | 0% | 28.0 / 47.3 deg |
| Motion Decoder | **16.1% / 16.9% / 15.5%** | **40.2 deg** | 0.9% | 19.3% | **28.3%** | **53.5 / 81.0 deg** |
| ridge probe on `z` | 6.7% / 8.3% / 7.4% | 14.5 deg | 3.6% | 5.6% | 13.2% | 35.1 / 60.9 deg |

The body's own range is 29.4 deg (CF) and 47.8 deg (FT). The decoder commands swings 1.9x and
1.7x wider than that, up to 40 deg past the limit, in every clip. Its TC amplitude is correct to
within 1.3 deg. This is the `gain` of 0.30 from F5 made physical: over-amplified distal joints
drive the leg into a pose the body cannot hold.

Because it needs only the body's joint range and not matched expert data, this measure carries
over to a body with no ground truth and across embodiments, where B1 has its own limits.

Single clips mislead here. On clip 0 alone the decoder veers 25.2 deg against the probe's 57.9
and looks the steadier of the two; that is the decoder's straightest clip and the probe's worst.
Across three clips the decoder's heading errors are +25.2, -37.5 and +64.8 deg. Report all
clips, or the wrong predictor wins.

![Gait, decoder](results/wm/stage1/figures/gait_stage1_100ep_framed_runB_epoch020_medium_clip0.png)

*Decoder-driven gait above, ground truth below. Black is stance. The stance blocks fragment and
the left and right tripods stop alternating.*

![Gait, probe](results/wm/stage1/figures/gait_probe_ridge_runB_medium_clip0.png)

*Probe-driven gait above, ground truth below, same axes. Stance blocks keep roughly the right
length and phase.*

Videos: `results/wm/replay/replay_*.mp4`, predicted on the left, ground truth on the right.

### F4d. Interpolation and extrapolation fail at different stages

Fold 2 holds out `short` (leg scale 0.5) while training on `long` (1.0) and `medium` (0.75), so
the test body lies outside the training range rather than between the training bodies. Axis
here runs 0 = `long`, 1 = `medium`, and the correct position for `short` is 2.803 (CF).

![Both folds](results/wm/stage1/figures/axis_both_folds.png)

| Stage | fold 1, held-out medium (bracketed) | fold 2, held-out short (outside), epoch 6 | fold 2, epoch 20 |
|---|---|---|---|
| frozen `e_t` | 0.499 | 1.507 | 1.507 |
| latent `z` | **0.335** | 1.202 | **1.098** |
| decoder output | **0.188** | 1.052 | **1.026** |
| correct | 0.357 | 2.803 | 2.803 |

A position of 1.0 means "identical to the nearest training body". Both the ITM and the decoder
pull the unseen body back to that value, and pull harder the longer training runs: `z` moves
1.202 to 1.098 and the decoder 1.052 to 1.026 between epochs 6 and 20. That is the
copy-the-nearest-body behaviour visible in the aggregate (model 7.00 against copy-medium 6.96)
seen at the level of the representation.

The diagnosis therefore differs by fold and must not be stated as one:

| | fold 1, bracketed | fold 2, outside the range |
|---|---|---|
| encoder | correct, 0.499 against 0.357 | already short, 1.507 against 2.803 (54%) |
| ITM | correct, 0.335, 94% of the way | makes it worse, 1.10 |
| decoder | **the failing stage**, 0.188 | drives it to 1.03 |

With the body bracketed, the ITM absorbs the curvature of the leg-scale to joint-offset map and
only the decoder discards it. Outside the range, the frozen encoder is already the largest
single gap: a perfect ITM and decoder reading `e_t` could reach 1.507 of a required 2.803.

This bounds the architectural fix. A linear readout can only pass the representation's position
through, so it helps exactly where the representation is right:

| | ridge probe on `z` | Motion Decoder | probe advantage |
|---|---|---|---|
| fold 1, medium | **5.13 deg** | 11.04 deg | **2.2x** |
| fold 2, short | 21.65 deg | 23.89 deg | 1.1x |

Both beat the no-learning baseline for fold 2 (29.43 deg), but neither is close to correct.

**Generalisation here comes from coverage, not from the model.** A body between the training
bodies is recovered to 94 percent at the latent; a body outside them is pulled back to the
nearest one at every stage. Training sets must bracket the bodies they are meant to generalise
to, and no change to the decoder substitutes for that.

### F5. Two training bodies cannot define a curve

The motion loss sees exactly two points: "looks like long -> long's offsets" and "looks like
short -> short's offsets". Every function through those two points has identical loss, so
nothing constrains the space between them. A network trained on two well-separated clusters
fits two plateaus rather than a ramp, and an input in between falls toward the nearer plateau.

The clearest evidence is what changing the loss weighting does. Standardising by within-body
spread instead of pooled spread (F9) raises CF and FT from 0.12 to 0.95 of TC's weight, and
the model then does move along the morphology axis -- but without control:

| Run | axis CF | axis FT | all-joint RMSE, deg |
|---|---|---|---|
| pooled std, epoch 6 | 0.11 | 0.10 | 8.99 |
| pooled std, epoch 20 | 0.15 | 0.14 | 10.95 |
| within-body std, epoch 6 | 0.25 | 0.24 | 9.87 |
| within-body std, epoch 18 | 0.61 | 0.59 | 17.30 |
| correct answer | 0.36 | 0.30 | 0 |

With the corrected weighting the model swings from undershooting (0.11) to overshooting
(0.61) as training proceeds. It is not converging on the right answer from either side; it is
drifting through it, because nothing in the objective marks where the right answer is.

### F6. Trivial baselines beat the model on this task

![Interpolation failure](results/wm/stage1/figures/interpolation_failure.png)

*Left: both training bodies are reconstructed to under 1.2 deg per joint while the held-out body
is not. Right: shown the medium body, the model's output is closer to a training body's geometry
than to the correct answer.*

Held-out `medium`, RMSE in degrees per joint:

| Predictor | TC | CF | FT | all |
|---|---|---|---|---|
| mean of the two training bodies' commands | 0.04 | 5.33 | 10.26 | 6.68 |
| model, epoch 6 | 2.02 | 9.64 | 12.05 | 8.99 |
| model, epoch 20 | 1.35 | 11.34 | 15.14 | 10.95 |

Held-out `short` (fold 2, `--train_morphs long medium`), motion MSE in the run's own units:

| Predictor | MSE |
|---|---|
| linear extrapolation, `medium + (medium - long)` | 1.91 |
| copy `medium` ground truth | 6.96 |
| model, epochs 1 to 28 | 6.93 to 7.07 |
| mean of the two training bodies | 10.77 |
| predict the training mean | 12.54 |

On fold 2 the model is indistinguishable from copying the nearest training body, and a linear
extrapolation that knows only the ordering of leg lengths beats it by 3.7x. The held-out score
does not move at all across 28 epochs while validation improves 8x.

Caveat: these baselines use time-aligned ground-truth commands from the training bodies, which
the model does not receive. F3 shows the model recovers phase to 1.35 deg, so the comparison
is meaningful, but it is not a like-for-like input.

### F7. Between-body differences on this dataset are 92 to 99 percent constant offset

The IK retargeting drives every body along one shared Cartesian foot trajectory, scaled to
fit. The resulting joint commands are near-affine transforms of each other:

| Pair | RMSE, deg | after removing a per-joint constant | fraction that is offset |
|---|---|---|---|
| long vs medium, CF | 11.86 | 1.10 | 99% |
| long vs medium, FT | 14.88 | 4.08 | 92% |
| short vs medium, CF | 22.08 | 5.88 | 93% |
| long vs medium, TC | 0.59 | 0.23 | 84% |

Changing leg length shifts posture and barely changes the movement pattern. This is why the
averaging baseline in F6 is so strong, and it is a property of how the data was generated
rather than of the world model.

The relationship is nonlinear, which is the opening: `medium` at leg scale 0.75 sits at 0.5
on the scale axis but at 0.30 to 0.36 on the joint-offset axis. No linear baseline can be
right about that. A model that reads appearance and produces the correct curve would beat
every baseline in F6 -- but two training bodies can only express a straight line.

## Pitfalls, with the evidence that they are real

### F8. Aggregate error hides which joints transferred

![Per-joint traces on the held-out body](results/wm/stage1/figures/action_trace_stage1_100ep_framed_runB_epoch020_medium.png)

*Predicted (red, dashed) against IK ground truth (black) for all 18 joints on the held-out body.
Left column is TC and tracks exactly; the CF and FT columns are over-amplified and out of phase.
Ground-truth frames are the input, so this is action reconstruction, not closed-loop control.*

The 18-joint average on the held-out body reads 0.208 in standardised units, which looks
healthy. Split by joint type, against each joint's own mean over the clip:

| Joint type | MSE | constant baseline | verdict |
|---|---|---|---|
| TC | 0.006 | 1.001 | 164x better than a constant |
| CF | 0.382 | 0.121 | 3x worse than a constant |
| FT | 0.236 | 0.211 | no better than a constant |

TC scores near zero and drags the mean down. Report `motion_mse_per_joint_type` from
`wm.evaluate`, not the average.

TC is also the wrong joint to celebrate: the three bodies' TC commands differ by only 0.58 to
1.17 deg, so there is almost nothing there to transfer. Cross-morphology claims rest on CF and
FT, the joints that set how high and how far the foot goes.

### F9. The standardisation scale silently reweights joints

Standardising the motion target by the spread of all training bodies pooled together counts
the posture gap between bodies as signal amplitude. Measured on `long` plus `short`:

| Joint type | pooled std, deg | within-body std, deg | weight in the loss |
|---|---|---|---|
| TC | 17.1 | 17.1 | 1.00 |
| CF | 18.4 | 7.9 | 0.12 -> 0.95 |
| FT | 31.4 | 20.7 | 0.11 -> 0.84 |

TC keeps full weight because all three bodies move it by the same amount. The joints that
differ between bodies -- the ones the whole experiment is about -- were receiving an eighth of
the gradient. Fixed by `within_body_std` in `wm/config.py`, on by default.

Correcting this did not improve held-out RMSE (F5), so it was not the bottleneck, but it is
the correct scaling and it changes what the model does with the morphology axis.

### F10. The trivial baseline of 1.0 does not hold on a held-out body

Motion MSE is standardised by the training bodies' statistics, so "predicting the mean costs
1.0" is true only on those bodies. On held-out `medium` the same trivial predictor costs
0.495, and that body's own mean costs 0.444. Claims of the form "N times better than no skill"
must use the held-out body's own baseline. `wm.evaluate` reports both as
`predict_training_mean` and `predict_this_body_mean`.

### F11. Validation on held-out episodes cannot detect cross-body failure

![Validation against held-out body](results/wm/stage1/figures/heldout_sweep_two_seeds.png)

*Same configuration and same `seed: 0` on two different GPUs. Left: validation on unseen episodes
of the training bodies improves by an order of magnitude. Right: the held-out body does not, and
the two runs disagree by up to 2.1x.*

Two runs of the identical configuration, from 3,086 to 30,860 gradient steps:

| | run A | run B |
|---|---|---|
| validation motion, unseen episodes of the training bodies | 0.0118 -> 0.0011 (11x better) | 0.0125 -> 0.0012 (10x better) |
| held-out body | 0.295 -> 0.220 | 0.140 -> 0.203 |

Ninety percent of the compute budget buys an order of magnitude on validation and nothing
measurable on the held-out body. A new body is a different distribution, not a held-out sample
of the same one. Fold 2 shows the same shape: validation 8x better, held-out flat across 28
epochs.

### F12. Held-out scores need error bars; in-distribution scores do not

Same configuration, same `seed: 0`, different GPU. How far apart the two runs land:

| Metric | median ratio | worst |
|---|---|---|
| reconstruction, train and validation | 1.003 | 1.005 |
| motion, train and validation | 1.14 to 1.18 | 1.38 |
| held-out body | 1.247 | 2.103 |

Floating-point rounding is enough to move held-out performance by a factor of two while the
models remain identical to 0.3 percent in-distribution. This is the numerical signature of
F5: extrapolation is underdetermined, so arbitrarily small differences pick different answers.
Single-run held-out numbers are not interpretable.

### F13. More episodes of the same bodies does not help

| Run | Episodes | Steps | Held-out MSE |
|---|---|---|---|
| stage1_6ep_clipped | 6 | 9,750 | 0.166 |
| stage1_100ep_clean | 100 | 9,500 | 0.179 |
| stage1_100ep_clipped | 100 | 30,880 | 0.422 |

A 16x increase in episodes of the same two bodies changes nothing. Data budget spent on
episodes is wasted; spend it on bodies.

### F14. Peak transfer arrives in the first tenth of training

![Held-out sweep](results/wm/stage1/figures/heldout_sweep_runB.png)

*Every snapshot re-scored on 2,600 identical cached held-out pairs. The dotted line at 1.0 is
predicting the mean; the grey curve is the same model with the latent zeroed.*

Re-scoring every snapshot on 2,600 identical cached held-out pairs
(`wm.sweep_checkpoints`), transfer is best between 3,086 and 9,258 gradient steps of 30,860 and
does not improve after. Selecting a checkpoint on this curve would leak the test body, so the
curve is for reporting compute cost, not for choosing a model.

## What more bodies fix

Everything above was measured with two training bodies differing along one axis. This section
uses `data/ik_walk_8body`: nine bodies generated by scaling coxa, femur and tibia
independently (`sim/scene/make_leg_morphology.py`), 30 clips each, 0 percent of frames edge-clipped.
Two bodies were dropped because they stumble -- both have femur 0.6 with tibia 1.0, and their
head height falls to 0.03 m against 0.111 m for the rest. Five bodies train; `c08f09t09`
(0.8, 0.9, 0.9) is held out and lies inside their convex hull; `c06f06t06` is held out in a
second run and lies outside it.

![The nine bodies](results/wm/dataset/morphology_bodies.png)

### F15. The morphology space is three parameters but two dimensions

Scaling the coxa barely moves the joint commands, so the family of bodies is thinner than the
parameterisation suggests:

| Change, other segments fixed | Command change |
|---|---|
| coxa 1.0 to 0.6 | **0.73 deg** |
| tibia 1.0 to 0.6 | **28.63 deg** |

An SVD of how the five training bodies deviate from their mean gives 82.4 percent to the first
direction, 17.5 percent to the second, and 0.0 percent to the remaining three. The first
correlates with tibia at -0.93, the second with femur at -1.00. The coxa is the shortest
segment and sits against the body, so shortening it barely moves the foot and the IK has almost
nothing to compensate for.

Reconstructing the held-out body from a k-dimensional basis of the training bodies:

| k | RMSE deg |
|---|---|
| 0 (the average body) | 11.489 |
| 1 | 0.483 |
| **2** | **0.203** |
| 5 | 0.174 |

Two numbers place the held-out body to 0.2 deg. The decoder emits 18 free numbers per timestep.

### F16. Five bodies cut held-out error 3.1x and beat the no-learning baseline

Held-out `c08f09t09`, RMSE deg per joint, `m3d_bracketed` at epoch 6:

| Predictor | TC | CF | FT | all |
|---|---|---|---|---|
| best possible linear mixture | -- | -- | -- | **0.18** |
| **model** | 2.25 | 3.59 | 4.53 | **3.57** |
| copy the nearest training body | 0.22 | 4.30 | 4.18 | 3.46 |
| mean of the five training bodies | 0.15 | 19.29 | 4.85 | 11.48 |
| predict this body's own mean | 18.24 | 6.30 | 11.83 | 13.07 |

Against the two-body setup's 11.04 deg this is 3.1 times better, and it clears the averaging
baseline that beat every earlier run (3.57 against 11.48). The `z`-ablation gap rises from 3-4x
with two bodies to **10-37x** with five, so the latent became far more load-bearing.

It is still 20 times worse than a linear mixture of the training bodies, and indistinguishable
from copying the nearest one.

### F17. Bracketed against outside, with everything else held fixed

`m3d_bracketed` and `m3d_outside` share training bodies, data, normalisation and
hyperparameters, and differ only in which body is held out, so their held-out numbers are
directly comparable:

| | bracketed `c08f09t09` | outside `c06f06t06` | ratio |
|---|---|---|---|
| best over 12 epochs | **0.0764** | 0.9341 | 12x |
| mean, epochs 1-6 | 0.0902 | 1.9707 | 22x |
| mean, epochs 7-12 | 0.1034 | 1.5439 | 15x |
| z-ablation gap | 10-37x | 1.6-5.1x | |

Bracketing is worth 10 to 30 times in error and about 7 times in how much the latent
contributes. This reproduces F4d on an independent dataset with a properly matched control.

More bodies also help the unbracketed case, which two bodies did not:

| | trend, late epochs over early |
|---|---|
| `fold_short`, 2 training bodies, outside | **1.02x** (flat across 41 epochs) |
| `m3d_outside`, 5 training bodies, outside | **0.78x** (22 percent better) |

### F18. The decoder takes the body from the latent, not from the frame

Every body walks the same expert episodes, so at a given timestep two bodies are at the same
point in the gait cycle and the decoder's two inputs can be crossed. `m3d_bracketed` epoch 8,
between two bodies whose commands differ by 28.63 deg (`scripts/diagnostics/swap_pathway.py`):

| frame from | latent from | RMSE vs c10f10t10 | RMSE vs c10f10t06 |
|---|---|---|---|
| c10f10t10 | c10f10t10 | **1.38** | 28.68 |
| **c10f10t10** | **c10f10t06** | 27.35 | **3.48** |
| **c10f10t06** | **c10f10t10** | **11.75** | 23.81 |
| c10f10t06 | c10f10t06 | 28.14 | **2.02** |

Row two is the result: given body A's frame and body B's latent, the decoder emits body B's
commands to within 3.48 deg, almost as accurately as when both inputs agree. The frame it is
looking at makes no difference.

The preference strengthens throughout training. At epoch 20 the crossed case reaches **2.08 deg**
against the latent's body, closer than at epoch 8 and nearly as accurate as when both inputs
agree (1.22 deg), while row three's two columns spread from 4.6 deg apart at epoch 6 to 14.5 deg
apart at epoch 20.

This inverts the architecture's intent. `z` is meant to be a body-independent latent action and
`x_t` the context that says which body; the model uses them the other way round.

### F19. It is keying off 11 percent of the latent while ignoring the whole frame

Decomposing the variance of `z` across the five training bodies at matched gait phases:

| Source | Share |
|---|---|
| where in the gait cycle | **64.1%** |
| which body it is | **11.1%** |
| interaction and residual | 24.8% |

| Distance in `z` | |
|---|---|
| between gait phases of one body | 20.03 |
| between bodies at one phase | 14.98 |

So `z` is doing what it was designed for; the gait dominates it. But a linear probe still
recovers the body from `z` at **0.724** against a 0.200 chance level, and F18 shows that small
component is exactly what the decoder uses.

The frame carries leg lengths in full and the decoder ignores it. A lookup over five
well-separated codes is a cheaper way to reduce the training loss than reading geometry off
256x1408 tokens, and a lookup has no entry for a body that was never assigned a code. Fitting
which mixture of training bodies the model's answer resembles (`scripts/diagnostics/morphology_mix.py`)
shows the same thing from the output side: **0.883 of the weight sits on one training body**
where the best possible mixture spreads to 0.697, and the segment scales its answer implies are
(0.980, 0.975, 0.973) against an actual (0.80, 0.90, 0.90).

By epoch 20 the concentration rises to **0.947**, and it has switched which body it copies, from
`c10f10t10` to `c06f10t10`. The implied scales become (0.615, 0.991, 0.984), which is
`c06f10t10`'s own (0.6, 1.0, 1.0) almost exactly. Switching which entry it returns, rather than
converging on the answer, is what a lookup does. Held-out error over the same span goes 3.57 to
3.88 deg.

This is the mechanism behind every failure above. Coverage helps because it puts a closer code
in the table, not because the model learned to infer morphology.

### F20. The encoder carries morphology and generalises; the decoder does not use it

Everything above assumes the frame carries the body's geometry. This tests it directly, with no
world model involved: fit a regression from the mean-pooled frozen embedding `e_t` to the three
segment scales on the five training bodies, then apply it to a body it has never seen.

Predicted against actual segment scale:

| Body | | coxa | femur | tibia |
|---|---|---|---|---|
| c10f10t10 | train | 0.985 / 1.00 | 0.999 / 1.00 | 0.998 / 1.00 |
| c06f10t10 | train | 0.616 / 0.60 | 0.998 / 1.00 | 0.998 / 1.00 |
| c10f10t06 | train | 0.970 / 1.00 | 0.998 / 1.00 | 0.601 / 0.60 |
| c06f10t06 | train | 0.633 / 0.60 | 0.998 / 1.00 | 0.601 / 0.60 |
| c10f06t06 | train | 0.996 / 1.00 | 0.606 / 0.60 | 0.602 / 0.60 |
| **c08f09t09** | **held out, bracketed** | **0.850 / 0.80** | **0.939 / 0.90** | **0.898 / 0.90** |
| c06f06t06 | held out, outside | 0.872 / 0.60 | 0.683 / 0.60 | 0.710 / 0.60 |

Errors on the bracketed body are **0.050, 0.039 and 0.002**, from ridge regression on a
1408-dimensional average of patch tokens. The frozen encoder places a body it has never seen at
close to its true segment lengths, and a linear readout is enough to get them out.

Against that, the trained Motion Decoder's answer implies segment scales of
**(0.980, 0.975, 0.973)** for the same body (F19), where the truth is (0.80, 0.90, 0.90). A
5.2M-parameter decoder with the frame in front of it is further from the answer than a linear
map with 4,227.

**So the decoder is the failing component, and not for want of information or because the
information is hidden in some hard-to-reach form.** A single matrix on the average of the patch
tokens recovers it.

The capacity pattern from F4b reappears on this much simpler task. An MLP replacing the ridge:

| | training bodies | held-out bodies |
|---|---|---|
| linear | 0.020 / 0.003 / 0.001 | 0.161 / **0.061** / **0.056** |
| MLP | **0.005 / 0.006 / 0.011** | 0.130 / 0.115 / 0.121 |

Better on what it saw, roughly twice as bad on what it did not, on a task that is only
predicting three numbers.

Two secondary readings. The coxa is the worst-predicted segment (0.161 against 0.061 and 0.056),
matching F15: it barely changes the joint commands and it barely changes the image. And the
unbracketed body is recovered far less well -- 0.872 predicted against 0.60 actual. The condition
on the result is exact and computable in advance:

> **The probe predicts a new body to within 0.03 only if that body can be made by mixing the
> bodies it was fitted on. If it cannot, the error jumps to 0.16-0.17.**

Solving for the closest reachable mixture in segment-scale space, with non-negative weights
summing to one:

| body | true scales | closest mixture of the training bodies | distance |
|---|---|---|---|
| **c08f09t09** | (0.80, 0.90, 0.90) | **(0.80, 0.90, 0.90)** | **0** |
| c06f06t06 | (0.60, 0.60, 0.60) | (0.80, 0.80, 0.60) | 0.283 |
| c10f10t06 | (1.00, 1.00, 0.60) | (1.00, 0.80, 0.80) | 0.283 |

`c06f06t06` is unreachable because driving the femur to 0.6 requires all the weight on
`c10f06t06`, which forces the coxa to 1.0; taking weight off it to lower the coxa pushes the femur
back up. `c10f10t06` fails the mirror of this: the only body with a short tibia also has a short
femur, so the two cannot be separated.

**This does not settle whether the encoder carries the information.** A readout fitted on five
points cannot reach past them regardless of what the encoder holds, and the observed pattern --
all three scales pulled toward the middle of the training data -- is what any regressor does
outside its fitting range. Capacity is not the issue: an MLP on the same five bodies is no better
(held-out mean absolute error 0.130/0.115/0.121 against ridge's 0.161/0.061/0.056, and worse on
the bracketed body). What the measurement supports is the conditional above, which is enough to
use it -- see F35.

**Why the decoder cannot do what the probe can.** The probe sees the mean over all 256 patch
tokens. The decoder sees those tokens through cross-attention with `z` as the query, so it only
retrieves what `z` asks for, and `z` is 64 percent gait phase (F19). Morphology is present in the
tokens and never queried. That also explains why removing the body code from `z` did not help
(F21): it forced the decoder onto a channel it has no mechanism to read.

### F21. Removing the body code from the latent moves the channel but does not help

`--lambda_adv 0.1` puts a gradient-reversal classifier on `z`. Against `m3d_bracketed`, which
differs only in that flag, averaged over ten epochs:

| | control | adversarial | change |
|---|---|---|---|
| held-out error | **0.097** | 0.118 | **1.21x worse** |
| z-gap, `zero_z` over held-out | 23.4x | 5.1x | latent used 4.6x less |
| x-gap, `zero_x` over held-out | 10.7x | **21.8x** | **frame used 2.0x more** |
| probe on `z` | 0.724 post hoc | plateau at **0.440** | body code partly removed |

The intervention did what it was designed to do. The decoder moved off the latent and onto the
frame, by a factor of two on the ablation that measures exactly that, and the shift keeps growing
-- x-gap climbs from 11.3x at epoch 1 to 31.9x at epoch 10, reaching 2.46x the control over the
last five epochs. **Transfer never improved.** The harder the decoder is pushed onto the frame,
the more clearly it fails to benefit.

The probe settles at 0.440 from epoch 7 onward, well above the 0.200 chance level and no longer
falling: the ITM and the classifier reach a standoff and the body code stops leaving. Running
longer changes nothing.

Read with F20 this is not ambiguous. Forcing the decoder onto the frame does not help because
the decoder cannot extract morphology from the frame, while a linear probe on the same
embeddings can. The shortcut was a symptom.

Below-chance classifier accuracy is a failure mode worth naming: with five bodies, chance is
0.200, and a 5-epoch smoke run drove the probe to 0.002. Being wrong 99.8 percent of the time
requires information, so it means the latent is rotating the code faster than the classifier
tracks it, not that the code is gone.

### F22. Giving the decoder direct access to the frame makes it use the frame less

F20 showed a ridge probe on mean-pooled `e_t` recovers a held-out body's segment scales to 0.05,
while the decoder does not. `--md_head pooled` gives the decoder that exact view: the mean over
patch tokens, projected and added as a residual straight onto the action. It is initialised to
zero, so training starts bit-identical to the `mlp` decoder, and unlike a concatenated input a
residual on the output cannot be down-weighted away.

An earlier concatenation design was tried first and the fusion layer suppressed the new path --
the frame-ablation gap fell from 12.5x to 7.1x on the smoke set. The residual form was adopted
to remove that escape route.

Against `m3d_bracketed`, differing only in `md_head`, over eleven epochs:

| | control | pooled | change |
|---|---|---|---|
| held-out error | 0.098 | 0.099 | **1.01x, identical** |
| z-gap | 21.1x | 29.6x | latent used 1.4x more |
| **x-gap** | **10.9x** | **1.4x** | **frame used 7.6x less** |

The frame ablation is the result. With the control, removing the frame costs a factor of eleven;
with the direct pooled path available, it costs a factor of 1.4, steady across the last five
epochs. **The decoder was handed the view that works and responded by relying on the frame
almost not at all**, holding transfer exactly level by leaning harder on `z`.

The residual can also be read directly rather than by ablation, and it says the same thing more
sharply. On the full run at epoch 6, where held-out error is at its best:

| | smoke, 975 pairs | full, 9,425 pairs |
|---|---|---|
| residual magnitude | 1.5-1.9 deg | **0.24-0.28 deg** |
| spread across bodies | 1.87 deg | 0.27 deg |
| spread across frames of one body | 2.80 deg | 0.84 deg |
| **ratio between / within** | 0.67 | **0.32** |

It is not a learned constant -- it sits 0.92 to 1.10 deg from what a zeroed frame produces -- but
it varies three times more with where the legs are than with how long they are, and it is
**0.9 percent** of the 28.6 deg that separates two training bodies. More data made it smaller and
the ratio worse, which is the same direction the frame ablation moved.

Crossing the inputs confirms it. At epoch 6, against the control at the same epoch:

| frame from | latent from | control, RMSE vs the latent's body | pooled |
|---|---|---|---|
| A | B | 2.74 | 4.24 |
| B | A | 16.52 | **3.42** |

In the second row the pooled decoder follows the latent **more** faithfully than the control
does, not less: a frame belonging to a body 28.63 deg away moves its answer by 3.42 deg, an
influence of 12 percent.

This closes the access explanation. Five interventions have now been tried:

| Intervention | Result |
|---|---|
| rescale the motion target (F9) | no change |
| shrink the decoder head (F4b) | 1.4 to 2.1x worse |
| remove the body code from `z` (F21) | frame used 2x more, transfer 1.21x worse |
| **give the decoder the pooled view (F22)** | **frame used 7.6x less, transfer level** |
| more training bodies (F16, F17) | **3.1x better, the only one that worked** |

What survives is the objective. `L_motion` asks for the right joint command on bodies the model
can see during training, and a lookup over five body codes in `z` satisfies that at lower cost
than reading leg geometry off pixels -- whatever route to the pixels is provided. Nothing in the
loss ever requires the mapping from appearance to morphology that transfer needs.

### F23. The reconstruction loss barely uses the latent, and it is 99 percent of the gradient

The latent is supposed to be an action because `L_recon` cannot predict the next frame without
it. That premise had never been checked. `L_recon` is an MSE on unnormalised V-JEPA2 embeddings,
so its value carries no meaning on its own and needs baselines.

Held-out body, `m3d_bracketed` epoch 20 and `stage1_100ep_framed_runB` epoch 20:

| horizon | FTM | copy `e_t` | FTM with `z` zeroed | **`z` helps** | FTM vs copy |
|---|---|---|---|---|---|
| 1 | 1.452 | 2.116 | 1.549 | **1.07x** | 1.46x |
| 2 | 1.778 | 2.756 | 1.910 | 1.07x | 1.55x |
| 5 | 2.494 | 3.646 | 2.620 | 1.05x | 1.46x |
| 10 | 3.187 | 4.431 | 3.294 | **1.03x** | 1.39x |

The forward model works -- it beats predicting that the frame does not change by 39 to 55
percent -- but **removing the latent costs it 3 to 7 percent**. Against the Motion Decoder, where
removing the latent costs 2,000 to 3,700 percent, the forward model is barely conditioned on it
at all.

This is not specific to the five-body run: the two-body run gives 1.04x at horizon 1. It is not
fixed by looking further ahead either; the contribution *falls* to 1.03x at horizon 10, because
by then the frame is unpredictable enough that the model falls back on an average, which needs
no action. One step of a small gait is largely predictable from the current pose, and ten steps
are largely not, and neither regime requires knowing what the action was.

Now weight it. With `lambda_recon = lambda_motion = 1.0`, recon sits at 1.6 and motion at 0.01:

> **99 percent of the gradient goes to a loss that does not need `z`, and 1 percent to the loss
> that does.**

So `z` is shaped almost entirely by `L_motion`, on one percent of the training signal, and
`L_motion` is the term that a lookup satisfies (F19). The latent is not a latent action in the
sense the architecture intends; it is whatever compresses enough to predict commands for five
known bodies, and a body code is the cheapest thing that does.

This sits underneath every earlier finding. The decoder keys off a body code in `z` (F18) because
nothing shaped `z` into anything else, and no change to the decoder -- capacity, access, or
adversarial pressure -- can repair a latent the objective never constrained.

Two untested consequences, both one config value and no new data:

| change | question it answers |
|---|---|
| `lambda_motion` raised to ~100 | with a comparable gradient budget, does `L_motion` still settle for a lookup |
| `lambda_recon` set to 0 | does dropping a term that contributes 3 to 7 percent help or hurt the latent |

### F25. Cross-augmentation makes the reconstruction target almost entirely noise

F23 found `L_recon` barely uses the latent while taking 99 percent of the gradient. This is why.

The FTM predicts view 2's next frame from view 2's current frame and a latent computed from
view 1. The two views carry independently sampled crops and brightness jitter, so part of the
target is noise no latent could predict. Measured on 40 frames of one clip, in the same units as
`L_recon`:

| | value |
|---|---|
| **augmentation noise** -- one frame, two views | **8.51** |
| **signal** -- consecutive frames, no augmentation | **1.97** |
| what the FTM is actually asked to close | 8.43 |
| an augmented view against the clean frame | 8.13 |

Splitting the augmentation shows no setting of it recovers the signal:

| augmentation | noise | noise / signal |
|---|---|---|
| crop 85-100% + jitter (current) | 8.42 | 4.39 |
| crop 85-100% only | 8.56 | 4.47 |
| crop 95-100% only | 6.76 | 3.53 |
| **jitter only, no crop** | **4.02** | **2.10** |
| crop 95-100% + jitter | 7.02 | 3.66 |

Crop is the larger term, but tightening it from 85 to 95 percent removes only 21 percent of the
noise, and photometric jitter **alone** -- which moves nothing in the image -- still produces
twice the signal. The frozen encoder is not invariant to brightness and contrast changes, which
is the assumption cross-augmentation rests on. Weakening the augmentation cannot fix this; the
best available setting still leaves noise at twice the signal.

**Noise is 4.33 times the signal**, and the augmentation accounts for 101 percent of the target
the FTM is trained on. The motion `z` could explain is at most 23 percent of `L_recon`; the
measured contribution is 3 to 7 percent.

So the latent is not useless to the forward model, it is buried. Ninety-nine percent of the
gradient goes to a term whose target is dominated by an unpredictable nuisance, and the one
percent that remains is `L_motion`, which a lookup satisfies (F19). **`z` was never under
pressure to become an action.**

Cross-augmentation exists for a reason -- without it the ITM can satisfy `L_recon` by copying
`x_{t+1}` into `z` instead of encoding the transition. The cost of that protection had simply
never been measured. Three ways to keep the protection and recover the signal, none tested:

One option can be ruled out immediately. Making the FTM's target a **clean** frame would make
the shortcut *more* attractive, not less: the protection comes from the target being randomly
augmented, so that no fixed content in `z` can predict it. A deterministic target is exactly what
a copy could hit.

**Dropping cross-augmentation is also ruled out, by F29.** The argument for dropping it was that
the shortcut buys little, since `z` improves `L_recon` by only 3 to 7 percent. That number was
measured while `z` had no job at all. Under `action_lag 1` the decoder can only reach `a_{t+1}`
through `z`, so compressing `e_{t+1}` into `z` is the cheapest way to satisfy `L_motion` -- and
the same compression makes `L_recon` trivial, because the FTM then only has to unpack it. One
shortcut now pays into both terms, which is precisely the degenerate solution cross-augmentation
exists to block. It stays on.

What that leaves for the FTM, none tested:

| change | what it would cost |
|---|---|
| augment in embedding space rather than pixel space | needs designing; the noise becomes controllable |
| augment the FTM's input more weakly than the ITM's | asymmetric, so the ITM still cannot copy |
| accept the FTM as inert and drop it | loses latent rollout, and with it deployment |

The measurement that decides between these is whether `z` under `action_lag 1` becomes a
compressed copy of `e_{t+1}` regardless: a probe from `z` back to the pose in frame `t+1`,
against a probe from `z` to the command difference `a_{t+1} - a_t`. A latent action should carry
the second and not the first.

### F24. Asking the loss for the mapping is what works, inside the range the data covers

**Scope, established after the fact by F28**: everything below holds for a held-out body inside
the range the training bodies span. On `c06f06t06`, whose morphology axis the training set never
demonstrates, the same flag makes transfer **1.35x worse** than the control.

**Scope against the source method.** LAC-WM has no term like this; the shared latent space is
meant to emerge from sharing the modules across embodiments. Its setting probably does not need
one: the shortcut this term closes -- recognise the body, recall its commands -- only pays when
knowing the body tells you the command, and in LAC-WM's data one robot performs thousands of
different manipulations. In ours each body performs exactly one behaviour, so body identity is
nearly the whole answer. This is an addition for the **cross-morphology** regime, not a correction
to the method.

`--lambda_cross` adds one term: decode body A's latent against body B's **frame**, supervised by
body B's command. Every body walks the same expert episodes, so at a given timestep they share
the intent and differ only in geometry, which makes the target well defined with no new data.
Reading the body out of `z` gives the wrong answer here by construction.

Against `m3d_bracketed`, differing only in that flag, over 25 epochs:

| | control | cross | |
|---|---|---|---|
| held-out error, mean epochs 1-10 | 0.0992 | **0.0760** | 23% better |
| held-out error, mean epochs 11+ | 0.0965 | **0.0715** | 26% better |
| best | 0.0764 | **0.0572** | |
| best, in degrees | 3.57 | **2.91** | |
| **x-gap** | 10.7x | **40-69x** | frame used 5x more |
| **z-gap** | 21x | **2.2-3.2x** | latent barely used |
| probe on `z` | 0.724 | 0.306 rising to 0.659 | |

Four independent measurements agree, and this is the first intervention where they all move the
same way rather than trading against each other.

**The swap test inverts completely.** At epoch 8, between two bodies whose commands differ by
28.63 deg:

| frame from | latent from | RMSE vs A | RMSE vs B |
|---|---|---|---|
| A | A | 1.17 | 28.70 |
| **A** | **B** | **1.18** | 28.70 |
| **B** | **A** | 28.49 | **1.49** |
| B | B | 28.45 | 1.57 |

Given body A's frame and body B's latent, the answer is body A's command to within **1.18 deg**,
against 1.17 when both inputs agree -- a difference of 0.01 deg. The latent no longer decides the
body; the frame does. In the control the same crossing produced the latent's body to 2.74 deg.

**It stops copying.** Mixture concentration on a single training body falls from 0.883 (control
epoch 6) and 0.947 (control epoch 20) to **0.540**, below the 0.697 that the best possible
mixture uses. Implied coxa scale moves from 0.615 to **0.784** against a true 0.80. And for the
first time the model beats copy-nearest-body: **2.91 deg against 3.47**.

**It does not degrade with training.** Every earlier run peaked by epoch 8 and then flattened or
worsened (F14). Here epochs 11-25 average **better** than 1-10, 0.0715 against 0.0760.

Two things to report honestly:

- **`z` is now barely used.** z-gap falls to 2.2-3.2x against the control's 21x. The decoder
  reads the body and most of the command from the frame. That is the intended direction taken far
  enough to raise a question about what the latent is still for.
- **The forward model is unchanged.** Removing `z` costs it 1.03x here against 1.07x in the
  control, so F23's finding stands: `L_recon` still does not constrain the latent. The cross term
  fixed the decoder's behaviour without repairing the term that was supposed to shape `z`.

Why this worked where four architectural changes did not: they all altered *how* the decoder
could reach the frame while leaving the question it was asked unchanged, and a lookup answered
that question at lower cost every time. This changes the question. A lookup over the training
bodies is now wrong by construction, and reading geometry from the frame is the only thing that
is right.

### F26. `lambda_cross` purifies the latent rather than emptying it, on bodies it trained on

The concern the low `z` ablation raises is that the latent has been hollowed out: if the decoder
reads everything off the frame, `z` may carry nothing, which would undercut Stage 2. It has not.

| | control epoch 20 | cross epoch 8 | cross epoch 27 |
|---|---|---|---|
| **foot-contact pattern decodable from `z`** | 0.757 | 0.744 | **0.787** |
| body decodable from `z` | 0.707 | 0.638 | 0.665 |
| **variance of `z`: gait phase** | 64.5% | **88.7%** | **83.4%** |
| **variance of `z`: body** | 8.8% | **1.2%** | **1.2%** |
| variance of `z`: interaction | 26.8% | 10.1% | 15.4% |

Eight contact patterns, majority class 0.144.

Behaviour is decoded from `z` as well as before or better. What changed is what else is in there:
the body's share of the variance falls **from 8.8 to 1.2 percent**, a factor of seven, and the
gait's share rises to 83-89 percent. `lambda_cross` achieves what the adversarial head was built
for and failed at (F21) -- and it does so as a by-product of a well-posed task rather than by
fighting the latent, so the gait information survives intact.

That also explains the low `z` ablation. In the control, most of the 21x cost of removing `z` was
the loss of the body code, not the loss of gait; with the body now read from the frame, removing
`z` costs only the gait, and a single frame already says a good deal about where the legs are.
**A z-gap of 2.2x is `z` no longer carrying work that was never its own, not `z` being empty.**

### F27. The fix improves the pose, not the distance

Physical replay of `m3d_cross` epoch 8 against its matched control `m3d_bracketed` epoch 6, both
on the held-out body `c08f09t09`, same three clips, same scene, same physics, open loop.

| | control ep 6 | cross ep 8 |
|---|---|---|
| mean R2 over 18 joints | 0.832 | **0.868** |
| mean RMSE | 3.40 deg | **2.75 deg** |
| duty-factor error against IK, 6 legs x 3 clips | 0.076 | **0.044** |
| commands outside the range this body ever uses | 7.7% | **5.4%** |
| **worst excursion outside that range** | **20.2 deg** | **5.5 deg** |
| forward distance as a fraction of IK | **93%** | 89% |

Per clip, forward distance in metres against an IK ground truth of 0.617, 0.600 and 0.594:
control 0.535, 0.565, 0.569; cross 0.521, 0.556, 0.529.

Two things to state plainly. **The distance did not improve** -- the control walks 93 percent of
the way and the cross model 89, despite 1.2x better joint accuracy. And the earlier result that
transfer covered *less than half* the required distance came from the **two-body** dataset; on
five bodies both runs reach 84 to 96 percent, so **coverage is what fixed the distance, not
`lambda_cross`**.

What `lambda_cross` fixed is the **quality of the pose**. The control commands the legs up to
20.2 deg outside any configuration this body adopts; the cross model's worst excursion is 5.5.
Its tripod index on clip 2 is -0.30 against an IK -0.34, where the control gives -0.09, which is
barely a tripod at all. Commands outside the reachable range are what make legs fold into the
abdomen once error accumulates, so this is the term that matters for closed-loop deployment even
though it does not show up in distance walked.

The 3 to 4 point distance difference is smaller than the spread across clips within the cross
model itself (84 to 93 percent), so it does not establish that the cross model walks less far.
Settling that needs more than three clips.

Figures: `results/wm/action_trace_*_c08f09t09.png`, `results/wm/gait/gait_*.png`, and the
side-by-side videos `results/wm/gait/replay_*.mp4`.

### F28. The model reads apparent size, not the ratios the commands depend on

`c06f06t06` was recorded as the extrapolation test, held out because it sits outside the convex
hull of the training bodies. **In command space it is not outside anything.** It is
`c10f10t10` with every segment scaled by 0.6, and the collector scales the IK foot targets by leg
length, so the two bodies are geometrically similar and their joint trajectories are identical to
**0.07 deg**. The body is physically smaller -- 0.084 m tall against 0.128, covering 0.371 m
against 0.569 -- but the correct answer for it is to copy a training body verbatim.

Neither model, evaluated on it without retraining (both had it held out already):

| predictor | RMSE deg | mean R2 |
|---|---|---|
| copy `c10f10t10` -- the correct answer | **0.07** | -- |
| predict this body's own mean | 12.73 | 0.00 |
| control `m3d_bracketed` epoch 6 | 13.92 | **-2.01** |
| **cross `m3d_cross` epoch 8** | **18.82** | **-4.63** |

Both are worse than the trivial predictor, on 12 of 18 joints with negative R2. The swing joint
(TC) survives at 0.69-0.93; the joints that set leg extension and height (CF, FT) collapse to
RMSE 2.4-3.6x the ground truth's own standard deviation.

**`lambda_cross` makes it worse, and the mixture analysis says exactly why.** The correct implied
scale here is (1.0, 1.0, 1.0), because scaling every segment together leaves the angles unchanged:

| | coxa | femur | tibia |
|---|---|---|---|
| correct in command space | **1.000** | **1.000** | **1.000** |
| control implies | 0.794 | 0.806 | 0.793 |
| **cross implies** | 0.909 | **0.691** | **0.671** |
| true geometry | 0.60 | 0.60 | 0.60 |

The cross model reads off the image that the femur and tibia are short -- and it reads it *well*,
0.691 and 0.671 against a true 0.60 -- then applies the command change that shortening those
segments *relative to the others* would require. Here nothing was relative: everything shrank
together and no command change was needed. The control, which reads the frame less, is wrong by
less. **`lambda_cross` did precisely what it was built to do, and that is what broke it.**

None of the five training bodies scales all three segments together (`c10f10t10 c06f10t10
c10f10t06 c06f10t06 c10f06t06`), so uniform scale is a direction the data never demonstrates. The
model treats each segment's apparent size independently and extrapolates, when the quantity that
actually determines the commands is the **ratio** between them.

This bounds F24. `lambda_cross` improves transfer to a body **inside** the range the training
bodies span, and the mechanism measurements (swap test, x-gap, mixture concentration) are real.
But what it learned is an interpolation across the bodies it saw, not a reading of geometry that
holds outside them -- and reading harder is worse where the data does not cover the direction.

Two consequences. The earlier record of `m3d_outside` as an extrapolation result (held-out MSE
1.71-2.61 against the bracketed body's 0.0992) measured a body whose answer was a training body,
so it understates nothing -- it is a failure on the easiest possible case. And the fix this points
to is a data fix, not a loss fix: include a uniform-scale family so the model can learn that
overall size does not change the commands.

### F29. The task never required dynamics, because the answer was in the input

The Motion Decoder takes `(e_t, z)` -- it never sees `e_{t+1}`. So the only reason `z` should
exist is to carry what the second frame adds. Substituting the second frame given to the ITM
measures whether it adds anything. Held-out body `c08f09t09`, 195 transitions:

| what the ITM is given as `e_{t+1}` | control ep 6 | cross ep 8 |
|---|---|---|
| the real next frame | 3.57 | 2.91 |
| **`e_t` again, no transition at all** | **3.96 (1.11x)** | **3.47 (1.19x)** |
| a real frame from a random other time | 9.65 (2.70x) | 6.10 (2.10x) |
| `e_{t-1}`, the transition backwards | 5.13 (1.44x) | 4.18 (1.44x) |
| the latent zeroed entirely | 19.24 (5.39x) | 6.04 (2.08x) |

**Deleting the transition costs 11 to 19 percent.** The commands move by 2.3 to 3.2 percent of
their own scale. Feeding a *wrong* transition costs more than feeding none -- `e_{t-1}` at 1.44x
and a random other frame at 2.10-2.70x, against the duplicate's 1.11-1.19x -- so the latent is
genuinely sensitive to the second frame. What the duplicate condition establishes is not that the
latent ignores the transition, but that **having the transition is worth only 11 to 19 percent**:
most of what the decoder needs is already in `e_t`.

**Re-measured on the corrected target** (`lag1_ctrl` ep 12 and `lag1_cross` ep 5), the duplicate
frame costs 1.36x and 1.23x against the original 1.11x and 1.19x. The correction roughly triples
the transition's contribution in the control and raises it slightly with the cross term, so it did
what it was designed to do; it simply was not what transfer was short of. `lag1_cross` also gives
the lowest reconstruction of any checkpoint at 2.82 deg.

**The cause is in the collector, not the model.** `sim/collect/collect_ik.py` applies `cmds[t]`, steps the
simulator, and only then captures `frames[t]`. So `frames[t]` is the *result* of `actions[t]`, and
the command that caused `frames[t] -> frames[t+1]` is `actions[t+1]`. Training asked for
`actions[t]` from `(e_t, e_{t+1})`: **the target was already visible in `e_t`**, which the decoder
gets directly. Nothing forced anything through `z`.

This is one finding that explains every earlier one:

| earlier finding | why |
|---|---|
| F23: the FTM does not need `z` (1.03x) | there is nothing `z` must supply |
| F19: the decoder does a lookup | `e_t` already answers the question; `z` only has to name the body |
| F26: `z` is 83-89% gait phase | gait phase is what `e_t` and `e_{t+1}` share |
| F27: pose improves but distance does not | the model was never asked about motion |

**The fix is `action_lag`, now 1 by default.** The decoder is asked for the command that caused
the transition, and since it never sees `e_{t+1}`, that answer can only arrive through `z`. Runs
recorded before 2026-08-09 are read back with `action_lag 0` by `wm.config.from_checkpoint`, so
their numbers are unchanged.

Moving the target one step is not enough by itself if the model can see both frames -- a ridge
probe on `[e_t, e_{t+1}]` gains the same 1.15x whether it is asked for `a_t` or `a_{t+1}`, because
whichever command is wanted, one of the two frames shows it. What makes the difference here is the
**decoder's** input being `e_t` alone. Consecutive commands differ by 3.44 deg on average, and no
function of `e_t` can recover that difference.

### F30. Deleting the forward model costs nothing *for action reconstruction*

`lambda_recon 0` on the five-body dataset, against `m3d_bracketed` differing only in that flag.
Both under the original target (`action_lag 0`), so they are directly comparable.

| | control | `lambda_recon 0` |
|---|---|---|
| `recon` | 1.6, falling | **9.40, flat across all 7 epochs** |
| held-out error | 0.0992 (epochs 1-10) | **0.1025** (epochs 1-7) |
| z-gap | 21x | **24-62x** |
| x-gap | 10.7x | **2.5-6.8x** |

The FTM learns nothing at all -- `recon` does not move from its initial 9.40 -- and the action
reconstruction is unchanged, 0.1025 against 0.0992. This confirms on real data what the 975-pair
smoke run suggested. **In the original pipeline the forward model contributed nothing**, which
follows directly from F29: with the target already visible in `e_t`, there was nothing for it to
supply.

One thing it did do. Removing it moves the decoder **onto `z` and off the frame**: the latent
ablation rises from 21x to 24-62x while the frame ablation falls from 10.7x to 2.5-6.8x. So
`L_recon` was pushing the decoder toward the pixels -- the direction F19 wanted -- while taking
99 percent of the gradient and buying no accuracy for it.

Bounded by F32: this says the forward-prediction term does not help the action decoder. It does
**not** say the forward model learned nothing -- rolled forward on its own output it beats a
frozen world at every horizon out to ten steps.

### F31. One frame nearly determines the command at every horizon

Ridge regression from a single frame's pooled embedding to the joint command at various offsets.
Six clips of `c10f10t10`, fitted on four, tested on two. The commands' own spread is 11.33 deg.

| target | RMSE deg |
|---|---|
| `a_t` | 4.61 |
| `a_{t+1}` | 4.89 |
| `a_{t+2}` | 5.17 |
| `a_{t+4}` | 5.33 |
| `a_{t+8}` | 5.23 |
| `a_{t+16}` | 5.03 |
| **`a_{t+32}`** | **4.45** |

**Predicting 32 frames ahead is as accurate as predicting the current command.** The gait is
periodic with a period near 22 frames, so a distant offset wraps back to a similar phase, and one
frame fixes the phase. The error never rises above 5.33 against a signal of 11.33 at any horizon
tested.

The transition is not worth *nothing*, and it is worth asking what it is worth, because the
obvious objection is that direction of travel cannot be read from a still image -- a leg at
mid-stroke looks the same going forward and going back. Measured directly, on the command
difference, which is pure direction:

| predict | frame `t` only | frames `t` and `t+1` | gain |
|---|---|---|---|
| `a_{t+1}`, the whole command | 5.02 | 4.54 | 1.11x |
| **`a_{t+1} - a_t`, the change** | **2.61** | **2.40** | **1.09x** |
| which feet are swinging | 0.815 | 0.840 | (accuracy, chance 0.5) |

The change has a standard deviation of 4.78 deg, so **one frame already explains 70 percent of its
variance**, and one frame identifies which feet are off the ground at 0.815 against a chance of
0.5. The ambiguity is real for a single leg and disappears for six coordinated ones: the
configuration of all 18 joints fixes the phase, because the other five legs say which half of the
cycle the ambiguous one is in.

So the second frame is worth 9 percent linearly and 19 percent to the trained model, not nothing
and not much. **No choice of target makes the transition necessary**, because what it adds is a
small correction to something already determined. This bounds F29: moving the target one step
forward was the causally correct thing to do and changes nothing, because `a_{t+1}` is nearly as
visible from `e_t` as `a_t` was.

Confirmed directly. `lag1_ctrl` against `m3d_bracketed`, identical but for `action_lag`:

| epoch | | held-out | z-gap |
|---|---|---|---|
| 1 | `action_lag 0` | 0.0925 | 34.0x |
| 1 | `action_lag 1` | 0.1429 | 13.0x |
| 2 | `action_lag 0` | 0.1219 | 24.9x |
| 2 | **`action_lag 1`** | **0.1215** | **11.3x** |

By epoch 2 the held-out error is the same to three decimals and the latent is needed *less*, not
more -- the opposite of what the correction was supposed to produce, and exactly what a
deterministic gait predicts.

**No target on this data can make the transition necessary for the action decoder.** Every insect
clip is forward walking at one speed, and a coordinated hexapod gait is close to a closed loop in
configuration space: knowing where you are on it tells you where you are going.

What this bounds is the **action-decoding** path, and only that. It says the joint command can be
read off one frame, so the latent cannot earn its place there. It says nothing about the forward
model's own competence, which F32 measures separately and finds intact. The right conclusion is
not "the world model is inert" but "**the action decoder was never the place to look for it**".

### F32. The forward model did learn dynamics; it was being measured at the wrong task

F23 and F30 established that the forward-prediction term does not improve action
reconstruction: removing it leaves the held-out error unchanged. That was read as the forward
model having learned nothing. **The reading was wrong, because reconstruction is not what a
forward model is for.**

Closing the forward model on its own output and rolling it forward, with the real latents
supplied by the ITM so the forward model is isolated, on un-augmented frames of the held-out
body, 162 rollouts across three clips:

| steps ahead | forward model | hold `e_t` still | constant velocity | vs hold |
|---|---|---|---|---|
| 1 | 1.53 | 2.11 | 5.78 | **1.38x** |
| 2 | 1.83 | 2.68 | 14.6 | **1.47x** |
| 3 | 2.07 | 3.05 | 27.6 | **1.47x** |
| 5 | 2.54 | 3.57 | 66.0 | **1.41x** |
| 8 | 3.21 | 4.12 | 155.8 | **1.28x** |
| 10 | 3.63 | 4.36 | 236.5 | **1.20x** |

It beats a frozen world at every horizon out to ten steps, and beats constant velocity by two
orders of magnitude at the far end. **The forward model can roll the world forward.** What it
cannot do is make `L_motion` easier -- and nothing ever asked it to, because the joint command
was already recoverable from `e_t` (F29, F31).

Two caveats. Holding `e_t` still is a weak baseline, and 1.2 to 1.5x over it is real but modest;
and the margin decays with horizon, from 1.47x at three steps to 1.20x at ten. The latents fed
in are the true ones, which isolates the forward model correctly but is easier than a rollout
where the latents would also have to be chosen.

**This matches what the source paper does with the module.** In LAC-WM the Motion Decoder is an
auxiliary regulariser; the deployed system predicts future *embeddings*, rolls them out for eight
steps, and selects actions by comparing predicted futures against a subgoal image. The action
decoder is not the output of the system. Measuring the forward model by whether it improves action
reconstruction was measuring it against a task the method never assigns it -- our framing error,
not theirs.

The correct statement is therefore narrower than F30 suggested: **the forward-prediction term is
not needed for action reconstruction on this data, and is not evidence that the world model is
inert.** Evaluating it requires rollout quality and, ultimately, planning success.

### F33. Signal and noise are two separate dials, and only turning both works

F25 measured the forward model's target as 8.4 of augmentation noise against 1.97 of signal, and
concluded that no augmentation setting recovers it -- the best available, photometric jitter with
no crop, still leaves noise at 2.10x the signal. That was correct and incomplete, because it only
turned one dial. **The signal is small because consecutive frames at 20 Hz barely differ**, and
that can be changed independently.

Embedding distance between frames a fixed number of steps apart, same clips, same encoder, against
an augmentation noise floor of 8.39:

| frames apart | real change | signal / noise |
|---|---|---|
| 1 (current) | 2.01 | **0.24x** |
| 3 | 3.09 | 0.37x |
| **5 (the source paper's stride)** | **3.58** | **0.43x** |
| 10 | 4.35 | 0.52x |
| 16 | 4.89 | 0.58x |

Widening the gap nearly doubles the signal at five steps, and then **saturates**: from five to
sixteen steps it gains only 3.58 to 4.89. The gait is periodic, so beyond half a cycle the frames
start returning to a similar pose, and the largest distance available is bounded by the diameter
of the cycle in embedding space.

Neither dial is enough alone. Together they are:

| gap + augmentation | signal | noise | ratio |
|---|---|---|---|
| **1 step, current augmentation** | 2.01 | 8.42 | **0.24x** |
| 5 steps, current augmentation | 3.58 | 8.42 | 0.43x |
| 5 steps, crop 95-100% + jitter | 3.58 | 7.02 | 0.51x |
| **5 steps, photometric jitter only** | 3.58 | 4.02 | **0.89x** |
| 10 steps, jitter only | 4.35 | 4.02 | **1.08x** |
| **16 steps, jitter only** | 4.89 | 4.02 | **1.22x** |

**0.24x to 0.89x is a 3.7x improvement, and at ten steps the signal exceeds the noise for the
first time.** This is the first configuration in which the forward model would be asked to predict
something mostly real.

It also gives the source paper's five-step action chunking a reason we can state: it is not only
about downsampling observation frequency, it is what makes the reconstruction target carry signal.

**The risk it reopens.** Weakening the augmentation restores the copying shortcut, and under
`action_lag 1` that shortcut pays into both loss terms at once. The counter-argument is that at
five to ten steps the latent would have to carry a frame 250 to 500 ms away through a 64-dimensional
bottleneck, which is a much harder thing to copy than a nearly identical neighbouring frame. That
is a hypothesis, not a measurement; the test is a probe from `z` to the future frame's content.

### F34. The clean extrapolation test: the model reproduces the gap in its own training data

`c06f06t06` was a degenerate extrapolation test (F28): its correct commands are a training body's,
to 0.07 deg. The properly designed one holds out the **tibia-short family**. Training on
`c10f10t10 c06f10t10 c10f06t06 c08f09t09`, held out `c10f10t06`.

**In every training body the femur and the tibia carry the same scale:**

| training body | coxa | femur | tibia |
|---|---|---|---|
| c10f10t10 | 1.0 | **1.0** | **1.0** |
| c06f10t10 | 0.6 | **1.0** | **1.0** |
| c10f06t06 | 1.0 | **0.6** | **0.6** |
| c08f09t09 | 0.8 | **0.9** | **0.9** |

The held-out body is the first in which they come apart: femur 1.0, tibia 0.6.

The thresholds were fixed before the run, from baselines computed directly on the commands:
predicting this body's own mean costs **15.99 deg**, the best non-negative mixture of the training
bodies costs **20.31**, copying the nearest costs **20.34**. Below 15.7 would mean the geometry is
being read; above 18 would mean interpolation.

Both bodies of that family were held out. Neither appears in training:

| predictor | `c10f10t06` | `c06f10t06` |
|---|---|---|
| predict this body's own mean | 16.01 | 15.75 |
| best possible mixture of training bodies | 19.58 | 18.43 |
| copy the nearest training body | 20.37 | 19.12 |
| **model, `lambda_cross 0.5`, best checkpoint** | **27.68** | **25.60** |

**What it implies about the geometry is the finding.**

| implied segment scale | coxa | femur | tibia |
|---|---|---|---|
| the truth | 1.00 | **1.00** | **0.60** |
| what the model says | 0.93 | **0.70** | **0.70** |
| the best any mixture could say | 0.99 | **0.62** | **0.62** |

The model ties the femur to the tibia -- and so does the best available mixture, because no
combination of bodies in which the two always move together can separate them. **The model's
error has the same shape as the gap in the data.** All 18 joints score negative R-squared; even
the fore-aft swing joints, which survive on every other held-out body, collapse here.

Two readings that must not be merged. **No interpolation can pass this test**: the best mixture
at 20.31 loses to predicting a constant pose at 15.99. And **the model is a further 1.36x worse
than that ceiling**, so it is not even interpolating optimally.

The matched control without the cross-body term, ten epochs, is **1.11x better**: mean 9.41
against 10.47, best 8.98 against 9.55, and equally flat -- its best epoch is the first one. This
repeats F28's pattern, that outside the range the training bodies span the cross term does not
help and is slightly worse, and it establishes that the failure belongs to the split rather than
to that term. The control's probe on `z` climbs from 0.519 to 0.762 over training, and its latent
ablation from 1.00x to 1.93x: with nothing opposing it the latent returns to being a body code.

**This is compositional generalisation, and it points at a data fix, not a loss or architecture
fix.** Bodies in which the femur and tibia differ can be generated with `sim/scene/make_leg_morphology.py`.

Complete, ten epochs. The held-out error never moves: 10.53 at epoch 1, 10.47 mean, best 9.55 at
epoch 4, drifting back to 10.71 by epoch 10, while training and validation both fall steadily.
The latent ablation averages 1.06x -- removing `z` changes nothing at all on this body.

### F35. The probe predicts which held-out bodies will transfer, before any training

The segment-scale probe (F20) was built to ask whether the frozen encoder carries morphology.
Set against what the trained models then did on three different held-out bodies, it ranks all
three correctly.

| held out | reachable by mixing the training bodies | probe error | trained model, deg | baseline to beat | outcome |
|---|---|---|---|---|---|
| `c08f09t09` | **yes, exactly** | **0.030** | **2.91** | copy nearest 3.47 | **beats it** |
| `c06f06t06` | no, 0.283 away | **0.155** | 18.82 | own mean 12.73 | loses |
| `c10f10t06` | no, 0.283 away | **0.172** | 27.68 | own mean 16.01 | loses |
| `c06f10t06` | no, 0.283 away | **0.172** | 25.60 | own mean 15.75 | loses |

The second column is pure geometry: the distance from the held-out body's segment scales to the
nearest non-negative mixture of the training bodies' scales. It needs no encoder, no model and no
run. Both failures sit the same distance outside, and both fail.

Each probe is fitted only on that run's own training bodies. The separation is a factor of five,
and **nothing else measured here orders the three correctly**: the best-mixture ceiling calls
`c06f06t06` trivially easy at 0.07 deg, and the model fails on it anyway.

The probe costs minutes on CPU with no training. Finding out by training costs roughly four GPU
hours per body. **Fitting it before choosing a train/held-out split says whether the split asks
for a direction the data spans**, and therefore whether the run will answer the intended question
at all.

Stated honestly: this was not designed as a diagnostic. Three measurements made for other reasons
happened to line up, which is stronger evidence than a planned result -- the measurement could not
have been selected to fit the outcome -- but three points establish an ordering, not a threshold.

It also does not need the open question in F20 resolved. Whether a large probe error means the
encoder lacks the information or means five fitting bodies cannot reach that far, the error
predicts the outcome either way.

### F36. The purification does not survive a held-out pair that includes an out-of-range body

F26 measured the latent's variance split on the five **training** bodies, because a balanced
body-by-phase grid needs every body present at every timestep of the shared expert episode. The
two held-out bodies also walk those episodes, so the same grid can be built from them alone. Two
rows is not five, so all ten **pairs** of training bodies give a like-for-like reference at
matching group size (`scripts/diagnostics/z_body_share.py`).

| body's share of the latent's variance | training bodies | training pairs | **held out** |
|---|---|---|---|
| old target, no cross (`m3d_bracketed` ep 6) | 11.3% | 7.2%, range 0.0-10.8 | 6.8% |
| corrected target, no cross (`lag1_ctrl` ep 12) | 10.4% | 6.7%, range 0.0-9.2 | 11.7% |
| old target, with cross (`m3d_cross` ep 8) | 1.2% | 0.8%, range 0.0-1.3 | 10.6% |
| **corrected target, with cross** (`lag1_cross` ep 5) | **0.8%** | **0.5%, range 0.0-0.8** | 8.6% |

The full two-by-two separates the two changes. **The corrected target alone moves the body share
by less than a percentage point** (11.3 to 10.4); **the cross term moves it by an order of
magnitude** (11.3 to 1.2); and the two compound, reaching 0.8 percent with the tightest spread
across the ten pairs. That ordering matches every other measure -- `lag1_cross` is the best run
by held-out error (0.0698) and produces the lowest reconstruction of any checkpoint (2.82 deg).

**The held-out column resists all four configurations**: 6.8, 11.7, 10.6, 8.6, with the plain
control the best of them.

**Scope, and it is narrower than it first appears.** The decomposition needs at least two bodies --
with one there is no between-body variance to decompose -- so the held-out column is computed on
`c08f09t09` **and** `c06f06t06` together. Those two are not the same kind of body: `c08f09t09` is
reachable exactly by mixing the training bodies, `c06f06t06` is 0.283 away from anything reachable.
**The 8.6 percent could be driven entirely by the out-of-range one, and this measurement cannot
separate them.**

The defensible claim is therefore: *the purification does not survive a held-out pair one of whose
members lies outside the training range.* That is consistent with F28, F34 and F35 rather than
being an independent limit on the mechanism, and it is a good deal less surprising.

Separating the two needs **two held-out bodies that are both inside the hull**, and `c08f09t09` is
the only in-hull body in the set -- every other body is a corner of the 0.6/1.0 grid, so any pair
necessarily includes an outside one. Generating intermediate bodies would supply the missing case:
if the body share stays near 1 percent on two in-hull held-out bodies, this finding folds into the
coverage story; if it jumps anyway, it is a genuine separate limit.

**Under the best configuration every training pair lies between 0.0 and 0.8 percent and the
held-out pair is 8.6 -- an order of magnitude above the top of that range.** The group-size
explanation is ruled out by the pairwise column, and "those two bodies are unusually different
from each other" is ruled out by the controls, which sit at 6.8 and 11.7 percent on the *same*
two bodies, inside their own training-pair ranges.

The mechanism follows from what the term actually constrains. `lambda_cross` requires that body
A's latent decoded against body B's frame yields B's command, **for the pairs present in
training**. A body outside that set produces a latent no constraint ever touched, so it is free to
carry whatever the encoder hands it, including apparent size.

This bounds F26, which should be read as measured on training bodies throughout.

**It is the same boundary as F28 and F34, found from a third direction.** Everything the model
learned holds within the span of its training bodies and does not extend past it:

| measurement | inside the training range | outside it |
|---|---|---|
| decoder's joint commands | 2.91 deg, beats copy-nearest | 25.60-27.68 deg, loses to a constant |
| encoder probe on segment scales | 0.030 | 0.155-0.172 |
| **latent's body content** | **1.2%** | **10.6%** |

Consequence for Stage 2: transferring to an unseen embodiment requires body-independence on a body
never trained on, and this is direct evidence that the mechanism does not provide it there.

Caveat: one contrast between two bodies. The pairwise column and the four-configuration sweep make
the comparison sound, but a third held-out body would make it solid.

### F37. What the simulator will and will not let us generate

Generating new morphologies turned out to be bounded, and the bound had not been noticed.

A two-link leg cannot place its foot closer to its own shoulder than `|femur - tibia|` -- the
triangle inequality, since the two links and the shoulder-to-foot distance are the three sides.
Below that the knee would have to fold past straight. The collector pulls every foot target to
**half** the hip-to-foot distance (`tgt = m1 + 0.5 * (expert_foot - m1)`), so the targets sit well
inside, and the closest one across all 30 episodes the dataset uses is **92.5 mm** from the
shoulder, with a spread of 92.5 to 93.7 mm across episodes.

**It is a step, not a gradient**, which is what makes it easy to miss:

| body | `\|femur - tibia\|` | targets inside the dead zone | IK residual |
|---|---|---|---|
| `c10f10t08` | 11.8 mm | 0.0% | 19 mm |
| `c10f10t10` base | 71.0 mm | 0.0% | 32 mm |
| **`c10f10t06`** | **94.6 mm** | **0.3%** | **2 mm** |
| `c10f07t09` | 132.5 mm | 24.2% | **809 mm** |
| `c10f08t10` | 139.6 mm | 27.3% | **349 mm** |

A body 2 mm past the limit loses 0.3 percent of its targets and walks normally; one 40 mm past
loses a quarter and does not. No body is ever *too far* -- 0.0% in every row -- so outer reach was
never the constraint, and leg length does not sort the failures at all (`c10f08t10` at 703 mm
fails while `c10f10t08` at 689 mm works).

`sim/scene/make_leg_morphology.py` now refuses to generate a violating body and prints the margin, with
`--force` to override. Three of the first six bodies attempted were infeasible; the rule would
have caught all three before any collection.

**What it bounds for the project.** The decoupling axis is usable in both directions but not
arbitrarily far. And `c10f10t06`, the held-out body of the tibia-short split, is itself 2 mm past
the limit, so **no feasible body can bracket it exactly** in segment-scale space -- searching all
182 bodies the rule permits on a 0.1 grid leaves a hull distance of 0.0707. What the experiment
turns on is the floor in *command* space, which is unaffected.

### F38. The shared trunk produces a switch, not a shared language

Stage 2 trained for the first time: one ITM, one FTM and one decoder backbone shared across an
18-DOF hexapod and a 12-DOF quadruped, with a per-embodiment output head. No cross-embodiment
term -- the source method has none, and claims the shared latent emerges from weight sharing alone.

Latent variance split by what explains it, using stance fraction as the phase label since the two
embodiments share no episodes (`scripts/diagnostics/z_embodiment_share.py`, `best.pt`, epoch 12):

| | share |
|---|---|
| gait phase | **39.6%** |
| **which embodiment** | **33.0%** |
| interaction | 27.4% |

Against the insect-only figure, where `lambda_cross` holds the *body* share at **0.8-1.2%** on
training bodies. **A third of the latent is a code for which robot this is**, and a linear probe
separates the two at **1.000**.

**Training did pull them together, and the picture shows it.** Between the frozen encoder and
the learned latent (`scripts/diagnostics/cross_embodiment_umap.py`, 2,104 frames):

| | embodiment probe | silhouette | cluster separation |
|---|---|---|---|
| frozen encoder `e_t` | 1.000 | **+0.671** | **4.01x** |
| learned latent `z` | 1.000 | **+0.140** | **0.77x** |

Weight sharing cut the silhouette by a factor of nearly five and brought the cluster means closer
together than the average within-cluster spread. That is a real effect and it is visible in the
projection: the encoder panel shows two clean, far-apart masses, the latent panel two much tighter
ones.

**But the latent panel still shows two separate clusters, and the probe is still 1.000.** A
prediction registered before running it -- that a 0.77x separation would look intermingled in the
projection -- was wrong. UMAP finds and sharpens cluster structure, so a representation that is
only weakly separated in 64 dimensions still resolves into two clean blobs in two. **The picture
overstates the separation relative to the full-space numbers**, which is the opposite of the
failure mode usually warned about.

**What this licenses about method, stated narrowly.** A UMAP cannot tell you how much embodiment
identity a latent retains: here the projection reads as cleanly separated while the silhouette is
+0.140 and the means sit closer than the spread, and elsewhere the reverse could hold. **The
decomposition and the probe have to be reported beside the figure**, because they are the only
things that quantify what the picture gestures at. That is a claim about what this class of
evidence can carry, not about anyone's result: our latent is not theirs, our setting is two
embodiments against three and thousands of transitions against 150,000, and nothing here says
LAC-WM's latent is or is not shared.

This is F36's boundary arriving from a third direction. There, `lambda_cross` drove body identity
out of the latent for bodies it trained on and failed to for bodies it had not seen. Here, with no
equivalent term, embodiment identity simply stays.

**Consequence for Stage 2**: the premise needs a mechanism that actively removes embodiment
identity, not a shared trunk. `lambda_cross` is that mechanism for morphology and does not port,
because it needs frames paired by shared intent and the two embodiments share no episodes (Q0).
That question now has a measured number behind it rather than a prediction.

Caveats, both real. The validation metric for this run is unusable -- `val_fraction 0.1` on 14 B1
clips leaves **67 transitions**, which balanced sampling then repeats to fill half of every
validation batch, which is why training loss reads nine times higher than validation. And the
learning rate reached zero at epoch 6 while validation was still falling at epoch 12, so the
schedule ran out before the model did.

### F39. SUPERSEDED BY F43 -- this measurement was an artefact of contaminated data

> Everything below was measured on `stage2_balanced`, which trained on two robots that collapse
> and rotate rather than walk, with the embodiments at 10.5:1 balanced only by repeating the B1
> data ten times an epoch (F42). Repeated on clean data, **the conclusion reverses**: removing the
> embodiment identity costs *less* than removing random directions, on both seeds. The identity is
> passive leakage, not load-bearing. See F43.
>
> Kept because the reasoning is sound and the reversal is the point: presence and use are
> different questions, and a contaminated dataset can make a passive quantity look functional.

### F39 (as measured, superseded). The embodiment identity in the latent is load-bearing, and it is smeared, not localised

F38 established that 33.0% of the latent's variance is the embodiment and that a linear probe
recovers it at 1.000. Both say identity is *present*; neither says anything *uses* it, and the
difference decides the intervention. If nothing downstream reads it, the 33% is passive leakage
from the frozen encoder and the fix is to remove the model's *ability* to carry it. If something
does, that ability cannot be removed without first supplying the information elsewhere.

The prior was passive, on structural grounds: the decoder's output head is *selected* by
embodiment, so identity is handed to it for free, and the FTM sees `x_t`, which is a picture of
the robot. Neither has to ask `z`. **That prior was wrong.**

Tested without training anything (`scripts/diagnostics/z_identity_ablation.py`, `stage2_balanced/best.pt`
epoch 12, 2,104 latents). Identity is linearly decodable, so peel the directions carrying it out
of the 64-D latent, re-run the decoder on the crippled latent, and compare against the cost of
deleting the same number of *random* orthogonal directions -- the floor for losing capacity with
nothing meaningful removed:

| latent | B1 | hexapod | mean vs intact |
|---|---|---|---|
| intact | 3.42 | 3.39 | 1.00x |
| **identity removed, 8 directions** | 4.03 | **7.45** | **1.69x** |
| random 8 directions removed | 3.79 | 4.13 | 1.16x |
| `z` zeroed entirely | 29.34 | 18.82 | 7.07x |

Degrees RMSE per joint, de-standardised per embodiment before converting, since MSE in
standardised units is not comparable across two action spaces with different per-joint spreads.

**1.69x against a 1.16x control: the identity is used.** And the cost is almost entirely one-sided
-- the hexapod pays **2.20x** against random's 1.22x, while the B1 barely moves (1.18x against
1.11x). The decoder reads "this is the hexapod" out of `z` despite its own head already encoding
that fact.

**Second result, which changes what the fix can look like.** Removing directions one at a time,
refitting the probe after each, the embodiment does not clear:

```
1.000 -> 0.941 -> 0.895 -> 0.865 -> 0.849 -> 0.842 -> 0.836 -> 0.819 -> 0.806
```

Eight directions gone and identity still decodes at **0.806 against a 0.500 chance level**. It is
not a subspace that can be excised; it is distributed across the latent. This retrospectively
explains the adversary's failure mode: a 5-epoch run drove the body probe to 0.009 against a 0.200
chance level, and below-chance means the code is being *rotated* faster than the classifier
tracks, not dropped. With the signal smeared, gradient reversal has nothing local to delete, so
scrambling is the only move available to it.

**Consequence.** The two interventions are not alternatives, and their order is forced. A side
channel alone relieves the *need* but leaves the *ability*, and nothing makes the model stop.
An adversary alone removes the ability while something still depends on the information, which is
what breaks. `cfg.ftm_embodiment_channel` supplies identity to the FTM as a separate latent token
so `z` is free to stop carrying it; only with that in place does pressure on `z` cost nothing.

The measurement to repeat after training with it is F38's 33.0%.

### F40. The vision path runs at 10.5 Hz, which puts it on the biological side of the trade-off

Asked which architecture the system addresses -- "few sensors and fast" like a robot at 50 Hz, or
"many sensors and slow" like an animal at 200 ms -- the answer needed a measured number rather
than an estimate.

V-JEPA2 ViT-g/16, 1B parameters and frozen, encoding a single frame with the frame-duplication
trick on a 2080 Ti: **94.9 ms, or 10.5 Hz**, averaged over 20 calls after warm-up. The ITM,
forward model and decoder together are a rounding error beside it.

| | sensors | loop time |
|---|---|---|
| biological | ~10^6 nerve endings | 200 ms |
| robot control | few | 20 ms, 50 Hz |
| **this pipeline's vision path** | one camera | **94.9 ms, 10.5 Hz** |

**We are nearer the animal than the robot, and that is not a defect to optimise away.** Vision is
not here for bandwidth or latency; it is here because it is the only channel in which an 18-DOF
hexapod and a 12-DOF quadruped are described in the same coordinates. They share no joint space,
no sensor correspondence and no midpoint (F38's premise). That property costs 95 ms per frame.

Consequence for deployment: the architecture is **two-rate by necessity**. Perception plans at
~10 Hz and control stabilises at 50 Hz, which is also the answer to the objection that
proprioception cannot be removed -- it is not being removed, it is doing the job vision cannot.
The cross-embodiment stance-fraction probe puts a number on which job that is: within an
embodiment the frozen encoder reads contact at 0.82-0.89x of the target's spread, across
embodiments at 1.04-1.16x, at or past the point where looking at the image is worth nothing, and
that is with pooling and appearance both controlled (F41).

### F41. Most of the cross-embodiment probe failure was the pooling, not the encoder

The frozen-encoder cross probe (F37) fitted a stance-fraction readout on one embodiment and applied
it to the other, on **mean-pooled** patch tokens, and reported 4.72x and 3.00x of the target's own
spread -- three to five times worse than predicting the average. That number is an artefact of the
reduction, and the honest figure is much smaller.

The encoder emits 256 patch tokens per frame, which have to be collapsed to one vector. Three
reductions, identical in every other respect:

| fitted -> tested | mean | bands (4 horizontal) | **max** |
|---|---|---|---|
| insect -> insect | 0.88x | 0.84x | 0.92x |
| B1 -> B1 | 0.89x | 0.89x | 0.92x |
| **insect -> B1** | **4.72x** | 2.74x | **1.32x** |
| **B1 -> insect** | **3.00x** | 2.35x | **1.06x** |
| embodiment cluster separation | 3.94x | 3.58x | **1.83x** |

The diagonals barely move. The cross cells fall by a factor of three to four. **Max uses exactly as
many features as mean, 1,408, so this is not capacity.**

**The mechanism is the last row.** Mean-pooled, the two embodiments' frames sit 3.94x apart
relative to their own spread -- a large constant offset, not a difference in how contact is
encoded. A ridge fitted on insects absorbs that offset into its intercept; applied to a B1 every
prediction shifts by it, and the error explodes. Max-pooling takes the strongest response per
dimension, which discards the offset, and separation drops to 1.83x with the cross error following.

**What survives.** Every cross cell is still at or above **1.00x**, so a readout fitted on one
embodiment and applied to the other is no better than ignoring the image. The frozen encoder still
hands over nothing usable across embodiments, and F38's premise stands. What does not survive is
the *magnitude*: "actively misleading, three to five times worse than guessing" was the pooling.

**Method consequence, and it generalises.** Mean-pooling is correct for quantities spread across
the frame -- segment scale is recovered at 0.050 from pooled tokens (F20), embodiment identity at
probe 1.000 -- and wrong for quantities confined to a few patches, since which feet are loaded
occupies perhaps 6-12 of 256. **The reduction has to be chosen for the quantity and reported with
the result.** A single-reduction number for a local quantity is not a property of the encoder.

**The second nuisance is appearance, and removing it makes the number stable.** The insect renders
orange and occupies about a quarter of the frame; the B1 renders grey and about three quarters.
Neither is behaviour. Standardising each embodiment by its own mean and spread removes both, using
only which dataset a frame came from and never the target -- standard unsupervised domain
adaptation. Cross cells, raw against controlled:

| cross cells | mean | bands | max |
|---|---|---|---|
| raw | **4.72x / 3.00x** | 2.74x / 2.35x | 1.32x / 1.06x |
| **appearance controlled** | 1.57x / 1.07x | **1.16x / 1.04x** | 1.22x / 1.02x |

Raw, the answer swings four-fold on the choice of pooling alone. Controlled, every reduction agrees
within 1.02-1.57x. **Reported at band-pooled and controlled, 1.16x and 1.04x**, because that is the
setting where the number is a property of the encoder rather than of our choices, and because the
best-transferring readout is the fair test of whether the information is present at all.

The claim is now its strong form: with colour, apparent size and pooling all controlled, the frozen
encoder still gives nothing usable across embodiments, and that cannot be explained away by the two
robots looking different.

Limit worth stating: per-embodiment standardisation needs a batch of the new robot's frames, so
this is domain adaptation rather than zero-shot. The setting already requires that, since a new
embodiment needs a new output head fitted on some of its data regardless.

**Where this does not transfer.** The same correction cannot be applied to F38's 33% embodiment
share in `z`: that quantity *is* the between-group mean difference, so centring per embodiment
would zero it by construction. Controlling appearance there means equalising it at render time --
same material colour, camera distance matched so both robots occupy a similar fraction of the
frame -- which is also what the 4-leg insect would need to separate leg count from appearance.

An untried third option: standardise the encoder features per embodiment **during training**, so
the constant appearance offset never reaches the ITM and `z` is never offered it as a code. Uses
only the embodiment label, which training has, and costs nothing the setting was not already
paying.

Reproduce all six cells with `--features {mean,bands,max}` and `--normalize` on
`scripts/diagnostics/cross_embodiment_probe.py`, from one cached encoder pass.

### F41b. Per-leg contact: the encoder sees a loaded leg clearly and describes it differently for each robot

The stance-fraction probe (F41) turned out to be unfixable, for a reason that only became clear
later: the B1 trots and the insect walks a wave, so the two have **no shared gait phase**, and
stance fraction is dominated by 0.5 on the B1 (86.6% of its frames) while the insect spreads
across 2/6, 3/6 and 4/6. The phase label predicted the embodiment.

Per-leg contact avoids all of it. "Is *this* leg loaded" needs no shared phase and no shared
aggregate, is binary and near-balanced on both robots so chance is exactly 0.500, and the four
corner legs correspond anatomically. The middle legs have no counterpart, which is the honest
asymmetry between a hexapod and a quadruped rather than a defect.

Band-pooled patch tokens, each embodiment standardised by its own statistics, split by clip,
balanced accuracy (`scripts/diagnostics/leg_contact_probe.py`):

| leg | insect -> insect | b1 -> b1 | insect -> b1 | b1 -> insect |
|---|---|---|---|---|
| left front | 0.848 | 0.931 | **0.487** | **0.469** |
| left hind | 0.774 | 0.959 | 0.590 | 0.608 |
| right front | 0.792 | 0.942 | **0.466** | **0.472** |
| right hind | 0.812 | 0.934 | 0.582 | 0.638 |
| **mean** | **0.806** | **0.941** | **0.531** | **0.547** |

**The diagonal is a genuine ceiling this time** -- 0.806 and 0.941 -- so the cross cells at 0.531
and 0.547 cannot be dismissed as "nothing there to predict", which is exactly what sank the
stance-fraction version. The encoder reads a loaded leg well within an embodiment and transfers
almost nothing across.

**And the pattern is specific.** Front legs transfer *below* chance, hind legs above it. That
tracks how differently the legs are used: the insect's front leg has duty **0.309** -- mostly in
the air, partly a feeler -- against the B1's **0.578**, while the hind legs are much closer, 0.591
against 0.500. Legs used similarly transfer a little; legs used differently transfer worse than
chance. That points at **behaviour**, not appearance, as what the frozen encoder fails to share.

**Run on the learned latent instead of the encoder, this becomes the most direct test of the
source method's central claim** -- that a shared latent emerges from weight sharing alone. Same
probe, same legs, `z` in place of `e_t`:

| | insect -> insect | b1 -> b1 | **insect -> b1** | **b1 -> insect** |
|---|---|---|---|---|
| frozen encoder `e_t` | 0.806 | 0.941 | 0.531 | 0.547 |
| `z`, `stage2_clean` | 0.811 | **0.986** | **0.373** | **0.401** |
| `z`, `adv_warm10` | 0.798 | 0.969 | **0.490** | **0.500** |
| `z`, centred | 0.851 | 0.978 | 0.559 | 0.468 |

**Training makes the two embodiments less comparable, not more.** The frozen encoder sits barely
above chance at 0.531/0.547; the trained latent falls to 0.373/0.401, which is *below* chance --
a readout fitted on one embodiment is systematically wrong on the other, so the latent encodes
"this leg is loaded" along an axis pointing the opposite way for each robot.

Capacity does not explain it. `z` is 64 features against `e_t`'s 5,632, so a capacity limit would
hurt everywhere; instead `z` is **better** within an embodiment (0.986 on the B1) and worse across.
Weight sharing produced a sharper per-robot code and a poorer shared one.

**The adversary repairs precisely this**, 0.373 -> 0.490 and 0.401 -> 0.500, which is the first
measurement where it does something specific and good rather than merely harmless. But **nothing
gets above chance**: the best any intervention achieves is back to where the frozen encoder already
was. No version of this pipeline makes a hexapod and a quadruped more comparable than V-JEPA2 left
them.

This is a more direct test than the UMAP or the variance share, and unlike both it needs no shared
gait phase.

Solver note: `LogisticRegression(max_iter=3000)` on 5,632 band-pooled features against ~1,900
samples does not converge in hours. `RidgeClassifierCV` has a closed form and picks its own
penalty, which is what n_features > n_samples calls for.

### F42. Two bodies in the dataset do not walk, and two more crab sideways. UNRESOLVED

Found by asking why the learned latent splits the hexapod frames into two clean groups. It does,
perfectly, by whether the femur is longer than the tibia -- and chasing that down turned up a data
problem that touches results already reported.

**The nine bodies in `data/ik_walk_8body`, measured on their own recorded head trajectories:**

| body | ratio f/t | dead zone | forward (m) | sideways drift (m) | verdict |
|---|---|---|---|---|---|
| c06f06t06 | 0.83 | 42.6 mm | +0.371 | 0.088 | fine |
| c10f06t06 | 0.83 | 42.6 mm | +0.664 | 0.062 | fine |
| c08f09t09 | 0.83 | 63.9 mm | +0.596 | 0.117 | fine |
| c06f10t10 | 0.83 | 71.0 mm | +0.579 | 0.149 | fine |
| c10f10t10 | 0.83 | 71.0 mm | +0.571 | 0.174 | fine |
| c06f10t06 | 1.38 | 94.6 mm | +0.368 | **0.353** | crabs sideways |
| c10f10t06 | 1.38 | 94.6 mm | +0.480 | **0.380** | crabs sideways |
| c06f06t10 | 0.50 | **208.2 mm** | **+0.057** | 0.328 | **does not walk** |
| c10f06t10 | 0.50 | **208.2 mm** | **-0.369** | 0.281 | **walks backwards** |

**The dead zone is the cause, not the ratio.** A two-link leg reaches nothing closer to its
shoulder than `|femur - tibia|`, and the closest commanded target sits at 92.5 mm. The Track A
bracket bodies settle which of the two matters: at ratios 1.04, 1.07 and 1.10 with dead zones of
11.8, 18.9 and 26.0 mm they walk normally, forward 0.33-0.39 m on drifts of 0.08-0.16 m. Ratio
alone is harmless; ratio is only dangerous because it drives the dead zone.

**`c10f06t10` got in through a bug already recorded**: the walk check used
`norm(h[-1,:2] - h[0,:2])`, which is unsigned, and reads 0.46 m for a body reversing away from the
start. The fix is to check the forward component and the lateral drift separately.

**And these two bodies were already known to be bad.** PROGRESS 18.2 records them walking
successfully in only 20-21 of 30 clips, and they were "cut" -- by leaving them out of
`train_morphs` in the Stage 1 runs. The clips stayed in the directory. Stage 2 takes a directory
and globs it, so **a convention enforced by naming bodies broke the moment a run stopped naming
them.** That is the real mechanism: not that nobody noticed, but that the exclusion lived in the
callers rather than in the data.

**Who is affected.**

| run | bodies | contamination |
|---|---|---|
| `m3d_cross`, `m3d_bracketed`, `lag1_*` | 5 training | **2 of 5 are the 94.6 mm crab-walkers** |
| `tib_cross` | 4 training, held out `c10f10t06` | training clean; **the held-out body is a crab-walker** |
| `bracket_cross` | 7 training | one crab-walker, the rest sound |
| **every Stage 2 run** | globs the whole directory | **both non-walking bodies included, ~22% of hexapod clips** |

Stage 2 passes `hexapod=data/ik_walk_8body` and `embodiment_split` globs `*.npz`, so nothing
selects bodies at all.

**What this does not invalidate.** Comparisons between Stage 2 runs. `stage2_balanced`,
`stage2_sidechannel` and `stage2_centered` share identical data, so the side-channel and centring
contrasts remain clean.

**What it does.** Every absolute Stage 2 number, including F38's 33.0% embodiment share, comes
from a model that spent about a fifth of its hexapod gradient on a robot that does not walk.

**And it bounds F33 / slide 8, measured rather than assumed.** `tib_cross` holds out `c10f10t06`,
which veers 0.380 m sideways against 0.06-0.17 m for every body it trained on. Scoring the *same
checkpoint* on three further held-out bodies that walk cleanly separates the geometry gap from the
gait (`scripts/diagnostics/score_body.py`):

| held out | ratio | dead zone | deg/joint | R^2 | frame ablation |
|---|---|---|---|---|---|
| **c10f10t06** | 1.38 | 94.6 mm | **27.76** | **-3.16** | frame helps |
| c10f10t08 | 1.04 | 11.8 mm | 13.49 | -1.07 | -1.6%, noise |
| c10f09t07 | 1.07 | 18.9 mm | 12.52 | **-0.42** | frame helps |
| c10f08t06 | 1.10 | 26.0 mm | 11.39 | **-0.47** | frame helps |

**R^2 is negative on all four**, so the femur/tibia gap is real and not an artefact of one body:
on every unseen ratio the model does worse than memorising that body's own average posture. **But
the headline body overstates it 3 to 7 times** -- 11-13 deg rather than 27.8, R^2 of -0.4 to -1.1
rather than -3.2. Against a command spread of 11.7 deg per joint, the honest statement is
"comparable to the signal", not "four times it".

**Method note that cost a run.** The first attempt at this retrained with a different held-out
body (`tib_sound`), which changed the weights as well as the test -- 18 clips per body instead of
30 -- and produced a "the frame is actively harmful, 1.34x" signal that the original checkpoint
does not show. One checkpoint against several held-out bodies isolates the body; retraining does
not.

**With the condition that makes it safe:** this works only because `tib_cross`'s *training* set is
already sound -- four bodies, all ratio 0.83, dead zones 42-71 mm. Where the training set is what
needs fixing, there is no way around a rerun: `m3d_cross` and `m3d_bracketed` carry two veering
bodies out of five, and every Stage 2 run carried two that do not walk at all. Re-scoring answers
"which body is being tested"; only retraining answers "which bodies it learned from".

**Biological note, raised by the user and correct.** The base insect is femur 342.9 mm against
tibia 413.9 mm, ratio 0.83. Ratio 1.38 inverts the proportion, which no stick insect has; femur
longer than tibia occurs in insects, but in orthopteran jumping legs rather than in Phasmatodea.
Those two bodies are not stick insects with short tibias.

**The admission rule, agreed and now in code.** A body is usable when `|femur - tibia|` stays
under the closest commanded target, 92.5 mm. Ratio is not the criterion:

| body | ratio | dead zone | forward | sideways | |
|---|---|---|---|---|---|
| c10f10t10 | 0.83 | 71.0 mm | +0.592 | +0.058 | reference |
| c10f10t08 | **1.04** | 11.8 mm | +0.467 | +0.117 | femur longer, walks straight |
| c10f08t06 | **1.10** | 26.0 mm | +0.408 | +0.051 | femur longer, straighter than the reference |
| c10f10t06 | 1.38 | **94.6 mm** | +0.396 | **-0.397** | veers off, yaws progressively |
| c06f06t10, c10f06t10 | 0.50 | **208.2 mm** | ~0 | tumbles | collapses within ~12 frames |

Femur longer than tibia is fine at 1.04 and 1.10. At 1.38 the body yaws steadily until it is
0.40 m off course, because a subset of its foot targets fall inside the dead zone.

**The contact labels are sound, checked because so much rests on them.** Stance fraction is the
shared phase label in the cross-embodiment probe and the phase term in the latent decomposition.
The force distribution is sharply bimodal -- a swing mode near 0.1 N, a stance mode near 6-10 N --
and the 0.27 N cut sits in the empty valley with **1.8% of samples within +/-0.07 N** of it. Not a
thresholding artefact. `scripts/dataset/plot_gait_quality.py`.

**The expert gait is a real stick insect's, so it is a variable wave, not a tripod.** Per-leg
contact periods on the reference come out at 6, 22, 18, 4, 9 and 20 frames, front-leg duty runs
0.29-0.44 against 0.60-0.78 for the middle legs, and the tripod separation is near zero for every
body. That is the expected reading for an animal and not a defect; every body inherits the same
variability from the same targets, so comparisons stay fair. Worth one narrow note: `c10f10t06` is
the only body with a **negative** separation, its legs grouping worse than arbitrary triples --
one more mark against a body already known to veer.

**Still to do, and nothing below is done yet:**

1. Drop the two non-walking bodies from every Stage 2 source. `sources` selects a directory, so
   this needs either body filtering in `embodiment_split` or a curated directory.
2. Rerun Stage 2 once on clean bodies and re-measure the 33.0%.
3. Fix the walk check to test forward displacement and lateral drift separately, signed.
4. Decide whether the 94.6 mm crab-walkers stay. They walk, so they are usable, but they are the
   bodies the extrapolation claim rests on and they are outside the animal's proportions.
5. Reach ratio diversity through the **coxa**, not by rescaling the foot trajectory. The coxa
   positions the shoulder without entering `|femur - tibia|`, and it is behaviourally almost free:
   coxa agrees with the gait split at ARI +0.038, and c10f10t10 against c06f10t10 -- a 40% coxa
   change -- have contact patterns agreeing at 0.984. Rescaling the trajectory per body would
   instead break `lambda_cross`, which is well defined only because every body walks identical
   expert episodes.

### F43. Stage 2 on clean data: transfer works, and the embodiment identity is passive

The first Stage 2 whose data is defensible. Every training body walks, the embodiments are
balanced by data rather than by repetition, validation is stratified across bodies, and a hexapod
body is withheld. Two runs, identical but for the seed, 60 epochs, converged (val moved 0.0001 per
epoch over the last six).

    hexapod   4 bodies x 4 clips x 65   = 1,040 pairs
    b1        2 policies x 6 clips      = 1,003 pairs     ratio 1.04:1
    held out  c08f09t09                                   never trained on
    excluded  c06f06t10, c10f06t10                        collapse and rotate (F42)
    withheld  c06f10t06, c10f10t06                        veer 0.35-0.40 m off course

**Stage 2 has a generalisation test for the first time.** With two embodiments neither can hold the
other out, so a hexapod body was withheld instead (`scripts/diagnostics/score_body.py`, same weights, one
unseen body):

| | seed 0 | seed 1 |
|---|---|---|
| deg per joint | 3.85 | 3.43 |
| **R^2 against the body's own mean** | **+0.87** | **+0.90** |
| latent zeroed | 0.365 | 0.444 |
| frame zeroed | 0.193 | 0.395 |

**Positive on both seeds**, where every Stage 1 held-out body scored -0.42 to -3.16. Both inputs
are used: zeroing the latent costs 5-7x, zeroing the frame 2.6-6.7x. Stage 1's `m3d_cross` scores
2.91 deg on the same body, so **learning a quadruped alongside costs about 30% of hexapod accuracy
and does not break it**.

Two caveats. `c08f09t09` is coxa 0.8, femur 0.9, tibia 0.9 -- inside the training range on every
axis, so this is interpolation, not the extrapolation F33 fails at. It was chosen because Stage 1
held out the same body, which is what makes the stages comparable. And the `zero_x` figures differ
2x between seeds, so every input-ablation ratio we report -- including the older z-gap and x-gap --
deserves that scepticism.

**The embodiment identity in `z` is passive, reversing F39.**

| | identity removed | random control, same count |
|---|---|---|
| contaminated (F39) | 1.69x | 1.16x |
| **clean, seed 0** | **1.03x** | 1.18x |
| **clean, seed 1** | **1.04x** | 1.14x |

Removing it costs *less* than removing arbitrary directions, on both seeds. Nothing downstream
reads it. `z` itself is heavily used -- zeroing it costs 7.6-8.3x -- so the latent does real work
and only its identity component is inert.

This explains the side channel's null result: `ftm_embodiment_channel` was built to relieve a
pressure that does not exist.

**The variance decomposition is not a usable number.**

| | seed 0 | seed 1 | spread |
|---|---|---|---|
| gait phase | 44.9% | 61.2% | 16.3 pts |
| which embodiment | 12.0% | 6.7% | **5.3 pts, nearly 2x** |
| interaction | 43.2% | 32.1% | 11.1 pts |
| **probe** | **0.994** | **0.992** | 0.002 |
| cluster separation | 0.39x | 0.24x | 0.15 |

`two_way` balances its grid by subsampling every cell to the smallest, which holds six latents.
Two seeds of one config disagree by a factor of two.

**And the phase axis is worse than under-sampled -- it barely exists.** Stance fraction takes only
**8 distinct values** across both embodiments (0.167, 0.25, 0.333, 0.5, 0.667, 0.75, 0.833, 1.0),
dominated by 0.5, so the quantile edges collapse onto each other:

| requested bins | edges | bins actually occupied |
|---|---|---|
| 3 | [0.5, 0.5] | **2** |
| 4 | [0.5, 0.5, 0.5] | **2**, giving numbers identical to 3 bins |
| 6 | [0.5, 0.5, 0.5, 0.5, 0.667] | **3** |

So the grid was never 2 x 6 x 6 = 72 cells; it was 24 to 36. The embodiment share reads **32.0% at
three bins and 12.0% at six** on the same checkpoint -- a 2.7x swing from a parameter meant to be
cosmetic.

**And the deeper fault is that the two axes are not independent.** Splitting the distribution by
embodiment shows where the pile-up at 0.5 comes from:

| | frames at 0.5 | distinct values | std |
|---|---|---|---|
| hexapod | 41.8% | 5 | 0.147 |
| **B1** | **86.6%** | 4 | 0.094 |

The B1 is a trot -- diagonal pairs, two feet down 86.6% of the time, almost no other state. The
insect is an animal's wave: 22% at 2/6, 42% at 3/6, 29% at 4/6. With more B1 frames as well
(1,143 against 792), the B1 alone supplies **75% of all frames sitting at 0.5**.

Which means the phase label predicts the embodiment: at 0.5 a frame is probably B1, away from 0.5
it is probably the insect. A two-way decomposition into "row = embodiment, column = phase" assumes
those factors are separable, and here **they are half the same variable**. That, not the cell
count, is why the answer moves with the bin parameter.

**F38's headline 33.0% came from this measurement and should be withdrawn**, not merely caveated.

Stage 1's `z_body_share` is unaffected: insect bodies walk identical expert episodes, so at a given
timestep every body is in the same phase by construction and the grid can use the timestep
directly. Stage 2 had to invent a shared phase label, and the general point is that **two robots
with genuinely different gaits may not have a shared phase to measure at all** -- a limit of the
setting, not of the tool.

The claim belongs on the probe and the ablation, which reproduce to three decimals:

> The embodiment is fully decodable from the latent at 0.99, and nothing uses it -- removing it
> costs less than removing random directions.

Untried: `--bins 3` doubles the latents per cell, which would say whether the decomposition is
under-sampled rather than unusable.

**Method fix this forced.** Five diagnostics each carried a hardcoded `INSECT_BODIES`, in three
different versions, and three of them silently scored a model on bodies it had never seen --
`z_identity_ablation` read 15.99 deg where the trained bodies read 1.45.
`wm/evaluate.py:training_bodies(cfg)` now derives the list from the checkpoint's own config.

### F43b. The ablations replayed as behaviour, and what the per-joint error was hiding

Ajan Blink asked twice for the same thing: never present a number without the gait beside it, and
run ablations that show the *behaviour* degrade rather than the metric. Every ablation up to here
was numeric. These are the same ablations driven through CoppeliaSim on the held-out body
`c08f09t09`, clip ep101, 65 steps (`wm/predict_actions.py --ablate`, then
`sim/render/render_wm_prediction.py`).

| | forward (m) | heading | mean feet down | RMSE deg | out-of-range commands |
|---|---|---|---|---|---|
| ground truth (IK) | **+0.592** | +13.8 deg | 3.00 of 6 | -- | 0.0% |
| intact | +0.374 | +14.0 deg | 2.74 | 3.98 | 7.9%, worst 15.4 deg |
| identity removed (8 dirs) | +0.345 | +24.8 deg | 2.75 | 4.18 | 7.5%, worst 15.0 deg |
| frame zeroed | +0.297 | +26.8 deg | 3.09 | 7.26 | 3.2%, worst 17.6 deg |
| **latent zeroed** | **+0.100** | **+59.6 deg** | 3.12 | 9.68 | 6.7%, worst 8.9 deg |

**Zeroing the latent stops the robot walking.** 0.100 m against 0.592, turning 59.6 degrees off
course, with the front-right foot down **9%** of the time against 51% for ground truth -- it drags
a leg and spins. That is the 7.63x from the ablation table made visible.

**And the intact model walks only 63% as far as the reference** while scoring 3.98 deg per joint
and R^2 +0.87. Against a command spread of 11.7 deg that error reads small; behaviourally it is a
robot covering two-thirds of the ground, with 7.9% of its commands outside the range this body
ever uses. **This is exactly the gap Ajan Blink named: the metric and the behaviour do not tell the
same story.** Reconstruction R^2 should never again be reported for this pipeline without the
replay beside it.

**Identity removal is nearly free behaviourally, with a caveat.** Forward distance 0.345 against
0.374 supports the 1.03-1.05x from two independent code paths. But heading degrades, 24.8 against
14.0 degrees. One clip, and heading is the noisiest of these measures, so it is not evidence
against the ablation result -- but "invisible" overstates it.

Method note: the replay path reproduced the identity ablation independently -- 1.05x here against
1.03x in `scripts/diagnostics/z_identity_ablation.py` -- which is worth more than either number alone.

Two bugs surfaced building this, both of the kind that returns plausible numbers instead of
failing. Rewriting `load_model` dropped `itm.load_state_dict`, so the ITM ran randomly initialised:
14.51 deg, and `zero_z` scored *better* than the intact model, since a zero latent is less harmful
than a random one. And `load_clip` is hexapod-specific -- B1 clips key their commands as `action`
-- so the identity basis crashed on the B1 side.

### F44. A third embodiment: Stage 2 features transfer to a 4-leg insect, few-shot

The strongest result in the project, and the first that tests the thesis claim rather than
inspecting the latent.

**The body.** The stick insect with its middle legs removed, `ML,MR` -- a 4-leg insect. Built by
ghost-removing the legs at runtime from the base scene, so no new scene file exists or is needed,
and driven by the *unchanged* six-leg IK gait, so no policy was trained. The four remaining legs
are geometrically identical, so the commands already computed for FL, HL, FR and HR still apply;
dropping the six middle columns turns the 18-D command into 12-D. Of the three leg-loss variants
only this one walks: front-loss tips and lies diagonal by frame 55, hind-loss rears vertical at
frame 27 and collapses.

Data: `data/ik_4leg_middleloss_clean9`, 9 clips passing the walk check, collected with the
training set's framing (`--cam_dx -0.6 --spawn 0 0 --scale 0.5 --travel 0.8 --warmup 20`).
Several accepted clips still drift 0.19-0.20 m laterally, so this is a probe set, not a training
dataset.

**The test has to be few-shot, not zero-shot.** The 4-leg action space is 12-D, and so is the B1's,
but the coordinates mean different things -- matching dimensionality is not matching semantics.
So: freeze V-JEPA, the ITM, the FTM and the decoder backbone; add a new 12-D head; fit only that
head on a few 4-leg clips; score held-out clips. The control is the identical procedure on a
**random backbone**, same architecture, same data budget.

| split | pretrained Stage 2 | random backbone | gain |
|---|---|---|---|
| A | 1.86 deg | 5.06 | 2.72x |
| B | 1.67 | 4.72 | 2.83x |
| C | 1.71 | 5.18 | 3.03x |
| **mean** | **1.75 +/- 0.10, R^2 +0.967** | **4.99 +/- 0.24, R^2 +0.743** | **2.86x** |

**And it is sample efficiency, not just accuracy.** Over clip budgets 1, 3, 5, 7 with three splits
each, the pretrained backbone beats random by 2.6-2.9x at every budget -- reaching with **one clip**
(2.56 deg) what the random backbone never reaches with seven (4.78 deg).

**The latent is doing work, not just the frame.** F31 established that one frame nearly determines
the command in this data, so the obvious worry is that the 4-leg result is that same shortcut.
Ablating the latent while refitting the head each time:

| | test deg | R^2 |
|---|---|---|
| real `z` | **1.86** | +0.96 |
| zero `z` | 2.49 | +0.94 |
| shuffled `z` (real latents, permuted within clip) | 3.35 | +0.88 |
| random backbone | 5.06 | +0.74 |

Zero-`z` still beats random, so the frame plus the pretrained backbone carries a lot -- the F31
effect is real and present. But `real_z` beats `zero_z` and strongly beats `shuffled_z`, so an
**aligned** transition latent adds something a permuted one cannot. The cautious claim: few-shot
transfer uses both the frame representation and the latent, and is neither zero-shot latent
control nor pure frame-to-action fitting.

**The commands physically walk.** Open-loop replay of the predicted actions in CoppeliaSim, all
four held-out split-A clips, ghost-removed middle legs:

| clip | predicted fwd/lat | IK fwd/lat | feet down pred/IK | out-of-range |
|---|---|---|---|---|
| ep101 | +0.660 / -0.233 m | +0.701 / -0.239 | 2.55 / 2.55 of 4 | 1.8% |
| ep130 | +0.665 / -0.188 | +0.694 / -0.181 | 2.58 / 2.60 | 2.4% |
| ep6 | +0.713 / -0.168 | +0.692 / -0.277 | 2.51 / 2.55 | 1.4% |
| ep69 | +0.650 / -0.167 | +0.655 / -0.273 | 2.58 / 2.65 | 2.4% |

Stable 4-leg walking, closely matching the IK reference; in two clips the prediction drifts *less*
than the reference. Still open-loop reconstruction, not closed-loop control.

**It does not repair bad demonstrations.** On `data/ik_4leg_middleloss_badtest` -- 8 clips that
move forward but veer 0.22-0.31 m -- the pretrained backbone still beats random (2.31 vs 6.75 deg,
R^2 +0.94 against +0.53), and the replay veers where the IK veers. Matching a bad clip means the
correspondence was learned; it should not be framed as gait correction.

### The three interventions on embodiment identity, all measured

| | 4-leg few-shot, split A | held-out hexapod R^2 | probe after 8 dirs removed | `z` zeroed |
|---|---|---|---|---|
| `stage2_clean` | 1.86 deg | +0.87 | 0.738 | 7.63x |
| `stage2_clean_centered` | 1.88 | +0.89 | 0.697 | **9.96x** |
| `stage2_clean_adv_warm10` | **1.66** | +0.88 | **0.598** | **4.44x** |
| random backbone | 5.06 | -- | -- | -- |

**Centring does nothing.** The online probe reaches **1.000** during training, having started at
0.594 and climbed back -- the model relearns identity within 25 epochs with the offset already
removed. Centring subtracts the *first moment*; two robots differ in shape, silhouette and leg
count, which vary frame to frame and survive it. The lesson from F41 -- that an offset wrecks a
readout -- applies to a **linear readout fitted on one embodiment**, not to a nonlinear model
trained on both.

**The adversary is the only lever that moves anything**: best 4-leg result and lowest residual
identity.

The apparent cost -- zeroing `z` drops from 7.63x to 4.44x, suggesting a weaker latent -- does not
survive checking. That was measured on the *decoder*. Rolling the **forward model** on its own
output, true latents supplied, held-out body, 162 rollouts:

| steps ahead | clean | adversary | centred |
|---|---|---|---|
| 1 | 1.38x | 1.37x | 1.38x |
| 3 | 1.52x | 1.51x | 1.53x |
| 5 | 1.48x | 1.47x | 1.48x |
| 10 | 1.30x | 1.30x | 1.29x |

Identical within 1% at every horizon, all beating a frozen world by ~1.5x and constant velocity by
two orders of magnitude. **`z` still carries everything the world model needs.**

What changed is the decoder's balance between its two inputs:

| | `zero_z` | `zero_x` |
|---|---|---|
| clean | 0.365 | 0.193 |
| adversary | **0.140** | **0.266** |

The decoder now depends *less* on the latent and *more* on the frame -- which is the direction
F18 and slide 4 ask for, since the failure they name is the decoder reading the body out of `z`
while ignoring the frame it is holding. Consistent with the 4-leg result improving and held-out
R^2 holding. So the adversary leaves the world model intact and moves the decoder toward reading
geometry from pixels; it is a candidate to replace the baseline rather than a trade-off.

> **Last paragraph corrected by F46 (2026-08-14).** The swap test, run on these same two
> checkpoints, shows the decoder still answers with the *latent's* body under the adversary. The
> `zero_z`/`zero_x` shift above is real but overstates it: zeroing an input is out of distribution,
> so those ratios compare runs rather than locating the pathway. "Moves the decoder toward reading
> geometry from pixels" holds directionally; "a candidate to replace the baseline" does not.

**Measurement trap this cost.** `scripts/diagnostics/score_body.py` did not apply the stored
`embedding_offsets`, so a centred checkpoint scored on raw embeddings read **15.10 deg, R^2 -0.95**
-- reported as "centring breaks transfer" before the bug was found. With the offset applied it is
3.61 deg, R^2 +0.89. The same omission does *not* affect `fit_4leg_head`, because the new head is
fitted on the target data and absorbs a constant offset into its bias; a frozen head cannot.

## The setup this points to

1. **Bodies, not episodes.** Sixteen times more episodes of two bodies changed nothing (F13);
   going from two bodies to five cut held-out error 3.1x (F16). Roughly 30 episodes each across
   6 to 8 bodies costs the same to collect as 100 across 3.
2. **Bracket the test body.** A body inside the training hull scores 10 to 30 times better than
   one outside it, with everything else identical (F17). State which regime a reported number
   comes from.
3. **Check that the morphology axes do anything.** Coxa scale moves the joint commands by 0.73
   deg where tibia moves them by 28.63, so a three-parameter family is two-dimensional in the
   space that matters (F15). Measure this before designing a held-out body around an axis.
4. **Keep the shared foot trajectory.** It is what makes "the same latent action across
   different bodies" well defined. The averaging baseline it creates is beaten by learning the
   curvature the linear mixture cannot express, which needs bodies rather than different gaits.
5. **Report the baselines.** Copy-nearest-body, the training-body average, and the best linear
   mixture belong in every held-out table. On this data the mixture reaches 0.18 deg, so it is a
   ceiling as well as a baseline.
6. **Report per joint type, against the held-out body's own baseline, over at least three runs.**
   F8, F10 and F12 each independently make a single aggregate number misleading.
7. **Do not select on the held-out curve.** F14. Fix the budget in advance and report the final
   checkpoint, or report the whole curve.

Four interventions have been tried on the model and none worked (F9, F4b, F21, F22). The last
two are the informative failures: removing the body code from the latent pushed the decoder onto
the frame and made transfer worse, and handing it the frame directly made it use the frame 7.6x
less while transfer stayed level. Access is not the constraint and neither is capacity.

That leaves the objective, and F23 locates the problem inside it. `L_recon`, the term that is
supposed to make `z` an action, contributes 3 to 7 percent of the forward model's accuracy while
taking 99 percent of the gradient; `L_motion`, the term that actually needs `z`, gets the
remaining 1 percent and is satisfied by a lookup. No change to the decoder can repair a latent
that nothing shaped.

Three untested directions, none needing new data: reweight the two terms, drop `L_recon`, or
require the same latent to decode correctly against a *different* body's frame (`lambda_cross`,
which the shared expert episodes make possible).

## What this enables

The result that stands on its own is a mechanism, not a score.

**A frozen video encoder carries robot morphology in a form that is linearly decodable and that
generalises to a body it has never seen.** Ridge regression on mean-pooled V-JEPA2 embeddings,
fitted on five bodies, recovers a sixth body's three segment scales to 0.050, 0.039 and 0.002
(F20). Nothing supervises this and the encoder was never trained on robots.

**The world model trained on top of it does not use that.** It identifies the body from an
11-percent component of its own latent rather than from the frame (F18, F19), which is a lookup
over the bodies it saw. Its answer for the held-out body implies segment scales of (0.98, 0.98,
0.97) where the truth is (0.80, 0.90, 0.90) -- worse than the linear probe, with 5.2M parameters
against 4,227.

**The gap is architectural, and the diagnosis is specific.** The decoder reaches the frame only
through cross-attention with the latent as the query, so it retrieves what the latent asks for,
and the latent is 64 percent gait phase. Morphology sits in the tokens unqueried. Three
interventions confirm the diagnosis from different directions: rescaling the motion target does
nothing (F9), shrinking the decoder makes it worse (F4b), and removing the body code from the
latent moves the decoder onto the frame by 1.7x and still makes transfer 1.23x worse, because
there is no mechanism there to read it with (F21).

**What does work is coverage,** and only within the training hull. Five bodies instead of two cut
held-out error 3.1x and beat the no-learning baseline for the first time (F16); a body outside
the hull scores 10 to 30 times worse than one inside it under an otherwise identical control
(F17), and the frozen encoder fails to extrapolate in the same way (F20).

For the cross-embodiment claim in `OPEN_QUESTION.md`, the encoder result is the load-bearing one:
it says vision carries body geometry in a usable, generalising form, which is the premise the
whole argument rests on. The baselines that beat the model here -- copy-nearest-body, averaging,
linear mixture -- all interpolate inside a shared 18-D joint space. An 18-DOF hexapod and a
12-DOF quadruped have no shared joint space and no midpoint, so those baselines do not exist
there. That is the asymmetry the thesis argues for: vision is a common space where
proprioception is not.

### F45. Cross-embodiment pairing has no usable label on this data, and the probe says so before the run

`lambda_cross` is the only intervention of six that improved Stage 1 transfer (F24), so the
obvious move for Stage 2 is to port it. It cannot be ported as written: it decodes body A's
latent against body B's frame supervised by B's command, which is well posed only because every
insect body walks the *same expert episodes*. The hexapod and the B1 share none. The pairing has
to be rebuilt from something both robots record, and per-leg contact is the candidate -- it needs
no shared gait period, and the four corner legs correspond anatomically (F41b).

Measured before spending a run, on 150 hexapod clips over 5 sound bodies and 14 B1 clips
(`scripts/diagnostics/pairing_feasibility.py`):

| label | overlap | hexapod frames pairable | intent, hexapod | intent, b1 |
|---|---|---|---|---|
| `n_feet_down`, 0-4 | 0.572 | 98.9% | 0.913 | **0.998** |
| `diagonal`, which pair is loaded | 0.711 | 100% | **0.918** | 0.605 |
| `corner_pattern`, 16-way | **0.240** | **33.8%** | 0.630 | 0.524 |

`intent` is matched-pair command distance over random-pair distance, **within a single body**, so
geometry is held constant and only the label varies. **1.0 means the label carries nothing**, and
that is the number that decides it: `L_cross` supervises the decoder with the *partner's* command,
so a label that does not imply a similar command on one robot cannot mean "the same intent" across
two. That is a **wrong target, not a noisy one**, and wrong targets do not average out.

**No label in the family is usable, and the two failure modes are opposite.** The 16-way pattern
carries real intent on both robots, 0.630 and 0.524, but leaves **two-thirds of hexapod frames
with no partner at all**. The coarse labels cover everything precisely *because* they discard what
made them meaningful: `n_feet_down` is a coin flip on the B1 at 0.998, `diagonal` is one on the
hexapod at 0.918. Coarsening to raise coverage is the same operation that destroys the meaning,
which is why both columns have to be read together.

**The 16-way table shows the mechanism.** The B1 spends **84.6%** of its time in two patterns,
`0110` and `1001` -- the two diagonals of a trot. The hexapod spreads over all sixteen with
nothing above 15%, and **nine of the sixteen are hexapod-only**. This is the quantified form of
the claim that there is no physically correct answer to what "the same phase" means for a six-leg
wave and a four-leg trot: it is not that the answer is hard to choose, it is that on this data no
choice is both covered and meaningful.

**Checked before trusting it.** The B1's `foot_contact` is already binary, so its 0.5 threshold is
a no-op; the hexapod uses the validated 0.27 N; and overall duty matches closely, 0.533 against
0.515. The mismatch is not an artefact of two differently-calibrated sensors.

**What this does not establish.** Only 14 B1 clips, 1,143 frames. And both datasets are
behaviourally narrow -- one IK wave gait, one RL trot at a commanded velocity -- so part of the
non-overlap is the F31 constraint restated rather than a fact about hexapods and quadrupeds. That
cuts *for* the behavioural-diversity direction in Q11: a dataset with varied gaits and speeds is
the one change that would widen the overlap, and this measurement says how much it would have to
widen to matter.

**The method here is slide 10's, reused.** A few minutes of CPU on recorded data, no encoder and
no training, deciding whether a run can answer the question it is meant to ask. The Stage 1
version predicted which held-out bodies would transfer; this one says a mechanism cannot be built
at all yet, which is worth more than the run that would have discovered it in four GPU-hours.

### F46. Stage 2 has the Stage 1 pathology, and the adversary does not fix it

Every argument for porting `lambda_cross` to Stage 2 assumed the Stage 1 diagnosis carries over.
It had never been checked here: the swap test that established the pathology (F18) had not been
run on any Stage 2 checkpoint. It has now.

Two hexapod training bodies whose own commands differ by **21.13 deg**, decoded through the
hexapod head, 195 transitions each (`scripts/diagnostics/swap_pathway.py --embodiment hexapod`):

| checkpoint | frame from | latent from | vs `c10f10t10` | vs `c10f06t06` | follows |
|---|---|---|---|---|---|
| `stage2_clean` | c10f10t10 | c10f06t06 | 21.03 | **6.98** | latent |
| `stage2_clean` | c10f06t06 | c10f10t10 | **5.26** | 20.19 | latent |
| `adv_warm10` | c10f10t10 | c10f06t06 | 18.76 | **8.04** | latent |
| `adv_warm10` | c10f06t06 | c10f10t10 | **6.87** | 18.04 | latent |

The diagonals are near-identical across the two runs -- 4.85/5.97 against 4.89/5.98 -- so both
reconstruct their own bodies equally well and the crossed cells are comparable.

**Stage 2 has the pathology.** `stage2_clean` answers with the *latent's* body to within 7 deg
while sitting 21 deg from the body whose frame it is holding. Same shape as F18's Stage 1 control,
which answered within 3.5 deg on a 28.6 deg spread. **This is the first direct evidence that a
cross term is needed in Stage 2 rather than inherited as an assumption.**

**The adversary does not fix it.** It narrows the margin from 3.0-3.8x to 2.3-2.6x, roughly a
quarter of the way, and never approaches 1.0 where following the frame would begin. `lambda_cross`
reversed this test outright in Stage 1 (F24).

**Why F44 read it more favourably.** F44 measured the pathway by zeroing an input: `zero_z`/`zero_x`
went 0.365/0.193 to 0.140/0.266, which looks like a reversal. Zeroing is out of distribution, so
those ratios compare runs rather than locating the pathway -- the standing caveat in
`scripts/README.md`, and exactly the case it was written for. The swap test feeds real embeddings
and disagrees. **Two measurements of "which input does the decoder use" pointed opposite ways, and
the one with the known confound was the optimistic one.**

**What this opens.** The pathology lives *within* the hexapod head, where four bodies share one
18-D output -- not between embodiments, where per-embodiment heads already select structurally and
the identity in `z` is passive (F43). And those four bodies walk identical expert episodes, so
`lambda_cross` is as well defined there as in Stage 1. F45 rules out *cross-embodiment* pairing; it
says nothing about this. `MultiEmbodimentPairs` currently builds no `partners` map and emits no
`cross_x_t`, so the term is simply never computed -- a code gap, not a data one, and the `group`
field it needs is already loaded.

### F47. The 4-leg's latent lands on the base body, so its transfer is lookup, not composition

F46 crossed the decoder's inputs between two bodies of one embodiment. The same swap is exactly
definable *across* embodiments for the 4-leg insect and nothing else: it is the base stick insect
with the middle legs ghost-removed, driven by the **unchanged** six-leg IK gait, so its 12
commands are the base body's corner-leg columns **bit-identically** (verified: max difference
0.0000 deg) and it walks the same expert episodes. None of F45's pairing problem applies, because
intent is shared by construction.

Frame from the short body `c10f06t06`, latent from the 4-leg video, decoded through the hexapod
head, corner columns only, 195 transitions (`scripts/diagnostics/swap_embodiment.py`):

| frame from | latent from | RMSE vs frame's body | RMSE vs latent's body | follows |
|---|---|---|---|---|
| 4-leg | 4-leg | 5.68 | 5.68 | control |
| `c10f06t06` | `c10f06t06` | 5.39 | 5.39 | control |
| `c10f06t06` | 4-leg | 19.30 | **5.78** | latent |
| 4-leg | `c10f06t06` | 21.46 | **6.91** | latent |

The candidate answers differ by 21.62 deg, and both are commands the hexapod head already produces
for training bodies, so neither is out of its reach.

**Read alone this looks like the thesis property** -- a latent inferred from an embodiment never
trained on still drives the output. It is not, and the latent itself says why. Distances between
latents at matched timesteps, normalised by the latent's own spread:

| pair | distance |
|---|---|
| 4-leg vs base `c10f10t10` | **0.578** |
| 4-leg vs short `c10f06t06` | 1.148 |
| base vs short | 1.103 |
| base vs **itself at a random timestep** (chance) | 0.981 |

**Two things follow, and the second undercuts the first reading.**

**Body identity dominates gait phase in `z`.** Two *different* bodies at the *same* timestep sit
1.103 apart while the *same* body at a *random* timestep sits 0.981 apart. Which body you are
matters more than where you are in the stride -- F46's pathology, now located in the latent rather
than inferred through the decoder.

**The ITM barely registers the missing legs.** The 4-leg sits 0.578 from the base body, far closer
than same-body-different-phase. It reads the 4-leg video as "the base body, at this phase". So the
swap result is the same lookup F46 found, succeeding *because the 4-leg's geometry is a training
body's geometry*.

**What this costs the Stage 2 claim.** The 4-leg was designed as the compositional test: insect
appearance, quadruped leg count. It does not currently test composition, because only leg count is
novel and the latent is mostly tracking geometry. F44's few-shot numbers stand as measured -- 2.86x
over a random backbone is a real comparison -- but the mechanism story changes: the backbone
transfers well partly because the target's geometry is in distribution. **The 4-leg tests novel leg
count, not a novel body.**

**The fix is one collection run and no new tooling.** Ghost-remove the middle legs from a
*held-out* geometry rather than the base one -- `c08f09t09` is already withheld from Stage 2
training -- so leg count and geometry are both novel. Then the few-shot comparison measures what
slide 15 claims it measures. Only middle-loss walks (front-loss tips, hind-loss rears, F44), so
the variant is forced; render it before collecting, since a geometry change can break a gait that
worked on the base body.

> **Done, and the conclusion survives -- see F48.** The rebuilt body gives 1.91 +/- 0.08 deg
> against a random backbone's 5.45 +/- 0.16, a **2.85x** margin where the base-geometry body gave
> **2.86x**. The confound above was real in the evidence and did not change the answer.

### F48. Rebuilt on held-out geometry, the 4-leg transfer survives unchanged

F47 found that the 4-leg's geometry was a *training* body's, so the few-shot result could have
been riding on that rather than on transfer. This is the control that decides it: the same leg
removal (`ML,MR`) applied to **`c08f09t09`**, which Stage 2 withholds from training, so leg count
and geometry are both unseen. Same collection settings, same few-shot protocol, same checkpoint,
5 train clips, 300 epochs.

| split | pretrained Stage 2 | random backbone | margin |
|---|---|---|---|
| A | 1.80 deg, R^2 +0.97 | 5.25, +0.74 | 2.92x |
| B | 2.00, +0.96 | 5.47, +0.70 | 2.74x |
| C | 1.94, +0.96 | 5.64, +0.68 | 2.91x |
| **mean** | **1.91 +/- 0.08** | **5.45 +/- 0.16** | **2.85x** |

Against the base-geometry body (F44): 1.75 +/- 0.10 against 4.99 +/- 0.24, **2.86x**.

**The margin is identical -- 2.85x against 2.86x.** Both absolute numbers get slightly worse on a
body whose geometry was never trained on, which is the expected direction, and the *ratio the
claim rests on* does not move. So F47 identified a real flaw in the **evidence** and not in the
**conclusion**: the few-shot transfer was not an artefact of the target's geometry being in
distribution. Slide 15's claim now rests on a body that is genuinely held out on both axes.

**Dataset.** `data/ik_4leg_c08f09t09_clean10`, 10 clips kept from a 30-episode sweep, built the
same way `clean9` was (9 kept from 30) -- collector settings identical, walk check from
`wm.bodies`.

**A physical side-finding worth recording, because it explains the yield.** Removing the middle
legs **flips the direction of lateral drift**:

| body | lateral drift | sign |
|---|---|---|
| 6-leg `c10f10t10` | +0.134 m | 0/30 negative |
| 6-leg `c08f09t09` | +0.095 | 4/30 negative |
| 4-leg base | -0.180 | 9/9 negative |
| 4-leg `c08f09t09` | -0.225 +/- 0.050 | **30/30 negative** |

Every single 4-leg clip veers the same way, tightly, while the 6-leg bodies veer the other way.
This is a systematic consequence of the missing middle legs, not instability and not a property of
one body, which is why roughly two thirds of episodes fail the 0.20 m lateral gate on both 4-leg
builds. The selection is therefore the same operation on both, and the F44/F48 comparison is not
confounded by it -- but note that both datasets are the low-drift tail of a biased distribution,
so neither is a straight-walking body.

## Files

- `results/wm/stage2/4leg_head/c08f09t09_fewshot.csv` -- the three splits behind F48
- `scripts/diagnostics/pairing_feasibility.py` -- is a cross-embodiment `L_cross` definable (F45)
- `scripts/diagnostics/swap_pathway.py` -- also takes `--embodiment` for Stage 2 checkpoints (F46)
- `scripts/diagnostics/swap_embodiment.py` -- does an unseen embodiment's latent drive the
  decoder, or is it read as a training body (F47)
- `wm/predict_actions.py` -- reconstruct joint commands for any body, in radians
- `scripts/diagnostics/swap_pathway.py` -- cross the decoder's two inputs between bodies (F18)
- `scripts/diagnostics/morphology_mix.py` -- which mixture of training bodies an answer resembles (F19)
- `scripts/diagnostics/morphology_axis.py` -- where each stage places a held-out body (F4, F4d)
- `sim/scene/make_leg_morphology.py` -- generate a body by scaling segments independently
- `results/wm/dataset/morphology_bodies.png` -- the nine bodies
- `results/wm/cache/z_by_body.npz` -- latents behind the variance decomposition in F19
- `scripts/diagnostics/plot_action_trace.py` -- per-joint predicted against ground truth, with R^2
- `wm/sweep_checkpoints.py` -- re-score every snapshot on identical cached embeddings
- `results/wm/stage1/figures/interpolation_failure.png` -- F4 and F6 in one figure
- `results/wm/stage1/figures/heldout_sweep_two_seeds.png` -- F11 and F12
- `results/wm/cache/axis_embeddings.npz` -- embeddings behind the axis positions in F4
- `scripts/diagnostics/z_identity_ablation.py` -- is the embodiment in the latent used, or only present (F39, F43)
- `scripts/diagnostics/score_body.py` -- one checkpoint against several held-out bodies, no retraining (F43)
- `scripts/dataset/write_run_log.py` -- regenerate `results/wm/RUNS.md` before deleting any checkpoint
- `results/wm/OVERNIGHT.md` -- the clean Stage 2 results in one place, with what is withdrawn
- `results/wm/cache/stage2_embeddings.pt` -- cached encoder pass behind F39, rebuilt on demand and
  gitignored: every patch token at full width is 2.9 GB
- `scripts/figures/make_track_figures.py` -- the coverage, variance-share and probe-matrix figures
- `scripts/dataset/compare_ratio_gaits.py` -- side-by-side video and contact diagram across the
  femur/tibia boundary, from recorded frames (F42)
- `results/wm/dataset/ratio_gaits_ep6.mp4` -- the five bodies walking, ordered by ratio
- `results/wm/README.md` -- per-run metrics
