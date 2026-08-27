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

**Scope against the source method.** LAC-WM has no *cross-decoding* term like this. **Corrected by
F67**: the sentence that stood here said its shared latent space "is meant to emerge from sharing
the modules across embodiments", and that is wrong -- LAC-WM's motion-decoding loss is an explicit
alignment term, and its own Figure 2 shows the space is disjoint without it. What LAC-WM lacks is
this term, not any term. Its setting probably does not need this one: the shortcut this term closes -- recognise the body, recall its commands -- only pays when
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

> **Superseded in its numbers by F53, not in its conclusion.** Re-measured on
> `ik_walk_m3d_clean` the same way, the errors are uniformly lower — 3.00 / 3.40 / 2.86 for
> `a_t` / `a_{t+8}` / `a_{t+32}` — because this fitted 18 clips rather than 4, and 4 clips is
> 264 samples against 1,408 features. The measured cycle is **19 frames, not 22**, identical
> across all five bodies, and 32 is not a multiple of it, so "a distant offset wraps back to a
> similar phase" is not the mechanism: open-loop IK plus a fixed phase makes *every* horizon
> predictable, whole cycle or not. Use F53's table.

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
not here for bandwidth or latency; it is here because it describes an 18-DOF hexapod and a 12-DOF
quadruped in the same coordinates **without being given a description of either**. They share no
joint space, no sensor correspondence and no midpoint (F38's premise). That property costs 95 ms
per frame.

**Do not write that proprioception cannot do this.** Morphology-agnostic proprioceptive control
exists -- joints as a token set over the kinematic graph -- so the defensible claim is not that
joint space cannot be made to work across bodies. It is that those methods must be **handed the
kinematic tree**, and a camera has to be handed nothing. Check the specific references before
citing them; the point stands regardless of which papers are named.

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

### F49. Retrained on clean data: the mechanism strengthens, its accuracy gain shrinks, and coverage becomes the strongest result

All five Stage 1 runs the deck cites, retrained by `scripts/retrain_stage1.sh` on data where every
training body walks and at the corrected `action_lag 1`. `tib_ctrl` is included, so slide 3's
named control exists for the first time. Scored with `scripts/diagnostics/score_body.py`; raw
numbers in `results/wm/stage1_correct/measurements/heldout_scores.csv`.

| run | held out | deg | R^2 | `zero_z` | `zero_x` |
|---|---|---|---|---|---|
| `m3d_cross` (0.5) | c08f09t09 | **3.44** | +0.81 | 0.917 | **1.621** |
| `m3d_bracketed` (0.0) | c08f09t09 | 3.67 | +0.79 | 0.729 | **0.083** |
| `tib_cross` (0.5) | c10f10t08 | 12.67 | -0.78 | 2.349 | 1.991 |
| `tib_ctrl` (0.0) | c10f10t08 | 13.41 | -0.41 | 5.404 | 1.510 |
| `bracket_cross` (0.5, 6 bodies) | c10f10t08 | **3.27** | **+0.89** | 0.699 | 1.037 |

**The pathway result is stronger than it ever was on contaminated data.** On the matched m3d pair
the control's `zero_x` is **0.083** -- deleting the frame entirely costs almost nothing, so the
decoder is not reading it at all -- against `zero_z` 0.729. With the cross term the order inverts:
`zero_x` 1.621 against `zero_z` 0.917, so the frame matters *more* than the latent. A **16x swing
in which input the decoder depends on, from one flag**, and the cleanest statement of F18 and F24
in the project.

**The accuracy gain shrank by three.** 3.44 against 3.67 is 6%, where the contaminated pair read
2.91 against 3.57, or 18%. `lambda_cross` still wins and still repairs the pathway outright; what
it buys in *degrees* is much smaller once the veering bodies are gone. Q13 predicted cleaning the
data would strengthen Stage 1's claims -- it strengthened the mechanism and weakened the headline
number, which is not what was predicted and should be said plainly.

**Coverage is now the strongest result in Stage 1.** Four bodies that tie femur to tibia score
12.67 deg at R^2 **-0.78**; six bodies that decouple them score 3.27 deg at R^2 **+0.89**, at
matched volume (96 training clips both sides). The contaminated version of this comparison moved
27.68 to 16.10 and landed exactly on the constant-pose baseline without passing it. **The clean
version crosses zero R^2 and beats every baseline**: filling the named gap does not merely improve
extrapolation, it removes the failure.

**Read with three caveats, none optional.**

- **The held-out body changed.** `tib_*` and `bracket_*` now hold out `c10f10t08` (ratio 1.04)
  rather than `c10f10t06`, which does not walk (F42). With the six-body set spanning 0.83-1.10 the
  held-out body sits *inside* the hull, so `bracket_cross` is interpolation where `tib_cross` is
  extrapolation. That is exactly what coverage is meant to do, and also why the two are not a
  like-for-like difficulty comparison.
- **The m3d pair differs from the deck in two things**, data and `action_lag`, so old against new
  is not attributable to either alone.
- **The three 10-epoch runs peaked at epoch 10**, still improving when the budget ended. Not
  converged.

**One metric disagreement to settle before quoting slide 8.** `tib_cross` beats `tib_ctrl` in
degrees, 12.67 against 13.41, and loses on R^2, -0.78 against -0.41. Degrees pool raw error while
R^2 is in standardised units where low-variance joints weigh more, so both orderings are correct
and answer different questions. Slide 8 should name which one it reports.

### F49b. Every deck measurement re-run on the clean checkpoints, and what moved

The deck's numbers all came from checkpoints trained on the contaminated data. Re-measured on the
retrained pair, with the same scripts and the same held-out body. Recorded here rather than in the
deck: the slides state the current numbers only, without narrating what they used to be.

| measurement | contaminated (ctrl / cross) | clean (ctrl / cross) | verdict |
|---|---|---|---|
| swap test, which input decides | latent / **frame** | latent / **frame** | unchanged, and sharper |
| latent variance: gait | 64.5% / 88.7% | 81.9% / **92.6%** | same direction |
| latent variance: body | 8.8% / 1.2% | 12.4% / **3.4%** | same direction, 7x becomes 3.6x |
| behaviour decodable from `z` | 0.757 / 0.744 | 0.729 / **0.732** | now improves rather than dips |
| transition removed costs | 1.36x / 1.23x | 1.28x / 1.34x | unchanged in kind |
| **latent deleted costs** | 3.97x / **2.61x** | 2.88x / **3.48x** | **reversed** |
| replay distance, share of IK | 93% / 89% | 85% / **90%** | reversed |
| replay heading deviation | 3.7 / 6.8 deg | 11.8 / **5.5 deg** | reversed |
| commands outside joint range | 7.7% / **5.4%** | 6.1% / 6.4% | advantage gone |
| worst excursion | 20.2 / **5.5 deg** | 8.2 / **3.9 deg** | unchanged in kind |

**Three claims did not survive** and are gone from the deck: that the cross term reduces the
decoder's dependence on the latent, that it lowers the *frequency* of out-of-range commands, and
that neither model veers more than the IK reference. On one clean clip the reference walks almost
straight, -0.8 deg, and both models veer to +11.5 and +15.0.

**The reversal in the latent-deletion row is the informative one.** Under contamination the cross
term *reduced* the decoder's need for `z`; on clean data it *increases* it, 2.88x to 3.48x. Both
readings are correct for the data they were measured on, and together they say what `z` was
carrying in each case. With veering bodies in training, `z` was largely a body code, so a decoder
pushed onto the frame stopped needing it. With those bodies gone, `z` is 92.6% gait, so a decoder
that reads the body from the frame still depends on `z` for the movement. The division of labour
-- **frame carries which body, latent carries what movement** -- is only visible once the data is
clean, and it is what slides 5 and 11 now report.

**A swap-test detail worth keeping.** On the clean cross run the crossed rows score 4.79 and 5.84
against uncrossed 4.77 and 5.88: substituting the other body's latent moves the answer by 0.04
deg. Not a preference for the frame -- the latent contributes nothing to the body question at all.

**Checked on both saved checkpoints.** `best.pt` is selected on validation *total*, which is ~99%
reconstruction, while every number reported is action accuracy -- so the two could in principle
disagree. Scoring `best_motion.pt` for all five runs returns the **same figures to two decimals**
(3.44 / 3.67 / 12.67 / 13.41 / 3.27 deg, identical R^2), differing only in the third decimal of
`zero_z`. The same epoch won on both criteria, so the selection rule does not affect anything in
the deck.

### F50. Insect-only features transfer to the B1, weakly but consistently

Every cross-embodiment number so far was measured on a body Stage 2 trained on, or on the 4-leg
insect that F47 showed the model reads as the body it was cut from. This is the question asked on
the one genuinely different robot available: **hold the B1 out entirely, train on stick insects
only, and fit a B1 head few-shot.**

Backbone is `stage1_m3d_cross` -- four insect bodies, never a quadruped -- frozen, with a fresh
12-D B1 head. Control is the same head on a random backbone at an identical budget. Protocol,
fitting and metrics imported from `fit_4leg_head` so the scales match
(`scripts/diagnostics/fit_b1_head.py`).

| split | train / test | pretrained | random backbone | margin |
|---|---|---|---|---|
| random | 5 / 9 | 20.49 deg, R^2 +0.35 (+/- 4.41) | 23.80, +0.22 | 1.16x |
| random | 7 / 7 | 16.05 deg, R^2 +0.62 (+/- 0.58) | 20.09, +0.48 | **1.25x** |
| random | 9 / 5 | 15.62 deg, R^2 +0.68 (+/- 0.70) | 20.48, +0.51 | **1.31x** |
| **velocity-stratified** | 7 / 7 | **15.98 deg, R^2 +0.58** | 20.49, +0.39 | **1.28x** |

The margin is stable at **1.25-1.31x** wherever both arms produce a usable head, and grows slightly
with the clip budget.

**Transfer is real.** A backbone that has seen only stick insects makes a quadruped action head
measurably cheaper to fit than random features do, consistently across splits and budgets.

**And it is not a speed-generalisation artefact.** The B1 set is 2 policies x 7 commanded
velocities, so a random split leaves unseen speeds in the test set and conflates "new robot" with
"new speed". Stratifying -- one policy's clip of each speed in train, the other in test, so both
sides cover all seven velocities -- gives **1.28x against the random 7/7 split's 1.25x**.
Controlling the confound does not move the result.

> Corrected 2026-08-18: this line read "against the random split's 1.29x", which contradicts the
> table above it -- 20.09 / 16.05 = 1.25x, and no measurement in this entry produces 1.29x. The
> table is arithmetically self-consistent and is the number to use. Found by checking the two
> against each other; worth doing for every ratio quoted in prose beside a table.

**Read the margin only where both arms work.** At five clips the two backbones score R^2 -0.14 and
-0.12 on a single split: neither produces a usable head, so their ratio compares two failures. The
three-split average at that budget is positive but swings +/- 4.41 deg. Nine clips, or the
stratified seven, is where the comparison means anything.

**Do not read 1.28x against the 4-leg's 2.85x as "transfer is 2.2x weaker".** The tasks differ in
difficulty independently of embodiment: the B1 set spans two policies and seven speeds with unseen
combinations in the test set, while every 4-leg clip is one behaviour at one speed. Both show
transfer; the B1's is smaller. That is as far as the comparison goes.

**What this fixes in the argument.** The Stage 2 story rested on a 4-leg body the model does not
perceive as new (F47), which left "transfer to a genuinely different robot" untested. It is now
tested, on the one such robot in the project, and the answer is a small positive rather than the
null that F41b and F45 predicted -- the frozen encoder shares little (0.531 / 0.547 across on
per-leg contact) and the behaviour distributions barely overlap, yet something still carries.

**All of the transfer travels through the latent, and none through the frame.** Ablating `z` on
the velocity-stratified split, three splits, everything else identical:

| arm | deg | R^2 | margin over random |
|---|---|---|---|
| pretrained, real `z` | **16.01** | +0.577 | **1.28x** |
| pretrained, `z` zeroed | 20.86 | +0.342 | **0.98x** |
| pretrained, `z` shuffled within clip | 22.04 | +0.283 | 0.93x |
| random backbone | 20.49 | +0.393 | -- |

**Zeroing `z` removes the entire margin.** A backbone trained on insects, with its latent deleted,
scores 20.86 against a random backbone's 20.49 -- the same, within noise. So the decoder trunk's
learned processing of `e_t` carries **nothing** to a quadruped; every bit of the advantage arrives
through the ITM's latent.

**And the latent's value is its alignment, not its distribution.** Shuffling real latents within a
clip scores 22.04, *worse than supplying none at all*. A latent that does not belong with the frame
it is paired with is an active liability.

**This locates the lever.** The part of the pipeline that transfers across embodiments is the one
module that encodes *change* rather than appearance. It also sharpens the target: `z` reads a
loaded leg at 0.986 within an embodiment and 0.373 across (F41b), below the frozen encoder's
0.531 -- and since `z` is the only thing carrying transfer, raising that 0.373 is the thing to aim
at. Making the two robots' behaviour distributions overlap (F45's measured root cause) is the lever
with a mechanism behind it: the ITM learns from transitions, so shared transitions are what it
needs to find shared structure.

**The forward model does not transfer -- it is actively harmful.** Rolling the same insect-trained
FTM on B1 video, true latents supplied, against holding the frame still:

| steps ahead | on B1 video | on its own held-out insect body |
|---|---|---|
| 1 | **0.63x** | 1.52x |
| 3 | **0.57x** | 1.72x |
| 5 | **0.63x** | 1.69x |
| 10 | **0.71x** | 1.46x |

**Below 1.0 at every horizon**: on a quadruped the forward model predicts *worse than assuming
nothing moves*, while the same weights beat a frozen world by 1.5-1.7x on the body they were
trained near.

**So the three modules separate cleanly**, and only one of them crosses:

| module | across embodiments |
|---|---|
| ITM, the latent `z` | **transfers** -- 1.28x |
| decoder trunk's use of the frame | does not -- 0.98x, the same as random weights |
| FTM | **harmful** -- 0.57-0.71x, worse than doing nothing |

**This matters most for deployment.** The method's deployed form rolls the forward model, compares
imagined futures against a goal and picks actions from that comparison -- so the FTM, not the
motion decoder, is the module a planner depends on. A forward model that scores below a frozen
world cannot support planning on that robot, however cheaply its action head fits. **The current
model could not be deployed on the B1 whatever the few-shot numbers say.**

It also corrects a conclusion drawn one measurement too early. `z` transferring made the latent
look like the whole story; the module that *consumes* `z` fails on the same data, so the limit is
not located in the latent alone.

**What it rules out.** Adding bodies within either family does not touch this, and neither does
anything acting on the frame pathway -- that pathway was measured to carry nothing across
embodiments.

**A prediction that did not hold, recorded because it was mine.** I framed this test expecting a
null and wrote that expectation into the script. Two measurements pointing one way is a reason to
run the experiment, not a reason to report its answer in advance.

### F51. Coverage repairs the decoder and barely touches the forward model

F50 found the forward model harmful across embodiments. Three follow-ups locate why, and they
separate two modules that had been treated as one problem.

**The architecture can model a quadruped; the Stage 1 model simply never saw one.** Same rollout,
same B1 video, two checkpoints:

| FTM rolled on B1 video | 1 step | 3 | 5 | 10 |
|---|---|---|---|---|
| `stage2_clean`, trained on insects **and** B1 | **1.39x** | **1.53x** | **1.52x** | **1.34x** |
| `stage1_m3d_cross`, insects only | 0.63x | 0.57x | 0.63x | 0.71x |

Identical architecture and objective. Exposure alone moves the forward model from *worse than a
frozen world* to *1.5x better than one*. The limit is not the design.

**But coverage inside a family barely helps it generalise.** `tib_cross` (4 bodies, femur tied to
tibia) against `bracket_cross` (6 bodies, decoupled), rolled on the **same** held-out body at
matched volume:

| steps ahead | 4 bodies | 6 bodies |
|---|---|---|
| 1 | 1.23x | 1.29x |
| 3 | 1.27x | **1.38x** |
| 5 | 1.22x | 1.31x |
| 10 | 1.04x | 1.07x |

**Better at every horizon, by 5 to 8 percent.** On the *same pair of checkpoints* the motion
decoder improved 12.67 deg to 3.27 and R^2 -0.78 to +0.89 -- a factor of 3.9 (F49). **The same
intervention that transforms the decoder moves the forward model almost not at all.**

**And multi-embodiment training costs the forward model something.** On a held-out *insect* body,
`stage1_m3d_cross` rolls at 1.46-1.72x while `stage2_clean`, which added the B1 to training, rolls
at 1.30-1.52x. Sharing the trunk with a quadruped makes the insect forward model slightly worse.
Small, but it is a price nobody had measured, and it will matter if more embodiments are added.

**What this means for the plan.** The two goals do not respond to the same lever:

| goal | coverage helps? |
|---|---|
| action head fits cheaply on a new robot | **yes, enormously** -- 3.9x |
| forward model rolls a new robot forward | **barely** -- 5-8% |

So adding bodies, or even adding embodiments, is not a route to a forward model that works on an
unseen robot: it needs that robot inside its training distribution. Since the source method's own
transfer is a LoRA finetune on 7,265 target-robot trajectories rather than zero-shot, the
comparable question is not whether a frozen forward model generalises but **how few target clips
it takes to adapt one** -- the same sample-efficiency framing already used for the action head.

### F52. The forward model adapts to the B1 from one clip, and insect pretraining is worth 7x

The question F51 leaves. ITM and FTM from `stage1_m3d_cross` fine-tuned on N clips of B1 video
against the same architecture from random init, encoder frozen in both, scored by rolling the
forward model on its own output and dividing by holding the frame still. Three splits per budget,
1,000 optimiser updates each, four held-out clips.

```
.venv/bin/python3 -u scripts/diagnostics/finetune_ftm.py --clips 1 3 5 7 9 --splits 3
```

Measured on com7; its log stayed on that machine, so the table below is the record.

| clips | model | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|
| 1 | pretrained | **1.02x** | 1.00x | 0.94x | 0.83x |
| 1 | scratch | 0.89x | 0.68x | 0.64x | 0.66x |
| 3 | pretrained | 1.17x | 1.11x | 1.02x | 0.89x |
| 3 | scratch | 0.97x | 0.92x | 0.85x | 0.80x |
| 5 | pretrained | 1.23x | 1.17x | 1.09x | 0.95x |
| 5 | scratch | 0.98x | 0.97x | 0.91x | 0.85x |
| 7 | pretrained | 1.32x | 1.29x | 1.19x | 1.00x |
| 7 | scratch | 1.01x | 1.06x | 1.00x | 0.90x |
| 9 | pretrained | **1.37x** | **1.36x** | **1.26x** | **1.05x** |
| 9 | scratch | 1.01x | 1.10x | 1.06x | 0.96x |

**Pretrained beats scratch in all twenty cells**, and both curves rise monotonically with the
budget, so this is not noise.

**The headline is sample efficiency, not the ceiling.** Pretrained reaches 1.0x at h=1 with **one**
clip; scratch needs **seven** to reach the same place, and at h=3 scratch does not clear 1.0x until
seven clips either. Insect pretraining is worth roughly **7x fewer target clips**.

**The two curves separate rather than converge.** From five clips on, scratch is flat at h=1 --
0.98x, 1.01x, 1.01x -- while pretrained keeps climbing, 1.23x, 1.32x, 1.37x. The margin widens from
1.15x at one clip to 1.36x at nine. Scratch is not slowly catching up; it has saturated on what
this much target data can teach from cold.

**Against F51 this is a reversal of kind.** The frozen forward model scored 0.57-0.71x -- worse
than predicting no motion. One clip of the target robot takes it to 1.02x. What does not transfer
is not the representation but the calibration to the new body's dynamics, and that is cheap.

**Ten steps ahead clears break-even at nine clips, and only for the pretrained arm.** Pretrained
runs 0.83 / 0.89 / 0.95 / 1.00 / **1.05x** across the five budgets; scratch runs 0.66 / 0.80 /
0.85 / 0.90 / 0.96x and never crosses. The longest horizon measured is where the two arms differ
in kind rather than degree -- one gets there, the other does not.

**Budget cap.** Fourteen clips with four held out leaves ten as the largest clean budget. A run at
eleven was measured before this was noticed and put one of the four test clips inside the training
set -- `train = order[:n]` and `test = order[-4:]` overlap once `n + 4 > 14`. Its numbers
(1.46 / 1.42 / 1.31 / 1.08x pretrained) follow the same trend but are **not reportable**. The
script now refuses the overlapping budget outright.

**Trap this run cost.** `adapt()` originally summed gradients over every span before stepping, so
`--steps` counted epochs, cost scaled with the clip budget, and the sweep was 374,400
forward+backward passes and 13 hours. Batching the *device transfer* is what the memory limit
requires; batching the *optimiser* is not. See `scripts/README.md`.

### F53. F31 re-measured on the clean dataset: same conclusion, lower errors, and the cycle is 19

F31 was fitted on **four clips** -- 264 samples against 1,408 encoder features. `ik_walk_m3d_clean`
has 26 clips of `c10f10t10`, so the same ridge now sees 18. Nothing here uses a trained
checkpoint, so this was never stale from the Stage 1 retrain; it was stale from the dataset.

| predict from one frame | F31 (4 clips fitted) | F53 (18 clips fitted) |
|---|---|---|
| command spread | 11.33 deg | 11.34 deg |
| `a_t` | 4.61 | **3.00** |
| `a_{t+8}` | 5.23 | **3.40** |
| `a_{t+32}` | 4.45 | **2.86** |
| second frame, on the change | 1.09x | 1.11x |

The two runs agree on the spread to 0.01 deg, so the protocol reproduces; the errors fall because
the earlier fit was badly underdetermined.

**Pooling all five bodies gives the same answer.** 140 clips, spread widened to 15.04 deg by
between-body variance, errors 3.87 / 4.45 / 3.67 -- which is 26 / 30 / 24 percent of the signal
against 26 / 30 / 25 percent for the single body. The claim does not depend on that choice.

**The gait cycle is 19 frames, not 22**, measured by autocorrelation and identical across all five
bodies (range 19-19, as expected from replayed expert episodes). 32 is not a multiple of 19, so
F31's "a distant offset wraps back to a similar phase" was the wrong mechanism even though its
conclusion held. The right one is that the commands are open-loop IK: one frame fixes the phase
and everything downstream of it is determined, at any horizon.

### F54. Insect pretraining transfers dynamics *and* a foothold in the feature space, and only the second one is what makes adaptation cheap

F52 measured that insect-pretrained ITM+FTM adapt to the B1 from ~7x fewer clips than random init.
`scratch` controls for "any initialisation", but two very different readings both predict that:

    the model learned how legged bodies move, and that survives a change of robot
    the model learned the shape of V-JEPA2's feature manifold, nothing about motion

Two arms, identical in every respect except **what the ITM is given as its second frame**. Same
100 clips of `ik_walk_m3d_clean`, same architecture, 15,000 minibatch steps, same seed. ITM+FTM
only -- no decoder, no cross term -- since that is all `finetune_ftm.py` loads.

| arm | second frame |
|---|---|
| `real` | `e_{t+1}`, the frame that actually follows |
| `shuffled` | `e_s`, random `s` in the **same** clip, never `s == t` |

**Not undertrained.** At 5 clips `real` reaches 1.17 / 1.15 / 1.09 / 0.98x against
`stage1_m3d_cross`'s 1.23 / 1.17 / 1.09 / 0.95x, and `scratch` reproduces F52 to within 0.03x on a
different run and checkpoint. The arms sit close enough to the full Stage 1 checkpoint for the
comparison to mean something.

**After finetuning, the two arms are indistinguishable.**

| clips | arm | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|
| 1 | real | 1.03x | 1.03x | 1.00x | 0.90x |
| 1 | shuffled | 1.00x | 1.03x | 0.99x | 0.88x |
| 5 | real | 1.19x | 1.17x | 1.10x | 0.98x |
| 5 | shuffled | 1.18x | 1.19x | 1.11x | 0.98x |
| 9 | real | **1.31x** | 1.30x | 1.21x | 1.05x |
| 9 | shuffled | **1.31x** | 1.33x | 1.24x | 1.05x |

At nine clips the three-split ranges lie on top of each other -- 1.29-1.34 against 1.28-1.34.

**Frozen, before any finetuning, they are not.**

| arm | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| `real` | **0.54x** (0.53-0.54) | 0.51x (0.50-0.52) | 0.53x (0.52-0.55) | 0.59x (0.58-0.61) |
| `shuffled` | **0.39x** (0.38-0.39) | 0.45x (0.44-0.46) | 0.49x (0.48-0.51) | 0.57x (0.55-0.59) |

**The ranges are disjoint at h=1 and the advantage decays with horizon** -- 1.38x, 1.13x, 1.08x,
1.04x. `real` was trained on one-step pairs and its edge appears at exactly the horizon it was
trained on, which is the internal consistency check this rests on.

**Both arms are competent in-domain, which is what licenses reading the B1 numbers at all.** The
same frozen checkpoints, on 40 `ik_walk_m3d_clean` clips the pretraining never saw:

| arm, frozen on held-out insect clips | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| `real` | **1.38x** (1.38-1.39) | 1.37x (1.35-1.38) | 1.24x (1.23-1.26) | 1.04x (1.03-1.07) |
| `shuffled` | 1.33x (1.31-1.35) | **1.46x** (1.44-1.48) | **1.34x** (1.32-1.37) | **1.10x** (1.08-1.14) |

Two things fall out of this table.

**The embodiment gap, stated as plainly as it gets: 1.38x to 0.54x on the same weights.** Both arms
clear 1.0x at every horizon on insects and neither comes close on the B1. Nothing was retrained
between those two measurements, so the B1 failure is not undertraining -- it is the change of
robot.

**And each arm is best at the horizon it was trained on.** `real` saw only `Δ=1` pairs and wins at
h=1, then loses to `shuffled` at h≥3 with disjoint ranges. `shuffled` saw pairs averaging 21.9
frames apart -- stride scale -- and wins at every multi-step horizon. **Training the forward model
on adjacent frames produces a worse multi-step rollout than training it on stride-scale pairs**,
in-domain, by 7 to 8 percent at h=3 and h=5. That is a design consequence, not a curiosity: the
FTM is used by rolling it forward many steps, and it is currently trained on the one horizon where
that advantage does not apply.

> The cross-embodiment picture is the mirror image and should not be conflated with it. On the B1,
> `real` leads at *every* horizon, by a margin that shrinks with distance. In-domain the long
> baseline buys better rollout; across robots the little that survives is the one-step structure.
> Both readings come from single frozen evaluations and neither has been repeated on a second
> pretraining seed.

**So pretraining supplies two separable things.**

| what | measured by | size |
|---|---|---|
| transferable dynamics | the frozen comparison | real, **1.38x at h=1** -- but both arms are far below 1.0x, so it is useless on its own |
| a foothold in V-JEPA2's feature space | the post-finetune comparison | **this is what produces the 7x** |

A thousand optimiser steps on B1 clips teach more about B1 dynamics than insect pretraining ever
carried, so the frozen 1.38x is overwritten. **This does not say pretraining is unnecessary**: at
nine clips, pretrained scores 1.31x against `scratch`'s 1.01x, and that gap never closes. It says
the part of pretraining that survives finetuning is familiarity with the shared representation --
which is precisely what a camera provides and a joint space does not.

**Read `shuffled` narrowly; it does not remove motion.** Its partners average 21.9 frames apart
against a measured gait cycle of 19 (F53), so a shuffled pair shows the body at two points of a
stride rather than at no motion at all. Measured on the joint commands, pose distance runs 3.44 deg
at one frame, peaks at **17.67 deg at half a cycle**, and falls back to 5.81 deg at a full cycle;
under uniform sampling 14.6% of pairs land within one frame of the same phase. What the arm removes
is *adjacency*, not movement.

**Which sets the scale the whole design should be read at.** The insect expert data runs at
**20 Hz** (`sim_time` in `expert_66k_aug3c_fcontact.csv`), so a clip is 3.30 s, a stride is 0.95 s,
and `t -> t+1` is **50 ms -- one nineteenth of a stride, and 19% of the pose change a half-stride
carries**. That single number reconciles three results that looked in tension: slide 11's second
frame being worth only 1.11x, `shuffled` matching `real` after finetuning, and `real`'s frozen edge
living at h=1 and dying by h=10. **The informative window is stride-scale; one timestep is the
wrong unit to have built the comparison on.**

### F55. The switch is the mechanism, and it has one named cause (synthesis, not a new measurement)

Nothing here is new data. It is the chain that four existing findings make when read together, and
it is written down because it identifies the single thing left to fix.

| step | the evidence already recorded |
|---|---|
| no cross-embodiment frame pairing exists | **F45** — no label is both covered and meaningful; contact patterns either pair everything and mean nothing, or mean something and pair a third |
| so `lambda_cross` cannot be applied | it needs to know two frames show the same intent; insect bodies walk identical expert episodes, the hexapod and B1 share none |
| so nothing forces one `z` to mean the same thing on both robots | **F43/F46** — embodiment is decodable from `z` at 0.994 yet deleting it costs 1.03x, *less* than deleting random directions. The label is present and inert |
| so the trunk partitions rather than shares | the per-leg contact probe reads **0.986 within** an embodiment and **0.373 across**, below the frozen encoder's 0.531 and below chance. Two codes pointing opposite ways, not one shared code |
| so learned dynamics do not travel | **F54** — what survives the change of robot is the foothold in the *encoder's* space, which is shared by construction because V-JEPA2 is one model. What the trunk learned does not travel, because the trunk learned to split |

**Stage 1 is the positive control for the last three steps.** There `lambda_cross` *is* definable —
every body walks the same expert episode — and it **reverses the swap test outright**. In Stage 2,
without it, the adversary narrows the same test from 3.0-3.8x to 2.3-2.6x and never approaches 1.0.
An adversary removes *decodability*; it does not install *shared meaning*, which is exactly what the
probe shows: it moves the cross cells to 0.490 and 0.500, chance, and no further.

**What this predicts, so it can fail.** If a pairing can be defined and a cross term trained on it:
the cross-embodiment contact probe should rise above 0.531, the swap test should move toward 1.0,
and the frozen forward model should clear the 0.57-0.71x it currently scores on the B1. If pairing
is fixed and none of those move, this chain is wrong.

**This also re-weights Q14's remaining lever.** Widening behavioural overlap is not "more data is
better" — it is the precondition for a pairing to exist at all, which is what unlocks the term that
stops the partitioning. That makes it the one intervention aimed at the cause rather than at a
symptom.

### F56. The two gaits have different numbers of degrees of freedom, which is why no tight pairing exists

F45 found that no contact label is both covered and meaningful and left it there. This measures the
reason, and the reason is not "we have not found the right label yet".

**Anchor phase at front-left touchdown, then ask where every other foot lands.** If one leg's phase
fixes the whole body, the other legs land at a repeatable phase. Concentration is the circular
resultant length: 1.0 is perfectly repeatable, 0.0 is uniform.

| B1 | mean phase | concentration | | hexapod | mean phase | concentration |
|---|---|---|---|---|---|---|
| FL | 0.00 | **1.00** | | FL | 0.00 | **1.00** |
| RR | 0.05 | **1.00** | | ML | 0.84 | **0.22** |
| FR | 0.49 | **0.99** | | HL | 0.13 | **0.09** |
| RL | 0.55 | **0.99** | | FR | 0.47 | **0.24** |
| | | | | MR | 0.29 | 0.24 |
| | | | | HR | 0.43 | **0.07** |

**The B1 is a textbook trot** — FL and RR together, the other diagonal half a cycle later, all four
determined by one. **The insect's other five legs are near-uniform.** It walks a variable wave, as
the expert recording is a real animal, so its gait state needs roughly six loosely coupled numbers
where the B1's needs one.

**Consequence, and it is structural rather than empirical.** Any low-dimensional label that fully
describes the B1's gait state must underdetermine the insect's. A pairing built on one will supply
a **wrong** partner command rather than a noisy one, which is exactly F45's condition 3 and exactly
why it failed. This is not a search problem.

**Measured on the best task-space label we could build.** Phase anchored at touchdown, crossed with
Froude number `v / sqrt(g*h)`:

| label | overlap | hexapod pairable | b1 pairable | intent hexapod | intent b1 |
|---|---|---|---|---|---|
| F45 feet-down 0-4 | 0.572 | 98.9% | — | — | **0.998** |
| F45 diagonal | 0.711 | 100% | — | **0.918** | — |
| F45 corner pattern | 0.240 | **33.8%** | — | 0.63 | 0.52 |
| **phase x Froude** | **0.578** | **100%** | **100%** | 0.647 | **0.278** |

Better coverage than anything in F45 and a much better intent ratio on the B1, but **0.647 on the
hexapod is loose**, and the asymmetry between 0.647 and 0.278 is the leg-DOF result above showing
up in the pairing.

**Froude is the part that works.** The hexapod averages **0.155** and the B1 **0.159**, on hip
heights of 0.13 m and 0.56 m. The two robots walk at nearly the same Froude number despite a
four-fold size difference, which is why task space overlaps where contact does not, and it holds
without reference to phase.

> **A trap this measurement walked into first.** The phase was initially the Hilbert phase of the
> joint commands' first principal component, which gave intent ratios of 0.362 / 0.265 -- better
> than anything here. That number was **circular**: the phase was derived from the commands it was
> then tested against. Re-derived from contact, independent of the commands, the hexapod's intent
> ratio moved 0.362 -> 0.647 and phase alone moved 0.437 -> 0.895, near meaningless. Any label
> computed from the thing it is scored on will pass condition 3 for free.

**What this rules in and out.** It rules out finding a tighter frame-level pairing by searching for
a better label. It leaves two routes: make the insect's gait regular so one phase does determine
its configuration (expensive -- new foot-trajectory generation and every Stage 1 number retrained),
or drop frame-level pairing for a constraint that does not need a bijection, aligning only the part
of `z` the two robots can share.

### F57. The insect walked one speed, which made every body-level question unanswerable

F56 established that body-level motion is the only level where the two robots overlap -- Froude
0.155 against 0.159 at hip heights of 0.13 m and 0.56 m. The obvious next step was to read body
speed out of `z` and see whether it transfers. That measurement was run and returned nonsense:
`insect->b1` -0.284 and `b1->insect` -0.142 from the **frozen encoder**, worse than predicting the
mean, before any training was involved.

**The measurement was broken, not the representation.** Froude has two sources of variation and
they were not separated:

| | between clips (commanded speed) | within a clip (body rocking) | ratio |
|---|---|---|---|
| hexapod | 0.0188 | 0.0714 | **0.26** |
| B1 | 0.0255 | 0.0170 | **1.50** |

**The expert walks one speed.** Across 1,000 episodes its forward velocity has a standard deviation
of 0.0086 m/s on 0.454 -- **1.9 percent**. A readout fitted on the B1 learns commanded speed; one
fitted on the insect learns how far the body rocks within a stride. Those are different quantities
that happen to share a name, and asking one to transfer to the other is not a question with an
answer.

**The fix is retiming, not a synthetic gait.** `collect_ik.py --speed` resamples the shared foot
path along time: the same Cartesian path through fewer samples covers the same ground in less time.
**Every leg is resampled by the same time map**, so the inter-leg phase relationships F56 measured
-- the variable wave that makes the insect's gait high-dimensional -- are untouched. This replays a
real animal's coordination at a different tempo, which is what the animal itself does, rather than
authoring a tripod and discarding the recording.

`data/ik_walk_speed5`: five speeds, 0.72 to 1.10, 67 clips from 75 after `walk_check`.

| | insect, five speeds | B1 |
|---|---|---|
| mean Froude | **0.164** | 0.164 |
| range | 0.113-0.221 | 0.121-0.216 |
| sd between clips | 0.0284 | 0.0266 |
| **signal / rocking** | **1.45** | 7.28 |

Matched to the B1 by design: seven commanded speeds there, five retimings here, the same Froude
band. The second axis differs on purpose -- the B1 has two gait policies, the insect five
morphologies -- because an insect has no policy to switch.

**Re-measured on data that has speed in it, the frozen encoder is positive across robots**: +0.012
and +0.079, against -0.284 and -0.142 on the single-speed set. V-JEPA2 carried shared body-level
structure the whole time and there was nothing to read it against.

**And training destroys it.** Trained on the five-speed data with no new term, `z` gives -4.163 and
-5.595.

**Behavioural diversity does not fix cross-robot body-speed transfer.** (It does fix other things -- see F61, which corrects the broader claim this paragraph originally made.) Widening the insect from one speed to
five moves `b1->insect` from -24.359 to -5.595 -- four-fold -- and the sign is still wrong and still
two orders below the frozen encoder. `OPEN_QUESTION.md` Q14 named behavioural diversity as the one
lever left; this is the first test of it, and on its own it does not close the gap.

**Two measurement traps, both recorded because both cost a run.**

The window matters more than it looks. Between-clip variation against within-clip rocking reads
**0.63 at a five-frame window and 2.17 at a stride-length window** on the same clips. The rocking is
real body motion at stride frequency, not noise -- targeting the raw per-frame value hands a readout
mostly rocking and almost no speed. Same lesson as F54's `t -> t+1` being 50 ms.

And a dataset can be "diverse" and still be a lookup table. Five discrete speeds is five numbers to
memorise, which is what happened in F58.

---

### F58. A body-motion term shared across embodiments, and the first cross-embodiment result that beats the frozen encoder

F55 traced the switch behaviour to a single absence: `L_motion` supervises `z` through
**per-embodiment** heads onto 18-D and 12-D joint commands with no correspondence between them, so
nothing in the objective ever asks one latent to mean the same thing on both robots. LAC-WM's
equivalent term targets a hand-unified end-effector pose every arm has; their own EAC-WM ablation,
identical weight sharing without that pressure, produces visibly disjoint per-dataset clusters.

**The locomotion equivalent is body motion.** `wm/models/body_motion.py`: decode forward Froude
number from `z` through **one head shared by every embodiment**. A per-embodiment head here would
reintroduce exactly the freedom the term exists to remove.

Matched pair on `data/ik_walk_speed5` + `data/b1_framed`, one flag apart, 60 epochs, one seed.

| | insect->insect | b1->b1 | **insect->b1** | **b1->insect** |
|---|---|---|---|---|
| frozen encoder | 0.666 | 0.750 | **+0.012** | **+0.079** |
| `z`, no term (control) | 0.624 | 0.155 | -4.163 | -5.595 |
| **`z`, + `L_body`** | **0.676** | **0.879** | -1.931 | **+0.407** |

**`b1->insect` is +0.407 against the frozen encoder's +0.079 -- 5.2x.** Nothing in this project had
cleared that bar on any cross-embodiment measurement. The adversary reaches chance and stops (F38,
F59); `lambda_cross` is not definable here at all (F45, F56).

**Three of four cells beat the frozen encoder**, and `b1->b1` went 0.155 -> 0.879, past the
encoder's 0.750: training now *preserves* B1 body speed where before it destroyed it.

**Both ingredients were needed.** Five speeds without the term leaves the sign wrong (F57). The term
without the data has nothing to learn from.

**What it costs.** Val motion 0.0245 -> 0.0367, **+50%** on the metric slide 14 reports. Val recon
is unchanged, 1.5990 -> 1.6040. **Superseded by F65**: at `lambda_body 0.1` the cost is -2 percent
with the transfer unchanged within seed spread, so this is the weight's price and not the
mechanism's.

**The embodiment probe stops saturating.** The control reaches 1.000 by epoch 32 and stays; the
`L_body` run plateaus at 0.954-0.963 and never closes. First thing in Stage 2 to hold it below 1.0
-- the adversary never managed it. Read against F43 though: identity is decodable at 0.994 and
costs 1.03x to delete, so a probe reduction is not by itself evidence of shared meaning. The
body-motion probe is.

**Three things are still wrong, and they point somewhere specific.**

`insect->b1` is still negative, -1.931. The direction that fails is the one fitted on the noisier
side: signal-to-rocking is **7.28 on the B1 and 1.45 on the insect** (F57). A readout fitted on the
clean side transfers; one fitted on the noisy side does not.

**The head memorises.** Train loss 0.077 against 0.855 on held-out clips, where the target is
standardised so **1.0 is exactly "predict the mean"** -- an 11x gap, and no generalisation at any
epoch. Twelve distinct speed values across 32 clips is a lookup table: `z` encodes which clip this
is and the head reads off that clip's speed.

**But a weak readout is not a weak representation.** The head failed and the probe -- a fresh ridge
fitted on `z` directly -- found +0.407. `z` acquired transferable structure the jointly-trained head
never exploited. An earlier reading of this run called the term "satisfied by memorisation with no
shared meaning" on the strength of the val loss alone; that was wrong in the second half.

**Next step follows from the memorisation, not from the weight.** A continuous speed target within
each clip -- retiming with a ramp rather than a constant factor -- removes the lookup table and
raises the insect's signal-to-rocking ratio at the same time. Raising `lambda_body` would only
press harder on a target that has 12 values.

**One seed.** `stage2_clean` was run with two and the pair should be repeated before this carries
weight alone.

---

### F59. The adversary's 4-leg benefit disappears on a test body that was actually held out

F47 found the 4-leg test body had been cut from a *training* body and F48 re-measured the headline
margin on a build cut from held-out `c08f09t09`, where it survived unchanged: 2.85x against 2.86x.
The deck then kept using the base-geometry build for **everything else on that slide**. Re-measuring
the rest, one claim does not survive.

Same protocol throughout: frozen backbone, new 12-D head, five training clips
(`ep28,69,93,101,113`), five held out, identical random-backbone control at 5.35 deg.

| | base geometry | held-out `c08f09t09` |
|---|---|---|
| `stage2_clean` | 1.86 deg | 1.99 deg |
| `stage2_clean_adv_warm10` | **1.66 deg** | **2.13 deg** |
| | adversary 11% *better* | adversary **7% worse** |

**The apparent gain was the target's geometry being in distribution, not better transfer.** On a
body the world model never saw, adversarial identity removal costs a little rather than buying
anything. Combined with F38 (repairs the leg probe to chance and no further) and the Stage 1 result
that it made transfer 1.2x worse, the picture is consistent: **the adversary moves identity metrics
and never moves transfer.**

**What does survive re-measurement, and gets stronger.** The `z` ablation:

| held-out 4-leg | real aligned `z` | zero `z` | shuffled `z` | random backbone |
|---|---|---|---|---|
| base geometry | 1.86 | 2.49 (1.34x) | 3.35 (1.80x) | 5.06 |
| **held-out geometry** | **1.99** | 3.02 (**1.52x**) | 4.14 (**2.08x**) | 5.35 |

An aligned latent matters *more* when the target geometry was never trained on, which is the
direction that supports the claim rather than the one that undermines it. The few-shot curve also
reproduces: 2.63x / 2.81x / 2.97x / 2.94x at budgets 1/3/5/7, against 2.61 / 2.72 / 2.91 / 2.80 on
the base build.

**The rule this establishes.** Any claim measured on a body cut from a training body is provisional
until re-measured on a held-out one. F48 demonstrated that for one number and the demonstration was
not generalised; two of three re-measurements held and one reversed, which is roughly the hit rate
that makes the check worth doing every time.

---

### F60. Ramping the speed inside each clip flips the direction that would not transfer

F58 left one cell failing. `insect->b1` read -1.93 on seed 0 and +0.20 on seed 1 -- the only cell
that ever changed sign -- while `b1->insect` reproduced at +0.407 and +0.377. The diagnosis was
that the failing direction is the one *fitted on the noisier side*: between-clip speed variation
against within-clip rocking is **7.28 on the B1 and 1.45 on the insect** (F57), and a readout
fitted on the noisy side does not transfer.

**The fix follows from the memorisation, not from the weight.** F58 measured the shared head at
0.077 train against 0.855 on held-out clips, where the target is standardised so **1.0 is exactly
"predict the mean"**. Five constant speeds give 12 distinct values across 32 clips, which is a
lookup table: `z` encodes which clip this is and the head reads off that clip's speed. Raising
`lambda_body` presses harder on a target that has 12 values. Making the target *continuous* removes
the table.

`collect_ik.py --speed_end` sweeps the rate across the clip instead of holding it constant. The
source path is walked at a linearly varying rate, so the sampling positions are the running sum of
that rate; renormalising to span the path exactly keeps distance constant and puts the whole change
into elapsed time. **All legs share the time map**, so inter-leg phase is untouched -- verified at
0.056 lag against the constant case's 0.061.

`data/ik_walk_speed7` = the five constant speeds plus both ramp directions, 91 clips from 105 after
`walk_check`. Both directions, because with up-ramps alone "later in the clip" and "faster" are the
same thing and a readout could learn clip position instead of speed.

| | body loss, train | held out | gap |
|---|---|---|---|
| five constant speeds, seed 0 | 0.0775 | 0.855 | 11.0x |
| five constant speeds, seed 1 | 0.0862 | **1.060** | 12.3x |
| **ramped** | 0.1010 | **0.705** | **7.0x** |

On the constant set the head sat **at or above 1.0** -- no better than ignoring `z` entirely. On the
ramped set it reaches 0.705, so it extracts something from clips it has never seen. First
generalisation the head has shown.

**And the failing cell flipped.** One seed, `speed7`, **both arms at epoch 60**:

| | insect->insect | b1->b1 | **insect->b1** | **b1->insect** |
|---|---|---|---|---|
| frozen encoder | 0.676 | 0.753 | **-0.046** | +0.131 |
| control, no term | 0.664 | 0.167 | **-7.083** | -2.357 |
| **+`L_body`** | **0.798** | **0.879** | **+0.544** | **+0.435** |

`insect->b1` went -1.93 -> +0.20 -> **+0.544**, and the frozen encoder is *negative* in that cell,
so the model is not preserving structure V-JEPA2 supplied -- it is creating structure the encoder
did not have. `b1->insect` holds at +0.377, matching seed 1 of the constant set exactly.

**Read against the control, not the encoder.** The encoder row is scored on **one frame** while `z`
is built from **two**, so `z` has motion available and the encoder does not -- that comparison is
loaded in our favour and should not carry the claim. The control has identical two-frame access and
identical data, differing in this one term, and sits at -7.102.

**Costs and caveats.**

Val motion 0.0166 -> 0.0259, **+56 percent** on the metric the deck reports as its headline.
**Superseded by F65**: this is the cost at `lambda_body 0.5`; at 0.1 it is -2 percent.

**One seed on `speed7`**, and `insect->b1` is precisely the cell that has already flipped once. A
second seed is running.

**`lambda_body 0.5` was never swept.** It was copied from `lambda_cross` because that value works
in Stage 1. Whether 0.1 keeps the transfer at a quarter of the cost is unmeasured, and it decides
whether this reads as a trade or as nearly free.

**The head still memorises**, 7x train-to-val against the constant set's 11-12x. Reduced, not
solved.

---

### F61. The switch was a property of the data, and the leg probe was never evidence

F55 built a causal chain ending "the trunk partitions by robot instead of sharing", and slide 19
rested on it. Two of the three measurements behind that chain do not survive contact with the
speed-varied data, and one of them was never valid.

**The swap test reverses on speed-varied training data, with no loss term.** Body A's frame with
body B's latent, on `c10f10t10` and `c10f06t06`, whose commands differ by 21.1 deg:

| trained on | reads body identity from | strength |
|---|---|---|
| `ik_walk_8body`, one speed | **the latent** | 3.1x / 3.8x |
| `ik_walk_speed5`, five speeds, **no term** | **the frame** | 2.9x / 3.9x |
| `ik_walk_speed7`, seven conditions, **no term** | **the frame** | 5.0x / 3.7x |
| `ik_walk_speed7` + `L_body` | the frame | 4.8x / 4.5x |

Almost a mirror image, and **the controls did it** -- `lambda_body 0.0`, and the checkpoint confirms
no shared head was ever built. `L_body` adds nothing here (4.8/4.5 against 5.0/3.7, within noise).

**The confounds were checked.** Both runs trained on the same four bodies (`ik_walk_8body`'s nine
minus the non-walkers leaves five; `stage2_clean` held out three and `speed5` held out one, landing
on the same four), the same 5 clips per body, the same 60 epochs, the same architecture. And
scoring `stage2_clean` on the *speed-varied clips* still gives the latent at 3.1x/3.8x, so it is
the training data that matters and not the evaluation data.

This is what `lambda_cross` does in Stage 1 and what the adversary never managed (F59). **It was
achieved by making the insect walk more than one speed.**

**Isolated 2026-08-18.** `ik_walk_8body` and `ik_walk_speed7` are separate collection runs, so
"speed variation" needed separating from "the newer dataset". `stage2_8body_matched` retrains on
the **old** data with the **new** split -- `heldout_bodies c08f09t09` alone, 5 clips per body, 60
epochs, same architecture and seed.

| trained on | reads body identity from |
|---|---|
| `ik_walk_8body`, 3 bodies held out (`stage2_clean`) | **latent**, 3.1x / 3.9x |
| `ik_walk_8body`, **1 body held out** (matched) | **latent**, 3.2x / 4.3x |
| `ik_walk_speed7`, 1 body held out | **frame**, 2.9-5.0x |

**Changing the split does nothing.** The old data still produces the switch. Ruled out alongside
it: clips per body (5 in both) and frame clipping -- both datasets measure **0% of frames touching
the image edge**, so the framing-default fix is not the difference either.

What remains between the two datasets is the speed variation.

**And the per-leg contact probe was never evidence about the latent.** Slide 14 led with `z`
reading a loaded leg at 0.377 across, below the frozen encoder's 0.531 and below chance. Measured
again:

| | insect->insect | b1->b1 | insect->b1 | b1->insect |
|---|---|---|---|---|
| frozen encoder | 0.806 | 0.941 | 0.531 | 0.547 |
| `stage2_clean` | 0.802 | 0.986 | 0.377 | 0.398 |
| `speed7` control | 0.842 | 0.989 | **0.586** | 0.515 |
| `speed7` + `L_body` | 0.808 | 0.937 | 0.536 | 0.513 |

That 0.586 looks like the first leg-level cross number to beat the encoder. **It is one leg.** Right
hind scores 0.931 and the mean is 0.586, so the other three average **0.471 -- below chance**. In
the `L_body` run the same leg scores 0.404 while the other three average 0.580: **a different leg
carries it.**

**And the matched run settles it.** `stage2_8body_matched` and `stage2_clean` train on *identical
data* and differ only in the split. The leg probe reads **0.377 and 0.562** -- a 0.19 swing -- while
the swap test on those same two runs reads 3.1x/3.9x against 3.2x/4.3x and the forward model differs
by **0.014**. A metric that moves 0.19 on a change the representation does not register is not
measuring the representation. Per leg the matched run reads **0.30 / 0.42 / 0.74 / 0.79**: two below
chance, two well above, in one run.

**A four-leg mean with one leg jumping is not a transfer result**, and F56 already said why the
quantity cannot transfer -- one leg's phase fixes all four of the B1's and almost none of the
insect's other five. **A bad score on an ill-posed question is evidence about the question.** The
probe was how the problem was found; it should not have been the headline number, and the deck now
leads with body speed instead, which is a quantity both robots genuinely share.

**What still stands.** Body-speed transfer across robots fails on every control regardless of data
-- -4.60/-24.36 on one speed, -7.10/-2.33 on seven conditions -- and only `L_body` moves it (F58,
F60). That is the failure the loss term exists for, and it is the one the data does not fix.

---

### F62. `best.pt` selected on a metric only one arm had, and it broke a matched pair

`best.pt` is written when validation **total** improves. `total` carries whichever loss the run
enables, so a run with `lambda_body 0.5` is checkpointed on a quantity its control does not have.
The body term generalises poorly -- F60 measured 0.10 train against 0.71 held out -- so it makes
validation total noisy and its minimum arrives early.

| run | `best.pt` epoch |
|---|---|
| `speed7` control | **59** |
| `speed7` + `L_body` | **28** |
| `speed5` control | 60 |
| `speed5` + `L_body` | 49 |

**Every comparison drawn from `best.pt` compared a half-trained model against a fully trained one**,
in a design whose entire point is that the two arms differ in one flag.

**It produced one wrong conclusion.** The forward-model rollout on B1 video read 1.42x for the
control against 1.37x for the body arm at one step, widening to 1.33x against 1.15x at ten -- which
reads as the term costing the forward model something. Re-run from `last.pt`, epoch 60 against
epoch 60:

| steps ahead | control | +`L_body` |
|---|---|---|
| 1 | 1.42x | 1.42x |
| 3 | 1.56x | 1.55x |
| 10 | 1.33x | 1.31x |

**Identical.** The whole gap was the epoch difference. `L_body` neither costs the forward model
anything nor buys it anything.

**And it understated the result it was supposed to support.** At matched epochs every cell of the
body-motion probe improves:

| | insect->insect | b1->b1 | insect->b1 | b1->insect |
|---|---|---|---|---|
| epoch 28 (`best.pt`) | 0.783 | 0.809 | +0.432 | +0.377 |
| **epoch 60 (`last.pt`)** | **0.798** | **0.879** | **+0.544** | **+0.435** |

**Which comparisons survived, and why.** The swap test and the leg probe (F61) draw their
conclusion from *control against control* -- `stage2_clean` at epoch 60, `speed5` control at 60,
`speed7` control at 59 -- so they were epoch-matched already. The body-motion probe compared arms,
but the mismatch ran *against* the winning arm, so its result was conservative rather than wrong.
Only the forward-model test had the mismatch aligned with the effect it was measuring.

**Fixed at the source.** `compute_losses` now emits `selection = lambda_recon * recon +
lambda_motion * motion` -- the two terms every run has -- and `best.pt` selects on that. `total`
still reports everything. Same reasoning that already keeps the probe loss out of the selected
number, which `losses.py` had commented on and which the body term was added without following.

---

### F63. Behavioural variety moves the forward model, the module nothing else moved

F51 measured the forward model as the least responsive part of this pipeline: the coverage
intervention that took the decoder from 12.67 deg to 3.27 -- a factor of 3.9 -- moved the forward
model **5-8 percent**. Nothing else has moved it at all.

Giving the insect five speeds instead of one moves it by the same order, and does so on every
comparison available.

**Design, because the obvious version of this test cannot work.** `stage2_clean` trained on
`ik_walk_8body` and `stage2speed7ctrl` on `ik_walk_speed7`, so scoring on either set puts one model
in-distribution and the other out. Scoring on **both** separates the two explanations: if each wins
at home it is distribution match, if one wins everywhere it is a better forward model. Rolled on
its own output against holding the frame still, four clips per cell, **both checkpoints at epoch
60**.

| body | scored on | | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|---|
| `c10f10t10` | `ik_walk_8body` | `stage2_clean` | 1.38x | 1.57x | 1.50x | 1.29x |
| | | **`speed7` ctrl** | **1.43x** | **1.66x** | **1.60x** | **1.37x** |
| | `ik_walk_speed7` | `stage2_clean` | 1.35x | 1.47x | 1.43x | 1.23x |
| | | **`speed7` ctrl** | **1.41x** | **1.58x** | **1.56x** | **1.35x** |
| `c10f06t06` | `ik_walk_8body` | `stage2_clean` | 1.32x | 1.44x | 1.40x | 1.22x |
| | | **`speed7` ctrl** | **1.39x** | **1.54x** | **1.50x** | **1.30x** |
| | `ik_walk_speed7` | `stage2_clean` | 1.27x | 1.40x | 1.38x | 1.19x |
| | | **`speed7` ctrl** | **1.34x** | **1.52x** | **1.50x** | **1.29x** |

**24 of 24 comparisons**, two bodies at opposite ends of the leg-length range (0.77 m and 0.47 m),
both evaluation sets, six horizons each. Mean gain **+7.0 percent**; **+5.6 at short horizons and
+7.8 at long ones**, so it grows with rollout length -- the direction that matters for a module
whose only job is to be rolled.

**Distribution match is ruled out by the cell that should have gone the other way.** On
`ik_walk_8body` clips -- which `stage2_clean` trained on and `speed7` never saw -- `speed7` still
wins at every horizon on both bodies. A model predicts the other model's training data better than
its owner does.

**This is the fourth thing behavioural variety fixed today**, and the running total is worth stating
plainly:

| | fixed by |
|---|---|
| the decoder reads body identity from `z` rather than the frame | **data** (F61) |
| a body-level question is answerable at all | **data** (F57) |
| the forward model's rollout | **data**, +7% (this) |
| `z` carries body speed across the two robots | **the loss term** (F58, F60) -- every control stays at -7.1 |

**Isolated 2026-08-18, same run as F61.** `stage2_8body_matched` -- old data, new split -- rolled on
the same two evaluation sets:

| scored on | | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|
| `ik_walk_8body` | `stage2_clean` | 1.38x | 1.57x | 1.50x | 1.29x |
| | **matched, old data** | 1.38x | 1.57x | 1.49x | 1.26x |
| | `speed7` control | **1.43x** | **1.66x** | **1.60x** | **1.37x** |
| `ik_walk_speed7` | `stage2_clean` | 1.35x | 1.47x | 1.43x | 1.23x |
| | **matched, old data** | 1.35x | 1.46x | 1.41x | 1.19x |
| | `speed7` control | **1.41x** | **1.58x** | **1.56x** | **1.35x** |

**The two single-speed runs differ by 0.014 on average across twelve cells.** The split does
nothing. Against the *matched* run, speed variation gains **+7.9 percent on 12 of 12 horizons** --
so the gain is not the newer collection, the different split, or the framing fix.

**Smaller caveat.** Four clips per cell and two bodies. The effect is consistent in sign across
every one of the 24, which is what makes it credible at this sample size rather than the magnitude
of any single number.

---

### F64. The shared head only constrains the latent if it cannot tell the robots apart

F58 added a body-motion head shared by both embodiments and it worked: `insect->b1` +0.544,
`b1->insect` +0.435, against controls at -7.08 and -2.36. But that head read `z` alone, which is
**not** what LAC-WM does -- their motion decoder is `MD(x_t, z_t)`, conditioned on the observation.
Ours deviated from the source method and from our own `MotionDecoder`, and the deviation was
recorded in a docstring rather than raised as a decision.

Two objections to it, both fair. It sits badly with a thesis whose claim is that vision carries the
information. And "the frame would let it take a shortcut" was asserted, never measured.

**So it was rebuilt properly**: one head on `MotionDecoder`'s shared `features(x_t, z)`, no
embodiment key, gradient reaching the trunk. `stage2_speed7_bodyframe`, same data, same weight,
same control, both at epoch 60.

| | insect->insect | b1->b1 | **insect->b1** | **b1->insect** |
|---|---|---|---|---|
| frozen encoder | 0.676 | 0.753 | -0.046 | +0.131 |
| control, no term | 0.664 | 0.167 | -7.083 | -2.357 |
| `z`-only head | **0.798** | **0.879** | **+0.544** | **+0.435** |
| **frame + `z` head** | 0.680 | 0.347 | **-10.475** | **-57.170** |

**The corrected version is worse than adding nothing.**

**And the cost told the same story before the probe did.** Val motion against the control: the
`z`-only head costs **+55 percent**, the frame version only **+12**. That looked like the redesign
being cheaper. It was the term having stopped doing anything.

**The mechanism is not the shortcut I assumed.** `scripts/diagnostics/body_head_ablation.py` zeroes
one input at a time on the trained frame-conditioned head:

| input | body loss | vs real `z` |
|---|---|---|
| real `z` | 0.3425 | 1.00x |
| **`z` zeroed** | 0.7932 | **2.32x** |
| frame zeroed | 0.6795 | 1.98x |

**The head uses `z`.** Deleting it costs 2.32x, so `z` is carrying speed -- just a *robot-specific*
code for it. The frame tells the head which robot it is looking at, so it learns one mapping per
robot and `z` never has to agree with itself across embodiments.

**That is the per-embodiment head problem re-entering through the image.** We removed the embodiment
*key* and handed the head a photograph, which identifies the robot just as well.

**The rule this establishes.**

> A shared decoding head constrains the latent only if it is **blind to embodiment**. Any input that
> identifies the robot lets it decode conditionally, and conditional decoding is what the term
> exists to prevent.

**Why LAC-WM does not hit this.** Their target is end-effector and camera pose in a setting where a
still frame does not hand you the answer. Body speed *is* readable from one still frame -- the
frozen encoder scores R^2 **0.676** on it within an embodiment -- so conditioning on the frame gives
the head everything it needs and the bottleneck learns nothing. **The formulation does not port; the
principle behind it does.**

`z` is built from two frames, so the blind head still reads vision -- through the bottleneck being
shaped, which is the point. Nothing in the deployed system is denied the image: the ITM, the FTM and
the joint heads all take `e_t`. Only this one auxiliary head, whose job depends on having no
shortcut, does not.

**Kept reproducible.** `cfg.body_sees_frame` defaults to False and rebuilds the failed variant when
set, because a negative result nobody can re-run is not much of one.

**What this costs the earlier claim.** The `+0.544` stands, and its architecture is now justified by
measurement rather than by assertion. What does not stand is describing it as LAC-WM's mechanism
ported over: the port was tried, and it fails here for a reason specific to locomotion.

---

### F65. The 56 percent cost was the loss weight, not the mechanism -- and the probe is the noisy part

F58 and F60 reported `lambda_body 0.5` costing **+55 percent on validation motion**, and called it
the trade the mechanism demands. It was not. 0.5 was copied from `lambda_cross`, where it works in
Stage 1, with no argument that it should transfer to a different loss on a different target.

Swept at `lambda_body 0.1`, everything else identical, same control (`stage2speed7ctrl`, epoch 60):

| | val recon | val motion | cost | body loss, train | held out |
|---|---|---|---|---|---|
| control | 1.5608 | 0.0167 | -- | -- | -- |
| λ=0.5 | 1.5655 | 0.0259 | **+55%** | 0.1010 | 0.705 |
| **λ=0.1** | **1.5603** | **0.0164** | **-2%** | 0.1032 | **0.668** |

**A fifth of the weight reaches the same body loss at no measurable cost**, and validation recon and
motion both land marginally *better* than the control. The mechanism is close to free at the right
weight.

**But the transfer numbers are noisier than they look, and this nearly produced a false claim.**
The first reading was "λ=0.1 improves alignment as well" -- +0.675 against λ=0.5's +0.544. Then seed
1 of λ=0.5 landed at **+0.749**:

| | insect->b1 | b1->insect |
|---|---|---|
| control | -7.083 | -2.357 |
| λ=0.5 seed 0 | +0.544 | +0.435 |
| λ=0.5 seed 1 | **+0.749** | **+0.704** |
| λ=0.1 seed 0 | +0.675 | +0.624 |

**λ=0.1 sits inside λ=0.5's own seed spread.** One seed cannot separate the weights on transfer, and
claiming it did would have been reading noise.

**Where the pipeline is and is not seed-stable:**

| metric | spread across two seeds |
|---|---|
| val total | **0.7%** |
| val recon | **0.9%** |
| val motion | 14% |
| **probe `insect->b1`** | **27%** |

Training is stable to under a percent. The probe is not, because it is a *downstream* measurement of
a property nothing directly optimises -- a fresh ridge fitted on `z` and applied across robots, where
small differences in the latent's geometry become large differences in transferability.

**The effect still dwarfs it.** Control-to-treatment on the probe is **7.6**; the seed spread is
**0.20**, a factor of 38. So *"the term makes body speed transfer"* is far outside seed variance.
*"This weight beats that weight"* is not, and needs a second seed before it is claimed.

**What to report.** The range across seeds -- `insect->b1` **+0.54 to +0.75** -- not a single run.
And the cost as a property of the weight, with λ=0.1 as the operating point, since it is a
within-seed comparison against a shared control and therefore does not suffer the same problem.

**Still one seed at λ=0.1.** The zero cost is the only thing separating the two weights, and it has
not been reproduced.

---

### F66. Correlating the two readouts' *predictions* is the stable statement; correlating their weights is chance

F65 left the probe as the noisy part -- `insect->b1` moving 27 percent across two seeds whose
training metrics move under one. The R^2 cells are a harsh way to ask the question: a readout is
fitted on one robot and applied to the other, so it is charged for scale and offset on top of
direction, and it is unbounded below, which is why the control reads **-7.083** with no way to say
how bad that is.

`body_motion_probe.py` now also reports **agreement**: fit a readout on each robot separately, run
both over the same frames, correlate the outputs. Bounded, symmetric, and blind to the scale and
offset R^2 punishes.

| | insect->b1 R^2 | b1->insect R^2 | **agreement** |
|---|---|---|---|
| frozen encoder | -0.046 | +0.131 | **0.313** |
| control (λ=0) | -7.083 | -2.357 | **-0.014** |
| λ=0.5 seed 0 | +0.544 | +0.435 | **0.845** |
| λ=0.5 seed 1 | +0.749 | +0.704 | **0.915** |
| λ=0.1 seed 0 | +0.675 | +0.624 | **0.898** |

**Two things become sayable that R^2 could not say.**

The control's -7.083 becomes **-0.014**: the two robots' speed readouts are *uncorrelated*. Not
inverted, not distorted -- unrelated. And the frozen encoder's -0.046 becomes **0.313**: V-JEPA2
already leaves a partial shared ordering, which a single linear readout simply cannot exploit. Both
are more precise claims than "no transfer", and neither was available from a number with no floor.

**It is meaningfully more stable, but it does not rescue the weight comparison.** Across the two
λ=0.5 seeds the spread is **8 percent** of the mean against R^2's 32 -- a factor of four -- yet the
*ordering* survives (0.845 < 0.898 < 0.915 against 0.544 < 0.675 < 0.749). Since both metrics rank
the three runs identically, part of the seed gap is a real difference in the latent's geometry, not
only readout brittleness. **λ=0.1 still sits inside λ=0.5's seed spread on this metric too**, so
F65's refusal to separate the weights stands, now on the measurement better suited to it.

**Comparing the fitted weight vectors directly does not work, and was the first thing tried.** `z`
is 64-D and heavily correlated, so a ridge's coefficients are not identified -- most of their norm
lies in low-variance directions that barely move a prediction. Measured that way every run sat at
chance, **0.014 to 0.085**, including the run with the best transfer of all:

| | insect->b1 R^2 | |cos(w_A, w_B)| |
|---|---|---|
| control | -7.083 | 0.085 |
| λ=0.5 seed 0 | +0.544 | 0.054 |
| λ=0.5 seed 1 | **+0.749** | **0.014** |

A readout that transfers at +0.749 cannot be built on an axis orthogonal to the other robot's. That
contradiction is what exposed the error. Correlating predictions instead weights each direction by
how much the data varies along it, so the unidentified part drops out.

**General form: to ask whether two fitted models agree, compare what they predict, not what they
weigh.** In any correlated feature space the parameters carry a large component the predictions
never see.

**Two corrections that fell out of reporting `r` beside R^2.**

*The asymmetry between the two directions was mostly calibration.* `b1->insect` moves 0.435 -> 0.704
across seeds, 62 percent, while its `r` moves **0.852 -> 0.863**, 1.3 percent. The direction is the
most stable quantity in the whole table; the swing is gain and offset. The sentence above that reads
"part of the seed gap is a real difference in the latent's geometry" holds for `insect->b1` (r 0.743
-> 0.879) and **not** for `b1->insect`.

*Embodiment identity is **linearly** carried by the per-feature mean and scale.* A classifier held
out **by clip** reads **AUC 1.000** on raw `z` and **0.441 / 0.459** once each embodiment is
standardised -- which is the representation the probe measures in. So the transfer results are not
identity leaking through a linear channel.

**But "one cloud, shifted" overstates it, and this is now measured rather than read off a picture.**
`scripts/diagnostics/identity_linearity.py` runs three classifiers on the same features:

| classifier | raw `z` | standardised |
|---|---|---|
| linear (logistic) | 1.000 | **0.460** |
| nonlinear (random forest) | 1.000 | **0.999** |
| nonlinear (MLP) | 1.000 | **1.000** |

**Standardising removes identity from what a straight line can use, and from nothing else.** The
UMAP was right to show the robots apart. Two things follow, and they must be kept apart: the probe's
transfer numbers are trustworthy, because a linear ridge cannot exploit what a linear classifier
cannot find; and **no claim of the form "the latent forgets the body" is available to us**, because
a nonlinear reader recovers the robot exactly.

This is the `center_embeddings` lesson again -- subtracting each robot's mean embedding let the
online probe climb back to 1.000 within 25 epochs. First and second moments are not where identity
lives.

(An earlier version of this check reported 0.212, below chance. It split folds by *frame*, and
neighbouring frames of one clip are near-duplicates. Splitting by clip is the fix.)

**What to report.** Agreement as the headline -- **-0.014 for the control against 0.85 to 0.92 with
the term** -- with the R^2 cells kept beside it as the practical question of whether one robot's
readout can be *used* on the other, and `r` beside those to say which part of a weak cell is
direction and which is calibration.

**Spearman was measured and dropped.** It tracked Pearson within 0.013 on every run, so the
straight-line assumption costs nothing and the column was not worth carrying.

---

### F67. Read against the source paper: our architecture is the *baseline* it beats, plus a patch

The comparison in F24 and F64 was made against a summary of LAC-WM, not the text. Read properly
(`doc/LATENT ACTION ROBOT FOUNDATION WORLD MODELS FOR CROSS-EMBODIMENT ADAPTATION.pdf`, ICLR 2026
submission), three things stated here were wrong and one was right for a better reason than the one
given.

**Wrong 1: "LAC-WM has no alignment term."** It has exactly one, and it is the same shape as ours:

> "LAC-WM uses continuous latent actions and mitigates shortcuts through an **auxiliary motion
> decoding loss**" -- `L = λ_recon·L_recon + λ_motion·L_motion`

And its Figure 2 is our control experiment: "IDM trained **without** the motion decoder (MD), where
Agibot and Egodex cluster together but separate from Droid." **No auxiliary head, disjoint space.**
That is `stage2speed7ctrl` at r = -0.048, arrived at independently.

**Wrong 2, then wrong again in the other direction.** The first version of this entry said their MD
never emits different-dimensional output and concluded our per-embodiment heads were the whole
divergence. Appendix A.2 says otherwise:

> "In **Droid**, it is **ten-dimensional** for a single arm... In **Agibot**, the end-effector pose
> has **twenty dimensions**, ten per hand... In **EgoDex**, the end-effector pose includes the
> nine-dimensional wrist pose **plus sixty dimensions representing finger positions**... for a total
> of **138 dimensions** for both hands."

Ten, twenty-nine and one hundred forty-seven (with the 9-D camera pose). **The paper does not say
how one decoder emits three different widths**, and it should not be guessed at: per-dataset final
layers, one wide layer with each dataset supervising only its own slots, and zero-padding all three
to a common width are all consistent with what is written ("the resulting cross-attended features
are fed into **an MLP** to produce motion outputs"). An earlier version of this entry asserted
per-dataset heads as fact; that was an inference, and the masked or padded designs would mean the
output weights are **shared**, which is a stronger form of the mechanism than the one described
here, not a weaker one.

What does not depend on the answer: their labels all live in **one physical space**, and ours do
not. So having per-embodiment output heads is *not* what separates us from
LAC-WM, and the "we built EAC-WM" reading is too strong.

**What actually separates us is the coordinate the heads predict, not their number or size.**

| | their per-dataset outputs | ours |
|---|---|---|
| quantity | wrist pose, fingertip positions, camera pose | **joint angles** |
| coordinate | 3-D position and rotation in a common physical frame | **body-specific joint space** |
| shared meaning across bodies | **yes** -- a fingertip at (x,y,z) means the same thing for a human hand and a robot hand | **no** -- "leg 3 TC angle" has no referent on a B1 |

Different *counts* of the same physical quantity is their entire setup: one arm, two arms, two hands
with fingers. Different *quantities* is ours. **A shared trunk can align representations whose heads
differ in width; it cannot align heads whose outputs have no common referent.**

Their labels are also far richer than ours -- 10 to 147 dimensions against our **one**. The claim in
F66 that a one-dimensional target aligns one direction is not a limitation of the method; it is what
we asked for.

**Wrong 3: "the unified coordinate is their main target."** It is not. `L_recon`, next-frame
prediction, carries the task; the MD is auxiliary and its stated purpose is to *mitigate shortcuts*,
with alignment as the demonstrated consequence. Gait-level detail in their setting survives in `z`
through frame prediction, not through the motion label.

**Right, for a better reason: the frame-conditioned head (F64).** Their MD *does* see the frame --
"`z_t` serves as a query in a cross-attention module over visual tokens extracted from the current
frame", `â_t = MD(x_t, z_t)` -- which is the design that scored -10.5/-57.2 for us. The paper never
states the condition under which that is safe, because it never has to: **its target is a delta**
("**delta** human hand poses, robot end-effector actions, and camera motion"), and a single still
cannot supply a delta. Ours is a *state* a still supplies at R^2 0.676. So F64's rule is not a
deviation from the paper; it is the hidden precondition the paper satisfies for free.

**They also measure it with UMAP, on 7,000 embeddings from three datasets.** Our own note above --
that a UMAP read as cleanly separated while the silhouette was +0.140 -- applies to that evidence
too, and the `r`/agreement numbers in F66 are a quantity their figure does not report.

**The "sufficiency" argument made here first was wrong and is withdrawn.** It said manipulation has
a unified label that nearly determines the task while locomotion does not, so their recipe cannot
port. But a 7-DoF arm reaching a 6-DoF pose has a null space too, and EgoDex's label includes every
fingertip -- they did not discard the fine articulation, they *labelled* it. The asymmetry was an
artefact of comparing their 147-dimensional label to our one-dimensional one.

**What the port actually requires** -- **items 1 to 3 are withdrawn by F69**, which measures that
the foot coordinate adds only the part that does not transfer, and that the binding constraint is
behavioural coverage rather than the choice of coordinate. Item 4 stands:

1. **The shared coordinate for locomotion is foot position, normalised by leg length.** Their
   end-effector is where the body touches the world; ours is the feet. Six feet against four is the
   same kind of mismatch as one arm against two hands.
2. **The normalisation is the step locomotion adds.** Their robots are of comparable size; ours have
   hip heights of 0.13 m and 0.56 m, so raw foot positions announce the embodiment and walk straight
   into F64. Dividing by leg length is the same move Froude makes for speed.
3. **Their `z` is split -- "the first half decodes the end-effector pose, and the second half decodes
   the camera pose".** The locomotion counterpart is feet in one half and body twist in the other:
   end-effector maps to feet, camera pose maps to how the body itself moves.
4. **F64 is the precondition they satisfy for free** (their target is a delta; ours is a state).

Every measurement in F58-F66 stands. What moves is that they describe **the joint-angle-target
version** of the pipeline, which the plan now supersedes.

---

### F68. Chunking does not turn our target into a delta -- looking *forward* does

F67's plan said to copy LAC-WM's 5-step action chunking so our motion target would stop being a
state a single frame supplies. **Measured before spending a retrain on it, and the premise is
wrong.** `scripts/diagnostics/target_window_sweep.py` fits a readout from **one frame's** frozen
embedding to the forward speed averaged over W steps:

| read from one frame, R^2 | W=1 | W=2 | **W=5** | W=10 | **W=20 (1 s)** |
|---|---|---|---|---|---|
| insect | 0.627 | 0.645 | **0.670** | 0.595 | **0.246** |
| b1 | 0.491 | 0.480 | **0.420** | 0.295 | **0.361** |

**Shorter windows are no harder to read from a still, and for the insect they are slightly easier.**
The reasoning that made chunking look right is a manipulation intuition: a still of an arm says
little about where the gripper goes next. A still of a *walking* robot shows the leg configuration,
which is most of the gait phase, which largely fixes the instantaneous velocity. Locomotion leaks
short-horizon motion into a single frame in a way manipulation does not.

**What does change it is the direction of the window, not its length.** Today's target is body speed
smoothed over one second **centred** on the frame -- frames t-10 to t+10 -- and reads **0.676**. The
same one-second span taken **forward** from the frame, `(x[t+20] - x[t]) / 20dt`, reads **0.246**.
The frame sits at the start of the window instead of its middle, so it stops being half the answer.

**Both halves of the condition, and a forward horizon passes both.** F64 requires the target to be
unrecoverable from the head's other inputs; it equally requires the target to be recoverable from
`z`, or the auxiliary loss adds noise rather than a constraint. Reading the same targets out of a
trained `z` (`--ckpt stage2speed7body`):

| R^2 | | W=1 | W=2 | **W=5** | W=10 | **W=20** |
|---|---|---|---|---|---|---|
| insect | frame | 0.627 | 0.645 | **0.670** | 0.595 | **0.246** |
| insect | **z** | 0.539 | 0.557 | **0.611** | 0.664 | **0.422** |
| b1 | frame | 0.491 | 0.480 | **0.420** | 0.295 | **0.361** |
| b1 | **z** | 0.610 | 0.609 | **0.578** | 0.464 | **0.538** |

At the source method's 5-step chunk the frame beats `z` on the insect (0.670 against 0.611); at a
one-second forward horizon `z` leads by 0.18 on both robots.

**That comparison is much weaker evidence than it first looks, and the first version of this entry
oversold it.** `z = ITM(e_t, e_{t+1})` is built from **two** frames. On any forward-looking target it
therefore holds information a single frame cannot have, so `z > frame` is close to tautological and
says little about whether the target is a good one. Add that the two feature spaces have different
widths (64 against 5,632), that this is one seed of one checkpoint, and that `z` explains only 0.42
and 0.54 in absolute terms, and the margin cannot carry a claim on its own.

**The measurement F64 actually calls for is incremental**, because the head receives the frame and
`z` *together* and can use both: does the frame add anything **on top of** `z`? No gain means
nothing to shortcut to; a large gain means the shortcut survives however far ahead `z` is. That is
`source = z+frame` in the same script, and it is the number to quote.

> **Caveat, and it needs fixing before this is quoted.** `W` counts steps, and the two robots run at
> different rates -- 0.05 s for the hexapod, 0.02 s for the B1. **W=20 is 1.0 s of insect and 0.4 s
> of B1**, so the columns are not time-matched across rows. The trend within each robot stands; the
> cross-robot comparison at a fixed W does not. Re-run matched by seconds, or by fraction of a
> stride, before this number goes in a thesis.

**Consequence for the plan.** Step 2l ("chunk actions to 5 steps") is withdrawn as stated.
**Superseded by F69**: choosing a horizon by the `z`-minus-frame margin is also the wrong lever --
the constraint is that only one shared *behaviour* varies in our data, not how the target's window
is shaped.

---

### F69. What the two robots share is at body level, and only one channel of it varies

F67 proposed decoding **foot geometry** as the locomotion analogue of LAC-WM's end-effector, and F68
proposed reshaping the target's time window. Both were arguments from the source paper's structure.
Checked against measurements this project already had, **the coordinate was never the binding
constraint** -- and the foot proposal is contradicted by our own data.

**Three gates a shared auxiliary target has to pass**, assembled from four separate measurements
that were each made for another purpose:

| candidate | does it vary? | does it hide the robot? | does its variation mean the same thing on both? | |
|---|---|---|---|---|
| duty factor | **no** -- 0.533 against 0.515 (F45) | yes | -- | fails |
| lateral speed | yes | **no** -- AUC 0.788 | -- | fails |
| which leg is loaded | yes | yes | **no** -- transfers at **0.373**, below the frozen encoder's 0.531 and below chance (F41b) | fails |
| **forward Froude** | **yes** | **yes** | **yes** -- agreement 0.85 to 0.92 (F66) | **passes** |

We screened these one at a time over months without noticing we were applying a single rule. Only
one quantity we own passes it.

**Why the foot target fails, in terms of the same gates.** Foot motion in body frame splits in two:

| part | what it is |
|---|---|
| stance feet move backward at body speed ÷ leg length | **body speed rewritten** -- nothing beyond `lambda_body` |
| which foot is in stance when | **the gait**, which is the 0.373 row above |

**Everything a foot target adds beyond body speed is exactly the part measured not to transfer.**
The B1 trots and the insect walks a six-leg wave, so a per-leg readout fitted on one is
systematically wrong on the other. This is F45's structural argument arriving from a third
direction: coarsen a leg-level label enough to describe both robots and you have destroyed what made
it meaningful.

**So the shared head reached one axis for a reason that has nothing to do with coordinates.** Body
twist is six-dimensional in principle. In our data:

| channel | status |
|---|---|
| forward speed | varies -- five speeds plus ramps |
| lateral speed | **zero in every B1 clip** |
| yaw rate | **constant per policy** |
| acceleration | only at each clip's start and stop |

**Five of six channels are constants.** A shared head trained on that can align exactly one
direction, which is what it did. There was never more available to align.

**The blocker is behavioural coverage, and clearing it also unlocks the published mechanism.** With
one behaviour, the current state fixes the future, so `z` holds nothing a single frame lacks -- which
is why F64 forced us to blind the head, and why the alignment left the forward model at 1.42x
either way. Give one state several possible futures and the frame can no longer say whether the
robot is about to turn, stop or accelerate. **Then the frame-conditioned decoder LAC-WM publishes
should run here as written**, instead of in the blinded variant we had to build.

**Withdrawn by this entry**: F67's foot-coordinate proposal and F68's "pick a horizon by the
`z`-minus-frame margin". Both measured the wrong lever. The window sweep in F68 stands as a
description of the target -- short windows stay frame-readable, forward ones less so -- but it is not
the fix.

---

### F70. Every body channel splits into behaviour and gait, and only behaviour crosses robots

F69 argued that the shared head reached one axis because only one body channel varies in our data.
That was an inference from what the collectors command. `scripts/diagnostics/channel_screen.py`
measures it per channel, and the mechanism is sharper than the argument was.

**A body-velocity channel is a sum of two timescales**: the slow part is what the robot is doing,
the fast part is where it is in its gait cycle. Scoring each channel raw and smoothed over about one
stride separates them, on the same clips and the same checkpoint:

| channel | timescale | varies | robot AUC | insect->b1 | b1->insect |
|---|---|---|---|---|---|
| forward | per frame | 1.00 | 0.623 | **-1.453** | **-2.078** |
| **forward** | **smoothed** | **1.00** | **0.529** | **+0.544** | **+0.435** |
| lateral | per frame | 1.07 | 0.705 | -0.999 | -1.939 |
| lateral | smoothed | 1.05 | **0.834** | -2.647 | -0.250 |
| vertical | per frame | 0.78 | 0.505 | -1.333 | -3.883 |
| vertical | smoothed | **0.20** | 0.510 | -0.471 | -3.460 |

**The forward channel is its own control.** Same axis, same clips, same weights -- the only
difference is whether the velocity is averaged over a stride, and it moves the cross-robot readout
from **-1.45 to +0.54**. Nothing else can account for that. So `lambda_body` does not work because
forward is a special direction; it works because forward is the one channel with a slow component
that is not zero.

**Three channels, three different gates failed** (the gates are F69's):

- **forward** varies, hides the robot at 0.529, and transfers. The only one that passes.
- **vertical** collapses when smoothed, 0.78 -> **0.20**. The bob is almost entirely gait; on level
  ground the stride-averaged height barely moves. Nothing there to align, however large the
  per-frame row looks.
- **lateral** keeps its variation, 1.05, but its robot AUC *rises* to **0.834** once smoothed. The
  slow part of lateral is "the B1 never drifts sideways and the insect does" -- embodiment identity,
  not shared behaviour. This is the 0.788 that forced `BODY_CHANNELS = (0,)`, now located: it is the
  slow component that betrays the robot, not the gait wobble.

**Smoothing is not uniformly good, and the two AUC columns show it.** It isolates behaviour, so if
the behaviour itself differs between the robots the identity gets *sharper* -- forward 0.623 ->
0.529, lateral 0.705 -> **0.834**.

**This is the measured form of the argument for more behaviour.** The other channels are not hiding
in the data waiting to be extracted: their slow components are either zero (vertical, and yaw and
pitch by the same logic) or robot-specific (lateral). **Turning, strafing and slopes would create
slow components that do not exist today**, which is the difference between "we could not align it"
and "there was nothing there to align".

**Still to check**: roll, pitch and yaw cannot be screened at all yet -- `collect_ik.py` records
`head` as a position and no orientation, so the hexapod has no measured body attitude. Adding it is
a few lines at collection time and a full recollection to use.

**And the balance is not what it is usually described as.** With `clips_per_body hexapod=7` the
training set is **2,780 hexapod frames against the B1's 1,143 -- 2.43:1**, not equal. What is equal
is *gradient steps*: `balance_embodiments` repeats the B1 about 2.4 times an epoch. `config.py`
already carries the warning ("balancing the sampler is not balancing the data") and it applies here.
Separately, the probe and the UMAP read the **uncapped** directory and see **5.9:1**, so any quoted
ratio has to say which of the two it means. Steps 2m and 2n.

---

### F71. A second insect gait, and the four wrong ways to get one

Step 2k needs behaviours the hexapod does not have -- turning, backing, speed range -- and
`--turn_bias`, the collector's existing steering knob, was measured on 2026-08-20 not to steer:
at most 8 degrees of heading against 30 degrees of natural gait wander, because it adds a constant
to a joint and moving where a leg sits does not change how far it pushes.

**What works: a joint-space oscillator, ported from the lab's `student_Locomotion_Control_olaf_6legs`
scene.** Two sinusoids a quarter cycle apart drive the three joints of every leg, signs flipped
between the two tripod groups. `--gait cpg` in `sim/collect/collect_ik.py`.

**It is a commandable gait, not a tripod.** The *drive* is a tripod -- the sign pattern puts FL HL MR
against ML FR HR exactly -- but the *contacts* are not, and calling it one anywhere below is wrong.
Measured over three repeats at the best settings: within-group agreement **0.641 +/- 0.008** against a
clean tripod's 1.0, across-group **0.691 +/- 0.007**, duty factors spread **0.227** across the six
legs. The lab's own Olaf scene does not produce one either -- its middle legs carry 260-636 N while
its corner legs carry 0.006-25 N -- so this is the behaviour of the pattern, not a porting error.
What the gait is good for is that it holds heading and takes commands; grouping is not what it
delivers.

| | travel | hip | Froude | heading change | wander |
|---|---|---|---|---|---|
| recorded wave | +0.59 m | 0.129 m | 0.161 | **+29°** | 1.29 |
| **oscillator** | +0.37 m | 0.132 m | 0.102 | **-6° +/- 2** | 2.28 |

**It holds its heading better than the recording does.** Speed comes from `--cycles`.

**Correction, 2026-08-22: `--turn` does not steer, and the heading numbers first written here were
read off the wrong axis.** `frame_axes` (collect_ik.py:140) establishes that in this scene the
abdomen's **z** runs front-to-back and its **x** is vertical; taking heading as rotation about z
therefore measures roll. On the same clip that convention reads **+79 deg** where the correct one
reads **-6**, against a measured wander of 2.28 -- the sign that it is wrong is that a clip which
barely deviates cannot have yawed most of a right angle. Re-measured over three repeats each:

| | heading | vs straight | travel | lateral | lateral/forward |
|---|---|---|---|---|---|
| straight | -6 +/- 2 | -- | 0.37 +/- 0.00 | -0.12 | -0.32 |
| `--turn -0.3` | +8 +/- 2 | **+14** | 0.21 +/- 0.01 | -0.09 | -0.44 |
| `--turn +0.3` | -4 +/- 3 | **+2** | 0.29 +/- 0.00 | 0.12 | 0.43 |
| `--spin 0.8` | -79 +/- 2 | **-73** | 0.09 +/- 0.01 | -0.37 | -4.16 |
| `--strafe 0.8` | +60 +/- 4 | +66 | 0.11 +/- 0.04 | **0.62 +/- 0.01** | 6.84 |

`--turn +0.3` moves the heading **2 degrees** and `-0.3` moves it 14 -- neither monotonic nor
symmetric, and both far inside `--spin`'s 73. **What `--turn` actually does is brake**: travel drops
from 0.37 to 0.21 m. That follows from its mechanism, which scales down one side's amplitude, so
both of that side's legs take shorter steps and the robot slows more than it yaws. The withdrawn
claim was `+0.3 -> +30 deg, -0.3 -> -18`.

**Use `--spin` for turning.** It is the drive F72's matched-yaw table is built on, so nothing there
is affected. `--turn` and `--turn_bias` were removed from `collect_ik.py` on 2026-08-22.

**Sideways is a gait, not a modifier -- and that is what makes it work.** `--strafe 0.8` on top of
forward walking translates 0.62 m sideways but yaws 66 degrees doing it. Splitting the drive by leg
row locates the yaw but does not fix it: the front row alone yaws **+1 deg**, the middle **+12**, the
hind **+80**, so it is not a front-against-hind couple and no set of per-row gains cancelled it --
the best of four tried still left 20 degrees. **Turning the fore-aft swing off does.** The yaw was
never coming from the sideways drive; it came from that drive **fighting a swing that was still
running** -- the hind legs asked to stride forward and splay outward in the same stroke, and the body
twists.

Three further corrections took it from a crawl to a real gait, and each was found by measuring
rather than by tuning:

*The splay-lift phase was inherited from forward walking and is wrong for sideways.* A leg has to
splay while planted and fold while lifted; `--ft_phase 0.125` had it splaying in the air and
dragging on the ground, which showed up as **80% of the stroke lost to slip** -- 0.24 rad of splay on
a 0.5-0.77 m leg should carry 0.12-0.18 m and carried 0.03. Sweeping the full circle, **0.5 --
exact antiphase, which is what the mechanics ask for -- is three times faster per cycle**.

*`--spin` was multiplying by zero in the one configuration that needed it.* It scaled by `amps[0]`,
the fore-aft amplitude, which this gait sets to **0.00**. A sweep of `--spin 0.15` against `0.25`
moved the heading by 1 degree and read as "spin cannot cancel this yaw" when spin had never been
applied. `--spin_amp` now gives it an amplitude of its own; **F72's matched-yaw table runs at
`amps[0] = 0.25` and is unaffected**.

*The residual yaw is a constant, so it subtracts.* Sideways at antiphase yaws **+20 deg +/- 2** --
repeatable, therefore cancellable. `--spin` moves it about 150 deg per unit and monotonically:

| `--strafe` | `--spin` | sideways | per cycle | forward | heading |
|---|---|---|---|---|---|
| -0.8 | 0 | 0.30 +/- 0.01 | 0.056 | +0.01 | +20 +/- 2 |
| -0.8 | 0.15 | 0.33 +/- 0.03 | 0.060 | +0.06 | -2 +/- 2 |
| +0.8 | -0.15 | 0.55 +/- 0.06 | 0.096 | -0.02 | -17 +/- 6 |
| **+0.8** | **-0.18** | **0.42 +/- 0.06** | **0.073** | +0.06 | **-2 +/- 4** |
| +0.8 | -0.20 | 0.39 +/- 0.04 | 0.068 | +0.08 | +3 +/- 5 |

Cancelling the yaw always costs distance -- `--spin 0` runs 0.55 m and 38 degrees off -- so the
setting is the largest `--spin` the heading can be brought to zero with, not the largest travel.

**The sideways gait is therefore, one setting per direction** (re-derived at `--scale 0.65`):

    right   --amps 0.00 0.20 0.30 --ft_phase 0.5 --strafe  0.8 --spin -0.24 --spin_amp 0.25
    left    --amps 0.00 0.20 0.30 --ft_phase 0.5 --strafe -0.8 --spin  0.19 --spin_amp 0.25

both with `--gait cpg --ik_iters 8 --scale 0.65 --symmetric`. **Right runs 0.52 m +/- 0.06 at -1 deg
of heading and is pure -- sideways travel is 1.00 of total displacement. Left runs 0.30 m +/- 0.01
at -0 deg, purity 0.98.** At the old `--scale 0.5` the same two ran 0.42 and 0.33.

**Widening the foot path raises the extend ceiling from `A[2]` 0.30 to 0.40, and 0.40 still loses.**
The headroom argument says take the larger: 0.40 clips nothing at this scale and travels **0.75 m**
against 0.30's 0.58. But it also yaws **-56 deg** against -38, and cancelling that costs more than
the extra distance is worth -- at zero heading, 0.30 lands at **0.54 m and 0.94** while 0.40 lands
at 0.46 m and purity 0.94, walking 0.17 m forward while it crabs. **The quantity to maximise is
travel after the yaw is removed, not before**, and a gait that generates less yaw to begin with
starts ahead. `A[2] = 0.50` clips three joints and is out regardless.

*The splay-lift antiphase is unchanged by the wider path*, as its mechanical argument predicts:
`ft_phase` 0.375 and 0.625 travel 0.15 and 0.06 m against 0.5's 0.54. Sideways is now the faster of the two per cycle, which is the opposite of where this
started.

**The two directions are not mirror images and must not be generated by flipping a sign.** They
differ in distance (0.42 against 0.33) and they need *different* yaw trims (-0.18 against +0.15),
because the animal is not symmetric: the leg pairs are 0.771, 0.489 and 0.638 long and the walking
pose the oscillator is centred on is itself asymmetric. Collect and measure each direction
separately.

**Reading a direction off the abdomen frame needs care.**
`frame_axes` identifies the abdomen's **z** as the fore-aft axis but not which end of it is the
front, and it points **aft**: its dot product with the direction straight walking actually travels
is **-0.96**. Taking it as forward reports left and right swapped. Heading *changes* survive this --
a constant 180 degree offset cancels in a difference, so every `a[-1] - a[0]` above stands -- but
any statement about which way the robot went does not. Fix the sign against straight walking's own
displacement before using it.

**The extend amplitude cannot be raised, and the reason is a joint limit rather than traction.**
Widening the splay is the obvious way to ask for more distance and it fails: `A[2]` at 0.35, 0.40
and 0.50 travels 0.09, 0.08 and 0.20 m against 0.30's 0.42, and the direction reverses twice.
Checking the commands against `sim.getJointInterval` shows why -- from 0.35 upward, three joints
(`FR_2 MR_2 HR_2`) are commanded past their limit and the waveform is clipped. At 0.30 nothing is
clipped and the margin is **0.06 rad**, in both strafe directions. This follows from the balance
knob: `|right| = A[2](1 + |strafe|)` puts 90% of the splay on one side, so raising `A[2]` pushes
only the side that is already full. **0.30 is the ceiling of this parameterisation, not of the
robot** -- the standing pose of the extend joints sits about 2.5 rad, 0.64 from the +pi limit and
5.6 from the other, so re-centring the bias would roughly double the available swing. Untried.

**Two things not to assume.** The amplitude is a **narrow optimum** -- at `ft_phase 0.125`, `--strafe`
0.6 and 1.0 scatter by a factor of ten across identical runs while 0.8 holds to +/- 0.03 -- so
retuning any one number needs the repeats. And **the two sideways directions are not mirror images**:
`--strafe +0.8` and `-0.8` at the same phase gave -0.55 and +0.30, different distances and different
yaw, so left and right have to be measured separately rather than assumed symmetric.

**Speed: widen the foot path, do not run the oscillator faster.** `--scale`, the shared foot-path
scale about each body's hip, defaults to **0.5** -- feet pulled halfway in, so that the shortest
legs in `ik_walk_8body` could still reach a shared absolute coordinate. Once the collection is one
long-legged body (F15: the hexapod morphology space is two-dimensional and a held-out body is
reconstructed from a 2-D basis to 0.203 deg, so extra bodies add almost nothing), that constraint
has no purpose -- **and it reaches the oscillator too**, because `--gait cpg` centres itself on the
mean of the IK commands, which were solved against the compressed targets. Three repeats:

| `--scale` | Froude | heading | hip | wander | frames | FT headroom |
|---|---|---|---|---|---|---|
| 0.5 | 0.100 +/- 0.000 | -6 +/- 2 | 0.132 | 2.23 | 66 | 0.29 rad |
| **0.65** | **0.131 +/- 0.003** | **-0 +/- 0** | 0.176 | 1.67 | 66 | 0.51 rad |
| 0.8 | 0.200 +/- 0.002 | -3 +/- 2 | 0.219 | 1.37 | 55 | 0.74 rad |

**Every column improves at once, which is not what tuning usually looks like** -- further, straighter
and less wandering. **0.65 is adopted**: it puts the hexapod at Froude 0.131 against the *already
collected* B1 clips at `--vx 0.30` (0.135), so the two robots match at a speed **neither has to
crawl to reach**, and the 66-frame clip length is preserved (0.8 finishes the travel gate early and
loses a fifth of its transitions).

**Two consequences to carry.** The body rides at **0.176 m against the recorded animal's 0.129**, so
this gait is matched to the B1 rather than to the insect it is modelled on -- a deliberate choice,
not a drift. And the **extend-joint headroom nearly triples, 0.29 to 0.51 rad**, which withdraws the
conclusion below that `A[2]` cannot be raised: it was clipped because the standing pose was
compressed, not because the robot ran out of joint.

**Superseded: `--cycles 9.3` also reaches the animal's Froude.** At the default 6 it walks
at **0.100** against the recorded wave's 0.161 and the B1's 0.17, and a clip that differs from its
cross-robot partner in two channels at once screens neither. `--cycles` scales it cleanly and the
body height does not move, which matters because Froude divides by it:

| `--cycles` | Froude | heading | travel | hip |
|---|---|---|---|---|
| 6 | 0.100 +/- 0.000 | -6 +/- 2 | 0.38 | 0.132 |
| 9 | 0.164 +/- 0.000 | -14 +/- 0 | 0.62 | 0.134 |
| **9.3** | **0.161 +/- 0.004** | **-5 +/- 0** | 0.61 | 0.134 |
| 10 | 0.197 +/- 0.006 | +11 +/- 5 | 0.74 | 0.134 |
| 12 | 0.225 +/- 0.003 | -0 +/- 0 | 0.79 | 0.129 |

**9.3 lands on 0.161, the recorded animal's own value, and within 5% of the B1's 0.17**, but it
drifts more on video and `--scale` gets the same speed with every metric improving instead. The
heading column is not noise -- its spread inside a row is zero and only the row changes -- so it is set by
where in the stride the clip happens to end, not by instability.

**The sideways gait cannot follow.** It runs at Froude **0.119** and `--cycles` above 6 destroys it
(9 travels 0.05 m, 12 reverses), so the sideways condition sits at a different speed from the
forward one by construction. Anything comparing sideways across the two robots has to match speed
some other way, or accept the mismatch and say so.

**The speed ladder, matched level by level to the B1's own clips.** At `--scale 0.65`, `--cycles`
spans the quadruped's whole measured Froude range with the body height staying flat (0.176 to 0.171,
3%), which `--scale` could not do -- it moved the hip 0.176 to 0.219, and a hexapod whose height
tracks its speed against a B1 whose does not would plant that correlation straight in the body-pose
channel being screened.

| `--cycles` | hexapod Fr | B1 `--vx` | B1 Fr | gap | heading | frames |
|---|---|---|---|---|---|---|
| 5.8 | 0.121 +/- 0.002 | 0.30 | 0.128 | -5% | +4 +/- 1 | 66 |
| 6.4 | 0.136 +/- 0.001 | 0.34 | 0.143 | -5% | -7 +/- 0 | 66 |
| 7.1 | 0.155 +/- 0.006 | 0.38 | 0.161 | -4% | -4 +/- 1 | 66 |
| 8.15 | 0.172 +/- 0.001 | 0.40 | 0.170 | **+1%** | -4 +/- 3 | 66 |
| 8.5 | 0.187 +/- 0.002 | 0.46 | 0.194 | -4% | -1 +/- 2 | 66 |
| 8.8 | 0.206 +/- 0.005 | 0.50 | 0.209 | **-1%** | +3 +/- 8 | 59 |

The hexapod runs about **5% slow throughout** -- a systematic offset rather than scatter, so it is a
consistent bias and not a per-level mismatch.

**The response has a plateau, and reading it as a ceiling would have cost two levels.** Between
`--cycles` 7.1 and 7.7 the Froude barely moves: 7.4 and 7.7 both return **0.159**, which is 2.6%
above 7.1's 0.155 while 7.1's own scatter is +/- 4%. Two clips there would be one speed wearing two
B1 labels, which is worse than a gap because the gap is honest. **The climb resumes immediately
after**, and 8.15 gives the tightest pair in the table. Levels have to be placed by measuring the
response, not by interpolating between its ends.

**There is a second plateau at 8.3-8.65 and a narrow bad spot at 8.9, and both were found only by
sweeping finely.** 8.3, 8.5 and 8.65 all return ~0.188, so the level there is a choice of *which*
point on a flat region -- 8.5 is the one to take, heading **-1 +/- 2** against 8.3's +6 +/- 1 and
keeping 66 frames where 8.3 drops to 61. And 8.9 reads heading **-14 +/- 13** while 8.8, 0.1 away,
reads **+3 +/- 8** at a *better* speed match. Reading 8.9 as the ceiling would have truncated the
range at `vx 0.46` for no reason; it is a single unstable point with clean ground on both sides.

The top level's heading spread, +/- 8, is still the loosest in the table against +/- 1 to 3
elsewhere. It is the cost of covering `vx 0.50` and is declared rather than hidden.

**The behaviour set is therefore: straight, `--spin` at four levels (turn), the two sideways gaits,
and `--cycles` at six levels (speed) -- all at `--scale 0.65`.**

**All three are one equation, and the `cpg_commands` docstring now derives it.** Three oscillators
share a clock; each joint takes a mirrored drive plus an un-mirrored one; and because the two sides
of the body are mirror images, **whether a term is mirrored is what decides the behaviour** -- a
mirrored term moves both sides the same way through the world and the robot walks, an un-mirrored
one cancels the fore-aft parts and what is left depends on which joint received it. The useful
question about any of these gaits is therefore *which joint carries a left-right amplitude
imbalance*, in radians per leg:

| | TC sweep | CF lift | FT extend | imbalance |
|---|---|---|---|---|
| straight | 0.250 both | 0.200 both | 0.300 both | none |
| `--spin 0.8` | **0.450 / 0.050** | 0.200 both | 0.300 both | **9:1 at TC** |
| sideways | 0.037 both | 0.200 both | **0.060 / 0.540** | **1:9 at FT** |

**CF is identical in all three** -- it is the clock, not a drive. `spin` and `strafe` are balance
knobs rather than throttles, giving `|left| = A(1 + knob)` against `|right| = A(1 - knob)`, so at
`knob = 1` one side's joint stops moving altogether. That is the mechanism behind the narrow
optimum: `--strafe 1.0` collapses to 0.12 m from 0.8's 0.33 because at 1.0 one side is switched
off.

**Three things had to be right, and the metric that caught the last one was the video.**

*Oscillate around the animal's walking pose, not the model's spawn pose.* Using the scene's default
joint angles put the abdomen at **0.284 m against the recording's 0.129**, and at a different height
for every behaviour -- 0.248 turning, 0.103 backing. Froude divides by that height, so a posture
that moves between behaviours makes the one quantity both robots share incomparable. The mean of
the IK commands is the pose the recording walks in, and costs one IK pass to get.

*Mirror **every** joint between the sides, not just the sweep.* Whether a joint angle swings a leg
forwards or backwards depends on which way its axis points, and the two sides of a mirrored body
point opposite ways. Mirroring only the fore-aft joint left the lift joints fighting each other
across the midline and the robot walked a **90 degree arc** while its start-and-end displacement
still passed `walk_check`. Mirroring all three took the heading change from **-89° to -1°**.

*`walk_check` cannot see this, and neither could the metric added to replace it.* Start-and-end
distance passes a body that yaws its way across the floor; a path-length ratio passes a smooth
curve. **Watching the clip is what caught it** -- the body turns to face the camera by two thirds of
the way through, which no number being printed at the time reflected. Net heading change is now
printed per clip, and it is the number to read.

**And a separate finding that applies to every clip ever collected here.** IK was solved with **one**
solver call per frame, which is a fixed number of steps and not a solution. On the recorded path it
keeps up; on any re-timed path it does not, and reports as if the target were unreachable when it is
simply not converged. `--ik_iters 8` takes the recorded gait's residual from **0.28 / 36.97 mm to
0.00 / 0.00** with no change to how it walks. Every dataset before 2026-08-21 carries the residual.

**The four routes that failed, because the reasons generalise.** The code for all of them was
removed from `collect_ik.py` on 2026-08-22 -- `--gait tripod` with `synth_tripod`, `--no_rephase`,
`--trim`, and `--strafe_gain` -- so this table is the only remaining record. A measured failure left
in an argument list reads like an option.

| tried | measured | why |
|---|---|---|
| ellipse built from the recording's mean and range | residual **365 mm** | a leg's reachable set is a curved shell; a box measured around it includes corners that are not in it |
| the same, with the sweep taken from planted frames only | **343 mm** | still a box, and lowering the stance line to the ground made it worse |
| re-timing the recorded wave into a tripod | residual **0.00 mm**, travel **-0.43 m**, body at forty degrees | leg paths authored for a gait that lifts one leg at a time do not support a body when three lift together. **A gait has to be designed as a whole; re-phasing one does not give another** |
| levelling the planted feet so three could support the body | **345 mm** | the abdomen frame is pitched with the body, so flat ground appears as a slope in it and the recorded feet already lie on that slope |

**The tripod is a second gait, not a replacement.** The recording is a variable wave -- one leg's
phase barely predicts the others, 0.07-0.24 concentration against a B1's 0.99-1.00 (F56) -- and that
looseness is what F45 cites for why no cross-robot frame pairing exists. A hexapod that walks both
against a quadruped that only trots is more coverage, not less contrast. **Collect them as separate
conditions**, or a later alignment gain will not be attributable to behaviour rather than to the
insect's gait having been made more B1-like.

---

### F72. Yaw is commandable on both robots, and their dimensionless ranges overlap

F70 left the shared target at one channel because the other five are constants in our data. The
hexapod now turns (F71's `--spin`), the B1 always could (`--wz`), and the question is whether the
two can be made to do **the same thing** rather than merely both do something.

**Nondimensionalise the turn the way Froude nondimensionalises the speed**: `ŵ = ω sqrt(h/g)`, with
`h` the hip height. Two robots at the same `ŵ` are turning at the same rate relative to their own
scale.

**Re-derived 2026-08-22 at `--scale 0.65`** (F71: the foot path was widened, which moves the standing
pose and roughly triples the turn rate per unit of `--spin` -- 0.4 gave 0.0180 at the old scale and
gives 0.0557 at the new one). The B1 side is untouched, since its clips are already collected and
its four levels already measured; only the hexapod commands move:

| ŵ | hexapod `--spin` | ŵ measured | B1 `--wz` | ŵ measured | gap |
|---|---|---|---|---|---|
| ~0.007 | 0.05 | 0.0081 +/- 0.0006 | 0.00 | 0.0067 | +21% |
| ~0.021 | 0.15 | 0.0200 +/- 0.0007 | 0.08 | 0.0209 | -4% |
| ~0.041 | 0.29 | 0.0388 +/- 0.0009 | 0.19 | 0.0407 | -5% |
| ~0.077 | 0.56 | 0.0736 +/- 0.0004 | 0.40 | 0.0772 | -5% |

Three of four match within 5%; the first row's 21% is a small denominator -- the absolute gap is
0.0014, inside the gait's own wander, and both rows are "walking straight". **The hexapod holds
Froude 0.132 / 0.131 / 0.122 / 0.121 across the whole yaw range against the B1's 0.120-0.135**, so
speed and yaw are still separable channels rather than one confounded behaviour.

*Superseded, for the record -- the same table at the old `--scale 0.5`:*

| ŵ | hexapod `--spin` | ŵ measured | B1 `--wz` | ŵ measured |
|---|---|---|---|---|
| ~0.007 | 0 | 0.0075 | 0.00 | 0.0067 |
| ~0.019 | 0.4 | 0.0180 | 0.08 | 0.0209 |
| ~0.042 | 0.8 | 0.0428 | 0.19 | 0.0407 |
| ~0.082 | 1.2 | 0.0872 | 0.40 | 0.0772 |

**The ranges overlap across their whole span**, and the hexapod side is repeatable to a standard
deviation of **0.002 rad/s or better** over two runs at each level. **This is the first pair of
matched commands this project has for any channel other than forward speed.**

**Turning does not disturb the speed on either robot, within the range actually used.** At the
adopted scale the hexapod holds 0.121-0.132 across the four matched levels and the B1 0.120-0.135.
Pushed further it does bite -- `--spin 1.2` at `--scale 0.65` drops Froude to **0.054** -- so the
independence is a property of this range, not of the gait. At the old scale the hexapod held
0.095-0.112 and the B1 0.166-0.170, so yaw and forward vary independently and can be
screened as separate channels rather than as one confounded behaviour.

**The speed gap is closed by slowing the B1, not by speeding the hexapod.** The oscillator walks at
Froude ~0.10 against the B1's ~0.17, and a clip that differs from its partner in two channels at
once screens neither. Both directions were measured on 2026-08-22:

*The hexapod can be sped up, at a cost.* `--cycles 9.3` reaches Froude **0.161 +/- 0.004** -- the
recorded animal's own value -- with the body height unchanged. On video it walks, but it drifts
more, and the sideways gait cannot follow at all (it runs at 0.119 and `--cycles` above 6 destroys
it: 9 travels 0.05 m, 12 reverses).

*The B1 can be slowed for free.* `--vx 0.22` gives Froude **0.101**, against the hexapod's 0.100 at
its default `--cycles 6`. Tracking is linear across 0.18-0.30, the trot is intact -- **2.08 feet
down**, hip 0.561 m unchanged -- and the lateral drift ratio stays at **-0.32**, which happens to be
the hexapod's straight-walk figure exactly.

*And it costs nothing in the table above, because turn rate and speed are independent on the B1:*

| `--wz` | w_hat at `--vx 0.22` | w_hat at `--vx 0.40` |
|---|---|---|
| 0.08 | 0.020 | 0.021 |
| 0.19 | 0.040 | 0.040 |
| 0.40 | 0.078 | 0.077 |

**So the matched set is the hexapod at `--cycles 6` with `--spin` 0 / 0.4 / 0.8 / 1.2 against the B1
at `--vx 0.22` with `--wz` 0.00 / 0.08 / 0.19 / 0.40 -- the pairing above, unchanged, now at a
matched Froude of ~0.10 as well.** Slowing the quadruped keeps the hexapod on the settings its gait
was tuned and watched at, and leaves this whole table valid rather than re-deriving it.

**Measuring this at all took three corrections, and each was giving confident wrong numbers.**

| measured from | read straight walking as | why |
|---|---|---|
| `/head` orientation | -151 deg of turn | the head segment sways 129 deg every stride |
| `/abdomen`, Euler angles | -99 deg | beta sits at -84 deg, a hand's breadth from gimbal lock |
| **`/abdomen`, quaternion, unwrapped trend** | **+15 deg, 16 deg of sway** | matches the video |

`collect_ik.py` now records `body_quat` per frame. Nothing before 2026-08-21 has it, which is why
roll, pitch and yaw could not be screened at all (F70).


---

### F73. Equal feet do not give equal contacts: the body decides, not the legs

F71's oscillator leaves the middle legs' contact bars short and broken, and the kinematics say why.
The three leg pairs measure **0.771, 0.489 and 0.638** long, so one amplitude gives three strokes:
measured on `c10f10t10`, the front feet lift **0.111** and the middle feet only **0.045**, and the
front feet reach **0.038** deeper than the middle pair and **0.056** deeper than the hind. Whichever
feet reach lowest carry the robot, so the contact pattern follows leg length rather than the phase
the oscillator asks for.

**The obvious fix was built, and it made the gait worse.** `scripts/diagnostics/tune_legs.py` solves
two numbers per leg against the kinematics -- a gain on the lift and extend amplitudes so every foot
rises the same distance, an offset on the same joints so every stroke bottoms out at the same height
-- and it converges cleanly, closing the lift spread from **0.072 to 0.0000** and the depth spread
from **0.056 to 0.0001**. Three repeats each, same settings otherwise:

| | in-group | across | feet down | duty spread | forward |
|---|---|---|---|---|---|
| untuned | **0.641** +/- 0.008 | **0.691** +/- 0.007 | 3.21 | **0.227** | **0.37 m** |
| feet levelled and lifts equalised | 0.451 +/- 0.010 | 0.511 +/- 0.003 | 2.90 | 0.465 | 0.22 m |

Every column is worse, the repeats are tight enough that none of it is noise, and **the duty spread
-- the very quantity the correction targets -- doubles**. Only MR's own duty improves, 0.41 to 0.57,
which is what makes this worth writing down rather than deleting: the intervention did exactly what
it was designed to do to the leg it was aimed at, and the gait still got worse.

**Why: the correction is computed in the body frame, and the body does not hold still.** Foot
heights are equalised relative to the abdomen, which is the same mistake F71's fourth failed route
made -- flat ground appears as a slope in a frame that pitches with the body. Under the tuned
settings the body's attitude swings **3.5x more** over a stride than untuned and the hip sits
**0.114 m** against **0.132 m**. Equalising six feet against a rocking frame does not put them on
one plane; it feeds the rocking.

The general form, which is F71's third failure again from another direction: **a gait is a
closed loop through the body, and per-leg geometry is an open-loop correction to it.** Getting a
clean tripod out of this animal needs the pose solved against the *world*, or the timing adapted
from contact -- which is what the lab's actual CPG (Larsen et al. 2023) does and this ported
demonstration oscillator does not.

**Kept as it stands.** `--legtune` is wired into `cpg_commands` and off by default. The settings that
stand are `--gait cpg --ik_iters 8 --amps 0.25 0.20 0.30 --ft_phase 0.125 --symmetric`, now confirmed
over three repeats rather than a single clip.

---

### F74. The two robots' clips were recorded at different frame rates, and every cross-embodiment number was computed across that gap

The insect collector records at **20 Hz** and `render_b1_replay.py` rendered **one frame per MuJoCo
rollout step, which is 50 Hz**. Both sides then stored a clip as a sequence of frames with no rate
written down, so the mismatch never surfaced:

| | frames | duration | **per stored transition** |
|---|---|---|---|
| hexapod | 66 | 3.30 s | **50 ms** |
| B1 (`data/b1_framed`, and today's first pass) | 99-126 | ~2.0-2.5 s | **20 ms** |

The ITM is handed `(e_t, e_t+1)` and asked for the latent of "the transition". **On one robot that
transition is 2.5x longer than on the other.** Everything computed across the pair inherits it --
F43/F46's below-chance cross-embodiment sharing, F51's forward model that does worse than predicting
no motion, F58's per-channel AUCs, F45's pairing feasibility. None of those numbers are wrong about
the data they were given; the data was not comparable.

**It also interacts with a finding we already had.** F70 measured that the body channels only cross
robots once averaged over a stride rather than a frame -- forward speed goes from **-1.45 raw to
+0.54 stride-averaged**. A stride-length window is 19 frames on the hexapod and 48 on the B1, so
stride-averaging was partly *correcting for this* without anyone knowing, which is consistent with
why it helped so much more than the size of the effect seemed to warrant.

**Fixed by subsampling frames, not by changing the physics.** `--fps 20` on the replay keeps the
rollout at 50 Hz -- the policy is untouched and still runs at the rate it was trained at -- and
renders every 2.5th step. Two details that were wrong in the first attempt and are worth stating
because both are silent:

*Index the proprioception by the frames actually rendered.* The save block sliced `T[k][:n]`, which
equals "the steps that became frames" **only** when every step became one. Under subsampling it
pairs frame `i` with a different moment's joint angles, desynchronising vision from proprioception
without any error.

*Derive `dt` from the requested rate, not from the first gap.* A 2.5-step stride rounds to gaps of
2 and 3 alternately, so `idx[1] - idx[0]` reports **0.04 s** where the mean interval is 0.05 -- a
20% timing error inside the fix for a timing error.

`--max_frames` was added alongside it so every condition yields the same clip length whatever its
speed. The B1 set is now **66 frames at 3.30 s**, identical to the insect, with Froude back on
calibration (0.133 against the 0.128 expected, 0.217 against 0.209).

**`data/b1_framed` still carries the old rate.** It is not deleted, because the results that cite it
have to remain reproducible, but nothing new should be measured against it.


---

### F75. Matching a magnitude is not matching a channel: the two robots were turning opposite ways

F72 pairs the hexapod's `--spin` to the B1's `--wz` on the dimensionless turn rate **w_hat**, and
w_hat is defined as a magnitude. The pairing therefore says nothing about which way each robot
turns, and they turn opposite ways:

| | speed conditions | sideways | turn conditions | sign consistent |
|---|---|---|---|---|
| hexapod | +0.49 +/- 2.66 | -0.51 +/- 0.76 | **-14.89 +/- 10.41** | yes, negative |
| B1 | +2.16 +/- 0.29 | +1.28 +/- 0.61 | **+8.74 +/- 6.16** | yes, positive |

signed yaw rate, deg/s. **The calibration could not see this**, because every table built to match
the two sides reported `|w_hat|` and the two columns agreed to within 5%.

**What it costs.** Pooled over the twelve conditions, signed yaw separates the robots at **AUC
0.871** -- the yaw value alone says which robot it came from, which is the exact gate F69 added
after lateral speed failed it at 0.788. A shared 6-DOF body target would then be trained on a
channel where `+w_hat` means "turning left" on one robot and "turning right" on the other, so the
head could reduce its loss by learning which robot it is looking at instead of how fast it is
turning -- the same shortcut F43/F46 measured the trunk already taking.

**A second, smaller version of the same thing.** The B1's yaw is positive in *every* condition,
including the ones commanded straight: **+2.16 deg/s walking forwards, +1.28 sideways**, with a
standard deviation of 0.29 -- a constant bias, not scatter. The hexapod's straight conditions
scatter about zero instead (+0.49 +/- 2.66). Even with the turn sign fixed, a constant offset in
one robot's yaw is a free embodiment cue.

**Fixed** by re-collecting the hexapod's four turn conditions at negative `--spin`, since positive
`--spin` yaws negative. The B1 bias is left in and declared; removing it means either retraining the
policy or subtracting a per-robot mean, and subtracting a per-robot mean is exactly the kind of
per-embodiment correction this project exists to avoid.

**The rule this establishes.** A quantity matched across two robots has to be matched as a **signed
vector in a shared world frame**, and the direction convention checked against something physical --
which way the robot actually travelled -- rather than assumed from an axis name. `|w_hat|` agreeing
to 5% is compatible with the robots turning opposite ways.


---

### F76. Re-screening the channels on matched behaviour: the frozen encoder holds two, the trained latent holds none

F70 screened the body channels and only forward speed passed, and it named the cause: **the other
five were constants in our data**, because both robots only ever walked forwards. `data/beh12_*`
removes that -- twelve matched conditions per robot spanning speed, turn and sideways travel,
balanced 4/4/4, 48 clips a side. This is the re-test.

`screen_behaviour_channels.py` scores four channels (F70's three plus **yaw**, which the old screen
had no rotational channel for) at two timescales against F69's three gates, on the frozen encoder --
`stage2speed7body` predates both F71's wider foot path and F74's frame rate and has never seen
either robot's current data, so its `z` says nothing here.

**A held-out clip is not a held-out behaviour, and the difference reverses the result.** Within a
condition the four clips agree to **2-10% of the between-condition spread** on both robots -- they
are one behaviour recorded four times, not four samples. A clip-level 70/30 split therefore leaves
near-duplicates of a training behaviour in the test set, and a readout can score by recognising a
behaviour it has already seen. Splitting on **condition** holds whole behaviours out. Five seeds
each, smoothed rows, frozen encoder:

| channel | robot AUC | hex->b1 | b1->hex | | by clip: b1->hex |
|---|---|---|---|---|---|
| **forward** | 0.66 +/- 0.11 | **+0.36 +/- 0.10** | -1.08 +/- 1.34 | | -0.56 +/- 0.39 |
| lateral | 0.68 +/- 0.04 | -0.16 +/- 0.44 | +0.04 +/- 0.45 | | +0.20 +/- 0.19 |
| vertical | 0.63 +/- 0.03 | -1.72 +/- 0.57 | -1.94 +/- 0.49 | | -1.74 +/- 0.36 |
| yaw | 0.72 +/- 0.10 | -0.82 +/- 0.23 | **+0.10 +/- 0.19** | | **+0.31 +/- 0.06** |

**Yaw's clip-split result was the artefact.** It read +0.31 +/- 0.06, positive on all five splits and
the tightest number in the table, and it was reported as the one finding that survived. Held out by
behaviour it is **+0.10 +/- 0.19** -- zero. The tightness was the duplicates: five splits of the same
leak agree with each other.

**Forward is the opposite** -- it gets *stronger and tighter* under the harder split, +0.21 +/- 0.13
to **+0.36 +/- 0.10**, hexapod to B1. A readout for walking speed fitted on the insect generalises to
a quadruped performing behaviours it never saw. That is a real cross-embodiment transfer, and it is
the same single channel F70 already had.

**Scope: this measures the frozen encoder, which is the state *before* any training.** F66 puts
forward speed at 0.31 frozen against 0.85-0.92 trained, so a frozen number does not decide whether a
channel is usable -- judged by its frozen value the one channel that works would fail too. What
follows answers "which channels are shareable **without being taught to be**", and nothing more
(F77).

F70's stated cause was
that five of six channels were constants in our data. They are not constants any more -- yaw spans
w_hat 0.007-0.076 and lateral spans Froude 0.07-0.19, on both robots, matched to within 10%, with
speed held apart from yaw so the channels are separable. **Coverage was supplied, and untrained the new channels
still sit at zero**: lateral and yaw at chance, vertical strongly negative and not varying once
smoothed. What training would do to them is untested.

**Three ways to read that, and the first was missed for several hours.** Either **the channels have
simply not been taught** -- the frozen encoder is the before condition and F66 shows a 3x gap
between before and after on the only channel ever trained (F77, and this is the live one). Or
forward speed is genuinely the only body quantity a hexapod and a quadruped share, a result about
morphology rather than about our pipeline. Or twelve behaviours is too few to resolve effects of
this size, since holding out by condition leaves about four test behaviours against spreads of
+/- 0.2 to 1.3. **The way to tell
them apart is more distinct behaviours, not more clips**: adding repeats of the twelve we have adds
copies, as the 2-10% within-condition figure shows.


---

---

### F77. The screen measured the untrained baseline, and a length scale cannot move it

F76 reported that supplying matched behaviour left no new channel shareable. **That conclusion was
drawn from the frozen encoder**, because `stage2speed7body` predates both F71's wider foot path and
F74's frame rate and has never seen either robot's current data. The frozen encoder is the *before*
condition, and this project's own measurement of what training does to that condition is F66:

| forward speed | readout correlation |
|---|---|
| frozen encoder | **0.31** |
| trained, no body term | -0.01 |
| trained, shared body head | **0.85 / 0.90 / 0.92** |

**Judging forward speed by its frozen value of 0.31 would have condemned the one channel that
works.** Yaw's frozen value is 0.10. Nothing has yet asked what training would do to it, so "matched
behaviour did not make any new channel shareable" should have read "no new channel is shareable
*without being taught to be*", which is a different claim and a much weaker one.

**Two hypotheses were on the table for why yaw sits at zero, and one is now dead.**

*That the dimensionless group is wrong.* w_hat = omega sqrt(h/g) uses hip height, which is the right
scale for walking -- an inverted pendulum over its own height -- but turning acts through the moment
arm of the planted feet, not through height. The two disagree sharply: measured from each model, the
B1 is **3.19x taller** and the hexapod's stance is **1.39x wider** (mean stance radius 0.576 m
against 0.414 m), so the two candidate scales differ by a factor of **4.4** in the ratio between the
robots.

**It cannot be the explanation, and the reason is structural.** `transfer()` standardises both target
vectors before fitting, and a length scale is an affine rescale of omega, so it cancels exactly.
Measured, held out by condition:

| yaw, smoothed | hex->b1 | b1->hex | robot AUC |
|---|---|---|---|
| height scale | -0.630 | +0.270 | 0.637 |
| stance-radius scale | -0.640 | +0.269 | **0.571** |

Transfer identical to three decimals. **The scale does matter for the robot gate** -- stance radius
hides the embodiment better, 0.637 to 0.571 -- and for choosing which `--spin` levels to collect,
but not for whether the channel transfers.

**And the obvious test of the scaling is circular.** Comparing how well each scale makes the
*collected* conditions line up says only that the data was collected to match one of them: under
height the four levels agree to -33/-19/-3/+5%, under stance radius to +41/+70/+105/+119%, because
the `--spin` values were solved against the height version. A choice of dimensionless group is a
modelling decision; the only non-circular test is to collect under each and compare what transfers.

**So the live question is the one not yet asked: train on this data and measure again.** F66 is the
precedent -- the body term moved forward speed from 0.31 to 0.90, and the gap between the frozen and
trained columns is larger than the gap between any two channels in the frozen column.


---

### F78. The B1's heading controller was proportional-only, and both the bias and the fix for it were traps

With yaw a candidate for the shared target, the B1's yaw had to be checked rather than assumed. It
is **positive in every condition**, including the ones commanded straight: **+2.16 deg/s walking
forwards, +1.28 sideways, sd 0.29** -- a constant, not scatter, where the hexapod's straight walking
scatters about zero (+0.49 +/- 2.66). Pooled over conditions that constant separates the robots, so
a shared yaw target would let the head read "which robot is this" instead of "how fast is it
turning" -- the shortcut F43/F46 measured the trunk already taking.

**The cause is structural, not a tuning accident.** `rollout_b1_mujoco.py` commanded yaw as
`HEAD_K * yaw_err` with `HEAD_K = 0.5`: proportional only. A P controller cannot reject a constant
disturbance below `disturbance / gain`, and the policy has a slight inherent turn. Every clip ever
rolled out carries it.

**The obvious fix produced a number that matched and data that did not.** Adding an integral term at
`ki = 5.0` took the standing drift to +0.33 deg/s and the steady turn rate to **0.0732 against a
target of 0.0736** -- a 0.5% match, and wrong. Inside a clip the turn rate ran **0.19, 0.017, 0.024,
0.15, 0.035**: a lightly damped oscillation with a ~3 s period against a 162-step clip. Only the
average over a much longer window matched. **The clips would have been controller ringing labelled
as steady turns.**

**The rule this establishes.** Verify a matched quantity by its **time course inside one clip**, not
only by its mean. A steady-state average agreeing to 0.5% is compatible with the signal underneath
it oscillating through the whole clip.

**Tuned against three requirements at once**, since satisfying one alone is what produced the trap:

| kp / ki | standing drift | turn level 3 | ringing (sd/mean in a clip) |
|---|---|---|---|
| 0.5 / 0 (shipped) | 0.0064 | 0.0604 | 0.06 |
| 0.5 / 5.0 | 0.0001 | 0.0616 | **oscillates, ~3 s period** |
| **2.5 / 1.0** | **0.0000** | **0.0743** | **0.04** |
| 4.0 / 1.5 | -0.0005 | 0.0720 | 0.02 |

`--head_kp 2.5 --head_ki 1.0`: no standing drift, turn matched to +1% with the `--wz` levels
unchanged, and **less ringing than the hexapod's own gait** (0.04 against 0.10). Re-collected on it,
the target reads hexapod 0.0148 / 0.0353 / 0.0736 against B1 0.0146 / 0.0346 / 0.0714 -- within 3%
-- and the non-turning conditions sit at +/-0.002 on both robots instead of +0.007 on one.

**One asymmetry survives and is real.** At the hardest turn the hexapod's forward speed falls to
**0.028** while the B1 holds **0.096**: a sharp turn costs the insect its speed and costs the
quadruped nothing. That is morphology, not a defect, but it means the *forward* channel carries some
embodiment information at that condition, and the robot-AUC gate should be read with it in mind.


---

### F79. The body target was measured in the world frame, so "forward speed" was partly a rotation measurement

Reviewing the collected conditions, the hexapod's forward speed appeared to collapse when it turned
hard -- **0.132 to 0.026** across the four turn levels -- while the B1 held 0.131 to 0.102. That was
written up as morphology: a sharp turn costs the insect its speed and costs the quadruped nothing.
**It is not morphology. It is not a controller difference either. It is the frame.**

`body_motion` differenced **world x and y**. Its "forward" channel was therefore walking speed
multiplied by how much the robot still happened to point along world x. Straight walking hides this
completely, since both robots start along +x -- and every dataset before `data/beh12_*` contained
only straight walking, which is why it survived this long. Projected onto each robot's own heading:

| | world-x | **body forward** | | world-x | **body forward** |
|---|---|---|---|---|---|
| | hexapod | hexapod | | B1 | B1 |
| turn level 0 | 0.132 | **0.135** | | 0.131 | **0.131** |
| turn level 3 | 0.026 | **0.128** | | 0.102 | **0.132** |

**Neither robot slows down when turning.** Both walk at a constant speed through all four turn
levels. Supervised on the world-frame version, the shared body head would have been taught that
turning means slowing down, by different amounts on the two robots -- a difference between their
turn rates wearing the label "speed", and a free embodiment cue in the one channel that works.

**Fixing the frame also rescued the channel that had been written off.** In the body frame the
sideways conditions read **-0.114 against -0.114** and **+0.162 against +0.159**, signs and
magnitudes agreeing. In the world frame lateral could never match, because it meant "world y"
regardless of which way each robot faced -- which is a large part of why F58 measured it separating
the robots at AUC 0.788 and concluded it was "an embodiment label in disguise". That conclusion was
drawn about a quantity that was partly an orientation measurement.

**And the forward direction is not the same as the fore-aft axis.** Projecting onto the hexapod's
abdomen axis gives **-0.135** where the B1 gives **+0.131** for the same behaviour -- the abdomen's
z points aft, so it must be negated. Same behaviour, opposite sign, and a shared head could only fit
both by learning which robot it was looking at. `forward_axis()` now carries the correction, checked
against the direction straight walking actually travels.

**The rule this establishes.** A body-relative quantity has to be computed in the **body frame**,
and the frame's orientation verified against measured motion. World-frame differencing is correct
only while every robot walks in a straight line along the same axis, which is a property of the old
datasets rather than of the quantity.


---

### F80. Two B1 policies separate the robot from its controller, and most of the B1's "character" was the controller

`sim/assets/b1_policy/` holds two trained policies -- `base_gait3` at 2.0 Hz and `base_1.7hz_sym`
at 1.7 -- and only the first had ever been used. Running both is the only way this project has to
ask whether a B1 property belongs to the **robot** or to **one training run**, and the answer for
several of them is the latter.

| at `--vx 0.30` | `gait3` | `sym` | hexapod |
|---|---|---|---|
| lateral drift | **-0.022** | **+0.004** | -0.04 to +0.01 |
| standing yaw | 0.0049 | **0.0008** | 0.0029 |
| stride rate | 2.00 Hz | 1.67 Hz | -- |

**The B1's constant sideways drift is `gait3`, not the B1.** It had been carried since the first
collection and treated as a property of the quadruped.

**So is its lean into turns**, which had a plausible morphological story attached -- a narrow stance
leaning in against the hexapod's wide stance swinging out:

| turn level | `gait3` lateral | `sym` lateral | hexapod |
|---|---|---|---|
| w_hat 0.0006 | -0.025 | +0.005 | -0.003 |
| w_hat 0.0148 | -0.029 | +0.005 | +0.001 |
| w_hat 0.0353 | -0.033 | +0.005 | +0.017 |
| w_hat 0.0736 | **-0.040** | **+0.004** | **+0.044** |

`gait3` leans further in the harder it turns; `sym` is flat through all four. **Two policies on the
same body disagreeing settles it: the body is not doing this.** The hexapod's outward swing, which
neither B1 policy reproduces, survives as the one real difference.

`sym` also tracks the matched turn levels to **1.5%** (0.0147 / 0.0347 / 0.0726 against 0.0148 /
0.0353 / 0.0736) where `gait3` runs +3 to +23% high.

**Both are kept, two clips each per condition, and the reason is not fairness.** Every clip of a
condition had been the same limit cycle at a different phase -- within-condition spread was 2-10% of
between-condition spread, so four clips said one thing four times and the effective sample size was
**twelve behaviours, not forty-eight clips** (F76). Two policies at different stride rates give the
B1 genuinely different dynamics within a condition: forward spread rises to 0.112-0.122 while the
condition mean stays on target, because the commands are solved **per policy** against the same
hexapod-side target.

**The cost is that half the clips carry `gait3`'s lateral bias.** That is acceptable only because
lateral is excluded from the target -- and this makes the exclusion permanent rather than pending a
re-test, since the channel now mixes a real quantity with a known per-policy artefact.


---

### F81. The inverse model cannot run at control time, and the source paper's answer is a second encoder

Planning was scoped as `e_t -> selector -> z_t -> Motion Decoder -> a_t`, with the selector sampling
in latent space. Reading LAC-WM's method rather than its architecture section shows that is not how
it works, and the reason is structural:

> "Since future observations, required by the IDM, are unavailable at inference time, we train an
> **action projector** that maps explicit actions into the latent action space, enabling the world
> model to directly consume raw action inputs."

**Our ITM has the same problem.** `z_t = ITM(e_t, e_{t+1})` needs the next frame, which at control
time is the thing being decided. **The ITM is a training and analysis module and can never be in the
loop.** Everything measured so far -- every transfer number, every probe -- reads `z` off two
ground-truth frames, which is why `predict_actions.py` says it is reconstruction rather than control.

**The two loops differ in which space the search happens in, and that decides what can go wrong:**

| | sample in latent space | sample in action space (LAC-WM) |
|---|---|---|
| | `z -> MD -> a` | `a -> projector -> z -> FDM` |
| every candidate executable? | **no** -- a sampled `z` need not correspond to any real behaviour | **yes**, by construction |
| decode `z -> a` at run time? | required, and the MD must be right for the new body | **not needed** -- the action is what was sampled |
| Motion Decoder's role | the controller | **auxiliary loss during pretraining only** |

The second row is the one that matters. Sampling in latent space makes the whole loop depend on a
Motion Decoder generalising to a body it was not trained on -- **never tested here** -- while
sampling in action space removes the decode step entirely.

**It also corrects a claim of ours.** "A new body needs only video" is stronger than what LAC-WM
claims for itself: the abstract says *"adapt quickly to previously unseen robot embodiments through
finetuning"*, and the projector is fitted on the target robot's own action data. The defensible form
is that **video is what lets the world model span incomparable bodies; the projector still needs
actions from the target robot** -- which is cheap rather than free, and F52 already measured how
cheap: one B1 clip clears break-even, nine clear every horizon.

**What this changes about the build.** The projector is a small network trained on data already
collected, and it replaces both the latent-space selector and the requirement that the MD transfer.
The closed loop becomes: sample candidate actions, project them, roll the FDM, score, execute the
winner directly.

**Correction: the FDM is fine-tuned too, and the paper says so explicitly.** An earlier version of
this entry flagged as unknown whether adaptation touches the FDM's weights. Section 3.2 states the
procedure -- **three stages, LoRA rank 2**:

> "First, we fine-tune the IDM and FDM of LAC-WM end-to-end using LoRA with rank 2. Second, we
> freeze the FDM and train the action projector from scratch to map explicit actions into the latent
> space. Third, we jointly fine-tune the projector and FDM end-to-end using LoRA with rank 2. At
> inference time, we use only the action projector and FDM to perform action-conditioned imagined
> rollouts."

So **a new robot costs a projector *and* LoRA adapters on the FDM**, not a projector against a frozen
world model. `fit_projector.py` freezes both, which is the conservative version -- if it works
frozen, the cheaper option suffices; if it does not, LoRA is the documented fallback rather than an
invention.


---

### F83. The shared body head is what creates cross-embodiment transfer, and it creates it channel by channel

Three arms on `data/beh12_*`, identical except for the body term. Held out by condition, frozen
`best.pt`, smoothed rows -- the behaviour rather than the gait phase:

| | forward hex->b1 | forward b1->hex | yaw hex->b1 | yaw b1->hex |
|---|---|---|---|---|
| control, no body term | **-28.918** | **-43.075** | -45.254 | -29.550 |
| body head, forward only | **+0.761** | **+0.641** | -8.872 | -9.791 |
| body head, forward + yaw | +0.482 | +0.323 | **+0.606** | -0.031 |

**Without the term nothing transfers, and not merely at zero.** A readout fitted on one robot applied
to the other is dozens of times worse than predicting a constant. F43/F46 measured the trunk becoming
a switch; this is what that costs, on data where the two robots perform matched behaviours at matched
speeds. **Matched behaviour alone does not produce a shared code.**

**F66's result survives the frame-rate fix and improves.** The published figure is +0.54 / +0.68 /
+0.75 in one direction; here the same term gives **+0.761 and +0.641, positive in both**. F74's
mismatch -- 20 ms of B1 against 50 ms of insect per stored transition -- did not manufacture that
result.

**A channel transfers when it is supervised and not otherwise.** Yaw is -8.9 / -9.8 in the arm where
it is absent from the target and **+0.606** in the arm where it is present, on identical data and an
identical architecture. This is the question `data/beh12_*` was built to ask, and the answer is that
the collection was necessary but not sufficient: **behaviour has to vary *and* the channel has to be
taught.** F76 measured the first half alone and found nothing, correctly.

**The channels compete.** Adding yaw roughly halves forward: +0.761 / +0.641 to +0.482 / +0.323. A
6-DOF shared target is therefore not free, and "widen the target" is a trade rather than an upgrade.
Whether the cost is capacity, gradient share, or the near-zero yaw of eight of the twelve conditions
is untested.

**Training loss cannot see any of this.** The two body-head arms are indistinguishable on every
logged number -- val 1.7602 against 1.7596, motion 0.2183 against 0.2186 -- while their transfer rows
differ completely. And **`probe` saturates in all three arms** (0.936 control, 0.982, 0.994), so it
cannot discriminate either; `z` is near-perfectly separable by embodiment even with no body term.
Identity in `z` and shared readout direction are independent properties, and only the post-hoc screen
measures the second.

**A side effect worth its own line: the term cuts per-embodiment action error by 38%** -- val motion
0.3517 to 0.2183 at no cost to reconstruction (1.5580 to 1.5400). One dimensionless number shared
across two robots makes each robot's own 18-D and 12-D joint decoding substantially better, which is
LAC-WM's stated "mitigates shortcuts" mechanism measured rather than asserted.

**Five condition-level splits, and they change what can be claimed.**

| arm | channel | hex->b1 | b1->hex | seeds positive |
|---|---|---|---|---|
| forward-only | forward | **+0.610 +/- 0.140** | **+0.573 +/- 0.240** | 5/5, 5/5 |
| forward-only | yaw | -5.230 +/- 2.383 | -10.268 +/- 3.059 | 0/5, 0/5 |
| forward+yaw | forward | +0.196 +/- 0.278 | +0.400 +/- 0.107 | 3/5, 5/5 |
| forward+yaw | yaw | +0.367 +/- 0.274 | -0.415 +/- 0.556 | 4/5, 1/5 |

**The causal claim survives completely.** Supervising yaw moves it **-5.23 to +0.37** and **-10.27 to
-0.42**, with **no overlap between the two arms in either direction** -- the worst seed of the
supervised arm beats the best seed of the unsupervised one by a wide margin. Identical data,
identical architecture, one term different.

**Yaw does not reach usable transfer.** It goes from catastrophic to approximately zero: +0.367 with
a spread of 0.274, and negative in four of five seeds the other way. The single-seed reading of
+0.606 / -0.031 overstated it.

**The cost is larger than one seed showed.** Forward hex->b1 falls **68%**, 0.610 to 0.196, and from
five seeds positive to three. **The practical conclusion is to use the forward-only target**: adding
yaw sacrifices most of the channel that works to buy one that merely stops being harmful.

**"Yaw carries less signal" was proposed as the cause and is refuted.** Both channels are
standardised pooled across robots, each by its own statistics, and the loss weights them equally.
Decomposing the standardised targets into between-condition and within-clip variance:

| channel | between-condition sd | within-clip sd | signal share |
|---|---|---|---|
| forward | 0.828 | 0.339 | **0.86** |
| lateral | 0.746 | 0.286 | 0.87 |
| yaw | **0.972** | 0.387 | **0.86** |

Identical signal share, and yaw's between-condition spread is the **larger** of the two. Its gradient
is not noise and the competition is not explained by signal quality.

**What the distribution does explain is the instability, which is a different question.** Forward's
variance is spread across the conditions; yaw's is concentrated in about six of them, with fourteen
sitting within +/-0.4 of zero:

    forward  -1.34 -1.30 -1.21 -1.19 | +0.31 ... +1.40
    yaw      -1.97 -1.30 | -0.39 ... +0.32  (fourteen conditions) | +0.65 +0.90 +1.99

The screen holds out about four of twelve conditions, so **a split that removes two or three turn
levels removes most of yaw's test signal**. That is why the yaw row reads +/-0.274 and +/-0.556 where
forward reads +/-0.140 -- leverage in the evaluation, not instability in the model.

**A longer smoothing window was the one free option, and it is refuted.** F70 established that these
channels cross robots only at stride scale, but never that every channel needs the *same* scale, and
yaw's noise floor is 2.6x forward's -- so a slower window might have recovered it. Measured on the
forward+yaw arm, seed 0:

| window | yaw smoothed | forward smoothed | yaw "varies" |
|---|---|---|---|
| **1.0 s** (default) | **+0.606 / -0.031** | +0.826 / +0.328 | 0.35 |
| 1.5 s | +0.594 / -1.063 | +0.564 / +0.396 | 0.34 |
| 2.5 s | -0.116 / -0.465 | -0.116 / +0.448 | 0.33 |

**Monotonically worse.** The `varies` column says why: on a 3.30 s clip a 2.5 s window averages the
signal away along with the noise -- vertical collapses from 0.14 to 0.07 -- and `mode="same"`
convolution pulls both ends toward zero on top. The default is already at the useful limit.

**So the competition remains unexplained**, with capacity and optimisation the remaining candidates,
and it is separable from the variance: more turn levels would tighten the *measurement* without
necessarily changing the *trade*.


---

### F84. Transfer R^2 is not comparable across these two datasets, and three attempts each found a different confound

F83 reports a control at -28.9 where the published control (F66) reads **-7.083**. The obvious
question is whether the behaviour collection made things worse. **It cannot be answered by comparing
those numbers**, and the record of trying is worth keeping, because the attempt is the natural one.

| attempt | result | why it does not compare |
|---|---|---|
| as published | control -28.9 vs -7.083 | F83 holds out **whole behaviours**; F66 held out clips of behaviours it had trained on. F76 measured that gap directly -- yaw read +0.31 by clip and +0.10 by condition on identical data |
| same split protocol | control -16.7 vs -7.083 | the new data asks for forward speed while the robot is **turning or strafing** in eight of twelve conditions. The old data was forward walking only, so "predict forward speed" meant predicting the one thing that varied |
| same split, speed conditions only | body head **-0.98 / +0.32** vs +0.54 / +0.44 | the restriction **removes variance rather than isolating the comparison** |

The third is the instructive one. Restricting to the four speed conditions looks like the clean
match and is the worst of the three:

| subset | forward sd | clips |
|---|---|---|
| all twelve conditions | 0.066 | 48 |
| speed conditions only | **0.040** | **16** |
| old `ik_walk_speed7` | 0.045 | 91 |

**R^2 is variance explained.** The speed-only subset has *less* forward variance than the old data
and a third of the clips, so after a 70/30 split it tests on about five clips over a narrower range.
The same representation scores worse on it -- +0.70 on the full set against -0.98 here -- with
nothing about the model changed.

**The two datasets differ in composition, dynamic range, clip count, frame rate and split protocol
at once**, and each of those moves R^2 independently. There is no slice that holds them all fixed,
and each fix for one introduced another. **Stop looking for one.**

**What replaces the comparison.** Every claim in F83 is measured inside a single dataset with one
loss term as the only difference -- same script, same clips, same split, same seed set. That is a
controlled comparison and the cross-dataset one never was. The result does not need to beat the old
number: it rests on the shared head turning **-16.7 into +0.70** on data where both robots perform
matched behaviours across three modes, which the old dataset could not have tested at all because it
contained only forward walking.

**The general rule.** A transfer R^2 is a statement about a representation **and** the distribution
it was scored on. Quoting one across datasets is quoting half a measurement -- the same trap as
`motion` MSE being incomparable across datasets and `action_lag` settings, which this project already
records in `wm/README.md`.


---

### F85. Yaw fails in the noise floor, not in the behaviour, and the noise floor is an asymmetry we introduced

F83 leaves yaw at +0.367 +/- 0.274 one way and -0.415 +/- 0.556 the other, after supervision moved it
from -5.2. The obvious question is whether that ceiling is the data. It is, and the mechanism is
specific.

**Yaw is matched where it is signal and mismatched where it is noise.**

| | hexapod | B1 | tell the robot from yaw alone |
|---|---|---|---|
| turn conditions (4 of 12) | -- | -- | **AUC 0.506** |
| speed conditions | -0.0003 +/- **0.0160** | +0.0019 +/- **0.0050** | |
| sideways conditions | -0.0010 +/- 0.0058 | +0.0014 +/- 0.0029 | |
| all eight non-turning | | | AUC 0.588 |

In the four conditions where both robots actually turn, the calibration holds and the channel is
indistinguishable between them. In the eight where neither turns -- **two thirds of the dataset** --
the hexapod's yaw has **three times the spread**. This is after stride-window smoothing, so it is
between-stride wander rather than gait rocking.

**That explains the direction asymmetry exactly.** A readout fitted on the B1, whose yaw is nearly
constant when not turning, has little to learn, and fails on the hexapod's wander: -0.415, negative
in four of five seeds. Fitted on the hexapod, part of what it learns is real turn signal, which
carries: +0.367.

**"The asymmetry is ours" was the hypothesis, and it is refuted.** The B1 was given a PI heading
controller on 2026-08-22 (F78) while the hexapod's oscillator was open loop, so the obvious reading
was that we closed one robot's heading loop and not the other's. `--head_kp/--head_ki` were added to
the insect collector to test it, modulating the oscillator's own `--spin`. Measured:

| gains | yaw sd | lateral | forward | wander |
|---|---|---|---|---|
| open loop | **0.0130** | 0.04 | +0.55 | 1.76 |
| kp 0.5 ki 0.2 | 0.0141 | 0.02 | +0.57 | 1.72 |
| kp 1.0 ki 0.4 | 0.0149 | 0.02 | +0.56 | 1.70 |
| kp 2.5 ki 1.0 | **0.0177** | **0.00** | **+0.70** | **1.48** |
| B1 | 0.0050 | | | |

**Every gain makes travel better and the yaw channel worse.** The controller holds heading by
steering continuously, so net drift falls while the instantaneous yaw *rate* becomes more variable --
it optimises net heading, and the target is a rate. There is no setting that closes the gap: the
floor is the open-loop 0.0130, still **2.6x the B1**.

**So the gap is the gait, not a missing controller.** The hexapod's sprawling wide stance swings its
body more per stride than a compact trot, and the residual survives stride-window smoothing. **This
is a real difference between the two robots and cannot be collected away.**

**That collapses the two-arm design proposed below** -- there is no artefact to remove, so a "clean"
arm and a "dirty" arm would differ in nothing relevant. The conclusion transfers to the objection
that prompted it: **the pipeline has to tolerate a difference like this, because collecting better
does not remove it.** And it currently cannot -- F44's three invariance methods moved nothing, the
adversary shifts `probe` without changing transfer, and the body head leaves identity fully decodable.

**The controller is kept and defaulted off.** It genuinely improves the walk -- lateral 0.04 to 0.00,
forward +27%, wander 1.76 to 1.48 -- but it raises the noise in the channel it was built to clean and
shifts Froude 0.126 to 0.162, which would force re-calibrating the whole speed ladder. Net negative
for the experiment it exists to serve.

**A second one the same test exposes, which is not ours.** Forward separates the robots at **AUC
0.869** in the sideways conditions: the hexapod still creeps forward while strafing (0.013-0.029)
where the B1 does not (0.002-0.009). That is a real capability difference, and it means the *forward*
channel identifies the robot in a third of the conditions.

**The design this was going to justify, kept because the reasoning survives the refutation.** The
objection to simply cleaning the data is correct -- a method that needs datasets matched to 2% is not
a method, since real cross-robot data is whatever each robot's controller produced. Two kinds of
difference have to be separated before deciding anything:

    behaviour differences    the robots doing different things -- the method exists to handle these
    collection artefacts     one robot has heading control and the other does not -- ours, not theirs

Leaving an artefact in does not test robustness; it tests whether the method survives our own
inconsistency. The plan was to collect both arms and read the gap between them as the robustness
number. **The test above removed the premise: the yaw gap is the second kind of difference, not the
first**, so there is no artefact to remove and the two arms would be the same dataset twice.

**What was explicitly ruled out, and still is: matching the two robots' noise floors after the fact.**
Forcing the hexapod's yaw to resemble the B1's would be fitting the dataset to the method. Since
giving both robots the same heading controller does not close the gap either, **nothing at the
collection level closes it**, and the difference has to be either tolerated by the model or declared
as a limit on the channel.

**And the deeper gap this exposes.** The pipeline has **no working mechanism for suppressing a
nuisance difference**: F44 tried three invariance methods and none moved transfer, the adversary
moves `probe` without changing what transfers, and the body head creates shared meaning while leaving
identity fully decodable (`probe` 0.94-0.99 in every arm including the control). Removing artefacts
by collecting better is currently the only lever we have, which is worth stating plainly rather than
presenting clean data as a design choice.


---

### F86. Within-condition diversity has to be measured on the input, not on the target

The B1 was given a second policy so its clips within a condition would be genuinely different
gaits (F80). The hexapod never got the equivalent: `--episodes 6,926,521,625` selects four expert
episodes, but the expert **is one gait** -- F57 measured 1.9% speed variation across a thousand
episodes -- so the four standing poses differ by **0.0007 rad**, four hundredths of a degree.

**Two measurements disagree about whether that matters, and only one of them is asking the right
question.**

| | hexapod | B1 |
|---|---|---|
| within-condition sd of the **target** (forward) | 0.0016-0.0062 | 0.0006-0.0088 |
| mean pairwise correlation of the **joint commands** | **1.000** | **0.127** |

By the target the two robots look equivalent, and this was reported as "no asymmetry to fix". By the
input they are not remotely equivalent: **the hexapod's four clips are the identical command
sequence**, and the target varies only because the simulation's physical response does. The B1's
four are two genuinely different gaits.

**The target measure cannot see this and never could.** Two clips can decode to the same Froude from
completely different images, which is precisely the invariance a shared readout is supposed to learn.
Spread in the target is not evidence of diversity in the input -- and if anything, *matched* clips
with *different* gaits are the ideal, while spread in the target means the condition has been
smeared into two.

**That also re-reads F80.** The B1's larger target spread was written up as its two policies
providing diversity. Half of it is the two policies not being speed-matched -- `sym` travels less
than `gait3` at the same command -- which smears the condition. The diversity is real and visible in
the command correlation; the target spread is a separate and mildly unwanted thing.

**The fix, and why it is not the same fix as the yaw one.** These are two different problems that
were briefly conflated:

    input diversity   the hexapod walks one way, the B1 two      -> a second parameterisation
    yaw instability   only ~4 test conditions survive a split    -> more turn levels

More clips per condition does not add test conditions, so it cannot address the second. More
conditions does not make the hexapod's gait more varied, so it cannot address the first.

**How large the asymmetry actually is, measured where the model works.** Command correlation is the
wrong scale to judge it on, because it standardises per column and so reports shape rather than
values -- the four hexapod clips differ by a constant **0.0007 rad** of bias, which it erases. That
difference is not nothing: legged contact dynamics amplify it, and the same configuration run three
times gave heading changes of **+21, +21 and -5 degrees**. In V-JEPA2 embeddings:

| | within condition | between conditions | ratio |
|---|---|---|---|
| hexapod, one gait | 6.430 | 16.518 | **0.389** |
| B1, two policies | 16.888 | 27.151 | **0.622** |

**The B1's two policies buy real diversity** -- 2.6x further apart in absolute terms, 1.6x relative
to its own spread -- so this is not what any four clips of a legged robot look like. **But the
hexapod is not degenerate**: at 0.389 the encoder sees four meaningfully different clips, not one
repeated. The asymmetry is moderate and real, not the near-duplication that command correlation
implied.

**And it is not what limits yaw.** A readout fitted on the B1 sees more varied views of the same
behaviour and should generalise better, yet `b1->hex` is the direction that fails (-0.415). F85's
noise-floor mismatch explains that; diversity does not. **Two separate problems, and conflating them
cost three collection attempts.**

**Second parameterisation, chosen so the pair is matched on the target and different in the gait:**

    speed and turn (8 conditions)   A: --ft_phase 0.125    B: --ft_phase 0.0
                                    Froude 0.126 against 0.123, hip 0.176 against 0.171
    sideways (4 conditions)         A: lift 0.20           B: lift 0.24
                                    sideways travel 0.48 both, Froude 0.114 against 0.113

Two axes were rejected by measurement. **Stride rate** works and is the closest analogue to the B1's
2.0/1.67 Hz -- `cycles 5.8 / amps 0.25` and `cycles 7.2 / amps 0.21` both give Froude 0.117 exactly
-- but the faster variant needs `cycles` above 10 at the top of the speed ladder, past where drift
sets in. **Lead** is inert: Froude 0.124-0.127 across `--lead` 0.20 to 0.35. And `ft_phase` cannot be
the sideways axis, since 0.5 is the antiphase that makes that gait work at all -- 0.375 and 0.625
collapse it to 0.15 and 0.06 m.

**Both parameterisations were collected and both failed.** `ft_phase` 0.125 against 0.0 moved command
correlation only **1.000 to 0.935** against the B1's 0.127 -- an eighth-cycle phase shift is a
perturbation of one gait, not a second one -- and the pairing does not survive the speed ladder:
matched at `cycles 5.8` (-7%), it reads **-39% at c8.8**, smearing the condition far worse than the
duplication it was meant to fix. The sideways variant did not vary anything: lift 0.20 to 0.24 is a
pure amplitude scale, the same gait louder, and correlation stayed at exactly 1.000.

**The failure is structural, not a bad choice of axis.** The B1's diversity comes from **two
independently trained controllers**. The CPG is one generator whose every parameter is coupled to the
behaviour -- stride rate and amplitude set the speed directly, `lead` does nothing, `ft_phase` couples
unpredictably. **You cannot change how this gait walks without changing how fast it walks.** What
would work is a different *phase pattern* rather than a different parameter -- a metachronal wave
against the current tripod ordering, which is also what stick insects actually do at low speed --
and that is an implementation rather than a sweep. Left undone; the failed collection is in
`data/beh12_hex2/` and was not merged.


---

### F87. The forward model barely reads the latent, and `L_recon` is most of the reason

`L_recon` asks the FTM to predict `e_{t+1}` from `e_t` and `z_t`. For a periodic gait at constant
speed, `e_{t+1}` is largely guessable from `e_t` alone -- so nothing forces the model to read `z`.
Measured with `ftm_uses_z.py`, holding one input fixed and sweeping the other:

| control arm | sweep z | sweep e | frame dominates | z / one step |
|---|---|---|---|---|
| speed conditions | 1.376 | 38.567 | **28x** | **0.027** |
| turn conditions | 1.654 | 37.985 | 23x | 0.033 |
| sideways conditions | 3.163 | 35.392 | 11x | 0.068 |

**The latent accounts for 2.7% of the distance the embedding actually moves in one frame.** And
replacing it with a latent from a *different behaviour* costs almost nothing:

| control, hexapod, speed | correct | wrong behaviour | difference |
|---|---|---|---|
| one step | 1.6333 | 1.6374 | **0.25%** |
| rollout at 8 | 3.1603 | 3.1804 | **0.6%** |

**Without the body term, `L_recon` is not training the latent in any meaningful sense.**

**And forward walking is the worst case, exactly as predicted.** `sweep z` runs **speed 1.376 <
turn 1.654 < sideways 3.163** -- the harder the behaviour is to guess from the previous frame, the
more the model is forced to read the latent. This is the strongest argument yet for the behaviour
collection, and a better one than the transfer numbers: variety is not decoration, it is what puts
pressure on the objective.

**The body term partly fixes it, and this is invisible in the training log.**

| | control | forward body head | change |
|---|---|---|---|
| sweep z (speed) | 1.376 | 4.257 | **3.1x** |
| frame dominance | 28x | 9.3x | 3.0x better |
| correct vs wrong behaviour | 0.25% | 3.5% | **14x** |
| rollout, real vs shuffled | 0.6% | 5.3% | 8.8x |

`recon` moves 1.5580 to 1.5400 between these two runs -- a 1% change that looks like nothing.
Underneath it, the forward model's dependence on the latent **triples**. **`L_body` is not only
reshaping the decoder; it is what makes the world model read its own conditioning input.**

**It also explains why the cross-embodiment rollout looks so flat.** `cross_latent_rollout.py` on
the body-head arm reads own 1.7010, other 1.7565, random 1.7659 -- **the entire dynamic range is
3.8%**, because `z` barely moves the prediction to begin with. A gap-closed of 0.145 / 0.183 is not
evidence that the latent fails to cross robots; it is a measurement with almost no room to move.
(The control reads 0.222 and **-0.420**, wildly inconsistent, which is what a near-ignored input
looks like.)

**Which loss term is training the latent -- and the loss values say the opposite of the truth.**
At convergence the objective reads `recon` 1.5400, `motion` 0.0110, `body` 0.7837 on a held batch,
so by value it is 79% reconstruction. Taking each term's gradient with respect to the **same** `z`
(`loss_gradient_balance.py`):

| term | lambda | share of loss | \|dL/dz\| x lambda | **share of gradient** |
|---|---|---|---|---|
| recon | 1.00 | 79.3% | 0.0004 | **5.1%** |
| motion | 1.00 | 0.6% | 0.0011 | 12.3% |
| body | 0.50 | 20.2% | 0.0071 | **82.5%** |

**Reconstruction is most of the loss and almost none of the gradient into the latent**, and the two
measurements confirm each other from opposite directions: the FTM barely reads `z`, so barely any
gradient flows back through it. The smallest term by weight is doing nearly all of the work on `z`.

*The 82.5% is overstated and the direction is not.* That run standardised `body_motion` on its own
hexapod batch, giving a body loss of 0.7837 against the training log's 0.0362 -- `train.py` stores
`body_stats` precisely because they are pooled across both embodiments and cannot be recomputed from
one robot. Rescaling puts it near **body 50%, motion 37%, recon 13%**. Recon is small either way.

**This decides what to fix, and rules out the obvious move.** Lowering `lambda_recon` or normalising
it would do nothing, because reconstruction was never dominating the gradient. **The weighting is not
the problem; the prediction task being too easy is** -- which is F54's finding and step 2i, already
measured and never acted on: at 20 Hz `t -> t+1` is 50 ms, a nineteenth of a stride, and stride-scale
pairs roll better at every horizon. LAC-WM does the same thing by chunking actions into five-step
sequences, *"which we found in practice improves world model learning"*.

**And it explains the control arm's collapse.** With no body term, `z` receives gradient only from
`motion` and `recon` -- the two smallest contributors -- which is why its cross-embodiment transfer
reads -28.9 while its `recon` looks indistinguishable from the body-head arm's.

**What to change, and what not to.** Lowering `lambda_recon` is the obvious move and is ruled out:
reconstruction is 99% of the loss and 18% of the gradient, so weakening it removes gradient without
rebalancing anything. Two changes are supported by the measurement:

*Raise `lambda_body`.* It has the **largest raw `|dL/dz|` of the three terms** -- 0.0019 against
motion's 0.0011 and recon's 0.0004 -- and is then halved by `lambda_body 0.5`, landing at 38.8%.
At 1.0 it would lead outright. Per unit of loss it pulls on `z` **78x** harder than recon does.

*Widen the pair.* `cfg.frame_stride` now sets how far apart the ITM's two frames sit (implemented
2026-08-24). This attacks the cause rather than the balance: at stride 1 the next frame is guessable,
so no weighting can make a latent necessary that the task does not need. F54 already measured
stride-scale pairs rolling better in-domain, and **also measured them losing at every horizon across
robots**, so this must be scored on transfer as well as on z-usage.

**Consequence for the closed loop, and the reason this was run before building it.** A planner
samples candidate actions, projects them to latents, rolls the FTM and picks a winner. If changing
`z` moves the prediction by 3-8%, every candidate scores nearly the same and there is little to
choose between them. Not fatal -- the body-head arm is 3x better than the control, and sideways is
1.6x better than forward -- but a planner built on the control arm would have had no signal at all.


---

### F88. Widening the training pair buys the latent and sells everything else

F87 ended with *"widen the pair"* as the supported fix. Implemented as `cfg.frame_stride`, then a
second time as `cfg.action_chunk` after the first attempt broke the decoder. **Both are now
measured, and the conclusion is not the one either implementation was written to reach.**

| hexapod, speed conditions | stride 1 | stride 5 + chunk | stride 10 + chunk |
|---|---|---|---|
| `sweep z` -- how much the FTM reads the latent | 4.257 | 7.662 | **12.279** |
| `z` per step | 0.083 | 0.150 | **0.240** |
| cost of a latent from another behaviour | 3.5% | 6.0% | 5.6% |
| **joint decoder**, val motion at one step | **0.218** | 0.906 | 0.879 |
| **transfer**, forward smoothed, seed 0 | **+0.826 / +0.328** | -0.102 / +0.050 | -0.447 / +0.102 |

**F87's prescription works on exactly the thing it was prescribed for.** The forward model's use of
the latent nearly **triples**, and a latent from the wrong behaviour now costs more than it did at
stride 1. LAC-WM's chunking is not a detail of their pipeline; it does what they say it does.

**And it costs both of the things this project is about.** The joint decoder goes to the level of
predicting the mean, and cross-embodiment transfer goes to zero -- not catastrophically negative
like the no-body-term control, simply **absent**.

**The intermediate diagnosis was wrong, and its own fix refuted it.** The first arm widened only the
frames, and the collapse was attributed to the mismatch: `z` summarising k steps while `L_motion`
scored it against one. Chunking the action target fixes that mismatch exactly, and it moved the
decoder **0.911 to 0.906** -- nothing -- while doubling `sweep z` from 3.728 to 7.662. **The
mismatch was real and was not the cause.**

**What the numbers say the cause is.** Four independent readings point one way: a wider pair turns
`z` into a **clip identifier** rather than a movement code.

| observation | consistent with a clip code |
|---|---|
| `sweep z` rises 2.9x | a clip identifier predicts that clip's future very well |
| decoder trains to 0.06 and validates at 1.28 | it memorises the identifier, it does not generalise from it |
| transfer falls to zero | an identifier is robot-specific by construction |
| `probe` rises 0.94 to **0.997** | `z` becomes *more* separable by embodiment, not less |

At stride 1 the pair difference is dominated by gait phase (F19: 64%), which is generic. At stride 5
to 10 it is dominated by which clip this is.

**This is F54, confirmed with a mechanism.** F54 measured a long-baseline arm winning every
in-domain horizon and losing every cross-robot one, and could not say why. The why is that the
information a wider pair adds is not shared between robots.

**Consequence for the objective, and it is a real constraint rather than a bug.** `z` being weakly
read by the forward model (F87) and `z` transferring across robots (F83) are **in tension**. Every
route to making the latent more predictive of the next frame that we have measured makes it more
robot-specific. The body term is the one intervention that moved both in the same direction --
`sweep z` 1.376 to 4.257 **and** transfer -28.9 to +0.610 -- and it did so by adding a shared
target, not by making the prediction task harder.

**So the planner's problem stands and this does not solve it.** Rolling a forward model that moves
3-8% under changes to `z` gives a planner little to choose between candidates. Widening the pair
raises that number and destroys the transfer the plan is supposed to cross with. **The remaining
direction is a shared target the frame does not already supply**, not a harder prediction task.

**What is kept.** `cfg.frame_stride` and `cfg.action_chunk` stay in the code, defaulting to 1 and
"follow frame_stride". They are measured, documented and off. No number in the deck or in any other
finding uses them.


---

### F89. Which kind of variety a backbone was pretrained on does not change how it adapts to a new robot

Two hexapod-only backbones, matched on everything except the axis this project is built around:

| | bodies | behaviours | clips | transitions |
|---|---|---|---|---|
| `beh12_hexonly` | **1** | **12** | 48 | 2,779 |
| `m3d_body` | **4** | **1** (forward) | 48 | 2,860 |

Both frozen, both with the shared body head, both adapted to the same 48-clip B1 set by
`finetune_ftm.py` at identical budgets and splits.

| 9 clips | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| 1 body, 12 behaviours | 1.25x | 1.12x | 1.01x | 0.86x |
| 4 bodies, 1 behaviour | 1.25x | 1.11x | 0.99x | 0.81x |

**Indistinguishable, and closer to each other than either is to its own spread across splits** --
the h=10 ranges are 0.76-0.96 and 0.76-0.87. The whole curve matches, at every budget from one clip
to nine, and so does the displacement profile: at 9 clips the predicted-versus-actual motion runs
0.54 / 0.82 / 0.97 / 1.17 against 0.53 / 0.82 / 0.97 / 1.23.

**What transfers to a genuinely different robot is generic, not morphological and not behavioural.**
F13 measured that adding *episodes* of the same bodies changes nothing while adding *bodies* does;
that held within the hexapod family and on the motion decoder. It does not extend to a forward model
crossing to a quadruped: there, neither axis moves the number.

**Consequence, and it saves work.** Collecting more hexapod bodies, or more behaviours per body,
will not improve few-shot adaptation to a new robot. A third embodiment is still needed -- for the
scaling claim and for testing whether the shared body head helps a *new* robot (F82, F83) -- but
**not for this**, and it should not be justified on this basis.

**What this does not say.** It compares two pretraining sets, not pretraining against nothing. The
gap to random initialisation is large and grows with budget: at one step, 1.25x against 0.92x, and
scratch never clears break-even at any budget measured. Pretraining is worth a great deal; *which*
variety it contains is what makes no difference.

**A second reading of the same run: the better one-step model is the less stable roller.**

| 9 clips, predicted displacement / actual | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| pretrained | **0.54** | 0.82 | 0.97 | **1.17** |
| scratch | 0.88 | 0.85 | 0.86 | 0.91 |

The pretrained arm is conservative at one step -- moving half as far as it should, and being right
about the direction, which is why it scores 1.25x. Closed on its own output ten times that becomes
an overshoot at 1.17, and it falls to 0.86x. Scratch holds ~0.9 throughout and lands near the
hold-still baseline by default. **A hypothesis that scratch had simply learned to predict no motion
is refuted**: predicting no motion scores exactly 1.00x by construction and would show a
displacement near 0.

---

### F90. The forward model can rank candidate actions, and the sensitivity ratios said it could not

F87 measured the frame outweighing the latent **28x** in the control and **9.3x** with the body
head, and concluded: *"if changing `z` moves the prediction by 3-8%, every candidate scores nearly
the same and there is little to choose between them."* **That inference is wrong, and this is the
measurement that shows it.**

`plan_discriminates.py` asks the question the closed loop actually asks. Given K candidate action
sequences, one of them the true one, project each through the fitted action projector, roll the FDM
from the same starting frame, and score against the true future. **The inverse model is not used at
all** -- this is the deployment path LAC-WM describes, projector and FDM only.

`beh12_hexonly`, 19 held-out clips over 12 conditions, 200 trials, top-1 rate:

| distractors drawn from | phase | chance | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|---|
| another behaviour | free | 12.5% | 74.5% | 82.5% | 80.0% | 63.5% |
| another behaviour | **aligned** | 12.5% | 69.5% | 68.5% | 68.5% | 64.5% |
| **same behaviour, other level** | free | 25% | 73.9% | 76.9% | 75.4% | 62.7% |
| **same behaviour, other level** | **aligned** | 25% | **57.8%** | 55.1% | 57.8% | 53.1% |

**The hardest row is the one that matters and it holds.** Telling `speed_vx0.30` from 0.38, 0.40 and
0.50, with every distractor taken at the same frame index so the gait phase cannot break the tie:
**57.8% against a 25% chance level over 147 trials -- about nine standard errors** -- with top-2 at
76.2%. That is a planner with something to choose between.

**Why the sensitivity ratios mislead.** `sweep z` measures the **magnitude** by which the prediction
moves when the latent changes. Ranking needs only that the movement be **consistently in the right
direction**. A small displacement that is reliably ordered ranks perfectly; a large one that is
noisy does not. The two quantities are close to independent, and every alarm raised in F87 and F88
about the planner was raised on the wrong one.

**Two design conclusions, one of them a withdrawal.**

*The horizon does not matter, and the apparent sweet spot was an artefact.* Free-phase distractors
show a peak at h=3 (74.5 -> 82.5); phase-aligned distractors are flat (69.5, 68.5, 68.5, 64.5). The
peak was the model rejecting distractors for being at the wrong point of the gait cycle, which a
planner cannot rely on because its own candidates all start from the same frame. **Any planner tuned
to h=3 on the first reading would have been tuned to nothing.**

*The projector costs about 15 points and is worth fixing.* The `latent` arm -- the candidate's own
`z` from the ITM, the best any projector could emit -- runs 10 to 17 points above the projector arm
in every condition (hardest row: 72.8% against 57.8%). **This is the first number that prices
LAC-WM's stage 3**, jointly fine-tuning the projector and the FDM, which takes 35k of their 60k
adaptation iterations.

**What this does not establish.** One robot, the one it was trained on, in simulation, ranking
recorded action sequences rather than sampled ones, and scored against a true future rather than a
goal. It says the forward model carries usable ranking signal, which is the precondition the closed
loop needed and which nothing before this had tested. **It does not say the loop closes.**

---

### F91. A planner over recorded behaviours picks the right one, except for how hard to turn

`plan_open_loop.py` runs the control-time path -- action projector and forward model, no inverse
model -- over the frames of a held-out demonstration. At each step it scores all twelve recorded
behaviours against the demonstration's frame `h` ahead and picks one. `beh12_hexonly`, 24
demonstrations, 478 decisions, horizon 5, replanning every third frame.

| | rate | chance |
|---|---|---|
| exact condition | **58.2%** | 8.3% |
| right behaviour family | **90.0%** | 25.0% |

**Split by family it stops being one number.** Counting the behaviour each demonstration was held in
longest:

| family | exact | what the failures are |
|---|---|---|
| speed | **9 / 9** | -- |
| sideways | **6 / 6** | -- |
| turn | **2 / 9** | every miss is turn -> turn at the wrong rate |

**It knows it is turning and cannot say how hard.** `s0.05 -> s0.15`, `s0.29 -> s0.56`,
`s0.56 -> s0.29`. No turn demonstration was ever taken for a walk or a strafe.

**This is the yaw limitation arriving for the third time by a third route.** F83 measured yaw
transfer going from catastrophic to approximately zero under supervision and never to usable. F85
measured its noise floor as an asymmetry between the two robots' gaits. Neither predicted anything
about a planner. **A planner built on the same latent resolves speed and lateral travel and does not
resolve turn rate** -- three independent measurements, one conclusion, and this is the one that says
what it costs at run time.

**The cross-family confusions are one cell of the matrix.** 17 of the 20 are `speed_c5.8` taken for
a turn -- the slowest walk against turning, which share a low forward speed. Nothing confuses fast
walking with anything.

**"The turn levels are too close together" is the obvious explanation and it is refuted.** Measuring
how far apart the conditions actually sit in body-motion space, in units of their own spread across
clips:

| closest pairs in `data/beh12_hex_flat` | separation / own noise |
|---|---|
| `speed_c7.1` / `speed_c8.15` | **1.7x** |
| `speed_c8.15` / `speed_c8.8` | 1.8x |
| `turn_s0.05` / `turn_s0.15` | **2.7x** |
| `turn_s0.29` / `turn_s0.56` | **6.6x** |
| `turn_s0.15` / `turn_s0.29` | 6.8x |

**The turn levels are two to four times better separated than the speed levels, and the planner
resolves speed 9/9 and turn 2/9.** It succeeds on the closest axis and fails on the furthest one, so
the failure is not about fine distinctions -- it is specific to **yaw**.

That places this beside F83, where yaw goes from catastrophic to approximately zero under
supervision and never to usable, and F85, where its noise floor is an asymmetry between the two
gaits. **Three measurements, three methods, one channel.**

**What this is not.** Open loop over recorded frames: the observations are the demonstration's, not
the ones the planner's own actions would have produced, so error cannot accumulate. It is the
necessary condition -- a planner that cannot pick the right behaviour from the demonstration's own
frames cannot do it from frames it caused -- and it is cheap. **The closed loop is what remains.**

---

### F92. The loop closes, and the planner's accuracy does not survive being fed its own consequences

**The loop runs.** `sim/control/close_loop_ik.py` drives the hexapod in CoppeliaSim from vision
alone: camera -> V-JEPA2 -> twelve candidate behaviours projected and rolled through the forward
model -> execute the winner -> step. **No inverse model, no kinematics, no human command.** Fifteen
runs, three commitment settings, five repeats each, 20 steps against a `speed_c5.8` demonstration.

| | commit 1 | commit 5 | commit 10 |
|---|---|---|---|
| S.R. survival | **5/5** | **5/5** | **5/5** |
| S.R. behaviour class | **5/5** | **5/5** | **5/5** |
| S.R. speed (within 15%) | 1/5 | 0/5 | 0/5 |
| median speed error | **21.8%** | 40.2% | 35.4% |
| behaviour changes per 20 steps | 12-13 | 3 | 1 |

**It walks, it stays up, and it walks too slowly.** Every run holds the right behaviour *class* and
none reaches the commanded speed.

**A dithering explanation was proposed and refuted by its own fix.** Replanning every step produced
12-13 behaviour changes in 20 steps, and the natural reading was that switching gait mid-stride
costs distance. Committing to a choice removes the switching and makes the result **worse**, 21.8%
to 40.2%. What commitment actually does is **lock in the first choice**, and the first choice is
almost always wrong.

**Two failures, and separating them is what the numbers show.**

| turn picks, on a demonstration that never turns | steps 0-4 | steps 5-19 |
|---|---|---|
| commit 1 | 80% | 48% |
| commit 5 | 80% | 67% |
| commit 10 | **100%** | 39% |

*The cold start is nearly all wrong.* After warmup the robot is standing still; there is no motion
in the frame to read, and a slow turn is the behaviour that best explains a nearly static scene.
Committing for ten steps then spends half the clip turning, which is where the extra 14 points of
speed error come from.

*And it does not recover to open-loop accuracy afterwards.* **F91 measured this same planner
choosing the exact condition on 9 of 9 forward demonstrations when the frames came from the
demonstration.** Here, on frames its own actions produced, 39-67% of steady-state picks are a turn.
Same planner, same candidates, same demonstration -- **the only difference is who generated the
observations.**

**That is covariate shift, and open loop cannot see it by construction.** The planner's early
mistakes move the robot into states its selection was never validated on, and the errors compound.
It is the classic failure of behaviour cloning under closed-loop execution, and finding it is
precisely what closing the loop was for: every measurement before this one -- F90's ranking, F91's
selection -- scored the planner on trajectories it did not produce.

**What this does and does not establish.** One robot, the one it trained on, one demonstration,
20 steps. **It establishes that the loop closes and the robot survives**, which was the milestone,
and it establishes the next problem precisely enough to act on. It does not establish that the
controller works.

**Started in motion, it works.** `--warm_start N` executes the demonstration's own commands for N
steps, then hands over. Scored **only on the planned steps**, against the demonstration over the
**same step window**:

| | warm 0 | warm 5 | warm 10 |
|---|---|---|---|
| S.R. speed | 1/5 | 0/5 | **5/5** |
| median speed error | 19.9% | 20.3% | **3.5%** |
| worst | 23.0% | 26.9% | **7.0%** |
| S.R. behaviour / survival | 5/5 | 5/5 | **5/5** |

**Ten steps of warm start turns a 1-in-5 failure into 5-in-5 at a fifth of the error.**

**And not because the planner chooses better.** Its pick distribution barely moves --
`speed_c5.8` 40 / 43 / 48%, `turn_s0.05` 35 / 28 / 28% across the three settings. What changes is
**the state it inherits**: handed a robot already walking, roughly-right behaviours maintain the
speed; handed one accelerating from rest, its early wrong picks leave the robot slow and twenty
steps is not enough to recover.

**"Start it in motion" is a deployment assumption, not a defeat.** No gait controller is expected to
go from a dead stop to a commanded speed inside one second.

**Two defects in the scorer were found on the way, both ours and both in the flattering direction
had they gone unnoticed.** It counted warm-start steps -- the demonstration replayed -- as the
planner's own work, worth about 2 points. And it scored our 10-step window against the
demonstration's whole 66 frames, comparing a start-up transient against a settled walk; correcting
it moved warm 10 from 7.4% to **3.5%**, so that one was penalising the result rather than inflating
it. Both are fixed and both are why the numbers above are quoted rather than the first ones.

**Over a full clip and all three behaviours, it meets the criteria.** 66 steps with 10 warm, so
**56 planned steps**, three demonstrations, three repeats each:

| demonstration | channel scored | speed error, three repeats | passes |
|---|---|---|---|
| `speed_c5.8` | forward | 14.3% / 4.0% / 6.7% | **3/3** |
| `turn_s0.05` | forward | 21.6% / 6.8% / 5.4% | 2/3 |
| `side_R_lvl0` | lateral | 26.4% / 7.0% / 11.9% | 2/3 |

| | rate |
|---|---|
| S.R. speed | **78%** (7/9) |
| S.R. behaviour class | **100%** (9/9) |
| S.R. survival | **100%** (9/9) |
| median error | **7.0%** |

**A third scorer defect, found here and in the punishing direction.** The criterion divided by the
**forward** Froude whatever the behaviour. On a sideways clip that reference is 0.015 -- near zero by
construction -- so the three `side` runs read 23-34% while tracking their *lateral* speed to within
7%. Scored on the channel the demonstration is actually about, they read 26.4 / 7.0 / 11.9. The
channel is chosen from the demonstration and never from the run, or a controller that drifted into
another behaviour would be graded on the one it drifted to.

**The behaviour selection holds up over the longer clip.** On the sideways demonstration the planner
spends **0-2%** of steps on a turn and holds `side_R_lvl0` for 26-35 of 49; on the turn
demonstration, 61-84% on turns. It switches often -- 16 to 36 changes over 49 steps -- and the modal
choice is right every time.

**What is still not established.**

*The yaw channel was not tested.* `turn_s0.05` is gentle enough that forward motion still dominates
it, so the scorer graded it on forward. **F91 predicts turn *rate* is what a planner on this latent
cannot resolve, and that prediction remains untested in the loop** -- it needs `turn_s0.29` or
`turn_s0.56` as the demonstration.

*It must be started in motion.* Ten warm steps, and from a standstill it fails at 1/5.

*One robot, one demonstration per behaviour, three repeats.* And the first repeat is the worst in
all three families -- 14.3, 21.6, 26.4 against 4-12% for the later two -- which is unexplained.

*Nobody has watched it.* The numbers say upright and on-speed. They do not say whether the gait
looks like walking, and this project's own rule is to render before believing.


---

### F93. A planner over recorded action sequences cannot drive the B1, and the reason is the robot rather than the method

The closed loop on the hexapod (F92) chooses among **recorded action sequences**. That is the design
decision that lets it work on a robot whose kinematics are unknown: no gait generator to author, no
IK, no URDF -- on a new robot you already have the few clips slide 15 adapts the forward model on,
and those clips are the candidate set.

**It requires the sequence to be a plan.** Replayed open loop, `data/b1_traj` at its native 50 Hz:

| trajectory | steps | survived | |
|---|---|---|---|
| `fwd_vx0.2` | 300 | 289 | fell |
| `fwd_vx0.3` | 300 | 154 | fell |
| `fwd_vx0.4` | 300 | 72 | fell |
| `fwd_vx0.5` | 300 | 58 | fell |

**0 of 8, and survival falls monotonically with commanded speed.**

**The two robots differ in what a recorded action *is*.**

| | what writes the joint targets | replayable |
|---|---|---|
| hexapod | IK and a CPG, from a clock. **No state is read.** | **yes, exactly** |
| B1 | a PPO policy reading base orientation, joint state and a phase clock at 50 Hz | **no** |

The B1's action is a *response*; without the state it was responding to there is no reason for it to
hold. The faster the gait the sooner the divergence bites, which is what the survival column shows.

> **Superseded in part by F101, and the correction is about episode length.** This entry was read
> afterwards as "no physics loop is available on the B1". The survival column above says something
> narrower -- **289, 154, 72 and 58 steps at 50 Hz**, six seconds at the slowest command -- and the
> closed loop is three. Run for three seconds, the forward behaviour survives a full episode and
> the loop clears chance on two demonstrations of three. **The claim that holds is that a recorded
> B1 sequence stops being re-issuable after about three seconds**, not that it never was.

**So the closed loop is not portable to the B1 as built, and this is a property of the target robot,
not a gap in the method.** Everything upstream of execution transfers -- F90's ranking and F91's
selection are both measured on B1 latents, and the forward model adapts to the B1 in three clips
(F92, slide 15). What does not transfer is the assumption that a recorded sequence can be re-issued.

**What a B1 closed loop would need, and why neither option is free.**

*Command its own policy.* Let the world model choose `(vx, vy, wz)` and let the PPO controller
execute. That gets real physics and a stable robot -- **and it is a task-space action, which is
precisely the thing this project argues you should not need** (slide 16). Using it concedes the
question rather than answering it.

*Replace the candidates with a closed-loop primitive.* A per-robot low-level controller the planner
selects among, rather than a sequence it replays. That keeps the joint-space claim and adds a
component per robot, which is the cost the recorded-sequence design existed to avoid.

**Reported rather than worked around.** The measurement cost one script and no training, and it was
run before anything was built for the B1 -- which is the point of asking whether something is
possible before asking how well it works.

---

### F94. The twelve behaviours reproduce on a held-out body, and the weak sideways gait reverses on it

`data/beh12_c08f09t09_flat`: the same twelve conditions collected on `c08f09t09`, the body every
Stage 1 result holds out. The recipe is now in `scripts/dataset/collect_beh12.py` rather than in
twelve commands nobody wrote down.

| | c10f10t10, forward | c08f09t09, forward |
|---|---|---|
| `speed_c5.8` -> `c8.8` | 0.135 0.166 0.210 0.200 | **0.129 0.158 0.200 0.215** |
| `turn_s0.05` -> `s0.56`, yaw | | **-0.007 -0.024 -0.037 -0.088** |

Both ladders rise monotonically on the new body, and **64 of 66 condition pairs sit more than twice
their own spread apart**. The two that do not -- `speed_c8.15`/`c8.8` at 1.2x and
`turn_s0.05`/`s0.15` at 1.4x -- are the same two the original set is closest on.

**The weak sideways gait travels the wrong way on this body.**

| | c10f10t10 | c08f09t09 |
|---|---|---|
| `side_L_lvl0` | +0.019 | **-0.045** |
| `side_L_lvl1` | +0.070 | +0.148 |
| `side_R_lvl0` | -0.022 | **+0.017** |
| `side_R_lvl1` | -0.124 | -0.131 |

At `--strafe ±0.8` both bodies crab the way they are told. At `±0.4` the shorter-legged one crabs
the other way. F71 recorded the direction reversing twice across an amplitude sweep and called the
optimum narrow; this is that, appearing as a **morphology** dependence rather than a parameter one.
**The gait's direction is not a property of the command alone.**

*It does not invalidate the planning test and it does mislabel two conditions.* Demonstration and
candidates both come from this body, so a planner that picks `side_L_lvl0` gets `side_L_lvl0`'s
motion and the loop is self-consistent. What is wrong is the **name**: on this body it strafes
right. Anything reading the label rather than the measured channel would be wrong.

**Two collection notes worth keeping.** The collector's walk check is a *forward*-walking gate, so
its `FAILS forward` and `BACKWARDS` flags fire on every sideways and hard-turning clip and mean
nothing there -- except that here `BACKWARDS` on the two weak strafes turned out to be real, which
is why flags get checked against measurements rather than dismissed. And the achieved values are not
matched to `c10f10t10` and are not meant to be: a single-body planning test never compares across
bodies, so **separability is the standard and value-matching is not** (`--separability` against
`--verify` in the collection script).

---

### F95. A two-layer MLP restores control of a body the world model has never seen, halfway

The closed loop of F92 run on `c08f09t09` -- the body every Stage 1 result holds out -- with the
same backbone, same candidates, same demonstrations, same scene, same warm start. Three behaviours,
two repeats, 49 planned steps each.

| | S.R. speed | S.R. behaviour | S.R. survival | median error |
|---|---|---|---|---|
| the body it trained on | 7/9 | 9/9 | 9/9 | **7.0%** |
| **held-out body**, projector from the trained body | 1/6 | 5/6 | 6/6 | 37.1% |
| **held-out body**, projector refitted on it | 2/6 | **6/6** | 6/6 | **19.2%** |

**Nothing in the world model is adapted.** V-JEPA2, the ITM and the forward model are frozen
throughout; the only thing that changes between the last two rows is a **two-layer MLP mapping
actions to latents**, fitted on the new body's own clips in a few minutes with no gradient through
anything else.

**It halves the error and restores the behaviour class outright.** Selection becomes decisive as
well as correct: on the sideways demonstration the refitted arm holds `side_R_lvl1` for 30-34 of 49
steps with 18-20 changes, where the un-refitted one settled on the *weak* strafe -- which on this
body travels the wrong way (F94) -- and switched about 30 times.

**And it does not restore the speed.** 2/6 against 7/9, median 19.2% against 7.0%. **The projector
accounts for about half the degradation and the forward model's ignorance of this body accounts for
the rest** -- which is the split LAC-WM's three-stage adaptation assumes, with stage 1 adapting the
FDM before the projector is fitted at all.

**Why this is the encouraging half.** The expensive component transfers: a forward model trained on
one hexapod predicts a differently-proportioned one well enough to keep it upright 6/6 and to pick
the right behaviour 6/6. The cheap component is what needs the new robot, and it needs only clips of
it -- **no kinematics, no URDF, no gait authored for it**, which is the deployment claim this
project is built on.

**The failure mode is walking the wrong way, not falling over.** Median body speed in any
direction, across all planned steps, is **0.097** zero-shot and **0.114** refitted against the
demonstrations' 0.13-0.14. A stopped or fallen robot reads near zero. It keeps walking at 70-85% of
the demonstrated speed throughout, and survival is 6/6 in both arms.

**"It tracks at first and diverges" is the warm start, and it is worth saying because it is what a
viewer sees.** Both arms replay the demonstration's own commands for ten steps, so up to the
handover the two videos are the same trajectory. Measured on the planned steps alone, the
zero-shot arm's pick accuracy is **flat** -- 42% over the first third, 39% over the last -- so it
does not degrade with time; it is simply not tracking. The arm whose picks *do* decay is the
refitted one, 48% to 21%, and its score improves anyway: on the turn demonstration it settles into
`turn_s0.29`/`s0.56`, the wrong **level** of the right behaviour, which walks forward at the same
speed. **What decays is level accuracy, not behaviour quality** -- F91's yaw limitation once more,
in a form the three success criteria cannot see.

**What is not established.** One held-out body, one family, three demonstrations, two repeats.
Speed accuracy is not recovered and the obvious next step -- adapting the forward model on a few
clips of the new body, then refitting the projector against the adapted ITM -- is the source
method's stages 1 and 3 and is **not built**: `finetune_ftm.py` adapts and scores without saving a
checkpoint to plan with.

---

### F96. Across embodiments the planner defaults instead of selecting, and the success criteria pass anyway

The B1 cannot be driven by re-issuing recorded actions (F93), so the **selection** half was measured
without the execution half: `sim/control/close_loop_kinematic.py` poses the body from per-step
body-frame deltas of whichever behaviour the planner chose. **There is no physics; the robot cannot
fall and its survival column is not evidence.** What the loop does test is the one thing open-loop
scoring cannot -- the frames the planner sees next are produced by what it chose now.

`beh12_hexonly` as the backbone: **the ITM and forward model have never seen a quadruped.** The
projector's B1 head *has* been fitted on `beh12_b1_flat`, so this is projector-adapted and
world-model-naive.

| demonstration | speed error | class | verdict |
|---|---|---|---|
| `speed_vx0.30` | 8.6% | forward | S B . |
| `turn_wz0.40` | 3.5% | forward | S B . |
| `side_R_lvl1` | **83.1%** | forward | - - . |

**Two of three pass and the passes are meaningless.** What the planner actually chose:

| demonstration | most-chosen | second | family accuracy |
|---|---|---|---|
| `speed_vx0.30` | `speed_vx0.50` 17/49 | `turn_wz0.00` 14/49 | 53% |
| `turn_wz0.40` | `speed_vx0.50` 21/49 | `side_L_lvl1` 7/49 | **8%** |
| `side_R_lvl1` | `speed_vx0.50` 18/49 | `turn_wz0.00` 15/49 | **10%** |

**`speed_vx0.50` is the top choice for every demonstration, including the sideways one.** The planner
is not selecting; it is defaulting to the fastest forward walk. The first two demonstrations *are*
forward walks, so a mixture of `speed_vx0.50` at 0.206 and `turn_wz0.00` at 0.126 averages onto
their 0.126-0.130 by arithmetic rather than by choice. **The only demonstration whose right answer
is not "walk forward" fails, at 83%.**

**The criterion is insensitive whenever most candidates move the same way**, which on the B1 set is
eight of twelve. Reported without the choice distribution beside it, this run reads as 67% success.
It is 8-10% family accuracy on the two demonstrations that discriminate.

**And it locates the binding constraint, which is the useful part.** On a held-out body of the *same
family*, refitting the projector recovered half the degradation and restored the behaviour class
outright (F95). Here the projector was already fitted on the target robot and selection still
collapses -- so **across families it is the forward model's ignorance of the robot that binds, not
the action-to-latent map.** That is what slide 15 measures from the other side: a frozen forward
model scores 0.57-0.71x on the B1, worse than predicting no motion, and needs three clips of it to
clear break-even.

**The experiment that follows is therefore specific**: adapt the forward model on N B1 clips, refit
the projector against the adapted ITM, and re-run this. It is LAC-WM's stages 1 and 3, and
`finetune_ftm.py` adapts and scores without saving a checkpoint to plan with, so it is **not built**.

---

### F97. The action projector is 2.8x harder to fit on the B1, for the same reason its actions cannot be replayed

Building LAC-WM's stage 1 (`wm/adapt.py`, which adapts the ITM and forward model and **saves a
checkpoint**, where `finetune_ftm.py` adapted and discarded) made stage 2 measurable on the B1 for
the first time. Both stages ran; they do not behave alike.

**Stage 1 works on nine clips.** Rolling on held-out B1 clips, against holding the frame still:

| horizon | frozen | after 9 clips | predicted / actual displacement |
|---|---|---|---|
| 1 | 0.68 | **1.16** | 0.55 |
| 3 | 0.66 | 0.98 | 0.90 |
| 5 | 0.69 | 0.86 | 1.10 |
| 10 | 0.73 | 0.74 | 1.31 |

From worse-than-nothing to better-than-nothing at one step, which is slide 15's curve reproduced
with the weights kept.

**Stage 2 does not.** The action projector fitted against the adapted ITM, scored as a rollout gap
against a mean-`z` baseline where **below 1.0 is better than knowing nothing**:

| projector fitted on | rollout gap |
|---|---|
| the same 9 clips | **1.301** -- worse than the baseline |
| all 48 clips | 0.841 |

**And more data is not the explanation.** The same script, the same clip count and the **unadapted**
checkpoint, fitting both robots at once:

| | rollout gap |
|---|---|
| hexapod, 48 clips | **0.230** |
| B1, 48 clips | **0.640** |

**2.8x harder on the same latent space, before any adaptation is involved.** The difficulty is a
property of the robot.

**The mechanism is narrower than it first looked, and F98 corrects the reading given here.** The
tempting explanation -- the B1's action is a policy's response to state, so it carries nothing a
planner can use -- is **wrong, and was measured rather than assumed**: a classifier reads the
behaviour off B1 actions alone at 85% family accuracy against a 28% chance rate (F98). The
information is in the action. What fails is the *target* stage 2 regresses onto: `z_ITM` encodes a
particular frame-to-frame transition, which depends on gait phase and state as well as behaviour,
so `a -> z` is one-to-many even where `a -> behaviour` is not. **The projector is not starved of
signal; it is asked to resolve a distinction its input does not determine.**

**Adapting the forward model does not rescue it, and that is the confirmation rather than a
surprise.** Running the kinematic loop with the adapted checkpoint and the best projector available:

| | frozen | adapted |
|---|---|---|
| S.R. speed | 2/3 *(by accident, F96)* | 0/3 |
| S.R. behaviour | 2/3 | 1/3 |
| family accuracy, `speed` / `turn` / `side` | 53% / 8% / 10% | 18% / 10% / 39% |
| most-chosen, every demonstration | `speed_vx0.50` | `side_R_lvl1` |

**It swapped one default for another.** The top choice is still the same candidate for all three
demonstrations; adapting the forward model changed *which* behaviour it defaults to, not whether it
selects. Sideways rises 10% to 39% and forward collapses 53% to 18%.

Which is what a projector at 0.841 predicts: **the latent it emits is close to uninformative**, so
the forward model is ranking candidates on something near noise, and improving the ranker cannot
help. The chain is complete -- **response-shaped actions -> a one-to-many `a -> z` -> an
unfittable projector -> nothing for the planner to rank** -- and the forward model is not the link
that fails.

**Consequence for the deployment claim, which has to be narrowed.** "Record a few clips of the new
robot and control it" holds where the robot's own controller is open loop. Where it is a learned
feedback policy, the action projector needs either far more data or LAC-WM's stage 3 -- joint
fine-tuning of projector and FDM, 35k of their 60k adaptation iterations, and **still not built**.
Stage 3 is not an optimisation in that setting; it is the step that lets the forward model move
toward a latent the projector can actually reach. **It is now built (`wm/adapt3.py`) and its first
result is in F98.**

---

### F98. The forward model refuses to use a quadruped's action, and the objective is why -- not the robot

F97 read stage 2's failure on the B1 as a property of the robot. **Building stage 3 and measuring
it says otherwise, and three results have to be read in order.**

**1. The action is not the problem.** A classifier trained on actions alone, held out by clip, has
to name which of twelve behaviours is being performed:

| | one frame | five frames |
|---|---|---|
| hexapod joint targets | 68% | **100%** |
| B1 policy actions | 61% | **80%** |
| *chance* | 8% | 8% |

By behaviour family the B1 reaches **85% against a 28% chance rate** (family chance is not 1/12 --
the families hold unequal numbers of conditions, and reading it as 8% reports chance as success).
The hexapod is cleaner, as a clock-driven action should be, but **the B1's action carries the
behaviour plainly.** The explanation F97 reached for -- a response-shaped action has nothing a
planner can use -- is measured and false.

**2. Stage 3 with an honest budget still fails, and fails in a specific way.** `wm/adapt3.py`,
15k steps, lr 1e-4, 24 clips, the faithful MSE form:

| step | train | /hold | **/mean-z** | family |
|---|---|---|---|---|
| stage 2 | 2.05 | 0.830 | 1.005 | 25% |
| 3000 | 0.80 | 0.799 | 0.994 | 19% |
| 15000 | **0.35** | **0.805** | **0.993** | 19% |

Training loss falls sixfold and held-out prediction genuinely improves -- **and `/mean-z` never
moves.** The forward model's answer given the projected action equals its answer given the mean
latent, to three decimals, at every checkpoint. **It learned B1 dynamics and discarded the action
channel entirely**, and family selection sits at chance throughout.

**3. Changing the objective fixes it, with everything else held constant.** Adding an InfoNCE term
-- the true action must reach `e_t+1` more closely than actions from other behaviours, negatives
drawn at the same time index so phase is not the giveaway:

| | /mean-z | cond | family |
|---|---|---|---|
| stage 2 | 1.005 | 8% | 25% |
| stage 3, MSE | 0.993 | 6% | 19% |
| stage 3, **+ InfoNCE** | **0.62** | **29%** | **50%** |
| *chance* | -- | 8% | 28% |

Same data, same robot, same architecture, same budget. **Only the loss changed.**

**Why MSE cannot get there from here.** The action-dependent part of `e_t+1` is a small fraction of
its variance, and at adaptation time there is a large unconditional gap -- "what does a quadruped
look like" -- available without touching the action at all. Gradient descent banks the large win
and is never obliged to earn the small one. During *pretraining* no such shortcut exists, which is
why the hexapod's forward model learned to use `z` and this one did not. **MSE rewards prediction;
planning needs discrimination.** They coincide when training from scratch and separate under
adaptation. This is Q7 -- the objective -- answered by measurement.

**The boundary is the family, and it is not memorisation.** The same hexapod-only checkpoint, the
same protocol, three sources:

| | /hold | family (chance 28%) |
|---|---|---|
| hexapod, the trained body | 0.692 | **84%** |
| hexapod, a body it has never seen | 0.776 | **62%** |
| B1 | **1.476** | 35% |

Arm 2 is the control F97 lacked: an unseen hexapod body still discriminates far above chance, so
arm 1 is not recall. **And the comparison understates itself** -- the projector used here was
fitted on `beh12_b1_flat`, so arm 3's projector had seen every clip it is tested on while arm 2's
had never seen its body. The disadvantaged arm wins by 27 points.

**What this changes.** Cross-family control is not blocked by the target robot's action space. It
is blocked by an adaptation objective that permits a shortcut, and **LAC-WM's three stages are MSE
throughout** -- the contrastive term is ours. It recovers about half the distance from B1's 35% to
the within-family 62%. Not solved; no longer a wall.

**A caveat that applies to every closed-loop number in this project, found while trying to add one
here.** The same loop command, same checkpoint, same demonstration, run twice **inside one
simulator session is identical to the pixel** -- and run against a *different* CoppeliaSim
instance, is not. Two sessions gave **6% agreement** on the chosen candidates, with rendered frames
differing (`frames.sum()` 1.4473e9 against 1.4649e9). The planner reads frames, so a rendering
difference too small to see changes what it picks.

Two consequences. **Loop results are comparable only within one simulator session** -- the hexapod
runs behind F95 all came from one, 03:20-03:33, and are internally consistent. And **the decision
margin is thin**: a planner whose answer turns on differences this small is not deciding with
confidence, which is the same story the 30-57% selection rates tell. A B1 loop number was nearly
written into the slides from a session that could not be reproduced, and the reproducible session
was markedly worse. **It was pulled rather than reported**, and re-measured properly: one
simulator, three demonstrations, run twice, **100% agreement between the repeats.**

| demonstration | family accuracy | most-chosen |
|---|---|---|
| turning | **59%** | `turn_wz0.19` |
| sideways | 24% | `turn_wz0.40` |
| forward | 0% | `turn_wz0.00` |
| *chance* | *28%* | |

**One of three clears chance, and all three top choices are turns.** The amplitude tracks the
demonstration; the behaviour does not. That is a step past the frozen and MSE-adapted models, which
answer with one identical candidate for every demonstration, and it is not behaviour selection.

**So 57% ranking on held-out clips did not convert, and the gap is the next problem.** The loop
adds two things clip-level scoring does not have -- error compounding across steps, and a state the
planner drove itself into rather than one that was recorded. The contrastive term fixed *what the
forward model attends to*; nothing here yet addresses *what happens when its small errors
accumulate under its own control*.

---

### F99. Evidence tables from the Stage 1 diagnostic slides, kept because the slides were cut

The week-13 deck was reduced from twenty-five slides to thirteen so that it states results rather
than the path to them. **Every conclusion on the cut slides is already written up above; some of
the raw table rows were not, and existed only inside the deck file.** They are reproduced here so
the record does not depend on a presentation document. Source: `report/update_slide_full.md`.

**Cut slide 5 — Four changes that did not help, and the one that did**

| frame from | latent from | matches `c10f10t10` | matches `c10f06t06` | follows |
|---|---|---|---|---|
| c10f10t10 | c10f06t06 | **4.79** | 21.64 | **the frame** |
| c10f06t06 | c10f10t10 | 21.59 | **5.84** | **the frame** |

**Cut slide 6 — What is inside the latent, with and without the cross-body loss**

| | Without | With |
|---|---|---|
| variance explained by **gait phase** | 81.9% | **92.6%** |
| variance explained by **which body it is** | 12.4% | **3.4%** |
| variance explained by neither, the interaction | 5.7% | 4.1% |
| **foot-contact pattern decodable from it** (8 patterns, majority class 0.172) | 0.729 | **0.732** |
| which body it is, decodable from it (4 bodies, chance 0.250) | 0.764 | **0.694** |

**Cut slide 8 — The limit: everything ties the femur to the tibia, because the data does**

| the same weights, asked about | deg | **R²** |
|---|---|---|
| `c08f09t09` — femur 0.9, tibia 0.9, **inside the range** | **3.44** | **+0.81** |
| `c10f10t08` — femur 1.0, **tibia 0.8**, the first time they differ | 13.35 | **−0.34** |

| | coxa | femur | tibia |
|---|---|---|---|
| **The truth** | 1.00 | **1.00** | **0.80** |
| The trained decoder, from its output commands | 1.000 | **0.681** | **0.681** |
| The linear probe on the frozen encoder | 0.920 | **0.843** | **0.843** |
| The best any mixture of training bodies could say | 0.809 | **0.600** | **0.600** |

*Where the failure sits*

| joint | what it moves | R² |
|---|---|---|
| **TC**, thorax-coxa | swings the leg fore and aft | **+0.46 to +0.83 — still works** |
| **CF**, coxa-femur | lifts the leg | −0.53 to +0.05 |
| **FT**, femur-tibia | the knee | **−0.45 to −3.99** |

*Where the failure sits*

| held out | femur/tibia | deg per joint | **R²** |
|---|---|---|---|
| c10f10t08 | 1.04 | 13.35 | **−0.34** |
| c10f09t07 | 1.07 | 11.63 | **−0.14** |
| c10f08t06 | 1.10 | 10.51 | **−0.33** |

**Cut slide 9 — Testing the diagnosis instead of asserting it**

With decoupled bodies added to training, the probe's femur-tibia gap opens to **0.182 against a
true 0.200** -- the coupling the four tied bodies could not express is recovered once the data
contains it.

| | coxa | femur | tibia |
|---|---|---|---|
| the truth | 1.00 | **1.00** | **0.80** |
| probe fitted on the 4 tied bodies | 0.955 | **0.819** | **0.819** |
| probe fitted on all 6 | 0.973 | **0.954** | **0.772** |

**Cut slide 10 — The same measurement predicts, before training, which bodies will transfer**

| Training set | Held out | **Mixture gap** | **Probe error** | Model, deg | R² | Outcome |
|---|---|---|---|---|---|---|
| 4 bodies, spanning | `c08f09t09` | **0.000** | **0.021** | **3.44** | +0.81 | **beats copy-nearest, 3.47** |
| 6 bodies, decoupled | `c10f10t08` | **0.063** | **0.034** | **3.27** | **+0.89** | **beats every baseline** |
| 4 bodies, all tied at 0.83 | `c10f10t08` | **0.141** | **0.082** | 12.67 | −0.78 | loses to the body's own mean |

**Cut slide 11 — One frame nearly determines the command**

*The transition is worth about a third*

| what the ITM is given as `e_{t+1}` | control | with the cross term |
|---|---|---|
| the real next frame | 3.71 deg | **3.37 deg** |
| **a copy of `e_t`, no transition at all** | **1.28x** | **1.34x** |
| `e_{t-1}`, a wrong transition | 1.67x | 1.65x |
| a frame from a random other time | 3.54x | 3.44x |
| the latent zeroed entirely | 2.88x | 3.48x |

**Cut slide 12 — The forward model was being judged on the wrong task**

| steps ahead | forward model | hold the frame still | constant velocity | **beats holding still by** |
|---|---|---|---|---|
| 1 | 1.39 | 2.11 | 5.78 | **1.52x** |
| 3 | 1.78 | 3.05 | 27.6 | **1.72x** |
| 5 | 2.12 | 3.57 | 66.0 | **1.69x** |
| 10 | 2.98 | 4.36 | 236.5 | **1.46x** |
---

### F100. The forward model is doing the selecting, and one step of it is enough

**The challenge this answers.** If the candidate library already contains recorded behaviours of
the target robot, the world model may be doing very little -- picking whichever action most
resembles what it was trained on, with the rollout as expensive decoration. The planner spends
**sixty forward-model calls per control step** (twelve candidates, five steps) where a policy
spends one, so this is worth settling rather than assuming.

Three scoring rules, identical candidates, identical held-out clips, argmin of each
(`scripts/diagnostics/does_rollout_matter.py`):

| rule | how the goal enters |
|---|---|
| `rollout` | roll the FDM `h` steps on `proj(a)`, score the predicted frame against the goal frame |
| `direct` | score `proj(a)` against `ITM(e_t, e_goal)` -- **no forward model at all** |
| `blind` | score `proj(a)` against the mean latent -- **the goal is not used** |

Behaviour-family accuracy on the B1, against a 28% chance rate:

| horizon | rollout | direct | blind |
|---|---|---|---|
| 1 | **62%** | 38% | 33% |
| 3 | **65%** | 36% | 32% |
| 5 | **65%** | 38% | 32% |
| 10 | **67%** | 37% | 34% |

**The rollout carries the selection.** Deleting the forward model costs **24 points** and lands
`direct` within five points of `blind`, which does not look at the goal at all. **The world model
is predicting, not pattern-matching**, and an earlier reading of the horizon sweep that suggested
otherwise was reading the wrong experiment: that sweep varied *how far* to roll, which is not the
same question as *whether* to roll.

**And one step is nearly all of it.** Going from one step to ten adds five points, where going
from none to one adds twenty-four. The planner can be run at **twelve forward-model calls per
control step instead of sixty** -- a fivefold cut for three points.

**The gap this exposes is the one that matters.** These rules score 62-67% on recorded clips while
the same checkpoint in the closed loop follows one demonstration of three. The difference is not
the scoring rule -- it is that offline the current frame always comes from a real clip, and in the
loop it comes from wherever the planner drove. **Compounding error under the planner's own control
is the binding constraint**, not the choice of objective at a single step, and that is an argument
for moving the search into training -- a policy distilled in imagination is trained on the states
it actually reaches.

---

### F101. The B1 closed loop runs in physics after all, and F93's "impossible" was an episode-length claim

**F93 measured that replayed B1 actions fall -- 0 of 8 -- and that conclusion was carried forward as
"a physics loop on the B1 is not available".** The table it reported says something narrower:
survival of 289, 154, 72 and 58 steps at 50 Hz, which is **six seconds** at the slowest command.
The closed loop is three. Nobody asked how long the robot stays up, only whether it eventually
fell.

**Replaying the beh12 clips at the rate the planner runs them** -- 50 ms per decision, not the
policy's 20 ms, since the clips are 20 Hz:

| clip | at 20 ms/step | **at 50 ms/step** |
|---|---|---|
| forward | 66 / 66 | **66 / 66** |
| turning | 66 / 66 | 28 |
| sideways | 66 / 66 | 27 |

The first column is the trap: stepping 20 ms per row simulates 40% of the episode and reports
survival the robot never earned. **An earlier version of this measurement did exactly that and
concluded all three survive.**

**So the loop was built** (`sim/control/close_loop_b1_physics.py`): **MuJoCo holds the physics,
CoppeliaSim poses a body from MuJoCo's state and returns the camera image.** The split is
mandatory rather than convenient -- the B1's policy walks only in MuJoCo, and rendering the B1
from MuJoCo while the insect comes from CoppeliaSim would let the encoder separate the robots by
render style instead of by morphology.

**The first version of this loop started the robot standing, and that was a defect.** The clips
were recorded with the spawn-to-walk transient cropped, so their first action is a command for a
robot **already mid-stride**. Applying it to a robot standing still asks a leg to continue a swing
it never began: body height leapt **0.435 -> 0.665** in six steps, a third above its own stance
height, visible in the rendered video as a jump the demonstration never makes. Seeding MuJoCo from
the demonstration's first frame -- joint angles, joint velocities, height, orientation -- removes
it. **The numbers below are the seeded runs; the standing-start runs are reported only as the
comparison.**

| | seeded from the demonstration | standing start |
|---|---|---|
| forward | **65 / 65**, family 58% | 65 / 65, family 75% |
| turning | **65 / 65**, family 51% | fell at 29, family 35% |
| sideways | **65 / 65**, family 38% | fell at 37, family 68% |
| peak body height | 0.57-0.60 | 0.67-0.70 |
| *chance* | *28%* | *28%* |

**The robot now stands through every episode**, and the artefact that inflated two of the
standing-start scores is gone. Scored on the project's criteria: **survival 3/3, behaviour class
2/3, speed 0/3** -- errors of 25%, 40% and 95%.

**So the quadruped holds itself up under the planner and does not track speed.** Family accuracy
averages 49% against 28% chance, above chance on all three and far from the 90% the hexapod reaches
on recorded frames.

**The planner is not what makes a robot fall.** In the standing-start runs sideways survived **37**
steps under the planner against **27** replaying its own clip. Re-deciding every step holds the
robot up slightly longer than re-issuing a recorded sequence.

**What this replaces.** `close_loop_kinematic.py` exists because the physics loop was believed
impossible; its survival column passes by construction and had to be reported as `n/a`. **The B1
now has a survival number that means something**, and the honest form of F93 is that a recorded B1
action sequence holds for about three seconds and not six.

---

### F82. Positioning against LAC-WM: the shared quantity is not the action space, and that is the whole difficulty

F67 established that the divergence from LAC-WM is **the coordinate the heads decode into**, not
their number or size. Reading their evaluation sections sharpens what that costs us and what it buys.

**Their embodiments differ in degrees of freedom but share a physical space.** The motion decoder's
targets are wrist poses, fingertip positions and camera poses -- a fingertip at `(x, y, z)` means the
same thing for a human hand and a robot gripper. The unified latent action space is therefore
unifying representations of quantities that are **already commensurable**.

**Where the value actually is, and it is not the coordinate on its own.** The setting this method is
for is a robot about which **nothing is known** -- no kinematic tree, no URDF, no action labels, only
video of it moving. Morphology-agnostic *proprioceptive* control exists but must be handed the
kinematic graph; a camera has to be handed nothing. What the world model supplies is knowledge of
**how to drive joints so the result is locomotion**, which is the expensive part of bringing up a new
robot. The coordinate argument below explains *why the problem is hard*; this is what makes it worth
solving.

**And it sets the research question, which the three arms answer.** Can a pipeline work with a
**joint-space action target** at all, given no shared action space exists? Measured:

| | within-robot joint error | cross-robot transfer |
|---|---|---|
| joint target, no body term | 0.3517 | **-28.9 / -43.1** |
| joint target + body term | **0.2183** | **+0.610 / +0.573** |

**A joint-space target works within a robot on its own** -- the control arm decodes 18-D and 12-D
commands without any shared supervision. **It does not cross robots without the body term**, and the
term also improves the within-robot decoding by 38%. So the defensible conclusion is not "joint
targets work" or "joint targets fail", but the conditional one: **a joint-space action target
transfers across incomparable embodiments only when a shared body-motion term is present** -- and
that is a result, not a caveat.

**A weak version of our claim, and why it fails.** It is tempting to say ours are "one or two
dimensionless numbers rather than a 6-DOF pose in a common frame". That is a description of a data
limitation dressed as a design principle: **body pose is shared by every embodiment** -- a rigid body
has a pose whatever moves it -- and nothing in principle stops us supervising all six. We supervise
two because the oscillator does not populate the rest: vertical fails the variation gate at **0.13**
by construction, since `--scale` is held fixed precisely so body height does not track speed, and
roll and pitch were never varied at all. Populating them needs a controller that can command body
attitude, which means the kinematic knowledge this project exists to avoid needing.

**A second weak version, also wrong.** An earlier draft said "there is no command for travel at
Froude 0.16". **There is** -- the B1's policy is commanded in m/s and `--vx 0.30` means exactly that.
Velocity-conditioned control is the standard interface for a legged policy. What is true is much
narrower: *our hexapod's* oscillator takes stride frequency, so `--cycles -> Froude` had to be
calibrated empirically (F71). That is a property of the controller we built, not of locomotion.

**The version that survives, and it inverts the argument.** Body velocity is not merely shared, it is
**commandable on both robots** -- which is precisely why it is not our action space. Two observations
follow.

*Commensurable is not equivalent.* 0.3 m/s on a 0.176 m insect and a 0.561 m quadruped are not the
same behaviour; one is near its limit and the other is strolling. Same units, different meaning. The
dimensionless form is what makes them the same behaviour, which is why every match in
`data/beh12_*` is on Froude and w_hat rather than on m/s and rad/s.

*Decoding the shared space would dissolve the problem.* If the Motion Decoder emitted body velocity,
both robots would share a 3-DOF action space, the correspondence would be **given rather than
learned**, and there would be nothing for a latent action to bridge. **We decode joint angles
because the disjoint space is the one worth crossing** -- 18-D against 12-D with no correspondence
between any pair of dimensions.

That is the real difference from LAC-WM, and it is a choice rather than a limitation. Their targets
-- end-effector and camera poses -- live in the space a manipulator is naturally commanded in, so
their latent maps onto something directly actuable and the hard part is unifying *representations* of
commensurable quantities. Ours maps onto a space where **no dimension of one robot corresponds to any
dimension of the other**, and the shared quantity is carried by a separate head whose only job is to
stop the trunk becoming a switch. F83 measures what that head is worth: without it, transfer is
-28.9.

**What we can and cannot reproduce from their evaluation.**

| their section | ours | status |
|---|---|---|
| 5.1 UMAP of action embeddings | `scripts/figures/plot_z_umap.py` | have it |
| 5.2 **action latent transfer**, scored in pixels (PSNR/LPIPS/FID/FVD) | same measurement as our readout R^2, in a better medium | **worth building** -- see below |
| 6.1 action-conditioned imagined rollout, 8 frames | `scripts/diagnostics/latent_rollout.py` (F51) | have it, different metrics |
| 6.2 planning by action selection, task success rate | nothing | the closed loop, F81 |
| 6.3 **scaling with number of embodiments** | **cannot** -- we have two | state as a limitation |

**5.2 is the one to build, and the measurement needs no decoder.** They condition the FDM on
observations from one embodiment while feeding action embeddings derived from **another**. That is a
question about the *latent*, where F51 asked about the *forward model* -- different component.
`scripts/diagnostics/cross_latent_rollout.py` does it in embedding space with two bracketing
baselines, because the clips are not phase-synchronised (F45) and a raw error would mostly measure
that.

**Correction: their FDM predicts an embedding, exactly as ours does.** An earlier version of this
entry said it was a video generator, which is why image metrics were available. It is not. Section
3.1: *"to predict the next visual embedding x_hat_{t+1}, such that x_hat_{t+1} = FDM(x_t, z_t)"*,
with `L_recon` an MSE on embeddings. **The image metrics come from a separate component**:

> "Both models use a pretrained V-JEPA2 RGB tokenizer for image encoding, which is frozen. **We use
> a custom V-JEPA2 RGB decoder to decode the predicted image embeddings into RGB image space.**"

So the architectural distance is smaller than it looked: **frozen V-JEPA2 tokenizer, FDM predicting
embeddings, and a 64-dimensional action embedding -- all three identical to ours.** The only piece we
lack is the RGB decoder, and it is needed for *presentation*, not for the measurement.

**6.3 is a limitation to declare, not to attempt.** Their headline is that LAC-WM's downstream
performance **scales positively** with the number of pretraining embodiments while EAC-WM's degrades.
Two embodiments cannot produce that curve, and adding hexapod bodies would not count: that is
within-family, which this project already established solves the wrong problem.

**Success metrics for a locomotion analogue of 6.2**, since theirs (contact, lift, place) are
manipulation-specific. The twelve matched conditions are already the task set:

    S.R. speed       |Fr_achieved - Fr_commanded| / Fr_commanded  < 15%
    S.R. behaviour   correct class by dominant channel: forward / turn / sideways
    S.R. survival    body height held above threshold, did not fall

Two departures from their protocol. **Report the graded error beside the binary rate** -- "0.14
against a commanded 0.16" carries more than pass/fail, and twelve conditions give a binary rate very
few points. And **survival is not optional for us**: a manipulator that fails a grasp is still
standing, a legged robot that fails falls over, and a success rate that ignores it flatters a
controller that reaches the speed by lunging.


---

## Files

- `sim/collect/collect_ik.py --gait cpg` -- joint-space oscillator giving the hexapod a second
  gait plus steering and a speed range, without IK (F71)
- `scripts/diagnostics/inspect_scene.py` -- list a CoppeliaSim scene's joints and read out any
  attached script; how the Olaf controller was recovered
- `scripts/diagnostics/tune_legs.py` -- solves a gain and an offset per leg so six unequal legs
  trace the same stroke; converges, and makes the gait worse, which is the point of F73
- `results/wm/dataset/figures/gait_legtune.png` -- the contact raster with and without it (F73)
- `data/ik_walk_speed7` -- five constant speeds plus both ramp directions, 91 clips (F60)
- `scripts/diagnostics/body_head_ablation.py` -- zero `z` or zero the frame on a trained body head
  and see which input the loss actually depends on (F64)
- `scripts/diagnostics/identity_linearity.py` -- linear against nonlinear classifiers on raw and
  standardised `z`; shows standardising hides identity from a straight line only (F66)
- `scripts/diagnostics/channel_screen.py` -- score each body-motion channel at two timescales
  against F69's three gates; the evidence that more behaviour is needed (F70)
- `scripts/diagnostics/target_window_sweep.py` -- is a candidate motion target readable from a
  single frame, swept over window length and direction (F68)
- `scripts/diagnostics/body_motion_probe.py` -- cross-robot Froude readout; reports both
  transfer R^2 and the seed-stable `agreement` between the two robots' readouts (F66)
- `scripts/figures/plot_body_head_design.py` -- slide 19's three-panel figure: what we had, what
  failed, what works
- `scripts/diagnostics/latent_rollout.py` -- rolls the forward model on its own output. Needs
  `--data_dir` and `--glob` on any cross-embodiment checkpoint: the config's `morph` and `data_dir`
  are stale single-morphology defaults and produce a silent zero-clip run with NaN everywhere (F63)
- `scripts/diagnostics/swap_pathway.py` -- which input the decoder reads the body from. On a
  merged speed set `--episodes` needs the block-encoded numbers, not the source ids (F61)
- `sim/collect/collect_ik.py --speed_end` -- sweep the speed across a clip so the body-speed target
  is continuous rather than one value per clip (F60)
- `data/README.md` -- which dataset to use for what, and the two collection flags that are not
  optional
- `data/ik_walk_speed5` -- five retimed speeds matched to the B1's Froude band (F57). Episode
  numbers carry the speed as a block of a thousand, so cross-body pairing stays inside one speed
- `sim/collect/collect_ik.py --speed` -- time-retime the shared foot path; every leg by the same
  map, so inter-leg phase is untouched (F57)
- `scripts/dataset/merge_speed_dirs.py` -- merge per-speed collections, rewriting `expert_episode`
  so a 92-frame clip and a 60-frame clip never pair as "episode 6" (F57)
- `scripts/dataset/preview_clips.py` -- watch clips straight from the npz before training on them
- `scripts/diagnostics/body_motion_probe.py` -- does a body-level readout transfer where a
  leg-level one does not; `--insect_dir` selects the dataset (F57, F58)
- `wm/models/body_motion.py` -- the shared body-speed head, one for all embodiments (F58)
- `wm/data/embodiment.py: body_motion, BODY_CHANNELS` -- the target, and why the lateral channel is
  excluded: it separates the robots at AUC 0.788 against forward speed's 0.543 (F58)
- `results/wm/stage2/4leg_head/fewshot_curve_c08f09t09.csv` -- the few-shot curve on held-out
  geometry (F59)

- `results/wm/stage2/measurements/ftm_cross_embodiment.csv` -- the rollouts behind F51

- `results/wm/stage2/measurements/b1_transfer.csv` -- the B1 few-shot splits (F50)

- `results/wm/stage1_correct/measurements/heldout_scores.csv` -- the five retrained runs (F49)
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

