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
parameter. F15 onward use `data/allocentric/fwd_hex8body`, five training bodies differing in three.

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
uses `data/allocentric/fwd_hex8body`: nine bodies generated by scaling coxa, femur and tibia
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
> `fwd_m3d` the same way, the errors are uniformly lower — 3.00 / 3.40 / 2.86 for
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

**The nine bodies in `data/allocentric/fwd_hex8body`, measured on their own recorded head trajectories:**

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

Stage 2 passes `hexapod=data/allocentric/fwd_hex8body` and `embodiment_split` globs `*.npz`, so nothing
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

F31 was fitted on **four clips** -- 264 samples against 1,408 encoder features. `fwd_m3d`
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
100 clips of `fwd_m3d`, same architecture, 15,000 minibatch steps, same seed. ITM+FTM
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
same frozen checkpoints, on 40 `fwd_m3d` clips the pretraining never saw:

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

`data/_archive/ik_walk_speed5`: five speeds, 0.72 to 1.10, 67 clips from 75 after `walk_check`.

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

Matched pair on `data/_archive/ik_walk_speed5` + `data/allocentric/fwd_b1_50hz`, one flag apart, 60 epochs, one seed.

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

`data/allocentric/fwd_hex7speed` = the five constant speeds plus both ramp directions, 91 clips from 105 after
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
| `fwd_hex8body`, one speed | **the latent** | 3.1x / 3.8x |
| `_archive/ik_walk_speed5`, five speeds, **no term** | **the frame** | 2.9x / 3.9x |
| `fwd_hex7speed`, seven conditions, **no term** | **the frame** | 5.0x / 3.7x |
| `fwd_hex7speed` + `L_body` | the frame | 4.8x / 4.5x |

Almost a mirror image, and **the controls did it** -- `lambda_body 0.0`, and the checkpoint confirms
no shared head was ever built. `L_body` adds nothing here (4.8/4.5 against 5.0/3.7, within noise).

**The confounds were checked.** Both runs trained on the same four bodies (`fwd_hex8body`'s nine
minus the non-walkers leaves five; `stage2_clean` held out three and `speed5` held out one, landing
on the same four), the same 5 clips per body, the same 60 epochs, the same architecture. And
scoring `stage2_clean` on the *speed-varied clips* still gives the latent at 3.1x/3.8x, so it is
the training data that matters and not the evaluation data.

This is what `lambda_cross` does in Stage 1 and what the adversary never managed (F59). **It was
achieved by making the insect walk more than one speed.**

**Isolated 2026-08-18.** `fwd_hex8body` and `fwd_hex7speed` are separate collection runs, so
"speed variation" needed separating from "the newer dataset". `s2_fwd_hex8-b1_ctrl` retrains on
the **old** data with the **new** split -- `heldout_bodies c08f09t09` alone, 5 clips per body, 60
epochs, same architecture and seed.

| trained on | reads body identity from |
|---|---|
| `fwd_hex8body`, 3 bodies held out (`stage2_clean`) | **latent**, 3.1x / 3.9x |
| `fwd_hex8body`, **1 body held out** (matched) | **latent**, 3.2x / 4.3x |
| `fwd_hex7speed`, 1 body held out | **frame**, 2.9-5.0x |

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

**And the matched run settles it.** `s2_fwd_hex8-b1_ctrl` and `stage2_clean` train on *identical
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
`fwd_hex8body` and `s2_fwd_hex7-b1_ctrl` on `fwd_hex7speed`, so scoring on either set puts one model
in-distribution and the other out. Scoring on **both** separates the two explanations: if each wins
at home it is distribution match, if one wins everywhere it is a better forward model. Rolled on
its own output against holding the frame still, four clips per cell, **both checkpoints at epoch
60**.

| body | scored on | | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|---|
| `c10f10t10` | `fwd_hex8body` | `stage2_clean` | 1.38x | 1.57x | 1.50x | 1.29x |
| | | **`speed7` ctrl** | **1.43x** | **1.66x** | **1.60x** | **1.37x** |
| | `fwd_hex7speed` | `stage2_clean` | 1.35x | 1.47x | 1.43x | 1.23x |
| | | **`speed7` ctrl** | **1.41x** | **1.58x** | **1.56x** | **1.35x** |
| `c10f06t06` | `fwd_hex8body` | `stage2_clean` | 1.32x | 1.44x | 1.40x | 1.22x |
| | | **`speed7` ctrl** | **1.39x** | **1.54x** | **1.50x** | **1.30x** |
| | `fwd_hex7speed` | `stage2_clean` | 1.27x | 1.40x | 1.38x | 1.19x |
| | | **`speed7` ctrl** | **1.34x** | **1.52x** | **1.50x** | **1.29x** |

**24 of 24 comparisons**, two bodies at opposite ends of the leg-length range (0.77 m and 0.47 m),
both evaluation sets, six horizons each. Mean gain **+7.0 percent**; **+5.6 at short horizons and
+7.8 at long ones**, so it grows with rollout length -- the direction that matters for a module
whose only job is to be rolled.

**Distribution match is ruled out by the cell that should have gone the other way.** On
`fwd_hex8body` clips -- which `stage2_clean` trained on and `speed7` never saw -- `speed7` still
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

**Isolated 2026-08-18, same run as F61.** `s2_fwd_hex8-b1_ctrl` -- old data, new split -- rolled on
the same two evaluation sets:

| scored on | | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|
| `fwd_hex8body` | `stage2_clean` | 1.38x | 1.57x | 1.50x | 1.29x |
| | **matched, old data** | 1.38x | 1.57x | 1.49x | 1.26x |
| | `speed7` control | **1.43x** | **1.66x** | **1.60x** | **1.37x** |
| `fwd_hex7speed` | `stage2_clean` | 1.35x | 1.47x | 1.43x | 1.23x |
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
embodiment key, gradient reaching the trunk. `s2_fwd_hex7-b1_bodyframe0.5`, same data, same weight,
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

Swept at `lambda_body 0.1`, everything else identical, same control (`s2_fwd_hex7-b1_ctrl`, epoch 60):

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
That is `s2_fwd_hex7-b1_ctrl` at r = -0.048, arrived at independently.

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
trained `z` (`--ckpt s2_fwd_hex7-b1_body0.5`):

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
legs in `fwd_hex8body` could still reach a shared absolute coordinate. Once the collection is one
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
| B1 (`data/allocentric/fwd_b1_50hz`, and today's first pass) | 99-126 | ~2.0-2.5 s | **20 ms** |

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

**`data/allocentric/fwd_b1_50hz` still carries the old rate.** It is not deleted, because the results that cite it
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
five were constants in our data**, because both robots only ever walked forwards. `data/allocentric/beh12_*`
removes that -- twelve matched conditions per robot spanning speed, turn and sideways travel,
balanced 4/4/4, 48 clips a side. This is the re-test.

`screen_behaviour_channels.py` scores four channels (F70's three plus **yaw**, which the old screen
had no rotational channel for) at two timescales against F69's three gates, on the frozen encoder --
`s2_fwd_hex7-b1_body0.5` predates both F71's wider foot path and F74's frame rate and has never seen
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
drawn from the frozen encoder**, because `s2_fwd_hex7-b1_body0.5` predates both F71's wider foot path and
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
completely, since both robots start along +x -- and every dataset before `data/allocentric/beh12_*` contained
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

Three arms on `data/allocentric/beh12_*`, identical except for the body term. Held out by condition, frozen
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
identical architecture. This is the question `data/allocentric/beh12_*` was built to ask, and the answer is that
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
| old `fwd_hex7speed` | 0.045 | 91 |

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
`data/allocentric/beh12_hex2/` and was not merged.


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

| closest pairs in `data/allocentric/beh12_c10f10t10_flat` | separation / own noise |
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

`data/allocentric/beh12_c08f09t09_flat`: the same twelve conditions collected on `c08f09t09`, the body every
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

> **`S.R. behaviour` is a whole-run verdict, and F102 measures the per-step version.** These runs
> are in the right behaviour family on **47-71%** of their planned steps while scoring 100% here,
> because the run as a whole stays dominated by the right channel even when a third of the
> decisions go elsewhere. Both numbers are correct; this one is the flattering one.

**It halves the error and restores the behaviour class outright.** Selection becomes decisive as
well as correct: on the sideways demonstration the refitted arm holds `side_R_lvl1` for 30-34 of 49
steps with 18-20 changes, where the un-refitted one settled on the *weak* strafe -- which on this
body travels the wrong way (F94) -- and switched about 30 times.

> **Repeated fifteen times (five per demonstration) after F105 showed CoppeliaSim physics does not
> repeat.** Survival and behaviour class hold at **15/15 each** -- neither was a lucky draw. Speed
> passes on **47%** of runs, median error **19.0%**, and the spread is the part a single run hides:
>
> | | error on **its own channel** | within 15% |
> |---|---|---|
> | forward | 20% ± 12, range 2-36 | 2/5 |
> | turning, `s0.05` | **130% ± 105**, range 50-331 | 0/5 |
> | turning, `s0.29` | **13% ± 8**, range 4-27 | 4/5 |
> | turning, `s0.56` | **79% ± 6**, range 68-83 | 0/5 |
> | sideways | 39% ± 24, range 18-84 | 0/5 |
>
> **Turning was never scored on turning until now.** The criterion picked whichever channel was
> largest in the demonstration, and forward speed exceeds yaw in *every* turn condition on this
> body -- 0.136 against 0.088 even at `s0.56`. The 7% ± 2 this table used to report for turning was
> a **forward-speed** measurement on a clip whose yaw error is 130%. Graded on yaw, the run-level
> rate falls from **47% to 13%**, and only the middle turn rate is tracked at all.
>
> **Re-run again after F106 corrected two of the candidate library's twelve conditions**, since the
> first fifteen used the broken library: survival and behaviour hold at 15/15 either way, speed
> stays at 47% under the old criterion, median 19.0% to 17.7%. Sideways does not move, which is the
> same answer the direct test gave.
>
> **Sideways is inconsistent rather than broken** -- across both sets of repeats it has landed as
> low as 2% and as high as 84%. Earlier entries reporting a single sideways number near 74% were
> reporting one draw from that range.

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

> **Measured on the B1 only, and the hexapod does not agree at one step.** Same three rules on the
> held-out hexapod body: `rollout` **63%**, `direct` **89%**, `blind` 35% at horizon 1 -- deleting
> the forward model *helps* by 26 points. At horizon 5 the order returns to normal, `rollout` 80%
> against `direct` 59%. **The claim below holds for the quadruped at every horizon and for the
> hexapod only beyond one step.** Why the inverse model's reading of (now, goal) should beat a
> one-step rollout on one robot and not the other is unexplained.

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

| | seeded | standing start |
|---|---|---|
| survival, all three | **65 / 65** | 65/65, fell at 29, fell at 37 |
| peak body height | 0.57-0.60 | 0.67-0.70 |

> **A second defect, larger than the first, found by F102's diagnostics.** Both earlier versions of
> this loop **moved the camera with the robot**. Every clip the model trained on places the camera
> once, from the trajectory's first frame, and never moves it -- so the robot travels across a
> fixed view, and **the background sliding is the cue that says how far it went.** A following
> camera keeps the robot centred and deletes that cue. Its one-step forward-model error on those
> frames was **2.9x** the error on recorded clips while the frames themselves scored as barely
> novel, because a pooled embedding hardly registers the difference and the forward model entirely
> does. **`close_loop_kinematic.py` and `close_loop_ik.py` place the camera once and are
> unaffected; the defect was in this file alone.**

| | camera fixed | camera following *(discarded)* |
|---|---|---|
| turning | family **100%**, exact **95%** | family 51% |
| forward | family **71%** | family 58% |
| sideways | family 35% | family 38% |
| *chance* | *28%* | *28%* |

**Turning is what the defect was suppressing, and it is now the strongest result on the
quadruped**: the planner chooses `turn_wz0.40` -- the exact condition, not merely the family -- on
**52 of 55** planned steps. It was the behaviour that had failed every previous measurement, and
**rotation is read from the background turning**, which is exactly what a following camera
destroys.

**The robot stands through every episode.** Scored on the project's criteria: **survival 3/3,
behaviour class 2/3, speed 0/3** -- errors of 18.7%, 25.3% and 91.7%.

**So the quadruped holds itself up, resolves one behaviour outright, and tracks speed on none of
them.**

**The planner is not what makes a robot fall.** In the standing-start runs sideways survived **37**
steps under the planner against **27** replaying its own clip. Re-deciding every step holds the
robot up slightly longer than re-issuing a recorded sequence.

**What this replaces.** `close_loop_kinematic.py` exists because the physics loop was believed
impossible; its survival column passes by construction and had to be reported as `n/a`. **The B1
now has a survival number that means something**, and the honest form of F93 is that a recorded B1
action sequence holds for about three seconds and not six.

---

### F102. The loop is in the right behaviour on about half its steps, and "behaviour 100%" was never measuring that

**Every closed loop in this project reaches the right behaviour and the wrong speed** -- 33% within
15% on a held-out hexapod body, 0 of 3 on the quadruped. Four explanations were tested from files
already on disk, without running anything. **Three are refuted.**

| tested | result |
|---|---|
| the candidate library is too coarse to hit the demonstrated speed | **refuted** -- a behaviour travelling at the right rate was in the list on **9 of 9** runs, and was chosen on 3 |
| the score cannot see speed, only behaviour | **refuted** -- within-family score spread is **67%** of between-family spread |
| the score sees speed but orders it wrongly | **refuted where it matters** -- rank correlation between score order and speed-match order, inside the correct family: **+0.88** sideways, +0.25 forward, **-0.06 turning** |
| switching candidates often is what costs speed | **refuted** -- correlation between switch rate and achieved/demanded speed is **+0.14** |

**What is left is not a speed problem.** Counting how often the planner is in the *right behaviour
family*, step by step:

| | per-step in-family |
|---|---|
| hexapod, held-out body, six runs | **47%, 55%, 61%, 61%, 67%, 71%** |
| B1, physics, three runs | **35%, 71%, 100%** *(after the camera fix in F101; 38/51/58% before)* |
| *what these runs report as "behaviour"* | *100%, and 2/3* |

**Both numbers are correct and they answer different questions.** `S.R. behaviour` asks whether the
*run as a whole* ended up dominated by the right channel; this asks how often the *decision* was
right. A run can walk forward while a third of its steps chose something else, because the wrong
picks scatter and the right family stays the plurality. **Only the flattering one has ever been
reported.**

**And when it is in the right family it picks the right amplitude.** Mean speed of the chosen
candidates over the demonstration's, restricted to in-family steps: **0.90 to 1.35**, against
**0.31 to 1.08** when out-of-family steps are included. The earlier reading that the planner
"picks slow candidates" was an artefact of that averaging -- a forward candidate has near-zero
lateral speed, so counting it against a sideways demonstration drags the mean toward zero.

**A second term, which is F93 restated with a number rather than a new result.** Replaying a single
B1 clip alone, seeded exactly as the loop seeds it and with no planner involved, reaches **0.84,
0.76 and 0.99** of its own recorded speed. The B1's action is a policy's response to state, so
replaying it open loop drifts and the drift costs travel -- known since F93; the contribution here
is only the size. **The hexapod's actions come from IK and a clock and replay exactly, so this term
should be absent there and has not been measured.**

**The second term tested on the other robot, where F93 predicts it should vanish.** The hexapod's
commands come from IK and a clock and read no state, so replaying them should reproduce the motion
exactly. Replayed through the same physics the closed loop uses:

| | hexapod | B1 |
|---|---|---|
| forward | **1.06** | 0.84 |
| turning | **0.96** | 0.76 |
| sideways | 0.82 | **0.99** |

**The prediction holds for forward and turning and inverts for sideways.** Where the two robots
differ in the way F93 describes -- a clock-written command against a policy's response -- the
hexapod replays essentially exactly and the B1 loses a fifth to a quarter. **Sideways does the
opposite on both robots and F93 does not explain it.** One clip per condition, and the hexapod's
recording was made in a different simulator session from this replay, which F101's camera defect
showed is not a neutral difference -- **0.82 may be session variance rather than replay loss, and
separating them needs repeats inside one session.** Recorded as unresolved.

**The obvious fix for the B1 does not work, and the reason is instructive.** If the loss comes from
replaying a *response*, record the joint angles the robot actually reached and replay those as
targets -- a fixed table, exactly like the insect's. Measured: the robot stays upright and
**travels almost nowhere**, at 0.01, 0.06 and 0.28 of the recorded speed.

**A motor makes force from the gap between the target and where the joint is.** Achieved positions
lag their targets under load, so commanding the achieved position leaves no gap, no torque and no
push. **And the table we already replay is the right one**: `DEFAULT_IL + ACTION_SCALE x action` is
the sequence of *targets* the policy commanded, stored as 66 rows, structurally identical to the
insect's IK table. The difference between the robots is not table against policy -- both are
tables -- but that **the B1's table was written with reference to states the replay cannot
recreate.**

**A third term, first estimated as a residual and then measured directly.** Dividing the achieved
speed by (picks x replay) left 1.08 forward, 0.59 turning, 0.20 sideways -- **a residual, which
absorbs every error in the other terms**, so it was written down as a hypothesis. Tested directly
(`scripts/diagnostics/what_stitching_costs.py`): replay the exact command sequence the loop
executed, then replay the single candidate it chose most often, same seeding, same length, no
planner and no vision in either.

| B1 demo | switches | stitched | single clip | ratio | residual predicted |
|---|---:|---:|---:|---:|---:|
| forward | 44 | 0.122 | 0.102 | **1.19** | 1.08 |
| turning | 36 | 0.059 | 0.118 | **0.50** | 0.59 |
| sideways | 38 | -0.020 | -0.109 | **0.18** | 0.20 |

**The direct measurement lands on the residual's estimate in all three**, so the term is real:
**switching between recorded clips costs half the turning and four fifths of the lateral travel,
and costs forward nothing.** Both arms are replays, so selection quality and replay fidelity are
held out of the comparison -- the only difference is whether the sequence switches clips.

**The mechanism is directional cancellation.** Every candidate in the library travels forward to
some degree, so a stitched sequence still goes forward. Turning and strafing need the *direction*
to be held; interleaving clips that turn at different rates leaves the rotations partly cancelling.
This is why the sideways run misses its channel by 93% while picking the right amplitude whenever
it is in the right family -- it is out of that family on 62% of steps, and what remains cancels.

**And the diagnostics found a defect the metrics never would have.** Chasing why the loop's frames
gave a forward-model error 2.9x the recorded one -- when a novelty check said those frames were
barely unusual -- turned up a camera that followed the robot, which no summary statistic in this
project reports on. Fixing it took the turning demonstration from 51% to **100%** family and 95%
exact. **The contradiction between two measurements is what exposed it**, not either one alone.

**What to do with this.** The per-step family rate is the number that predicts everything else and
it is the one to improve. It is also the number that a distilled policy would attack directly,
since the planner's wrong picks come from states it drove itself into and was never scored on.
`scripts/diagnostics/why_speed_misses.py` and `does_score_see_speed.py` reproduce the table above.

---

### F103. Committing to a behaviour for three steps is the first thing that has hit a speed target

F102 measured that switching between recorded clips costs travel -- half the turning and four
fifths of the lateral, forward none -- by replaying the stitched sequence against a single clip.
**The loop switches because nothing stops it: `--commit` defaults to 1, meaning re-decide every
step, and no run in this project has ever used another value.** That default was never justified.

Holding the chosen behaviour for `commit` steps before deciding again, B1, physics, three
demonstrations:

| commit | speed within 15% | behaviour | survival | speed errors |
|---|---|---|---|---|
| **1** | **0 / 3** | 2/3 | 3/3 | 18.7%, 25.3%, 91.7% |
| **3** | **2 / 3** | 2/3 | 3/3 | **13.1%, 6.4%**, 85.5% |
| **5** | 1 / 3 | 2/3 | 3/3 | 39.0%, 9.8%, 94.8% |

**Three steps is the first setting anywhere in this project to clear the speed criterion on a
quadruped**, and it does it on two demonstrations out of three. Forward goes from 25.3% error to
**6.4%**, turning from 18.7% to **13.1%**.

**It is a trade, and the other side of it is selection.** Per-step behaviour accuracy on the
turning demonstration, which is the one the planner resolves best:

| commit | family | exact condition |
|---|---|---|
| 1 | **100%** | **95%** |
| 3 | 95% | 89% |
| 5 | 91% | 76% |

**Six points of exact-condition accuracy for two speed targets.** Re-deciding every step tracks the
behaviour better and executes it worse, because every switch interrupts the stride; committing
executes better and corrects later. **Five is past the optimum** -- it holds a choice long enough
that the recovery arrives too late, and turning's error triples back to 39%.

**Sideways does not move** -- 91.7% to 85.5% -- because its problem is not stitching. It is out of
the right behaviour family on **65%** of steps (F102), and committing only holds a wrong choice for
longer.

**This was predicted by F102 and is the confirmation.** The stitching term was measured on replayed
sequences with no planner involved; if it is real, reducing the switching should return the speed
it costs, and it does. **The default of 1 was a free parameter nobody had questioned**, and on this
robot it was costing the loop the one criterion it had never met.

### It does not reverse on the hexapod -- the reversal was two tails of one distribution

**Reported first from one run per setting**, forward at 0.4% under `commit 1` against 39.7% under
`commit 3`, and read as a reversal driven by replayability. **Re-run five times per setting after
F105:**

| | commit 1 | commit 3 |
|---|---|---|
| forward | 23% ± 14, range **1-37** | 15% ± 13, range **4-40** |
| turning | 11% ± 5, range 4-19 | 13% ± 16, range 2-45 |

**The two distributions overlap almost entirely, and the original comparison took the bottom of one
against the top of the other.** `commit 3` is if anything slightly better on forward. **There is no
hexapod reversal**; the claim is withdrawn.

**The B1 half stands**, because MuJoCo repeats bit for bit (F105) and one run is the answer there:
speed 0/3 at `commit 1` against **2/3** at `commit 3`, errors 25.3% and 18.7% falling to 6.4% and
13.1%.

**So `commit` helps the quadruped and is neutral on the insect**, which is weaker than the
symmetric story and is what the measurements support. The mechanism behind the B1 half -- switching
clips costs travel, measured on replays with no planner in F102 -- is unaffected: that comparison
is replay against replay inside one session.

> **The lesson is procedural.** The reversal was an attractive result: it tied `commit` to F93's
> replayability and explained both robots with one idea. **It came from two single runs of a
> configuration whose spread was later measured at 1-37%.** Nothing about the story was wrong in a
> way inspection would catch -- only repeats caught it.

---

### F104. Offline ranking accuracy predicts the loop backwards, and the two robots fail sideways for opposite reasons

**Ranking on recorded clips and choosing inside the loop are not the same skill, and on the B1 they
are close to inverted.** Behaviour-family accuracy, same checkpoint, same candidates:

| behaviour | on recorded clips | inside the physics loop |
|---|---|---|
| sideways | **97-100%** | **35%** |
| turning | 55-63% | **100%**, exact condition 95% |
| forward | 32-36% | 71% |

**Sideways is the behaviour the ranking resolves best and the loop resolves worst.** Anything
inferred about the loop from clip-level scoring -- including the 62% headline in F100 -- describes
a different problem.

**The step sequences say what happens, and it is not lock-in.** Printing the first two dozen
planned choices:

    B1,      sideways demo    side turn turn turn side side side turn turn turn side turn turn ...
    hexapod, sideways demo    side side side side side side side side side side side side side ...

**The B1 oscillates; it does not commit to a wrong answer, it fails to hold the right one.** The
hexapod holds it on roughly 96% of steps -- **and still misses the lateral speed by 73.8%.**

**So the same behaviour fails on the two robots for different reasons.**

| | choice | execution |
|---|---|---|
| B1, sideways | **cannot hold it** -- 35% in family | -- |
| hexapod, sideways | holds it, ~96% | **cannot achieve it** -- 73.8% speed error |

**The B1's half traces back to F93 once more.** Commanding a sideways candidate does not make the
B1 strafe, because its recorded actions are policy responses and replay at 0.76-0.99 of their own
motion; the next frame is therefore not a sideways frame, the ranking that scored 97% on real
sideways frames no longer applies, and the choice flips. **The ranking is not wrong -- it is being
asked about a state the robot failed to reach.** The hexapod's commands replay at 1.06 and 0.96,
its frames stay sideways, and its choices stay correct.

**The hexapod's half is unexplained.** It picks correctly and does not travel: F102 measured its
sideways clip replaying at **0.82**, the worst of its three, and the loop reaches 0.26 of the
demonstrated lateral speed. Neither number has an account yet.

### Eight hypotheses tested and refuted in one day

Kept as a list because each was plausible enough to have been built on:

| | refuted by |
|---|---|
| the candidate library is too coarse | the right-speed behaviour was in the list on **9 of 9** runs |
| the score cannot see speed | within-family spread is 67% of between-family |
| the score orders amplitudes wrongly | rank correlation **+0.88** inside the correct family |
| switching often is what costs speed | correlation with switch rate **+0.14** |
| gait phase drift explains the prediction error | phase-broken control scored **1.33x**, below the in-phase control's 1.43x |
| replaying achieved joint positions fixes the B1 | the robot stands still -- no tracking error, no torque |
| sideways was already bad at ranking | it ranks at **97-100%**, the best of the four |
| the loop locks into its first choice | the B1's sideways run oscillates rather than locking |

**What survived is one root and one defect**: F93's replayability, which decides the frames the
loop sees and therefore everything downstream, and a camera that followed the robot (F101).

---

### F105. CoppeliaSim physics does not repeat, MuJoCo does, and it decides which numbers in this project can be read from one run

**Repeating one closed-loop configuration five times, changing nothing:**

| | lateral speed error, five runs |
|---|---|
| hexapod, `--commit 1` | **50% ± 12**, range **37-71%** |
| hexapod, `--commit 5` | 58% ± 23, range 23-92% |
| hexapod, `--commit 10` | 68% ± 35, range 27-111% |
| hexapod, `--commit 20` | 39% ± 30, range 10-87% |

**Nothing here is significantly better than `commit 1`.** 39 ± 30 against 50 ± 12 at n=5 is under
one standard error of the difference, and what longer commitment reliably does is **widen the
spread** -- fewer decisions per episode, so the outcome rides on whether three or four choices
happen to be right. **The single-run reading that suggested commitment fixes sideways (24.1% at
commit 20) is the bottom of a range that reaches 87%.**

**The two robots' loops are not equally repeatable, and it is the physics engine.**

| | engine | same settings, two runs |
|---|---|---|
| B1 | **MuJoCo**, seeded from the demonstration's first frame each time | **identical** -- choices, frames and body track, bit for bit |
| hexapod | **CoppeliaSim**, scene reloaded per run | 37-71% spread on the same configuration |

CoppeliaSim reloads the scene for every run and its solver and contact state do not come back
identical; MuJoCo is initialised explicitly here and does.

**What this means for numbers already reported.**

| | status |
|---|---|
| every B1 loop result -- F101, F103's B1 half, F104's B1 half | **safe from this.** MuJoCo repeats exactly, so one run is the answer |
| F95's hexapod loop -- survival 100%, behaviour 100%, 19.2% median error | **re-run fifteen times since.** Survival and behaviour hold at 15/15; speed passes 47% with median 19.0%. **The headline survives; the per-behaviour spread is wide and is now reported with it** |
| **F103's hexapod half** -- commit 1 at 0.4% forward against commit 3 at 39.7% | **one run each. Both are inside the spread measured here and the comparison does not stand as reported** |
| F102's hexapod replay fidelity -- 1.06, 0.96, 0.82 | one clip each, **and the recording came from a different session.** The 0.82 outlier may be session variance |

**The rule this sets.** A CoppeliaSim-physics number needs repeats before it carries a comparison;
a MuJoCo one does not. **That was not known while most of this project's loop results were being
collected**, and the hexapod ones were read as if a single run were the answer.

---

### F106. Two of the held-out body's four lateral conditions travel the wrong way -- a real defect, and not the one the loop was failing on

The loop misses lateral speed on the held-out hexapod and the misses are inconsistent -- five
repeats of one configuration span **2% to 64%** error (F105). Offline ranking pointed at an
asymmetry: `side_L` **96%**, `side_R` **77%** at horizon 5, the weakest family and the only one
that does not improve with a longer rollout. **A mirror-image pair should not differ by 19 points**,
and the first explanation reached for was the single fixed viewpoint -- strafing toward the camera
and away from it are not mirror images in the image.

**That was wrong. The conditions are not mirror images because two of them travel the wrong way.**
Median lateral speed per condition:

| | `side_L_lvl0` | `side_L_lvl1` | `side_R_lvl0` | `side_R_lvl1` |
|---|---|---|---|---|
| `beh12_c10f10t10_flat` | +0.071 | +0.185 | -0.118 | -0.186 |
| `beh12_b1_flat` | +0.066 | +0.152 | -0.119 | -0.169 |
| **`beh12_c08f09t09_flat`** | **-0.045** | +0.148 | **+0.017** | -0.131 |

**On the held-out body both `lvl0` conditions strafe opposite to their names**, with a standard
deviation of 0.001 across clips -- consistent, not noise. The two robots the recipe was authored on
are correct; only the body it was ported to is wrong.

**`collect_beh12.py` predicts exactly this.** Its own docstring says the commands are not portable
across bodies: a lateral recipe is a twist on the middle joint scaled about each body's hip, and
what produces a gentle left strafe on the base geometry does not on shorter legs. **At `lvl1` the
amplitude is large enough to survive the port; at `lvl0` it is small enough for the sign to flip.**

**So the 77% is not a viewpoint limitation and not a model limitation.** The ranking was asked to
separate `side_R_lvl0`, which travels **left**, from `side_L_lvl1`, which also travels left. They
are the same behaviour under different names, and confusing them is correct.

**Two things this invalidates, both of them mine.**

| | |
|---|---|
| the camera explanation, written into this entry an hour earlier | withdrawn -- the asymmetry is in the labels, not the optics |
| `corr(error, left-picks) = +0.72` over five runs | meaningless -- it counted `side_L_lvl0` as a left pick while that clip travels right |

**And it reaches further than sideways.** Every `in-family` figure for the lateral behaviours on
this body -- in F102, F104 and F105 -- groups conditions that move in opposite directions under one
label. **The forward and turning numbers are unaffected**; the lateral ones need the conditions
re-derived for this body before they mean anything.

**And the two `lvl0` conditions are not strafing the wrong way so much as barely moving at all.**
All three channels on the held-out body:

| | forward | lateral | yaw |
|---|---|---|---|
| `side_R_lvl0` | -0.009 | **+0.017** | +0.021 |
| `side_L_lvl0` | -0.012 | **-0.045** | -0.017 |
| `side_R_lvl1` | +0.016 | -0.131 | -0.002 |

**`side_R_lvl0` is motionless in every channel.** The recipe under-drives the shorter legs, and the
sign that survives is the residue rather than a strafe in the opposite direction.

**The check that exists does not catch this, and it was run.** `collect_beh12.py --separability`
asks whether the twelve conditions are further apart than their own spread; on this body it passes
-- **2 of 66 pairs below 2x, and the close pairs are speeds and turn rates, not the lateral ones.**
A condition that barely moves is still comfortably separable from one that moves a lot. **The
missing check is semantic: does `side_L` travel left, does `side_R` travel right, does each `lvl1`
exceed its `lvl0`.** Three lines against the achieved channels, and it would have caught this
before the body was used for the project's headline result.

**The fix is collection, not modelling** -- re-derive the two lateral levels for this geometry, as
`collect_beh12.py`'s own docstring says has to be done per body. **Done**: `--lvl0_strafe 0.7`
(0.4 is the base body's value, 0.6 flips the sign back but leaves the motion at a fifth of `lvl1`),
giving **+0.076 and -0.069** against a target of half of `lvl1`. The eight clips were replaced and
the dataset now passes both checks.

### And fixing it changed nothing in the loop

| | sideways speed error, five repeats |
|---|---|
| before the fix | 34% ± 22, range 2-64, within 15% on **1/5** |
| **after the fix** | **35% ± 23**, range 15-67, within 15% on **0/5** |

**Identical.** The dataset was genuinely broken and is now correct, and **it was not what the loop
was failing on.** The explanation offered above -- that the ranking was being asked to separate two
conditions that travel the same way -- is measured and does not hold.

**So the sideways failure survives every explanation tried today**: the library, the score, the
amplitude ordering, switch frequency, gait phase, lock-in, the camera angle, and now the condition
labels. What is left is that the loop resolves lateral travel worse than forward or turning on both
robots, for a reason nothing measured has reached. **It is the open question this round ends on.**


---

### F107. A quadruped walks forward from a stick insect's video, and the diagnostic that was meant to predict this got it backwards

**The demonstration the project exists for.** Every cross-embodiment number until now compares
*representations*: a readout fitted on one robot applied to the other, a latent's transfer score.
This is the control version -- **the goal image is a hexapod, the robot driven is the B1**. The
candidates stay B1 clips, because only those are executable, so only the goal crosses. That is the
only form this demonstration can take without a motion decoder that generalises across bodies,
which nothing here has.

Physics, MuJoCo, `--commit 3`, behaviour-family accuracy over planned steps against a **28%** chance
rate:

| behaviour asked for | goal is a **B1** clip | goal is a **hexapod** clip |
|---|---|---|
| forward | 42% | **67%** |
| sideways, `lvl1` | 31% | 2% |

**Forward walking crosses, and crosses better than the same robot's own video does.** The B1 stays
upright for the full episode and spends two thirds of its steps on forward candidates while looking
at a six-legged insect.

**Turning crosses in proportion to how much turning there is**, which only became visible after the
first attempt used the wrong clip. `turn_s0.05` is the gentlest of four turn rates and the scorer
classifies it as a *forward* clip -- its yaw is -0.007 against `turn_s0.56`'s -0.088:

| hexapod goal | its yaw | B1 chooses a turn |
|---|---|---|
| `turn_s0.05` | -0.007 | 18% -- **below chance** |
| `turn_s0.29` | -0.037 | 38% |
| `turn_s0.56` | **-0.088** | **47%** |

**A dose-response, and stronger evidence than any single point.** The reading it replaces --
"turning does not cross at all" -- came from testing turning with a clip that barely turns.

> **This table is withdrawn by F109.** Every run in it was warm-started with the same *turning* B1
> clip, and re-running with a forward warm start moves the two turn rows to 27% and 27% -- below
> chance, and no longer ordered. The dose-response was a property of the warm start. **Forward and
> sideways survive the change; turning does not.**

**So the ordering is forward 67%, a real turn 47%, sideways 2%**, against 28% chance. (**The turn
figure does not survive F109**; forward and sideways do.) **The split is
physical rather than representational.** Forward travel looks the same from the side whatever the
leg count: the body translates across the frame. A strong turn also reads across, because the whole
body rotates. Strafing does not -- a hexapod crabs by twisting its middle joints and a quadruped
does something else entirely. **The latent carries what the two robots share and not what they do
differently**, which is what a morphology-agnostic action is supposed to mean and is the first time
this project has shown it acting on a robot rather than in a readout.

> **The same clip has been standing in for "turning" throughout.** `hexapod_ep1001` is
> `turn_s0.05`, and it is the turn demonstration behind F95's hexapod loop as well -- where turning
> scored best of the three behaviours at 7% ± 2. **That number is a forward-speed measurement on a
> clip the scorer calls forward.** The hexapod loop has not been run on a real turn.

**And the measurement built to predict this predicted the opposite.** `z_crosses_bodies.py` asks,
for each hexapod clip, whether the nearest B1 clip shares its behaviour:

| | clip-level retrieval | in the loop |
|---|---|---|
| turning | **100%** | 18% at `s0.05`, **47%** at `s0.56` |
| forward | 19% | **67%** |
| sideways | 0-50% | 2% |

**Backwards on forward, and on turning it is right about the direction for the wrong reason** --
retrieval pooled all four turn rates together, so its 100% is dominated by the strong turns the
loop also handles best.** Averaging `z` over a clip and taking a nearest neighbour is not what the
planner does: the planner scores `FDM(e_t, proj(a))` against the goal, so the B1's *current state*
is inside every comparison, and a hexapod latent never has to sit near a B1 latent. **The cheap
proxy was wrong in the direction that would have cancelled the experiment** -- it said to run
turning, which fails, and not forward, which works.

> **Three defects found while getting here, all in one file.** Two runs differing only in the goal
> robot wrote the same output name and the first was lost; the goal's embedding was centred with
> the driven robot's offset; and the automatic verdicts printed by both diagnostics -- "reachable",
> "not reachable" -- were threshold comparisons of two overlapping means. **The numbers were right
> and the sentences the scripts printed about them were not.**

---

### F108. Turn rate was never scored in any closed-loop run, and the criterion is why

`S.R. speed` grades a run on **the channel largest in the demonstration**. That was written when
every clip walked forwards and it has been wrong for turning ever since: **forward speed exceeds
yaw in all four turn conditions on both bodies**, 0.136 against 0.088 even at the strongest,
`turn_s0.56`. The criterion therefore graded every turning run on **forward speed** and yaw was
never measured in a loop at all.

Graded on yaw, on the held-out hexapod, five repeats each:

| turn rate | demonstrated yaw | yaw error | within 15% |
|---|---|---|---|
| `s0.05` | -0.007 | **130% ± 105** | 0/5 |
| `s0.29` | -0.037 | **13% ± 8** | 4/5 |
| `s0.56` | -0.088 | **79% ± 6** | 0/5 |

**Only the middle rate is tracked.** The gentlest is missed by more than its own magnitude; the
strongest is missed by 79% with a spread of 6, so it fails consistently rather than noisily.

**And `turn_s0.05` is the clip every turning claim in this project rests on.** It is the turn
demonstration in F95's loop, in the commitment sweeps, and in the first cross-embodiment attempt in
F107. **Its celebrated 7% ± 2 is a forward-speed measurement on a clip that barely turns.**

| | before | graded on the named channel |
|---|---|---|
| `S.R. speed`, held-out hexapod, 15 runs | 47% | **13%** |
| median error | 17.7% | 36.2% |

**Survival and behaviour class are unaffected** -- 15/15 each -- because neither depends on the
channel choice. **What falls is the claim that the loop tracks the commanded rate**, which was
resting on turning passing 5/5.

**This is F91 arriving in the loop.** F91 measured a planner over recorded behaviours resolving
speed 9/9, sideways 6/6 and turn **2/9**, with every turn miss being a turn at the wrong rate --
*it knows it is turning and cannot say how hard.* That was a ranking measurement on recorded
frames. **The same shape now appears in physics**, and it was hidden for as long as the criterion
looked at the wrong channel.

**The fix is one line and it is semantic, not statistical**: a condition named `turn` is graded on
yaw, `side` on lateral, `speed` on forward. Choosing by magnitude is what let a turn be graded as a
walk -- the same failure mode as F106's separability check, which asked whether conditions differ
without asking whether they differ **in the way their names claim**.

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
`data/allocentric/beh12_*` is on Froude and w_hat rather than on m/s and rad/s.

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


---

### F109. The loop's turning is set by the ten steps that precede it, not by the goal

**Found by watching, not by reading.** The cross-embodiment videos showed the B1's head drifting at
the start of every run and then, on the `turn_s0.29` goal, arcing *left* while the insect in the
goal pane turned right. Neither is visible in any number reported in F107, because
behaviour-family accuracy counts **which label the planner picked** and says nothing about which
way the robot went.

**The confound is mine.** Every run in F107 passed `--demo data/allocentric/beh12_b1_flat/b1_ep1301.npz`, which
is `turn_wz0.40`. `--demo` supplies the starting state *and the first ten actions*, so the loop
opens by executing a turn regardless of what the goal asks for.

**The control: change only the warm start.** `b1_ep1301` (turning) versus `b1_ep2`
(`speed_vx0.30`, forward), same checkpoint, same goals, same `--commit 3`. Yaw is the median over
planned steps; family chance is 33% for `speed` and `turn`, 17% for `side_R`.

| hexapod goal | goal yaw | turn warm: yaw / family | forward warm: yaw / family |
|---|---|---|---|
| `ep1` `speed` | +0.003 | +0.041 → **-0.031** / 67% | -0.019 → **-0.023** / **84%** |
| `ep1200` `turn_s0.29` | -0.038 | +0.041 → **+0.004** / 38% | -0.020 → **-0.016** / 27% |
| `ep1300` `turn_s0.56` | **-0.086** | +0.041 → **-0.025** / 47% | -0.021 → **+0.005** / 27% |
| `ep2301` `side_R` | -0.003 | +0.041 → +0.002 / 2% | -0.019 → -0.022 / 0% |

**The robot's yaw tracks the warm start and ignores the goal.** Shifting the warm start by 0.06
shifts every planned yaw with it; varying the goal across a thirty-fold range of commanded yaw
moves it by almost nothing. **`ep1300`'s correct sign under the turning warm start was a
coincidence** -- it reverses when the warm start does.

**And the family numbers move with the warm start too**, which is the more damaging half. The two
turn goals fall from 47% and 38% to 27% and 27%, crossing from above their 33% chance rate to
below it. **F107's turn dose-response was measuring the clip I chose to start the robot with.**

| | turn warm | forward warm | verdict |
|---|---|---|---|
| forward | 67% | **84%** | above chance both ways -- **stands, and is stronger without the turning warm start** |
| turning | 47% / 38% | 27% / 27% | crosses the chance line when the warm start changes -- **withdrawn** |
| sideways | 2% | 0% | below chance both ways -- **fails, as already reported** |

**What crosses embodiments in an actual control loop is forward travel, and only that.**

> **Corrected by F126 and then reframed by F127 (2026-08-30). Read all three or none.**
> "Forward is the robust behaviour and turning the broken one" is not a fact about the behaviours.
> F126 measured the ordering **inverting** on same-robot goals -- turning 53% at one step rising to
> 89% at ten, forward weakest at 30-46%. **F127 then showed why neither ordering answers the
> question asked here**: under a mismatched goal the planner still scores 56-70% against the
> *demonstration* and 18-23% against the goal it was shown, below chance. The loop is not
> conditioning on its target, so **"which behaviour transfers" is not measurable from these runs at
> all** -- what the numbers rank is how identifiable each behaviour is from its own dynamics.
> Turning is the most classifiable family; forward the least. **Do not carry "forward crosses,
> turning does not" into a write-up.**

**Cutting the warm start entirely does not rescue the turn, and the B1 walks fine without it.**
`--warm_start 0` seeds the state from the demonstration's first frame and hands the planner a
standing robot from step 0. All four goals survive **65 of 65 steps** at 0.054-0.119 m/s, so the ten
steps were never load-bearing for locomotion. Family accuracy across all three settings:

| | warm = turn clip | warm = forward clip | no warm start | chance |
|---|---|---|---|---|
| forward | 67% | 84% | 71% | 33% |
| turning | 47% / 38% | 27% / 27% | 37% / 34% | 33% |
| sideways | 2% | 0% | 0% | 17% |

**Forward clears chance in all three and is the finding; turning straddles it in all three and is
not.** Removing the warm start only changes *which* bias the yaw carries -- it is uniformly negative
(-0.019 to -0.021) under the forward warm start and uniformly **positive** (+0.010 to +0.035)
without one, where it is the robot's own drift. **Over all thirteen cross-embodiment runs, goal yaw
and achieved yaw correlate at -0.33 with 46% sign agreement** -- no relationship, at chance.

**So the warm start was concealing the turn failure rather than causing it.** What the planner
controls is *forward or not forward*: the two turn goals do slow the robot down (0.054 and 0.084
against 0.117 m/s for the forward goal), so the goal is read as "not straight ahead" -- and then
nothing selects which way to rotate.

**Two lessons for the measurement, both general.** *Family accuracy is a count of labels and cannot
see direction* -- a run that turns the wrong way scores identically to one that turns the right
way, so any condition with a sign (turning, strafing) needs sign agreement reported separately from
magnitude. And **a warm start is an intervention, not setup**: ten steps of commanded motion at
50 ms is half a second of the three-second episode, and the loop never escapes it. Every future
closed-loop comparison holds the warm-start clip fixed *and* neutral, or varies it as a control.

> **Third time in this project a defect surfaced from looking at the video rather than the table**
> -- after the standing-start jump and the camera that followed the robot. The tables were
> internally consistent in all three cases.


### F110. The warm start replays the goal's own actions, and what it was hiding is an entry transient

**The same-robot loops hand the planner the answer for ten steps.** `--demo` supplies both the goal
clip and the warm-start actions, so a same-robot run opens by executing the correct behaviour. The
scorer already excludes those steps (`lo = warm`, every run labels them `warm:`), **but it cannot
exclude their consequence**: at step 11 the body is already in the state the right behaviour
produces, and F109 showed the loop mostly continues whatever it was doing.

**Measured by removing it.** `--warm_start 0` on the held-out body, five repeats per goal:

| goal | family picks | speed error | behaviour class |
|---|---|---|---|
| `ep1` forward | 56% -> **75%** | 23.0% -> **12.9%** | 100% -> 100% |
| `ep1001` `turn_s0.05` | 46% -> 23% | 86.8% -> 154.4% | 100% -> 100% |
| `ep2301` `side_R_lvl1` | 91% -> 69% | 26.2% -> 58.1% | 100% -> 80% |
| **all 15** | | 36.2% -> 58.8% | **15/15 -> 14/15** |

**Forward improves when the hint is removed; everything else degrades.** So the line is not
turning-versus-the-rest, it is **forward versus every departure from walking straight**. F95's
15/15 is 14/15 without the warm start and its median speed error nearly doubles: the headline
survives, its margin does not.

**What the warm start was hiding is an entry transient, not an inability.** Splitting the planned
window into thirds on the two real turn goals:

| | yaw 1/3 | 2/3 | 3/3 | turn picks | switches |
|---|---|---|---|---|---|
| `ep1200`, warm start | -0.038 | -0.043 | -0.035 | 76% | 26 |
| `ep1200`, none, `--commit 1` | -0.003 | -0.008 | -0.024 | 34% | 42 |
| `ep1200`, none, `--commit 3` | +0.020 | -0.028 | **-0.031** | 47% | **18** |
| goal | | | **-0.038** | | |

**The planner enters the turn unaided, over roughly two thirds of the episode**, and committing for
three steps halves the switching and gets it there faster -- 63% of the commanded yaw at `commit 1`,
**82% at `commit 3`**. The whole-episode score barely moves (77.3% to 75.4%) because the median
still includes the entry. **Fifty-nine steps shows the transient and not the settled turn**;
whether it converges to the commanded rate needs longer episodes and is untested.

**`ep1300` is the exception and is not evidence of control.** Without a warm start it reaches
-0.026 while its last-third picks are 74 forward candidates against 10 turn ones, and those forward
candidates have *positive* yaw of their own (+0.003 to +0.009). **The rotation cannot be attributed
to what it chose** -- at 35 switches in 59 steps no candidate runs long enough to express its own
behaviour, which is F102's stitching cost.

**The closed-loop checkpoint has no shared body target at all, and that took two wrong answers to
establish.** `wm/data/embodiment.py` computes three body channels -- forward, lateral, yaw -- while
`BODY_CHANNELS = (0,)` and `wm/runs/beh12_hexonly/config.yaml` sets `body_dim: 1`,
`body_channels: ['0']`, which reads as "forward is the only channel taught to cross". **It is not,
because that run has only one robot in it**: `sources: hexapod=data/allocentric/beh12_c10f10t10_flat`. `lambda_body`
supervises forward *within the insect*, with no quadruped to share it with. Stage 3 then trains
`proj` and `ftm` only (`wm/adapt3.py`), leaving the motion decoder and its body head untouched, so
**the body head plays no part in the B1 path whatsoever.**

`screen_behaviour_channels.py --split condition` on that exact checkpoint, smoothed rows:

| channel | hex->b1 | b1->hex |
|---|---|---|
| forward | **-0.112** | -0.514 |
| lateral | -2.763 | -5.485 |
| yaw | -2.971 | -0.764 |
| vertical | -7.005 | -1.549 |

**Nothing transfers, forward included -- and forward crosses in the loop at 67 / 84 / 71%.** So
whatever carries forward travel across the two robots is V-JEPA2's own features plus the stage-3
adaptation, **not a shared body target**, and any sentence attributing the cross-embodiment result
to `lambda_body` is wrong.

**This does not contradict F83, it exposes a gap between them.** F83 asked whether `lambda_body`
creates a shared code and answered yes, on `stage2_*` checkpoints trained on both robots. Today's
probe asks whether *the checkpoint that closes the loop* has one, and answers no, because that
checkpoint is stage 1. Both are right about different models, and F83's +0.761/+0.641 must not be
quoted about the model that closes the loop.

**The gap: no checkpoint is both correct and cross-embodiment.** `stage2_*` trains on two robots but
on `ik_walk_*` + `fwd_b1_50hz`, from before F74's frame-rate fix; `beh12_hexonly` is on the corrected
data but has one robot. **So every closed-loop cross-embodiment number was produced without ever
using the mechanism this project measured as the thing that creates transfer** -- and forward still
crosses at 67-84%.

**That makes the next run obvious rather than speculative**: stage 2 on `beh12_c10f10t10_flat` +
`beh12_b1_flat`, `lambda_body 0.5` against an `0.0` control, then stage 3 and the loop on both. It
is the first experiment that would carry the body-head result all the way to a controller, and
widening `body_channels` becomes testable on top of it rather than instead of it. Heavy; it belongs
on fibo7.

**Fourth time a cheap proxy has failed to predict the loop**, after `z_crosses_bodies` ranked
turning first and forward last (F107). A linear readout of pooled `z` is not what the planner
computes -- it scores `FDM(e_t, proj(a))` against the goal, with the driven robot's own state inside
every comparison. **Screens of this kind are not evidence about the loop in either direction.**

**The initial pose is not the explanation, and it was the natural guess.** The robot holds
`cmds[0]` -- the goal clip's first pose -- for 20 warmup steps, and that pose is identical within a
condition and distinct between them, so it is a cue for all twelve. But its distance from the grand
mean runs **the wrong way**: forward is the *least* distinctive at 0.159 rad, against `turn_s0.56`
at 0.331 and `side_R` at 0.361. **If a distinctive starting pose helped, sideways would be the
behaviour that worked.** What is left is that forward translation is simply the largest thing in
the camera's view, which is the same reading F107 arrived at from the cross-embodiment side.

---

### F111. Across embodiments the loop transfers the kind of motion and not the amount of it

**The test forward travel needed, and it was available all along.** Froude is dimensionless by
construction -- 0.18 m/s on a 0.13 m insect and 0.30 m/s on a 0.56 m quadruped are the same number
(F56) -- so a cross-embodiment *speed* target is measurable even though the README had it as `n/a`.
Seven hexapod forward goals spanning Froude **0.129 to 0.222**, a 1.72x range, driving the B1 with
`--commit 3` and no warm start:

| hexapod goal | goal Froude | mean `vx` picked | B1 Froude |
|---|---|---|---|
| `ep1` `speed_c5.8` | 0.1289 | 0.373 | 0.1170 |
| `ep100` `speed_c7.1` | 0.1582 | 0.367 | 0.0718 |
| `ep101` `speed_c7.1` | 0.1609 | 0.361 | 0.0526 |
| `ep200` `speed_c8.15` | 0.2007 | 0.375 | 0.1490 |
| `ep201` `speed_c8.15` | 0.1996 | 0.362 | 0.1290 |
| `ep300` `speed_c8.8` | 0.2067 | 0.354 | 0.0692 |
| `ep301` `speed_c8.8` | 0.2216 | 0.373 | 0.0771 |

> **Read with F127.** The failure of forward *amount* to transfer, measured here, is real and
> stands. The implied contrast with turning does not: F127 shows the planner does not condition on
> its goal in either embodiment, so a behaviour ranking taken from these loops measures how
> classifiable each behaviour is, not what transfers. F126 read the same gap as a
> cross-embodiment-metric failure and **that reading is withdrawn.**

**corr(goal, achieved) = +0.074.** The B1 walks at an unrelated speed, and it is not the body
failing to deliver: **corr(goal, mean `vx` selected) = -0.167**, so the planner does not even choose
faster candidates for faster goals. Every goal draws the same mixture, mean `vx` between 0.354 and
0.375 against a 1.72x spread in what was asked for.

**The library is not the limit either.** The B1's forward candidates cover the goal range: `vx0.30`
at Froude 0.126, `vx0.38` at 0.160, `vx0.40` at 0.174, `vx0.50` at 0.206, against goals of
0.129-0.222.

**And the same loop does control speed when the goal is the same robot.** `hex_unseen_commit3`: 50% of runs
inside the 15% band, median error **14.8%**. The graded control is present within an embodiment and
absent across one, on the dimensionless quantity built to make the comparison fair.

| | family selection | speed tracking |
|---|---|---|
| same robot | 100% behaviour class | **50% within 15%**, median 14.8% |
| across embodiments | **66-80%** against 33% chance | **corr +0.074** -- none |

**So the defensible cross-embodiment claim is narrower than "a quadruped walks forward from an
insect's video" implies: the loop transfers the *kind* of motion, not the *amount*.** That answers
the obvious objection to a one-behaviour result -- "isn't forward just the largest thing in the
frame?" -- in the direction that concedes it. What crosses is a category, and category is what a
large translation in the image can carry.

**Reported against pre-registered criteria.** The three readings -- controlled, weakly tracked, not
tracked -- were written down before the runs finished, after a day in which several of this
project's own scripts printed verdicts that were threshold comparisons of overlapping means.

---

### F112. The pretrained latent does not cross embodiments at all -- the adaptation objective is what crosses it

**Asked because the claim was worded to invite it.** "The model never saw the quadruped during
pretraining" is true and reads as though a hexapod latent generalises to a four-legged robot. The
arms that test it were already on disk: the same loop, the same goals, `--warm_start 0`,
`--commit 3`, four checkpoints differing only in how much adaptation they received and under which
loss.

**Four clips per condition, not four reruns.** The B1 loop is MuJoCo and repeats bit for bit
(F105), so a rerun returns the identical number; the spread has to come from the recorded goal clip.

| adaptation | forward, `speed_c5.8` | turning, `turn_s0.56` |
|---|---|---|
| frozen world model, projector fitted only | **5% +/- 0** | 2% +/- 2 |
| ITM+FDM adapted separately, MSE | **28% +/- 5** | 12% +/- 2 |
| projector+FDM adapted **jointly**, MSE | **32% +/- 7** | 2% +/- 2 |
| projector+FDM adapted jointly, **+ InfoNCE** | **74% +/- 3** | 32% +/- 5 |
| *chance* | *33%* | *33%* |

Every run stayed upright for all 65 steps in every arm.

**The frozen arm is not weak, it is below chance.** A world model trained on the insect alone, with a
two-layer projector fitted to map the quadruped's 12-D actions into it, selects forward candidates
**5%** of the time against a 33% rate from guessing. **The pretrained latent does not transfer to
this robot in any usable sense**, and every sentence implying otherwise has to go.

**The two MSE arms are chance, and the joint one is the control that makes this an ablation.**
Adapting on 24 B1 clips under MSE -- what the source method's three stages do throughout -- buys
**28%** separately and **32%** jointly against 33%. **Fine-tuning jointly is not what does it**,
which was the obvious competing explanation and the first thing a reader would ask.

**The contrastive arm is the result.** The same file, the same 24 clips, the same architecture, the same
`adapt3.py` code path; **only `--lambda_nce` differs**, and forward selection goes
**32% +/- 7 to 74% +/- 3**. The same arm reaches 84% with a forward warm start (F109).

**Forward does not overlap**: the joint-MSE arm's best run is 38% and the contrastive arm's worst is
71%. **Turning clears chance in no arm at all** -- the contrastive arm's 32% +/- 5 sits on the 33%
line, and the earlier single-run 37% that read as a result is the top of that band. Both MSE arms
land *below* chance on turning, at 2% and 12%: they avoid turn candidates systematically rather than
failing to find them.

**Training budget runs the wrong way to explain it.** `stage3_b1_full` (MSE) ran **15,000** steps
and `stage3_b1_nce` **12,000** -- the losing arm got 25% more optimisation.

**This relocates the contribution rather than reducing it, and it lands where the project already
claimed one.** F98 measured MSE adaptation discarding the action channel -- `/mean-z` 0.993, family
19% -- and a contrastive term restoring it to 50%, **on recorded clips**. Here the identical
mechanism decides a physics loop:

| | measured on | MSE | + InfoNCE | chance |
|---|---|---|---|---|
| F98 | recorded B1 clips | 19% | 50% | 28% |
| F112 | **B1 walking in MuJoCo, goal from an insect** | 32% | **71%** | 33% |

**So the cross-embodiment result is a claim about the adaptation *objective*, isolated to the loss
term and not to the stage, the data, the architecture or the budget.** The defensible sentence: *a world model pretrained on a six-legged
insect cannot drive a quadruped at all; adapting it on 24 target clips under MSE leaves it at
chance; adding a contrastive term to that adaptation is what makes an insect's video steer the
quadruped forward.* Sideways fails at every rung, and **turning does not clear chance at any rung** once four
clips are measured -- the 37% that looked like a result is the top of a 28-37 band around a 33%
chance rate.

> **The frozen arm's 38% on sideways is not a result.** With forward at 5% the loop is defaulting onto
> `side_*` candidates, which is what a planner that cannot discriminate looks like when one family
> happens to sit closest. F96 named this failure mode; the way to see it is that all of
> 1's columns are within the spread of "always pick the same thing".

---

### F113. The quadruped walks out of its own camera frame, and the insect never does

**Found from a preview video, not from a table.** `results/dataset/preview_beh12/` shows the B1
partly outside the image in the sideways clips. Measured across all 96 clips -- robot mask taken as
the difference from each clip's own median background, since the robot moves and the backdrop does
not:

| | smallest visible area, as a fraction of that clip's own maximum | frames whose silhouette touches an image edge |
|---|---|---|
| **B1** `side_R_lvl1` | **42%** | 100% |
| **B1** `side_L_lvl0` / `lvl1` / `side_R_lvl0` | 87 / 85 / 90% | 100% |
| **B1** forward and turning | 69-76% | 36-47% |
| **hexapod**, all twelve conditions | 83-94% | **0%** |

**The insect never touches an edge in any of its 48 clips; the B1 does in every sideways frame and
in a third to a half of the rest.** And it is not confined to the recorded data -- inside the closed
loop the B1 drops to **43-66%** visible on cross-embodiment goals and **47%** on its own sideways
goal, while the hexapod loop stays at **89-93%**.

**This is a data defect on one robot only, and it lands on the behaviour that fails everywhere.**
Sideways is at or below chance in every arm of F112, on both robots in F104, and survived nine
refuted explanations (F106) -- all nine of which were on the deployment side.

**It is not established as the cause, and the reason is in the same table.** Forward clips clip
almost as badly (69-76% visible, 36-47% of frames touching an edge) and forward is the one behaviour
that works, at 74% +/- 3. **Framing is a real defect and a confound; calling it the explanation
would repeat the mistake F106 already made twice.**

**Fixing it is not a one-line change.** The camera would have to move back, which changes every B1
frame, which means re-recording `data/allocentric/beh12_b1_flat` and redoing the stage-3 adaptation that all of
F112 rests on. The insect side needs nothing. **Until that is done, every B1 number in this project
carries a framing asymmetry that the hexapod numbers do not.**

**The floor was checked and is not the problem, which took two measurements to establish.** A
bigger robot shows more ground, so the ground was the natural next suspect. Background sd on the
pixels the robot never covers reads **5.53 for the B1 against 3.74 for the insect** -- but splitting
that into a smoothed component and its residual shows where it lives:

| | total sd | smooth (lighting falloff) | high-frequency (floor texture) |
|---|---|---|---|
| B1 | 5.53 | **5.30** | 0.99 |
| hexapod | 3.74 | **3.37** | 1.02 |

**Both floors are effectively untextured, and identically so** -- 0.99 against 1.02. Essentially all
of the difference is a smooth lighting gradient, stronger in the B1's scene. **Three readings drawn
from the raw sd are therefore withdrawn**: that the encoder could key on floor pattern for position,
that the two scenes' floors let it separate the robots without looking at either, and that the
texture had to be fixed before the camera. A smooth gradient carries far less of any of that than a
pattern would.

**What survives is the framing and the apparent size.** Moving the camera back still shows more of
the lighting gradient, and flattening the lighting is still worth doing while the scene is open --
but it is a tidy-up, not the reason to re-render.

**Target for the fix, so it can be checked rather than eyeballed:** a B1
bounding box near **118 px** of 256 -- the insect's -- against its present 157, with **0%** of
frames touching an edge, which is what the insect already achieves in all 48 of its clips.

**Fixed, and verified against the target rather than by eye.** Both scenes ship an identical 15-deg
camera at the same height and distance, which was deliberate -- but **an identical camera is not an
identical view**. The field is 2.11 m wide at the robot; the insect is 0.97 m across and travels up
to 0.78 m, needing 1.75 m and fitting, while the B1 is 1.29 m across and travels up to 1.56 m,
needing 2.85 m. Widening the B1's perspective angle, one clip per condition re-rendered from the
states already stored in `beh12_b1_flat`:

| perspective angle | frames touching an edge | B1 bounding box |
|---|---|---|
| 15 deg (as shipped) | **62%** mean, 100% on all four sideways | 157 px |
| 21 deg | 18% sideways, 2% forward | 119 px |
| **25 deg** | **0% on all twelve conditions** | 94 px |
| *hexapod, unchanged* | *0%* | *118 px* |

**25 deg is the choice, and it trades apparent size for completeness.** The B1 goes from 33% larger
than the insect to 20% smaller. That is the better trade in one direction only: **clipping removes
information from the image, and a size difference does not** -- and the two robots genuinely differ
in size fourfold, so rendering them to look equal would be erasing the fact the experiment is about.

**`--cam_fov` and `--floor_scale` are on both `render_b1_replay.py` and
`close_loop_b1_physics.py`, defaulting to 24 and 3 on the latter.** They have to agree: a loop that plans on frames differing from its adaptation
set in any static way -- angle, spawn, floor -- is measuring that difference.

**The first enlarged floor buried the robot, and the metrics all passed anyway.** `sim.scaleObjects`
grows a box without moving its centre, so a 3x floor -- 0.2 m thick at z=-0.1 -- became 0.6 m thick
at the same centre and **lifted the walking surface from z=0.000 to z=+0.200**. The B1 stands at
z=0, so it was rendered 20 cm below ground with its feet cut off. Clipping, background sd, worst
background edge and between-clip spread all improved on that render; **not one of them can see a
robot sunk into the floor**, because the floor is static and the robot's visible silhouette is still
a robot. It was caught by looking at the picture, and the count of pixels above 200 -- the specular
dots on the feet -- is the number that separates the two: 24 in the original render, **0** with the
floor lifted, 9 with it put back.

`--floor_scale` now measures the surface before and after and translates it back, printing
`surface +0.000 -> +0.000` on every clip.

**Moving the camera instead of widening it was tried first, on the grounds that it keeps the
insect's exact lens, and it does not work.** The B1 sits at image y=0.35 where the insect sits at
0.49, so it is framed high and clips the top edge; **pulling back along the optical axis shrinks the
robot without moving it in frame**, leaving the sideways clips at 100% clipped even at 1.7x
distance. Shifting the camera to recentre makes the robot *larger* and brings the floor edge into
shot. Widening the angle is the only motion that adds room on the side the robot is leaving.

**Re-rendered and verified, `data/allocentric/beh12_b1_fov25`** (`scripts/dataset/rerender_b1_framing.py`,
`--cam_fov 24 --spawn 0 0 --floor_scale 3`). Checking the framing turned up a second defect that had nothing to do
with it: **the B1's camera was never pinned to a fixed world point**, so every clip carried its own
background, where the insect's is identical across all 48. `--spawn` exists for exactly that and was
not used when the set was built. Both are fixed in the same pass:

**And the first re-render was rejected on sight**, because widening the view reached the far edge of
the scene's 15 m floor and drew a straight band across the upper third of every frame. **Both of the
criteria written in advance passed on it** -- 0% clipping, backgrounds consistent to 0.22 -- because
the band is identical in every clip, so a between-clip measure cannot see it. Raising the lights
from 2.5 m to 6 and 12 changed nothing; only more floor did. `--floor_scale 3`:

| | before, 15 deg | `data/allocentric/beh12_b1_fixed` | hexapod |
|---|---|---|---|
| frames touching an edge, mean | 61% | **0.0%** | 0.0% |
| worst single clip | 100% | **0%** | 0% |
| background sd | 5.48 | **3.67** | 3.80 |
| worst background edge | 6.34 | **3.50** | 4.32 |
| background spread between clips | 2.79 | **0.23** | 0.14 |
| bounding box | 136 px | 99 px | 115 px |
| robot-to-floor contrast | 24.7 | 25.5 | 39.8 |

**The angle was chosen by sweeping, and the sweep had to run on all 48 clips rather than one per
condition.** At 23 degrees a representative clip from each of the twelve conditions gave 0%, and the
full set gave 1% -- two `side_L_lvl1` clips that travel further than their condition's
representative. **Within-condition spread is not visible in a one-per-condition check**, which is
exactly the shortcut that made the sweep quick. 24 degrees clears all 48.

| angle | clips still clipping | bbox |
|---|---|---|
| 15 (shipped) | 61% of frames | 136 px |
| 21 | 16% | 112 px |
| 23 | **2 of 48** | 103 px |
| **24** | **none** | 99 px |
| 25 | none | 92 px |

**The remaining gap to the insect is colour, not framing.** The B1's grey against a grey floor gives
a robot-to-background contrast of 25.5 where the insect's orange gives 39.8. That gap predates all
of this -- the original render was 24.7 -- and widening the view improved it slightly. **Changing it
means recolouring a robot, which would confound the framing fix with an appearance change in the
same re-fit**, so it is left alone and recorded.

**Every B1 column now sits at or inside the insect's**, where before it was worse on all four.
**`worst background edge` is in the table because it is the one that caught the band**, and it was
added only after the eye caught what the other two missed.

**The clip-to-clip background variation was noise, not a shortcut** -- two clips of the *same*
condition differed by 4.03 against 4.43 between conditions, so it never encoded the label. It is
still worth removing: it was a nuisance the encoder had to absorb on one robot and not the other.

**One number got worse and is reported rather than buried.** The B1's background against the
*insect's* moved 5.47 to **6.41** grey levels, because a 25-degree view takes in a different amount
of floor than the insect's 15. **While the two scenes use different angles that gap cannot close**;
it buys a robot that is never cut in half. The within-robot consistency, which is what every clip of
that robot shares, improved twelvefold.

**What is left is a re-fit, not a re-rollout.** MuJoCo never ran again -- every clip stores `base_pos`,
`base_quat` and `joint_pos`, so the physics was replayed from the file. **Stage 3 still has to be
refitted on `beh12_b1_fov25`, and until it is, every B1 number in this project stands on 15-deg
frames from unpinned cameras**, including all of F112.

> **Fourth defect this project found by looking rather than reading**, after the standing-start
> jump, the camera that followed the robot, and the loop turning the wrong way. The dataset's own
> `--separability` check passes on all of these clips, because it measures where the *body* went and
> never asks whether the camera saw it.

---

### F114. One of the four turn levels does not turn, and it is the forward clip under another name

**Spotted in a preview video.** `b1_turn_wz0.00` walks straight, because `wz = 0.00` is a commanded
yaw rate of zero. Measured, it is not merely weak -- it is **the same behaviour as the forward
clip**:

| condition | forward Froude | yaw |
|---|---|---|
| `speed_vx0.30` | **+0.1259** | **+0.0008** |
| `turn_wz0.00` | **+0.1259** | **+0.0008** |
| `turn_wz0.08` | +0.1288 | +0.0146 |
| `turn_wz0.19` | +0.1297 | +0.0359 |
| `turn_wz0.40` | +0.1295 | +0.0760 |

Identical to four decimals in both channels. **The twelve-condition set contains eleven behaviours.**

**The insect's weakest turn is milder but not a duplicate.** `turn_s0.05` reaches yaw -0.0072
against forward's +0.0026 to +0.0088 -- the same magnitude as straight walking's drift, though at
least the opposite sign. F107 already flagged that clip as barely turning; this is the quadruped's
version of the same problem, one step worse.

**It biases the two families in opposite directions, which is worse than a constant offset.**
Choosing `wz0.00` is *correct* behaviour for a forward goal and is scored as a miss; it is *wrong*
behaviour for a turn goal and is scored as a hit. Recomputing F112's ladder with `wz0.00` counted as
what it does rather than what it is called:

| | forward goal, as labelled | by behaviour | turn goal, as labelled | by behaviour |
|---|---|---|---|---|
| frozen | 5% | 5% | 2% | 2% |
| separate, MSE | 28% | **37%** | 12% | **5%** |
| joint, MSE | 32% | 32% | 2% | 2% |
| joint, **+ InfoNCE** | 74% | **83%** | 32% | **23%** |
| *chance* | *33%* | ***42%*** | *33%* | ***25%*** |

Chance moves too: forward holds five of twelve conditions and turning three.

**Both conclusions survive and both numbers move.** Forward goes to **83% against 42%**, still twice
chance; turning goes to **23% against 25%**, still not clearing it. **The contrastive result is not
an artefact of this defect** -- it is the arm that gains most on forward and loses most on turning,
which is what a planner that actually reads the goal should do.

**The `--separability` check cannot see it, and the reason is general.** It verifies that each level
exceeds the one below *on its own channel*, and 0.0146 > 0.0008 passes. **Nothing checks that a
family's weakest level is distinguishable from a different family.** Adding that is one comparison:
the lowest level of each family must differ from the forward clips by more than the forward clips
differ among themselves.

**The fix worth making is a new level, not a relabelling.** Renaming `wz0.00` to `speed` is free and
leaves turning with three levels against everything else's four. Collecting `wz0.60` instead
restores the balance **and extends the quadruped's turn range past the insect's**, which is
presently the other way around -- the B1 tops out at yaw 0.076 where the insect reaches 0.088, so
the two robots' strongest turns are not matched either. `rollout_b1_mujoco.py --wz` already does it,
and the re-render is happening anyway.

---

### F115. F75's sign flip was recorded as fixed and was not, and it decides every turning result

**`direction_plan.md` says "all four are fixed".** Measured again on 2026-08-28, on the data every
result in this project uses:

| | `beh12_b1_flat` | `beh12_c08f09t09_flat` |
|---|---|---|
| weakest commanded turn | **+0.0146** | -0.0241 |
| middle | **+0.0359** | -0.0372 |
| strongest | **+0.0760** | -0.0878 |

**The two robots still turn opposite ways.** F75 diagnosed this on 2026-08-22 and the fix was
written down rather than applied.

**This decides the turning result rather than influencing it.** A hexapod goal turning one way was
scored against a B1 candidate library that only turns the other, so **"turning does not cross
embodiments" was true by construction** -- F107's dose-response, F109's sign disagreement, F111,
F112's 23-32% at chance. None of them is evidence about the model. They are evidence that the
dataset asks for a left turn and offers only right ones.

**Forward and sideways are unaffected**: forward has no sign to disagree about, and the sideways
conditions carry their own direction labels, which F106 already checked and corrected.

**The replacement levels are calibrated, and the calibration is the part that was missing.** F72
paired the two robots on *commanded* turn rate and reported agreement within 3%; what they *achieve*
does not agree -- the insect's four levels reach 0.0072 / 0.0241 / 0.0372 / 0.0878 while the B1's
reach 0.0008 / 0.0146 / 0.0359 / 0.0760, so only the third pair is matched. Sweeping the B1 at
`--vx 0.30` gives an almost perfectly linear response, and the commands that land on the insect's
achieved values are:

| insect level | its achieved yaw | B1 command |
|---|---|---|
| `turn_s0.05` | -0.0072 | **`--wz -0.064`** |
| `turn_s0.15` | -0.0241 | **`--wz -0.153`** |
| `turn_s0.29` | -0.0372 | **`--wz -0.223`** |
| `turn_s0.56` | -0.0878 | **`--wz -0.491`** |

Forward Froude stays 0.120-0.129 across that range, so the speed match is not disturbed. **This also
fixes F114 in the same pass**: the weakest level becomes a real turn instead of the forward clip.

**What blocks the re-collection is not the physics.** `rollout_b1_mujoco.py` starts every run at
(0, 0) deterministically, so four runs of one command are identical -- yet the four clips of each
existing condition start at different points, which means they were cut as four windows from one
longer rollout. **The script that cut them is not in the repository**, so reproducing the set means
choosing the windowing again, and that choice decides how much the four clips of a condition
overlap. It should be made deliberately rather than inferred.

**The fix is a re-rollout of twelve clips, not a re-render.** `rollout_b1_mujoco.py --wz` takes a
signed rate, so negating the three turn levels regenerates them; and F114 wants `wz0.00` replaced by
a real fourth level in the same pass. **Nothing about turning can be claimed either way until that
is done.**

> **The lesson is about the document, not the data.** A finding recorded as fixed, with no
> measurement re-run afterwards, was carried for six days and shaped four later findings. Anything
> marked fixed should name the check that would fail if it regressed -- here, one line comparing the
> sign of the two robots' turn conditions.

---

### F116. Forward selection survives a forward model that ignores its action, and that is what the contrastive term is for

**The whole pipeline was refitted on `data/allocentric/beh12_b1_flat`** -- stage 1, stage 2, stage 3 -- because
the set the original was fitted on turns the opposite way from the insect (F115), files the forward
clip under `turn_wz0.00` (F114) and clips the robot in 61% of frames (F113). **The two stage-3 arms
now differ in `--lambda_nce` and in nothing else**: same 24 clips, 12,000 steps each, batch 8. The
original pair differed in batch and in budget as well, MSE having had 15,000 steps to the
contrastive arm's 12,000.

Ten cross-embodiment runs per arm, `--warm_start 0 --commit 3`, hexapod goals, four goal clips for
forward and turning and two for sideways:

| | forward | turning | sideways | upright | turn sign correct | yaw, last third |
|---|---|---|---|---|---|---|
| MSE | **53% +/- 5** | 22% +/- 3 | 19% +/- 1 | 10/10 | 4/4 | **-0.0475** |
| **+ InfoNCE** | **54% +/- 0** | **43% +/- 11** | 20% +/- 3 | 10/10 | 4/4 | -0.0353 |
| *chance* | *33%* | *33%* | *17%* | | *50%* | *-0.0878 commanded* |

**Forward is identical under the two objectives.** 53% against 54%, both about 1.6x chance.
**F112's 32% against 74% does not reproduce**, and the most likely reason is in F114: on the old
set `turn_wz0.00` *was* the forward clip, so an arm that chose it on a forward goal walked forward
correctly and was scored as a miss. The gap that made the contrastive term look decisive was partly
a label.

**What the term does buy is turn selection.** 43% against 22%, with MSE **below** its 33% chance
rate, and the same ordering offline -- family 52% against 23% on held-out clips. That is a real
effect and it is the one that survives.

**But the body disagrees with the label count, and it is worth saying which is which.** Both arms
turn the right way on 4 of 4 goals -- the first time this project has measured that at all, since
before F115 the two robots turned opposite ways -- and **MSE achieves *more* rotation**, -0.0475
against -0.0353 in the last third. The picks explain it:

| on turn goals | MSE | + InfoNCE |
|---|---|---|
| `speed_vx0.50` (yaw +0.0007) | **35%** | 22% |
| `turn_w0.075` (yaw -0.0750) | 15% | 9% |
| `turn_w0.037` (yaw -0.0371) | -- | **27%** |

**MSE alternates between the fastest straight clip and the hardest turn**; the contrastive arm picks
the turn level nearest the goal's rate. Median yaw rewards the first strategy and behaviour-family
accuracy rewards the second. **Neither reaches the commanded rate** -- 54% and 45% of it.

**Turning crosses embodiments, weakly, and this is the first time the question could be asked.**
Before the sign was fixed a hexapod goal turning one way was scored against a library that only
turned the other, so every earlier turning number measured the dataset.

**The `/mean-z` column resolves what looked like a contradiction, and it is the sharpest number
here.** F98 built that ratio to catch a forward model that ignores its action input: 1.0 means the
answer given the real action equals the answer given the mean one.

| | `/mean-z` | forward in the loop | turning in the loop |
|---|---|---|---|
| MSE | **0.977** | 53% | 22% |
| + InfoNCE | **0.493** | 54% | 43% |
| *chance* | | *33%* | *33%* |

**The MSE arm ignores the action channel and still selects forward at 53% against 33%.** That is the
result to take seriously: **forward "cross-embodiment control" does not require the forward model to
use the action at all** -- a 2% residual sensitivity is enough to rank the speed family above the
others. Forward is separable in the frame, and picking it is not evidence that the world model is
working.

**So the contrastive term does help, and only where help is needed.** What it changes is whether the
model uses the action -- 0.977 to 0.493 -- and that shows up exactly in the behaviour that cannot be
chosen without it: turning, where the collapsed arm sits **below** chance at 22% and the contrastive
arm clears it at 43%. On forward, where a nearly collapsed model already scores 1.6x chance, it buys
nothing.

**This also reframes F100.** Deleting the rollout there cost 30 points, but the arm measured was one
that used its action channel. A model that has already stopped using it loses far less, because the
selection it was making did not depend on the action in the first place.

**The old gap was the data, and that was tested rather than assumed.** Training MSE on the corrected
set with the *original run's* configuration -- batch 16, 15,000 steps, the two things that differed
from the new arms -- gives **58% forward and 36% turning**, against the original's 32% and 2% on the
old data. Same objective, same optimiser settings, same clip count; **only the data differs, and it
moves forward by 26 points and turning by 34.**

| MSE arm | batch | steps | data | forward | turning |
|---|---|---|---|---|---|
| the original | 16 | 15,000 | old | 32% | 2% |
| refit, original config | 16 | 15,000 | **v2** | **58%** | **36%** |
| refit, matched to the contrastive arm | 8 | 12,000 | v2 | 53% | 22% |

**And that exposes a flaw in the comparison above.** MSE at batch 16 and 15,000 steps reaches 36% on
turning where the batch-8, 12,000-step arm reaches 22%, so **the configuration is worth 14 points to
MSE and the "matched" ablation gave it the weaker one**. Against the stronger MSE the contrastive
arm's turning advantage is 43% against 36%, seven points, inside its own +/-11 spread. **The claim
that the contrastive term buys turn selection is not yet established** -- it rests on a comparison
that varied the objective and the budget together, which is the same fault this project criticised
in the original run. Both arms are being retrained at batch 8 and 15,000 steps to settle it.

**Sideways is at chance under both** -- 19% and 20% against 17% -- which is now the fourth
independent measurement of the same failure and the first on data with no known defect in it.

---

### F117. The two hexapod bodies turn opposite ways, and F94 left the cell that would have shown it blank

**Found while auditing what data each checkpoint came from.** The world model is pretrained on
`beh12_c10f10t10_flat` (`c10f10t10`) and every cross-embodiment goal comes from `beh12_c08f09t09_flat`.
Their turn ladders have the same magnitudes and opposite signs, on **every one of the 32 clips**:

| condition | `c10f10t10`, pretrained on | `c08f09t09`, the goal source |
|---|---|---|
| `turn_s0.05` | **+0.0029 +0.0047 +0.0043 +0.0006** | -0.0062 -0.0082 -0.0062 -0.0082 |
| `turn_s0.15` | +0.0138 +0.0135 +0.0152 +0.0140 | -0.0221 -0.0270 -0.0274 -0.0199 |
| `turn_s0.29` | +0.0357 +0.0370 +0.0358 +0.0368 | -0.0377 -0.0366 -0.0377 -0.0366 |
| `turn_s0.56` | +0.0762 +0.0759 +0.0790 +0.0788 | -0.0857 -0.0899 -0.0857 -0.0899 |

**F94 already had this in front of it.** Its comparison table prints the held-out body's yaw ladder
and leaves the training body's cell **empty**, so the reversal was never displayed. The same finding
caught the sideways reversal on that body and reported it as a headline; turning reversed in exactly
the same way and went unnoticed for a week, because one column was not filled in.

**Confirmed by looking at the walked paths, and the cross-checks that disagreed were my own error.**
Two quick tests written on the spot both gave nonsense -- an endpoint quaternion difference reading
the same sign for both bodies, and a net-rotation ladder that *decreased* with commanded spin. I
attributed that to the insect's body swaying every step. **That explanation was wrong.** Both
helpers assumed `(w, x, y, z)`, and `wm/data/embodiment.py` says in its own docstring that the
hexapod's `body_quat` is **`(x, y, z, w)` off an abdomen whose z axis points aft**, while the B1's
is MuJoCo's `(w, x, y, z)`. The canonical `yaw_rate()` handles both and was right throughout; a
second reading of it reproduces the table above to the fourth decimal.

**The failure was inventing a plausible cause instead of reading the code that documents the trap** --
the same shape as trusting F75's "fixed" without re-measuring. Use `yaw_rate()`; do not hand-roll a
quaternion helper for these two robots.

`results/wm/dataset/figures/turn_paths_three_sets.png` plots the head's path for all four turn
levels of each set, smoothed over one stride. **The two insect bodies curve in visibly opposite
directions and the B1 curves with the goal body**, which settles it and also validates
`yaw_rate()`: its signs match the paths, where the naive checks did not.

**It is most likely morphology rather than a defect.** Both bodies run the same CPG at the same
`--spin` levels, and the leg-ratio change that reverses the weak sideways gait (F94) plausibly
reverses the turning gait too. **But it means `turn_s0.56` does not denote the same behaviour across
bodies**, in a dataset whose entire purpose is matched behaviour.

**Three consequences, in order of how much they cost.**

1. **The model is pretrained on turns of one sign and asked about goals of the other.** Nothing in
   `beh12_c10f10t10_flat` ever showed the insect turning the way the goal clips turn.
2. **The B1 was matched to the goal body** (F115), which is right for the loop -- goal and candidate
   library agree -- but it makes the quadruped disagree with the pretraining insect.
3. **Any hexapod-to-hexapod turning comparison across these two bodies is sign-inconsistent**,
   including F95's held-out-body loop.

**What it does not do is invalidate F116.** There the goal source and the candidate library are both
negative, so selection is being asked a coherent question; the pretrained model's exposure to the
opposite sign is a handicap both arms carry equally.

**Closed in the collector rather than in a note.** `collect_beh12.py --separability` checked
`side_L` and `side_R` for sign but turning only for **size** -- `abs(w) < 0.5 * abs(turn_s0.56)` --
which is the same `|w_hat|` blindness F75 diagnosed between the two robots. It now fails a body
whose turn levels disagree with each other, and takes `--turn_sign` to fail a body whose turns
oppose the one the goals come from. Run over the three sets in use:

| | turn signs | |
|---|---|---|
| `beh12_c10f10t10_flat`, pretrained on | **+** | consistent |
| `beh12_c08f09t09_flat`, the goals | **-** | consistent |
| `beh12_b1_flat`, the candidates | **-** | consistent |

Each set is internally consistent, which is why nothing ever failed; the disagreement is only
visible across sets, and no check compared sets until now.

**Resolved by flipping everything onto the pretraining body, and the choice was economic.**
`c10f10t10` is the expensive artefact -- re-pretraining costs far more than re-collecting -- so the
held-out body and the B1 were re-collected to match it rather than the other way round.
`collect_beh12.py` gained `--spin_sign`, verified on one clip before committing to sixteen: the same
level at `--spin -0.56` gives **+0.0814** where `+0.56` gave -0.0878, the same size with the sign
reversed. All three sets now agree:

| | level 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `beh12_c10f10t10_flat`, pretraining | +0.0032 | +0.0141 | +0.0363 | +0.0775 |
| `beh12_c08f09t09_flat`, held out | +0.0069 | +0.0215 | +0.0407 | +0.0863 |
| `beh12_b1_flat`, the candidates | +0.0105 | +0.0268 | +0.0401 | +0.0807 |

**And the direction can be stated in the image, which is the frame that matters.** Taking the
principal axis of the robot's silhouette and following it through a clip, all three sets now rotate
**anticlockwise on screen** -- +140 deg for the pretraining insect, +152 for the held-out one, +54
for the B1. The two scenes ship identical cameras, so on-screen sense is a shared language; **being
unable to name the direction from the picture would concede the project's own claim that vision
carries shared meaning across bodies.**

> **The lesson is the blank cell, not the sign.** A comparison table with an empty column reads as
> "not applicable" and hides "not checked". Fill every cell or say why it is missing.

---

### F118. The corrected turn ladders reproduce independently, and the aggregation is part of the number

**Re-measured on 2026-08-29 from a fresh session, before spending GPU on the rebuild**, because
F117 records the fix and F115 records a fix that had not happened. All three sets turn the
same way and the four levels are matched:

| | level 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `beh12_c10f10t10_flat`, pretraining | +0.0032 | +0.0141 | +0.0363 | +0.0775 |
| `beh12_c08f09t09_flat`, held out | +0.0069 | +0.0215 | +0.0407 | +0.0863 |
| `beh12_b1_flat`, the candidates | +0.0105 | +0.0268 | +0.0401 | +0.0807 |

Identical to F117's table to four decimals, from a separate reading of the clips. **F117's fix is
confirmed, not merely recorded.**

**Reproducing it requires the project's aggregation, and the first attempt did not use it.** Taking
the **mean** of `yaw_rate()` over a clip's frames gives +0.0029 / +0.0148 / +0.0353 / +0.0736 on the
pretraining set -- same signs, same ordering, wrong in the third decimal, and 5% low at the top
level. The convention is **median over the frames of a clip, then mean over the clips of a
condition**, which is what `achieved()` and `_channels()` in `scripts/dataset/collect_beh12.py` do,
at a hard-coded `dt` of 0.05. The cause is the smoothing: `yaw_rate` convolves over a one-second
window in `same` mode, so the first and last half-windows are averaged against zero padding and pull
a mean down; a median ignores them. Trimming those frames instead overshoots to +0.0813.

**Made clean before the rebuild ran, not recorded as a caveat.** `wm/adapt.py` permuted the whole
48-clip directory, so at seed 0 its nine adaptation clips included `b1_ep100` and `b1_ep1300` --
two of the twelve clips `adapt3` uses as the **candidate library**, the set the planner picks from
-- and three of its twelve validation clips. It now takes `--train_clips`, both sheets source the
same 24-clip list from `scripts/b1_stage3_clips.sh`, and stage 1 draws its nine from those 24:
`b1_ep1002, 102, 1102, 1302, 2, 2201, 2302, 301, 302`, overlapping the candidates and the
validation clips in nothing. Whether the deleted `adapted_b1_v2.pt` had the same contamination is
**not in the logs** -- it recorded its `train_paths` inside the checkpoint, and the checkpoint was
deleted.

**Two things the fix does not do, recorded so neither is discovered later as a surprise.**

1. **The pool changed, so stage 1's own numbers are no longer comparable to F52 or F97.** It was
   nine clips drawn from 48 and is now nine drawn from 24. The rollout ratios `wm/adapt.py` prints
   are against a different held-out set; quote them only against runs under the new rule.
2. **The draw was not stratified, and now is.** Nine clips can cover at most nine of the twelve
   conditions, and a plain permutation covers fewer: the old pool gave **6/12** with `turn_w0.075`
   appearing three times and one sideways clip in total, the new pool gave **8/12** with
   `speed_vx0.50` twice -- better by accident, not by construction. `--stratify` walks the families
   in turn and takes an unused condition from each, giving **9/12 conditions, three per family**,
   and the rebuild sheet passes it. The clips are `b1_ep1002, 102, 1102, 1202, 2, 2001, 201, 2101,
   2201`. Condition and family are read off each clip, never parsed from its filename.

**And the selection is now checkable without a GPU.** It lived inside `main()`, so seeing which
clips stage 1 would take meant loading a 383 MB checkpoint and the encoder -- which is why nothing
caught the contamination for the length of the original run. It is `wm.adapt.select_clips(paths,
clips, test_clips, seed, stratify)`, callable on its own.

**A third leak, found in the stage 2 log on 2026-08-29 and closed the same evening.**
`fit_projector` fitted on **all 48 B1 clips**, the twelve candidates and twelve validation clips
included. It is milder than the stage 1 one -- the projector is only stage 3's starting point and
stage 3 retrains it for 15,000 steps at `lr_proj 1e-3` under a different objective, where stage 1's
forward model is what stage 3 scores and is only nudged at `lr_ftm 1e-5` -- but **how much of the
memorised `(a, z)` survives 15k steps was never measured**, so "mild" was an argument rather than a
number. Both sheets now pass `--exclude $HOLDOUT`, the 24 non-training clips, listed in full with
their `.npz` suffix: `fit_projector --exclude` matches by prefix, and `b1_ep100` is a prefix of
`b1_ep1000` through `b1_ep1003`.

The rebuild was 19 minutes into the six stage-3 seeds when this was found and was restarted rather
than annotated. **None of the three leaks could change the MSE-against-contrastive ordering** -- all
six runs share one stage 1 checkpoint and one projector -- but each would have sat as a caveat
beside every absolute number the run produces.

**So a turn-ladder number is only comparable to another one aggregated the same way.** Quote the
table above, or say which reduction produced the number.

---

### F119. Three seeds settle it: the contrastive term is what makes the quadruped selectable, and MSE is below chance

**The measurement F116 asked for, on the corrected data and with the three leaks of F118 closed.**
Six stage-3 runs, three seeds per arm, everything else identical -- same stage 1 checkpoint, same
stage 2 projector, same 24 clips, 15,000 steps, batch 8. Averaged over the last 21 evaluations
(step 10,000 onward), a window fixed before the runs and applied to both arms:

| arm | family | cond | `/mean-z` | `/hold` |
|---|---|---|---|---|
| **contrastive** (`--lambda_nce 1`) | **54.8% +/- 1.1** | 28.6% | **0.490** | 0.891 |
| **MSE** (`--lambda_nce 0`) | **21.6% +/- 0.3** | 7.8% | **0.985** | **0.802** |
| *chance* | 28% | 8% | -- | -- |

Per seed, the two arms do not come within 30 points of each other: contrastive 53.7 / 55.0 / 55.9,
MSE 21.3 / 21.7 / 21.8. **F116's ambiguity is gone** -- there, MSE at one budget reached 36%
against contrastive's 43%, inside that arm's own spread, and the two had been compared at different
budgets.

**MSE is not merely worse at selecting, it is below chance.** 21.6% against 28%, and `cond` at 7.8%
against 8% is the rate of guessing. `/mean-z` 0.985 says why: the forward model returns the same
answer for the real action as for the mean one, so there is nothing for a planner to rank and the
sub-chance score is what ranking noise looks like.

**And the arm that selects is the arm that predicts worse.** MSE ends with the lower training loss
(0.69 against 1.12) and the better held-out prediction (`/hold` 0.802 against 0.891). **The two
capabilities dissociate cleanly, and the objective decides which one you get** -- which is the
claim, stated at three seeds rather than one.

**Reproduces F98 on data that no longer carries the four B1 defects**, and slightly better on both
arms: F98 read `/mean-z` 0.62, cond 29%, family 50% for contrastive and 0.993 / 6% / 19% for MSE.

**What this is not.** It is offline discrimination over recorded clips, not a closed loop: the
model ranks twelve candidate actions against a known next embedding. Whether the ordering survives
into physics is the next measurement, and F100's warning applies -- forward selection has cleared
chance before under a model whose `/mean-z` was 0.977.

`scripts/diagnostics/summarise_stage3_seeds.py` produces the table; `adapt3` stores only the final
step in its checkpoint, and the final step is not the run -- `family` wanders about four points
between evaluations, so nce seed 0 ends on 57% against a 53.7% mean. Log kept at
`results/wm/stage3_b1_seeds_2026-08-29.txt`.

---


### F120. The objective's 33-point offline lead is 6 points in physics, and neither arm turns

**The closed loop, run on the checkpoints of F119.** Twelve episodes: two stage-3 arms against six
goal clips -- two per behaviour family, both sideways signs -- with MuJoCo carrying the weight and
CoppeliaSim supplying the camera. Goals are the `...3` validation clips, in no candidate library and
in no training set. Nothing differs between the arms but `--ckpt`.

**Selection, measured over the planner's own 330 picks per arm** rather than over what it achieved:

| | offline (F119) | in the loop |
|---|---|---|
| contrastive | 54.8% | **41%** |
| MSE | 21.6% | **35%** |
| chance | 28% | 28% |

**The gap collapses from 33 points to 6**, and MSE moves from *below* chance to *above* it. Whatever
the offline discrimination measures, most of it does not survive being asked the same question on
frames the loop drove itself into.

**What the robot actually did**, median over the planned steps, dimensionless:

| goal | target | contrastive | MSE |
|---|---|---|---|
| forward, `speed_vx0.30` | 0.122 | **0.096** | 0.077 |
| forward, `speed_vx0.50` | 0.197 | 0.049 | 0.058 |
| turn, `turn_w0.037` | yaw 0.052 | 0.026 | 0.006 |
| turn, `turn_w0.075` | yaw 0.063 | 0.006 | 0.010 |
| sideways left | lateral +0.173 | +0.009 | +0.025 |
| sideways right | lateral -0.178 | **+0.012** | -0.017 |

**Survival is 12 of 12 and the speed criterion is 0 of 6 in both arms.** The signs are mostly right
and the magnitudes are 5-20% of target: the quadruped **barely turns and barely strafes**, rather
than turning the wrong way. The one reversed run is the contrastive arm on sideways right.

**So the defensible sentence about the objective is offline-only.** F119's contrast is real and
three-seeded; F120 says it does not carry into control on this robot, which is what F100 warned
about from the other direction -- selection scores and loop behaviour have come apart before.

> **`S.R. behaviour` cannot be quoted from these runs.** `score_closed_loop.py` decides the class by
> the largest channel, and forward speed exceeds yaw in every turn condition on both robots (F108),
> so a turning goal and a walking outcome are both "forward" and the check passes for free. It
> reads 67% against 50% here and neither number means what it appears to. The speed column was
> fixed to use `channel_for`; `ok_class` was not. **Fix it, and re-report every `S.R. behaviour`
> that has been quoted.**

Runs in `results/wm/closed_loop/b1_{nce,mse}_s0_b1_ep*/`.

---


### F121. With the turn signs finally agreeing, cross-embodiment control is still at chance and the quadruped does not turn

**The first cross-embodiment loop run on data where all three sets turn the same way** (F118).
Goal frames come from the held-out insect `c08f09t09`, candidates stay B1 clips because only those
are executable, and `--demo` is held fixed at one forward B1 validation clip so that only the goal
varies (F109, F110). Twelve episodes, two stage-3 arms, six goals.

**Selection, over 330 picks per arm:**

| | contrastive | MSE | chance |
|---|---|---|---|
| same-robot goals (F120) | 41% | 35% | 28% |
| **insect goals** | **32%** | **33%** | 28% |

**Both arms are at chance and the two are indistinguishable.** The objective that separates by 33
points offline and by 6 points on same-robot goals separates by **-1** here.

**What the quadruped did**, median over the planned steps:

| goal | target | contrastive | MSE |
|---|---|---|---|
| `speed_c5.8` | 0.131 | 0.070 | **0.101** |
| `speed_c8.8` | 0.222 | 0.069 | **0.105** |
| `turn_s0.29` | yaw 0.041 | 0.004 | -0.005 |
| `turn_s0.56` | yaw 0.088 | -0.002 | 0.009 |
| `side_L_lvl1` | lateral +0.160 | +0.013 | +0.002 |
| `side_R_lvl1` | lateral -0.140 | +0.016 | -0.004 |

**The robot walks forward whatever it is shown.** Yaw is 2-10% of target and twice carries the
wrong sign; lateral is 1-11% of target. Survival is 12 of 12 and the speed criterion is 0 of 6 in
both arms. On forward the **MSE arm is the better of the two**, at 23% and 53% error against 47%
and 69%.

**So the sign correction was necessary and is not sufficient.** Before F117 the turning question was
unaskable -- an insect turning one way against B1 candidates turning the other. It is now askable
and the answer is that nothing crosses: what the loop does with an insect goal is what it does with
no useful goal at all.

**Two things this does not settle.** The fixed forward demonstration starts the robot walking
forward and supplies ten warm-start commands, so "walks forward regardless" is the null the design
makes easiest to fall into; a neutral or standing demonstration is the control that has not been
run. And this is one seed per arm -- F119's three seeds cover the offline claim only.

Runs in `results/wm/closed_loop/b1_hexgoal_{nce,mse}_s0_*/`.

**`score_closed_loop.py` could not grade these until today.** It read the reference from `--demo`,
which in a cross-embodiment run is the neutral B1 clip rather than the goal, so every such run was
being scored against a forward walk it was never asked for. It now prefers the recorded `goal` when
that differs from the demonstration, takes `--goal_dir`, and picks the scored channel from the
*reference's* condition. Same-robot numbers are unchanged -- `b1_nce_s0_b1_ep3` reads 20.9% before
and after.

---


### F122. Committing three steps buys the turn back and it is still not the goal's turn

**Thirty-six cross-embodiment episodes**, two stage-3 arms against six insect goals at three loop
settings: `commit 1 / warm 10` (F121), `commit 3 / warm 10`, and `commit 3 / warm 0`. Survival is
36 of 36.

**Committing changes execution, exactly as F102 and F103 predict.** The quadruped's achieved yaw
range widens from -0.005..+0.021 at commit 1 to +0.004..+0.046 at commit 3, and the contrastive
arm's `turn_s0.29` lands within **0.9%** of its goal -- the first `S` on any cross-embodiment run.

**It is not a turning result, and the control is inside the same batch.** A *forward* goal produces
as much yaw as a turning one:

| setting, contrastive arm | turn goals | forward goals |
|---|---|---|
| commit 3, warm 10 | +0.041, +0.046 | +0.023, **+0.037** |
| commit 3, warm 0 | +0.060, +0.035 | **+0.054**, +0.009 |

The 0.9% match is a robot yawing 0.02-0.05 whatever it is shown, meeting a goal that happens to ask
for 0.041.

**Goal yaw against achieved yaw, six cells of six runs**: r = -0.30, +0.76, +0.33 for contrastive
across the three settings and -0.18, -0.07, -0.66 for MSE. **The sign of the effect flips between
settings within one arm**, which at n = 6 is what no relationship looks like. F109 measured -0.33
and 46% sign agreement on the old data; that survives the correction.

**Selection stays at chance under every setting:**

| | contrastive | MSE | chance |
|---|---|---|---|
| commit 1, warm 10 | 32% | 33% | 28% |
| commit 3, warm 10 | 34% | 35% | 28% |
| commit 3, warm 0 | 36% | 29% | 28% |

**So the sign defect was real, was fixed, and was not what stood between this pipeline and a
cross-embodiment turn.** Before F117 the question was unaskable; asked properly, on data where all
three sets turn the same way, the answer is that **forward travel crosses and turning does not** --
F109's sentence, now without the confound that could have explained it away.

**And the warm start is no longer load-bearing either.** At `warm 0` the robot starts standing and
still survives 65 of 65 in all twelve runs, so the ten steps were never holding the loop up. What
they were doing is what F109 said: setting the yaw the robot then keeps.

Runs in `results/wm/closed_loop/b1_hexgoal_{nce,mse}_s0_c{1,3w10,3w0}_*/`. The `--commit` and
`--warm_start` settings are in the directory names because two settings cannot be compared without
naming them.

---


### F123. The cross-embodiment goal metric was the fault, and a mean shift recovers it

**The planner scores a raw MSE between a predicted B1 embedding and the goal embedding**, and in a
cross-embodiment run that goal is a *hexapod* frame. Embodiment is strongly decodable from these
embeddings (F43, F46), so the distance is dominated by which robot is in the picture. No correction
was applied: the checkpoints carry no `embedding_offsets` and `center_embeddings` is false.

**Tested offline, no simulator, one checkpoint (`stage3_b1_nce_s0`), 8 demonstrations, 95
decisions.** `plan_open_loop.py` gained `--goal_dir` / `--goal_embodiment` so the goal can come from
another robot, and `--center`, which translates the goal clip into the driven robot's mean
appearance -- **only the goal moves**, since `e_t` is also the forward model's input and shifting it
would trade one confound for another.

| goal | exact condition | right behaviour |
|---|---|---|
| the B1's own clips | 19.4% | 47.2% |
| **insect, raw MSE** | **4.2%** | **34.7%** |
| **insect, goal mean-shifted** | **23.2%** | **55.8%** |
| *chance* | 8.3% | 25.0% |

> **The reading below is withdrawn by F125.** The control it names as unrun was run the same
> evening: with the goal swapped for a *different behaviour*, the corrected score is **55.8%
> again, unchanged to the decimal**, and the score against the clip the planner was actually
> shown is 22.1% against a 25% chance rate. **The planner is not reading the goal.** The
> measurements in this entry are correct; the sentence "the mean shift recovers cross-embodiment
> selection" is not.

**Uncorrected, exact-condition selection is *below* chance and the behaviour family is barely above
it. Corrected, both clear the same-robot baseline.** The mean shift is one subtraction: no
retraining, no architecture change, nothing learned.

**This locates the failure of F120-F122.** Every cross-embodiment loop in this project scored its
candidates against an insect goal in raw embedding space, so the planner was ranking on a distance
whose leading term is "this is the wrong robot" -- a term identical for every candidate in its
constant part and dominated by appearance in the part that varies. The loop's 32-36% is what that
metric supports.

**What it does not establish.** This is open loop: frames come from recorded clips, not from states
the robot drove itself into, and the closed loop has to survive both. It is one checkpoint and one
seed. And a mean shift is a first-order correction only -- if the two embodiments also differ in
variance structure, per-dimension standardisation or a learned alignment may be needed, and the
recovery above being *complete* rather than partial is the evidence that a mean is enough here.

**Next**: apply the shift in `wm/policy/planner.py` and rerun the twelve cross-embodiment episodes,
which is the same test in physics. Also worth re-reading F107's retrieval proxy, which asked a
related question and was wrong in the direction that cancelled the experiment.

---


### F124. The goal shift that fixes selection offline does not fix it in the loop

**F123's correction, applied in physics.** `close_loop_b1_physics.py` gained `--center_goal`, which
translates the insect goal clip into the driven robot's mean appearance -- the B1 reference taken
from its own demonstration clip, since live frames cannot supply it at step 0, and only the goal
moves. Twelve episodes, both arms, otherwise identical to F122's `commit 3, warm 0` cell.

| | offline (F123) | in the loop |
|---|---|---|
| raw MSE, behaviour | 34.7% (chance 25%) | 36% / 29% (chance 28%) |
| goal shifted, behaviour | **55.8%** | **36% / 34%** |

**Offline the shift is worth 21 points; in the loop it is worth nothing.** Survival stays 12 of 12,
the speed criterion stays 0 of 6, and achieved forward speed still covers a fraction of the goal
range (0.072-0.093 against goals of 0.017-0.218 for the contrastive arm -- *narrower* than without
the shift).

**It does move which behaviour is confused for which**, which is why the overall figure hides it:

| | sideways | forward | turning |
|---|---|---|---|
| contrastive, raw | 16% | **52%** | 40% |
| contrastive, shifted | 18% | **32%** | **58%** |
| MSE, raw | 16% | 43% | 29% |
| MSE, shifted | **32%** | 33% | 36% |

Turning rises 40% to 58% and forward falls 52% to 32% in the contrastive arm. The shift redistributes
selection across families rather than improving it: chance is 33% per family, so the shifted
contrastive arm is at chance on forward and above it on turning, having been the reverse.

**Yaw still does not track the goal.** r = -0.14 (contrastive) and +0.13 (MSE) with 83% sign
agreement in both -- and the sign agreement is uninformative because every goal in the corrected data
turns positive and the robot's own drift is positive (F109).

**So hypothesis (a) is confirmed as a real defect and refuted as the explanation.** The metric was
wrong, correcting it recovers offline selection completely, and the loop does not benefit -- which
leaves (b), the five-step rollout on frames the robot drove itself into, as the remaining candidate.
That is the next test: `does_rollout_matter.py` and `loop_frames_are_off_manifold.py` on these
checkpoints, and a one-step loop against the five-step one.

**A control that has not been run and should be, before any of this is written up.** The shift uses
the B1's own demonstration clip as the appearance reference, so the goal the planner sees is
"insect movement content, B1 appearance". Whether the offline gain came from the insect's content or
merely from the goal landing on the B1's manifold is answered by shuffling which insect behaviour is
paired with which demonstration: if the score holds up under a mismatched goal, it was never reading
the content.

Runs in `results/wm/closed_loop/b1_hexgoal_{nce,mse}_s0_c3w0ctr_*/`.

**Cosmetic defect in the new code**: the line printing the shift norm computes it after the shift is
applied, so it always reports 0.00. The shift itself is correct; the diagnostic print is not.

---


### F125. The planner does not read the goal at all, and every offline selection number in this project shares the confound

**The control F124 called for, run immediately.** `plan_open_loop.py` gained `--mismatch`, which
pairs each B1 demonstration with an insect goal from a **different behaviour family** and reports
two numbers: the picks scored against the demonstration, and against the clip the planner was
actually shown. With a matched goal these are the same number by construction, which is why nothing
before now could tell them apart.

Contrastive checkpoint, 8 demonstrations, 95 decisions, goal mean-shifted as in F123:

| goal shown | scored against the demonstration | scored against the goal |
|---|---|---|
| matched (F123) | 55.8% | 55.8% *(identical by construction)* |
| **a different behaviour** | **55.8%** | **22.1%** |
| *chance* | 25% | 25% |

**Showing the planner a goal from a different behaviour family does not move its picks by a single
point.** It scores 55.8% against the demonstration either way, and below chance against what it was
shown. **The 55.8% is a readout of `e_t` -- the quadruped's own current frame -- not goal-following.**

**So what the mean shift actually did.** It did not make insect content readable. It made the goal
term nearly constant across candidates, and the ranking fell back on the part driven by `e_t`, which
correlates with the demonstration's condition at 55.8%. That is why the correction was worth 21
points offline (F123) and nothing in the loop (F124): in the loop, `e_t` is a frame the robot drove
itself into and its condition is whatever the robot is currently doing, so a readout of it is not a
controller.

**The confound is not confined to this experiment.** Any offline selection score measured with a
goal drawn from the demonstration's own future -- which is every one this project has reported,
including the same-robot 47.2% baseline above and F100's 62% -- cannot separate "picked the
candidate that reaches the goal" from "named the behaviour already visible in the current frame".
**The mismatch control is cheap and should be attached to every such number before any of them is
quoted again.**

**What this does and does not say about the world model.** It does not say the forward model is
useless: F119's discrimination gives it the true next embedding and it ranks at 54.8% against 28%,
which is a different question and still stands. It says the *planning objective as implemented* --
minimise the distance from a rolled-out prediction to a goal embedding -- is not being solved by the
goal in a cross-embodiment setting, in either the raw or the corrected metric.

**Where this leaves the hypothesis table.** (a) is a real defect that explains the offline gap and
nothing else; the correction's apparent success was the confound. (b), the five-step rollout, is
still untested and is now joined by a sharper question: **does the goal term influence the argmin at
all, under any metric?** `does_rollout_matter.py`'s `blind` arm answers a version of that and should
be run first, before `loop_frames_are_off_manifold.py` -- if the goal does not move the choice, the
manifold question is downstream of a more basic failure.

**The full control matrix, and the raw metric is the best of the three.** Two further cells were
run: the raw metric under a mismatched goal, and a dataset-level mean -- averaged over 12 clips of
each robot rather than over the single goal clip -- on the hypothesis that a clip's mean carries its
behaviour and subtracting it removes both.

| metric | scored against the demonstration | **scored against the goal shown** |
|---|---|---|
| raw MSE | 33.7% | **38.9%** |
| goal shifted by the clip mean | 55.8% | 22.1% |
| goal shifted by a dataset mean | 49.5% | 25.3% |
| *chance* | 25% | 25% |

**The raw metric does follow the goal, weakly**: 38.9% against a 25% chance rate, and above its own
demonstration-tracking. **Every centring tried destroys that** and replaces it with an `e_t` readout,
the dataset mean no better than the clip mean -- so the "behaviour lives in the clip mean" reasoning
is refuted too. Adding the driven robot's appearance to the goal, however it is estimated, makes the
ranking answer "which prediction looks most like a typical B1", which correlates with the current
state and not with the goal.

**So the honest number for cross-embodiment goal-following is 38.9% against 25%, uncorrected**, and
hypothesis (a) closes with the metric already at its best. `--center` is kept in the script because
the negative result is worth being able to reproduce, and defaults to off.

Logs: `/tmp/ol_mismatch.log`, `/tmp/ol_mismatch_raw.log`, `/tmp/ol_dsmean_mismatch.log`; the pairing
is a deterministic family rotation at matched level.

**Two silent pairing failures were hit building this control**, both from the two robots naming the
same thing differently. Matching demonstration to goal by condition string kept only the sideways
clips, since `speed_vx0.30` and `speed_c5.8` share no prefix; and a rotation written over
`side_L`/`side_R` matched almost nothing, because the recorded `behaviour` field holds three values
-- `speed`, `turn`, `side`. The first reported 0.0% from 12 decisions and the second reported a
clean-looking 66.7% from 2 demonstrations of 8. **Both failed by skipping, not by raising.** The
script now refuses to start if any demonstration is left unpaired.

---


### F126. The forward model works, the rollout earns its cost, and the whole failure is the cross-embodiment goal

**Hypothesis (b) tested and closed.** `does_rollout_matter.py` on the contrastive stage-3
checkpoint, 48 clips, 12 candidates, scored on 2,340 transitions from clips no candidate came from,
**same-robot goals**:

| horizon | rollout | direct | blind | per family (rollout) |
|---|---|---|---|---|
| 1 | **61%** | 31% | 34% | side_L 100%, side_R 100%, speed 30%, turn 53% |
| 3 | **71%** | 32% | 38% | side 98/100%, speed 46%, turn 67% |
| 5 | **73%** | 35% | 36% | side 96/100%, speed 39%, **turn 82%** |
| 10 | **72%** | 34% | 35% | side 86/100%, speed 31%, **turn 89%** |
| *chance* | 28% | 28% | 28% | |

> **The goal-following reading of this table is withdrawn by F127.** The mismatch control was run
> the same night: with the goal frame taken from a **different behaviour of the same robot**, the
> rollout still scores 56-70% against the *demonstration* and only **18-23% against the goal it was
> shown**, below the 28% chance rate. The rollout is doing real work and it is not the work of
> reaching a goal. Read every number below as "names the behaviour the robot is already in", not as
> selection.

**Rolling the forward model is worth 27-37 points over ignoring the goal, and deleting it costs
almost all of that** -- `direct` sits within a few points of `blind` at every horizon. The world
model is a predictor here, not a similarity function, and the sixty forward-model calls per control
step buy exactly what they are supposed to. **The horizon that plans best is 5, and 10 is no worse**,
which is the opposite of the fragility that was assumed.

**And turning is the behaviour the rollout helps most** -- 53% at one step rising to 89% at ten,
the only family that improves monotonically with horizon. Sideways, which fails everywhere else in
this project, is at 86-100%. Forward is the *weakest* family here at 30-46%.

> **This inverts a claim the project has repeated, and the inversion is the point.** F109 and F111
> report forward as the behaviour that transfers and turning as the one that does not, and the
> "forward crosses, turning does not" framing was built on them. Both are **cross-embodiment**
> measurements, and F126 localises the cross-embodiment goal comparison as the one broken component
> -- so that ranking was a property of the broken metric, not of the behaviours. With the metric
> sound, turning is the strongest family and improves with horizon, which is mechanically sensible:
> a turn is a sustained low-frequency signal a longer rollout accumulates. **Correction notes have
> been added to F109 and F111 so the two cannot be read as jointly true.**

> **And the localisation below is withdrawn too.** F127 measures the planner failing to condition
> on its goal *within one robot*, where no cross-embodiment metric is involved, so "everything works
> except the cross-embodiment comparison" is false. F125's metric defect is real and sits on top of
> a planner that was not using its target in either setting.

**Put beside F125 this locates the failure precisely.** Same checkpoint, same candidates, same
projector, same rollout:

| goal comes from | selection | baseline |
|---|---|---|
| **the B1 itself** | **73%** | 36% (blind) |
| **the insect** | **38.9%** | 25% (chance) |

**Everything in the pipeline works except the comparison between a predicted quadruped embedding and
an insect goal embedding.** Not the objective (F119), not the projector, not the rollout, not the
horizon. And F125 measured that the obvious repair -- removing each robot's appearance offset -- makes
it worse under every estimator tried.

**What follows, and it is a design change rather than a fix.** A raw embedding distance was never a
shared coordinate; it was hoped to become one because both robots pass through one frozen encoder.
The project already owns a quantity that *is* shared by construction: the **body-motion head** that
`lambda_body` supervises, in dimensionless units both robots are measured in (F58, F65, F66).
Scoring a candidate by the body motion its rollout predicts, against the body motion the goal clip
shows, asks for the same comparison in a space where the two robots are commensurable -- rather than
hoping appearance cancels. That is the next experiment.

**Caveat on the numbers above**: same-robot goals here come from clips the candidates did not come
from, but the goal is still the demonstration's own future, so F125's confound applies to this table
too -- part of the 73% may be a readout of `e_t` rather than goal-following. The mismatch control
has not been run in this script. **It should be, before 73% is quoted anywhere.**

---


### F127. Even within one robot the planner does not follow its goal; the rollout is a state classifier

**The control F126 called for, run before its 73% was quoted anywhere.** `does_rollout_matter.py`
gained `--mismatch`, which takes each demonstration's goal frame from a clip of a **different
behaviour family of the same robot** and adds a column scoring the rollout's pick against the family
the goal actually came from.

| horizon | rollout vs demonstration | direct | blind | **rollout vs the goal shown** |
|---|---|---|---|---|
| 1 | 56% *(61% matched)* | 36% | 34% | **18%** |
| 3 | 64% *(71%)* | 26% | 38% | **18%** |
| 5 | 68% *(73%)* | 30% | 36% | **23%** |
| 10 | 70% *(72%)* | 24% | 35% | **21%** |
| *chance* | 28% | 28% | 28% | 28% |

**Showing the planner a goal from a different behaviour costs it 3-7 points of agreement with the
demonstration and leaves it below chance on the goal.** The 73% was never selection. It is the
same confound as F123, now measured **within one embodiment**, where no appearance mismatch and no
cross-robot metric can be blamed.

**This is the deepest correction of the session and it reframes the whole diagnosis.** F126 read the
gap between same-robot 73% and cross-embodiment 38.9% as "everything works except the
cross-embodiment comparison". It is not: **the goal drives the argmin in neither setting.** The
cross-embodiment metric is a real defect (F125) sitting on top of a planner that was not using its
goal to begin with.

**What the rollout is actually doing, and it is not nothing.** It beats `blind` by 22-35 points
under mismatch, so the forward model contributes real information -- but the information is *which
behaviour this robot is currently in and how it continues*, not *which action reaches a target*.
`direct` sits at or below `blind` throughout, so that information arrives specifically through
rolling the dynamics. **A good predictor, wired as a state classifier.**

**The turning result survives this and is worth keeping.** Turning still rises 39% to 88% with
horizon under mismatch, and sideways stays at 85-100%: the ordering F126 reports is a property of
how well the rollout identifies a behaviour from its dynamics, which is a real and useful capability
-- it is just not planning. F109 and F111's "forward crosses, turning does not" remains corrected.

**Consequence for what to do next.** The criterion for any proposed fix is now **the `roll/goal`
column, not the score against the demonstration**, which is passable without reading the goal at
all. That applies to the body-motion-head scoring proposed in F126: **predicted body motion against
the goal clip's body motion must clear 28% on a mismatched goal**, or it has changed the coordinate
without changing what the planner uses.

**And it puts a number on the teacher-student argument (Q16).** Candidate scoring here is not
failing to pick the best candidate -- it is not conditioning on the request. Emitting a policy makes
the request part of training rather than something a run-time argmin has to honour.

Log: `/tmp/rollout_mismatch.log`.

---


### F128. Scoring in the shared coordinate restores goal-conditioning, and the shared coordinate is too narrow to use

**The falsification F127 demanded, run with its control in the same pass so no number could stand
alone.** `scripts/diagnostics/score_by_body_motion.py` scores a candidate by the body motion it
produces rather than by embedding distance:

    score(a) = | body_head(proj(a)) - forward speed of the goal clip |

The goal enters as a dimensionless physical quantity, so nothing has to cancel and no embedding
distance is involved. Matched and mismatched goals are computed in the same run.

**Same-robot goals, and this is the first rule in the project that conditions on its goal:**

| horizon | matched | mismatched vs demonstration | **mismatched vs the goal shown** |
|---|---|---|---|
| 1 | 49% | 33% | **49%** |
| 3 | 47% | 31% | **49%** |
| 5 | 45% | 28% | **49%** |
| 10 | 47% | 25% | **51%** |
| *chance* | 28% | 28% | 28% |

Against the embedding rule under the identical control -- 18-23% against the goal, 56-70% against
the demonstration (F127) -- **the ordering inverts.** The picks follow the target and stop following
the current frame. **The coordinate was the thing standing between this planner and goal-conditioned
selection**, and it is not the world model: this rule uses no rollout, no `e_t`, and no forward
model at all.

**Cross-embodiment, the same rule does not separate**, and the reason is measurable rather than
mysterious:

| horizon | matched | mismatched vs demonstration | mismatched vs goal |
|---|---|---|---|
| 1 | 48% | 48% | 44% |
| 5 | 48% | 47% | 47% |
| 10 | 50% | 49% | 49% |

**The two columns agreeing is the signature of a near-constant pick**, not of goal-following.

**Why, and it is the head rather than the data.** The two datasets are calibrated to each other --
B1 `speed_vx0.30` walks at 0.126 against the insect's `speed_c5.8` at 0.129, `speed_vx0.50` at 0.206
against 0.215, and every condition pairs within a few percent. What fails is the head's reading of
the candidates:

| candidate | true forward speed | `body_head(proj(a))` |
|---|---|---|
| `side_L_lvl0` | **-0.008** | **0.079** |
| `side_R_lvl1` | 0.013 | 0.119 |
| `speed_vx0.30` | 0.132 | 0.129 |
| `speed_vx0.50` | **0.215** | **0.126** |
| `turn_w0.075` | 0.127 | 0.102 |

**True spread 0.223, predicted spread 0.059 -- a 3.8x compression, correlation +0.60** -- and the
four speed levels are read as identical to three decimals. The head can say "sideways or not" and
nothing finer, so the argmin saturates: any target above the achievable band selects the same
candidate, which is why matched and mismatched agree.

**So the same-robot 49% is two-way discrimination on a compressed signal**, not speed tracking, and
the cross-embodiment 44-49% is one candidate chosen almost regardless of the request. Both numbers
are honest and neither is a planner.

**What this changes about the pivot argument.** F127 said the loop's argmin was never conditioned on
the command, which read as an argument that run-time search cannot be made to honour a goal. That is
too strong: **it can, and this is the demonstration** -- put the score in a coordinate the two bodies
share and goal-conditioning appears immediately. The limit that remains is the *width* of the shared
coordinate: `body_dim 1`, forward speed alone, compressed 3.8x. Turning and sideways are not in the
shared space at all, which is exactly the channel-competition problem F83 measured and did not solve.

> **The "too narrow" diagnosis is corrected by F129.** The head is not compressed and the coordinate
> is not narrow: on the hexapod, the body it was trained on, it reads forward speed at correlation
> **+0.99 and compression 1.0x**, matching every family to three decimals. The 3.8x compression
> measured here is what that head does on a **B1**, which it has never seen -- `beh12_hexonly` is a
> hexapod-only pretrain and neither `wm/adapt.py` nor `wm/adapt3.py` touches the motion decoder. The
> tables above are correct; the conclusion drawn from them is not.

**The concrete next experiment is therefore a pretraining change, not a planner change**: widen the
body head to yaw and lateral (`body_channels 0,1,2`) and check its calibration against measured
speed before scoring anything with it. F83 says the channels compete; F128 says a one-channel head
is not enough to plan with. Those are the same question and it is now the blocking one.

Log: the tables above are reproducible with `--goal_dir data/allocentric/beh12_c08f09t09_flat`.

---


### F129. The shared head is exact on the robot it trained on and constant on the one it never saw

**Measured before spending ten hours widening a head that might not have been the problem.** The
body head is a two-layer MLP on `z`; it can be evaluated on either robot with the weights already on
disk. Fed the ITM's latent from real frames:

| robot | correlation | true spread | predicted spread | compression |
|---|---|---|---|---|
| **hexapod**, the body it trained on | **+0.99** | 0.194 | 0.194 | **1.0x** |
| **B1**, never seen by this head | +0.20 | 0.197 | 0.061 | 3.2x |

Per behaviour family, dimensionless forward speed, true against predicted:

| | hexapod | B1 |
|---|---|---|
| sideways | 0.022 -> **0.018** | 0.003 -> 0.106 |
| forward | 0.160 -> **0.161** | 0.159 -> 0.112 |
| turning | 0.129 -> **0.128** | 0.122 -> 0.115 |

**On the hexapod the shared coordinate is exact.** It separates a standing-still sideways gait from
a fast walk from a turn, to three decimals, from the latent alone. **On the B1 it returns the
dataset mean for everything** -- 0.106, 0.112, 0.115 against a `body_stats` mean of 0.109 -- which is
the correct behaviour of a model asked outside its domain, not a defect in the head.

**The cause is structural and is written in the code.** `beh12_hexonly` is pretrained on
`sources hexapod=...` alone, and both adaptation stages say so explicitly: `wm/adapt.py` --
"the motion decoder is not adapted at all"; `wm/adapt3.py` fine-tunes projector and forward model.
**The shared head has never seen a B1 latent at any point in the three stages.** F128 scored
candidates with it anyway and read the resulting flatness as the coordinate being too narrow.

**So the instruction to widen the head to `body_channels 0,1,2` is right and would have been
uninterpretable as issued.** Widening it on a hexapod-only pretrain produces a three-channel head
that has still never seen the target robot; a negative result would say nothing about channel
competition. **The pretrain has to contain both embodiments**, which is the condition F83 measured
channel competition under -- and F83's run is forward-walking only and carries the frame-rate defect
(F74), so it has to be redone on `beh12_*` regardless.

**What this does not rescue.** F128's same-robot result stands unchanged: scoring in the shared
coordinate is the only rule in this project that has ever conditioned on its goal (49-51% against
the goal, 25-33% against the demonstration). That was measured on the B1 with a head that reads the
B1 as a constant, which makes it a **lower bound** -- the rule conditioned on its target while its
coordinate was returning noise.

**Adaptation is a second, independent way the shared coordinate is lost, found while validating the
tool.** The same head, the same B1 clips, two checkpoints:

| ITM supplying the latent | correlation | compression |
|---|---|---|
| `beh12_hexonly/best.pt`, unadapted | **+0.76** | 2.2x |
| `stage3_b1_nce_s0.pt`, after stages 1 and 3 | **+0.23** | 3.2x |

**Adapting the forward model to the B1 makes the shared head read the B1 worse**, and `wm/adapt.py`
states the mechanism in its own docstring -- "stage 2 ... against the *adapted* ITM, since stage 1
moved what `z` means". The head was fitted against the pretrained latent space; stage 1 moves that
space and nothing moves the head with it. **So the pass bar can be met on the pretrain and lost by
the checkpoint the planner actually uses**, and both have to be measured. The run sheet now runs
`body_head_calibration.py` before stage 1 and again after stage 3 for exactly this reason.

**The tool agrees with the hand measurement**, which is what licenses the pass bar: on the same
checkpoint it reads B1 at +0.23 / 3.2x against the manual +0.20 / 3.2x, and the hexapod at
+0.99 / 1.0x either way. The 0.03 is 240 samples against 288. **The earlier disagreement was two
different checkpoints being compared, not two different readings.**

**The run this calls for**, and it is the one to hand to com7:

    --sources hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat \
        --lambda_body 0.5 --body_dim 3 --body_channels 0 1 2

then stages 1-3 on top, then per-channel calibration on **both** robots before anything is scored
with it. **The pass bar, set in advance**: all three channels within ~1.5x compression on both
robots, and `roll/goal` above 28% on mismatched cross-embodiment goals. If forward calibrates and
yaw does not, that is F83's channel competition reproduced on corrected data and the bottleneck is
the pretraining objective rather than head width -- **stop there and report it, rather than tuning.**

---


### F130. One head does serve both robots; it had simply never been fitted on the second one

**F129 said the shared head reads the B1 as a constant because nothing ever adapts it. This fits it
and asks what happens.** `wm/fit_body_head.py` freezes everything and trains the body head alone --
about 8k parameters, `z_dim -> body_hidden -> body_dim` -- on latents the checkpoint's own ITM reads
off real consecutive frames, so the question is what is *in* `z` rather than what the projector can
reach. Split by clip, 20% of the B1 held out, evaluated on held-out clips only.

**The premise that kept the decoder out of adaptation has expired.** `wm/adapt.py` excludes it
because it "plays no part at control time, so adapting it would cost time for a module the planner
never calls" -- correct until F128 put `body_head(proj(a))` on the control path.

| head | B1 held-out clips | hexapod |
|---|---|---|
| **as shipped**, never fitted on a B1 | +0.23, compression 3.2x | **+0.99, 1.0x** |
| fitted on the **B1 alone** | **+0.81, 1.5x** | +0.74, 0.5x |
| fitted on **both robots at once** | **+0.81, 1.7x** | **+0.97, 1.0x** |

Held-out MSE against predicting the target's mean, on the B1: **0.967 before, 0.514 after** -- the
latent carried the signal all along and nothing was reading it.

**Outcome (1) of the three named in advance: the shared coordinate is real and the head was simply
unadapted.** Fitted on both, one head holds the hexapod at +0.97 and the B1 at +0.81 with no
trade -- per family, hexapod sideways 0.020 -> 0.017, forward 0.160 -> 0.155, turning 0.131 -> 0.130.

**But fitting on the B1 alone reproduces outcome (2), and that is worth keeping.** It costs the
hexapod its correlation (0.99 -> 0.74) and **flips the sign of its sideways prediction** -- true
+0.020 read as -0.067 -- because the two robots' sideways gaits genuinely differ: the B1 strafes at
about zero forward speed (-0.008) while the insect still travels +0.020 while strafing. A head shown
only the quadruped learns "sideways means no forward motion" and carries that onto a body where it
is false. **The shared coordinate holds only if the fit sees both robots**, which is exactly what
the two-embodiment pretrain in `scripts/com7_pretrain_body3.sh` provides.

**What this changes about the plan.** The 3-channel pretrain is still needed -- yaw and lateral are
absent from the shared coordinate entirely, and one channel cannot separate turning from strafing
(F128). But **a pretraining run is no longer the only route to a calibrated head**: fitting it after
the fact takes minutes and recovers most of the gap, so the pretrain is buying the extra channels
rather than the calibration.

**Two things this does not yet show.** The B1's compression is 1.7x against a 1.5x bar, so the head
tracks the ordering better than the range. And nothing here has been scored: whether a calibrated
head raises `roll/goal` on mismatched cross-embodiment goals is the next measurement and the one
that matters.

> **Scored, and it made selection worse -- see F131 for why and for the fix.** Fitting against the
> ITM's latent lifts the ITM path from +0.20 to +0.79 and leaves the *projector* path, which is the
> one the planner feeds it, at +0.44 with its range 2.5x too wide. Same-robot goal-following fell
> from 49-51% to 19-34%. **A head has to be fitted on the latent it will actually be shown.**

Checkpoints: `stage3_b1_nce_s0_bodyfit.pt` (B1 only), `stage3_b1_nce_s0_bodyfit_both.pt` (both).
Logs `/tmp/bodyfit*.log`, `/tmp/bodycal_{fit,both}.log`.

**A defect in the first version of the fit script, caught by the calibration tool refusing to load
its output**: it built the motion decoder with a placeholder action width and then saved the whole
decoder, overwriting the checkpoint's real 12-D B1 output head with a 1-D one. It now updates the
`body_head.*` keys inside the checkpoint's own decoder state and leaves every other head untouched.
**A saved checkpoint that no longer loads is the lucky version of this mistake**; the same edit that
silently kept a wrong head would not have announced itself.

---


### F131. Fitted on the latent it is actually shown, the shared coordinate follows the goal -- and it crosses embodiments

**F130 fitted the body head against the ITM's latent and selection got worse.** The reason is the
one `wm/fit_projector` already documents for its own target: fit against what will be consumed.
`body_head(proj(a))` is what the planner evaluates, and `a -> z` is one-to-many (F97), so the
projector's latents occupy a different region from the ITM's.

| head, on B1 | ITM latent | projector latent |
|---|---|---|
| as shipped | +0.20, 3.2x | +0.44, 2.8x |
| fitted on ITM latents | **+0.79**, 1.5x | +0.44, **0.4x** *(range 2.5x too wide)* |

`wm/fit_body_head --latent projector` fits the same 8k parameters against the latents the planner
supplies. Held-out MSE over predicting the mean: **0.854 before, 0.132 after.**

**Scored with the mismatch control in the same pass, so no number stands without it:**

| | horizon 1 | 3 | 5 | 10 | chance |
|---|---|---|---|---|---|
| **same robot**, vs the goal shown | **76%** | **79%** | **82%** | **86%** | 28% |
| same robot, vs the demonstration | 18% | 16% | 14% | **7%** | 28% |
| **insect goals**, vs the goal shown | **36%** | **37%** | **35%** | **38%** | 28% |
| insect goals, vs the demonstration | 26% | 26% | 26% | 25% | 28% |

**Same-robot selection follows the goal at 86% while tracking the demonstration at 7%, below
chance.** No reading of that is compatible with the frame-reading confound that killed F123, F126
and F128's cross-embodiment column: the rule now ignores what the robot is doing and answers what it
was asked for.

**And insect goals clear chance at every horizon** -- 35-38% against 28% -- with demonstration
tracking at exactly chance. **This is the first cross-embodiment selection number in this project
that survives its own control.** It is modest, and modest is what one channel buys.

**What is actually shared here, stated precisely.** Not the latent: the head is fitted per robot on
that robot's own projector latents. What is shared is the **target quantity** -- dimensionless body
motion, measurable from outside with no kinematic model, and identical in meaning on a 6-legged
insect and a 12-DOF quadruped. The planner compares a prediction in that quantity against a goal
expressed in it. **That is the claim the project has been trying to make, and it is the first form
of it that a control has not destroyed.**

**Limits, all of them.** One channel -- forward speed only, so 36% is what remains after turning and
sideways are invisible to the score. One checkpoint, one seed, offline, no simulator. The head is
fitted on the target robot's actions, which is the same cost the action projector already pays and
is honest about (F52's few-shot budget), but it is not zero. **And the 3-channel pretrain is now
clearly worth running**: the coordinate works and it is one-dimensional.

Checkpoint `stage3_b1_nce_s0_bodyfit_proj.pt`; logs `/tmp/bodyfit_proj.log`,
`/tmp/score_proj_{same,cross}.log`.

---


### F132. Frames and rollout only: the quadruped's selection follows an insect's video above chance

**F131's result was real and was not the claim.** Its scoring rule reads
`body_head(proj(a))` against the goal clip's *measured* forward speed -- **no frame is consumed at
scoring time, the forward model is never called, and the target comes off a recorded trajectory
rather than out of a video.** It is action-to-speed regression matched against a number. Three
conditions were run to separate what each component contributes, cross-embodiment goals,
mismatch control in every one:

| | what the candidate side uses | where the target comes from |
|---|---|---|
| **A** | `body_head(proj(a))`, no frames, no rollout | the goal clip's measured trajectory |
| **B** | roll the FDM h steps from `e_t`, read the transition with the ITM | the goal clip's measured trajectory |
| **C** | the same rollout | **the goal robot's own frames**, `body_head(ITM(g_t, g_t+h))` |

**Selection against the goal actually shown, mismatched pairing, chance 28%:**

| horizon | A | B | **C** |
|---|---|---|---|
| 1 | 36% | 34% | 28% |
| 3 | 37% | 32% | **35%** |
| 5 | 35% | **45%** | **39%** |
| 10 | 38% | 39% | **42%** |
| *tracking the demonstration instead, C* | -- | -- | 25-27% |

**C is the condition the project's claim needs and it clears chance from three steps on**, rising
with horizon to 42% while its demonstration-tracking stays at chance. In C **nothing but pixels and
the world model is involved**: the candidate's future is predicted by rolling the forward model, the
insect's request is read out of the insect's frames by the same shared head, and no recorded
trajectory value enters either side.

**And the world model earns its place.** B and C both roll the FDM, and both improve with horizon --
B peaks at 45% at five steps, C at 42% at ten -- where A, which never calls it, is flat at 35-38%
across every horizon. **A rule that ignores the rollout cannot get better with more of it**, and the
horizon dependence is the signature that the prediction is contributing.

> **This paragraph is withdrawn by F135.** The comparison it makes is between conditions that differ
> in *two* things -- A is also handed a measured trajectory value where C has to read its target out
> of frames -- so the horizon signature cannot be assigned to the rollout. **Mode D**, which takes
> C's vision-only target and deletes the rollout, reaches **41% at horizon 3** against C's best of
> 42%. The two overlap. **The rollout's contribution is not established**, and the argument above
> that "a rule that ignores the rollout cannot get better with more of it" is wrong: D's target
> window widens with the horizon too, so D varies with it as well. Everything else in this entry
> stands.

**What C's failure at one step says.** A single-step transition of a hexapod carries very little
body motion to read, so the target is mostly noise there; three steps is enough. That is a property
of the readout window, not of the coordinate.

**Stated at its true strength.** This is offline selection among twelve recorded B1 behaviours,
one checkpoint, one seed, forward speed only, on held-out clips. It is **not** a closed loop and not
a controller. What it does establish, with the control that killed every previous version of this
claim: **a quadruped, shown a stick insect's video and nothing else, picks its own behaviour to
match what the insect is doing, above chance, using a world model to predict what each of its
options would do.**

Logs `/tmp/f132_{A,B,C}.log`; `scripts/diagnostics/score_by_body_motion.py --mode {A,B,C}`.

---


### F133. The contrastive objective is what makes the rollout usable, and mode A never needed it

**A replication attempt on an independent stage-3 checkpoint turned into the control that explains
F132.** The MSE-arm checkpoint differs from the contrastive one in a single flag; both were fitted
the same way and scored the same way, cross-embodiment goals, mismatch control throughout.

| | mode A -- no rollout, no frames | mode C -- rollout and frames |
|---|---|---|
| **contrastive** | 36 / 37 / 35 / 38% | 28 / **35 / 39 / 42%** |
| **MSE** | 36 / 38 / 36 / 37% | 28 / **24 / 22 / 26%** |
| *chance* | 28% | 28% |

**Mode A is identical under both objectives**, which is what it should be: it never calls the
forward model, so a model that ignores its action input is no handicap. `body_head(proj(a))` against
a measured speed is action-to-speed regression and the objective cannot touch it.

**Mode C works only for the contrastive arm.** The MSE arm sits at or below chance at every horizon
and gets *worse* with more rollout. F119 measured why on the same checkpoints: MSE adaptation leaves
`/mean-z` at 0.985 -- the forward model returns the same answer for the real action as for the mean
one -- so rolling it accumulates nothing about the candidate.

**This is the link between the two halves of the project, and it was missing until now.** F119 is a
statement about a training objective; F132 is a statement about cross-embodiment selection. **They
are the same statement**: the contrastive term is what makes the forward model's rollout carry
action-specific information, and mode C is the only rule that consumes that information. Remove the
term and the rule collapses while its rollout-free sibling is untouched.

**What it does and does not do for F132's robustness.** It is not the seed replicate that is still
outstanding -- `stage3_b1_nce_s{1,2}` are on com7 and have not been transferred. **But a checkpoint
that differs in one flag and fails exactly where that flag predicts is stronger evidence than a
second seed of the same configuration would have been**, because it rules out "any stage-3
checkpoint scores 42% under mode C".

Logs `/tmp/f133_{A,C}_mse.log`, `/tmp/f132_{A,C}.log`.

---


### F134. Three shared channels calibrate on both robots at once, and F83's channel competition does not reproduce

**The com7 run F129 asked for**, `beh12_hex-b1_body3`: pretrained on **both** embodiments with
`body_channels 0 1 2` -- forward, lateral and yaw, dimensionless, observed from outside. Five hours,
then stages 1-3, with `body_head_calibration.py` run before adaptation and again after.

**Straight off the pretrain, every channel on every robot:**

| | forward | lateral | yaw |
|---|---|---|---|
| **hexapod** | **+0.99, 1.0x** | **+0.98, 1.2x** | **+0.98, 1.0x** |
| **B1** | **+0.99, 1.0x** | **+0.97, 1.2x** | **+0.97, 1.0x** |

**Every cell clears the 1.5x bar set in advance.** Per family the head reproduces the physics:
hexapod `side_L` lateral 0.116 -> 0.111 against `side_R` -0.151 -> -0.119, both signs correct;
B1 turning yaw 0.038 -> 0.038; forward speed to three decimals on both bodies.

**F83 does not reproduce, and that finding should be treated as superseded.** It measured adding yaw
costing forward 68% and buying yaw at +0.37 +/- 0.27 -- the result that made channel competition the
expected outcome and was written into this run's pass bar as the likely failure. **On corrected data
with both embodiments in the pretrain, there is no trade**: forward is +0.99 with yaw present, the
same as it was alone (F129). What F83 measured was a two-embodiment pretrain on forward-walking-only
data carrying the frame-rate defect (F74), and one channel of variation to learn from.

**Adaptation degrades it, exactly as F129 predicted, on a pretrain F129 never saw:**

| after stages 1 and 3 | forward | lateral | yaw |
|---|---|---|---|
| hexapod | +0.88, 1.5x | +0.81, **2.7x** | +0.83, 1.1x |
| B1 | +0.82, **1.7x** | +0.60, **3.8x** | +0.80, 1.4x |

Correlations fall from +0.97-0.99 to +0.60-0.88 and three of six cells fail the bar. **This is the
second, independent confirmation that stage 1 moves the latent and nothing moves the head with it**
-- and F131 already has the repair: refit the head on the latents the planner supplies, which took
minutes and lifted held-out MSE from 0.854 to 0.132 on the one-channel model.

**So the run bought what it was meant to buy.** The shared coordinate is three-dimensional and
calibrated on both bodies; turning and strafing are now *in* it, where F128 could only separate
"sideways or not". **What has not been measured is whether they help**: the 42% bar (F132) is
forward speed alone, and the three-channel version has to beat it under mode C with the mismatch
control. That needs the checkpoints, which are still on com7.

Log: `paste_from_com7.txt`, run `beh12_hex-b1_body3`, commit ba12c71.

---


### F135. Deleting the rollout costs nothing: the result is the coordinate, not the world model

**The control that prices the world model, and the reason it was needed.** F132 compared mode A
(action-to-speed regression against a measured trajectory value, no frames, no rollout) with mode C
(rollout on the candidate side, target read from the goal robot's frames) and read the difference as
the world model contributing. **Those two conditions differ in two things at once.** Mode D fixes
that: C's vision-only target, A's rollout-free candidate score.

    A   body_head(proj(a))            vs  measured speed of the goal clip
    C   body_head(ITM(e_t, rolled))   vs  body_head(ITM(g_t, g_t+h))
    D   body_head(proj(a))            vs  body_head(ITM(g_t, g_t+h))     <- C's target, no rollout

Cross-embodiment goals, mismatch control, selection against the goal actually shown, chance 28%:

| horizon | A | C | **D** |
|---|---|---|---|
| 1 | 36% | 28% | **33%** |
| 3 | 37% | 35% | **41%** |
| 5 | 35% | 39% | **39%** |
| 10 | 38% | 42% | **33%** |

**D matches C.** Its best is 41% against C's 42%, they are equal at horizon 5, and D is *better* at
one and three steps. **Rolling the forward model buys nothing measurable here**, and F132's
horizon-signature argument does not survive: D's target window widens with the horizon as well, so
varying with horizon is not evidence about the rollout.

**What still stands, and it is the substantive half.** Cross-embodiment selection clears chance with
**vision on both sides and no recorded trajectory value anywhere** -- 33-41% against 28%, with
demonstration-tracking at chance. The insect's request is read out of the insect's frames by the
shared body head, the quadruped's options are scored in the same physical coordinate, and the
mismatch control that killed F123, F126 and F128 does not kill this. **The result is the shared
coordinate. It is not the world model.**

**Which sharpens what the project can claim and what it cannot.** "A quadruped selects its behaviour
from an insect's video" is measured. "A world model plans the selection" is not: the best rule
measured so far reads a target out of the goal video, maps each candidate action to the same
quantity, and takes the nearest -- no prediction of the future required. **F127 said the planner was
not conditioning on its goal; the fix turned out to be the coordinate, and the coordinate does not
need a rollout to work.**

**Two things that follow.** The forward model remains a strong dynamics predictor (F126: rollout
beats blind by 22-35 points at classifying behaviour) -- it is simply not what makes this selection
work, which is the same dissociation F133 found from the other direction. And mode D should be run
on the MSE arm: if D works there too, the objective's role is confined to the rollout-based rules
and F133's link between F119 and selection narrows accordingly.

**Mode D on the MSE arm is at chance too** -- 33 / 24 / 26 / 28% -- which is not what the F133 story
predicts on its own. **A and D share the entire candidate side**, `body_head(proj(a))`, and differ
only in where the target comes from; A works on the MSE arm and D does not. The ITM is identical
across the two arms (stage 3 trains the projector and forward model, not the inverse model), so what
differs is the **head**, refitted per arm on that arm's projector latents. *Hypothesis, untested*:
the contrastive stage 3 leaves the projector's latents in the region the ITM occupies, so a head
fitted on one reads the other, and MSE's does not.

**The per-family split of the one-channel mode D, which is the baseline the 3-channel run has to
beat** (per-family chance: speed 33%, turn 33%, side_L 17%, side_R 17%):

| horizon | pooled | side_L | side_R | speed | turn |
|---|---|---|---|---|---|
| 1 | 33% | 13% | 31% | 23% | **62%** |
| 3 | **41%** | 20% | 39% | 52% | **62%** |
| 5 | 39% | 25% | 38% | 45% | **54%** |
| 10 | 33% | 17% | 33% | 18% | **57%** |

**Turning is the best-identified family on a forward-speed-only coordinate**, at 54-62% against 33%,
because turn clips occupy a narrow band of forward speed (0.119-0.141) that separates them from fast
walks and from strafing. `side_L` sits at chance throughout: it is the family one channel cannot see.

**A defect worth recording**: the first mode-D run reported 0% on every column with a healthy sample
count, because the goal-encoding block still tested `mode == "C"` and every sample hit a `continue`
past the counters. **An empty loop that prints a table looks exactly like a measurement.** It now
raises instead of skipping.

Log `/tmp/f135_D.log`.

---


### F136. Three shared channels: cross-embodiment selection reaches 70%, and strafing goes from invisible to near-perfect

**The 3-channel pretrain (F134), carried through the repair F129 and F131 established.** Stage 1
and 3 degrade the head -- forward +0.82/1.7x, lateral +0.60/**3.8x**, yaw +0.80/1.4x on the B1 --
so the head was refitted on the latents it is actually shown: `proj(a)` and the ITM's, on the B1,
plus the hexapod's ITM latents, everything else frozen, 20% of the B1 held out.

**Calibration after the refit, on held-out clips** (F134's pretrain numbers are untouched and remain
the record of what the pretrain itself produced):

| | forward | lateral | yaw |
|---|---|---|---|
| **B1**, held out | **+0.95, 1.1x** | **+0.83, 1.3x** | +0.91, 1.6x |
| **hexapod** | **+0.97, 1.1x** | **+0.96, 1.2x** | **+0.95, 1.1x** |

Five of six cells clear the 1.5x bar; the B1's yaw misses by 0.1.

**Selection, cross-embodiment, mismatch control, per behaviour family. Chance is 28% pooled and,
per family, 33% for speed and turn and 17% for each sideways direction.**

**Mode D** -- target read from the insect's frames, no rollout:

| horizon | pooled | side_L | side_R | speed | turn |
|---|---|---|---|---|---|
| 1 | **70%** | 86% | 79% | 50% | 54% |
| 3 | 68% | 92% | 65% | 62% | 45% |
| 5 | **70%** | 87% | 69% | 41% | 64% |
| 10 | **70%** | **100%** | 84% | 43% | 39% |
| *tracking the demonstration* | 18-21% | | | | |

**Mode C** -- the same target, with the forward model rolled on the candidate side:

| horizon | pooled | side_L | side_R | speed | turn |
|---|---|---|---|---|---|
| 1 | 33% | 21% | 62% | 43% | 24% |
| 3 | 43% | 42% | 57% | 55% | 30% |
| 5 | 44% | 43% | 64% | 55% | 28% |
| 10 | 37% | 32% | 40% | **86%** | 21% |

**Three results, in order of size.**

**1. The pass bar is met and then some.** Both sideways directions and turning clear their chance
rates under mode D at every horizon: side_L 86-100% against 17%, side_R 65-84% against 17%, turning
39-64% against 33%. **The channels transfer.**

**2. Widening the coordinate is worth far more than the world model was.** Pooled mode D goes
**33-41% on one channel to 68-70% on three** -- a doubling -- while demonstration-tracking falls to
18-21%, below chance. The single largest change measured in this line of work, and it came from the
target quantity rather than from anything in the planner.

**3. Strafing was never a hard behaviour; it was an invisible one.** `side_L` reads 13-25% on the
forward-only coordinate, at or below its 17% chance rate, and **86-100%** once lateral speed is in
the score. Every earlier finding that sideways "fails on every measurement" was measuring a channel
the score could not see. **That is a correction to a claim this project has repeated since F102.**

**And mode C is worse than mode D again, more clearly than before.** 33-44% against D's 68-70%,
with turning at 21-30% -- below its chance rate. Rolling the forward model does not merely fail to
help here; on three channels it **destroys** the turning signal, and the one place it wins is
forward speed at horizon 10 (86%). F135's reading stands and hardens: **the result is the shared
coordinate, and the rollout is currently subtracting from it.**

**Limits.** One checkpoint, one seed, offline selection among twelve recorded behaviours, held-out
clips. The B1 head is fitted on the B1's own actions and latents, which is the same cost the action
projector pays. Nothing here is a closed loop.

Checkpoint `wm/runs/beh12_hex-b1_body3/stage3_b1_nce_s0_bodyfit_proj.pt`; logs
`/tmp/f136_{fit,cal,C,D}.log`.

---


### F137. Without a library there is nothing to select: naive action search never finds locomotion

**The test the deployment claim needs, and F135/F136 never ran.** Those measured the world model as
a selector over twelve recorded B1 clips and found the shared coordinate does the selecting better
alone. **But the clips supply the "how"** -- the coordinate says "produce this body motion" and a
recorded clip already knows which joint sequence does. Remove the library and that gap is what a
world model is for. `sim/control/plan_without_library.py` puts the goal in the shared coordinate,
read from an insect's frames, and replaces the library with **sampled** action sequences:

    condition 1   body_head(ITM(e_t, FDM rolled h steps))  vs the goal   -- planning over unrecorded futures
    condition 2   body_head(proj(a))                       vs the goal   -- the same goal, no prediction
    random        a uniform pick from the same bank                      -- the floor

Both conditions see the identical bank; the winner of each is **executed in MuJoCo** and the body
motion it actually produced is measured, so nothing is scored on a model's own prediction.

| | distance to the goal | forward | lateral | yaw |
|---|---|---|---|---|
| world model | 0.331 | 0.186 | 0.082 | 0.248 |
| no rollout | 0.313 | 0.193 | 0.091 | 0.214 |
| **random** | **0.347** | 0.187 | 0.075 | 0.273 |

**All three are the same, and the reason is not that the world model failed.** Every sample produces
roughly the same motion, so no scoring rule can separate them. **The bank contains no solutions.**

**Measured directly, and it does not depend on the noise scale**, 24 samples per setting executed in
physics:

| noise, in units of each joint's own sd | upright | forward achieved | yaw achieved |
|---|---|---|---|
| 0.10 | 24/24 | -0.092 .. -0.085 | -0.246 .. -0.229 |
| 0.25 | 24/24 | -0.096 .. -0.079 | -0.254 .. -0.224 |
| 0.50 | 24/24 | -0.100 .. -0.079 | -0.292 .. -0.181 |
| 1.00 | 24/24 | -0.108 .. -0.025 | -0.304 .. -0.097 |
| *the recorded clips* | | **-0.003 .. 0.206** | **0.010 .. 0.076** |

**The robot stays upright and travels backwards while rotating, at every scale including 0.1 sd** --
which is essentially the dataset's mean pose held constant. A static average posture is not a gait,
and smoothed noise around it never becomes one: **locomotion needs periodic joint trajectories, and
a low-passed random walk is not in that family.** The reachable set of this sampler does not
intersect the goals.

**So the result is about the search space, not the scoring rule.** `wm/policy/planner.py` already
says this in its own docstring -- "sampling 18 continuous dimensions produces postures that do not
walk and that the forward model has never seen" -- and it is now measured on the quadruped: **raw
action-space search does not recover the "how" on a body with no demonstrations, and a world model
cannot rank its way out of a bank with no answers in it.**

**What this does to the three options.** Hand-authored periodic parameters (a CPG) would put gaits
in the bank and are exactly the per-robot knowledge this project claims not to need. **Optimising
the action sequence through the world model** -- CEM or gradients rather than uniform sampling -- is
the honest remaining version of "give the world model its actual shot", and is F138. **And learning
a policy** (Q16, teacher-student) is the option this result argues for most directly: if the "how"
cannot be found by run-time search, it has to be learned once and stored in weights.

**The comparison F135 drew stands unchanged.** With a library, the coordinate beats the rollout.
Without one, neither rule has anything to choose between. **Nothing here shows the world model
earning its place, and nothing here shows it failing at its job** -- the experiment could not put the
question.

Log `/tmp/f137.log`; results in `results/wm/closed_loop/plan_without_library/`.

---


### F138. The forward model's imagined state barely depends on the action, so teacher-student cannot proceed as it stands

**The prerequisite check for distillation, on the tightest bar yet.** F126 asked whether a rollout
ranks behaviours; a policy trained on imagined states needs the imagined *state* to be right.
`scripts/diagnostics/rollout_fidelity.py`, the three-channel two-embodiment pretrain
`beh12_hex-b1_body3/best.pt`, rolled on **48 held-out insect clips it never trained on**, actions
teacher-forced from the ITM so the forward model is measured alone:

| horizon | error / holding still | predicted over actual displacement | latent perturbed 1 sd | a real latent from another state |
|---|---|---|---|---|
| 1 | **0.732** | 0.41 | 0.741 | 0.775 |
| 3 | 0.702 | 0.63 | 0.708 | 0.767 |
| 5 | 0.764 | 0.76 | 0.768 | 0.826 |
| 10 | **0.978** | 1.02 | 0.984 | 1.013 |

**One-step ratio 0.732, growth +0.027 per step, and by ten steps 0.978 -- indistinguishable from
predicting no motion at all.**

**1. Fidelity.** The rolled state beats holding still by 27% at one step and by nothing at ten. The
usable imagination horizon is about five steps, a quarter of a second, and that is an upper bound
measured with the *true* actions supplied.

**2. Cause: both, and the objective is the larger half.** A 0.732 one-step ratio is not a
compounding failure -- nothing has compounded yet. The model is mediocre at the very first step,
which is a statement about what the pretraining objective asked for: reconstruction and the
auxiliary readouts, never rollout fidelity. The slope adds the rest.

**3. And the sharper finding, which neither of those captures: the state prediction hardly uses the
action.** Perturbing the latent by a full standard deviation moves the error from 0.732 to **0.741**
-- one percent. Substituting a real latent from a state the model never followed costs **six**
percent. **The forward model is predicting where this robot goes next mostly from where it is**,
which is why "on-manifold versus off-manifold" does not separate here: there is no action-manifold
dependence to leave.

**This is the `/mean-z` pathology of F98 and F119 in the pretrain itself**, on a held-out body, not
merely in MSE adaptation. F119 measured the contrastive term repairing it *during adaptation*
(`/mean-z` 0.985 to 0.49); nothing repairs it during pretraining, where the same objective is used.

**Verdict on teacher-student, which is what this was run to decide.** **It does not proceed as it
stands.** Distillation trains a policy against futures the world model imagines; if those futures
move by 1-6% when the action changes, the policy receives almost no signal about its own actions and
will learn whatever the state alone predicts. **A capped horizon does not fix this** -- the defect is
already present at one step, where there is nothing to cap.

**What it points at instead, in order.** A pretraining objective with a rollout-fidelity term and an
action-conditioning term -- the contrastive term already exists and is measured to restore action
sensitivity in adaptation (F119); applying it in pretraining is the smallest change that addresses
the measured cause. **That is a pretraining finding, and it is the first time in this project the
pretraining objective has been implicated by a direct measurement rather than by elimination.**

**Scope.** One pretrain, one seed, hexapod held-out clips, teacher-forced actions. The same
measurement on the B1 and on the contrastive-adapted checkpoints has not been run and would say
whether adaptation's contrastive term also repairs *state* fidelity or only ranking.

Log `/tmp/f138_hex.log`.

---


### F139. Contrastive adaptation buys action-conditioning and spends state fidelity; MSE does the reverse

> **The verdict of this entry is withdrawn by F140, and the cause is in its own last line.** Every
> number below rolls the forward model on the **ITM's** latents. F139 itself measured that the
> forward model is action-sensitive only in the **projector's** region -- and then drew its
> conclusion from the path it had just shown to be the wrong one. Re-measured on the projector path,
> the contrastive arm reads **state fidelity 0.710, not 1.370**, with `/mean-z` 0.476. **It has both
> properties at once**, and "no cheap fix exists because no checkpoint has both" is false. The
> tables below are correct as measurements of the ITM path; the conclusion drawn from them is not.

**Run before any re-pretrain, to decide cheap fix against expensive retrain.** The same state-rollout
diagnostic as F138, on B1 clips, across three checkpoints that differ only in what was done after
pretraining. Actions teacher-forced from the ITM.

| checkpoint | h=1 | 3 | 5 | 10 |
|---|---|---|---|---|
| pretrain, unadapted | 1.573 | 1.712 | 1.750 | 1.781 |
| **contrastive**, stage 3 `--lambda_nce 1` | **1.370** | 1.848 | 2.071 | 2.221 |
| **MSE**, stage 3 `--lambda_nce 0` | **0.592** | 0.862 | 1.008 | -- |

*ratio of the rolled state's error to the error of holding `e_t` still; below 1.0 beats predicting
no motion.*

**The unadapted pretrain is worse than holding still at every horizon**, which is F51 restated at
the state level: a hexapod-only forward model does not predict a quadruped. Both adaptations improve
on it; only MSE's crosses below 1.0.

**The MSE arm predicts the state far better than the contrastive one** -- 0.592 against 1.370 at one
step, and it is still at break-even by five steps where the contrastive arm is at 2.07 and rising.
That is the same trade F119 recorded from the loss side (`/hold` 0.802 against 0.891) and it is much
larger at the state level than that number suggested.

**The reconciliation, which was necessary before anything could be concluded.** F119 reports
`/mean-z` 0.49 on the contrastive arm -- the real action halving the error against the average one --
while F138 measured a 1-6% effect from perturbing the latent. **Both are correct and they feed
different latents:**

| action sensitivity, contrastive arm, h=1 | |
|---|---|
| `/mean-z` with the **projector's** latents (F119) | **0.49** |
| `/mean-z` with the **ITM's** latents (measured here) | **0.965** |
| 1 sd perturbation of the ITM's latent | 1.370 -> 1.366 |
| a real ITM latent from another state | 1.370 -> 1.425 |

**The forward model is action-sensitive only inside the region the projector produces.** Stage 3
trains it jointly with the projector on `proj(a)` inputs and on nothing else, so that is the only
subspace where its output responds to the action. The ITM's latents -- which is what F138 fed -- are
outside it.

**This changes F138's verdict in one direction and confirms it in the other.**

**Better than F138 concluded**: a distilled policy would emit actions, and actions reach the forward
model *through the projector*, which is the sensitive path. **Action-conditioning is not the
blocker** -- `/mean-z` 0.49 is a real dependence, and contrastive adaptation is what produces it.

**Worse than F138 concluded**: the arm that has the action-conditioning has **state fidelity of 1.370
at one step and 2.221 at ten** -- its imagined states are further from the truth than assuming the
robot froze. Teacher-student would train a policy against futures that are wrong in the ordinary
sense, not merely action-insensitive. And the arm with usable state fidelity is the one F119 measured
at `/mean-z` 0.985, which has no action-conditioning at all.

**So the answer to the question this was run to settle: neither, and that is the finding.** There is
no cheap fix, because no existing checkpoint has both properties; and a re-pretrain with the
contrastive term alone would reproduce the contrastive arm's trade rather than escape it. **The
objective needs a rollout-fidelity term *and* an action-conditioning term together, and the evidence
that they are in tension is now direct**: every checkpoint measured pays for one with the other.

**Do not start a re-pretrain on the strength of "apply contrastive properly".** F119's contrastive
term is measured here degrading exactly the quantity distillation depends on.

**Scope.** One seed per arm, B1 clips, teacher-forced ITM actions, one-channel checkpoints for the
adapted arms. The projector-latent `/mean-z` is quoted from F119 rather than re-measured at every
horizon.

Logs `/tmp/f139_*.log`.

---


### F140. Measured on the path a policy would drive, the contrastive arm already has both properties

**F139 asked whether the fix was cheap or expensive and answered from the wrong latent path.** The
forward model is action-sensitive only inside the region the projector produces -- F139 measured
that itself -- and then rolled on the ITM's latents to judge state fidelity. `rollout_fidelity.py`
gained `--latent projector`, the path a distilled policy would actually drive, and the picture
changes:

| B1 clips, one step | state fidelity | `/mean-z` across clips | `/mean-z` within a clip |
|---|---|---|---|
| **MSE**, `--lambda_nce 0` | **0.585** | 0.969 | 0.975 |
| **contrastive**, `--lambda_nce 1` | **0.710** | **0.476** | 0.951 |

*(the same contrastive checkpoint on the ITM path reads 1.370 and 0.965 -- F139's numbers)*

**The contrastive arm predicts the state better than holding still and reads the action**, at the
same time, on the path that matters. **The trade is mild**: full action-sensitivity costs 0.585 to
0.710, about a fifth of the state fidelity, and buys 0.969 to 0.476.

**And that already clears the bar the sweep was written to look for** -- state under about 0.8 with
meaningful action-sensitivity -- **at lambda 1, with no sweep and no re-pretrain.**

**The `/mean-z` reconciliation, which had to be settled first.** F119 reports 0.49 and F138/F139
reported a 1-6% effect; both are right and they average different things:

| what the baseline latent is | contrastive arm |
|---|---|
| the mean across the whole dataset -- `wm/adapt3`'s, and F119's | **0.476** |
| the mean within the clip being predicted | 0.951 |

> **F147 corrects what the within-clip number means.** It is measured at a **fixed magnitude**,
> where the gait is periodic and a frame fixes the phase, so the action is redundant by
> construction (Slide 11). Let the magnitude vary inside the same behaviour family and the same
> checkpoint reads **0.485**, against 0.476 across all behaviours. **The model is not blind within a
> behaviour; it is blind where there is nothing to see.** Quote 0.951 as a property of the task,
> never as the model discarding information.

**Across behaviours the action matters; within one behaviour it barely does.** That is the same
shape as F111 -- the kind of motion transfers and the amount does not -- now visible in the forward
model's own predictions, and it is a real limit on fine control rather than a measurement artefact.
The tool prints both.

**Consequences.**

**Teacher-student is not blocked by the objective.** F139's verdict was the strongest argument
against it and it does not survive. What remains true from F138 is the horizon: the contrastive arm
degrades from 0.710 at one step to worse than hold-still by five, so imagined rollouts stay short.

**The lambda sweep is no longer a decision and becomes a refinement.** Its question was whether any
lambda gives both; lambda 1 does. Running it would map whether an intermediate value keeps 0.476
while recovering state toward 0.585, which is worth hours but decides nothing. The sheet is written
(`scripts/com7_lambda_sweep.sh`) and does not need to run before anything else.

**The lesson is the one this session keeps paying for.** F139 had the fact that decided its own
verdict -- action sensitivity lives in the projector's region -- inside the same entry, and did not
apply it. **Read a diagnostic's own finding back onto the diagnostic before drawing from it.**

Logs `/tmp/f140_*.log`.

---


### F141. Teacher-student, designed against what the measurements actually allow (design only, not run)

**Written before anything is executed, because the two surviving constraints decide the shape of the
experiment and one of them decides what counts as success.** Q16 committed to teacher-student on
2026-08-28 for a reason no measurement has touched: **scoring candidates needs a target-robot policy
that already performs the behaviours, and learning one needs only the ability to actuate.** F140
removed the objection that grew after that decision -- the objective does not trade state fidelity
against action-sensitivity once both are measured on the projector path.

## What the measurements permit

| constraint | measured | consequence for the design |
|---|---|---|
| **horizon** | contrastive arm 0.710 at one step, worse than frozen by five (F138, F140) | imagine **h <= 3**, never 5. The teacher's signal is short-horizon or it is noise |
| **magnitude** | `/mean-z` **0.476** across behaviours, **0.951** within one (F140) | the model reads **which** movement, not **how much**. A policy distilled from it can be asked for a behaviour and not for a rate |
| **coordinate** | three channels calibrate on both robots, selection 70% cross-embodiment (F134, F136) | the goal is a body-motion vector, not a frame and not a joint target |
| **search** | sampled joint commands never produce locomotion (F137) | the policy cannot be found by run-time search; it has to be fitted, which is what this is |

## The design

    goal g      a body-motion vector -- forward, lateral, yaw, dimensionless -- read from a video
                by the shared head. Cross-embodiment by construction (F136)

    teacher     from state e_t, for each candidate action, roll the FDM h <= 3 steps on proj(a) and
                read body_head(ITM(e_t, rolled)). The label is the action whose imagined motion is
                nearest g

    student     pi(e_t, g) -> joint targets, trained on those labels, on states the student itself
                reaches. One forward pass at run time, no library, no rollout, no camera if the
                state is proprioceptive

**On the insect**, where the forward model is worth rolling. The B1 is where it is deployed, not
where it is learned, and that ordering is Q16's and unchanged.

**The candidate set at training time is not a library.** It is whatever the student currently emits
plus exploration around it; the teacher ranks those with the world model. Nothing has to already
walk -- which is the whole point of the change, and the reason F137's negative result does not
block this: F137 searched **once, blind, at run time**; this searches **repeatedly, locally, around
a policy that is improving**.

## The success criterion, fixed now

**Primary, and it is what the magnitude limit permits**: on a held-out goal, the student produces
the **right behaviour family** -- forward, turning, strafing, and the sign of the turn and the strafe
-- above the chance rate, on a body with no candidate library, with the goal read from another
robot's video. Direction is reported separately from family (F109).

**Secondary, and expected to fail**: the achieved *magnitude* within a family tracks the goal's.
F111 measured 0.074 correlation for speed across embodiments and F140 explains it -- within a
behaviour the forward model barely distinguishes actions, so a teacher built on it cannot label
"faster" reliably. **Report the correlation and expect it near zero.**

**Is behaviour-type control a full result or a partial one? Partial, and it should be written as
partial.** A controller that can be told *walk / turn left / strafe right* but not *how fast* is not
a locomotion controller in the sense a robotics reader expects, and calling it one would be the
overclaim this project has corrected five times in one session. **What it would be is the first
demonstration that a body-motion goal read from one robot's video produces a working controller on
a robot with no demonstrations of its own** -- which is the deliverable Q16 named and which LAC-WM
does not produce, since it stops at selecting among what its VLA proposed.

**The honest sentence, drafted now so it cannot drift later**: *a quadruped with no recorded
behaviours is driven to the behaviour a stick insect's video specifies, in kind but not in
magnitude, by a policy distilled from a world model that was never shown a quadruped.* Every clause
in that is measurable and three of them are already measured.

## What would make it a full result, and is out of scope here

Magnitude control needs a forward model that distinguishes actions **within** a behaviour --
`/mean-z` within-clip is 0.951 and would have to fall. Nothing measured suggests the contrastive
term does that; it is a within-behaviour discrimination problem and the term's negatives are drawn
from other behaviours. **That is a separate experiment and naming it here keeps the partial result
from being sold as the whole one.**

## The hole in the design above, found before building it

**At initialisation the student is random, so the teacher ranks random joint targets -- which is
F137 exactly.** F137 measured that sampled joint commands on a quadruped never produce locomotion at
any noise scale, and nothing about the insect changes the argument: an 18-D joint target drawn near
a mean pose is a posture, not a gait, and the teacher would be choosing the least-bad of a bank with
no answers in it. **"Search locally around a policy that is improving" only escapes F137 if the
policy starts somewhere that moves.**

**So the design needs a bootstrap, and where it comes from decides whether the claim survives.**

| bootstrap | what it costs the claim |
|---|---|
| recorded clips of the **target** robot | the library returns; the claim collapses to candidate scoring with extra steps |
| recorded clips of the **source** robot (the insect) | **acceptable**: the insect is the robot we are transferring *from*, and having data for it is the premise, not a concession |
| random exploration with a survival/motion reward | honest but it is reinforcement learning, and the project's own AMP branch was abandoned for producing gaits too poor to trust (`PROGRESS.md` §13) |

**The second is the only one that keeps the claim and it changes the pipeline's shape**: clone the
student on the *insect's* recorded actions to get something that moves, then improve it on the
*target* robot with the world-model teacher and goals in the shared coordinate. The target robot
still needs no recorded behaviours, which is the property Q16 was protecting.

**But it introduces a transfer the design did not have**, since an insect-cloned policy emits 18-D
insect joint targets and the quadruped takes 12. The action spaces are disjoint -- that is this
project's whole premise -- so the clone cannot be copied across. **What can cross is the goal, which
is the shared coordinate, and that is already measured (F136).** The student would therefore have to
be per-robot and bootstrapped per-robot, which means the target robot needs *some* way to move
before the teacher is useful.

**That is the real open question this design surfaces, and it should be settled before any code:**
**where does the target robot's first motion come from, if not from a library and not from
reinforcement learning?** Until that has an answer, teacher-student on the B1 inherits F137's
result. **On the insect it is buildable today** -- clips exist, replay is exact (F102, 1.06) -- so
the honest first experiment is teacher-student **within** the insect, which tests the mechanism
without claiming the transfer.

## Before it runs

The student trains on states it reaches, so it needs a simulator in the loop on the insect
(CoppeliaSim, one instance, GUI -- F98). Budget is the same order as a stage-3 adaptation per
iteration, and the horizon cap makes each teacher call cheap. **Nothing here is measured yet; this
entry is a design and should be cited as one.**

---


### F142. Teacher-student on the insect: the reference, the bar, and the cloning control (teacher stage pending)

**An engine test on the easy case: same robot on both sides, no transfer claimed or measurable.**
The question is only whether short imagined rollouts can train a policy that walks. If they cannot
here -- one robot, a trustworthy short-horizon forward model, legitimate replayable clips -- no
cross-embodiment result rescues the direction.

## Step 1, fixed before anything was trained

`hexapod_ep100` (`speed_c7.1`) replayed through the insect's own physics, 66 steps, three seconds:

| | |
|---|---|
| the clip's recorded head displacement | 0.6454 m |
| **replayed** | **D_real = 0.6566 m** |
| replay ratio | 1.017 *(F102 measured 1.06 on this robot)* |
| **the bar, 50% of D_real** | **0.3283 m** |

## Step 2, the pre-registered criterion

**Both, over the whole three-second window, or it fails:** upright throughout -- head height never
below 0.6 of its settled value -- **and** at least 0.3283 m travelled. A statue passes the first and
fails the second; a lurch does the reverse. **The render is diagnosis only and cannot promote a
numeric failure.**

## The cloning control, which is also the bootstrap

F137 measured that a policy starting from noise never walks, so the student is bootstrapped by
cloning the insect's own forward-walking clips -- honest on this robot, which is the body the
project has data for. **Cloning alone is therefore also the control: if it clears the bar, the
teacher has added nothing.** Held-out cloning error 0.065 after 2,000 epochs.

| cloned student, run in the insect's physics | |
|---|---|
| travelled | **0.2349 m = 36% of D_real** |
| upright the whole window | **yes**, minimum head height 0.1400 against 0.1501 settled |
| **verdict** | **FAIL**, on distance |

**It walks and it under-travels.** The failure mode is not collapse -- the robot stays up for the
full three seconds -- it is a gait that covers a third of the ground the recorded one does. That is
the informative failure: **the control does not pass, so anything the teacher stage adds will be
attributable.**

Video at `results/wm/closed_loop/f142_video/f142_bc_vs_real.mp4`, recorded walk on the left and the
cloned student on the right, and `f142_bc_student.mp4` alone.

## What is not measured yet, and why

**The teacher stage is not built**, because the insect-side forward model cannot yet label. Measured
on the projector path, held-out insect clips: state fidelity **0.757** at one step and **0.727** at
two -- inside the bar -- with `/mean-z` across clips at **0.966**. **The rollout is good and deaf**:
a teacher whose prediction moves three percent when the action changes ranks noise. The quadruped
side does not have this problem because it has had a contrastive stage 3 (0.476, F140), and
`scripts/com7_stage3_hexapod.sh` applies the same term on the insect with the gate written into it.

**This entry stops here deliberately.** The bar, the reference and the control are recorded before
the teacher exists, so the teacher's number cannot be graded against a bar chosen after seeing it.

Logs `/tmp/f142_*.log`; runs in `results/wm/closed_loop/f142_*`.

---


### F143. The insect teacher passes its gate on the body it will teach, and fails it on a body it will not

**F142's teacher could not label** -- the insect-side pretrain rolled well and ignored the action,
`/mean-z` 0.966 across clips. `scripts/com7_stage3_hexapod.sh` applied F119's contrastive term on the
insect, 15,000 steps, 24 of the 48 `c10f10t10` clips, with the other 24 -- including
`hexapod_ep100`, the clip F142 takes its goal and `D_real` from -- held out. **The gate was written
into the sheet before the run.**

**On the body it was adapted to and that F142 uses:**

| | before stage 3 | after |
|---|---|---|
| state fidelity, h=1 | 0.705 | **0.739** *(adapt3's own held-out split)* |
| `/mean-z` across clips | 0.955 | **0.583** |
| exact condition picked | 53% | **85-90%** |
| behaviour family | 84% | **95%** *(chance 28%)* |

Re-measured with `rollout_fidelity.py` over all 48 clips -- optimistic, since 24 were trained on,
and reported as such: **0.592 at one step, 0.715 at two, 0.796 at three**, `/mean-z` across clips
**0.534 / 0.578 / 0.621**, and a wrong-but-real latent now costs **64%** at one step where before it
cost three.

**Gate passed.** Sensitivity arrives, fidelity stays inside the 0.8 bar out to three steps, and the
horizon cap F141 fixed at `h <= 3` is exactly where the ratio crosses 0.8. **F142's teacher stage
can be built.** The trade is the same shape and size as the B1's (F140: 0.585 to 0.710 buying 0.969
to 0.476) -- **F140's refutation of F139 holds on a second robot.**

**And on a body it was not adapted to, the gate fails**, which is a new limit and not a
contradiction:

| held-out **body** `c08f09t09` | before | after |
|---|---|---|
| state fidelity, h=1 | **0.757** | **1.052** -- worse than a frozen frame |
| `/mean-z` across clips | 0.966 | 0.774 |
| a wrong latent costs | 3% | **19%** |

**The contrastive term buys action-sensitivity everywhere and pays for it in cross-*body*
prediction.** Within the body it is adapted to, the payment is 5% of fidelity; across bodies it is
the whole margin. That is a real constraint on where a teacher trained this way can be used, and it
is a *cross-morphology* statement -- Stage 1's territory -- rather than a cross-embodiment one.

**What it means for the direction.** F142 teaches on `c10f10t10` and is graded on `c10f10t10`, so it
is inside the region where the teacher is sound. **A teacher expected to generalise to an unseen
body would need something this adaptation does not provide**, and nothing measured says the term can
be tuned to give both -- that is the lambda sweep's question
(`scripts/com7_lambda_sweep.sh`), now with a reason to run it that F140 had removed.

Log in `results/wm/`; the com7 run is `beh12_hex-b1_body3/stage3_hex_nce_s0.pt`, commit 3923cfa.

---


### F144. The engine test fails: teacher-student on the easy case walks less far than cloning alone

**Judged against the bar locked in F142 before the teacher existed**, on `c10f10t10`, the body F143
validated the teacher for. Goal and reference from `hexapod_ep100`, held out of the teacher's stage-3
training and of the cloning split.

| | distance | of `D_real` = 0.6566 m | upright the full 3 s | verdict |
|---|---|---|---|---|
| **cloning only**, the control | 0.2349 m | **36%** | yes | **FAIL** |
| **teacher-student**, 10 DAgger rounds | 0.2042 m | **31%** | yes | **FAIL** |
| *the bar* | 0.3283 m | 50% | required | |

**The clone reproduced its earlier number exactly (36%, 0.2349 m) in the same simulator session**,
so the comparison is not a session artefact.

**The pre-registered reading for 31% is the third bracket: the teacher adds nothing.** It made the
policy slightly worse. **The engine test on the easy case fails.**

**What the run looked like.** Ten episodes, 66 teacher labels each, 32 candidates per step ranked by
rolling the forward model three steps and reading the body motion the rolled transition implies,
cloning pairs kept in the buffer throughout. Distance during training wandered between 0.09 and
0.35 m with no trend, and the fitting loss fell from 0.067 to 0.055 -- **the student was fitting its
labels; the labels were not worth fitting.**

**Neither policy falls.** Both stay upright for the whole window and cover a third of the ground.
The failure is a gait that does not travel, not a collapse, in both arms.

Video `results/wm/closed_loop/f142_video/f144_real_clone_taught.mp4` -- recorded walk, clone, taught,
side by side. **Diagnosis only; it does not overturn the numbers above.**

**The most likely mechanism, stated as a hypothesis and not measured.** Candidates are Gaussian
perturbations of the student's own action at 0.5 of each joint's standard deviation. If the
teacher's ranking of those perturbations is close to arbitrary, DAgger trains the student on noise
around itself, which is exactly the small degradation observed. **The measurement that would settle
it is label quality**: execute the labelled action and the student's own from the same state and
compare the body motion each produces. That was not run, and F143's `/mean-z` 0.534 is *not* that
measurement -- it says the model separates a real action from an average one, not that it orders
small perturbations of one action correctly.

**Scope of the teacher, from F143 and repeated here because it bounds any reuse**: this teacher is
validated on `c10f10t10` only. On `c08f09t09` its state fidelity is 1.052, worse than a frozen
frame. **Do not use it off `c10f10t10`.**

**What this does to the direction.** The pre-registration says stop, and the honest report is that
teacher-student, in the form specified in F141 and built here, does not train a walking policy on
the case designed to be easiest for it. **It does not say distillation from world models cannot
work**; it says this teacher, ranking local perturbations at this horizon, produced no usable
signal. The distinction matters for what gets written and neither half should be dropped.

Runs `results/wm/closed_loop/f144_*`; logs `/tmp/f144_*.log`; students in `wm/runs/students/`.

---


### F145. The teacher ranks behaviours and cannot rank perturbations of one -- the mechanism of F144

**Characterisation of a failed test, not an appeal against it.** F144's bar was pre-registered and
failed; this asks only why, at two scales, with the same teacher and the same scoring rule the
labeller used.

**Local -- the perturbations the F144 teacher actually ranked, judged in the simulator.** At twelve
branch points along a held-out clip, the student's own action and the teacher's pick out of 32
Gaussian perturbations were each executed from the same state for three steps, and the body motion
each produced was compared with the goal:

| | |
|---|---|
| teacher's pick closer to the goal | **4 of 12 = 33%** |
| a coin | 50% |
| the teacher kept the student's own action | **0 of 12** |
| mean distance to the goal, student | **0.1299** |
| mean distance to the goal, teacher | **0.1304** |

> **Read with F147.** The candidates here are perturbations of one action at one magnitude, and the
> physics barely separated them -- 0.1304 against 0.1299. **A ranker cannot order what the outcome
> does not distinguish**, so this is partly a property of the task and not only of the teacher.

**The teacher's ranking of local perturbations is worse than a coin and its labels are
indistinguishable from the student's own output** -- 0.1304 against 0.1299 over twelve states. It
changed the action every single time and gained nothing by it. **That is the mechanism of F144**:
DAgger trained the student on noise around itself, which is exactly the small degradation the bar
measured, 36% to 31%.

**Coarse -- the twelve recorded conditions as candidates, same rule, same states:**

| | |
|---|---|
| pick shares the goal's behaviour family | **55% of 120 states** |
| chance | 33% |

**Above chance and far below F143's 95%, and the two are not in conflict** -- they ask different
questions. F143's number comes from `adapt3`, which is handed the *true next embedding* and asked to
rank actions against it. This is handed only a **goal body motion** and has to reach it, which is
the harder and more deployment-like task. **55% against 33% is a real but coarse ability.**

**So the answer is "coarse, not fine", and it is the one that decides what may be tried next.** The
teacher can tell walking from turning from strafing well enough to beat chance; it cannot tell a
good version of walking from a slightly different one. That is the same wall as F111's speed
correlation of 0.074 and F140's within-clip `/mean-z` of 0.951, now measured on the labelling path
and judged by physics rather than by the model.

**What this rules in and out for any future distillation.** A scheme whose teacher must **choose
among behaviours** has a signal to work with -- 55% against 33%. A scheme whose teacher must
**refine within a behaviour**, which is what improving a gait requires and what F141's design asked
for, does not. **F144 is not a tuning failure and no amount of DAgger rounds, candidate count or
horizon fixes it**, because the ordering the labels depend on is not there to be sharpened.

**F144's bar stands and this does not reopen it.** Runs and logs `/tmp/f145_*.log`;
`scripts/diagnostics/teacher_label_quality.py`.

---


### F146. The next pretraining objective, pre-registered (plan, not run)

**F138 traced the action-insensitivity to pretraining and F145 measured what it costs**: the
forward model orders behaviours and not perturbations, so anything that needs the second -- planning
within a behaviour, distillation, magnitude control -- has no signal to use. **ActSWM (arXiv
2607.26712, 2026) names the pathology Context Collapse and publishes a fix**; CD-LAM (2607.09185)
reports the fragility-under-small-perturbations half and Dueling World Models (2608.06706) the
loss-improves-while-actions-merge half. *References supplied by the advisor and not verified from
this machine.*

**Three terms, together, replacing the current pretraining objective.**

**1. A rollout-level action-sensitivity hinge.** Penalise similarity between a rollout driven by the
real actions and one driven by the **null** action, accumulated over K steps rather than one.
**The null is each embodiment's standing stance** -- per-embodiment vectors, identically defined as
"the pose that body stands still in" (F148). Not the zero vector, which collapses both robots, and
**not the dataset-mean pose**, which collapses the B1 and drifts the insect: a null that means
*still* on one robot and *falling* on the other would make a cross-embodiment sensitivity comparison
measure the nulls instead of the models. **A one-step
penalty cannot see our failure**: F138 measures 0.732 at one step decaying to 0.978 by ten, and the
horizon is what a policy consumes.

**2. A frozen action-readout, and it is a NEW module -- not the ITM.** A small head, **randomly
initialised and never trained**, that scores the separation between the two rollouts. The sensitivity
loss backpropagates through the latent inputs only, never through the readout itself.

> **Design decision, locked: do not freeze the ITM.** The ITM produces the `z` that the action
> projector is fitted to imitate (`wm/fit_projector`), so freezing it at random weights makes `z`
> arbitrary and breaks the projector -- and with it every control-time path, since the projector is
> how the model is driven when the future frame is unavailable. **The new readout consumes nothing
> downstream**: its only job is to force gradient into the forward model and the latent so that
> transitions are genuinely separated. Nothing reads its output but the loss. **The ITM keeps
> training exactly as it does now.** **This targets a trap we measured rather than a
hypothetical one**: F139 and F143 found the contrastive term producing sensitivity that lives only
in the projector's region and only on the body it was adapted to -- `/mean-z` 0.534 on `c10f10t10`
and state fidelity 1.052 on `c08f09t09`. A learnable readout can relocate the signal to wherever it
is measured instead of separating the transitions.

**3. Keep the prediction loss.** Prediction plus hinge plus frozen readout. F140 established the
trade is a knob rather than a wall -- 0.585/0.969 for MSE against 0.710/0.476 for contrastive, a
fifth of the fidelity buying the whole sensitivity -- so asking for both at once is not asking for
something no checkpoint has shown.

**Pre-registered expectations, fixed before the run.**

| | expectation |
|---|---|
| usable imagination horizon | **the primary target**: past **h = 3**, where the state ratio crosses 0.8. This is the clean, confirmed Context Collapse -- 0.732 at one step to 0.978 at ten on the pretrain (F138) |
| magnitude **ranking** | **open, and pre-declared neither way** (F147). The prediction already *reacts* to magnitude -- `/mean-z` 0.485 across sizes -- and ranks it poorly: turning 39-64%, forward 41-62%, against 33% chance (F136). **Reacting and ranking are different and only the first is established**, so this run is not a test of whether the fix repairs ranking, and a null there refutes nothing |
| **action-sensitivity across disjoint embodiments** | **the open question, and the contribution** |

**The third row is ours and not ActSWM's.** They test within one body. **F143 is the measurement
that makes the cross-embodiment version a real question**: the naive contrastive repair holds on the
robot it was adapted to and collapses on another one, so "does this fix survive a change of body"
has a documented negative answer for the obvious approach.

**Two things this plan does not address, named so they are not quietly folded in.** Where a target
robot's first motion comes from without a recorded library and without reinforcement learning
(F137, F141) -- distillation needs a policy that already moves, and removing the library is the
point. And whether a repaired forward model would rescue candidate scoring, which F135 showed does
not need a rollout at all.

**Status: plan. Nothing here is measured.** The three-channel pretraining run it builds on is
complete and evaluated (F134, F136).

---


### F147. The within-behaviour blindness is the task, not the model -- and that re-labels two earlier findings

**F146 is the ActSWM plan; this is the prerequisite that was almost skipped.** Slide 11 measured a
property of the *task*: at one speed the gait is periodic, a single frame fixes the phase, and the
command follows -- removing the transition entirely costs 28-34%, a second frame is worth 1.11x, and
predicting 32 frames ahead is as accurate as predicting the present. **That is real redundancy and
no objective should remove it.** F138/F140/F145 measured a *model* failure. The two were being read
as one fact, and the number that conflated them -- `/mean-z` 0.951 within a clip -- was measured at
**fixed magnitude**, where the task is redundant by construction.

**The measurement that separates them: a third baseline.** The mean latent of the *same behaviour
family at other magnitudes*, where the action genuinely is not redundant.

| `/mean-z`, h = 1, projector path | within one clip | **within the family** | across all behaviours |
|---|---|---|---|
| **B1**, contrastive stage 3 | **0.951** | **0.485** | 0.476 |
| **insect**, contrastive stage 3 | 0.697 | 0.597 | 0.534 |

**On the B1 the answer is unambiguous.** Holding the speed fixed, the action changes the prediction
by 5%. Letting the speed vary, it changes it by more than half -- and 0.485 is indistinguishable
from the across-everything 0.476. **The model is not blind within a behaviour. It is blind within a
behaviour at one magnitude, which is the regime where there is nothing to see.**

**So two earlier readings have to be corrected.**

**F140's within-clip 0.951 is not evidence of collapse.** It is Slide 11's task property, measured
on the forward model instead of the decoder. It should be quoted as "the action is near-redundant at
fixed magnitude", never as "the model ignores the action within a behaviour".

**F145's local-ranking failure is at least partly the same thing.** The teacher ranked
perturbations of one action at 33% against a coin's 50% -- and the physics agreed with it: executing
the teacher's pick and the student's own gave mean distances to the goal of **0.1304 and 0.1299**.
**The world itself barely distinguished those actions.** A ranker cannot order what the outcome does
not separate. The teacher's labels were uninformative because the candidates were, not only because
the model was.

**What survives unchanged, and it is what F146 is for.** Context Collapse is still measured and
still a model failure: on the **pretrain**, held-out insect clips, the rolled state goes 0.732 at
one step to **0.978** at ten, and a full-standard-deviation perturbation of the action changes the
answer by **1%** (F138). That is at every magnitude, not one, and it is horizon decay rather than
task redundancy. **The ActSWM rebuild is aimed at that, and this finding does not weaken it** -- it
removes a piece of evidence that was never valid for it.

**What it does change about expectations, and the distinction has three levels rather than two.**
`/mean-z` 0.485 says the prediction **reacts** to actions of a different magnitude. It does **not**
say the model **orders** them correctly, and F136 measures that it does not do so well: at three
channels, turning and forward are the *weakest* families in cross-embodiment selection -- turning
39-64% and forward 41-62% against a 33% chance rate, where sideways reaches 65-100%.

| | status |
|---|---|
| within one magnitude: does the action matter at all | **no, and correctly so** -- Slide 11's task property, not a target |
| across magnitudes: does the prediction **react** | **yes**, 0.485 -- not collapsed |
| across magnitudes: does it **rank** them correctly | **not well, and open** -- F136's weakest families |

**So F146 should be judged primarily on horizon** -- 0.732 at one step to 0.978 at ten is the clean,
confirmed Context Collapse and the thing the hinge targets. **Whether it also improves magnitude
*ranking* is open and must not be pre-declared either way**, solved or unfixable.

**And the still-open question**: whether a model could ever rank actions *within* a behaviour at a
fixed magnitude, or whether the outcome genuinely does not depend on them. The 0.1304 against 0.1299
says the second, on twelve states of one robot. **That is a limit on control precision that no
objective fixes**, and it should be written as a property of the task.

Logs `/tmp/f147_{hex,b1}.log`; `rollout_fidelity.py --family_mean`.

---


### F148. The null action: holding the settled pose, on both robots. The zero vector falls.

**A prerequisite for F146, not a detail.** The action-sensitivity hinge contrasts a rollout on the
real actions against one on a **null** action. If the null makes the robot collapse, the hinge is
trained to separate *walking* from *falling* -- trivial, and it teaches nothing about the action
channel. Three candidates, each held constant for three seconds, on both bodies:

| B1, MuJoCo | travel | jitter | forward | lateral | yaw | min height | verdict |
|---|---|---|---|---|---|---|---|
| hold the settled pose | 0.008 m | 0.61 mm | +0.000 | -0.000 | -0.000 | 0.462 | **still** |
| **the standing stance the clips start in** | 0.008 m | **0.17 mm** | +0.001 | -0.000 | -0.000 | 0.502 | **still** |
| the dataset-**mean** pose | **0.628 m** | 0.36 mm | -0.001 | +0.001 | -0.001 | **0.126** | **FALLS** |
| **the zero vector** | **1.077 m** | 16.9 mm | -0.001 | **-0.104** | -0.037 | **0.088** | **FALLS** |

| hexapod `c10f10t10`, CoppeliaSim | travel | jitter | forward | lateral | yaw | min height | verdict |
|---|---|---|---|---|---|---|---|
| hold the settled pose | 0.0002 m | 0.03 mm | -0.000 | +0.000 | +0.000 | 0.151 | **still** |
| **the standing stance the clips start in** | 0.0002 m | 0.04 mm | -0.000 | +0.000 | +0.000 | 0.151 | **still** |
| the dataset-**mean** pose | 0.063 m | 0.04 mm | +0.000 | -0.000 | -0.000 | 0.150 | drifts |
| **the zero vector** | 0.034 m | 0.04 mm | -0.000 | -0.001 | +0.000 | **0.019** | **FALLS** |

**The literal zero vector is the wrong null on both robots and for the same reason.** These action
spaces are **joint targets**, not torques, so zero commands every joint to angle zero -- a specific
and unsupportable posture. The B1 collapses and slides **1.08 m** doing it, at a lateral speed of
-0.104, which is faster sideways than any recorded strafing clip. The insect folds to a fifth of its
standing height. **A hinge built on that would be scoring "did the robot fall", and the answer would
be yes every time.**

**Holding a pose works on both, and the two variants are equivalent here.** Commanding the settled
pose back to itself and commanding the dataset's neutral stance both give body motion of zero to
three decimals in every channel, no fall, and sub-millimetre jitter. On the insect they coincide by
construction: the warm-up holds the clip's first command, so the pose it settles into *is* the
neutral stance.

**A fourth candidate was proposed and it fails: the dataset-**mean** pose.**

| | B1 travel | B1 verdict | hexapod travel | hexapod verdict |
|---|---|---|---|---|
| the standing stance | 0.008 m | still | 0.0002 m | still |
| **the dataset-mean pose** | **0.628 m** | **FALLS**, height 0.126 against 0.50 | 0.063 m | drifts, 150x the stance |

**A mean over a gait cycle averages swing against stance and produces a posture that exists in no
frame and holds nothing up.** F137 measured the same thing from the other side: sampling around the
mean pose at 0.1 sd sent the quadruped backwards while rotating. **"The dataset-mean pose" is not
the null.**

**The choice for F146, stated so it cannot be misread: the standing stance of each embodiment** --
the pose that robot's clips start in and settle into. It is **per-embodiment by necessity** (18-D
and 12-D, different vectors) and **identically defined on both** (the pose that body stands still
in), which is the property the cross-embodiment test depends on.

**That last point is the one that matters for the contribution.** If the null meant *still* on one
robot and *collapsing* on the other, a cross-embodiment sensitivity comparison would be measuring
the difference between the two nulls rather than between the two models. **The stance is the only
candidate that reads 0.000 in every channel on both bodies**; the mean falls on one, and the zero
vector falls on both.

**One property worth stating before the rebuild.** This null is *a pose*, not an absence of
actuation. The robot is being actively held, so the contrast is "commanded to hold still" against
"commanded to move", not "actuated" against "unactuated". **That is the right contrast for a
joint-target action space** and it should be written that way in the objective, because "zero
action" in a torque-controlled system would mean something else entirely.

`scripts/diagnostics/null_action.py`; log `/tmp/f148.log`.

---


### F149. Gradient balance on the three-channel pretrain: reconstruction is 96-99% of the loss and 22-41% of the gradient

**The pre-rebuild baseline.** F87 measured this on `beh12_body_fwd`, a one-channel checkpoint, and
its first reading over-standardised the body target on an insect-only batch. `loss_gradient_balance`
now takes the statistics from the checkpoint -- pooled across both embodiments, which is why
`train.py` stores them -- so the correction is applied by construction. Re-measured on
`beh12_hex-b1_body3/best.pt`, `body_dim 3`, 24 transitions, each term's gradient taken with respect
to the **same** `z`:

| hexapod | lambda | loss | share of loss | \|dL/dz\| raw | x lambda | **share of gradient** |
|---|---|---|---|---|---|---|
| recon | 1.00 | 1.5266 | **98.8%** | 0.0029 | 0.0029 | **40.6%** |
| motion | 1.00 | 0.0046 | 0.3% | 0.0016 | 0.0016 | 22.8% |
| body | 0.50 | 0.0266 | 0.9% | **0.0052** | 0.0026 | **36.6%** |

| B1 | lambda | loss | share of loss | \|dL/dz\| raw | x lambda | **share of gradient** |
|---|---|---|---|---|---|---|
| recon | 1.00 | 1.3131 | **95.6%** | 0.0028 | 0.0028 | **22.3%** |
| motion | 1.00 | 0.0304 | 2.2% | 0.0058 | 0.0058 | **46.3%** |
| body | 0.50 | 0.0589 | 2.1% | **0.0079** | 0.0039 | 31.4% |

**Reconstruction is 96-99% of the loss and 22-41% of the gradient into the latent.** The gap between
the two columns is the whole point: **the loss values were never the right thing to read**, and F23's
"99 percent of the gradient goes to reconstruction" was an inference from loss magnitudes, not a
measurement of gradients. It should not be quoted.

**The body term has the largest raw `|dL/dz|` on both robots** -- 0.0052 and 0.0079 against
reconstruction's 0.0029 and 0.0028 -- and is then halved by `lambda_body 0.5`. **The smallest term
by weight pulls hardest on the latent.**

**Compared with F87's one-channel run** (recon 5.1%, motion 12.3%, body 82.5% before rescaling,
about 13/37/50 after), the three-channel pretrain is **more balanced**: reconstruction's share rises
from about 13% to 22-41% and the body term's falls from about 50% to 31-37%. Widening the shared
target from one channel to three spread the gradient rather than concentrating it.

**What this is for.** It is the number the ActSWM rebuild (F146) will be measured against. The
rebuild adds a rollout-separation term scored through a frozen readout, and **the claim it has to
support is that gradient into the latent increases and does so through the new term**, not that the
loss curve moves. Re-measuring this table after the rebuild is the check.

**And it rules out the same intervention F87 ruled out**, on a different checkpoint: lowering
`lambda_recon` removes gradient without rebalancing anything, because reconstruction was not
dominating it to begin with.

`scripts/diagnostics/loss_gradient_balance.py`; logs `/tmp/f149_{hex,b1}.log`. Batch 24, one robot at
a time -- two concurrent runs OOM an 11 GB card.

---


### F150. Three wiring checks before the rebuild. All pass, and the third sets the bar

**Nothing here trains anything.** Each check is cheap now and expensive after five hours of
pretraining on a mis-wired objective. `scripts/diagnostics/check_actswm_wiring.py`.

## 1. The null-action contrast is wired to the stance, and the two rollouts do differ

Rolled from the same `e_t`, once on the real action's latent and once on the **standing stance's**
(F148). The stance is a real posture -- `|a|` up to 2.545 on the insect and 3.364 on the B1 -- and
not the zero vector, which would be a fall.

| body | h=1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| hexapod, real-vs-null over real-vs-`e_t` | 0.078 | 0.135 | 0.180 | 0.211 | **0.236** |
| B1 | 0.089 | 0.117 | 0.124 | 0.135 | **0.146** |

**The separation exists and is small**: after five steps the two rollouts differ by 24% of how far
the real one has travelled from its starting frame on the insect, and 15% on the B1. **That is the
quantity the separation term has to grow**, and it is non-zero, so the term has something to push
on. Had it been ~0 the hinge would have had nothing to work with; had the null been the zero vector
it would have been large and meaningless.

## 2. The frozen readout takes no gradient and passes gradient through

A new module, `[e_t, e_t+1] -> action`, 725,778 parameters, randomly initialised with
`requires_grad = False` throughout. One backward pass through it into a trainable forward model:

| | |
|---|---|
| gradient reaching the readout's own parameters | **0 tensors** -- correct |
| gradient reaching the forward model | **312 tensors, total norm 27.56** -- flows, correct |
| the ITM | **not instantiated for training, untouched** |

**The mechanism is confirmed**: the readout cannot move, so the only way to lower a loss measured
through it is to make the transitions themselves separable. That is precisely what a *learnable*
readout would avoid doing, and what F139 and F143 measured our contrastive repair avoiding -- the
sensitivity lived in the projector's region and on one body only.

## 3. The starting sensitivity, against the same null

`/mean-z` here is the rolled error on the real action over the rolled error on the null action.
**1.0 means the action changed nothing.**

| body | h=1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| hexapod | **0.956** | 0.923 | 0.894 | 0.871 | 0.863 |
| B1 | **0.936** | 0.921 | 0.937 | 0.922 | 0.925 |

**This is the number the rebuild must move, and it is worse than the earlier ones suggest.** F140's
0.476 and F149's more balanced gradient both come from *adapted* checkpoints or from a mean-latent
baseline. Measured on the **pretrain** against the **stance null** -- the exact contrast the hinge
will use -- the action buys 4-14% at every horizon on both robots.

**And the horizon direction differs between the bodies.** On the insect sensitivity improves with
horizon, 0.956 to 0.863: a longer rollout accumulates the difference. On the B1 it is flat at
0.92-0.94. **A rollout-level term is the right shape for the insect's curve and an open question for
the B1's**, and that difference is worth watching in the rebuilt run rather than pooling the two.

**Verdict: all three pass. The wiring is correct and the pre-rebuild baselines are recorded**
(F149 for gradient share, this entry for sensitivity). What remains before the run is the lambda
values, and the full pretrain has not been started.

Log `/tmp/f150.log`.

---


### F151. The short run says stop: the null cannot be an action at pretraining, and 400 steps made prediction worse

**A 400-step continuation of the three-channel checkpoint under the ActSWM objective**, ActSWM's
Table 3 values where they are not physics-dependent -- `alpha_pred` 1.0, `lambda_hinge` 0.5, margin
0.3, `lambda_readout` 1.0, readout hidden 512, frozen. `wm/actswm_short.py`. **This is not a
pretrain and does not stand in for one.**

## The two settings that were not copied, and why

**K = 5, not their 12.** F140 and F150 measure the rolled prediction crossing "worse than a frozen
frame" by five steps on this checkpoint. **Hinging separation at horizons where the prediction is
already broken trains on noise.** Ready to drop to 3.

**H stays at 1, not their 32.** Checked in the code rather than assumed: `wm/train.py:140` drives
the forward model as `FTM(e_t, z)` -- **a single frame, 256 tokens, no temporal stack.** 32 is not a
hyperparameter here, it is a different architecture, and nothing in F138 or F150 implicates context
length.

**`lambda_sig = 0.09` was not applied.** Which term it scales is not identifiable from the number
alone, and guessing would put a value in the log under a name that may not mean what the paper
means.

## The blocker the check was run to find

**The null cannot be an *action* at pretraining time.** F148 defined it as each body's standing
stance, which is an action -- and **pretraining has no action projector**. `wm/train.py` drives the
forward model with the ITM's `z`; the projector is fitted *afterwards*, against a frozen ITM
(`wm/fit_projector`), and the pretrain checkpoint has no `projector` key at all. A hinge built on
`proj(stance)` therefore has **no path back to `z`**, and that is exactly what the first run
measured:

| first attempt, null = `proj(stance)` | \|dL/dz\| hinge | \|dL/dz\| readout |
|---|---|---|
| hexapod | **0.00000** | **0.00000** |
| B1 | **0.00000** | **0.00000** |

**The pretraining-compatible null is `ITM(e_t, e_t)`** -- the latent of *nothing happened*. It needs
no projector, carries the same meaning as the stance, and restores the gradient path. **F148's
stance null remains correct for anything measured through the projector**, which is every
adaptation-stage and control-time measurement; the two nulls are for two different stages and both
belong in the write-up.

## What 400 steps did, with the corrected null

| | \|dL/dz\| pred | hinge | readout | `/mean-z` | prediction / frozen frame |
|---|---|---|---|---|---|
| hexapod, before | 0.00778 | 0.00108 | 0.00224 | 1.046 | **0.658** |
| hexapod, after | 0.00393 | **0.00006** | 0.00217 | 1.096 | **0.798** |
| B1, before | 0.00443 | 0.00044 | 0.00367 | 1.010 | **0.714** |
| B1, after | 0.01856 | 0.00390 | 0.00582 | 1.033 | **0.904** |

**Gradient does enter the latent from both new terms** -- that half of the wiring works.

**Prediction degrades on both bodies**: 0.658 to 0.798 on the insect and 0.714 to **0.904** on the
B1, which is nearly the frozen-frame baseline. **`/mean-z` moves the wrong way**, 1.046 to 1.096 and
1.010 to 1.033.

**And the separation term is unstable.** Across the run it read 0.019, 0.137, **0.496**, then
**0.008** -- it overshoots the 0.3 margin, the hinge switches off, and it collapses back. The
hexapod's hinge gradient falls to 0.00006 by the end, which is a term that has stopped acting.

## Reading, and what to change before five hours are spent

**Do not start the full pretrain.** Three things point the same way:

1. **The margin is too aggressive for our scale.** 0.3 in cosine distance is a large ask when the
   real and null rollouts start 0.078-0.089 apart (F150). The term either saturates or overshoots,
   and both were observed inside 400 steps.
2. **K = 5 is still too long.** Prediction degraded most on the B1, whose sensitivity is flat with
   horizon (F150) -- the body with least to gain from a long rollout lost the most accuracy.
3. **`/mean-z` against the latent null starts at ~1.0**, not the 0.86-0.96 F150 read against the
   stance null. **The two nulls are not interchangeable as a baseline**, and the rebuild's
   improvement must be quoted against whichever null it trains on.

**Proposed next step, one short run each, not the pretrain**: margin 0.1 with K = 3, and a
prediction-anchored variant that raises `alpha_pred`. **The check did its job** -- a five-hour run
under these settings would have produced a worse forward model and a hinge that had switched itself
off.

Logs `/tmp/f151.log` (the zero-gradient first attempt) and `/tmp/f151b.log`.

---


### F152. The margin was the problem: at 0.1 the hinge stops oscillating and stays alive. Prediction still slips.

**Two 400-step runs, `K = 3` and margin `0.1`, everything else as F151.** The diagnosis there was
that a 0.3 cosine margin is 3-4x the separation the model starts with, so the hinge bought
separation by breaking prediction, overshot, switched itself off, and collapsed --
0.019, 0.137, **0.496**, 0.008, with its gradient dying to 0.00006.

**A** = margin 0.1, K 3, `alpha_pred` 1.0.
**B** = the same with `alpha_pred` **3.0**, to anchor prediction while separation grows. *(The
instruction called this the "readout predicted-transition weight"; it is applied here to the
prediction term, which is what anchors accuracy. Flagged rather than silently reinterpreted.)*

## The oscillation is gone

Separation over training, eight points across the run, margin 0.1:

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| **A**, hexapod | 0.005 | 0.007 | 0.006 | 0.012 | 0.018 | 0.020 | 0.013 | **0.022** |
| **A**, B1 | 0.005 | 0.004 | 0.007 | 0.005 | 0.009 | 0.010 | 0.007 | **0.011** |
| **B**, hexapod | 0.006 | 0.007 | 0.005 | 0.009 | 0.013 | 0.007 | 0.011 | **0.012** |
| **B**, B1 | 0.004 | 0.004 | 0.005 | 0.005 | 0.010 | 0.007 | 0.006 | **0.008** |

**Rising and holding, on both bodies, in both runs.** Nothing approaches the margin, so the hinge
never switches off, and there is no 0.496-then-0.008 collapse anywhere. **The margin was the cause,
and 0.1 fixes it.**

## The hinge stays alive

| \|dL/dz\| hinge | before | after |
|---|---|---|
| F151, margin 0.3, hexapod | 0.00108 | **0.00006** -- dead |
| **A**, hexapod | 0.00089 | **0.00130** |
| **A**, B1 | 0.00038 | **0.00058** |
| B, hexapod | 0.00089 | 0.00061 |
| B, B1 | 0.00038 | **0.00016** |

**A keeps the hinge acting on both bodies and B weakens it**, which is what tripling the prediction
weight should do: the hinge is now a smaller share of a larger loss.

## Prediction still slips, and B slips less

| prediction error / frozen frame | before | after |
|---|---|---|
| F151, margin 0.3, hexapod | 0.658 | 0.798 |
| F151, margin 0.3, B1 | 0.714 | **0.904** |
| **A**, hexapod | 0.642 | 0.779 |
| **A**, B1 | 0.686 | 0.807 |
| **B**, hexapod | 0.642 | **0.721** |
| **B**, B1 | 0.686 | **0.760** |

**B halves the damage** -- the B1 goes to 0.760 where margin 0.3 sent it to 0.904 -- **and neither
run holds prediction where it started.** `/mean-z` does not move usefully in either: A reads
1.026 / 1.051, B reads 1.020 / 0.984, from 1.100 / 0.994.

## Verdict against the acceptance criteria

| | A | B |
|---|---|---|
| hinge gradient alive | **yes** | weakened on the B1 |
| separation stable, not oscillating | **yes** | **yes** |
| prediction not degrading | **no**, 0.642 to 0.779 | **closest**, 0.642 to 0.721 |

**Two of three pass and the third does not, in either run. Still no full pretrain.**

**What the numbers say to try next, in order of what they support.** Separation rises steadily and
stays an order of magnitude below the margin, so **the hinge is not saturating -- it is being
outpaced**: 400 steps is short, and a longer run at margin 0.1 with `alpha_pred` 3 is the setting
the evidence points at rather than another margin change. **And the prediction slip may be an
artefact of continuing from a converged checkpoint** at a learning rate meant for pretraining --
which a from-scratch run would not have, and which this test cannot distinguish. That limit is worth
stating before anyone reads 0.721 as a property of the objective.

**Locked, and not re-litigated here.** `H = 1` is architectural (`wm/train.py:140`), `lambda_sig` is
LeWM-specific and stays out, and the two nulls belong to two stages -- `ITM(e_t, e_t)` in
pretraining, `proj(stance)` at projector evaluation. **Never compare `/mean-z` across them**: the
same checkpoint reads ~1.0 against one and 0.86-0.96 against the other.

Logs `/tmp/f152_{A,B}.log`.

---


### F153. The ActSWM rebuild: settings, pre-registration, and how criterion 3 becomes measurable (run pending)

**The objective is implemented and the run is handed to com7.** `wm/train.py` gained the two terms
behind `lambda_hinge` and `lambda_readout`, both zero by default so every run before 2026-08-31
reproduces unchanged. Smoke-tested end to end: one epoch on both embodiments starts, the config
records every new field, nothing crashes. `scripts/com7_pretrain_actswm.sh`.

## The settings, each with the measurement behind it

| | value | why not ActSWM's |
|---|---|---|
| margin | **0.1** | at 0.3 the term overshoots, switches off and collapses -- 0.019, 0.137, 0.496, 0.008, gradient to 0.00006 (F151). At 0.1 separation rises and holds on both bodies (F152) |
| `K` | **3** | the rolled prediction crosses "worse than a frozen frame" by five steps (F140, F150); hinging past that trains on noise |
| `H` | **1** | `wm/train.py` conditions the forward model on one frame. 32 is a different architecture |
| `lambda_recon` | **3.0** | tripling the prediction weight halved the accuracy loss in the short run (F152, B) |
| `lambda_hinge` / `lambda_readout` | 0.5 / 1.0 | adopted |
| `lambda_sig` | **not used** | SigReg is LeWM-specific; this is V-JEPA2 and the term is not guessed in |
| null | **`ITM(e_t, e_t)`** | the stance is an *action* and pretraining has no projector; a hinge on `proj(stance)` puts zero gradient into `z` (F151) |

## Why criterion 3 is only now measurable

**The short runs could not answer it and no number of them could.** They fine-tune a converged
checkpoint at a pretraining learning rate, so prediction can move for reasons that have nothing to
do with the hinge, and "the hinge breaks prediction" cannot be separated from "continuing to train
this checkpoint moves it". **The confound is structural to fine-tuning and disappears only when the
hinge is present from step 0**, which is this run.

## Pre-registered, before the run

| outcome | reading |
|---|---|
| prediction healthy **and** `/mean-z` down **and** separation holding | **the rebuild works** -- Context Collapse addressed |
| prediction degrades toward the frozen frame, hinge present from step 0 | **now real, not an artefact**: the term is trading accuracy for separation. **Stop**, lower `lambda_hinge` or the margin |
| separation oscillates again | the margin is still too high for a from-scratch run; lower it or schedule it |

**Two rules on how the numbers are read, both from findings that were misread once already.**

**Per body, never pooled.** The insect's sensitivity improves with horizon and the B1's is flat
(F150), and the B1 lost accuracy fastest in every short run. **The contribution -- does
action-sensitivity survive across disjoint embodiments -- is a claim about both bodies separately**,
and pooling would hide the case that fails.

**`/mean-z` only against the null it was trained on.** The pretraining null is `ITM(e_t, e_t)` and
the baseline against it is about **1.0** (F151). The projector-stage null is the stance and its
baseline is **0.86-0.96** (F150). **The same checkpoint reads both, and cross-comparing them would
manufacture an improvement or hide one.**

**Baselines this run is measured against**: gradient share F149, sensitivity F150 and F151,
calibration F134, selection F136.

**Status: not run.** Five hours on com7.

---


### F154. The rebuild fails its pre-registered criterion: one-step prediction survives and everything past it explodes

**Fifty epochs from scratch on com7, both embodiments, the F153 settings.** `beh12_actswm`, commit
5306889, 01:44 to 08:07 -- six and a half hours. Judged against the pre-registration written before
it ran.

## Rollout prediction, against a frozen frame, per body

| horizon | hexapod (held-out body) | B1 | the old pretrain, hexapod |
|---|---|---|---|
| 1 | **0.743** | **0.707** | 0.732 |
| 2 | **3.331** | **3.693** | -- |
| 3 | 3.208 | 4.075 | -- |
| 5 | 3.139 | 4.007 | 0.764 |
| 10 | -- | -- | 0.978 |
| divergence horizon | **2** | **2** | > 10 |

**One step is untouched and everything after it explodes.** The old model was worse than a frozen
frame only by ten steps; this one is **three to four times worse by two**. `moves` -- predicted
displacement over actual -- goes from 0.41 to **3.2-4.0**: the rollout does not collapse toward
standing still, it flies apart.

**This is the pre-registered failure case and then some.** The reading fixed in F153 was "if
prediction degrades toward the frozen frame with the hinge present from step 0, that is real: stop."
It did not degrade toward the frozen frame; it went past it by a factor of four. **Stop is the
answer, and the settings are not the fix.**

## And the mechanism is legible

**The prediction loss is one-step and the hinge is not.** `L_recon` scores `FTM(e_t, z)` against
`e_{t+1}` and nothing else; the hinge acts at steps 1, 2 and 3. **At steps 2 and 3 the only force
on the rollout is the one telling it to move away from the null**, and the cheapest way to satisfy
that is to diverge. Step 1 is anchored by the reconstruction term and is exactly where the damage
stops -- 0.743 and 0.707, indistinguishable from the 0.732 baseline.

**So the missing piece is a multi-step prediction anchor**, not a smaller `lambda_hinge`. A hinge
over K steps needs a prediction loss over the same K steps, or it is unopposed everywhere the
reconstruction term does not reach. **Lowering the margin cannot fix an unopposed term; it only
slows it.**

## What survived

**The shared coordinate is untouched.** All three channels on both robots: hexapod +0.99 / +0.97 /
+0.98 at 1.0-1.2x compression, B1 +0.98 / +0.97 / +0.95. **The body head does not depend on the
rollout**, and F134's result stands on this checkpoint too.

**Gradient balance moved as intended**, against F149's baseline:

| | recon | motion | body |
|---|---|---|---|
| hexapod, F149 | 40.6% | 22.8% | 36.6% |
| hexapod, now | **54.4%** | 14.3% | 31.3% |
| B1, F149 | 22.3% | 46.3% | 31.4% |
| B1, now | **42.7%** | 28.7% | 28.6% |

Reconstruction's share rose because `lambda_recon` was tripled -- that part did what it was set to
do. It did not save the rollout, because tripling a **one-step** term cannot anchor steps two and
three.

## `/mean-z` cannot be read from this run

Hexapod reads 0.94 at one step and 1.04-1.11 beyond it; the B1 reads 0.96 then 0.67-0.74. **The B1's
apparent improvement is measured on a rollout whose error is four times a frozen frame's** -- a ratio
between two broken quantities. **No sensitivity claim can be made from a model that has stopped
predicting**, in either direction.

## A monitoring failure that is mine

**The run has no separation curve.** The epoch line prints `hinge` and `sep` only in a change made
*after* commit 5306889 was pushed, so six and a half hours produced no record of whether separation
rose, held, or oscillated. **The one diagnostic F151 and F152 established as essential was absent
from the run that needed it most.** The print is in the working tree; it must be committed before
anything else is launched.

## What the next run has to change

1. **A multi-step prediction loss over the same K the hinge spans.** This is the finding, and it is
   not a hyperparameter.
2. Only then revisit `lambda_hinge` and the margin.
3. **Commit the monitoring print first.**

Log kept from the com7 session; run `wm/runs/beh12_actswm/`.

---


### F155. One-step prediction does not need the action, so no weighting can make the hinge work

**F154's two candidate explanations, separated by one measurement.** Either `lambda_recon` is out of
balance -- F151 at 1.0 let the hinge break prediction, F153 at 3.0 smothered it, and the answer is
in between -- or the one-step task is solvable without reading `z` at all, in which case the balance
does not exist. `scripts/diagnostics/action_necessity.py` swaps what drives the forward model at a
single step while holding the state fixed: the real latent, F151's null `ITM(e_t, e_t)`, a real
latent from a random position in the same clip, the clip's mean, and no motion at all.

**The deciding number is `null/real`.**

| | real | null | shuffled | mean | hold still | **null/real** | shuffled/real | real beats null |
|---|---|---|---|---|---|---|---|---|
| **insect**, held-out body, all | 1.5511 | 1.5949 | 1.6531 | 1.6275 | 2.1055 | **1.028** | 1.066 | 70.2% |
| — sideways | 1.4458 | 1.5150 | 1.5591 | 1.5681 | 1.9057 | 1.048 | 1.078 | 86.9% |
| — speed | 1.5965 | 1.6425 | 1.7151 | 1.6637 | 2.2517 | 1.029 | 1.074 | 72.5% |
| — **turning** | 1.6132 | 1.6295 | 1.6878 | 1.6525 | 2.1660 | **1.010** | 1.046 | **51.4%** |
| **B1**, all | 1.3645 | 1.3994 | 1.4224 | 1.4021 | 1.9425 | **1.026** | 1.042 | 82.3% |
| — sideways | 1.3246 | 1.3475 | 1.3556 | 1.3477 | 1.8282 | 1.017 | 1.023 | 74.6% |
| — speed | 1.3799 | 1.4272 | 1.4572 | 1.4261 | 2.0343 | 1.034 | 1.056 | 85.4% |
| — turning | 1.3891 | 1.4233 | 1.4545 | 1.4325 | 1.9650 | 1.025 | 1.047 | 86.9% |

**Knowing the action that actually happened is worth under three percent, on both robots.** Telling
the model that *nothing* happened costs 2.6-2.8%; handing it an unrelated action from elsewhere in
the same clip costs 4-7%. Meanwhile predicting no motion at all costs **36-42%** -- the model is
predicting a great deal, and almost none of it from `z`.

**Turning on the insect is the sharpest reading and the worst.** Real beats null on **51.4%** of
samples, a coin flip, on the one behaviour family where the action is least redundant with the
frame. F119's within-clip objection does not apply here: this is not a periodic gait whose phase the
frame fixes, it is a direction.

## So the F154 diagnosis was wrong and the answer is not a lambda

**There is no balance point between 1.0 and 3.0.** The hinge asks the forward model to make the real
rollout differ from the null rollout, and F154 showed it will pay for that with long-horizon
accuracy. This finding says why it has to: **at one step the two rollouts are already nearly the
same thing**, so any separation must be manufactured somewhere the prediction loss cannot see. That
is exactly the divergence at h >= 2 that F154 measured. **Lowering `lambda_recon` will not create a
signal that is not in the task; it will only let the hinge break prediction faster.**

**This is F54 and F149's last line, now measured rather than suspected.** The prediction target is
the next embedding at the recorded frame rate, and one frame of legged locomotion is close enough to
the previous frame that the state carries it. **The objective has to be changed before any
separation term is worth weighting.**

## What has to change, and it is the target not the weight

The action must be made *necessary* to the prediction, which means widening what one step spans:

1. **Frameskip, or a longer lag between `e_t` and the target.** The cheapest test and it changes only
   the data pipeline. `action_lag` already exists.
2. **A multi-step prediction target**, which F154 independently asked for -- it anchors the horizon
   where the hinge acts *and* it makes each supervised step depend on more action.
3. Only after `null/real` is meaningfully above 1.0 does a hinge, or its weight, mean anything.

**Measure `null/real` first on any candidate change**, before training anything with a hinge in it.
It is minutes on a cached checkpoint and it is the precondition the whole ActSWM objective rests on.

## Scope, stated plainly

**Both rows are the pre-rebuild checkpoint `beh12_hex-b1_body3`, which is what exists locally.** The
rebuild `beh12_actswm` lives on com7 and has not been measured this way;
`scripts/com7_action_necessity.sh` runs both checkpoints there. The conclusion is expected to hold
because it is a property of the *task*, not of a set of weights -- but until that script has run,
**the numbers above are the F138 model's and must be quoted as such.**

*(Numbering: F154 is the F153 run outcome, logged before this diagnostic was requested. This is the
diagnostic that decides what follows it.)*

---


### F156. Lag 3 is the best target spacing on both robots, and the effect is real but small

**F155 said the target has to span more time; this measures how much.** At lag `k` the action is
`ITM(e_t, e_t+k)`, the target is `e_t+k`, and the forward model is applied once -- exactly what
training with frameskip `k` would ask of it. Two columns decide: `null/real` says whether the action
would *matter* at that spacing, `real/hold` says whether anything is predictable there at all.
Checkpoint `beh12_hex-b1_body3`, held-out body for the insect.

| lag | null/real, insect | real beats null | real/hold | null/real, B1 | real beats null | real/hold |
|---|---|---|---|---|---|---|
| 1 | 1.028 | 70.2% | 0.737 | 1.026 | 82.3% | 0.702 |
| 2 | 1.066 | 93.1% | 0.677 | 1.043 | 91.1% | 0.666 |
| **3** | **1.078** | **94.6%** | 0.676 | **1.053** | **93.4%** | 0.655 |
| 5 | 1.072 | 91.5% | 0.693 | 1.049 | 93.1% | 0.661 |

**Lag 3 is the peak on both robots and prediction never stops working.** `real/hold` sits between
0.65 and 0.74 at every lag -- there is no divergence horizon problem here, because this is one
application of the forward model rather than a roll. The failure mode F154 found does not appear in
this measurement and cannot be read from it.

**The sign becomes reliable, and that is the largest change.** Insect turning -- F155's coin flip at
51.4% -- reads **92.9%** at lag 3. Every family on both bodies moves from 70-87% to 88-97%. **A
weighting can act on a signal that is right nine times in ten; it cannot act on one that is right
half the time.**

**But the magnitude stays small.** The best `null/real` anywhere is 1.078, so the action still
accounts for under eight percent of prediction error on the insect and under six on the B1. F155's
number was 1.03; lag 3 roughly doubles it and no more.

## The structure the sweep exposes

**`real/hold` is flat across lags.** Holding still costs 2.11 at lag 1 and 3.06 at lag 3; the model
costs 1.55 and 2.07. **It removes about a third of the motion error at every spacing**, and the
action's share of that third does not grow with the gap. Widening the target makes every quantity
larger in proportion rather than changing what the model is using -- which is what a
representation-level limit looks like: V-JEPA2 embeddings three frames apart are still dominated by
what is static in the scene.

## The verdict sits between the two branches that were pre-registered

**A viable `k` exists in the sense that matters for weighting** -- lag 3 is a strict improvement over
lag 1 on both bodies and in every behaviour family, with prediction fully intact. **It is not a
"clearly beats" in magnitude.** Reporting it as one would be the overclaim.

**And the measurement is a lower bound.** Both the ITM and the forward model were fitted at lag 1, so
every lag-3 row is off-distribution for them. **A model actually trained at lag 3 could show more --
or could show that 1.078 was all there was.** That distinction cannot be settled by measuring an
existing checkpoint, and it is what a full pretrain would be betting six hours on.

## The cheap way to settle it before betting

**Pretrain short at lag 3 with the hinge off** -- `lambda_hinge = 0`, `lambda_readout = 0`, few
epochs -- and re-run this diagnostic on the result. That removes the off-distribution confound and
answers the only open question: does `null/real` at lag 3 rise above 1.078 once the model is trained
there? **If it does, the full pretrain has a target worth weighting against. If it stays near 1.08,
the problem is the representation and no frameskip reaches it**, and the next move is the encoder or
the prediction target itself rather than another objective.

**No full pretrain, and no lambda tuning, until that short run reports.** Tuning a weight against a
six-percent signal is what F153 already spent six hours proving does not work.

*(Both tables are the F138 checkpoint; `beh12_actswm` is on com7 and `scripts/com7_action_necessity.sh`
now sweeps lags there too.)*

---


### F157. The short lag-3 pretrain with the hinge off: pre-registered, wiring confirmed, awaiting com7

**One confound stands between F156 and a decision.** `null/real` peaked at 1.078 on the insect and
1.053 on the B1 at lag 3, but both the ITM and the forward model producing those rows were fitted at
lag 1, so every lag-3 measurement is off-distribution for them and 1.078 is a **lower bound**. It
cannot be used to reject lag 3. Training briefly at lag 3 removes that confound and nothing else.

`scripts/com7_pretrain_lag3.sh`, run `wm/runs/beh12_lag3_nohinge`, about ninety minutes.

| setting | value | why |
|---|---|---|
| `frame_stride` | **3** | the peak on both robots and in every behaviour family (F156) |
| `action_chunk` | 0, meaning follow the stride | **widening the stride without this is measurably wrong** -- `z` summarises k steps while `L_motion` scores one, which took validation motion from 0.218 to 0.928 (F88) |
| `lambda_hinge` | **0** | no separation pressure, so no explosion risk. F154's divergence came from a hinge acting where the prediction loss could not see |
| `lambda_readout` | 0 | same |
| `lambda_recon` | **1.0** | back to default; the 3.0 of F153 existed only to counter the hinge |
| `epochs` | 10 | enough to fit the new target, far short of a model worth keeping |

**This is not a candidate model and nothing may be tuned against it.**

## The decision, fixed before the run

| `null/real` at lag 3, trained there | what it means | what happens next |
|---|---|---|
| **clearly above 1.078** | the lag-3 task genuinely needs the action once a model learns it | add the hinge and pretrain fully, **then** tune `lambda` |
| **stays near 1.08** | the limit is the representation, not the objective -- no frameskip reaches it | the next move is the **encoder or the prediction target itself, not another objective term**. Report plainly and stop |

Read per body and per family. **Insect turning is the sharpest cell**: 51.4% at lag 1 (F155), 92.9%
at lag 3 off-distribution (F156).

## Wiring confirmed before spending the time

A one-epoch smoke run locally, `wm/runs/smoke_lag3`, checks the thing F88 says to check. The epoch
line prints `heads {'hexapod': 18, 'b1': 12}`, which is the **action dimension**, not the head width,
and reading the width off the checkpoint gives `heads.hexapod.3.weight (54, 512)` and
`heads.b1.3.weight (36, 512)` -- **18x3 and 12x3, so the command window did widen with the stride**
and the two halves of the objective describe the same interval. The saved config confirms
`frame_stride: 3`, `action_chunk: 0`, `lambda_hinge: 0.0`, `lambda_recon: 1.0`.

**One thing to watch in the full ten epochs.** Validation motion read **1.1396** after that single
epoch, which is around the level of predicting the training mean -- exactly the number F88 uses to
say the decoder has stopped working. After one epoch it means nothing; **if it is still near 1.0 at
epoch 10 the run has hit F88's failure and its `null/real` cannot be read**, because a decoder that
has stopped working says nothing about whether the action was available.

## Outcome: negative, and it removes the frameskip route entirely

Ten epochs, com7, commit 9fecfb3. **Training converged and the decoder lived.** Validation motion
sat at F88's dead-decoder level for five epochs -- 1.1346, 1.1072, 1.0244, 0.9989, 0.9727 -- then
broke through to **0.5360** by epoch 10. Probe 0.974, body 0.1436. The reading is valid.

**`null/real` at lag 3, measured on a model actually trained at lag 3: 1.032.** F156's
off-distribution lower bound was 1.078. **It went down, not up.**

| | measured at lag 3 | real beats null |
|---|---|---|
| lag-1 model, off-distribution (F156) | 1.078 | 94.6% |
| **lag-3 model, on-distribution** | **1.032** | **82.9%** |

## The comparison that settles it: each model at its own training lag

| model | its lag | null/real | real beats null | real/hold |
|---|---|---|---|---|
| `beh12_hex-b1_body3` | 1 | 1.028 | 70.2% | 0.737 |
| `beh12_lag3_nohinge` | 3 | **1.032** | 82.9% | 0.706 |

**Tripling the spacing bought four tenths of a percentage point.** The action is worth 2.8% at lag 1
and 3.2% at lag 3, and each model cuts about 30% of the motion error at its own spacing. **The
structure is identical; only the units changed.**

## F156's lag sweep was measuring model mismatch, not task structure

This is the finding to carry forward. **The apparent climb from 1.028 to 1.078 was the
off-distribution penalty, not the action becoming necessary.** Forcing a lag-1 model to predict
three frames ahead degrades it, and it degrades the *null* rollout slightly more than the real one,
which widens the ratio. Train the model where it is measured and the gap collapses back to 1.03.

The same artefact appears in this run with the sign reversed: the lag-3 model reads **1.005 at lag
1** with real beating null on 54.4% of samples -- worse than chance-adjacent, because lag 1 is now
*its* off-distribution -- and climbs to 1.047 at lag 5, which is off-distribution in the other
direction. **Every lag sweep on a fixed checkpoint measures how far the checkpoint is from home. It
is not a read on the task, and F156's recommendation of lag 3 rested on exactly that mistake.**

## The pre-registered branch, taken

**`null/real` did not rise clearly above 1.078; it fell to 1.032.** By the decision fixed before the
run: **the limit is the representation, not the objective. No frameskip reaches it. The next move is
the encoder or the prediction target itself, and not another objective term.**

**No hinge, no full pretrain, no `lambda` tuning.** F153 spent six hours proving a weight cannot
create a signal that is not there; F156 proposed frameskip as the way to create it; this run shows
frameskip does not create it either. **Three findings now point at the same place: V-JEPA2
embeddings of this scene are dominated by what does not move, at every spacing tested.**

## The one alternative explanation, stated so it is not hidden

**This run had 10 epochs; the lag-1 model it is compared against had 50.** Action-sensitivity could
in principle still be growing. Two things weigh against it: the run's own losses had largely
flattened by epoch 10 (val 2.7527 to 2.7373), and **F153's 50-epoch run did not produce
action-sensitivity either**. It is a real limit on the comparison and not, on this evidence, a
reason to spend another six hours.

## What survived, again

**Probe 0.974 and body 0.1436 at ten epochs.** The shared body coordinate forms under a tripled
stride, a rewired command window and a fresh initialisation. **It has now survived every
intervention that broke the prediction path**, which is worth stating in the deck: it does not
depend on the world model working.

---


### F158. What the action-blind prediction misses is mostly not the action

**Before any representation-level rebuild, what are the three percent?** F157 closed the
objective-level path; this asks whether the residual left by an action-blind prediction is
structured signal worth predicting directly, or noise.

    r  =  e_t+k  -  FTM(e_t, ITM(e_t, e_t))

`scripts/diagnostics/residual_structure.py`, checkpoint `beh12_hex-b1_body3` at its own lag of 1,
ridge solved in the dual so the full 360,448-dimensional embedding is used exactly, **split by clip**.

| | action R2, insect | action R2, B1 |
|---|---|---|
| **`r`**, the null residual | **0.786** | **0.274** |
| **`e_t` alone**, the control | 0.777 | 0.161 |
| `e_t+k - e_t`, raw difference | 0.641 | 0.063 |

**On the insect the residual carries nine thousandths more action than the bare frame does.** The
frame shows the pose, the pose is the command, and `r` adds essentially nothing to it. Per family it
is worse than the frame on two of three -- speed 0.774 against 0.805, turning 0.859 against 0.931 --
and better only on sideways, 0.730 against 0.610.

**On the B1 there is something, and it is small.** `r` reads 0.274 against the frame's 0.161, and
turning goes from **-0.166** -- the frame is worse than predicting the mean -- to 0.114. So a little
action information does live in the residual on the quadruped that the frame does not hold. **An R2
of 0.27 is not a signal to build a research direction on.**

## The pair test says `r` is not determined by the action

Taking each held-out transition and its nearest neighbour in action space:

| | matched-action pairs | random pairs | ratio |
|---|---|---|---|
| insect | 794,469 | 1,123,699 | **0.707** |
| B1 | 909,641 | 971,224 | **0.937** |

**Two states issued the same command leave almost as different a residual as two unrelated ones.**
If `r` were a function of the action this ratio would go to zero; on the B1 it is 0.94.

## Two things about the method that must be said

**The family-accuracy column is uninformative and I am not reporting it as evidence.** `r` separates
the families at 1.000 on the insect and 0.869 on the B1 -- but `e_t` alone reads 1.000 and 0.893.
The frame identifies the behaviour by itself, so the test cannot distinguish a structured residual
from a residual that merely remembers which clip it came from.

**`r` contains `e_t` implicitly**, since `FTM(e_t, z_null)` is a function of `e_t`. A probe on `r`
therefore has access to the frame whether or not the residual carries anything, which is exactly why
the control row is the measurement and the `r` row alone is not.

**And the pair test is loose**: nearest-neighbour in a continuous command space is not an identical
action, which biases the ratio toward 1. It is conservative in the direction of calling `r` noise,
so the insect's 0.707 should not be read as precise.

## The branch this takes

**`r` is not action-recoverable beyond what the frame already gives, so predicting `r` directly is
not a viable next direction.** Fitting a target that is 94% not-determined by the action on the B1,
and that adds one percent over the raw frame on the insect, reproduces the problem in a new place.

**The remaining path is the representation itself -- the encoder, or what it is asked to predict.**
That is the question to bring to the advisor. Four findings now converge: a weight cannot create the
signal (F153), a wider spacing does not create it (F157), and what the action-blind model misses is
not mostly the action (F158). **V-JEPA2 embeddings of this scene are dominated by what does not
move, and no term added on top of them reaches around that.**

**Measured on the lag-1 checkpoint only.** `beh12_lag3_nohinge` is on com7; the same script with
`--lag 3` would confirm there, and F156's lesson applies -- measure a checkpoint at its own stride.

---


### F159. A single frame reads the insect's command almost as well as a pair, and does not on the B1

**The paper measurement, in Yeom et al.'s metric.** They report V-JEPA inverse-dynamics R2 of 0.40
frozen and 0.85 with an ID head, and note that CALVIN's static tabletop lets per-frame appearance
substitute for temporal context. This asks how far that goes in legged locomotion, where a gait
makes the pose a near-complete statement of the command. Ridge in the dual on the full
360,448-dimensional embedding, **split by clip**, `beh12_c08f09t09_flat` held out for the insect.

| features | insect | vs single | B1 | vs single |
|---|---|---|---|---|
| **`e_t`, one frame** | **0.779** | -- | **0.161** | -- |
| `[e_t, e_t+1]`, the pair, their setup | 0.867 | +0.088 | **0.342** | +0.182 |
| `[e_t, e_t+3]`, a wider pair | **0.887** | +0.108 | 0.328 | +0.167 |

**On the insect one frame recovers 88% of what a pair recovers. On the B1 it recovers 47%.**

| family | insect, single | insect, pair | B1, single | B1, pair |
|---|---|---|---|---|
| sideways | 0.609 | 0.846 | 0.317 | 0.456 |
| speed | 0.814 | 0.861 | 0.263 | 0.482 |
| **turning** | **0.931** | 0.957 | **-0.166** | 0.120 |

**Insect turning is where the pose is nearly the whole command**: 0.931 from one frame, and the
transition adds 0.026. **B1 turning is the opposite extreme**: one frame is *worse than predicting
the mean* at -0.166, and even a pair only reaches 0.120.

## What this does and does not license

**The dissociation is real and it is sharpest exactly where it matters.** On the insect the command
is recoverable at R2 0.887 and still contributes under 3% of one-step forward prediction error
(F155). **Inverse-recoverable does not imply forward-necessary**, and that is now quantitative on
the body where recoverability is highest -- which is the strong form of the claim, not the weak one.

**But "the action is read almost completely from a single frame" is an insect statement.** The
quadruped reads 0.161 from a frame and 0.342 from a pair, so on the B1 the transition roughly
doubles what is recoverable and the total stays low. **The contribution statement in
`direction_plan.md` has been annotated to scope that sentence to the insect**; written unscoped it
is an overclaim a reviewer with our own table would catch.

**And the two failures are different failures.** The insect fails as "the pose already says it"; the
B1 fails as "not much is recoverable at all". F158 found the same asymmetry from the other side --
`e_t` predicting the action at 0.777 on the insect and 0.161 on the B1. Both break an ActSWM-style
objective, and only the first is the periodicity story.

## The cross-paper number is not a like-for-like comparison

Our single-frame 0.779 on the insect exceeds their frozen 0.40, and it is tempting to say
periodicity makes the problem twice as easy. **Different data, different action space, different
head, different split.** The comparison is worth one sentence of framing and must not be presented
as a controlled contrast.

## Scope

**This measurement is nearly checkpoint-independent** -- the features are frozen encoder embeddings
and only the action normalisation comes from the loaded config, which is a strength: it is a
property of V-JEPA2 on this data, not of anything we trained. Confirming on `beh12_lag3_nohinge` at
its own stride changes the *target* (a 3-step command window, 54 and 36 wide) rather than the
features, and is the F156-lesson check to run on com7:

    .venv/bin/python3 scripts/diagnostics/inverse_dynamics_r2.py \
        --ckpt wm/runs/beh12_lag3_nohinge/best.pt --data data/allocentric/beh12_c08f09t09_flat \
        --embodiment hexapod --pair_lags 3

---


### F160. The `lambda_body = 0` control: pre-registered, measurement path pinned

**The question a reviewer will ask that we cannot currently answer with a measurement**: did our own
coordinate objective cause the action-insensitivity? Every `null/real` number in the chain, back to
F87, comes from a checkpoint with the body term active at 0.5, taking 31-37% of the gradient (F149).

**What this control is not.** It is **not** a test of whether the body term explains the lead
finding. F159 measures action recoverability from **frozen V-JEPA2 embeddings**, which `lambda_body`
never touches, so the single-frame pose-is-the-command result is encoder-level and stands whatever
this run reports. **The narrower question, which is real and unmeasured**: does the body term
*additionally* starve `z` of joint-level detail by shaping it into three coarse body-motion numbers,
compounding the encoder limit?

`scripts/com7_lambda_body0_control.sh`, run `wm/runs/beh12_lag3_nobody`, about ninety minutes.

## Held identical to `beh12_lag3_nohinge`, one variable changed

| | value | |
|---|---|---|
| `frame_stride` / `action_chunk` | 3 / 0 | same as baseline |
| `lambda_hinge` / `lambda_readout` | 0 / 0 | same |
| `lambda_recon` | 1.0 | same |
| `epochs` | 10 | same |
| `body_dim` / `body_channels` | 3 / `0 1 2` | **kept** |
| **`lambda_body`** | **0.0** | **the single variable** |

**Keeping `body_dim` and zeroing only the weight is deliberate.** The modules, the latent width and
the saved buffers stay identical, so the only difference between the two checkpoints is the
gradient. Dropping `body_dim` instead would change the architecture and make the comparison a
different experiment.

## The measurement path, pinned before the run

F140 was withdrawn for comparing across two latent paths, so this is stated in advance and checked
against the code rather than asserted. **`action_necessity.py` constructs the ITM and the FTM and
nothing else** -- grepping it for `body_head`, `MotionDecoder` or `md` returns no lines -- and
`gather` reads only `embedding_offsets` from the checkpoint, which `offset_for` supplies regardless
of the body term. So on a `lambda_body = 0` checkpoint the path is byte-identical to the baseline's:

| | |
|---|---|
| null | `ITM(e_t, e_t)`, the same definition as F151, F155 and F157 |
| lag | **3**, this checkpoint's own stride, as F156 requires |
| data | the same held-out body for the insect, the same clips for the B1 |

**The script re-runs the baseline at lag 3 in the same log**, so the two tables cannot be mismatched
by transcription.

## The reading, fixed in advance

| `null/real` | what it means |
|---|---|
| **stays near 1.03** | the body term is innocent; the insensitivity is purely encoder-level. **This strengthens the paper** -- our own objective is not the cause, V-JEPA2 is |
| **rises** | the body term was stripping action detail from `z`. F159 still stands, but our pipeline **compounded** the encoder limit, and what we claim about `z` specifically has to change |

## One gap to close before the comparison can be read

**The B1 baseline number does not exist in any log I hold.** The `beh12_lag3_nohinge` run printed
`action_necessity` for the insect (1.032 at lag 3, real beating null on 82.9%) and then for the B1,
but only the insect table was transcribed back. **The insect comparison can be made today; the B1
comparison needs either the rest of that log or the re-run this script performs.**

## Outcome: the body term is innocent, and it was mildly helping

Ten epochs, com7, commit 3d86ebc, `wm/runs/beh12_lag3_nobody`. Both tables in one log, both at lag 3,
the same measurement path.

| at lag 3 | insect `null/real` | real beats null | B1 `null/real` | real beats null |
|---|---|---|---|---|
| baseline, `lambda_body` 0.5 | **1.032** | 82.9% | **1.009** | 86.1% |
| control, `lambda_body` 0 | **1.015** | 74.4% | **1.008** | 83.2% |

**Removing the body term made the action matter less, not more.** On the insect `null/real` fell from
1.032 to 1.015 and the sign became less reliable, 82.9% to 74.4%. On the B1 nothing moved, 1.009 to
1.008.

**The pre-registered reading was "stays near 1.03 means the body term is innocent".** It did not
rise; it fell. **The coordinate objective was not stripping action detail from `z` -- it was a small
positive contributor to what little action-sensitivity exists.**

**This strengthens the paper on the exact question a reviewer will ask.** "Did your coordinate
objective cause the action-insensitivity?" is now answerable with a controlled measurement, and the
answer is no: with the term removed entirely, the insensitivity is *worse*. The cause is upstream, in
the encoder (F159), which is what the contribution claims.

## Two things this log settles that were open

**The B1 lag-3 baseline, missing from the F157 transcript, is 1.009.** That completes F157's table
and makes its conclusion stronger, not weaker:

| body | at lag 1, lag-1 model | at lag 3, lag-3 model |
|---|---|---|
| insect | 1.028 | 1.032 |
| **B1** | 1.026 | **1.009** |

**On the quadruped, training at the wider spacing made the action matter *less*.** F157 concluded
frameskip does not reach the limit from the insect's +0.004; the B1 reads **-0.017**. Nothing about
the frameskip route survives this.

**And the morphology probe moved a long way.** Body identity is decodable from `z` at **0.974** with
the body term and **0.732** without it, against a chance of 0.5 on two bodies. **The coordinate
objective makes `z` substantially more body-identifiable.** Validation motion also ended slightly
worse without the term, 0.6134 against 0.5360.

## The scoping correction this forces, applied 2026-08-31

**`z` is not body-blind, and every document that said so has been changed.** A probe reading 0.974
is the first thing a reviewer checks against the word "morphology-agnostic", and it contradicts it
outright.

**The agnosticism lives in the shared body-motion coordinate -- forward, lateral, yaw, the same
physical quantities on both robots (F136) -- not in `z`.** The wording is now "`z` maps to a shared
body-motion coordinate" everywhere.

**This is a scoping correction, not a retraction, and it makes the architecture more coherent rather
than less.** The decoders are per-body (F129), so `z` *has* to carry body identity for them to work;
and removing that identity adversarially made transfer 1.2x worse while the frame was used twice as
much (F24). **A body-blind `z` was never what this pipeline wanted.**

Changed in `doc/direction_plan.md` (three places), `doc/START_HERE.md`, and
`report/presentation_proposal.md` (three places). **`report/update_slide.md` needed no change** -- it
never made the claim, and its line "remove body identity from `z` by adversarial training -> transfer
1.2x worse" already says the opposite. **Earlier findings are left as written**, this file being the
history; F107's "which is what a morphology-agnostic action is supposed to mean" is superseded by
this section.

**Do not conflate the two measurements.** `probe` is `MorphProbe(z)` classifying *which body*; it is
**not** the three-channel body-motion coordinate. The `lambda_body = 0` control has no trained body
head at all, so no body-motion number exists for it and none is quoted anywhere.

**Read `probe` correctly**: it is `MorphProbe(z)` classifying *which body*, not the three-channel
body-motion coordinate. The control has no trained body head at all, so no body-motion number exists
for it and none is quoted.

---


### F161. Novelty positioning: three named neighbours, and the scoped claim that survives all three

**The scoped contribution, recorded verbatim as owned by the user. The target-level clause was narrowed on 2026-08-31, after the flag below; this is the canonical wording:**

> In periodic visual locomotion, the joint action is inverse-recoverable from a SINGLE frame (F159:
> insect R2 0.78, turning 0.93) because gait phase makes the pose encode the command -- so the
> action is forward-redundant (F158: <3% of prediction error). This closes the loop between two
> prior observations -- adjacent-frame redundancy (AHA-WAM) and the action-invariant teacher-forcing
> solution (UWM-JEPA) -- with a measured mechanism, and shows the objective-level fix-family fails
> on it (ActSWM hinge, F153/F157) and that the residual-target route is closed (F158), since the
> residual of an action-blind prediction carries no action beyond the frame -- because there is no
> action-dependent forward signal to recover, only a redundant one.

**The unscoped version -- "visual world models cannot be action-conditioned in locomotion" -- must
never be written.** A reviewer holding AHA-WAM or UWM-JEPA would sink it, because both papers state
parts of it already.

## The three neighbours and what separates us from each

| paper | what they have | what we add |
|---|---|---|
| **AHA-WAM** (2606.09811) | adjacent frames are "redundant for control" / "weakly informative" -- **the same intuition**, used as architectural motivation for an asynchronous horizon split | they assume it as a design premise; **we measure it, and attribute it to gait periodicity** -- not measured, not tied to periodicity, and not locomotion in their work |
| **UWM-JEPA** (2605.25313, Sec 4, "Counterfactual Actions Require Counterfactual Targets") | names the structure behind F153/F157: a teacher-forced target contains the action's effect, admitting an **action-invariant solution** that ignores the action channel. Their fix is counterfactual targets | **we show the action-invariant solution is near-optimal in locomotion** because the pose already encodes the command (F159), so a target fix has nothing better to converge to |
| **Yeom et al.** (2606.07687) | V-JEPA carries inverse-recoverable action signal; CALVIN's static scene lets per-frame appearance substitute for temporal context | **inverse-recoverable is not forward-necessary**, measured; periodicity is the severe, structural form of their CALVIN exception |

**This is a measurement-and-mechanism contribution positioned between named neighbours**, not a
discovery of the phenomenon. Written that way it is defensible; written as a discovery it is not.

## Why the target-level clause was narrowed, kept so the reasoning is not lost

**As first written -- "target-level (counterfactual/residual targets, F158)" -- it overstated what
F158 ran.** F158 measured
whether the **residual** of an action-blind prediction is action-recoverable, and found it is not:
`r` adds 0.009 R2 over the bare frame on the insect, and matched-action pairs differ by 0.94 of a
random pair on the B1. That is strong evidence against a **residual** target.

**It is not a test of UWM-JEPA's actual fix.** A counterfactual target is constructed for an action
that was *not* taken -- it is a different object from the residual of the action that *was* taken,
and we have never trained against one. **As written, the sentence invites "you did not run our
method", which is the single most damaging thing a reviewer of that paper can say.**

**Narrowing was chosen over running the arm.** The clause now claims exactly what was measured,
and UWM-JEPA's counterfactual target is neither run nor claimed to fail. The load-bearing claim is
the measured mechanism -- single-frame recoverability plus forward redundancy -- and it never
depended on defeating every proposed fix. **If that counterfactual arm is ever trained, this is the
sentence that widens.**

## What the failure chain now supports, stated at its true strength

**Measured, on both bodies:** the action contributes under 3% of one-step prediction error (F155);
no weighting recovers it (F153); no frameskip creates it (F157); the action-blind residual does not
carry it (F158); and on the insect a single frame reads the command at R2 0.779 against a pair's
0.887 (F159).

**Not measured:** a counterfactual-target arm, and an encoder other than V-JEPA2. Neither is claimed.

---


### F162. Direction B killed in an afternoon: a motion representation does not break the redundancy and destroys the transfer

**The de-risking test, run before any encoder rebuild.** Direction B needs a representation that is
**both** action-necessary and body-shared, and those may trade off. The cheapest possible candidate
was used deliberately: **the temporal difference in V-JEPA2 space**, `m_t = e_t+1 - e_t`. No
training, no new encoder, no world model anywhere in `scripts/diagnostics/motion_rep_check.py`. If
even this cannot break the redundancy, a learned motion encoder is a much longer bet on the same
hope.

**Two of three pre-registered outcomes killed B. Both fired.**

## Part 1: the redundancy does not break

Action R2, split by clip, held-out insect body.

| body | family | `e_t` appearance | `m_t` motion | `[m_t, m_t+1]` pair |
|---|---|---|---|---|
| **insect** | all | 0.779 | **0.646** | 0.650 |
| | sideways | 0.610 | 0.451 | 0.412 |
| | speed | 0.812 | 0.721 | 0.736 |
| | **turning** | **0.931** | **0.783** | 0.826 |
| **B1** | all | 0.161 | **0.063** | 0.100 |
| | sideways | 0.317 | 0.063 | 0.089 |
| | speed | 0.263 | 0.193 | 0.233 |
| | turning | -0.166 | -0.076 | -0.029 |

**A single motion snapshot still reads the insect's command at R2 0.646, and its turn at 0.783.**
The drop from appearance is 0.133, seventeen percent of the way down from a number that was already
the problem. **Differencing does not remove the pose from the representation** -- an
appearance-organised space differenced is still appearance-organised.

**Adding a second motion snapshot buys 0.004.** The pair is not the fix either.

## Part 2: and the transfer is destroyed

Correlation of the fitted body-motion coordinate, **fitted on the insect and applied to the B1
without refitting**.

| representation | test | forward | lateral | yaw |
|---|---|---|---|---|
| `e_t` appearance | insect held-out | **0.98** | 0.95 | 0.96 |
| `e_t` appearance | **B1 unrefitted** | 0.63 | 0.43 | 0.07 |
| `m_t` motion | insect held-out | **0.60** | 0.52 | 0.84 |
| `m_t` motion | **B1 unrefitted** | **-0.05** | **0.07** | **0.00** |

**Differencing halves the coordinate within the insect and annihilates it across bodies.** Forward
transfer goes from 0.63 to -0.05; every channel lands at zero. **This is the fatal trade-off in its
plainest form: the transform gave up the one result that works and did not buy the property it was
meant to buy.**

## What this table is not

**The appearance rows are not F136.** This is a raw ridge from frozen embeddings to the coordinate,
with nothing trained; F136 fits through the model's body head from `z` and reports far higher
transfer. **The valid comparison here is appearance against motion under one identical protocol**,
which is what the test needed. **Do not quote 0.63 / 0.43 / 0.07 as the project's transfer numbers.**

## The decision

**Direction B is dead on the cheapest candidate, and dead on both criteria rather than one.** A
motion transform that halved the working result while leaving the action 83% as readable is not a
promising start that needs a better encoder; it is the trade-off the test was designed to expose,
arriving immediately.

**This does not prove no encoder could do it** -- a learned motion encoder is a different object
from a difference of an appearance encoder, and that is worth one honest sentence rather than a
claim. **It does mean the cheap evidence points the other way, and Direction A -- finish and write
up what is measured -- is what the afternoon bought.**

Reproduce: `scripts/diagnostics/motion_rep_check.py --ckpt wm/runs/beh12_hex-b1_body3/best.pt`.
The Part 2 label columns ran together in the original log; only the formatting was changed after.

---


### F163. Is the redundancy in the encoder or in the data? Pre-registered, collector built

**The hypothesis nothing so far has tested.** F159 measured the insect's command as readable from a
single frame at R2 0.779; F162 tested a motion transform of the **representation** and found it
neither breaks that redundancy nor survives it. **Neither touched the data.** On a steady gait the
pose fixes the phase and the phase predicts the next frame, so the action may be redundant **by
rhythm** rather than by anything about V-JEPA2.

**This changes the data, not the representation.** Blinding or corrupting `e_t` is the failed
"cripple the frame" trap; both arms here are collected identically and the encoder is untouched.
The only difference is what the robot was commanded to do.

## What was built

**`--cmd_noise` in `sim/collect/collect_ik.py`**, with `--noise_tau` and `--noise_seed`. Temporally
correlated noise added to the **final** joint command -- after the heading and oscillator branches,
before `actions.append` -- so **the logged `a_t` is the perturbed command that was actually sent**.
If it were not, the measurement would be void: the probe would be fitting commands the robot never
executed.

**Correlated, not white.** White noise at 20 Hz is filtered out by the joint controller and never
reaches the pose, which would produce a "perturbed" set that is visually identical to the clean one
and quietly answer nothing. `tau` is 5 steps.

**`scripts/dataset/collect_offrhythm.sh`** collects both arms through `collect_beh12.py`, so the
twelve conditions and the `condition` field come out in the form every measurement script already
reads.

| arm | `--cmd_noise` | |
|---|---|---|
| clean | 0.0 | the steady gait every set so far has used |
| noisy | 0.05 | off-rhythm exploration |

**Both arms in one sitting, and the clean arm is the control -- not `data/allocentric/beh12_c10f10t10_flat`.**
That set was collected on another day, and a camera or lighting difference between arms would be
indistinguishable from the effect being measured.

## The reading, fixed before collection

| single-frame action R2 on the noisy arm | what it means |
|---|---|
| **drops, and the pair-minus-single gap opens** | the redundancy is a **data** property. Direction B becomes "train on exploratory data" -- cheap, achievable, and it also unblocks the parked bootstrap problem |
| **stays high** | the redundancy is encoder-level, and the hard encoder rebuild is the only remaining B |

**The gap matters as much as the drop.** A single-frame R2 that falls while the pair R2 falls with
it means the perturbation destroyed readability rather than moving it into the transition, and that
is not the result being looked for.

## Two things that must happen before any number is quoted

**Watch the videos.** Six defects in this project were found by looking and none by the tables
passing at the time. **A robot that has fallen over is not off-rhythm walking**, and 0.05 rad of
correlated noise on a stick insect is not obviously safe. `walk_check` prints a verdict per clip.

**Check separability.** The script runs it on both arms. If the noisy arm no longer resolves its
twelve conditions, its clips are not twelve behaviours any more and the per-family rows mean
nothing -- a real risk, since the same recipe that "passed separability" once hid a body whose
sideways conditions had the wrong sign.

## First pass, free: the speed ramps we already collected say no

**`data/allocentric/fwd_hex7speed` already contains off-rhythm clips** -- 24 of its 98 are speed-ramped,
0.72->1.1 and 1.1->0.72, collected through `--speed_end` for F60. **24 ramped against 24 constant,
matched per body**, same measurement as F159:

| arm | `e_t` single frame | `[e_t, e_t+1]` pair | `[e_t, e_t+3]` pair | **pair - single** |
|---|---|---|---|---|
| constant speed | 0.739 | 0.759 | 0.764 | **+0.024** |
| **ramped speed** | **0.861** | 0.877 | 0.885 | **+0.024** |

**The transition adds the same +0.024 whether the speed is steady or ramping**, and single-frame
readability went *up* on the ramped clips.

**Only the gap is comparable between the arms.** The two have different action distributions and
R2 is relative to each arm's own mean, so 0.861 against 0.739 partly reflects that the ramped clips
sweep a continuous range while the constant arm has five discrete speeds. **The pair-minus-single
gap is what was pre-registered, and it did not move at all.**

**But a speed ramp is a weak perturbation, and F60 says exactly why**: it retimes the whole foot
path, and "all legs share the time map, so inter-leg phase is untouched" -- verified at 0.056 lag
against the constant case's 0.061. **It varies the rate of the rhythm rather than taking the robot
off it.** The pose still fixes the phase.

**So this lowers the prior on the data hypothesis without settling it, and it makes the `--cmd_noise`
collection more worth running rather than less**: the cheap already-collected form of "non-steady
gait" demonstrably does not move the number, so the noise arm is testing something genuinely
different. It also gives that arm a calibrated null -- **+0.024 from this project, on this
measurement** -- to be judged against.

Arms built as count-matched symlink sets; `wm/adapt3.gather` gained a `condition` fallback to
`behavior`, since the Stage 1 `fwd_*` sets never went through the twelve-condition flatten and
reading them used to raise.

## Outcome: the redundancy is a data property, and the pre-registered YES fired

Both arms collected in one sitting, `c10f10t10`, twelve conditions, two repeats, 24 clips each.

| | clean, `cmd_noise` 0 | **noisy, `cmd_noise` 0.05** |
|---|---|---|
| `e_t` single frame | **0.764** | **0.196** |
| `[e_t, e_t+1]` pair | 0.839 | 0.351 |
| `[e_t, e_t+3]` pair | 0.848 | 0.369 |
| **pair - single** | **+0.084** | **+0.173** |

**Single-frame readability collapses by 0.568 and the gap doubles.** Both pre-registered conditions
fired: the single frame drops *and* the pair-minus-single gap opens, which is the outcome that says
the redundancy is a property of the **data** rather than of V-JEPA2. The ridge on the noisy arm's
single frame also chose the **largest penalty in the grid**, 1e4, which is what a probe does when
there is little there to fit.

Per family, the sideways conditions go from 0.763 on one frame to **-0.047** -- worse than
predicting the mean -- while a pair recovers 0.305.

**And it is the opposite of the speed ramp.** Ramping the rate left the gap at +0.024 in both arms
(above). Perturbing the command doubles it. **The distinction is exactly F60's: a ramp shares one
time map across all legs and preserves inter-leg phase, so the pose still fixes the phase. Noise
does not.**

## The caveat that decides how far this can be pushed

**Part of the drop is tautological and must be said before the number is used.** The injected noise
is *exogenous*: by construction it is not determined by the current state, so no single frame could
predict it, and only the transition reveals it. That the gap opens is therefore partly a property of
having added random commands, not proof that a *useful* control signal became readable.

**What it does establish** is that the mechanism is data-side: when the command carries a component
the gait phase does not determine, the transition becomes the only way to read it, and the encoder
was never the obstacle to that. **What it does not establish** is that a world model trained on this
data would be useful, because the component it would learn to read is noise.

## And the noise level used is too high to train on

**The pre-registered separability check fails on the noisy arm**, and the clean arm passes it:

| arm | pairs closer than 2x their own spread | closest pair |
|---|---|---|
| clean | **0 of 66** | 2.4x |
| **noisy** | **24 of 66** | **0.3x** |

At 0.05 rad the noise is comparable to the differences *between conditions*: `speed_c7.1` against
`speed_c8.8` falls to 0.3x, and the semantic check reports turn levels disagreeing on direction.
**The twelve conditions are no longer twelve behaviours**, so the per-family rows above are
indicative only, and this exact set is not a training set.

**That makes the next step a sweep, not a rebuild.** There is a trade-off curve between breaking the
redundancy and keeping the behaviours distinguishable, and the useful noise level is the largest one
that still passes separability. **0.05 is above it. Nothing says 0.02 is.**

## Status of Direction B

**It is no longer dead, and the reason it looked dead was that F162 and the ramp both tested things
that leave the phase intact.** The remaining question is narrow and cheap: **does a noise level that
preserves separability still open the gap?** If yes, Direction B is "train on mildly exploratory
data" -- collection, not an encoder rebuild. If no, the two requirements are incompatible on this
data and the encoder is back to being the only route.

Bugs fixed in passing: `separability()` referenced an `args` it never received and crashed before
printing its verdict, so the clean arm's result had been invisible; `collect_offrhythm.sh` called it
with the wrong argument form.

---


### F164. The noise sweep: pre-registered, separability gates R2 at every level

**F163 established the mechanism and overshot the dose.** 0.05 rad took single-frame action R2 from
0.764 to 0.196 and doubled the pair-minus-single gap from +0.084 to +0.173 -- and destroyed the
dataset, 24 of 66 condition pairs below 2x their own spread against 0 of 66 clean. **The open
question is whether a level exists that breaks the redundancy while the twelve behaviours survive.**

`scripts/dataset/noise_sweep.sh`, levels **0.0, 0.02, 0.03**, correlated at tau 5, all three arms in
one sitting.

**A clean arm is recollected even though F163 has one.** If it reproduces 0.764 and +0.084 that also
retires the worry about comparing across sittings; if it does not, that is worth knowing before
anything else in the log is read.

## Separability gates R2, and the script prints it first

**A level that opens the gap by dissolving the behaviours has found nothing** -- it has collected one
noisy condition twelve times. The R2 table is only meaningful where the separability block passed.
This ordering is the whole design of the script.

## What a pass proves, fixed before the run

| outcome | what it means |
|---|---|
| **passes separability AND opens the gap** | **the mechanism is confirmed**: the redundancy is data-side and breakable with the behaviours intact. Direction B becomes "train on perturbed data" -- collection, not an encoder rebuild |
| **no level does both** | random perturbation cannot break the redundancy without destroying the behaviours. **B's cheap data form is dead**, and only meaningful perturbation or an encoder rebuild remain |

## What a pass does NOT prove, and this is the sentence that must survive into any slide

**That a world model trained on it would be useful.** The injected noise is **random by
construction**, so a model learns to read **jitter**, and jitter does not help control. The gap
opening is **partly tautological**: a component was added that only the transition can carry, and no
single frame could ever have predicted it.

**Mechanism is not usefulness.** **No result from this sweep licenses any claim about a working
controller**, and a sweep that passes must be written up as "phase-breaking is possible without
destroying the behaviours", never as "the world model can now be action-conditioned".

## The standing next question, which this run is only a proxy for

**The tautology-free version is *meaningful* phase-breaking motion** -- real turns, transitions and
speed-breaks that are not phase-locked, where the action carries intent rather than jitter. **This
sweep is the fast test of whether phase-breaking works at all**, before anyone invests in collecting
that. It is deliberately the cheap proxy and must not be mistaken for the thing itself.

**Nulls to read against:** clean, +0.084 gap at 0.764 single-frame; 0.05 rad, +0.173 at 0.196; and
the speed ramp's +0.024 in both of its arms, which is what a perturbation that preserves inter-leg
phase looks like.

## Outcome: 0.02 is the level, and the trade-off is graded rather than a threshold

Three arms, one sitting, 2026-08-31. **Separability read first, as designed.**

| `cmd_noise` | pairs < 2x spread | semantic check | single-frame R2 | pair R2 | **gap** |
|---|---|---|---|---|---|
| **0.0** | **1 of 66** | **FAILS** -- `speed_c8.8` weaker than `speed_c8.15`, 0.196 against 0.206 | 0.729 | 0.832 | **+0.102** |
| **0.02** | 11 of 66 | **passes** -- every condition moves as its name says, every level exceeds the one below | **0.383** | 0.581 | **+0.198** |
| 0.03 | 19 of 66 | FAILS -- `side_R_lvl0` lateral +0.019, should be negative | 0.344 | 0.516 | +0.172 |
| 0.05 (F163) | 24 of 66 | FAILS | 0.196 | 0.369 | +0.173 |

**0.02 is the best level on every axis that was pre-registered.** It is the **only** arm in the sweep
that passes the semantic check -- including the clean arm, which does not -- and its gap is the
**largest in the sweep at +0.198**, while single-frame readability halves from 0.729 to 0.383.
**Beyond it the gap stops growing and separability keeps degrading**, which is the shape of a
trade-off with an optimum rather than a monotone one.

## Honest reading of "passes"

**No level passes cleanly, and 0.02 does not either.** It takes unresolved condition pairs from 1 in
66 to 11 in 66, with the closest turn levels at 0.9x their combined spread. **That is a tenfold
degradation, not a free lunch**, and calling it a pass without that sentence would be the overclaim.

**The clean arm failing the semantic check matters too.** `speed_c8.8` against `speed_c8.15` is a 5%
inversion that exists with no noise at all, so the recipe is marginal on this body regardless, and
**0.02 passing that check is partly luck.** The separability *counts* are the sturdier column.

**And the separability check was built for the planner's candidate library, not for pretraining
data.** A planner cannot choose between two conditions it cannot resolve; a *pretraining* set mainly
needs varied actions and varied behaviour, and 11 unresolved pairs out of 66 is not obviously fatal
there. **That is an argument for using 0.02 as pretraining data and against using it as a candidate
library**, and it should not be used to wave away the gate.

## The clean arm reproduced, and it sets the noise floor on this measurement

F163's clean arm read **0.764 single-frame and +0.084 gap**; this sitting's reads **0.729 and
+0.102**. **So sitting-to-sitting variation is about 0.035 in single-frame R2 and 0.018 in the gap**,
which retires the cross-sitting worry and, more usefully, says what counts as a real difference.
**0.02's +0.198 against a null near +0.09 is roughly double, and far outside that.**

## What this confirms, exactly

**Mechanism confirmed.** The redundancy is data-side and can be broken while the twelve behaviours
remain individually correct in direction and ordering. **Direction B's cheap form is alive**:
perturbed collection, not an encoder rebuild.

## What it does not confirm, and this sentence goes on the slide unedited

**That a world model trained on this would be useful.** The injected noise is **random by
construction**, so a model learns to read **jitter**, and jitter does not help control. The gap
opening is **partly tautological**: a component was added that only the transition can carry, and no
single frame could ever have predicted it. **Mechanism is not usefulness**, and nothing here
licenses a claim about a working controller.

**Written up, this result is "phase-breaking is possible without destroying the behaviours" and
never "the world model can now be action-conditioned".**

## The standing next question, unchanged and now worth its cost

The tautology-free version is **meaningful phase-breaking motion** -- real turns, transitions and
speed-breaks that are not phase-locked, where the action carries intent rather than jitter. **This
sweep was the cheap proxy for whether phase-breaking works at all. It does.** That is what makes
collecting the meaningful version worth the investment, and it is still a separate thing that has
not been measured.

Data: `data/allocentric/beh12_c10f10t10_sweepn{00,002,003}_flat`; log `sweep_log.txt`.

---


### F165. Meaningful phase-breaking: pre-registered, and the usefulness control that noise failed

**Three attempts at breaking the redundancy, and this is the only one where the broken rhythm
carries something a controller could use.**

| | what it did | gap |
|---|---|---|
| F163, speed ramp | retimed the whole foot path; **inter-leg phase preserved** | +0.024, unchanged |
| F164, random `cmd_noise` | broke the phase with **jitter** | **+0.198**, and a model would learn to read noise |
| **F165, this** | breaks the phase because a **command changed** -- a stop, a speed break, a turn beginning | ? |

**This is also the exploratory set the parked bootstrap problem needs. One collection, two uses.**

## What was built

**`--spin_schedule` in `sim/collect/collect_ik.py`**, sharing `--schedule`'s `value@fraction`
grammar. **It is what makes a turn an event rather than a label**: with a constant `--spin`, every
frame of a turning clip says "turning clip" and **no frame is the one where the turn begins**, so a
probe reads the label off any frame and is never asked to predict the change.

Onsets sit at different fractions across conditions and in both directions, so "later in the clip"
and "turning" are not the same variable -- the reasoning F60 used when it collected ramps both ways.
Guarded against the heading loop, which regenerates frames from a single stored spin and would
discard the schedule silently.

`scripts/dataset/collect_intent.sh`: twelve conditions, four per axis, 24 clips.

| axis | conditions |
|---|---|
| speed | stop mid-clip, slow->fast, fast->slow, break then run faster |
| turn | onset late, onset early, a turn pulse, **a turn that reverses** |
| side | sideways left and right, each with a stop and a speed break |

**The clean control already exists** -- `data/allocentric/beh12_c10f10t10_sweepn00_flat` from F164's sitting,
0.729 single-frame with a +0.102 gap. **Sitting-to-sitting variation is about 0.035 and 0.018
(F164), so a real effect has to clear that**, and the script re-runs every measurement on that arm
so both tables land in one log.

## The three readings, fixed before collection

**1. The gap.** Does single-frame action R2 drop and the pair-minus-single gap open, as it did for
noise? Judged against the clean null near **+0.09-0.10** and noise's **+0.198**.

**2. Intent -- and this is the result.** `scripts/diagnostics/intent_recoverability.py` fits three
targets under one protocol: the instantaneous command `a_t`, **the command change `da = a_t+1 -
a_t`**, and the family label as a classification control.

| what opens the gap | what it means |
|---|---|
| **`da`** | a **change** was recovered -- something a controller could act on. **Direction B is real**, and the next step is to train a world model on this data and test it as a teacher |
| **`family` but not `da`** | a **label** was recovered. We already have that and already rejected it. **Meaningful perturbation adds nothing beyond what we hold** |
| neither, gap flat | meaningful phase-breaking is not enough and the redundancy is deeper than expected |

**3. Separability**, as context and **not** as a result. **Meaningful turns may well make the
behaviours *more* separable, and that would look like a win for entirely the wrong reason.** The
gate exists to catch a set that has dissolved, not to be scored.

## Two claims that are not licensed by any outcome here

**Separability improving is not success.** Written into the script header so it travels with the run.

**Nothing here licenses a claim about a working controller.** That requires training a model on this
data and testing it, which is a separate step and has not been done. **`da` being recoverable says
the signal exists, not that a model will use it** -- and this project has now twice measured a
signal that a trained model then ignored.

## Outcome: VOID as run. Two thirds of the set was never phase-broken

**The collection did not do what the script said**, and the numbers below cannot be attributed to
meaningful phase-breaking.

**`--gait cpg` silently discards `--schedule`.** `schedule_path` retimes the recorded *foot path*,
but the CPG branch keeps only `cmds.mean(0)` from that path as a bias pose and regenerates the
stroke from `--cycles`. **The eight `speed_*` and `side_*` conditions therefore had no within-clip
speed change at all**, while their command lines and the log said they did.

Measured directly from the collected clips, within-clip sd of the body coordinate:

| | intent arm | clean arm | did it land? |
|---|---|---|---|
| `speed_*` forward | 0.036-0.038 | 0.033-0.047 | **no** |
| `side_*` forward | 0.0055-0.0079 | 0.0042-0.0086 | **no** |
| **`turn_*` yaw** | **0.030-0.035** | 0.013-0.020 | **yes, roughly doubled** |

**`--spin_schedule` worked**, because it is applied inside `cpg_commands`. **`--schedule` did not**,
because it is applied before it. Only 4 of 12 conditions, 8 clips, were genuinely intentful.

## What the numbers said, recorded but not to be quoted as the answer

| target | intent arm, gap | clean arm, gap |
|---|---|---|
| `a_t` instantaneous command | +0.103 | +0.102 |
| **`da` command change** | **+0.011** | **+0.017** |
| family label | +0.005, accuracy 1.000 | +0.001, accuracy 1.000 |

**Nothing moved.** The action gap is identical to the clean null, and the intent gap is *smaller*
than the clean arm's -- both are near zero because `da` is already readable from a single frame,
0.836 and 0.764.

**In the turn family alone -- the part that did work -- the `da` gap is +0.020 against the clean
arm's +0.031.** Also nothing, but on 8 clips, which is too few to conclude from.

**So the pre-registered third branch is not established.** "Meaningful phase-breaking is not enough"
would be the reading if the set had been phase-broken; two thirds of it was not, and the third that
was is underpowered.

## The fix, and why it belongs in the oscillator

`cpg_commands` now takes a per-frame `pace`, and the phase advances by `cumsum(rate) - rate` instead
of `arange(frames)`. **A stop is then a stop in the oscillator's own terms** rather than a retiming
of a path the oscillator throws away. **`cumsum(rate) - rate` with `rate` identically 1 reproduces
`arange(frames)` exactly**, so every clip collected before this is bit-identical -- checked, not
assumed. Verified on the stop schedule: the phase freezes at 26 for the whole pause.

**The re-run is the real F165.** The turn conditions will reproduce; the eight others will change.

## The lesson worth more than the run

**A flag that is accepted, printed into the log, and then ignored is worse than one that errors.**
`--schedule` appeared in every condition line of `intent_log.txt`. The only reason this was caught
is that the within-clip variance was checked against the clean arm *before* the result was written
up -- **the separability gate and the R2 tables both passed and neither could have caught it.**

---


### F166. Meaningful phase-breaking, re-run with a working scheduler and a physical gate

**F165's collection is void and its numbers are quarantined.** `--gait cpg` discarded `--schedule`,
so eight of twelve conditions carried no within-clip change while the command lines, the log,
`walk_check`, the separability gate and the R2 tables all passed them. **The third pre-registered
branch was never tested. Direction B is not dead; the experiment did not run.**

**The collector fix is verified.** `cpg_commands` takes a per-frame `pace` and advances its phase by
`cumsum(rate) - rate`, which reproduces `arange(frames)` exactly at rate 1 -- checked, so every clip
collected before this is bit-identical -- and freezes the phase for the whole pause on a stop
schedule.

## The new required gate, and it is validated against the failure it exists for

`scripts/dataset/check_within_clip_intent.py`. **The statistic is the sd of the *smoothed* channel**,
averaged over one stride, because a gait oscillates the body at stride frequency whether or not
anything was scheduled -- that oscillation is most of the raw variance and is why the void run
looked unremarkable in both directions. Smoothing removes the gait and leaves the **envelope**: what
the robot was asked to do differently, and when.

Each family is checked on the channel it means -- speed on forward, turn on yaw, sideways on lateral
-- and **the bar is the clean arm's own maximum on that channel**, not an invented threshold.

**Run against F165's void set it separates the two halves cleanly:**

| conditions | envelope sd | clean max | ratio | |
|---|---|---|---|---|
| the eight `--schedule` ones | 0.0125-0.0244 | 0.0251 / 0.0334 | **0.50-0.90** | **fail** |
| the four `--spin_schedule` ones | 0.0282-0.0313 | 0.0132 | **2.13-2.37** | pass |

**It fails exactly the eight that were broken and passes exactly the four that worked.** The gate
now runs before the measurement in `collect_intent.sh` under `set -e`, so **nothing downstream
executes on a set that was never perturbed**.

## Everything else is unchanged, because the design was not what was wrong

The same twelve conditions, the same three readings, the same pre-registration:

**1. The gap** -- single-frame action R2 against the pair, judged on F164's clean null of 0.729 and
+0.102, with sitting-to-sitting variation of about 0.035 and 0.018.

**2. Intent, and this is the result** -- `da = a_t+1 - a_t`. A gap on the **family label** is a label
we already have and already rejected; a gap on **`da`** is a change a controller could act on.

**3. Separability**, as context. **Separability improving is not success.**

The void run's clean-arm baselines, which were measured on a set that was collected correctly and do
carry over: `da` reads **0.764 single-frame with a +0.017 gap**, and the family label is recovered at
**accuracy 1.000** from a single frame -- so the control is already saturated and cannot be the thing
that moves.

**Nothing here licenses a controller claim** even if `da` opens. That needs a model trained on this
data and tested, and this project has twice measured a signal a trained model then ignored.

## The methodological rule this earns

**A flag that is accepted, echoed into the log and then ignored is worse than one that errors**, and
no statistical gate catches it. **Every future intervention gets a physical check that it reached
the robot, run before the measurement and not after the write-up.** `check_within_clip_intent.py` is
that check for anything scheduled.

## The gate did its job twice more before any number was read

**First it blocked the re-run**: 8 of 12 conditions passed, and the four stop-based ones did not.
Checking the traces rather than the statistic showed **the stops had physically happened** --
`speed_stopmid` runs 0.198 -> 0.069 -> 0.194 through the clip against the clean arm's flat
0.16-0.18. **The gate's statistic was the problem, not the collection.**

**Every clip accelerates from rest and slows at the end**, which `schedule_path`'s own docstring
names, and that uncommanded transient was most of the steady arm's envelope -- `speed_c7.1` runs
0.081 -> 0.184 -> 0.131 while doing nothing. **Leaving it in sets the bar for an intended change at
the size of an unintended one.** Trimming 15% from each end removes it from **both** arms
identically. **Decided from the traces, before any R2 was run.**

**Re-validated on the void set, which still fails on exactly the same eight conditions**, so the
gate was corrected and not weakened.

**Then it rejected two conditions on the corrected statistic**: `side_L_stopmid` and
`side_R_stopmid`, at 0.34x and 1.21x. **That is the sideways recipe, not the scheduler** -- it
under-drives this body, `side_L_lvl0` walks at Froude 0.016-0.019, and **a stop is invisible in a
robot that is barely moving.** The remaining ten clear the gate at **2.1x to 5.6x** and were measured
as `data/allocentric/beh10_c10f10t10_intent2_flat`, 20 clips, axes deliberately unbalanced.

## Outcome: meaningful phase-breaking does not open the gap

| target | intent, gated | clean null |
|---|---|---|
| `a_t` instantaneous command | **+0.061** | **+0.102** |
| **`da` command change** | **+0.028** | **+0.017** |
| family label | +0.002, accuracy 1.000 | +0.001, accuracy 1.000 |

**The action gap did not open; it is lower than the clean arm's.** The intent gap is 0.011 above the
clean arm's, **inside F164's measured sitting-to-sitting variation of 0.018**. Nothing moved.

**`da` reads 0.743 from a single frame on intentful data.** Even when the robot genuinely stops
mid-clip and reverses a turn, one frame still says what the command is about to do.

**The pre-registered third branch, and this time the intervention is verified to have landed.**

## What the four interventions together now say

| intervention | landed? | action gap |
|---|---|---|
| clean | -- | +0.102 |
| speed ramp (F163) | phase preserved | +0.024, own null +0.024 |
| **random `cmd_noise` (F164)** | yes | **+0.198** |
| **meaningful intent (F166)** | yes, gate 2.1-5.6x | **+0.061** |

**Random jitter opens the gap and meaningful intent does not**, and both broke the phase. **So what
opens the gap is not phase-breaking -- it is the command being unpredictable from the pose.** Noise
qualifies because it is exogenous. **A command a controller would actually issue does not, however
non-periodic, because the body's configuration reflects the intent.**

## This refines the contribution statement, and the refinement is stronger

`direction_plan.md` currently attributes the redundancy to gait periodicity, following F159.
**F166 says periodicity is sufficient but not necessary**: intentful, non-periodic, verified
phase-broken motion stays just as pose-readable. **The mechanism is that the joint command is a
function of the visible pose**, and rhythm is one way that happens rather than the reason.

**Stated at its true strength**: this is evidence from one body, one gait generator and smooth
commanded motion, so the pose lags the command lawfully throughout. **It does not show that no
command distribution breaks the redundancy** -- F164's does, uselessly. It does show that the
obvious useful ones do not.

## Separability, as context only

3 of 66 pairs below 2x, all among the turn conditions -- `turn_onsetlate` against `turn_pulse` at
0.7x. **They differ in *when* the turn happens and the check compares clip means**, which is exactly
what a set built around onsets should look like. **Not a result in either direction**, as
pre-registered.

**Direction B's data form is now measured and negative for the useful case.** The cheap route --
perturbed collection -- opens the gap only with jitter a model cannot use.

---


### Note (not a finding): is a counterfactual-outcome measurement testable on the data we have?

**A scoping check of what exists, run on cached embeddings. Nothing was collected, trained or
decided.**

## 1. Can we construct `(e_t, alternative_action)` pairs that are meaningful?

**Yes, trivially, and it is the uninteresting half.** Every transition has 47 other clips to borrow a
command from, and the twelve conditions span three families, so "the robot is walking forward at
`e_t`, what about the `turn_s0.56` command" is one array index. `score_by_body_motion.py` and
`rollout_fidelity.py --mean_z` already do substitutions of this shape.

**The substitution is free. What it costs nothing to build is also what makes it unfalsifiable on
its own** -- there is no recorded frame saying what *would* have happened.

## 2. Do we have the ground truth? **Effectively no, and the reason is the project's own finding**

For every transition, the distance from its pose to the **nearest pose of a different behaviour
family**, against the distance to the **nearest pose of its own family in a different clip**. Same
body, 48 clips, every fourth transition.

| | insect `c10f10t10` | B1 |
|---|---|---|
| transitions | 765 | 768 |
| nearest same-family, other-clip pose | 802,310 | 663,101 |
| nearest **different-family** pose | 1,908,712 | 939,926 |
| ratio | **2.38x** | **1.42x** |
| **transitions with a cross-family pose as close as a typical same-family one** | **4 of 765 (1%)** | **166 of 768 (22%)** |
| action difference at those pairs, sd units per joint | **0.16** | **0.25** |

**On the insect there is no ground truth at all** -- four usable transitions out of 765, which is
nothing.

**On the B1 the 22% is a mirage, and the last row is why.** Two unrelated commands differ by about
**1.41** sd units per joint. The matched-pose pairs differ by **0.25**. **Where the poses match, the
actions match too** -- these are not "the same state doing different things", they are the same state
doing nearly the same thing, found in a clip that happens to carry a different label.

**This is the whole finding chain restated in the data's own geometry.** F159 measured that the pose
determines the command; **matching on pose therefore matches on action by construction**, so the
pairs a counterfactual needs cannot exist in this data *because* of what we already showed. The
family pairings confirm it -- almost all are `speed <-> turn`, the two families whose commands
overlap most.

**To get the ground truth the data would have to be collected for it**: the same body driven from
the same pose down two different commands, which needs a resettable simulator state, not a
re-labelling of what we have.

## 3. Would it differ from F158, or collapse back into it?

**It would differ, and the difference is real -- but only if question 2 is solved first.**

| | F158 | a counterfactual-outcome measurement |
|---|---|---|
| what is held | the action that **was** taken | the state |
| what varies | nothing | the action, across a **recorded** alternative |
| the target | `r = e_t+k - FTM(e_t, null)`, a residual of the model | `e_t+k^A` against `e_t+k^B`, two **measured** futures |
| what it can show | whether the model's own miss carries the action | whether the **world** distinguishes two actions from one state |
| depends on the model | yes, `FTM` is inside the target | **no** |

**That last row is the reason it is worth wanting.** Every measurement in F153-F166 is downstream of
either a trained model or an inverse probe. A counterfactual outcome would be a statement about the
**environment**: how far apart do two commands drive the same state, measured in the encoder's
space, with no model in the loop. **That is not a version of F158.**

**Without question 2 it does collapse**, and worse than into F158: substituting an action with no
recorded alternative future leaves only `FTM`'s prediction to compare against, which is asking the
model to score its own counterfactual. **That is not a measurement.**

## Where this leaves the idea

**Testable in principle, not testable on `data/allocentric/beh12_*`.** The blocker is not effort; it is that the
pairs are absent for the reason the project has been documenting, and **the near-miss on the B1 is
the mechanism showing up in yet another form** rather than an opening.

**No decision, no rebuild, no experiment** -- this was a scoping check. Script kept in the session
scratchpad, not in `scripts/`, because it answers a question rather than measuring a quantity.

---


### F168. Scoping note: `z` is not a two-frame quantity, it is a *more* pose-readable one than the action

**The question never asked all session.** F159 measured that the joint action is single-frame
readable. **`z = ITM(e_t, e_t+1)` is built from two frames by construction**, and our pipeline then
pushes it toward the action from both sides -- the projector fits `proj(a) ~ z`, the body head fits
`z -> motion`. So a redundant `z` could mean the transition itself is pose-determined, **or only that
we forced `z` to be the action**. Reading `z` from a single frame separates those.

Same probe as F159, target swapped. `beh12_hex-b1_body3`, held-out body for the insect.

| body | target | single frame | pair | gap |
|---|---|---|---|---|
| **insect** | `a_t` action, width 18 | 0.779 | 0.867 | +0.088 |
| | **`z`, width 64** | **0.856** | 0.895 | **+0.039** |
| **B1** | `a_t` action, width 12 | 0.161 | 0.342 | +0.182 |
| | **`z`, width 64** | **0.543** | 0.649 | **+0.106** |

**On both bodies `z` is read from one frame better than the action is, and the second frame adds
less to it.** Insect 0.856 against 0.779 with the gap halved; B1 0.543 against 0.161, more than
three times as readable. **A latent whose only input is a pair of frames requires less than one
frame's worth of extra information to recover.**

## The pre-registered branch, and what it does and does not license

**The second branch fired: `z` is at least as redundant as the action, so this is closed as a
direction.** There is no "something else" hiding in `z` -- no dynamics quantity that a single pose
fails to determine.

**But "the redundancy is intrinsic to the transition" is not established by this**, and the reason
matters. **`z` being *more* readable than `a_t` is not what "we forced `z` to be the action" would
predict** -- that would make them equally readable. Something additional is going on, and the
obvious candidate is that **`z` is substantially a behaviour label**: it is 64 numbers rather than
18, it is smoother, and a single frame recovers the behaviour family at accuracy **1.000** (F166)
and this checkpoint's body identity at **0.974** (F160). **A label is the easiest thing in the world
to read off one frame.**

So the honest statement is narrow and still decisive for the purpose: **the ITM's two-frame input
does not produce a latent that needs two frames to read.** Whether that is because the transition is
pose-determined or because `z` collapsed toward identity-plus-behaviour **is not separated here**,
and would need `z` with its behaviour-identity component removed before re-asking.

**Checkpoint-dependent, unlike every action-target number in this chain.** `z` is *this*
checkpoint's latent, trained at `lambda_body` 0.5, which F160 measured as making `z` markedly more
body-identifiable. A differently trained `z` could read differently, and nothing here says
otherwise.

## Why it was worth asking anyway

**It removes the most plausible remaining escape route.** "The pipeline forced `z` to equal the
action, so of course everything downstream inherits the action's redundancy" was a live explanation
for F153 through F166 and would have pointed at a cheap fix -- change what `z` is asked to be.
**Measured, `z` is not the bottleneck**: it is already less informative-per-frame than the thing it
was suspected of being crippled into.

*(Numbering: F167 is the unnumbered counterfactual scoping note above it; this keeps the label the
request used.)*

---


### Note (not a finding): prerequisite check for a resettable counterfactual

**Can the simulator reset to an exact state, so "same pose, different action" is clean?** Nothing was
collected and the counterfactual experiment was not run.

## 1. Exact state reset

**B1 / MuJoCo: yes, bit-identical -- but only with the right state flag, and the obvious one is a
trap.**

| state saved | same state, same command, twice, over 5 steps |
|---|---|
| `mjSTATE_FULLPHYSICS` (time, qpos, qvel, act) | max pose difference **2.83e-02 rad** |
| **`mjSTATE_INTEGRATION`** (adds warmstart, ctrl, applied forces) | **0.000e+00, bit-identical** |

**`mjSTATE_FULLPHYSICS` is what "save joint positions and velocities" means and it is not enough**:
the solver's warmstart carries over, and the residual it leaves is **larger than the one-step
counterfactual signal and comparable to the five-step one**. An experiment built on it would have
measured its own reset error. `mjSTATE_INTEGRATION` reduces the noise floor to exactly zero.

**Insect / CoppeliaSim: no, on the project's own evidence.** F105 repeated one configuration five
times and got 37-71%, and states plainly that "CoppeliaSim reloads the scene for every run and its
solver and contact state do not come back identical". There is also no full dynamics-state snapshot
in the API being used. **Not re-tested here** -- that needs the GUI -- but F105 is a direct
measurement of the same property. **A counterfactual experiment would be B1-only, which is a scope
limit on any cross-embodiment claim drawn from it.**

## 2. Counterfactual divergence, one state, two commands, noise floor exactly zero

Base displacement and its size in the rendered image. **The pixel columns are arithmetic from the
documented 2.85 m field across 256 px, not a rendered measurement.**

| h | turn `wz=0.6` base | px | sideways `vy=0.4` base | px |
|---|---|---|---|---|
| 1 | 0.44 mm | 0.04 | 0.85 mm | 0.08 |
| 3 | 1.34 mm | 0.12 | 6.03 mm | 0.54 |
| 5 | 2.05 mm | 0.18 | 13.21 mm | **1.19** |
| 10 | 4.59 mm | 0.41 | 32.96 mm | **2.96** |
| 15 | 6.20 mm | 0.56 | 64.59 mm | **5.80** |
| 30 | 11.97 mm | 1.07 | 207.21 mm | **18.61** |

**The second pre-registered branch: the futures diverge cleanly, and not at one step.** Against a
noise floor of exactly zero every number is real, but at h=1 the whole difference is **four
hundredths of a pixel**. **A world model cannot use what the camera cannot resolve.**

**The horizon the collection needs is set by the slowest behaviour, and the two differ by an order
of magnitude.** Sideways reaches a pixel by h=5 and six by h=15. **Turning is still at one pixel
after thirty steps** -- it rotates the body without moving it, and the base-displacement measure
understates that, but the quaternion difference only reaches 0.138 at h=30 as well.

**One caveat about the joint-angle column, which is why displacement is quoted instead**: it is
**non-monotone** -- 0.117 rad at h=15 falling to 0.020 at h=20 -- because the two gaits drift in and
out of phase. Joint difference oscillates; base displacement integrates. **A measurement that
sampled only h=20 would have reported the turn counterfactual as nearly dead.**

## 3. Throughput

**Physics is free**: reset plus a 5-step branch is **1.5 ms**, so a pair costs 3 ms and 300 pairs of
physics is **under a second**.

**The cost is rendering**, and the pipeline already separates it: the B1 is rolled out in MuJoCo and
**replayed kinematically in CoppeliaSim from stored states** (SIM_GUIDE 4.2). **CoppeliaSim's
non-determinism therefore does not touch the counterfactual** -- the physics that must be exact
happens in MuJoCo, and the renderer only draws states it is handed. **This architecture is what
makes the experiment possible at all**, and it is the reason the insect cannot do it: its physics
*is* CoppeliaSim.

At roughly 0.17 s per rendered frame, **300 pairs at a 15-step horizon is 9,000 frames, about 25
minutes**, plus encoding. **Comfortably inside a two-month budget** -- by three orders of magnitude
on the physics and one on the rendering.

## What this settles and what it does not

**Settled**: exact reset is available on the B1 and not on the insect; the divergence is real and
noise-free; the collection is cheap; **the horizon must be at least 10-15 steps and turning may need
more than 30.**

**Not settled, and it is the next cheap thing**: whether the divergence is visible **in the
embedding**. Everything above is physical. Two frames a fifth of a pixel apart will produce nearly
identical V-JEPA2 embeddings whatever the physics says. **Rendering one branch pair and measuring
embedding distance against the encoder's own noise floor is minutes of work and would settle it**;
it has not been done, and no claim here rests on it.

---


### Note (not a finding): the insect is viable too, and the reload noise is 16x below the signal

**Answering "does the insect need one continuous session?" -- it does not, and the reason is
measurable from clips we already have.** The two repeats of every `beh12` condition are driven by
**bit-identical commands** (`max|diff| = 0.00`), so repeat-against-repeat *is* the CoppeliaSim noise
floor and nobody had ever read it.

| h | noise, same commands | signal, different commands | ratio | signal in px |
|---|---|---|---|---|
| 1 | 0.35 mm | 7.39 mm | **21.2x** | **1.08** |
| 5 | 2.25 mm | 49.8 mm | 22.1x | 7.28 |
| 15 | 7.31 mm | 117 mm | 16.0x | 17.1 |
| 30 | 18.8 mm | 236 mm | 12.5x | 34.5 |
| 60 | 47.4 mm | 502 mm | 10.6x | 73.4 |

**Signal is 10-22x noise at every horizon and already a full pixel at one step**, where the B1 sits
at 0.04.

**F105 does not say the insect is unusable, and reading it that way was my error.** Its 37-71%
spread came from **closed-loop episodes**, where a planner re-decides on a diverged state and
compounds it. Open-loop scripted commands over tens of steps drift by millimetres. **Both robots are
viable and the insect is the stronger of the two for this.**

**The catch, and it is what the confirmation run exists for.** These numbers compare different
behaviours **from the same spawn**, diverging at frame 0 with no shared momentum. **That is the easy
version and it is optimistic.** The design worth collecting shares a prefix and branches from a
shared *pose*, where both futures start with identical velocity and contact state. **The 21x must
not be quoted as if it settled that.**

---

### Confirmation run sheet: does the *real* counterfactual design separate?

`scripts/dataset/confirm_counterfactual.sh`. **Pre-registered pass mark: signal above 3x noise on
position *and* heading, at every horizon, per arm -- including turning.**

**Heading is measured, not only position, and turning is why.** A robot that rotates barely moves
its head: the B1's turn arm reads **10.2 mm at h=25** -- under a pixel -- while its heading has moved
**13.6 degrees**. **A cell that passes on position and fails on heading is still a failure**; it
means the two futures end in the same place facing different ways, which is what turning *is*. It is
also the behaviour F136 found weakest, so it is broken out separately.

## B1: PASSED, and the orientation guard fired exactly as it was built to

| arm | h=1 | h=5 | h=15 | h=25 | noise |
|---|---|---|---|---|---|
| turn, position | 0.15 mm | 1.97 mm | 5.04 mm | **9.29 mm** | **0.0000** |
| **turn, heading** | 0.196 deg | 1.70 deg | **7.00 deg** | **13.3 deg** | **0.0000** |
| side, position | 1.01 mm | 13.8 mm | 85.5 mm | **180 mm** | **0.0000** |
| side, heading | 0.026 deg | 0.126 deg | 0.252 deg | 0.725 deg | **0.0000** |

**The noise floor is exactly zero at every horizon**, so every number above is the action and
nothing else, and the ratio is unbounded rather than merely large.

**Turning survives only because heading was measured.** At h=25 the turn branch has moved the base
**9.29 mm** -- **under one pixel** at the B1's 2.85 m field over 256 px -- while its heading has
moved **13.3 degrees**. **On position alone the strongest turn counterfactual reads as nothing.**
The guard was added because the B1's earlier same-state check showed 1.07 px of displacement against
a quaternion that had moved 0.138, and it fired on the real design.

**Sideways is the mirror image**: 180 mm of displacement and 0.7 degrees of heading. **Neither
channel alone passes both arms**, which is the argument for reporting both rather than picking one.

**An off-by-one the render caught and the numbers did not.** Heading was first referenced to the
frame at index `branch`, which is already the *first divergent* frame rather than the last shared
one. Rendered over a bit-identical prefix the two compasses read **+1.4 and +1.2 degrees where they
must read the same**, which is what exposed it; every heading figure was biased by one frame. The
table above is after the fix, and `h=1` now means the first frame after the split. **The video
caught it because it showed a quantity that had to be identical and visibly was not.**

## Possible retroactive check, flagged not scheduled

**Turning has read weak throughout this project** -- F136, candidate scoring, the closed-loop family
accuracy -- and **many of those metrics measure position where turning lives in heading**. Whether
that under-measurement explains part of the weakness is a real question and a cheap one to revisit
on existing logs. **Not a task now, and no earlier number is withdrawn on the strength of a
suspicion.**

## Insect: PASSES from h=10 on both channels, and turning passes on heading from h=5

Five arms sharing a straight-walking prefix to frame 33, then branching. **`turn` and `side` share
the commanded prefix exactly** -- `max|diff| = 0.00` over frames 0-32 -- so the branch is real
rather than assumed.

| arm | h | position | noise | x | heading | noise | x |
|---|---|---|---|---|---|---|---|
| turn | 1 | 4.32 mm | 2.75 | 1.6x | 0.107 deg | 0.124 | 0.9x |
| turn | 5 | 3.50 mm | 8.97 | 0.4x | **2.570 deg** | 0.207 | **12.4x** |
| **turn** | **10** | 23.1 mm | 7.09 | **3.3x** | **8.95 deg** | 0.078 | **115x** |
| turn | 25 | 134 mm | 13.7 | 9.8x | 30.0 deg | 0.722 | 41.6x |
| side | 5 | 21.4 mm | 8.97 | 2.4x | 1.89 deg | 0.207 | 9.1x |
| **side** | **10** | 55.1 mm | 7.09 | **7.8x** | 5.77 deg | 0.078 | **74.3x** |
| side | 25 | 194 mm | 13.7 | 14.2x | 20.1 deg | 0.722 | 27.9x |

**Both arms clear 3x on both channels from h=10.** **Turning clears it on heading from h=5, at
12.4x, while its position channel is at 0.4x** -- the pre-registered case, and it happened.

**The insect's noise floor is not zero and does not integrate.** Two identical-command runs drift to
about 26 mm by frame 35 and then back to 12 mm by frame 60: it is a **bounded gait-phase offset**,
not accumulating divergence. Heading drift stays near a degree throughout.

## Two corrections the run forced, both of which changed numbers

**Position was measured absolutely while heading was measured relative to the branch.** The position
channel was therefore charged for an offset already present at the split and common to signal and
noise alike -- on the insect, 26 mm of it. Both channels now measure displacement **since the last
shared frame**. Turn at h=10 moves from 1.6x to **3.3x** and side at h=10 from 1.5x to **7.8x**. **On
a bit-identical prefix the two references coincide, so every B1 number is unchanged** -- checked, not
assumed.

**The `faster` arm does not share its prefix and is excluded.** `--schedule` retimes the foot path,
`cpg_commands` takes its bias pose from `cmds.mean(0)` of that path, so a speed schedule shifts the
whole command sequence by a constant: `max|diff|` over the supposedly shared prefix is 5.7e-02
against 0.00 for the other arms. **The bias should come from the unscheduled path** -- a real defect
in the F166 collection route, found here and not fixed silently.

**A symlink bug ate the first attempt.** `ln -sf` resolves a *relative* target from the link's own
directory, and the failure was hidden by `2>/dev/null || true`. Absolute targets and no suppression
now, with an explicit readability check.

## The clips

`results/cf_confirm/insect_forward-vs-turn.mp4` and `insect_forward-vs-side.mp4`. Shared-prefix mean
pixel difference **0.304 and 0.327 of 255**, which is the gait-phase drift and is visually nil; the
turn pair ends **42.4 degrees apart** in heading. **Each panel carries a compass drawn relative to
the last shared frame**, so turning's divergence is visible where it actually lives.

**Every run renders, and the format is a merged side-by-side.** `scripts/render/merge_counterfactual.py`
plays both arms in sync through the shared prefix, holds on the last shared frame, then splits --
border colour, caption and a frames-since-split bar all change at the branch so the moment is
unmissable. **It also prints the mean pixel difference over the shared prefix, which should be near
zero**: if the prefix visibly differs, the two runs never shared a start and every number after the
branch is measuring that instead.

**Numbers decide pass or fail; the render is the check that the counterfactual is real.**

---


### F169. The embedding gate: the counterfactual is real in the world and invisible to the encoder

**The last de-risk gate, run before the bias fix and before any collection, because everything else
is upstream of it.** `branch_divergence.py` measures millimetres and degrees. **The world model is
never shown millimetres.** F158 and F159 established that these embeddings suppress information that
is physically present, so a counterfactual 13 degrees apart in the world may be a rounding error in
`e`.

Insect, shared prefix to frame 33, same displacement-since-last-shared-frame discipline, noise floor
= two runs of identical commands through the same encoder.

| arm | h | signal | noise | ratio |
|---|---|---|---|---|
| turn | 5 | 1168 | 1236 | **0.9x** |
| turn | 10 | 1229 | 1234 | **1.0x** |
| turn | 15 | 1323 | 1366 | **1.0x** |
| **turn** | **25** | 1432 | 1338 | **1.1x** |
| side | 10 | 1298 | 1234 | 1.1x |
| side | 25 | 1498 | 1338 | 1.1x |

**Nothing clears 3x. Nothing clears 1.3x.** Two futures that are **134 mm and 30 degrees apart in
the world** are, to the encoder, as far apart as two runs of the *same* command.

## The metric is not doing this, and the check that shows why

Raw distance and cosine agree with displacement-since-reference:

| h | raw turn/noise | raw side/noise | cos(fwd,turn) | cos(fwd,repeat) |
|---|---|---|---|---|
| 5 | 0.95 | 0.97 | 0.862 | 0.847 |
| 25 | **1.19** | **1.24** | 0.767 | **0.831** |

**And the scale line is the explanation.** Consecutive frames of one clip differ by **763-1010** in
these units, against a noise floor of about 1000. **One frame of gait motion moves the embedding as
far as twenty-five steps of behavioural divergence.** The embedding's geometry is dominated by leg
configuration, and the counterfactual is buried under it.

**This is the same fact as F19 and F26 from the outside.** They decomposed `z` and found 64-89% gait
phase; this measures the encoder's metric directly and finds the same thing. **F159's "the pose
determines the command" and this are one property seen from two directions.**

## Why 3x was the right bar and not a harsh one

**L2 in embedding space is the loss the forward model is trained with.** `L_recon` scores
`FTM(e_t, z)` against `e_t+1` by MSE, so **a divergence invisible in L2 is invisible to the
objective** -- not merely hard to detect, but absent from the gradient. The bar is the training
signal itself.

## The direction dies here, and it dies at a good place

**Everything upstream passed.** Exact reset on the B1, bit-identical. Shared prefixes verified to
`max|diff| = 0.00`. Physical divergence clearing 3x on both channels from h=10 on both robots,
turning surviving on heading exactly as pre-registered. **The design was sound and the physics
cooperated. The encoder does not.**

**Two months were not spent finding this out**, which is what the gate was for.

## What is left honestly open, and neither is a plan

**Cosine shows a small real separation** -- 0.767 against 0.831 at h=25, about 0.06 -- against a
phase variance of 0.17. **It is not nothing and it is not enough**, and a *learned* metric might
weight the difference that L2 drowns. That is a different project from the one being de-risked.

**A phase-aligned comparison is the only cheap escape not tried**, and it is ill-posed here: after a
branch the two arms' gait phases drift apart, so "the same phase" stops being defined. **Naming it
is not proposing it.**

**Measured on the insect only.** The B1's frames need CoppeliaSim, and its physical divergence at
h=25 was **9.29 mm against the insect's 134 mm** -- an order of magnitude smaller. **There is no
reading of that which makes the B1 pass a gate the insect failed.**

Reproduce: `scripts/diagnostics/embedding_divergence.py`. Clips and videos in `data/allocentric/cf_confirm/` and
`results/cf_confirm/`.

---


### F170. The egocentric de-risk gate: pre-registered, cheapest possible build

**F169 killed the counterfactual direction at the encoder: two futures 134 mm and 30 degrees apart
in the world sit 1.1x the noise floor in embedding space, because one frame of gait motion moves the
embedding as far as twenty-five steps of behavioural divergence.** The egocentric idea attacks that
from the other side -- a head camera cannot show the robot its own pose, so the pose can no longer
determine the command.

**It cannot be de-risked without building something**, since a head view needs a world to look at.
The build here is the minimum that can answer the question: **four coloured, mildly textured walls
around the spawn and the existing `vjepa_cam` parented to the robot.** Nothing more, and nothing
that should survive into a real environment.

**Texture is not decoration.** `set_floor_texture.py` measured both extremes as harmful --
repeating high-contrast edges alias under sub-pixel motion (r = -0.16), featureless patches get
repurposed as ViT scratch space (r = -0.20). **Optical flow is the entire signal the view change
exists to provide, and a blank wall provides none.** Per-wall colour is there for the same reason
turning needed heading in the counterfactual gate: a uniform room gives flow but no bearing, so a
robot that has turned sees what it saw before.

## The two questions, both must pass

**Q1 -- does egocentric break the single-frame redundancy?** Third-person reads the insect's command
from one frame at **R2 0.779** against a pair's 0.887 (F159). If a head view still does, the view
change fixed nothing, and **the leak has to be named**: legs in shot, the body visible, or a room
simple enough to localise from.

**Q2 -- does it preserve the shared coordinate?** Fit forward/lateral/yaw on the insect's egocentric
embeddings, test on the B1's **without refitting**. **Camera height and gait-induced bob differ
enormously between a stick insect and a quadruped**, so "how the world moves" may not mean the same
thing on the two bodies.

| outcome | reading |
|---|---|
| **Q1 pass, Q2 pass** | the direction is alive; the full environment is worth building |
| **Q1 pass, Q2 fail** | the world model is fixed and cross-embodiment is lost. **A trade-off, not a solution** -- report it, do not proceed on Q1 alone |
| **Q1 fail** | the view change did not break the redundancy; report the leak |

## The first build guessed, and the guess was wrong

**The camera was mounted on `/abdomen` -- the rear segment -- facing a direction assumed from an
axis convention.** `ROBOT_ROOT` is the abdomen and `TRACK` is `/head`; the camera went on the root.
Caught by looking at a frame, which is the only thing that could have caught it.

**The fix removes the guess rather than correcting it.** The camera pose is now derived from
geometry the scene can be asked for: the insect's forward is **`head - abdomen`**, the B1's is **the
direction its base actually travels** in the clip being rendered, and the sensor's `+z` is built
onto that vector directly with `setObjectMatrix`. **Neither can be wrong about a convention, because
neither uses one.** `--ego_forward` remains as an override that should never be needed.

## Step 0 is still a hard gate

**A camera facing the wrong way still produces 66 frames**, still passes every downstream script,
and would answer Q1 with a resounding **false pass** -- a view of the sky reveals no action either.
That is what nearly happened here and what voided F165. The run sheet renders and stops for a human
before it collects anything.



## What was built

| | |
|---|---|
| `sim/scene/ego_camera.py` | mounts the sensor on the head with a **measured** forward direction; builds the textured box |
| `--ego`, `--ego_forward`, `--ego_offset`, `--ego_box` | on `collect_ik.py` **and** `render_b1_replay.py`, so both robots see the same room through the same kind of camera |
| `scripts/dataset/ego_derisk.sh` | step 0, collection, Q1, Q2, and the paired head-view render |

**Q1 and Q2 reuse existing measurements unchanged** -- `inverse_dynamics_r2.py --target action` and
`motion_rep_check.py` part 2 -- so the egocentric numbers are directly comparable to the
third-person ones rather than being a new scale.

**Fixed in passing: `render_b1_replay.py --help` has never worked.** Two help strings contain
`100% clipped` and `36-100% of`, which argparse reads as `%c` and `%o`. Escaped.

## Design revision before building, on three points from prior work

**1. The ground is the primary flow source, not the walls.** A head camera on a walking robot spends
most of its frames looking at floor, and prior egocentric locomotion reads flow from **ground
texture** rather than wall landmarks (Hu et al. 2207.03386). The build now randomises the floor
every episode and treats the box as a backdrop that keeps the frame from ending in empty space.

**2. The four fixed wall colours were a self-inflicted Q1 failure, and the reasoning behind them was
backwards.** This file argued for per-side colour so that "bearing is readable" -- **which is
single-frame pose readability, the exact thing the egocentric view exists to break.** "See red,
facing north" is a landmark, and a landmark is a pose label. Q1 could have failed on a property
deliberately built in, and the failure would have looked like a fact about egocentric views.

**Fixed by randomising appearance per episode** (1703.06907): wall colours, wall textures and the
ground are all redrawn from `--ego_seed`, advanced per episode and repeat. Within one clip the walls
still differ -- they physically must -- but **no colour means a direction across the dataset**, so a
probe split by clip cannot use it on held-out clips. Walls are structure, not identity.

**3. The r = 0.72 texture result does not transfer and should not be tuned against.** It was
measured by panning a 2D texture past a viewport: **no depth, no parallax, no shadows.** A real
rendered egocentric view has near things moving faster than far ones, which is motion information a
2D pan cannot contain, and GeoLoco (2603.07624) pushes models toward 3D geometry over 2D texture
precisely because the two differ. **The texture measurements stand as a comparison between surfaces;
they do not establish that the signal is too weak.** Sufficiency is decided by Q1 in the rendered 3D
view, not by the proxy.

**The octave choice is therefore provisional**, kept because it beat the alternatives on the only
evidence available, and flagged as untested in 3D.

## Two guards locked in before collection, both against confounds this project has already been bitten by

**Guard 1 -- the two bodies must be collected from the same seed distribution, and ideally from
paired seeds.** The same integer produces the same room on either robot, so `insect seed k` and
`B1 seed k` differ **by body and by nothing else**, and the environment cancels in Q2.

**Unmatched seeds make Q2 unreadable in exactly F160's shape**: "the coordinate does not transfer
because the bodies differ" -- the result being tested for -- and "because the two sets were
collected in different-looking rooms" become the same number, with no way to separate them
afterwards. `--ego_seed` now exists on **both** `collect_ik.py` and `render_b1_replay.py`, and the
insect's advances by *repeat* only, so a caller setting it per **condition** gets matched rooms
across bodies for the same condition.

**Guard 2 -- randomisation must be verified sufficient before Q1 is trusted, not after it is
reported.** Per-episode randomisation is the *intent*; too few seeds and colour still correlates
with heading across the dataset, and a probe will find it. **Then Q1 measures the leak that
randomisation was introduced to remove**, and reports it as a property of egocentric views.

`scripts/diagnostics/check_appearance_leak.py` measures it on the frames themselves: a deliberately
crude colour summary -- mean RGB of the upper half where the walls are, plus an 8-bin hue histogram,
**no geometry** -- against heading, nearest-centroid, **split by clip**. If a summary that weak
predicts heading above chance, the leak is gross.

**It exits non-zero on a leak**, so it can gate the collection rather than being a report somebody
reads later. **The fix when it fires is cheap -- add seeds.** What is not cheap is finding out
afterwards that Q1's number was a landmark.

**Neither guard is a suggestion.** F160 is the case where an asymmetric null nearly made a control
unreadable, and F165 is the case where a silently-ignored flag produced eight conditions that passed
every gate while carrying no intervention at all. **Both were caught by a check that ran before the
result was believed.**

## Outcome: both questions pass, and turning is where it pays

48 clips per body, twelve conditions, four rooms fully crossed -- every room hosts every behaviour
and every behaviour appears in all four rooms, so appearance carries no information about which
behaviour a clip is. Seeds paired across bodies by slot.

**Guard 2 first, as locked.** Room colour predicts heading on held-out clips at **0.34x chance** on
the insect and **0.50x** on the B1 -- **below** chance, so there is no landmark and Q1 is measuring
the view.

## Q1: the redundancy breaks, decisively on the insect

| insect | single frame | pair | gap |
|---|---|---|---|
| allocentric (F159) | **0.779** | 0.887 | +0.108 |
| **egocentric** | **0.293** | 0.578 | **+0.285** |

| B1 | single frame | pair | gap |
|---|---|---|---|
| allocentric | 0.161 | 0.328 | +0.167 |
| **egocentric** | **0.097** | 0.259 | +0.162 |

**Single-frame readability falls by 0.486 on the insect and the pair-minus-single gap nearly
triples.** This is the first intervention in the whole chain that moves the quantity F153 through
F169 were all trying and failing to move. Per family the insect reads sideways at **-0.008** from
one frame, speed 0.421, turn 0.483 -- against 0.609, 0.812 and 0.931 allocentrically.

**The B1's gap is unchanged** at +0.162, because its single-frame readability was already low (0.161)
and there was little to remove.

## Q2: the coordinate still crosses, and yaw crosses far better

Fitted on the insect's egocentric embeddings, tested on the B1's **without refitting**:

| | forward | lateral | yaw |
|---|---|---|---|
| allocentric, insect held-out (F162) | 0.98 | 0.95 | 0.96 |
| allocentric, **B1 unrefitted** | 0.63 | 0.43 | **0.07** |
| egocentric, insect held-out | 0.77 | 0.53 | 0.61 |
| egocentric, **B1 unrefitted** | **0.50** | **0.39** | **0.64** |

**Yaw transfer goes from 0.07 to 0.64.** Turning is the channel this project has fought since F136
and which F169 nearly lost entirely, and it is the one the view change helps most -- which is what
the physics predicts, since a head camera sees rotation as global image flow whatever the body
underneath is doing.

**Forward and lateral transfer fall**, 0.63 to 0.50 and 0.43 to 0.39, and the within-insect fit falls
further, 0.98 to 0.77 on forward. **The coordinate is harder to read from a head view and it still
crosses.** That trade is the honest summary and it should not be reported as an unqualified
improvement.

## Two things this does not say

**`motion_rep_check.py` prints a verdict line of its own** -- "redundancy is NOT broken, Direction B
is dead" -- which belongs to **F162's** question about a motion *representation*, not to Q1. It is
computed from that script's Part 1 and must not be read as a verdict on this run.

**Nothing here is a trained model.** Q1 says the action is no longer readable from a pose; it does
not say a world model trained on this data will use the transition. F155 through F169 are a record of
signals that existed and were then ignored by a trained predictor. **The next step is to train one
and measure, and the pre-registration for that has not been written.**

## The pre-registered branch, taken

**Q1 pass and Q2 pass: the direction is alive and the full environment is worth building.** Every
number above comes from a room built to be the cheapest thing that could answer the question, so the
first real decision is what a proper environment should have that this one does not.

Data: `data/egocentric/beh12_c10f10t10_ego_flat`, `data/egocentric/beh12_b1_ego_flat`.

---


### Note (not a finding): what the shared coordinate actually requires, against Demo-JEPA

**Scoping by code inspection, to locate where a technical claim could sit.** Nothing was run.

## What the coordinate needs

| | this project | Demo-JEPA |
|---|---|---|
| **paired data across bodies** | **none.** `body_target = batch["body_motion"]` is that clip's own measurement; batches never pair an insect frame with a B1 frame, and `lambda_cross` -- the term that *would* pair them -- is **0.0** in the checkpoint every result is measured on | end-effector retargeting, to manufacture pairs |
| **temporal alignment across bodies** | **none.** No DTW, no soft alignment, no shared clock; each clip is consumed on its own timeline | GTCC |
| **labels** | **none by hand.** Forward and lateral come from differencing the body's recorded position, yaw from its quaternion, both divided by `sqrt(g*h)` and smoothed over one stride | -- |
| **transfer** | fit on the insect, applied to the B1 **with no refitting** (`motion_rep_check.py` part 2: one `mu`/`sd` and one solve, both from the insect's training half) | -- |
| **one head or two** | **one.** `body_head(z)` takes the latent alone and is shared across embodiments; only the *action* decoders are per-body | -- |

**So the answer is LESS supervision, and on three separate axes**: no pairing, no alignment, no human
labels. What makes it possible is that the target is a **physical quantity both bodies possess**
rather than a correspondence that has to be constructed -- forward, lateral and yaw are measured
from each robot separately, and Froude scaling is what puts them on one axis (F56: 0.155 against
0.159 across a fourfold size difference).

## What it does need, and this must not be left out of any claim

**Privileged state.** The target is differenced from the simulator's recorded body position and
orientation. On hardware that is odometry or motion capture, not vision. **The coordinate is not
learned from pixels; it is regressed onto a measured quantity**, and a paper that says "no
correspondence required" while quietly requiring pose ground truth is overclaiming.

**A stride-length smoothing window**, which is a locomotion-specific choice with a measured reason:
per-frame speed is dominated by body rocking, and between-clip variation sits at 0.63 of within-clip
rocking at five frames but 1.45 at a stride.

## Where the novelty candidate sits

**Not "we learn a shared latent" -- everyone does.** The candidate is narrower and defensible:
**a cross-embodiment coordinate that needs no pairing, no temporal alignment and no retargeting,
because the quantity chosen is one every legged body already has and Froude scaling makes
commensurable.** Demo-JEPA builds correspondence; this measures a quantity that is already common.

**And it is worth stating as a limitation of scope**: it buys exactly three channels. Anything the
two bodies do not share -- their joint spaces, their gaits, their contact patterns -- is not carried
by it, which is precisely what F82 and F83 found when the same question was asked of joint targets.

---


### Note (not a finding): gait shake and world motion are separable, and removing the shake helps across bodies

**Scoping a candidate method.** An egocentric camera carries two things at once: where the body is
going, which both robots share, and how the body shakes getting there, which is a tripod on one and
a trot on the other. If the second can be removed the first should cross better -- and that would be
a method rather than a setting. Q2 left this open by moving yaw from 0.07 to 0.64 while moving
forward and lateral the other way.

## They are separable, and by a wide margin

Decomposing each clip's camera yaw into a linear trend (the net turn) and the rest (the gait):

| | stride peak | share of the oscillation's power at that peak | trend's share of total variance |
|---|---|---|---|
| insect | **6 cycles/clip** | 0.53 | 0.57 |
| B1 | **6 cycles/clip** | 0.51 | 0.56 |

**The gait sits at 6 cycles per clip on both bodies and the net turn at 0 to 1**, so they occupy
different bands and a projection can separate them. **They are comparable in size** -- gait sd over
turn sd is 0.79 on the insect and 0.96 on the B1 -- so this is not a small correction to a large
signal.

## Removing it helps, modestly, and helps *across bodies specifically*

Per clip, `sin` and `cos` at the measured stride frequency plus two harmonics, subtracted from every
embedding dimension, clip mean kept. Coordinate fitted on the insect, applied to the B1 **without
refitting**.

| | | forward | lateral | yaw |
|---|---|---|---|---|
| egocentric | insect held-out | 0.71 | 0.29 | 0.61 |
| egocentric | **B1 unrefitted** | 0.45 | 0.38 | 0.57 |
| **gait removed** | insect held-out | 0.73 | 0.25 | 0.62 |
| **gait removed** | **B1 unrefitted** | **0.47** | **0.46** | **0.61** |

**All three channels improve across bodies and the within-body fit does not.** That asymmetry is the
interesting part: 0.71 -> 0.73, 0.29 -> 0.25 and 0.61 -> 0.62 within the insect, against +0.02,
**+0.08** and +0.04 on the B1. **Removing a body-specific component helps exactly where bodies
differ**, which is what the hypothesis predicts and what a generic denoising would not do.

## Answering the question Q2 raised

**Lateral comes back and then some**: 0.38 -> 0.46, against **0.43 allocentric**. **Forward does
not**: 0.47 against **0.63 allocentric**. So the drop Q2 showed is *partly* gait shake and partly
something else, and forward is where the something else lives.

## What this is and is not

**It is a viable method candidate.** Gait-locked shake is separable, it is body-specific, and
removing it improves cross-body transfer on every channel. "Extract embodiment-invariant ego-motion
from gait-coupled egocentric video" is a real problem with a measurable handle.

**The effect is modest and the method here is crude** -- three harmonics of one frequency estimated
per clip, subtracted linearly. That it works at all with a projection this blunt is the argument for
trying something learned; it is not evidence that a learned version would do better.

**And the baseline differs slightly from Q2's** -- 0.45/0.38/0.57 here against 0.50/0.39/0.64 --
because this uses every frame where `motion_rep_check` strides by three, on a different split.
**Only the two rows of this table may be compared with each other.**

Reproduce: `scripts/diagnostics/degait_coordinate.py`. `gram` now chunks both sides; the egocentric
sets are three times the size the function was written for and it died on the GPU rather than
degrading.

---


### F171. Which surface V-JEPA2 reads motion from, measured rather than inherited

**Asked because the egocentric room reused the floor's texture recipe on faith.** That recipe was
chosen for a *floor* under a *third-person* camera, to stop the background drowning the robot
(`set_floor_texture.py`). **An egocentric wall has the opposite job**: it is not background, it is
the entire signal.

**No simulator.** A large texture is panned past a 256 px viewport -- what a wall does when a robot
walks along it -- and the frames go through the pipeline's own encoder. Correlation between true pan
distance and embedding distance, four pan speeds:

| surface | r | note |
|---|---|---|
| blank | -- | no change at all |
| **checkerboard** | **-0.149** | **reproduces this project's own -0.16** (`set_floor_texture.py`) |
| white noise | 0.199 | |
| value noise, 5 octaves -- **the floor recipe** | 0.478 | what was inherited |
| **value noise, octaves 4/8/16** | **0.723** | what is used now |
| value noise, octaves 8/16/32 | 0.613 | |
| the same, tiled 4x4 | 0.474 | repeating costs a third of it |

**The checkerboard row is the method's own control.** It lands on the number this project measured
years of findings ago by a completely different route, which is why the ranking above is worth
trusting.

**The floor recipe's fine octaves at 64 and 256 are what cost it** -- the same high-frequency
aliasing the checkerboard finding is about, in a milder form. And **a repeating surface is ambiguous
about position**, so embedding distance stops tracking distance travelled: 0.723 falls to 0.474 when
the same texture is tiled.

## The limit, stated plainly

**The best surface tested reaches r = 0.72 and moves the embedding about twice as far as a one-pixel
wobble.** Egocentric rests on this relation being strong; measured on a 2D pan, it is not.

**And a 2D pan is not the question.** It has no depth, no parallax and no shadows, and GeoLoco
(2603.07624) pushes models toward 3D geometry over 2D texture precisely because the two differ.
**These numbers rank surfaces against each other; they do not establish that the signal is
sufficient.** That is what F170's Q1 answers, in the rendered scene.

Reproduce: `scripts/diagnostics/texture_for_vjepa.py`.

---

### Note (not a finding): ten defects in the egocentric scene, and what caught each

**Every number in F170 depends on the scene being what it claims**, so this records what it was not,
in the order the faults surfaced. **Three of the ten were caught by looking at a rendered frame and
would have passed every statistical gate**; two were caught only because a measurement disagreed
with geometry; and **two of my own measurements turned out to be the thing that was broken.**

| # | defect | how it presented | what caught it |
|---|---|---|---|
| 1 | camera mounted on `/abdomen`, the **rear** segment, facing a guessed axis | dark, flat frames | **looking at a frame** |
| 2 | `sim.createTexture` returns three values, two were unpacked | walls flat colour, "texture skipped" | reading the log line |
| 3 | shape colour **and** texture both tinted -> multiplied | near-black walls | **looking** |
| 4 | texture PNGs cached by filename, so a fixed recipe reused the broken files | looked exactly like a fix that did not work | timestamps on `/tmp` |
| 5 | 6 m tile on an 8 m wall -> one visible seam | vertical lines splitting each wall | **the user looking** |
| 6 | plane mapping, tried as the fix for 5, smears the texture | vertical bands | a four-way mapping-mode render |
| 7 | field of view set **before** `startSimulation`, which restores the scene file's value | every insect clip shot at 15 deg, not 90 | junction 19.4 deg against a predicted 6.0 |
| 8 | room built before the respawn, so the robot stood 2.8 m off centre | near wall at 1.2 m instead of 4 | the same measurement |
| 9 | scene floor 5 m inside an 8 m room | view ended in void before the wall | floor/room arithmetic |
| 10 | insect head carried **+7.53 deg up** while walking, camera parented to it | 13.6 deg junction against 6.0; 32% floor against the B1's 44% | measured over all 48 clips |

**Three more were cross-body asymmetries rather than defects**: the B1's camera aimed along its
*direction of travel* instead of its body axis (90 degrees off on sideways clips), its clips each
started at a different heading, and `--floor_scale` stretched its ground texture threefold **after**
it was applied. All three would have entered Q2 as differences between bodies that are not about
bodies.

**And two of my own instruments were the fault.** A junction detector using `argmax` picked the
ceiling edge on some frames and the floor edge on others, which read as the insect's view being
unstable when the metric was; and a "floor detail" figure compared different image regions on the
two bodies because the region was defined by that same detector.

**The rule this earns, beyond the ones already recorded:** when a measurement disagrees with
geometry, **check the measurement before changing the scene.** Two of the ten fixes above were
applied to a scene that was already correct.

Final settings and the code that carries them: `sim/scene/ego_camera.py`, `--view egocentric`.

---

### Note (not a finding): `data/` reorganised into allocentric and egocentric

**The split is the camera, not the behaviour.** Every set collected before 2026-09-01 moved to
`data/allocentric/`; the head-camera sets live in `data/egocentric/`. **A number measured under one
view cannot be compared with one measured under the other**, which is the whole subject of F170.

**278 path references were rewritten across 117 files** -- code, run sheets, docs and saved run
configs -- in three passes, because the first covered only git-tracked files and missed every script
written that day, and the second missed prose of the form `data/beh12_*`.

**34 symlinks broke and were repaired rather than re-collected.** `beh12_b1_flat_9clips` and
`beh10_c10f10t10_intent2_flat` are symlink farms with absolute targets. **A broken symlink presents
as a missing file several scripts deep**, not as a path problem, and it was caught by running a
diagnostic rather than by grep.

**Two smoke tests confirm the move is inert**: `check_within_clip_intent` and `branch_divergence`
both reproduce their pre-move numbers to every digit.

---


### Note (not a finding): `report/NUMBERS.md` and `report/retrain_and_remeasure.md` removed, and the two things they carried

**`retrain_and_remeasure.md` deleted on its own terms.** Its header read "delete this file once
slides 4-12 are re-measured; the conclusions belong in `FINDINGS.md`, not here." They are, and they
were.

**`NUMBERS.md` deleted because its subject no longer exists** -- it was the provenance ledger for
`presentation_v2.md` and `proposal_draft.md`, neither of which is in the repo. **Two things in it
would have been lost silently, so they are moved here.**

## 1. A number that does not reproduce is still cited in two live files

`NUMBERS.md` section 1 marked these ORPHANED after failing to recompute them:

| claim | documented | recomputed from repo data | |
|---|---|---|---|
| long body, distance | 4.125 +/- 0.434 m | **4.404 +/- 0.187 m** | **does not match** |
| medium body, distance | 3.562 +/- 0.015 m | 3.569 +/- 0.010 m | matches |

**The documented long-body figure is still quoted in `doc/PROGRESS.md:255` and
`sim/SOURCES.md:76`**, where nothing says it failed to reproduce. `SOURCES.md` calls it "bimodal --
lands on 4.479 *or* 3.593", which is consistent with a mean that moves between runs and is the
likeliest explanation.

**It is a Step -1 morphology-gap number and nothing current rests on it.** But F115 is this
project's record of a finding marked fixed that was not, and **a number that fails to reproduce
should not sit in two documents with no note attached.**

## 2. The standing rule, which is worth keeping

> Every number in a document should be regenerable by a command in this file. Anything that is not
> gets marked ORPHANED and does not go in front of an advisor.

**That role now belongs to the deck's measurement slide and to this file**, where each entry names
the script that produced it.

References to the deleted file were removed from `scripts/figures/plot_morphology_evidence.py` and
`sim/README.md`.

---


## Files

- `sim/collect/collect_ik.py --gait cpg` -- joint-space oscillator giving the hexapod a second
  gait plus steering and a speed range, without IK (F71)
- `scripts/diagnostics/inspect_scene.py` -- list a CoppeliaSim scene's joints and read out any
  attached script; how the Olaf controller was recovered
- `scripts/diagnostics/tune_legs.py` -- solves a gain and an offset per leg so six unequal legs
  trace the same stroke; converges, and makes the gait worse, which is the point of F73
- `results/wm/dataset/figures/gait_legtune.png` -- the contact raster with and without it (F73)
- `data/allocentric/fwd_hex7speed` -- five constant speeds plus both ramp directions, 91 clips (F60)
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
- `data/_archive/ik_walk_speed5` -- five retimed speeds matched to the B1's Froude band (F57). Episode
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
- `results/wm/closed_loop/b1_hexgoal_warmturn/` -- cross-embodiment control, turning warm start (F107)
- `results/wm/closed_loop/b1_hexgoal_warmforward/` -- the same goals with a forward warm start (F109)
- `results/wm/closed_loop/b1_hexgoal_arm4_nce_joint/` -- the same goals with no warm start at all (F109)
- `results/wm/closed_loop/hex_unseen_nowarm/` -- the F95 configuration with no warm start (F110)
- `results/wm/closed_loop/hex_unseen_turn_nowarm{,_c3}/` -- real turns with no warm start, `--commit` 1 and 3 (F110)
- `results/wm/closed_loop/b1_hexgoal_speedrange/` -- seven hexapod forward goals spanning 1.72x in Froude (F111)
- `results/wm/closed_loop/b1_hexgoal_arm{1,2,3,4}_*/` -- the four adaptation arms, four goal clips per condition (F112)
