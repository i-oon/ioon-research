# Stage 1 findings: what a two-body cross-morphology setup can and cannot show

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
| leg scale | dimensionless | 1.0 = base body; `medium` = 0.75, `short` = 0.5 |
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

![Morphology signal through the pipeline](results/wm/morphology_axis.png)

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

Reproduce: `.venv/bin/python3 scripts/morphology_axis.py --ckpt wm/runs/<run>/epoch020.pt`

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

**The bodies are uniformly scaled, so the task is close to affine.** `sim/make_leg_morphology.py`
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
held-out `medium`, all three clips (`sim/render_wm_prediction.py`, `scripts/wm_gait_report.py`):

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

![Gait, decoder](results/wm/gait/gait_stage1_100ep_framed_runB_epoch020_medium_clip0.png)

*Decoder-driven gait above, ground truth below. Black is stance. The stance blocks fragment and
the left and right tripods stop alternating.*

![Gait, probe](results/wm/gait/gait_probe_ridge_runB_medium_clip0.png)

*Probe-driven gait above, ground truth below, same axes. Stance blocks keep roughly the right
length and phase.*

Videos: `results/wm/gait/replay_*.mp4`, predicted on the left, ground truth on the right.

### F4d. Interpolation and extrapolation fail at different stages

Fold 2 holds out `short` (leg scale 0.5) while training on `long` (1.0) and `medium` (0.75), so
the test body lies outside the training range rather than between the training bodies. Axis
here runs 0 = `long`, 1 = `medium`, and the correct position for `short` is 2.803 (CF).

![Both folds](results/wm/axis_both_folds.png)

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

![Interpolation failure](results/wm/interpolation_failure.png)

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

![Per-joint traces on the held-out body](results/wm/action_trace_stage1_100ep_framed_runB_epoch020_medium.png)

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

![Validation against held-out body](results/wm/heldout_sweep_two_seeds.png)

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

![Held-out sweep](results/wm/heldout_sweep_runB.png)

*Every snapshot re-scored on 2,600 identical cached held-out pairs. The dotted line at 1.0 is
predicting the mean; the grey curve is the same model with the latent zeroed.*

Re-scoring every snapshot on 2,600 identical cached held-out pairs
(`wm.sweep_checkpoints`), transfer is best between 3,086 and 9,258 gradient steps of 30,860 and
does not improve after. Selecting a checkpoint on this curve would leak the test body, so the
curve is for reporting compute cost, not for choosing a model.

## The setup this points to

1. **More bodies, fewer episodes each.** F13 shows episodes are wasted and F5 shows bodies are
   the binding constraint. Roughly 30 episodes each across 6 to 8 bodies costs the same to
   collect as the current 100 across 3.
2. **Morphology with more than one dimension.** `sim/make_leg_morphology.py` currently takes a
   single `--factor` and scales coxa, femur and tibia together, which makes morphology a line.
   Scaling the three segments independently makes it a volume, and a held-out body can then be
   an unseen *combination* rather than a point between two others. That tests composition, not
   interpolation along a line.
3. **Keep the shared foot trajectory.** It is what makes "the same latent action across
   different bodies" a well-defined statement: every body executes the same task-space intent,
   and only the decoder differs. Giving each body its own gait would mix morphology with
   behaviour and weaken the claim rather than strengthen it. The averaging baseline of F6 is
   beaten by learning the nonlinear curve of F7, which needs bodies, not different gaits.
4. **Report the baselines every time.** Copy-nearest-body, linear interpolation and linear
   extrapolation belong in every held-out table. On this dataset they are strong.
5. **Report per joint type, with the held-out body's own baseline, averaged over at least
   three runs.** F8, F10 and F12 each independently make a single aggregate number misleading.
6. **Do not select on the held-out curve.** F14. Fix the budget in advance and report the
   final checkpoint, or report the whole curve.

## What this enables

The result in F1 stands on its own: a latent-action world model, trained only on video and
joint commands with no morphology input, learns to identify which body it is looking at and to
emit that body's joint offsets to within 0.06 deg, across bodies whose postures differ by 34
to 50 deg. Nothing supervises that. It is a concrete demonstration that visual observation
carries enough morphological information to condition a controller, which is the premise the
cross-embodiment argument rests on.

What F4 to F7 establish is the boundary of that result: identification is not generalisation,
and a two-point training set cannot teach the mapping from appearance to morphology, only two
memorised instances of it.

For the cross-embodiment claim in `OPEN_QUESTION.md`, none of the limitations above transfer.
The baselines that beat the model here all rely on interpolating between bodies in a shared
18-D joint space. An 18-DOF hexapod and a 12-DOF quadruped have no shared joint space, no
correspondence and no midpoint, so copy-nearest-body, averaging and linear extrapolation are
not merely weaker there -- they do not exist. That is exactly the asymmetry the thesis argues
for: vision is a common space where proprioception is not.

## Files

- `wm/predict_actions.py` -- reconstruct joint commands for any body, in radians
- `scripts/plot_action_trace.py` -- per-joint predicted against ground truth, with R^2
- `wm/sweep_checkpoints.py` -- re-score every snapshot on identical cached embeddings
- `results/wm/interpolation_failure.png` -- F4 and F6 in one figure
- `results/wm/heldout_sweep_two_seeds.png` -- F11 and F12
- `results/wm/axis_embeddings.npz` -- embeddings behind the axis positions in F4
- `results/wm/README.md` -- per-run metrics
