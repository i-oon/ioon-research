# Progress Update — Cross-Morphology and Cross-Embodiment Latent Action Models

Stick insect (*Medauroidea extradentata*) and Unitree B1, simulated in CoppeliaSim.

**Slides 1-12 are Stage 1**: one 18-DOF topology, several leg geometries, unchanged from the
previous update. **Slide 13 is Stage 2's position.** **Slides 14-26 are new**: an attempt to make
the world model action-conditioned that failed in six independent ways, the principle those six
measurements point at, the prediction that principle makes, the test of it -- **which half held and
half did not** -- and what four attempts to build on the result then measured.

**The arc, in one line.** We could not make the world model use the action, so we measured why; the
answer was a property of the *viewpoint* rather than of the model; and that property predicted a fix
which prior work had already adopted without explaining. **This is not "we tried things until one
worked."**

**Notation.** `e_t` is a V-JEPA2 observation embedding, `z` a latent action, `a` a joint command.
Where this deck describes a phenomenon ActSWM also reports — action-sensitivity, Context Collapse,
the real-against-null rollout contrast — it uses their terminology and **our** symbols; ActSWM's
`z` denotes an observation embedding and adopting it would collide with ours.

Slides 1 to 3 are background already covered previously. Stage 1's update starts at slide 4;
Stage 2's position is slide 13; everything from slide 14 is new since the last deck.

**Citations are separated from contributions throughout.** Each claim slide says what prior work
found, what we measured where they did not, and what is ours. The papers this deck leans on:
ActSWM (2607.26712), Yeom et al. (2606.07687), Demo-JEPA (2605.20811), Hu et al. (2207.03386),
AHA-WAM (2606.09811), UWM-JEPA (2605.25313), GeoLoco (2603.07624).

**Three claims this deck previously made are withdrawn**, each by a control that had not been run:
that the contrastive adaptation term is what crosses embodiments, that the closed loop selects
behaviours on the quadruped, and that sideways motion fails on every measurement. Slide 13 says
what replaced them.

---

## Slide 1 — Cross-morphology locomotion from a latent action model

Stage 1 progress update: what was built, what it measures, what it found.

**The problem.** A locomotion controller maps state to joint command. Change the leg lengths and
the same numerical command produces a different physical result — the robot stumbles, or stands at
a different height, or does not move. Every body needs its own commands for the same behaviour.

**The question this stage asks.** Can a model learn a latent action `z` from **video alone** — no
morphology label, no kinematics supplied — that separates *what movement is happening* from *which
body is doing it*, and then turn that latent into the correct body-specific joint command?

**The scope, stated precisely.** Stage 1 is cross-**morphology**, not cross-embodiment. All Stage
1 bodies share one 18-D joint space, six legs times three joints; only the geometry differs.
Stage 2 extends the same question to cross-embodiment with a quadruped, and tests a held-out
4-leg action space.

**Why it matters for what comes after.** If the latent really separates behaviour from body, the
same latent should drive a robot with a different number of legs. A camera gives that for free:
`256x256x3` is the same input space whatever the body, so one encoder serves both robots without
being told anything about either. Joint space has no such property — 18 numbers and 12 numbers with
no correspondence between them — so a proprioceptive method has to be **handed the kinematic
structure** before it can compare the two. Stage 1 is the test of whether the separation happens at
all, in the easy case where the joint spaces do match.

**What this update covers.**

| | |
|---|---|
| Slides 2-3 | what was built, the data, and how every number below is measured |
| Slides 4-7 | the central Stage 1 result: the geometry is readable, the model ignores it, what fixed that |
| Slides 8-10 | where it stops working, why, a test of that explanation, and a check that predicts it |
| Slides 11-12 | two facts about the task itself that bound what the latent can be worth |
| Slide 13 | Stage 2's position: the thesis, the gap, and what is proven against what is not |
| Slides 14-16 | **the attempt** — ActSWM's method rebuilt, six pre-registered routes closed, and the number underneath all six |
| Slides 17-19 | **the principle** — pose determines the future; what it explains in the literature and in our own record |
| Slides 20-23 | **the prediction tested** — egocentric breaks the redundancy, the coordinate survives, and the two contributions |
| Slides 24-28 | **the current diagnostic arc** — action selection fixed by the right scoring space, imagination-RL's wall localised to the rollout, the single-step-prediction synthesis ruled out every cheap fix, and the closing result: the signal is real but does not survive training — and how everything is measured |

---

## Slide 2 — The pipeline and its three trained modules

```
frame_t  --[frozen V-JEPA2]-->  e_t  --+--[ITM]--> z --[FTM]--> ê_{t+1}
                                       |
                                       +--[Motion Decoder]--> â
```

- **Encoder**: V-JEPA2 ViT-g/16, 1B parameters, **frozen throughout**, never trained on robots.
- **Trained**: three modules on top, about 5M parameters each.
- **Data**: CoppeliaSim, 20 Hz, fixed 256x256 side camera, joint targets in radians.

| Module | Input | Output | Why it exists |
|---|---|---|---|
| **ITM** inverse transition | `e_t`, `e_{t+1}` | `z ∈ ℝ^64` | Given a transition, what action produced it? |
| **FTM** forward transition | `e_t`, `z` | `ê_{t+1}` | Does `z` let you predict the next frame? |
| **Motion Decoder** | `e_t`, `z` | `â ∈ ℝ^18` | Can `z` be turned back into an executable joint command? |

**The objective, as inherited from LAC-WM.** Two terms:

```
z      = ITM(e_t, e_{t+1})
L_recon  = || FTM(e_t, z) - e_{t+1} ||^2          predict the next frame
L_motion = || MD(e_t, z)  - a      ||^2          recover the real joint command
L        = 1.0 * L_recon  +  1.0 * L_motion
```

**Two structural facts to hold on to.**

The Motion Decoder receives `e_t` but **never `e_{t+1}`**. Anything the second frame contributes
has to travel through the 64-number latent. That bottleneck is the whole design, and two of the
limits below turn out to hinge on it.

The two terms sit on very different scales — reconstruction around 1.5, motion around 0.01 in
standardised units — so weighting them equally at 1.0 does **not** give them equal influence.
**Reconstruction takes roughly 99 percent of the gradient in practice**, which means the term
meant to ground the latent in real commands is running on the remaining one percent.

---

## Slide 3 — The data, and how everything below is measured

Every run below draws from one of three directories, each built by linking only the clips where
the body actually walked — signed forward travel ≥ 0.30 m and lateral drift < 0.20 m. Bodies that
collapse or veer are excluded by name, not by hope.

| Dataset | Bodies | Size | Used by |
|---|---|---|---|
| `ik_walk_m3d_clean` | 4 training, all femur = tibia | 140 clips | `m3d_cross`, `m3d_bracketed` — slides 4-8 |
| `ik_walk_cov_narrow` | 4 training, all at femur/tibia 0.83 | 96 + 20 clips | `tib_cross`, `tib_ctrl` — slide 9 |
| `ik_walk_cov_wide` | 6 training, femur/tibia decoupled | 96 + 20 clips | `bracket_cross` — slide 9 |

The two coverage directories are **volume-matched at 96 training clips**, which is what lets slide
9 attribute its result to coverage rather than to more data.

- Commands come from **IK retargeting**: one shared foot trajectory in Cartesian space, solved
  separately per body. Same intended behaviour, genuinely different joint commands. Without this
  the transfer question would not be well posed — every body would receive the same command.
- Behaviour: forward walking only, one speed. This turns out to matter, and is picked up later.
- Held-out bodies are never trained on and are used only for evaluation.

| Tool | What it does | What it tells us |
|---|---|---|
| **Linear probe** | Fit a ridge regression on training bodies, apply to a held-out body | Is the information present and readable, independent of what the trained model does with it |
| **Swap test** | Give the decoder body A's frame with body B's latent | Does the decoder take the body from the frame or from the latent |
| **Input ablation** | Zero out `z`, or zero out `e_t`, and re-measure | Which of its two inputs the decoder actually depends on |
| **Mixture fitting** | Find the best combination of training bodies' commands explaining the output | Is the model interpolating, or copying one body |
| **Physical replay** | Drive the predicted commands through the same physics | Do the commands actually walk, not just score well per joint |

All error figures below are RMSE in degrees, pooled over all 18 joints and all timesteps, on a
body never trained on. The commands' own spread is about 11.7 deg per joint, so that is the number
to read every error against.

**One rule for every `x` in this deck: `comparison error ÷ our error`. Above 1.0 the comparison is
worse; 1.0 is a tie.** Only what we compare against changes, and each table names it:

| compared against | so `1.4x` means | slides |
|---|---|---|
| **a random backbone** — same head, same clips, untrained weights | pretraining is worth 1.4x | 15, 16 |
| **holding the frame still** — predicting no change at all | the forward model beats doing nothing by 1.4x | 11, 12, 13, 16 |
| **the same model with a part deleted** | that part was worth 1.4x — read as a *cost* of removing it | 14, 19 |

The one exception is flagged where it occurs: the Stage 2 transfer slides compare **R²**, where
higher is better, so that ratio runs the other way.

**"Control" throughout means the matched run**: identical data, identical split, identical
architecture, identical seed, with **one flag changed** — the cross-body loss turned off. Which
run that is depends on which split is being discussed, so it is named each time:

| Split | With the cross-body loss | Its control |
|---|---|---|
| 4 training bodies, held out `c08f09t09` (inside the range) | `m3d_cross` | `m3d_bracketed` |
| 4 training bodies tied at femur/tibia 0.83, held out `c10f10t08` (outside it) | `tib_cross` | `tib_ctrl` |

---

## Slide 4 — The geometry is in the frame, and the model does not use it

**The information is there.** A ridge probe from the **frozen** encoder to the three segment
scales, fitted on the four training bodies and applied to one it has never seen.

| Body | | coxa pred / true | femur | tibia |
|---|---|---|---|---|
| c10f10t10 | train | 0.978 / 1.00 | 0.999 / 1.00 | 0.999 / 1.00 |
| c06f10t10 | train | 0.623 / 0.60 | 0.998 / 1.00 | 0.998 / 1.00 |
| c10f06t06 | train | 0.997 / 1.00 | 0.602 / 0.60 | 0.602 / 0.60 |
| c06f06t06 | train | 0.602 / 0.60 | 0.600 / 0.60 | 0.600 / 0.60 |
| **c08f09t09** | **held out** | **0.836 / 0.80** | **0.914 / 0.90** | **0.914 / 0.90** |

Training rows are in-sample and near-exact, which is what makes the last row readable. Held-out
errors are **0.036, 0.014, 0.014** on a 0–1 scale, from **4,227 parameters** on a 1B-parameter
encoder that has never seen a robot, with nothing supervising it. **This is the premise the project
rests on.**

**The model trained on top does not use it.**

**Swap test** — give the decoder body A's frame with body B's latent, where the two bodies'
commands differ by 21.1 deg. It answers with **body B's** command to within 6.0 deg: it followed
the latent and ignored the frame.

**What geometry does each estimator think the held-out body has?** The probe reads it off the
encoder. The decoder is asked indirectly — fit its output as a mixture of the training bodies'
commands and read off the scales that mixture implies.

| Held-out `c08f09t09` | coxa | femur | tibia |
|---|---|---|---|
| **The truth** | **0.80** | **0.90** | **0.90** |
| Probe on the frozen encoder, 4,227 params | **0.836** | 0.914 | 0.914 |
| The trained decoder, 5.2M params | **0.622** | 0.962 | 0.962 |

Different estimators reading the same frame. The probe lands within 0.04 everywhere; the decoder
implies a **coxa 22% shorter than the body has**. The larger model is the one that misreads it.

> **What this table cannot test.** All four training bodies and the held-out one have femur equal
> to tibia, so neither estimator can produce two different numbers for them. 0.914 / 0.914 is
> interpolation along an axis the data spans. **They come apart once the data stops spanning the
> axis**, which is where both estimators fail together — measured a few slides on.

![the encoder places the unseen body correctly, the decoder does not](../results/wm/stage1/figures/encoder_vs_decoder.png)

**From the earlier three-body dataset** — an axis only draws cleanly with two training bodies — so
it shows on three bodies what the table above measures on four.

**Left**: everything placed on one line in joint-command space, 0 = long training body, 1 = short.

| | position |
|---|---|
| the held-out body's true commands | **0.30–0.36** |
| the 29k-parameter probe on the frozen encoder | **0.34** |
| the 5.2M-parameter trained decoder | **0.18–0.19** |

It is not at 0.5 because leg length and joint angle are not linearly related. The probe lands
inside the correct band; the decoder falls short, **pulled toward the training body it is nearest**.

**Right**: more trained capacity buys lower error on bodies it saw and higher error on the one it
did not. The 29k probe is the only predictor below the no-learning baseline on both.

**Reading**: the decoder learned to recognise which of the four training bodies it is looking at
and recall that body's commands. There is no entry for a body it has not seen.

---

## Slide 5 — Four changes that did not help, and the one that did

**Four attempts to fix it at the model**, same split, one flag each:

| What we changed | Result |
|---|---|
| Rescale the command target per body | No change |
| Shrink the decoder head to force generalisation | 1.4–2.1x worse |
| Remove body identity from `z` by adversarial training | Frame used 2x more, transfer 1.2x **worse** |
| Hand the decoder a pooled global view of the frame | Frame used 7.6x **less** |

Capacity, access and the contents of the latent were each ruled out. What remained was the
**objective**: nothing in the loss ever *required* reading geometry from pixels. Recognising the
body was cheaper and scored just as well.

**So change what the loss asks for.** Every body walks the same expert episodes, so at a given
timestep two bodies share the intent and differ only in geometry:

```
z^A      = ITM(e_t^A, e_{t+1}^A)                  the latent from body A's own transition
L_cross  = || MD(e_t^B, z^A) - a^B ||^2           A's latent, B's frame, B's command
L        = 1.0 * L_recon + 1.0 * L_motion + 0.5 * L_cross
```

**One term, weight 0.5, one extra decoder pass per batch** — `z^A` is reused, so no second encoder
or ITM pass. Reading the body out of the latent now gives the **wrong** answer by construction, so
the only way to be right is to read geometry from the frame.

Matched pair, identical but for `lambda_cross`, held-out `c08f09t09`:

| | Without | With |
|---|---|---|
| Error, deg | 3.67 | **3.44** |
| **cost of deleting the frame** (`zero_x`) | **0.083** | **1.621** |
| cost of deleting the latent (`zero_z`) | 0.729 | 0.917 |

**The second row is the result.** As a multiplier on each run's own error, **0.4x without the term
against 9.6x with it — a 22-fold difference in what the frame is worth**, from one flag.

![effect of the cross-body loss](../results/wm/stage1_correct/figures/cross_loss_effect.png)

**Left**: held-out error through training — the control spikes repeatedly (1.20 at epoch 13, 0.55
at 24) while the cross-body run is smooth and settles lower. **Right**: what each input is worth.
Read these *between* runs; zeroing an input is out of distribution, so control-against-cross is the
comparison that means something, not the raw multiplier.

**The swap test says something stronger.** One body's frame with the other's latent, commands
21.1 deg apart:

| frame from | latent from | matches `c10f10t10` | matches `c10f06t06` | follows |
|---|---|---|---|---|
| c10f10t10 | c10f06t06 | **4.79** | 21.64 | **the frame** |
| c10f06t06 | c10f10t10 | 21.59 | **5.84** | **the frame** |

Crossed rows score 4.79 and 5.84; uncrossed score 4.77 and 5.88. **Swapping the latent changes the
answer by 0.04 deg** — the decoder reads geometry from pixels and the latent contributes nothing to
that question.

**The two inputs end up with separate jobs.** The frame carries *which body*, the latent carries
*what movement* — `z` is 92.6% gait and 3.4% body (slide 6), and deleting it still costs 3.5x. That
division of labour is what the objective was supposed to produce and what reconstruction alone
never asks for.

---

## Slide 6 — What is inside the latent, with and without the cross-body loss

**What the cross-body loss does to the latent.** Split its variance by what explains it, and
separately ask what can still be decoded out of it:

| | Without | With |
|---|---|---|
| variance explained by **gait phase** | 81.9% | **92.6%** |
| variance explained by **which body it is** | 12.4% | **3.4%** |
| variance explained by neither, the interaction | 5.7% | 4.1% |
| **foot-contact pattern decodable from it** (8 patterns, majority class 0.172) | 0.729 | **0.732** |
| which body it is, decodable from it (4 bodies, chance 0.250) | 0.764 | **0.694** |

The body's share of the latent falls by a factor of **3.6** while the gait's share rises to 93
percent. The last two rows are the check that this is purification and not destruction: behaviour
comes out of the latent **slightly better** than before, so nothing was lost in the process — the
latent **stopped carrying a job that was never its own**, and kept the one that was.


---

## Slide 7 — The commands actually walk

Predicted commands driven open-loop through the same physics used to collect the data, on a body
never trained on. `m3d_cross` against its control `m3d_bracketed`, three clips. Cells are the mean
with the range in brackets; the last column counts clips where the cross-body run wins — same
episodes, same physics, so the comparison is paired.

| | IK | Control | With cross-body loss | cross wins |
|---|---|---|---|---|
| Forward distance, share of IK | 100% | 85% [74–91] | **90%** [89–91] | 1 / 3 |
| Heading deviation from IK | 0 deg | 11.8 [6.1–17.0] | **5.5** [0.6–12.3] | **3 / 3** |
| Commands outside the body's joint range | 0% | **6.1%** [5.8–6.2] | 6.4% [6.3–6.6] | 0 / 3 |
| Worst such excursion | 0 deg | 8.2 [4.0–**16.6**] | **3.9** [3.7–4.1] | 2 / 3 |

**Both walk, and on averages they are close. Read the ranges and the win column instead.**

**Heading is the only measure won outright** — closer to IK on every clip.

**The others say the cross-body run is steadier, not better.** It stays in a narrow band everywhere
(89–91% of the distance, worst excursion 3.7–4.1 deg) while the control matches it on two clips
then fails badly on the third — 74% of the distance, a 16.6 deg excursion. Both step outside the
joint range on ~6% of commands, so the *frequency* is a property of the task; what differs is the
worst case, and that decides whether a gait degrades gracefully or collapses.

> **Stated plainly:** on one clip IK walks almost straight (−0.8 deg) and both models veer, to
> +11.5 and +15.0. Neither reproduces a straight walk on demand.

![gait diagram, predicted vs IK](../results/wm/stage1_correct/gait/gait_stage1_m3d_cross_clip0.png)

Black is stance, white swing, 65 steps. Top block is the predicted commands, bottom the IK ground
truth; tripod alternation and stance durations line up, mean feet on the ground 3.02 of six against
the reference's 3.08. Video:
`results/wm/stage1_correct/gait/replay_stage1_m3d_cross_clip0.mp4`

![per-joint reconstruction on the held-out body](../results/wm/stage1_correct/figures/action_trace_m3d_cross_c08f09t09.png)

**Per joint** — black is ground truth, red the model, three clips end to end. The mean **R² 0.81**
averages three very different groups:

| joint | R² | RMSE |
|---|---|---|
| **TC**, fore-aft swing | **1.00 on all six legs** | 0.4–1.0 deg |
| **FT**, the knee | 0.81–0.91 | 4.0–4.5 deg |
| **CF**, the leg lift | **0.49–0.80** | 3.7–4.1 deg |

**CF is where it struggles, and in a specific way.** The red trace follows the *shape* of every
cycle and sits at the wrong *height* — it knows what the leg is doing and misplaces how high it
holds it. That is the same failure as slide 4's geometry read, where the decoder puts the coxa at
0.622 against a true 0.80: **the coxa sets leg height.** Two unrelated measurements land on the
same joint.

---

## Slide 8 — The limit: everything ties the femur to the tibia, because the data does

**Same model as the last four slides** — `m3d_cross`, same weights, same frozen encoder. Only the
body it is asked about changes.

| Training body | coxa | femur | tibia |
|---|---|---|---|
| c10f10t10 | 1.0 | **1.0** | **1.0** |
| c06f10t10 | 0.6 | **1.0** | **1.0** |
| c10f06t06 | 1.0 | **0.6** | **0.6** |
| c06f06t06 | 0.6 | **0.6** | **0.6** |

All four tie femur to tibia. Not by design — every body in the dataset where they differ is one
that does not walk (F36), so a training set of bodies that *do* walk is one where those two
segments have never moved apart.

| the same weights, asked about | deg | **R²** |
|---|---|---|
| `c08f09t09` — femur 0.9, tibia 0.9, **inside the range** | **3.44** | **+0.81** |
| `c10f10t08` — femur 1.0, **tibia 0.8**, the first time they differ | 13.35 | **−0.34** |

**Now ask three separate things what geometry `c10f10t08` has.**

| | coxa | femur | tibia |
|---|---|---|---|
| **The truth** | 1.00 | **1.00** | **0.80** |
| The trained decoder, from its output commands | 1.000 | **0.681** | **0.681** |
| The linear probe on the frozen encoder | 0.920 | **0.843** | **0.843** |
| The best any mixture of training bodies could say | 0.809 | **0.600** | **0.600** |

**All three give femur and tibia the same number** — a 5.2M-parameter decoder, a 4,227-parameter
readout of the raw encoder, and a mixture calculation with no learning in it at all, making the
identical mistake. The last row is why: **no combination of bodies in which the two always move
together can pull them apart.**

**And the size of that gap is geometry, not experiment.** The closest all-tied point to (1.00,
0.80) is (0.90, 0.90), a distance of **0.141** — exactly the mixture gap the probe reports, for any
all-tied training set, whichever bodies were used.

### Where the failure sits

![per-joint reconstruction on the tibia-short body](../results/wm/stage1_correct/figures/action_trace_m3d_cross_c10f10t08.png)

| joint | what it moves | R² |
|---|---|---|
| **TC**, thorax-coxa | swings the leg fore and aft | **+0.46 to +0.83 — still works** |
| **CF**, coxa-femur | lifts the leg | −0.53 to +0.05 |
| **FT**, femur-tibia | the knee | **−0.45 to −3.99** |

**The joint that fails worst is the one between the two segments the data could not separate**, and
the joint not involving the tibia still works. The body differs in the tibia and nothing else, and
the damage is localised accordingly — a fingerprint of the data gap, not a model that simply got
worse.

**Not one unlucky body.** Every unseen femur/tibia ratio available, same weights:

| held out | femur/tibia | deg per joint | **R²** |
|---|---|---|---|
| c10f10t08 | 1.04 | 13.35 | **−0.34** |
| c10f09t07 | 1.07 | 11.63 | **−0.14** |
| c10f08t06 | 1.10 | 10.51 | **−0.33** |

**Negative on all three** — worse than memorising the body's average posture — at 10–13 deg per
joint against a command spread of 11.7, comparable to the whole signal.

**The fix is more bodies where femur and tibia differ.** The scene generator already supports it.
A data gap, not a loss or architecture problem, and no regulariser touches it because the
information was never there.

---

## Slide 9 — Testing the diagnosis instead of asserting it

Slide 8 ends with an explanation, and an explanation makes a prediction: **if the femur and tibia
are tied because every training body ties them, then adding bodies where they differ should untie
them.** Two such bodies were generated and checked to walk: `c10f09t07` and `c10f08t06`, at
femur/tibia 1.07 and 1.10.

**Testing that needs two purpose-built runs, and here is why.** If the six-body set simply had more
clips, any improvement could be put down to more data. So a matched pair was trained instead —
**`tib_cross`** on four tied bodies and **`bracket_cross`** on those four plus the two decoupled
ones, at **96 training clips each**, holding out the same body from the same clips.

| | `tib_cross` — 4 bodies, femur tied to tibia | `bracket_cross` — 6 bodies, decoupled |
|---|---|---|
| training clips | 96 | 96 |
| bodies | 4 x 24 clips | 6 x 16 clips |
| held out | `c10f10t08` | `c10f10t08`, the same 20 clips |
| **the model** | **12.67 deg** | **3.27 deg** |
| **R² against the body's own mean** | **−0.78** | **+0.89** |

Only one thing differs between the columns: whether the training set contains bodies whose femur
and tibia move apart.

**A 3.9x improvement, and it crosses zero.** The four-body run is worse than memorising the
held-out body's average posture; the six-body run explains 89% of its variance and beats every
baseline. Filling the gap the diagnosis named does not merely improve extrapolation — **it removes
the failure.**

![the coverage experiment](../results/wm/stage1_correct/figures/coverage_experiment.png)

**A second prediction, costing no GPU at all.** Refit the encoder probe on the enlarged set and its
error on the held-out body should fall. It did — **0.082 → 0.034** — and the specific thing the
diagnosis named is what moved:

| | coxa | femur | tibia |
|---|---|---|---|
| the truth | 1.00 | **1.00** | **0.80** |
| probe fitted on the 4 tied bodies | 0.955 | **0.819** | **0.819** |
| probe fitted on all 6 | 0.973 | **0.954** | **0.772** |

**The four-body probe gives the femur and the tibia the identical number, 0.819 and 0.819** — it
cannot separate two quantities that never varied apart in anything it was fitted on. With the two
decoupled bodies added, the gap between them opens to **0.182 against a true 0.200**, recovering
91 percent of a separation that was previously invisible. The tying broke, and it broke in the
encoder's readout before any decoder was trained.

**Two things to state, because they qualify the number.**

**The held-out body moves inside the hull.** Both sides hold out `c10f10t08` at ratio 1.04. The
four training bodies all sit at 0.83, so for them that is extrapolation; the six-body set spans
0.83–1.10, so for them it is interpolation. **That is exactly what coverage is supposed to do** —
it converts an extrapolation problem into an interpolation one — but it means the two runs do not
face equally hard tasks, and the 3.9x measures the conversion rather than the same task done
better.

**These runs are 10 epochs and peaked at epoch 10**, still improving when the budget ended, where
the m3d pair on the earlier slides had 50. They are not converged, so both figures are a lower
bound rather than a settled value.

---

## Slide 10 — The same measurement predicts, before training, which bodies will transfer

The probe from slide 4 was built to answer a different question. Compared against what the trained
models actually did, it turns out to predict the outcome before a single epoch is run.

| Training set | Held out | **Mixture gap** | **Probe error** | Model, deg | R² | Outcome |
|---|---|---|---|---|---|---|
| 4 bodies, spanning | `c08f09t09` | **0.000** | **0.021** | **3.44** | +0.81 | **beats copy-nearest, 3.47** |
| 6 bodies, decoupled | `c10f10t08` | **0.063** | **0.034** | **3.27** | **+0.89** | **beats every baseline** |
| 4 bodies, all tied at 0.83 | `c10f10t08` | **0.141** | **0.082** | 12.67 | −0.78 | loses to the body's own mean |

**The bottom two rows are the same held-out body.** Nothing about the test changes between them —
same geometry, same clips, same frames. Only the training set does, and the outcome flips from
worse-than-memorising-a-pose to explaining 89 percent of its variance. That rules out the obvious
objection to a table like this, that some bodies are simply harder than others.

Both cheap columns order all three correctly. **The mixture gap is pure geometry** — the distance
from the held-out body's segment scales to the nearest convex mixture of the training bodies' —
and needs no encoder, no model and no data at all. **The probe error** needs one CPU pass of the
frozen encoder and a ridge fit, 4,227 parameters.

- **Neither threshold is zero.** The six-body split sits at a gap of 0.063 and succeeds
  comfortably; the boundary lies between 0.063 and 0.141. What matters is being *near* the hull
  the training bodies span, not inside it.
- **Cost: a few minutes on CPU, no training at all.** Cost of finding out by training: hours of
  GPU per run.

**This turns the limitation into a tool.** Before committing to a train/held-out split, fit the
probe on the training bodies and read off how well it recovers the held-out one. A large error
says the split is asking for a direction the data does not span, and the run will not answer the
question you meant to ask.

Speaker note, and this is the honest version: this was not designed as a diagnostic. The probe was
built to ask whether the encoder carries geometry at all, and only later compared against what the
runs did. That is worth more than a planned result, because the measurement could not have been
chosen to fit the outcome — but three points is enough to establish the ordering and not enough to
set a numeric threshold.

---

## Slide 11 — At one speed, one frame nearly determines the command

**Scope, and it decides how far this claim reaches.** Everything below is measured on **forward
walking at a single speed**, where the gait is periodic and one frame fixes the phase. It is a
structural fact about *that* task, not about the model.

> **Correction, added after F142.** This slide originally read "it does not extend to locomotion
> with varying speed" and pointed at a Stage 2 slide showing the action mattering once the magnitude
> varies. **That reading did not survive.** F142 asked the same question with a null-action baseline,
> on twelve conditions that include four speeds, and measured `null/real` = **1.03** — the action is
> worth under 3% whether or not the magnitude varies. **What this slide observed at one speed turned
> out to be general**, and slides 14 to 19 are what followed from taking it seriously.

### The transition is worth about a third

Substitute what the ITM is given as its second frame, on the held-out body, 195 transitions:

| what the ITM is given as `e_{t+1}` | control | with the cross term |
|---|---|---|
| the real next frame | 3.71 deg | **3.37 deg** |
| **a copy of `e_t`, no transition at all** | **1.28x** | **1.34x** |
| `e_{t-1}`, a wrong transition | 1.67x | 1.65x |
| a frame from a random other time | 3.54x | 3.44x |
| the latent zeroed entirely | 2.88x | 3.48x |

Read the middle rows first: **a wrong transition hurts more than a missing one**, and nonsense
hurts most, so the latent is genuinely sensitive to what the second frame contains — it is not
ignoring it.

Then the second row, which carries the conclusion. **Removing the transition entirely costs 28 to
34 percent.** The other two thirds of what the decoder needs is already in `e_t` alone.

The last row belongs to slide 5's division of labour: **deleting the latent costs the cross-term
run more, 3.48x against 2.88x.** Once the frame carries the body, `z` is left carrying the
movement, and the decoder cannot do without it.

### And the horizon does not matter

| Predict, from a single frame | now | 8 frames ahead | 32 frames ahead |
|---|---|---|---|
| Error, deg (signal spread 11.3 deg) | 3.0 | 3.4 | **2.9** |

- **Predicting 32 frames ahead is as accurate as predicting the present.** The commands come from
  inverse kinematics, which is open loop, and the gait is periodic with a measured cycle of 19
  frames — so one frame fixes the phase and every horizon after it follows.
- Six coordinated legs remove the ambiguity a single leg would have: one frame already identifies
  which feet are swinging with 81.5% accuracy, against 50% by chance.
- A second frame is worth only **1.11x** on the step-to-step change, so almost nothing is left in
  the transition for the latent to carry.

Measured on `c10f10t10` in `ik_walk_m3d_clean`, ridge from mean-pooled frozen encoder features,
18 clips fitted and 8 held out. Pooling all five bodies instead gives the same fractions of the
signal — 26, 30 and 25 percent — against a spread widened to 15.0 deg by the between-body
variance, so the claim does not depend on that choice.

**Why this bounds the design, and exactly how far.** At one speed the joint command cannot be where
the latent earns its keep, because the frame nearly determines it alone. That is a property of the
task, not a fault in the model, and it is the reason the next slide scores the forward model on
rolling the world forward instead.

**It bounds the action-decoding path at a fixed magnitude, and nothing wider.** Let the speed vary
and the same measurement reverses: the prediction responds to actions of a different magnitude
(`/mean-z` 0.485 against 0.951 within one speed). **Read the reversal as a property of that
metric's baseline, not as a rescue** — see the correction on slide 11: with a null-action baseline
the action is worth under 3% at every magnitude (F142).

---

## Slide 12 — The forward model was being judged on the wrong task

Every measurement above asks whether forward prediction helps **reconstruct the action**. It
does not. That is not what a forward model is for.

Closing it on its own output and rolling it forward, with the true latents supplied so the
module is isolated, on the held-out body, 162 rollouts:

| steps ahead | forward model | hold the frame still | constant velocity | **beats holding still by** |
|---|---|---|---|---|
| 1 | 1.39 | 2.11 | 5.78 | **1.52x** |
| 3 | 1.78 | 3.05 | 27.6 | **1.72x** |
| 5 | 2.12 | 3.57 | 66.0 | **1.69x** |
| 10 | 2.98 | 4.36 | 236.5 | **1.46x** |

- It beats a frozen world at **every horizon out to ten steps**, and beats constant velocity by
  two orders of magnitude. **The forward model can roll the world forward.**
- Reading the source paper confirms the mistake was ours: in LAC-WM the Motion Decoder is an
  **auxiliary regulariser**. The deployed system predicts future embeddings, rolls them eight
  steps, and picks actions by comparing predicted futures to a goal image. The action decoder is
  not the system's output. **We had made the auxiliary term the whole evaluation.**
- **The cross-body loss costs nothing here.** The control `m3d_bracketed` scores 1.52x, 1.72x,
  1.69x and 1.47x at the same horizons — **identical to two decimal places**. The term that fixed
  morphology reading leaves the world model's own competence untouched, which makes sense: it
  never touches the prediction loss.
- Honest limits: holding still is a weak baseline, 1.5-1.7x over it is real but not dramatic, and
  the margin decays with horizon. **And this is measured on a body the model trained near.** What
  the same module does on a genuinely different robot is a Stage 2 question, and the answer is not
  this one.

Speaker note: this is why the earlier slides say "the command can be read off one frame" rather
than "the world model does nothing". Those are different claims and only the first is supported.

---

## Slide 13 — Stage 2: the thesis, the gap, and where it stood

**The contribution.** Plan toward a goal defined in a coordinate **shared across bodies whose action
spaces have nothing in common** — 18-DOF six-legged insect, 12-DOF quadruped — with no kinematic
model, no retargeting, no controller already running on the target. **The only thing the two robots
share is what a camera sees.**

```
                     needs a kinematic model
                              ▲
                URMA          │          X-Morph
             (joint tokens    │      (URDF + retargeting)
              over the tree)  │
    ──────────────────────────┼──────────────────────────►  crosses leg count
                              │
                QWM           │        ███ THIS WORK ███
        (morphology params,   │        video only, 18-DOF ↔ 12-DOF
         quadrupeds only)     │
                              │        LAC-WM is off this map: manipulation,
                     needs no kinematics       where end-effector pose is already
                                               shared, and it *selects* over a VLA
```

**Everything that crosses leg count is handed a body model; everything that needs no body model
stays inside one leg count.** The lower-right quadrant is empty, and locomotion has no end-effector
pose to retreat to — 18 and 12 joint targets share no dimension.

```
   ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
   │  GOAL REPRESENTATION │──►│  WORLD MODEL         │──►│  CONTROLLER          │
   │  shared coordinate   │   │  dynamics            │   │  a robot being driven│
   ├──────────────────────┤   ├──────────────────────┤   ├──────────────────────┤
   │  ✔ PROVEN            │   │  ✘ DOES NOT WORK     │   │  ✘ NOT DONE          │
   │  70% vs 28% chance   │   │  and slides 14-19    │   │  every number here is│
   │  all three families  │   │  are why — the cause │   │  offline selection   │
   │  survives the control│   │  is measured         │   │  among recorded clips│
   └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
          slide 21                  slides 14-19               slides 25-26
```

**What changed since the last update.** Three claims this deck previously made are withdrawn, each
by a control that had not been run: that the contrastive term is what crosses embodiments, that the
closed loop selects behaviours on the quadruped, and that sideways motion fails everywhere. **The
measurements were real; the conclusions were not.**

**And one more since:** the middle box read *"⚠ diagnosed — Context Collapse, named and cited, fix
identified."* **We rebuilt ActSWM's fix and it failed** (slide 14). Context Collapse names the
symptom; slides 15 to 17 are the cause underneath it.

**Slide 11 is where the next act begins.** It reported, as an oddity at one speed, that a single
frame nearly determines the command. **Slides 14 to 19 are what happened when that oddity was
chased** — and slide 11 now carries a correction, because the oddity was not confined to one speed.


---


---

# Part 2 — The attempt, and what each failure measured

## Slide 14 — What we set out to build, and whose method it was

**The problem, in the field's own words.** A world model conditioned on an action should predict a
different future for a different action. **ActSWM (2607.26712) names the failure when it does not:
Context Collapse** — the predictor extrapolates from the observation context and becomes insensitive
to the action channel. **UWM-JEPA (2605.25313, §4) names the same thing from the training side**: a
teacher-forced target already contains the action's effect, so an *action-invariant solution* fits
the loss perfectly. **AHA-WAM (2606.09811)** takes adjacent-frame redundancy as a design premise and
splits its horizon asynchronously to avoid it.

**So the diagnosis was not ours and neither was the proposed fix.** ActSWM's remedy is a hinge that
forces the rollout under the real action apart from the rollout under a null action, plus a frozen
readout that must recover the action from the prediction. We rebuilt our pretraining to match it,
with their settings where we had evidence for them and our own where we did not.

| ActSWM setting | ours | why the difference |
|---|---|---|
| margin 0.3 | **0.1** | at 0.3 the term overshoots, switches itself off and collapses — separation read 0.019, 0.137, 0.496, 0.008 with its gradient dying to 6e-5 (F141); at 0.1 it rises and holds on both bodies (F141) |
| K = 12 rollout steps | **3** | our rolled prediction crosses "worse than a frozen frame" by five steps (F131, F141); hinging past that trains on noise |
| H = 32 context frames | **1** | our forward model conditions on one frame — 32 is a different architecture, not a hyperparameter |
| `lambda_sig` (SigReg) | **not used** | SigReg is LeWM-specific; this is V-JEPA2 and the term was not guessed in |

**Everything below was pre-registered.** Each run's reading was written down before it started, and
the entries in `doc/FINDINGS.md` show the criterion above the result. That matters for what comes
next: **six of these runs came out negative, and none of the criteria moved afterwards.**

---

## Slide 15 — Six independent measurements, one answer

**Read this as a chain of eliminations, not a list of failures.** Each row closed a hypothesis about
*where* the missing action-sensitivity lives, and each returned a number.

| # | hypothesis | measurement | result |
|---|---|---|---|
| F141/154 | the objective is wrong; ActSWM's hinge will fix it | full 50-epoch rebuild, both bodies | prediction fine at one step, **3.1–4.1× worse than a frozen frame at two**; no sensitivity gained |
| F142 | maybe the weighting was off | swap the real action for a null one and re-predict | **`null/real` = 1.03** — the true action is worth **under 3%** |
| F141 | one frame is too short a step | retrain at three-frame spacing, hinge off | 1.032 against 1.028. **Frameskip changes nothing** |
| F144 | the action lives in what the action-blind model *misses* | probe the residual for the command | adds **0.009 R²** over the bare frame |
| F148 | use a motion-organised representation instead | `e_t+1 − e_t`, same probes | redundancy survives; **cross-body transfer destroyed** |
| F154 | force two different actions from one state and look | bit-identical reset, branch, measure in embedding space | futures **134 mm and 30° apart** in the world sit at **1.1×** the noise floor in `e` |
| F146 | our own body-coordinate term caused it | remove it entirely, controlled | it got **worse** — the term was a small *positive* contributor |

**F146 is the control a committee asks for and it is the one that makes the rest mean something.**
"Did your own objective cause this?" is answerable: no, and removing it hurts.

**F154 is the sharpest, and it is worth watching rather than reading.**

![two futures a human separates instantly; the encoder does not](../results/cf_confirm/insect_forward-vs-turn.mp4)

`results/cf_confirm/insect_forward-vs-turn.mp4` — identical commands to the branch frame, then one
arm walks on and the other turns. **42 degrees apart by the end.** In embedding space that is
**1.1× the noise floor** — as far apart as two runs of the *same* command.

**The gap between what you see there and what the encoder encodes is the finding.**

---

## Slide 16 — The number underneath all six

**One quantity explains every row above.** In third-person locomotion video, the joint command is
readable from a *single frame*:

| | one frame | frame pair | what the transition adds |
|---|---|---|---|
| stick insect | **0.779** | 0.887 | +0.108 |
| — turning only | **0.931** | 0.957 | +0.026 |
| B1 | 0.161 | 0.342 | +0.182 |

**Prior work found the ingredient.** Yeom et al. (2606.07687) showed V-JEPA carries
inverse-dynamics-recoverable action structure (R² 0.40 frozen, 0.85 with a head) and **noted that
CALVIN's static tabletop lets per-frame appearance stand in for temporal context.**

**We measured how far that goes in periodic locomotion, which they did not.** A gait makes the pose
a near-complete statement of the command: one frame recovers **88%** of what a pair recovers on the
insect, and **97%** on turning.

**And this is the distinction that makes it a result rather than a restatement.** Inverse-recoverable
is not forward-necessary. The command is recoverable at R² 0.89 **and contributes under 3% of
one-step prediction error.** A model with nothing to gain from the action channel will not use it,
and no weighting on top can create a signal the task does not contain.

Speaker note, and the honest scope: this is measured on our two robots, in simulation, on twelve
behaviour conditions. We did not measure it on manipulation.

---

# Part 3 — The principle

## Slide 17 — Pose determines the future, so the action is redundant

> **When the agent's own configuration is visible and determines what happens next, the action
> carries no information the observation lacks — and a model trained to predict the next observation
> will ignore it.**

**It is not "the agent is in frame."** Manipulation puts the arm in frame and its world models work.
The condition is stronger: the visible configuration must determine the **future**, not merely reveal
the current command.

```
  THIRD-PERSON LOCOMOTION            MANIPULATION                   EGOCENTRIC LOCOMOTION
  ┌──────────────────────┐          ┌──────────────────────┐       ┌──────────────────────┐
  │   ▄▟█▙▄  whole body  │          │  arm ──►  ▢ object   │       │      the world       │
  │  ╱ │ ╲   visible     │          │          (separate)  │       │    (body unseen)     │
  └──────────────────────┘          └──────────────────────┘       └──────────────────────┘
   pose = command  R² 0.78           pose ≠ outcome                 pose invisible
   pose ⇒ next pose (gait)           object state is free           future depends on action
   ────────────────────────          ────────────────────────       ────────────────────────
   ACTION REDUNDANT                  action needed                  action needed
   world model collapses             world model works              ← the prediction
```

**Periodicity is what makes locomotion the severe case.** A gait is a limit cycle: the pose fixes the
phase and the phase fixes the next pose, so the pose determines not just what the robot is doing but
what it is *about to* do — which is the quantity a forward model is trained on.

**Two interventions separate rhythm from redundancy:**

| break the gait with… | did the action gain value? |
|---|---|
| random command noise (F150) | **yes** — gap +0.084 → **+0.198** |
| real stops, speed breaks, turn onsets (F152), verified to reach the robot at 2.1–5.6× | **no** — +0.061 |

**So the cause is not rhythm.** Any command a controller would actually issue is visible in the pose.

`results/deck/principle_allo_vs_ego.mp4` — the same behaviour under both viewpoints.

![the principle, seen](../results/deck/principle_allo_vs_ego.mp4)

Speaker note: the two panels are the same condition from two collections, not the same physical run —
CoppeliaSim does not repeat (F95).

---

## Slide 18 — The principle explains results that are already published

**The pieces are not ours. The connection is.**

| system | agent in frame? | pose determines the future? | works? | what they claim | what the principle adds |
|---|---|---|---|---|---|
| **Demo-JEPA** (2605.20811) — cross-embodiment latent-goal planning | yes, **and the object** | **no** — object state is independent of arm pose | **yes** | a shared latent goal space across embodiments, built by retargeting | why it *can* work: the arm's pose says nothing about where the block ends up, so the action stays informative |
| **Hu et al.** (2207.03386) — egocentric locomotion self-modelling | **no** | **no** — the body is unseen | **yes** | egocentric video suffices for locomotion self-modelling | **why egocentric is necessary**, which their paper does not claim |
| **ours, third-person locomotion** | yes | **yes** — the gait is a limit cycle | **no** | — | F141–F154 is the measurement of the collapse |
| CALVIN-style static manipulation | yes | partly | mixed | — | names Yeom's own exception rather than noting it |

## Three separate things, kept separate

```
  PRIOR WORK FOUND                          ┌──────────────────────────────────┐
    ActSWM       named Context Collapse,    │ four separate observations,      │
                 proposed rollout separation│ none connected to the others     │
    UWM-JEPA     named the action-invariant │                                  │
                 solution; fix = counterfac.│                                  │
    AHA-WAM      assumed adjacent-frame     │                                  │
                 redundancy as a design premise                                │
    Yeom et al.  noted static scenes let a  │                                  │
                 single frame substitute    └──────────────────────────────────┘
         │
         ▼
  WE MEASURED WHERE THEY DID NOT
    how far the substitution goes in PERIODIC locomotion
      one frame = 88% of a pair · 97% on turning
      and the recovered command is worth <3% of prediction
         │
         ▼   INVERSE-RECOVERABLE  ≠  FORWARD-NECESSARY
         │   so in locomotion the action-invariant solution is NEAR-OPTIMAL
         │   and a target-side fix has nothing better to converge to
         ▼
  OURS
    the mechanism that joins the four, measured in the severe case
    ⇒ a VIEWPOINT choice — not an objective, not a target — decides whether
      an action-conditioned world model can exist for a given task
```

**Two limits on this slide, stated because a reader will look for them.** We tested the
**residual-target** version of UWM-JEPA's route and closed it (F144, adds 0.009 R² over the bare
frame); **we did not train their counterfactual target and do not claim it fails.** And the rows
above read published results *through* the principle — they are not re-measurements of those systems.

---

## Slide 19 — Three failures, one cause — and one that is not

```
   candidate scoring          teacher-student           action-conditioning
     F118 · F126                F135 · F136                 F141-F154
         │                          │                            │
   swap the goal for            teacher ranks              six routes closed
   another behaviour:           behaviours but not         action worth
   costs 3-7 points             perturbations              <3% of prediction
   delete the rollout:                │                            │
   costs nothing                      │                            │
   41% vs 42%                         │                            │
         ▼                            ▼                            ▼
   ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
   │   MEASURED    │          │    PARTLY     │          │   MEASURED    │
   └───────┬───────┘          └───────┬───────┘          └───────┬───────┘
           │                  physics separated them             │
           │                  by 0.1304 vs 0.1299 — a            │
           │                  ranker cannot order outcomes       │
           │                  that are the same                  │
           │                          │                          │
           └──────────┬───────────────┴──────────────┬───────────┘
                      ▼                              ▼
        ╔═════════════════════════════╗   ╔═══════════════════════════╗
        ║  POSE DETERMINES THE FUTURE ║   ║ a second obstacle the     ║
        ║  removed by egocentric      ║   ║ viewpoint does not touch  ║
        ║  Q1:  0.779 → 0.293         ║   ╚═══════════════════════════╝
        ╚═════════════════════════════╝
```

![the teacher made it worse than cloning alone](../results/wm/closed_loop/f142_video/f144_labelled.mp4)

`results/wm/closed_loop/f142_video/f144_labelled.mp4` — **recorded walk 100% · cloning alone 36% ·
cloning + world-model teacher 31%.** The teacher subtracts.

**What this licenses:** two of the three share a mechanism, and the mechanism is gone.
**What it does not:** that any of them revive. **They were tested and they did not** -- slide 25.

**Measured** on two robots, twelve conditions, in simulation. **Not measured:** manipulation — slide
18 reads published results through the principle rather than re-measuring them.

---

## Slide 20 — Egocentric breaks the redundancy

**The cheapest test that could answer it**, built to be discarded: four textured walls and a ceiling
around the spawn, the camera moved onto the robot's head. **Not an environment.**

**The leak guard ran first and Q1 was not read until it passed.** Room colour predicts heading on
held-out clips at **0.34× chance** (insect) and **0.50×** (B1) — below chance.

| stick insect | one frame | pair | gap |
|---|---|---|---|
| third-person | **0.779** | 0.887 | +0.108 |
| **egocentric** | **0.293** | 0.578 | **+0.285** |

**Single-frame readability falls by 0.486; the gap nearly triples.** Sideways reads **−0.008** from
one frame — nothing at all — against 0.609 third-person. **This is the first thing in the F141–F154
chain to move the quantity all six were trying to move.**

`results/deck/q1_turn_both_bodies.mp4` — the same room, the same slot, two different bodies.

![the world turns the same way under either robot](../results/deck/q1_turn_both_bodies.mp4)

**What it does not say.** Q1 shows the action is no longer redundant with the pose. It does **not**
show that a trained world model uses the transition — this project's own record is of signals that
existed and were then ignored. **That measurement needs the trained model.**

---

## Slide 21 — The shared coordinate survives the change of view

**The risk this had to clear.** A head camera could break the redundancy and simultaneously destroy
the one cross-body result the project has. Fitted on the insect's egocentric embeddings, applied to
the B1's **with no refitting**:

| | forward | lateral | yaw |
|---|---|---|---|
| third-person, B1 unrefitted | 0.63 | 0.43 | **0.07** |
| **egocentric, B1 unrefitted** | **0.50** | **0.39** | **0.64** |

**Yaw transfer goes from 0.07 to 0.64.** Turning is the channel this project has fought since F127
and nearly lost in F154, and it is the one the view change helps most — which the physics predicts,
since a head camera sees rotation as global image flow whatever body is underneath.

**Forward and lateral fall**, and the within-insect fit falls further (0.98 → 0.77 on forward).
**The coordinate is harder to read from a head view and it still crosses.** That trade is the honest
summary; slide 23 is about recovering the part that was lost.

---

## Slide 22 — Contribution 1: a cross-embodiment coordinate that needs no correspondence

**What Demo-JEPA (2605.20811) requires to align two embodiments:** end-effector retargeting to
manufacture paired data, and GTCC for temporal alignment.

**What ours requires, read out of the code rather than asserted:**

| | Demo-JEPA | ours |
|---|---|---|
| paired data across bodies | **retargeting** | **none** — the target is each clip's own measurement; `lambda_cross`, the term that *would* pair them, is **0.0** in every checkpoint these results come from |
| temporal alignment | **GTCC** | **none** — no DTW, no shared clock |
| hand labels | — | **none** — forward and lateral from differencing position, yaw from the quaternion |
| transfer | — | fit on the insect, applied to the B1 **with no refit**, one shared head on `z` |

**Why it is possible: the target is a quantity both bodies already have, not a correspondence that
has to be built.** Froude scaling — dividing by `sqrt(g·h)` — is what puts them on one axis: the
insect averages 0.155 and the B1 0.159 across a fourfold size difference (F49).

**The scope limit, stated because a committee will find it.** The target is differenced from
simulator-recorded body pose. On hardware that is odometry or motion capture. **This coordinate is
regressed onto a measured physical quantity; it is not learned from pixels alone**, and Hu et al.
carry the same requirement. Anyone claiming "vision-only" here would be overclaiming.

**And it buys exactly three channels.** Anything the two bodies do not share — joint spaces, gaits,
contact patterns — is not carried by it, which is what F99 and F73 found when the same question was
put to joint targets.

---

## Slide 23 — Contribution 2: embodiment-invariant ego-motion, tested feasible

**The problem the view change creates.** An egocentric camera carries two things at once: **where the
body is going**, which both robots share, and **how the body shakes getting there**, which is a
six-legged tripod on one and a trot on the other — 15.6° of yaw sway against 6.8°.

**They are separable, measured.** Decomposing each clip's camera yaw into a linear trend and the
rest: the gait sits at **6 cycles per clip on both bodies**, the net turn at 0 to 1, and they are
comparable in size (gait sd over turn sd 0.79 and 0.96).

**Removing it helps, and helps across bodies specifically:**

| | insect held-out | | | **B1 unrefitted** | | |
|---|---|---|---|---|---|---|
| | fwd | lat | yaw | fwd | lat | yaw |
| egocentric | 0.71 | 0.29 | 0.61 | 0.45 | 0.38 | 0.57 |
| **gait removed** | 0.73 | 0.25 | 0.62 | **0.47** | **0.46** | **0.61** |

**The within-body fit does not improve and the cross-body fit does.** That asymmetry is the
hypothesis's own signature — a generic denoising would move both — and **lateral comes back past its
third-person value** (0.46 against 0.43).

**Status, stated precisely: proven feasible, not done.** The removal here is three harmonics of one
frequency estimated per clip, subtracted linearly. **That a projection this blunt works is the
argument for a learned version; it is not evidence that a learned version will do better.** Forward
does not recover (0.47 against 0.63 third-person), so part of that drop is something other than
gait shake and remains unexplained.

**No one has this.** Hu et al. use egocentric locomotion but do not separate gait from ego-motion;
Demo-JEPA aligns embodiments by retargeting rather than by removing body-specific motion.

---

# Part 5 — Where this stands

> **How to read Part 5.** Everything here is about **motor babble** — the standard way a robot that
> nobody has a controller for gets bootstrapped. B1 results and gecko results are kept apart: B1 was
> in pretrain, gecko never was. **All gecko work is on Slide 31 alone** and appears in no flow
> diagram or result table above it.
>
> **บทพูด (TH).** ตั้งแต่ part นี้ไปคือเรื่อง **motor babble** — วิธีมาตรฐานที่ใช้ตั้งต้นหุ่นที่ยังไม่มี controller
> ขอแยก B1 กับ gecko ให้ชัด: **B1 เคยอยู่ใน pretrain แต่ gecko ไม่เคย**
> **เรื่อง gecko อยู่ที่สไลด์ 31 หน้าเดียว** ไม่ปนกับแผนภาพหรือตารางไหนข้างบนเลย

---

## Slide 23.5 — Controller vs. what we test

```
  A CONTROLLER / POLICY                    WHAT WE ACTUALLY TEST (a "closed loop")
  ────────────────────                     ──────────────────────────────────────
  state ──▶ network ──▶ joint torques      goal ──▶ score 12 RECORDED clips ──▶ replay the winner
  invents the motion itself                picks from motions that already exist
  runs at 50 Hz forever                    one pick per step, 65 steps, then stops
  can produce a motion nobody recorded     can NEVER produce a motion nobody recorded
```

| | a controller/policy | our closed loop |
|---|---|---|
| where the motion comes from | invented by the network | **a library of 12 recorded clips** |
| can it fall over? | yes — it is real physics | **no — kinematic replay, the body is posed directly** |
| what "survival" proves | the policy is stable | **nothing. It cannot fall by construction** |
| what it DOES prove | — | **whether the goal was read and the right motion chosen** |

**We do not have a controller, and this deck never claims one.** What is measured is *selection*:
given a goal expressed in a shared physical coordinate, does the system pick the right behaviour for
a body it was never trained on. The motion itself is always something a human recorded earlier.

> **บทพูด (TH).** สไลด์นี้สำคัญมาก และผมเองก็สับสนบ่อย ขอแยกให้ชัด
> **Controller/policy จริง ๆ** = เอา state เข้า แล้วเน็ตเวิร์ก **คิดท่าเดินขึ้นมาเอง** ส่งแรงให้ข้อต่อ วิ่งตลอดเวลา และ **ล้มได้**
> **สิ่งที่เราทดสอบ (closed loop)** = มีคลิปที่อัดไว้แล้ว 12 คลิป ระบบแค่ **เลือกว่าจะเล่นคลิปไหน** ให้ตรงเป้า แล้ว replay
> เพราะฉะนั้น **มันล้มไม่ได้เลยโดยโครงสร้าง** — ตัวเลข "รอด 3/3" จึงไม่ได้แปลว่าเก่ง มันแปลว่าไม่มีอะไรให้ล้ม
> สิ่งที่มันพิสูจน์จริง ๆ คือ **อ่านเป้าหมายถูกไหม และเลือกท่าถูกไหม** เท่านั้น — **เรายังไม่มี controller และเดคนี้ไม่เคยเคลมว่ามี**

---

## Slide 23.6 — Froude, and how a goal is made

**Froude number** = speed made dimensionless by body size, `Fr = v / sqrt(g · L)`, where `L` is
roughly leg/hip height. Two robots of different size walking "the same way" get the **same Froude**
even though their m/s differ. That is the entire reason a hexapod's goal can mean anything to a B1.

| | hexapod | B1 | gecko |
|---|---|---|---|
| hip height `L` | ~0.09 m | ~0.56 m | **~0.06 m** |
| a normal walk, in m/s | 0.13 | 0.29 | 0.013 |
| **the same walk, in Froude** | **~0.13** | **~0.13** | **~0.04** |

Three channels are used, all dimensionless: **forward**, **lateral** (strafing), **yaw** (turning).

```
  HOW THE GOAL IS PRODUCED — two options, deliberately separated
  ─────────────────────────────────────────────────────────────
  physics goal :  read the recorded NUMBER off the source clip        (privileged, a sanity ceiling)
  vision goal  :  watch the source clip's video → ITM → body_head     (what deployment would do)

  HOW A CANDIDATE IS SCORED — two options, deliberately separated
  ──────────────────────────────────────────────────────────────
  DIRECT   a ──▶ projector ──▶ z ──▶ body_head ──▶ Froude     (no world model in the loop)
  ROLLOUT  e_t + z ──▶ FTM ──▶ imagined future ──▶ ITM ──▶ body_head ──▶ Froude   (world model in)

  score = | candidate's Froude − goal's Froude |   → pick the smallest
```

Crossing the two gives the 2×2 (A/B/C/D) on Slide 30. They are separated **on purpose**: bundling
them is what confounded an earlier version of this test.

> **บทพูด (TH).** ก่อนดูผล ขออธิบายสองคำที่ใช้ตลอด
> **Froude** คือความเร็วที่หารด้วยขนาดตัว `v / sqrt(g·L)` — หุ่นคนละขนาดที่ "เดินเหมือนกัน" จะได้ค่า Froude เท่ากัน
> ทั้งที่ m/s ไม่เท่ากัน **นี่คือเหตุผลเดียวที่เป้าหมายจากแมลงจะมีความหมายกับ B1 ได้** เราใช้ 3 ช่อง: เดินหน้า / ไถลข้าง / เลี้ยว
> **เป้าหมายผลิตได้ 2 แบบ**: อ่านตัวเลขที่อัดไว้ตรง ๆ (physics — เป็นเพดานอ้างอิง มีข้อมูลพิเศษ) กับ ดูวิดีโอแล้วถอดออกมา (vision — แบบที่ใช้จริงตอน deploy)
> **การให้คะแนนผู้สมัครก็ 2 แบบ**: **Direct** = แปลง action เป็น z แล้วอ่าน Froude เลย ไม่ใช้ world model
> **Rollout** = ใช้ world model จินตนาการอนาคตก่อน แล้วค่อยอ่าน — แล้วเทียบว่าใครใกล้เป้าที่สุด
> ที่ต้องแยกสองแกนนี้ออกจากกัน เพราะเวอร์ชันก่อนหน้าเราเอามามัดรวมกัน ทำให้สรุปไม่ได้ว่าตัวไหนเป็นตัวปัญหา

---

## Slide 24 — What the coordinate fixed

**Scoring in the right space is what turned selection on.** Frame distance reads the current frame,
not the goal. Rescoring by body-motion (Froude) distance fixes that immediately.

| selection rule | same-robot | cross-embod., 1ch | cross-embod., 3ch |
|---|---|---|---|
| frame/embedding distance | 18-23% (28% chance) | — | — |
| **Froude distance, no rollout** | **76-86%** | 35-38% | **68-70%** |
| + FTM rollout added back | — | — | 33-44%, turning destroyed |

| strafing | 1-channel | 3-channel |
|---|---|---|
| | 13-25% (17% chance) | **86-100%** |

### Limitation: coarse yes, fine no

Picking the right *kind* of motion works. Picking the right *amount* does not — and it did not
improve when the representation changed. Images could not separate fine magnitudes; Froude could
not either.

| question | result |
|---|---|
| which family? (walk / turn / strafe) | **works, crosses embodiments** |
| how much? rank 12 conditions | 28% exact, mean rank 2.33/12 |
| how much? 0.5-sd perturbations of one behaviour | **47% vs a 50% coin** |
| direction of correction | 0.867 — right way |
| extent of correction | 0.71 sd — wrong amount |

**The information is not missing.** Probed straight off the frozen embedding delta — no ITM, no FTM,
no trained head — the fine speed signal reads out clearly. The pipeline cannot use it; the encoder
did not discard it.

Six independent readout fixes, all null. **Not focused — the claim is the shared coordinate, which
is coarse and works.**

> **บทพูด (TH).** สไลด์นี้มีสองส่วน: **ส่วนที่แก้ได้** กับ **ข้อจำกัด**
> แก้ได้: เดิมให้คะแนนด้วยระยะห่างของภาพ ซึ่งไปอ่านเฟรมปัจจุบัน ไม่ได้อ่านเป้า พอเปลี่ยนเป็น Froude การทำตามเป้าโผล่มาทันที
> ขยายจาก 1 เป็น 3 ช่อง การเลือกข้ามหุ่นดีขึ้นเกือบเท่าตัว การไถลข้างจากมองไม่เห็นเลยเป็นเกือบสมบูรณ์
>
> ข้อจำกัด: เลือก**ชนิดท่า**ได้ เลือก**ความแรง**ไม่ได้ และ**ไม่ดีขึ้นเลยตอนเปลี่ยนวิธี** — ใช้ภาพก็ไม่ได้ ใช้ Froude ก็ไม่ได้ (47% เทียบเหรียญ 50%)
> **แต่ข้อมูลไม่ได้หาย** — probe ตรง ๆ บน embedding ดิบยังอ่านออกชัด แปลว่า pipeline ใช้มันไม่เป็น ไม่ใช่ encoder ทิ้ง
> ลองแก้มา 6 วิธี null หมด **เราไม่โฟกัสตรงนี้** เพราะ claim คือพิกัดกลางที่ข้ามหุ่นได้ ซึ่งเป็นงานหยาบ ๆ และอันนั้นทำได้แล้ว

---

## Slide 25 — Imagination-RL: the wall is the rollout

```
  looked like: the actor exploits the frozen FTM's blind spots
        │
        ▼  isolation test: 5,000 iterations, frozen FTM, no re-grounding at all
  actually was: the critic never converges, even against a STATIC target
        │
        ▼  four real stabilisation fixes, in order (table below)
  best Dreamer variant still fails the pre-registered bar
        │
        ▼  switch algorithm entirely (PPO -- no bootstrapped imagined-rollout value)
  lands on the SAME wall as the best Dreamer variant
        │
        ▼  shorten the horizon instead (GAMMA 0.99 → 0.95)
  gap does not shrink -- policy freezes into a static stance instead
        │
        ▼  every one of these iterates the SAME single-step predictor to build a return
  failure localises to the FTM's rollout, not the algorithm
```

| fix | MC-check relative difference (bar: < 0.25) |
|---|---|
| EMA target critic alone | fails outright (unbounded value drift) |
| + symlog critic, + return normalisation | 0.464 |
| hard/periodic target updates | 0.624 — worse |
| + two-hot distributional critic (DreamerV3's own fix) | **0.272** — best Dreamer variant, still fails |
| **PPO** (different algorithm, no imagined-rollout bootstrap) | **0.281** |
| PPO, GAMMA 0.99 → 0.95 (horizon ~100 → ~20 steps) | 0.306 — worse, policy went static (action variation −92%) |

**Two unrelated algorithms, same ~0.27-0.28 wall; shortening the horizon doesn't move it.** Not an
algorithm problem — every one of these returns iterates the same single-step predictor. The exact
failure mode Koopman Dreamer (2607.19719) names.

---

> **บทพูด (TH).** สไลด์นี้คือความพยายามใช้ RL บนโลกจำลอง (imagination) แล้วมันไม่ผ่าน
> ตอนแรกคิดว่า actor ไปหาช่องโหว่ของ forward model แต่จริง ๆ คือ **critic ไม่ลู่เข้าเลย** แม้เป้านิ่ง ๆ
> แก้ 4 อย่างตามตำรา Dreamer แล้วก็ยังไม่ผ่านเกณฑ์ เปลี่ยนไปใช้ PPO ซึ่งคนละอัลกอริทึมกันเลย ก็ไปชนกำแพงเดียวกัน
> แปลว่าปัญหาไม่ใช่อัลกอริทึม แต่เป็นการเอา forward model แบบ 1 สเต็ปมาม้วนต่อกันยาว ๆ

---

## Slide 26 — Sequence context does not fix it

```
  FTM(frame_t, action_t) → frame_t+1        one observation, one action
        │
        ▼  same action means different things at different points in a gait cycle
  one-action-many-outcomes, no phase context to disambiguate them
        │
        ▼  Yu / WMP: locomotion world models are recurrent-over-HISTORY
              frame-SEQUENCE + command-SEQUENCE → next state
        │
        ▼  premise checked on GROUND TRUTH first, no model in the loop
  signal exists in the data (table below) → not a task property, a model gap
        │
        ▼  every combination of {pooled, spatial-preserved} x {non-recurrent, recurrent}
             that is cheaply testable — command-sequence + delta-target held fixed throughout
  ALL FOUR fail the bar (table below), including the one that is genuinely
  spatial + recurrent at once — the widest miss of the four
```

Slides 24 and 25 both improved something real and both hit a wall, tracing to one place: the FTM is
**stateless, single-step** (confirmed — no hidden state between calls, not an RSSM). The standard
locomotion fix (Yu, WMP) conditions on frame-*history* and command-*sequence* instead.

| ground-truth check (Test 1, no model) | value |
|---|---|
| corr(\|Δaction\|, \|ΔFroude\|), within one fixed behaviour | +0.432 |
| permutation p-value (n = 20,000) | 0.002 |
| ridge, leave-one-out R² | 0.204 |

| kill-gate (command-sequence + delta-target throughout, bar: gap > 0.110) | spatial preserved? | recurrent? | gap |
|---|---|---|---|
| pooled mean-vector per frame → GRU | no | yes | +0.036 |
| learned attention-pool per frame → GRU | no — still pools every frame | yes | +0.058 |
| full token grid, self-attention across frames | yes | no | +0.048 |
| **ConvGRU: spatial hidden state, pooled only at final read-out** | **yes** | **yes** | **+0.069** |
| stateless single-step FTM (reference) | — | — | +0.042 |

**The genuine test — spatial detail flowing through a real recurrent state — fails by the widest
margin, not the narrowest.** So the sequence-context hypothesis is **ruled out, not unconfirmed**.
Test 1's signal still stands (the flatness is real, not a task property); no context-based fix
recovers it. Open next: capacity/optimisation of an end-to-end-trained model, not input
representation.

---

> **บทพูด (TH).** สไลด์นี้รวบว่าทำไมทั้งสองทางข้างบนถึงชนกำแพงที่เดียวกัน
> คำตอบคือ forward model ของเราเป็นแบบ **สเต็ปเดียว ไม่มีความจำ** ซึ่งงาน locomotion เขาไม่ทำกัน เขาใช้ประวัติ
> เราเลยเช็คก่อนว่าสัญญาณมีจริงไหมใน ground truth — **มีจริง** แล้วค่อยลองใส่ context 4 แบบ — **ตกหมดทั้ง 4 แบบ**
> อันที่ควรดีที่สุด (spatial + recurrent พร้อมกัน) กลับพลาดมากที่สุด สมมติฐานนี้เลยถูก **ตัดทิ้ง ไม่ใช่แค่ยังไม่ยืนยัน**

---

## Slide 27 — Stop-gradient clears the lever ~9×

```
  eliminated, each ruling out one cause (Slides 24-26 + offline probes):
    loss target · gradient share · sequence-context architecture (4 variants) · the encoder itself
        │
        ▼  what's left: the trained z-only head STILL fails the lever (+0.045 / +0.099)
             even though z demonstrably HAS the signal (offline rho 0.535, survives to proj(a))
        │
        ▼  isolate: freeze z, train a clean head on Froude ALONE (no competing L_recon/L_motion)
  PASSES -- matches/exceeds offline rho          → cause localised: joint-training competition for z
        │
        ▼  fix: z.detach() before L_body (L_state retired, redundant+worse)
             real 50-epoch retrain, BIAS-2, per-channel reading pre-registered before running
  PASS -- gap +1.008, ~9x the 0.110 bar
```

| action-lever, real retrain (`beh12_body_stopgrad`) | value |
|---|---|
| real z, median cos | 0.693 |
| mean z, median cos | -0.315 (anti-correlated, not just flat) |
| **gap (bar: 0.110)** | **+1.008** |

| channel | sign-agreement gap (real − mean) |
|---|---|
| forward | +0.238 |
| lateral | +0.031 (weakest, still positive) |
| yaw | +0.243 |

**Eight nulls were the diagnosis, not the failure** — each eliminated a cause, which is what
localised the real one to joint-training gradient competition rather than guessing it.

**Mechanism, precisely:** `L_body`'s gradient was not *pure* competition — it does real positive
work too (recon+motion alone develops only 32-76% of a jointly-shaped `z`'s signal). The fix trades
that contribution away to remove the competition, and still wins by more than the proxy predicted.

**Not yet shown: that it improves control.** The lever measures directional sensitivity, not
ranking or closed-loop behaviour — a gap this project has been burned by before (F116-F127).

---

> **บทพูด (TH).** สไลด์นี้คือผลที่คลี่คลายเรื่องทั้งหมด
> null ทั้ง 8 อันก่อนหน้าไม่ใช่ความล้มเหลว แต่คือการ **ตัดสาเหตุออกทีละอัน** จนเหลือสาเหตุจริง
> สาเหตุคือตอนเทรนรวมกัน loss หลายตัวแย่ง gradient ใน z กัน พอใส่ stop-gradient ให้ L_body ผลดีขึ้น ~9 เท่าของเกณฑ์
> **ข้อควรระวัง**: อันนี้วัดแค่ "ทิศทางไว" ยังไม่ได้แปลว่าคุมหุ่นได้จริง ซึ่งโปรเจกต์นี้เคยพลาดตรงนี้มาแล้ว

---

## Slide 28 — Motor babble, and what we added

```
  a robot nobody has a controller for
        │  flail semi-randomly, record (action, video, body motion)
        ▼
  clips that are NOT demonstrations — nobody is doing the task well
        │  fit: action ──▶ latent ──▶ physical body motion
        ▼
  the robot's own commands become readable in a coordinate shared with other robots
```

**What we contributed is calibration and convention, not a controller.** Babble gives a *map* from a
new body's commands to a shared coordinate. Turning that map into behaviour still needs our pipeline
to construct the motion — **that part is not working yet.** The method is widely used and has real
potential; this deck reports how far the calibration gets and where it stops.

### Zero-shot vs fine-tune (median ρ, `z = proj`, held-out)

| | clips needed | hexapod (in pretrain) | B1 (in pretrain) | gecko (never) |
|---|---|---|---|---|
| zero-shot, frozen head | **0** | 0.454 | 0.427 | 0.113 ⚠ |
| wrong "fine-tune" (`wm.train`) | 48 | — | 0.231 | not run |
| **correct staged adaptation** | **9** | — | **0.572** | 0.249 ⚠ |

**Nine clips.** Stage 1 adapts ITM+FTM on **9 babble clips, 1000 steps** — that is the whole cost of
a new body. Stages 2 and 4 refit two small heads (projector, and an 8.8k-parameter Froude head) on
the same babble set; no new recording. Zero-shot needs none of it and gets 0.427 on B1.

⚠ **The gecko column is not comparable and must not be read as one.** Its 0.113 was measured through
a body-frame bug found later and is withdrawn; its 0.249 comes from a later mechanism. Gecko is a
separate experiment — below.

**B1 is the defensible claim:** zero-shot to a body the world model saw in pretrain already
half-works (0.427); adapting on that body's own babble takes it to 0.572.

> **บทพูด (TH).** สไลด์นี้แนะนำวิธีที่งานเราอยู่ในนั้น: **motor babble**
> หุ่นที่ยังไม่มีใครทำ controller ให้ ก็ปล่อยให้มัน**ขยับมั่ว ๆ** แล้วอัดว่า สั่งอะไร → ภาพเป็นยังไง → ตัวเคลื่อนที่ยังไง
> คลิปพวกนี้**ไม่ใช่การสาธิตท่าที่ดี** มันแค่ทำให้เราแปลคำสั่งของหุ่นตัวใหม่เป็นพิกัดกลางได้
> **สิ่งที่เราทำคือ calibrate ไม่ใช่สร้าง controller** — สุดท้ายยังต้องให้ pipeline ไปประกอบท่าให้ **และส่วนนั้นยังไม่สำเร็จ**
> ตาราง: **B1 ดีขึ้นจริง 0.427 → 0.572** โดยใช้แค่ **9 คลิป / 1000 steps** — นี่คือต้นทุนทั้งหมดของหุ่นตัวใหม่
> ไม่ต้องอัดข้อมูลเพิ่ม stage อื่นแค่ fit หัวเล็ก ๆ สองอันบน babble ชุดเดิม ส่วนช่อง gecko **ห้ามอ่านรวมกัน** เดี๋ยวอธิบายแยก

## Slide 29 — The correct adaptation pipeline

```
  wrong mechanism, tried first: wm.train --init_ckpt
    jointly retrains ITM+FTM+decoder+body_head+probe under the FULL pretrain loss
    → B1 got WORSE than zero-shot (0.264 → 0.231), forward went negative
        │
        ▼  the actual question: "are we even running LAC-WM's adaptation?" — NO
  the real staged procedure, never assembled into one pipeline before:
    stage 1  wm.adapt          — fine-tune ONLY ITM+FTM on the new body's own clips
    stage 2  fit_projector     — refit a→z against the ADAPTED itm
    stage 3  wm.adapt3         — optional joint fine-tune (skipped)
    stage 4  fit_body_head     — refit the shared Froude head against the projector's own z
```

| B1, z=proj | forward ρ | lateral ρ | yaw ρ | median ρ |
|---|---|---|---|---|
| zero-shot (no adaptation) | 0.057 | 0.264 | 0.526 | 0.264 |
| `wm.train` joint retrain (wrong mechanism) | −0.068 | 0.231 | 0.326 | 0.231 |
| **correct staged adaptation** | **0.572** | **0.449** | **0.670** | **0.572** |

**Every channel more than doubles, forward included — never once positive under the wrong
mechanism.** The whole "B1 got worse" episode was a tooling error, not a result about the claim.

> **บทพูด (TH).** สไลด์นี้คือการแก้ที่ต้นเหตุ
> ตอนแรกเรา "fine-tune" ด้วยวิธีที่ผิดสนิท — มันไป retrain ทุกอย่างพร้อมกันด้วย loss ของการ pretrain
> ผลคือ B1 **แย่ลงกว่าไม่ทำอะไรเลย** (0.264 → 0.231) แล้วเราก็เสียเวลาไล่หาว่าทำไมมันพัง ทั้งที่คำถามที่ถูกคือ "เราใช้วิธี adaptation จริง ๆ หรือยัง" — **คำตอบคือยัง**
> พอใช้ขั้นตอน 4 stage ที่ถูกต้อง **ทุกช่องดีขึ้นเกินเท่าตัว** โดยเฉพาะช่อง forward ที่ไม่เคยเป็นบวกเลยในวิธีเดิม
> บทเรียน: **มันเป็นบั๊กเครื่องมือ ไม่ใช่ข้อสรุปว่าวิธีเราไม่เวิร์ก**

## Slide 30 — The 2×2: which half is broken

```
  goal source (physics / vision)  ×  candidate scoring (direct / rollout)
```

Same goal clip, same 12 candidates, scored per step (not dominant-pick):

| mode | candidates | goal | goal read error | top pick | % steps right family | |top pick − true goal| |
|---|---|---|---|---|---|---|
| **A** | direct | physics (privileged) | 0.0000 | turn_w0.008 | **78%** | **0.038** |
| **D** | direct | **vision only** | **0.0293** | turn_w0.008 | **80%** | **0.038** |
| B | rollout | physics (privileged) | 0.0000 | side_R_lvl1 | 47% | 0.290 |
| C | rollout | **vision only** | 0.0293 | side_R_lvl1 | 40% | 0.290 |

**Reading the goal from video costs nothing.** D is handed no recorded number and misreads the goal
by 0.029 in Froude — yet matches A exactly (same top pick, same distance, 80% vs 78%). The
misreading is smaller than the gap between the two closest candidates (0.033 vs 0.038), so it never
changes the ranking. **This is the claim: a vision goal performs like a privileged proprioceptive
measurement.**

**Rollout does not merely fail — it prefers the worst candidate.** `side_R_lvl1` is the *farthest*
of all 12 from the goal (0.290 against the best 0.033). Handing it a perfect goal (mode B, error
0.0000) does not help. The goal is not the problem; the world model in the scoring loop is.

**Reminder (23.5): selections from a library, not a controller.** Nothing here can fall.

**📹 VIDEO — C vs D.** Three panels, aligned by elapsed time (the two bodies record at 20 Hz and
50 Hz, so matching by frame index puts the goal 2.5× ahead). Goal panel shows both what the system
read and the true value; footer shows both errors.

| panel | shows |
|---|---|
| left | source body ego — the goal |
| middle | new body ego — what the loop sees |
| right | new body allocentric — **replayed ground truth, not control** |

> **บทพูด (TH).** สไลด์นี้แยกว่า**ครึ่งไหนของลูปพัง** โดยไขว้สองแกน: เป้ามาจากไหน × ให้คะแนนยังไง
> **A กับ D เท่ากัน** — D อ่านเป้าจากวิดีโอผิดไป 0.029 แต่**เลือกคลิปเดียวกัน ระยะห่างเท่ากัน** (80% เทียบ 78%)
> เพราะความผิดพลาดนั้น**เล็กกว่าช่องว่างระหว่างผู้สมัครสองตัวที่ใกล้ที่สุด** (0.033 กับ 0.038) มันเลยไม่เปลี่ยนอันดับ
> **นี่คือข้อเคลม: เป้าจากภาพทำงานได้เท่ากับการวัดด้วย proprioception ที่แอบดูตัวเลขจริง**
> **ส่วน rollout ไม่ใช่แค่พลาด แต่เลือกตัวที่แย่ที่สุด** — `side_R_lvl1` ห่างจากเป้าที่สุดใน 12 ตัว (0.290)
> และ**ต่อให้แจกเป้าที่ถูกต้อง 100% ให้ (mode B) ก็ยังพัง** แปลว่าปัญหาไม่ใช่เป้าหมาย แต่คือ world model ตอนให้คะแนน
> **ย้ำ:** นี่คือการเลือกคลิปจากคลัง ไม่ใช่ controller — มันล้มไม่ได้อยู่แล้ว

## Slide 31 — Gecko: the actual unseen body

**Separate from everything above.** B1 was in pretrain. Gecko was not — no URDF, no kinematics, no
expert clips, only babble. This is early work, kept apart on purpose.

### Two measurement bugs, found and fixed

| | |
|---|---|
| the stage-1 "gate" | sign inverted — it is anti-correlated with success, not a gate |
| gecko's body frame | built on an axis pointing straight up; all three channels scrambled |
| after both fixes | yaw went from **dead (−0.04) to strongest (+0.50)** |

### Where it stands

| stage-4 held-out (below 1.0 = the head learned something) | |
|---|---|
| broken frame, projector sees 1 action frame | 1.010 — dataset mean only |
| corrected frame, 1 action frame | 1.004 — still the mean |
| **corrected frame + 20 action frames** | **0.970** — first time below 1.0 |
| B1, same metric | **0.751** |

**Why 20 frames:** one action frame predicts gecko's motion at ρ 0.215; twenty predict it at
**0.736** ≈ B1's 0.770. Gecko's speed is set by gait *frequency*, invisible in one snapshot.

### The remaining gap is the camera

| what carries the signal | ρ | |
|---|---|---|
| actions (20-frame window) | **0.736** | ≈ B1's 0.770 — data is fine |
| **egocentric video** | **0.374** | B1's 0.747 — **the ceiling** |
| pipeline delivers | 0.249 | 67% of what the video allows |

Gecko's forward Froude is 0.038–0.048 vs B1's 0.126, flat across gait frequency. A robot that slow
barely moves between frames while its legs fill the view. **A limit of the sensor, not the
coordinate** — it bounds which bodies this method reaches.

**The 2×2 has not been run on gecko and should not be yet.**

> **บทพูด (TH).** หน้านี้แยกออกมาจากทุกอย่างข้างบน เพราะ **B1 เคยอยู่ใน pretrain แต่ gecko ไม่เคย**
> ไม่มี URDF ไม่มี kinematics ไม่มีคลิปครู มีแต่ babble — เป็นงานที่เพิ่งเริ่ม เลยจงใจแยกไว้
> **เจอบั๊กการวัดสองตัว**: อ่านเกตกลับด้าน กับ แกน forward ชี้ขึ้นฟ้า พอแก้แล้ว yaw ที่คิดว่าตายกลายเป็นดีที่สุด
> **สถานะตอนนี้**: ต้องแก้ทั้งสองอย่างถึงจะลงต่ำกว่า 1.0 ได้ (0.970) — B1 อยู่ที่ 0.751
> ที่ต้องให้ดู action 20 เฟรม เพราะ **ความเร็ว gecko มาจากความถี่การก้าว** ซึ่งดูเฟรมเดียวไม่มีทางรู้
> **ที่เหลือคือกล้อง**: action มีข้อมูลพอ (0.736) แต่วิดีโอ ego มีแค่ 0.374 — gecko เดินช้ามาก ระหว่างเฟรมแทบไม่ขยับ
> แต่ขาตัวเองบังเต็มจอ **เป็นข้อจำกัดของเซนเซอร์ ไม่ใช่ของพิกัดกลาง** และยัง**ไม่ควรรัน 2×2 กับมันตอนนี้**

---
