# Progress Update — Cross-Morphology and Cross-Embodiment Latent Action Models

Stick insect (*Medauroidea extradentata*) and Unitree B1, simulated in CoppeliaSim.

**Slides 1-12 are Stage 1**: one 18-DOF topology, several leg geometries, unchanged from the
previous update. **Slides 13-19 are Stage 2**: what crosses between two robots with no shared
action space, what the world model can and cannot do about it, the named failure that blocks the
rest, and what is being built next.

**Notation.** `e_t` is a V-JEPA2 observation embedding, `z` a latent action, `a` a joint command.
Where this deck describes a phenomenon ActSWM also reports — action-sensitivity, Context Collapse,
the real-against-null rollout contrast — it uses their terminology and **our** symbols; ActSWM's
`z` denotes an observation embedding and adopting it would collide with ours.

Slides 1 to 3 are background already covered previously. Stage 1's update starts at slide 4;
Stage 2 is new since the last deck and starts at slide 13.

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
| Slides 4-7 | the central result: the geometry is readable, the model ignores it, what fixed that, and what the fix did to the latent |
| Slides 8-10 | where it stops working, why, a test of that explanation, and a check that predicts it in advance |
| Slides 11-12 | two facts about the task itself that bound what the latent can be worth |
| Slides 13-15 | status, a first cross-embodiment run, and the B1: zero-shot fails, three clips buy one step |
| Slide 16 | we decode joint angles and LAC-WM does not — the divergence, and the term it forces |
| Slide 17 | what the shared axis did to the latent |
| Slide 18 | what it did not do: the forward model, and why |
| Slides 19-20 | what we contribute, and the pipeline from pretraining to a controller |
| Slide 21 | variety was necessary, supervision is what carries a channel |
| Slide 22 | the conclusion: what was asked, what the measurements say, and the scope |
| Appendix | last term's three questions, answered |

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
that does not walk (F42), so a training set of bodies that *do* walk is one where those two
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
structural fact about *that* task, not about the model, and **it does not extend to locomotion with
varying speed** — slide 15 shows the action mattering as soon as the magnitude varies. Two
measurements say it independently.

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
(`/mean-z` 0.485 against 0.951 within one speed, slide 15). **Read this slide as "at one speed",
never as "the action does not matter in locomotion".**

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

## Slide 13 — Stage 2: the thesis, the gap, and where we actually stand

**The contribution.** A world model that plans toward a goal defined in a coordinate **shared
across bodies whose action spaces have nothing in common** — an 18-DOF six-legged stick insect and
a 12-DOF Unitree B1 quadruped — with no kinematic model, no retargeting, and no controller already
running on the target robot. The only thing the two robots share is what a camera sees.

**The gap, stated against what exists.**

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
                              │        LAC-WM sits off this map:
                     needs no kinematics        manipulation, where end-effector
                                                pose is already shared, and it
                                                *selects* over a VLA's proposals

**Read the axes literally.** Everything that crosses leg count is handed a body model. Everything
that needs no body model stays inside one leg count. **The lower-right quadrant is empty**, and
locomotion has no end-effector pose to retreat to: 18 and 12 joint targets share no dimension.

**Where we stand, said plainly and separately from the goal.**

    ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
    │  GOAL REPRESENTATION │   │  WORLD MODEL         │   │  CONTROLLER          │
    │  shared coordinate   │──►│  dynamics            │──►│  a robot being driven│
    ├──────────────────────┤   ├──────────────────────┤   ├──────────────────────┤
    │  ✔ PROVEN            │   │  ⚠ DIAGNOSED         │   │  ✘ NOT DONE          │
    │  70% vs 28% chance   │   │  Context Collapse,   │   │  every number here is│
    │  all three families  │   │  named + cited,      │   │  offline selection   │
    │  survives the control│   │  fix identified      │   │  among recorded clips│
    └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
        slide 14                   slides 15-16                slide 17 = the plan

**Proven** — a quadruped, shown a stick insect's video and nothing else, picks which of its own
behaviours matches, and the choice survives the control that destroyed every earlier version of
this claim.

**Diagnosed** — the predictor extrapolates the next state from context while going insensitive to
the action it was given. We measured it before we knew it had a name.

**Not done** — the integration. The pieces are measured; the failure between them is now understood
rather than mysterious.

**What changed since the last update.** Three claims this deck previously made are withdrawn, each
by a control that had not been run: that the contrastive adaptation term is what crosses
embodiments, that the closed loop selects behaviours on the quadruped, and that sideways motion
fails everywhere. The measurements were real; the conclusions drawn from them were not. What
replaced them is stronger and is on the next slide.

Speaker note: the honest one-sentence position is *the goal representation crosses bodies and is
measured; the dynamics model that would act on it does not yet, and we know why.*

---

## Slide 14 — What crosses: the shared coordinate, and it is not the latent

    INSECT video ──► encoder ──► ITM ──► shared head ──►  goal:  (fwd, lat, yaw)
                                                                      dimensionless
                                                                          │
                                                                       compare
                                                                          │
    B1 candidate action ──► projector ──► shared head ──►  predicted: (fwd, lat, yaw)

**Nothing but pixels enters on either side** — no trajectory value, no joint correspondence, no
kinematic model. The two robots meet in a quantity measurable from outside that means the same thing
on both.

**The control that decides whether any of this is real:**

    matched goal    goal = "turn"   ──►  picks turn   ──► looks like success
                                                          BUT so does a rule that
                                                          just names what the robot
                                                          is already doing

    swapped goal    goal = "strafe" ──►  picks ???    ──► follows the goal  → real
                     (robot still walking)                follows the robot → fake

**Three earlier versions of this result died on exactly that swap**, each after it had already
shaped a plan.

**One channel against three, cross-embodiment, chance 28% pooled:**

| | pooled | sideways L | sideways R | forward | turning |
|---|---|---|---|---|---|
| forward speed only | 33-41% | **13-25%** | 31-39% | 18-52% | 54-62% |
| **all three channels** | **70%** | **86-100%** | 65-84% | 41-62% | 39-64% |
| chance | 28% | 17% | 17% | 33% | 33% |

Tracking the demonstration instead falls to **18-21%, below chance** — the picks follow the request
and stop following the robot's current state.

### ▶ `f136_strafe_reframe.mp4` — the clearest single result

    GOAL: insect strafing LEFT  │  one-channel score picks  │  three-channel score picks
                                │  side_R_lvl1              │  side_L_lvl1
                                │  strafes the OPPOSITE way │  correct side

**A forward-speed-only score cannot tell left from right, because both have the same forward
speed.** Add the lateral channel and the sign separates them immediately. **Invisible, not hard**,
in one picture.

### ▶ `f136_selection.mp4` — the same rule across all three families

Three rows, each *insect goal* beside *the quadruped behaviour the score picked*. **These are two
recorded clips: the behaviour it selected, not the robot it drove.** F136 measures selection; the
cross-embodiment physics loops this project has run are the chance-level ones, and using their
frames here would be the overclaim the mismatch control exists to prevent.

| goal | picked | reading |
|---|---|---|
| `speed_c7.1` | `speed_vx0.38` | exact match, but on only **3 of 10** steps — forward is a weak family |
| `turn_s0.29` | `turn_w0.075` | **responds but ranks poorly: right family (10/10), wrong level** |
| `side_L_lvl1` | `side_L_lvl1` | exact match, 5 of 10 |

**The middle row is the open problem made visible** — unanimous about *what* the insect is doing and
wrong about *how much*. That is slide 15's third level, in a picture.

**Two claims this deck made that the three-channel run overturns.**

**Sideways was never a hard behaviour. It was an invisible one.** 13-25% on a forward-only score,
at or below its 17% chance rate; **86-100%** once lateral speed enters it. Every report since the
behaviour-family work said strafing fails everywhere — we were measuring a channel the metric could
not see.

**There is no channel competition.** The earlier result — adding yaw costing forward 68% — was
measured on forward-walking-only data carrying a frame-rate defect. On corrected data with both
robots in the pretraining, all three channels calibrate at once: correlation +0.97 to +0.99 and
range compression 1.0-1.2x, on **both** bodies.

**What is shared is not the latent** — that is fitted per robot, and an 18-D and a 12-D command
reach it by different routes. What is shared is the **target quantity**, and that is the sentence
the control leaves standing.

---

## Slide 15 — What the world model can do, and the wall it hits

**Where the model does and does not separate actions.** `/mean-z` is our **action-sensitivity
ratio**: the rolled prediction's error on the real `z` over its error on a substitute `z`. Lower
means the action matters more. At one step:

| the action is compared against | B1 | insect | reading |
|---|---|---|---|
| the average of **this clip** — one magnitude | 0.951 | 0.697 | **ambiguous**: the gait is periodic at one speed, so the action may be genuinely redundant |
| the average of **this behaviour, other magnitudes** | **0.485** | 0.597 | the model **does** separate speeds |
| the average of **all behaviours** | 0.476 | 0.534 | it separates behaviours |

**The middle row settles what the first row could not** — hold the magnitude fixed and the action
changes the B1's prediction by 5%; let it vary and it changes it by more than half. **The model is
not blind inside a behaviour; it is blind where there is nothing to see.**

**But reacting is not ranking, and the distinction has three levels:**

    within one magnitude    does the action matter?     no -- and correctly so.
                                                        THIS IS ALL SLIDE 11 CLAIMS,
                                                        and it is a task property no fix targets
    across magnitudes       does it REACT?              yes, 0.485 -- not collapsed
    across magnitudes       does it RANK correctly?     not well, and OPEN

Turning and forward are the **weakest** families in cross-embodiment selection — 39-64% and 41-62%
against a 33% chance rate, where sideways reaches 65-100%. **The prediction moves when the magnitude
changes and still orders magnitudes poorly.** Across embodiments the speed a loop achieves and the
speed it was asked for correlate at **+0.074**.

**So the wall is narrower than it looked, and it is not one wall.** Separating behaviours works.
Reacting to magnitude works. *Ordering* magnitudes is weak and open. Ordering perturbations of one
action at one magnitude is not a wall at all — the physics barely separates them either. Three
attempts to build past this failed, and the failures were informative:

**The planner was never conditioned on its goal.** Swap the goal and its picks do not move. **The
fix was the coordinate, and the coordinate needs no rollout.**

**Run-time action search cannot find a gait** — with no recorded library, sampled joint commands
never produce locomotion at any noise scale.

**Teacher-student failed a bar fixed before it ran**, on the easiest possible case — same robot on
both sides, a forward model good to three steps, real clips to bootstrap from:

    real walk        ████████████████████████████████████████  0.657 m   100%
    the bar          ████████████████████                      0.328 m    50%
    cloning only     ██████████████                            0.235 m    36%   FAIL
    + the teacher    ████████████                              0.204 m    31%   FAIL

Both stayed upright for the whole three seconds. Neither travelled. **The teacher made it worse.**

### ▶ `f144_labelled.mp4` — recorded walk, cloning, and cloning plus the teacher

**The distances are burned into the frames**, because all three look like walking and only the
ground covered separates them — motion without travel. A clip of the taught policy alone would read
as success and contradict its own number, so it is not kept.

**The cause was measured, and it is not only the teacher.** It ranks behaviour families at **55%**
against 33% chance and perturbations of one action at **33%** against a coin — but **the physics
barely separated those candidates either**, 0.1304 against 0.1299 mean distance to the goal.

**So it is not a tuning failure and not only a model failure.** More rounds cannot sharpen an
ordering the world does not provide. **A teacher can be built on the choice between behaviours,
never on the refinement of one gait.**

---

## Slide 16 — The root cause has a name in the literature

**Two earlier slides narrowed what the latent's job could be. This one names what breaks it.**

    Slide 11   NOT decoding the joint command -- at ONE SPEED one frame supplies two thirds
               of it. A property of the TASK, not a failure, and true only at fixed magnitude
    Slide 12   NOT raw forward prediction -- the forward model was scored on the wrong thing
    ────────────────────────────────────────────────────────────────────────────────────
    so         action-CONDITIONED prediction is what is left --
               reacted to (0.485 across magnitudes) but not yet well ranked
               (39-64% turning, 41-62% forward), and Context Collapse is
               what breaks it over the horizon

**Context Collapse** — the predictor extrapolates the next state from the context it already has and
stops responding to the action it is conditioned on. Named in **ActSWM (arXiv 2607.26712, 2026)**.

    what it should do                    what it does
    ─────────────────                    ────────────
      e_t ──┐                              e_t ──────────────► ê_t+1
            ├──► FDM ──► ê_t+1                    ▲
      z   ──┘                              z ─ ─ ─┘  (1% of the answer)
      ▲
      the action decides where it goes     the state decides; the action is decoration

**The diagnostic is the one we were already running.** ActSWM contrasts a real-action rollout with
a **null-action** one; our `/mean-z` contrasts the real `z` against a substitute `z`. Same question,
reached independently.

**We arrived at the same diagnostic independently.** Our `/mean-z` compares a rollout on the real
action against one on the average action. On a body the model never trained on:

| | |
|---|---|
| **rollout prediction error** — the rolled state's error over a frozen frame's, one step | **0.732** |
| the same at ten steps | **0.978** — indistinguishable from predicting no motion |
| effect of perturbing the action by a full standard deviation | **1%** |
| effect of substituting a real action from another state | 6% |

**The forward model predicts where the robot goes next mostly from where it already is.**

**It comes from pretraining; fine-tuning inherits it rather than creating it.**

    PRETRAIN            reconstruction + readouts          never asks the prediction
                                                           to depend on the action
        │
        ▼
    ADAPT on robot A    + contrastive term                 0.955 ──► 0.583  repaired
        │
        ▼
    same model, robot B                                    0.757 ──► 1.052  worse than
                                                                            a frozen frame

**The repair is local to the body it was made on** — the trap slide 17's second term closes.

**Two more results name the two halves of our wall.** **CD-LAM (2607.09185)** reports latent-action
predictors *fragile under small perturbations* — slide 15's fine-ranking failure elsewhere.
**Dueling World Models (2608.06706)** reports the loss improving while actions become
indistinguishable — the shape of our stage-3 arm with the lowest loss and the worst sensitivity.

**So the failure is neither ours alone nor mysterious**, and it has a published fix.

**Why there is no side-by-side video of this**, where ActSWM has one: their world model predicts
pixels, so a real-action rollout and a zero-action rollout can be shown as two near-identical
videos. **Ours predicts V-JEPA2 embeddings and has no decoder back to images**, so the collapse is
visible only as numbers. That is a property of the architecture, not a missing figure.

*References supplied by the advisor; arXiv identifiers were not verified from this machine.*

---

## Slide 17 — The plan, and the part of it that is ours

**Adopt the fix where the defect is made: pretraining.**

                    ┌─────────────────────────────────────────────┐
    e_t, z real ───►│                                             │──► rollout A
                    │              forward model                  │
    e_t, z null ───►│                                             │──► rollout B
                    └─────────────────────────────────────────────┘
                          │                        │
                    prediction loss          ROLLOUT SEPARATION:
                    (keep accuracy)          push A and B apart over K steps,
                                             not at one
                                                    │
                                             scored through a fixed,
                                             randomly-initialised action-readout
                                             that is never trained

**1. The separation term is rollout-level.** Our sensitivity dies past three steps; a one-step
penalty cannot see that, and the horizon is what a distilled policy consumes.

**2. The readout is frozen — randomly initialised and never trained.** That is what stops it
cheating: a reader that learns cannot relocate the boundary to wherever the loss is looking. It
closes a trap we measured rather than a hypothetical one — the contrastive repair produced
sensitivity that lived only in the projector's region and only on the body it was adapted to.

*Our null is not ActSWM's zero action.* Their contrast is a zero-action rollout; ours is a **null
action**, the robot's standing stance, because these action spaces are joint targets rather than
torques and the literal zero vector collapses both robots (F148).

**3. The prediction loss stays.** The trade is **tunable, not structural** — measured: 0.710 fidelity
with 0.476 sensitivity together, against MSE's 0.585 and 0.969. A fifth of the accuracy buys all of
the sensitivity.

**The extension that is ours rather than ActSWM's.** ActSWM tests within a single body. **We ask
whether action-sensitivity survives across disjoint embodiments**, and we already know the naive
version does not: the contrastive repair holds on the body it was adapted to and collapses on
another. **That cross-embodiment robustness is the contribution** — the fix is theirs, making it
hold across bodies that share no action space is the open problem, and we have the measurement that
shows it is open.

**Pre-registered before the run, so the result cannot be reinterpreted after it.**

| | expectation |
|---|---|
| usable imagination horizon | **the primary target** — past three steps, where the rollout prediction error crosses 0.8. This is the clean, confirmed Context Collapse |
| magnitude **ranking** | **open, declared neither way.** The prediction already reacts to magnitude; ordering it is weak. A null here refutes nothing |
| action-sensitivity across embodiments | **the open question**; the naive fix fails it today |

**Deferred and named, not quietly dropped**: where a target robot's first motion comes from, if not
from a recorded library and not from reinforcement learning. Distillation needs a policy that
already moves; the library is what we are trying to remove. That is a separate question and it is
not answered here.

---

## Slide 18 — How everything above is measured

    frames ──► V-JEPA2 (frozen) ──► e_t
                                       │
     e_t, e_t+1 ──► inverse model ──► z │        training only: needs the future frame
     joint command ──► projector ──► z │        control time: no future frame available
                                       ▼
                          e_t, z ──► forward model ──► ê_t+1
                                       │
                                       └──► body head ──► (fwd, lat, yaw)  ← the shared coordinate

**One frozen encoder, three trained modules, and one head that both robots share.** The projector
exists because the inverse model needs the next frame, which at control time is the thing being
decided.

**The data.** Twelve matched behaviours per robot — four speeds, four turn rates, two sideways
levels each side — 48 clips each, at a common 20 Hz and 66 frames. Forward speed matched to 4% and
turn rate to 2% between the two robots, with Froude held at 0.12-0.13 on both sides.

**Three measurement rules this update enforces, each written after a result had to be withdrawn.**

**Every selection number needs the mismatched-goal control.** Report picks scored against the
demonstration *and* against the goal actually shown. With a matched goal the two are the same number
by construction, and three separate versions of the cross-embodiment claim survived on that
ambiguity until the control was run.

**Report per behaviour family, never pooled alone.** A pooled figure held steady at 36% across an
intervention while turning rose eighteen points and forward fell twenty. The aggregate was
actively misleading.

**Feed a model the latent it is actually shown.** The rollout prediction error measured on the
inverse model's latents read 1.370 for a checkpoint that reads 0.710 on the projector's — the path a policy would
drive. One number said the objective was structurally broken; the other said the trade is a knob.

**What each of the three numbers on slide 14 rests on.** 70% is one checkpoint and one seed,
offline, twelve recorded candidate behaviours, held-out clips. The three-channel calibration is one
pretraining run measured on both robots. The teacher-student failure is one seed against a bar
fixed in advance, with its cause measured in physics rather than inferred.

Speaker note: the discipline is the result as much as the numbers are — five conclusions were
withdrawn during this work, every one of them because a control had not been run, and every one of
them had already shaped a plan before it was checked.

---

## Slide 19 — What we do next

**The diagnosed failure.** Our forward model predicts the next embedding and stops responding to
the action that produced it. From `e_t`, a rollout on the real `z` and a rollout on a null `z`
converge: by ten steps the prediction is no better than a frozen frame, and a full-standard-
deviation change to `z` moves the answer by **1%**. This is **Context Collapse** (ActSWM, arXiv
2607.26712), and it comes from a pretraining objective that asks for reconstruction and readouts
and never asks the prediction to depend on `z`.

**The fix: three terms in pretraining.**

    (a) rollout separation      roll twice from the same e_t -- real z and null z -- and force the
                                two futures apart across the horizon, not at one step

    (b) frozen action-readout   a NEW module -- randomly initialised, never trained -- that scores
                                the separation. The model must separate the transitions instead of
                                relocating the signal to wherever it is measured. Closes F139/F143's
                                trap: our contrastive repair lived only in the projector's region,
                                on one body.
                                **Not the ITM.** The ITM makes the z the projector imitates;
                                freezing it at random weights would make z arbitrary and break
                                every control-time path. The new readout feeds nothing downstream

    (c) cross-augmentation      kept, unchanged. It fixes a different leak -- `z` copying the
                                future frame -- and nothing here replaces it

**The null action** (F148) is each body's **standing stance**: 18-D and 12-D, different vectors,
identically defined. Not the zero command and not the dataset-mean pose — both make a robot fall,
and a null meaning *still* on one body and *falling* on the other would corrupt the
cross-embodiment test before it starts.

**Pre-registered** (F146). Primary: does action-sensitivity survive past `h = 3`. **The
contribution**: does it survive hexapod ↔ quadruped, where our naive fix did not (F143). Magnitude
ranking is open — a null result there refutes nothing.
