# Weekly Update: Adopting the Reference Papers' Methods, and Testing Rollout

Period: 2026-09-20 to 2026-09-27 (findings F247–F274, plus one experiment in progress)

**Setup**
- **Pretraining body:** hexapod c10f10t10.
- **Test bodies:**
  - Unitree B1 (quadruped), adapted.
  - Hexapod c08f09t09 (held-out legs), zero-shot: no c08 data is used anywhere.
- **Model:** frozen V-JEPA2 encoder.
  - ITM: z = ITM(e_t, e_t+k).
  - FTM predicts e_t+k from (e_t, z).
  - Projector maps an action chunk to z.
  - Body head maps z to Froude (fwd / lat / yaw).
- **Froude:** dimensionless body velocity, 3 channels: fwd / lat / yaw (definition on the Metrics page).
- **Task:** given a goal Froude sequence from a held-out hexapod clip, select one candidate per step from the test body's clip library.

**Two selection mechanisms compared throughout**

| mechanism | score of candidate a (compared with the goal by L2) | uses the current state |
|---|---|---|
| direct | body(proj(a)) | no |
| rollout | body(ITM(e_t, FTM(e_t, proj(a)))) | yes (imagines the future from the current frame) |

Variants:
- **Rollout (library start):** the imagined future starts from the candidate's own recorded frame.
- **Rollout (current start):** it starts from the frame the controlled body is currently at.
- **Read-out window w:** the number of steps averaged before comparing to the goal.

---

## Metrics

Every table in this update uses one of these metrics. Each table names its metric by ID.

| ID | metric | definition | range / better |
|---|---|---|---|
| M1 | Error E | Mean Euclidean (L2) distance between achieved and goal Froude, over decision steps | ≥ 0 / lower |
| M2 | Normalised score (NS) | NS = (E_random − E) / (E_random − E_oracle), the same normalisation as the RL "human-normalised score". E_random: expected error of a uniformly random candidate. E_oracle: hindsight oracle, the best candidate at each step chosen knowing the outcome. Bounds: B1 0.031 / 0.126, c08 0.020 / 0.121 (oracle / random) | 0 = random, 1 = oracle / higher |
| M3 | Pearson r | Pearson correlation coefficient between predicted and true Froude, across the 24 candidate actions from the same start state; per channel fwd / lat / yaw | −1 to 1 / higher |
| M4 | Within-state Pearson r | M3 computed from one fixed start frame, then averaged over start frames | −1 to 1 / higher |
| M5 | η² state, η² action | Two-way variance decomposition (main effects) of the 24 × 24 start-frame × action grid P[s, a]: η² state = Var_s(mean_a P) / Var(P), η² action = Var_a(mean_s P) / Var(P) | 0 to 1 / for the world model: η² action higher, η² state lower |
| M6 | Top-1 retrieval accuracy | Per start state, subtract the mean over the 24 actions from both the predicted and the real 5-step embedding change; a prediction is correct if its **cosine similarity** is highest with the real change of the same action | chance 1/24 = 0.042 / higher |
| M7 | Next-frame MSE | Mean squared error of the predicted embedding vs the real one, held-out clips | ≥ 0 / lower |
| M8 | Relative MSE vs mean-latent baseline | MSE(FTM(e, proj(a)), FTM(e, z_true)) / MSE(FTM(e, z_mean), FTM(e, z_true)), held-out clips | 1.0 = no better than the dataset-mean latent / lower |
| M9 | Linear-probe accuracy | Logistic regression z → body identity, 5-fold cross-validation grouped by clip, classes balanced | chance 0.33 / lower = more shared |
| M10 | Cross-body R² | Coefficient of determination of a ridge regression z → Froude fitted on c10 only, evaluated on another body; < 0 = worse than predicting that body's mean | ≤ 1 / higher = more shared |
| M11 | k-NN mixing ratio (k = 10) | Fraction of each point's 10 nearest neighbours (standardised z) that belong to another body, divided by the fraction expected under random mixing | 0 = separated, 1 = fully mixed / higher = more shared |
| M12 | Ego-view check | Pearson r between the row-mean luminance profile of a rendered frame and that of correct training clips | 1 = identical profile; runs require ≥ 0.97 |

Froude (all metrics): dimensionless velocity with hip height h and gravity g. Forward and lateral speed in the body frame are divided by √(g·h), and yaw rate is multiplied by √(h/g).

---

## 1. Methods adopted from LAC-WM

**LoRA rank 2 in Stage 1 (F248)**

Metric: M8 relative MSE vs mean-latent baseline (lower is better), measured on both bodies after adapting to the B1.

| body | full fine-tune | LoRA rank 2 |
|---|---|---|
| hexapod (pretrained on, forgetting check) | 0.918 | 0.786 |
| B1 (adaptation target) | 0.307 | 0.319 |

**Stage 3: joint projector + FTM fine-tune, LoRA on the FTM (F265)**

Metrics: M7 next-frame MSE, M6 top-1 retrieval accuracy (chance 0.042), M2 NS on the B1 at w=11.

| | next-frame MSE | top-1 retrieval accuracy | NS direct | NS rollout (current start) |
|---|---|---|---|---|
| Stages 1, 2, 4 | 3.931 | 0.233 | +0.82 | +0.40 |
| + Stage 3 | 3.742 | 0.276 | +0.63 | +0.44 |
| + Stage 3, projector anchored to ITM z | 3.756 | 0.267 | +0.81 | +0.36 |

---

## 2. Why rollout lags direct (1): the FTM follows the start frame

**Start-frame × action grid (F252)**

Every start frame is paired with every action (24 × 24) and read as P[s, a] = body(ITM(e_s, FTM(e_s, z_a))). Metrics: M5 η² state / η² action (ranges over channels and steps), M4 within-state Pearson r.

| checkpoint, z source | η² state | η² action | within-state r fwd / lat / yaw |
|---|---|---|---|
| B1-adapted, proj(a) | 0.49–0.83 | 0.02–0.16 | 0.04 / 0.14 / 0.02 |
| B1-adapted, true z | 0.40–0.79 | 0.03–0.17 | 0.04 / 0.18 / 0.05 |
| pretrained, hexapod val, true z | 0.26–0.66 | 0.15–0.23 | 0.23 / 0.20 / 0.00 |

**Cycle loss (F253–F258)**

The cycle-consistency loss is an added training loss: ITM(e, FTM(e, z)) must return z. It is evaluated on rendered counterfactuals (the same start state rendered under each of the 24 actions). Metrics: M3 Pearson r, M6 top-1 retrieval accuracy.

| | control | cycle-consistency loss |
|---|---|---|
| read of the real outcome | 0.19 / 0.17 / 0.14 | 0.14 / 0.22 / 0.19 |
| read of the FTM prediction | 0.11 / 0.17 / −0.04 | **0.63 / 0.42 / 0.10** |
| top-1 retrieval accuracy (chance 0.042) | 0.137 | 0.146 |

Closed: the cycle-consistency loss's prediction is more readable than the real future itself, so its gain is z passing through the FTM, not better prediction.

---

## 3. Why rollout lags direct (2): action visibility and the read-out

**How far ahead the action becomes visible in the real future (F259)**

These are rendered counterfactual trajectories, with no model prediction involved. Metric: M3 Pearson r of body(ITM(e_t, e_t+k)) read on real frames.

| k (steps) | r fwd / lat / yaw |
|---|---|
| 1 | 0.20 / 0.15 / 0.15 |
| 2 | 0.28 / 0.26 / 0.19 |
| 5 | 0.53 / 0.62 / 0.44 |
| 11 | 0.60 / 0.77 / 0.77 |

**Read-out bottleneck, B1 stride 5 (F266–F268)**

Metric: M3 Pearson r.

| reading | r fwd / lat / yaw |
|---|---|
| direct: body(proj(a)) | 0.83 / 0.91 / 0.84 |
| ITM on the candidate's own recorded frames | 0.77 / 0.85 / 0.81 |
| ITM on the **real** outcome from a shared start state (= a perfect FTM) | 0.24 / 0.52 / 0.24 |
| rollout: ITM on the FTM prediction | 0.10 / 0.23 / 0.21 |
| rollout with a read-out refit on counterfactual pairs (held-out start states) | 0.48 / 0.63 / 0.72 |

Plugging that refit read-out into selection. Metric: M2 NS, B1, w=11.

| read-out | rollout, library start | rollout, current start |
|---|---|---|
| body head | +0.45 | +0.40 |
| counterfactual refit | +0.37 | +0.33 |

---

## 4. World-model stride: predict k frames per step with a k-command chunk (F260–F264)

Metric: M2 NS, read-out window w=11. Stages 2 and 4 of the B1 pipeline are fitted out-of-sample, on clips that are not in the candidate library. One seed per stride.

| stride | B1 direct | B1 rollout, library start | B1 rollout, current start | c08 direct | c08 rollout, library start | c08 rollout, current start |
|---|---|---|---|---|---|---|
| 1 | +0.66 | −0.04 | −0.25 | +0.39 | +0.35 | +0.26 |
| **5** | **+0.82** | **+0.45** | **+0.40** | +0.61 | **+0.38** | **+0.39** |
| 10 | +0.67 | +0.25 | +0.20 | **+0.64** | +0.24 | +0.27 |

Training pairs: 3,057 (stride 1) / 2,673 (stride 5) / 2,193 (stride 10).

Prediction of rendered 5-step counterfactual outcomes, M6 top-1 retrieval accuracy (chance 0.042): stride 1 0.161, stride 5 0.233.

---

## 5. Seed variance, and babble mixed into pretraining (F273)

These four stride-5 pretrains differ only in data or seed:
- **A:** beh24, 48 clips, one behaviour per clip (the stride-5 row on page 4).
- **A2:** A with a different seed.
- **B:** beh24 + 48 babble clips whose behaviour switches every 15–25 frames.
- **C:** beh24 + 48 steady babble clips, with the same drive distribution but no switch.

Metric: M2 NS, read-out window w=11.

| model | B1 direct | B1 rollout, library start | B1 rollout, current start | c08 direct | c08 rollout, library start | c08 rollout, current start |
|---|---|---|---|---|---|---|
| A | +0.82 | +0.45 | +0.40 | +0.61 | +0.38 | +0.39 |
| A2 (seed only) | +0.59 | +0.00 | +0.21 | +0.55 | +0.26 | +0.30 |
| B (switching) | +0.78 | +0.56 | +0.30 | +0.67 | +0.43 | +0.35 |
| C (steady) | +0.71 | +0.53 | +0.32 | +0.67 | +0.39 | +0.43 |
| stride 1 (1 seed), for reference | +0.66 | −0.04 | −0.25 | +0.39 | +0.35 | +0.26 |

Read-out of the B1's rendered counterfactual outcomes. Metric: M3 Pearson r, fwd / lat / yaw.

| A | B | C |
|---|---|---|
| 0.24 / 0.52 / 0.24 | 0.23 / 0.49 / 0.54 | 0.36 / 0.58 / 0.64 |

---

## 6. Is z shared across bodies? (F272)

z = ITM(e_t, e_t+k) is computed on recorded clips of three bodies. Metrics: M9 linear-probe accuracy (chance 0.33), M10 cross-body R², M11 k-NN mixing ratio.

| model | linear-probe acc | c10 → c08 R² fwd / lat / yaw | c10 → B1 R² fwd / lat / yaw | k-NN mixing ratio |
|---|---|---|---|---|
| stride 1 | 0.67 | +0.64 / −0.03 / −0.41 | −1.07 / −0.05 / −0.17 | 0.56 |
| stride 5 | 0.67 | +0.62 / +0.10 / −0.36 | −0.91 / −0.29 / −0.18 | 0.48 |

Figures: `results/deck/shared_latent/latent_stride{1,5}.png` (PCA, and UMAP with 3 seeds all shown).

Closed: z is shared between the two hexapods, but not with the B1, at either stride. The B1 is selected through its own fitted projector.

---

## 7. Physics closed loop (F274)

In this test the outcome depends on the body's state. The planner decides every 2 steps (w=11), and the score comes from how the simulated body actually moved over six goals. No run fell.
- **B1:** the chosen candidate's velocity command is executed by the B1's own walking policy in MuJoCo, and the egocentric view is rendered in CoppeliaSim.
- **c08:** the candidate's joint commands drive the physics-simulated hexapod in CoppeliaSim, with plain switching or phase-matched switching.

Metric: M1 error E of the simulated body's achieved Froude (lower is better).

| body | model | direct | rollout | random |
|---|---|---|---|---|
| B1 | A / A2 / B / C | 0.070 / 0.077 / 0.075 / 0.078 | 0.083 / 0.103 / 0.090 / 0.086 | 0.089 |
| c08 plain | A / A2 / B / C | 0.072 / 0.078 / 0.059 / 0.071 | 0.077 / 0.104 / 0.069 / 0.080 | 0.126 |
| c08 phase-matched | A / A2 / B / C | 0.065 / 0.073 / 0.060 / 0.065 | 0.100 / 0.104 / 0.093 / 0.087 | 0.111 |

Direct < rollout in 12 / 12 cells. Direct < random in 12 / 12 cells.

Figures and per-goal values: `results/deck/physics_closed_loop/`, `results/wm/closed_loop/physics/summary.txt`.

**Invalidated results and fixes (F256).** Metric: M12 ego-view check.

| issue | measurement | action |
|---|---|---|
| The closed-loop ego camera was created at 24° FOV; training clips use 90° | correct renders 0.980–0.995, that closed loop 0.67–0.70 | rollout closed-loop results before 2026-09-25 quarantined (`results/_invalid_F256/`) |
| Four B1 ego babble datasets were rendered with the wrong lens | 0.43–0.94 | excluded from use |
| The closed loop now refuses a first frame below 0.97 | physics runs above: 0.995 | guard in code |

Direct selection never reads the closed-loop camera, so direct results are unaffected.

---

## 8. Objective of the world model, and the experiment now running

**Objective**
- **Direct** matches an action to the motion it produced in recorded clips (recognition).
- **Rollout** must predict the consequence of an action from the current state, which is what the world model is for. The target is for rollout to select as well as or better than direct.

**Data property relevant to this objective (earlier measurement, `FINDINGS_old`, note before F154)**

Pose-matched pairs are transitions whose pose is as close to a pose from a *different* behaviour family as to a typical pose of its *own* family.

| | hexapod c10 | B1 |
|---|---|---|
| transitions whose nearest different-behaviour pose is as close as a same-behaviour pose | 4 / 765 (1%) | 166 / 768 (22%) |
| action difference at those pairs (in per-joint standard deviations; two unrelated commands differ by 1.41) | 0.16 | 0.25 |

In the pretraining data, the same state is almost never followed by different actions.

**Running now: pretraining on different action-variation schemes**

All arms use stride 5, 48 c10 clips each, and 2 seeds per arm. Everything else is identical (validation set, B1 pipeline, c08 zero-shot).

| arm | pretraining data | runs |
|---|---|---|
| 1 defined | beh24: one behaviour per clip | A, A2 (done) |
| 2 switch | babble only: new drive every 15–25 frames | 2 (training) |
| 3 rapid | babble only: new drive every 5 frames (= one model step) | 2 (queued) |

Pilot of arm 3 (4 clips): the achieved motion lags the command by about one segment (5–10 frames). The 2nd–98th percentile range of forward Froude is −0.14 to +0.13, against −0.21 to +0.22 for beh24.

**Measured on each arm**

| stage | metric |
|---|---|
| 1. Does the FTM use the action? | M5 η² state / η² action, M4 within-state r |
| 2. Can the read-out read counterfactuals? | M3 Pearson r on rendered counterfactual outcomes, M6 top-1 retrieval accuracy |
| 3. Does selection improve? | M2 NS (kinematic replay), M1 error E in the physics closed loop; direct vs rollout |

Expected completion: 2026-09-28.
