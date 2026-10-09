# Weekly Update: A Froude-Grounded Latent Action World Model Across Bodies

Period: 2026-09-20 to 2026-09-30

## The problem this period set out to fix

To choose an action, the world model imagines the future for each candidate action and reads off the resulting motion. That only works if **different actions give different predicted futures**.

![Same start state, different actions: real futures vs what each model's prediction looks like](../results/deck/weekly/action_future_strip.png)

- **Row 1:** one start state.
- **Row 2:** the real future 5 steps later under 6 different actions.
- **Rows 3–5:** each model's prediction, shown as the real future it is closest to. Green = it matches the right action; red = it matches another action's future.

Metric: top-1 retrieval accuracy of the predicted future among the 24 actions' real futures (chance 0.042), averaged over all start states.

| model | top-1 retrieval accuracy |
|---|---|
| start of the period (one frame per step) | 0.161 |
| five frames per step | 0.233 |
| current pipeline (below) | 0.266 / 0.271 (seeds S0 / S1) |

**Bodies.**
- **Pretraining body:** hexapod c10f10t10.
- **Test bodies:**
  - c10 itself (same body);
  - hexapod c08f09t09, other leg lengths, zero-shot;
  - Unitree B1 quadruped, adapted.

![The three bodies doing the same turn: third-person (top) and egocentric input (bottom)](../results/deck/weekly/bodies_turn.png)

Clips: [forward](../results/deck/weekly/bodies_forward.mp4), [turn](../results/deck/weekly/bodies_turn.mp4), [sideways](../results/deck/weekly/bodies_sideways.mp4).

**Task:** a goal Froude sequence comes from a held-out hexapod clip. The test body selects one candidate per step from its own clip library.

---

## Metrics

Every table names the metric it uses.

| metric | definition | range / better |
|---|---|---|
| Error E | Mean Euclidean (L2) distance between achieved and goal Froude over decision steps | ≥ 0 / lower |
| Normalised score (NS) | (E_random − E) / (E_random − E_oracle), as the RL "human-normalised score". E_random: uniformly random candidate. E_oracle: hindsight oracle, the best candidate per step knowing the outcome | 0 = random, 1 = oracle / higher |
| Pearson r | Pearson correlation between read and true Froude across the 24 candidate actions from the same start state; per channel fwd / lat / yaw | −1 to 1 / higher |
| Top-1 retrieval accuracy | Per start state, subtract the mean over the 24 actions from the predicted and the real 5-step embedding change; correct if the cosine similarity is highest with the same action's real change | chance 0.042 / higher |
| Body-ID probe accuracy | Logistic regression z → body identity, 5-fold CV grouped by clip, classes balanced | chance 0.33 (3 bodies) / lower = more shared |
| Cross-body R² | Ridge regression z → Froude fit on c10 only, evaluated on another body; < 0 = worse than that body's mean | ≤ 1 / higher = more shared |
| k-NN mixing ratio (k = 10) | Share of each point's 10 nearest neighbours (standardised z) from another body, divided by its value under random mixing | 0 = separated, 1 = mixed |
| Cross-body retrieval | For each c10 transition, the nearest neighbour in standardised z among another body's transitions: 1 − L2(Froude, neighbour's Froude) / mean L2(Froude, random transition of that body) | 0 = random, 1 = identical motion / higher |

- **Froude:** with hip height h and gravity g, forward and lateral speed in the body frame are divided by √(g·h), and yaw rate is multiplied by √(h/g).
- **Normalised-score bounds (oracle / random error):** B1 0.034 / 0.127 (library of 23 clips); c08 0.020 / 0.121; c10 0.0135 / 0.1535.

---

## 1. The pipeline

| part | design | shared or per body |
|---|---|---|
| encoder | frozen V-JEPA2, egocentric camera | shared |
| rendering | every body's room, floor and floor texture scaled with its camera height; rendered from the same camera poses, the hexapod and B1 scenes are indistinguishable to a frame-level body-identity probe (0.552, chance 0.5) | — |
| latent action | z = ITM(e_t, e_t+5): five frames per step, driven by a 5-command chunk, as in LAC-WM | shared |
| forward model | FTM predicts e_t+5 from (e_t, z) | shared |
| motion target that trains z | **Froude head: input z only (no frame), one head for all bodies, its gradient trains z**; the task-space counterpart of LAC-WM's end-effector / camera motion target | shared |
| other training losses | reconstruction; separation between real-z and null-z predictions; frozen read-out; 2-step rollout consistency | shared |
| action projector | the body's commands → z | **per body (the only per-body part)** |
| adaptation | Body in joint pretraining: fit its projector (Stage 2). Unseen body: LoRA on ITM/FTM (Stage 1) with the pretraining losses, then fit its projector. | shared head; per-body projector |

**Two selection mechanisms**

| mechanism | score of candidate a (L2 to the goal) | uses the current state |
|---|---|---|
| direct | Froude head(proj(a)) | no |
| rollout | Froude head(ITM(e_t, FTM(e_t, proj(a)))) | yes |

Rollout starting observation:
- **candidate's recorded frame:** the frame saved with that candidate action in its original clip. This is an offline replay comparison.
- **robot's actual current frame:** the observation at the decision point, shared by all candidate actions. This is the state a live selector must plan from.

Read-out window: 11 steps.

---

## 2. Adapting a body that was not in pretraining

**Now**

| stage | what is trained | data |
|---|---|---|
| 1 | LoRA on the ITM / FTM MLP layers, everything else frozen, with the pretraining losses (reconstruction, separation, Froude loss through the frozen Froude head) | a few clips of the new body |
| 2 | the new body's action projector (commands → z) | the new body's clips |

**Before**

| stage | what is trained | data |
|---|---|---|
| 1 | all ITM / FTM weights, with reconstruction and separation losses | a few clips of the new body |
| 2 | the new body's action projector (commands → z) | the new body's clips |
| 4 | the shared Froude head, refit | the new body's clips + hexapod clips, so the hexapod is still read |

**LoRA instead of updating all weights.** Updating all ITM / FTM weights makes the model forget the hexapod; LoRA adds a small low-rank change on top of the frozen weights. Metric: rollout error relative to always predicting the average z (lower is better, 1.0 = no better than the average).

| Stage 1 | hexapod (already trained on) | B1 (new body) |
|---|---|---|
| all weights updated | 0.918 | 0.307 |
| LoRA rank 2 | 0.786 | 0.319 |

With LoRA, 3 clips of the new body are as good as 15 (rollout error: hexapod 0.764 vs 0.766, B1 0.315 vs 0.307).

**LoRA capacity.** Metric: normalised score on B1 selection (w=11): direct / rollout from the candidate's recorded frame / rollout from the robot's current frame.

| Stage 1 | B1 selection |
|---|---|
| LoRA rank 2, 1000 steps | +0.68 / +0.56 / +0.32 |
| LoRA rank 8, 3000 steps | **+0.87 / +0.66 / +0.41** |
| LoRA rank 8, 3000 steps, current B1 data, S0 / S1 | +0.83 / +0.39 / +0.39, +0.74 / +0.29 / +0.29 |

First two rows: earlier B1 data, S0. The old pipeline on the current data (S0): (re-measuring).

---

## 3. Motion target: Froude only, or Froude + per-body joint commands

The two versions are identical except for an extra motion decoder that predicts each body's joint commands (18-d hexapod, 12-d B1). Two seeds each. Metric: normalised score (0 = random, 1 = oracle).

| test | with joint-command decoder, S0 | with, S1 | **Froude only, S0** | **Froude only, S1** |
|---|---|---|---|---|
| c10 same body, direct | +0.91 | +0.90 | +0.91 | +0.91 |
| c10, rollout (candidate's recorded frame) | +0.72 | +0.73 | +0.74 | +0.76 |
| c10, rollout (robot's actual current frame) | +0.64 | +0.59 | +0.61 | +0.66 |
| c08 zero-shot, direct | +0.80 | +0.79 | +0.78 | +0.79 |
| c08, rollout (candidate's recorded frame) | +0.47 | +0.54 | +0.56 | +0.63 |
| c08, rollout (robot's actual current frame) | +0.52 | +0.43 | +0.53 | +0.54 |
| B1 adapted, direct | +0.80 | +0.85 | +0.83 | +0.74 |
| B1, rollout (candidate's recorded frame) | +0.54 | +0.34 | +0.39 | +0.29 |
| B1, rollout (robot's actual current frame) | +0.46 | +0.38 | +0.39 | +0.29 |

B1 rows: B1 adapted with the page 2 pipeline on the current rendering. c10 and c08 need no adaptation.

- The hexapods: equal, Froude only slightly ahead on c08 rollout from the recorded frame.
- The B1: equal on direct; the joint-command decoder is ahead on rollout by about 0.1 (recorded frame +0.34 to +0.54 vs +0.29 to +0.39; current frame +0.38 to +0.46 vs +0.29 to +0.39).
- The current pipeline uses Froude only (one fewer per-body part); on the B1's rollout it costs about 0.1.

---

## 4. Physics closed loop

- **B1:** the chosen candidate's command is executed by the B1's own walking policy in MuJoCo.
- **c08:** the candidate's joint commands drive the physics-simulated body in CoppeliaSim.

Decision every 2 steps, six goals, no B1 run fell. Hexapod-only models; B1 on the earlier rendering. Metric: error E, mean L2 between achieved and goal Froude (lower is better).

In physics, direct selection follows the goal; rollout is close to or worse than random on the B1 (0.081 / 0.108 vs random 0.089) and 3–4× direct's error on c08 for the same goals.

![Physics closed-loop error: direct and rollout, B1 and c08](../results/deck/weekly/physics_summary.png)

| version | B1 direct | B1 rollout | c08 direct | c08 rollout |
|---|---|---|---|---|
| with joint-command decoder, S0 / S1 | 0.072 / 0.073 | 0.073 / 0.098 | 0.053 / 0.050 | 0.071 / 0.078 |
| **Froude only (current), S0 / S1** | **0.061 / 0.063** | 0.081 / 0.108 | **0.047 / 0.045** | 0.085 / 0.055 |
| random candidate | 0.089 | 0.089 | 0.126 | 0.126 |

Same goals, direct vs rollout selection (S0; error = mean L2 between achieved and goal Froude, lower is better):

| goal | c08 direct | c08 rollout | B1 direct | B1 rollout |
|---|---|---|---|---|
| turn | 0.024 | 0.106 | 0.078 | 0.086 |
| forward | 0.041 | 0.141 | 0.068 | 0.081 |

**B1** (goal hexapod | B1 in physics, third-person | B1's egocentric input, with Froude traces)

![B1 turn goal, direct selection](../results/deck/weekly/b1_physics_turn.png)
![B1 turn goal, rollout selection](../results/deck/weekly/b1_physics_turn_rollout.png)

Direct: [turn](../results/deck/weekly/b1_physics_turn.mp4), [forward](../results/deck/weekly/b1_physics_forward.mp4). Rollout: [turn](../results/deck/weekly/b1_physics_turn_rollout.mp4), [forward](../results/deck/weekly/b1_physics_forward_rollout.mp4).

**c08** (goal hexapod third-person | its ego view | c08 in physics, with Froude traces)

![c08 turn goal, direct selection](../results/deck/weekly/physics_turn.png)
![c08 turn goal, rollout selection](../results/deck/weekly/physics_turn_rollout.png)

Direct: [turn](../results/deck/weekly/physics_turn.mp4), [forward](../results/deck/weekly/physics_forward.mp4). Rollout: [turn](../results/deck/weekly/physics_turn_rollout.mp4), [forward](../results/deck/weekly/physics_forward_rollout.mp4).

---

## 5. Trace the rollout failure: FTM → ITM → Froude head

**Follow one candidate action through the blocks.** Solid arrows show rollout; dashed arrows show the real-future control and the direct shortcut.

```mermaid
flowchart LR
    A[Candidate action] --> P[Projector] --> Z[Action z]
    E[Current frame] --> F[1. FTM]
    Z --> F
    F --> Q[Predicted future frame] --> I[2. ITM]
    E --> I
    A -. apply from the same state .-> S[Real future frame]
    E -. same start state .-> S
    S -. ground-truth control .-> I
    I --> Z2[Recovered z′] --> H[3. Froude head] --> V[Predicted body motion]
    Z -. direct scoring skips future .-> H
    classDef ftm fill:#dbeafe,stroke:#2563eb,stroke-width:2px;
    classDef itm fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    classDef head fill:#fee2e2,stroke:#dc2626,stroke-width:2px;
    class F ftm;
    class I itm;
    class H head;
```

- **Selection gap on c10 (the pretraining body):** direct **+0.91**; rollout **+0.74–0.76** from each candidate's recorded frame, **+0.61–0.66** from the robot's actual current frame (normalised score, two seeds).
- **Test:** from one start state, apply each of the 24 candidate actions, render the real future, and read Froude at each point of the chain (24 start states per body).


| Froude reading (Pearson r, forward / lateral / yaw) | hexapod c10 (pretrained) | hexapod c08 (zero-shot) | B1 (adapted, page 2 pipeline) |
|---|---|---|---|
| direct from action | 0.99 / 0.92 / 0.98 | 0.98 / 0.80 / 0.95 | 0.88 / 0.79 / 0.78 |
| candidate's own recorded transition | 0.95 / 0.74 / 0.74 | 0.91 / 0.55 / 0.38 | 0.47 / 0.74 / 0.22 |
| **real** future from the same start state | 0.18 / 0.40 / 0.28 | 0.40 / 0.42 / 0.38 | 0.15 / 0.46 / 0.41 |
| **FTM-predicted** future from that state | 0.25 / 0.36 / 0.60 | 0.35 / 0.31 / 0.62 | 0.32 / 0.24 / 0.45 |

Mean of S0 / S1; 24 start states × 24 actions (c08: × 48). Futures rendered kinematically in each body's training scene; on the B1, futures executed by its walking policy in physics from a saved state give the same drop (0.12–0.19 / −0.43 to −0.10 / 0.45–0.54, earlier checkpoint).

- **Where the information is lost:** already at the real future. Even with a perfect FTM, the read of an alternative action from a shared start state is weak, on all three bodies, including the body the model was pretrained on.

**Is the motion still in z′, or is it the read-out?** Earlier B1 checkpoint (stride 5). A linear probe fitted on same-start alternative transitions, tested on held-out start states:

| read of z′ | real future | FTM-predicted future |
|---|---|---|
| model's Froude head (trained on one-action-per-clip pairs) | 0.24 / 0.52 / 0.24 | 0.10 / 0.23 / 0.21 |
| linear probe fitted on alternative transitions (one probe per column) | **0.58 / 0.60 / 0.56** | **0.48 / 0.63 / 0.72** |

| rollout selection, robot's current frame (w=11) | normalised score |
|---|---|
| model's Froude head | +0.40 |
| the probe (FTM-predicted fit) replacing the head | +0.33 |

**Conclusion.**
- z′ keeps decodable motion information (probe r ≈ 0.6 on held-out start states), but the Froude head, trained on one-action-per-clip transitions, does not generalise to same-start alternatives.
- Refitting the read-out alone on a small set of alternative transitions (8 start states, one room) did not improve selection (+0.40 → +0.33).
- The same failure occurs on the pretrained c10 body: a training-data problem, not specifically a cross-body one.
- Fixing it needs diverse alternative transitions across states and rooms, including multi-step imagined rollouts, during joint training of the ITM, FTM and Froude head (page 8).

---

## 6. Does z transfer across bodies?

The Froude head reads the same motion from c10, c08 and B1 only if their latents carry compatible meanings. We test this with a readout fitted on c10 and applied to the other bodies **without refitting**. Body-ID, mixing and retrieval test whether the wider 64-D space is shared too. Rollout selection remains the practical test.

| pretraining | body-ID probe | cross-body R² c10 → c08 | cross-body R² c10 → B1 | k-NN mixing | retrieval c10 → c08 | retrieval c10 → B1 |
|---|---|---|---|---|---|---|
| hexapod only (B1 not adapted) | (re-measuring) | | | | | |
| hexapod only, B1 adapted (page 2 pipeline; mean S0 / S1) | 0.68 | +0.70 / +0.34 / +0.17 | −0.73 / +0.11 / −0.04 | 0.47 | 0.38 | 0.25 (0.25 / 0.24) |
| joint, no similarity loss | 0.73 | +0.76 / +0.38 / +0.18 | −1.40 / −0.07 / −0.87 | 0.35 | 0.40 | 0.22 |
| joint + Froude-similarity loss | 0.73 | +0.71 / +0.30 / +0.04 | −1.59 / −0.09 / −1.06 | 0.36 | 0.44 | 0.24 |

Joint pretraining uses c10 + B1 (48 clips each, B1 on the current rendering). The similarity loss encourages transitions with similar Froude motion to have similar z.

![z of the three bodies, joint pretraining without similarity loss: PCA and UMAP, coloured by body (top) and behaviour (bottom)](../results/deck/shared_latent_jointD3/latent_jointD3_nosim_s0.png)

- z is shared between the two hexapods (c10 → c08 forward R² +0.71 to +0.76, retrieval 0.40–0.44).
- z is not shared with the B1 in any version: retrieval 0.22–0.25 (hexapods 0.38–0.44), forward R² negative. Joint pretraining and the similarity loss do not change this.

---

## 7. Selection when the B1 is in pretraining

Joint models are used as pretrained: only the B1's action projector is fitted. The joint models were pretrained before a B1 label correction (hard-turn clips); their projectors and all evaluation use the corrected data, and retraining is planned. The first column adapts the B1 to a hexapod-only model with the adaptation pipeline of page 2. Metric: normalised score (0 = random, 1 = oracle), w=11.

| test | hexapod-only pretraining; B1 adapted (S0 / S1) | joint, no similarity loss (S0) | joint + similarity loss (S0) |
|---|---|---|---|
| B1 direct | +0.83 / +0.74 | +0.92 | +0.94 |
| B1 rollout (candidate's recorded frame) | +0.39 / +0.29 | +0.61 | +0.45 |
| B1 rollout (robot's actual current frame) | +0.39 / +0.29 | +0.55 | +0.57 |
| B1 direct, goal read from vision | +0.60 / +0.46 | +0.84 | +0.84 |
| B1 rollout (current frame), goal read from vision | +0.25 / +0.35 | +0.56 | +0.57 |
| c08 direct (zero-shot) | +0.78 | +0.78 | +0.79 |
| c10 direct (same body) | +0.91 | +0.90 | +0.91 |
| c10 direct, goal read from vision | +0.82 / +0.84 | +0.87 | +0.85 |

- Joint pretraining gives the best B1 selection (direct +0.92 to +0.94, rollout +0.45 to +0.61) without a shared z: each body's projector and the shared head carry it.
- The similarity loss adds nothing measurable (one seed each).
- Reading the goal from vision instead of the recorded Froude costs 0.08–0.28 on B1 direct (least for the joint models) and 0.03–0.09 on c10 direct.

Unless marked, the goal is the goal clip's recorded Froude. "Goal read from vision": the goal clip's frames read through the model's own ITM + Froude head (11-step window); the picked action is still graded by its real Froude.

---

## 8. The training data: what it lacks, and the plan

**1. Counterfactual transitions.** Each training clip holds one command for its whole length, so from any state the model only ever sees one action and its future. It never sees "same state, different action, different future". Because of this the Froude read-out cannot judge another action's future: on the hexapod c10, Pearson r between read and true Froude (fwd / lat / yaw) is 0.95 / 0.74 / 0.74 on the clip's own future, but 0.18 / 0.40 / 0.28 when another action is applied from the same start state. The same drop on c08 (forward 0.91 → 0.40) and the B1 (0.47 → 0.15).

| | now | should be |
|---|---|---|
| actions seen from one state | 1 | many (all 24 commands from saved states) |
| B1 | one command per clip | save the MuJoCo state, run each command from it, record each future |
| hexapod | one command per clip | clips that switch command every ≥ 15 steps |
| training on alternatives | none | ITM, FTM and Froude head trained together on them, including multi-step imagined rollouts |
| rooms per start state | one | several |

**2. Frame / view style.** Each body is rendered in one style with small image changes, so the model can recognise a body's scene instead of its motion. Changing only the B1 floor tiling and texture scale drops the B1 direct normalised score from +0.94 to +0.24 / −0.08. We follow Egocentric VSM's view randomisation.

| | now | should be (Egocentric VSM) |
|---|---|---|
| crop | 85–100% | 10–100% |
| brightness | ±20% | ×0.1–10 |
| blur | none | kernel 3–41 |
| colour, noise | none | yes |
| ground textures | one per body | several, the same set for every body |
| lighting | fixed | varied |

**3. Amount.** 48 clips per body, ≈2.9k training pairs (frame t → t+5). LAC-WM uses 150k trajectories.

| | now | should be |
|---|---|---|
| clips per body | 48 | test 48 vs 48 + switching vs 96, then scale up |

☐ view augmentation: pilot ready (current, safe, Egocentric VSM strength; 25% of samples kept clean)  ☐ counterfactual branches: tool ready  ☐ randomised rendering, more data: planned

**Next**

| run | question |
|---|---|
| Pretraining with alternative transitions (physics branches on the B1, switching clips on the hexapod) | Does the read of an alternative action's future reach the own-transition level, and rollout reach direct? |
| View augmentation pilot | Does B1 selection survive a rendering change without adaptation, with no loss on the training rendering? |
| Gecko held out | Does a body never seen join, zero-shot and after a small adaptation? |

---
---

# Weekly Update: 2026-10-08 (draft)

Period: 2026-10-01 to 2026-10-08. The section above (2026-09-30) is kept as it was; every number in it was measured on the
earlier data and is superseded by the numbers below.

## Where last week ended, and this week's question

- **Last week:** direct selection followed the goal; rollout did not. A model trained on one command per clip could not
  read the future of a different action from the same state (Pearson r 0.15–0.46).
- **Hypothesis:** the training data never shows "same state, different action".
- **This week:**
  1. Rebuild the data so that it does, identically for both bodies.
  2. Check whether rollout now works, with the same tests as last week.
  3. Check that the result does not depend on how the scenes were rendered.
  4. Start the real test: a robot the model knows nothing about, adapted only from its own random motion.

![Last week: same start state, different actions; each model's prediction shown as the real future it is closest to](../results/deck/weekly/action_future_strip.png)

---

## The data, rebuilt

| | now | last week |
|---|---|---|
| splits | train / validation / held-out; no file, frame or room in two splits | test clips overlapped training |
| behaviours | 24 per body; clips cut from one long walk, every clip a different start state | hexapod: the same motion repeated in several rooms |
| Froude labels | at the centre of mass, 1 s average, never across a command switch | at the head, zero-padded edges |
| same state, other actions | **branches** (below) | none |
| both bodies | identical design, room seeds and amounts | different |

**How a branch is made.** Each behaviour's clip gets 3 branch points (early / middle / late in a step). At each point the
robot's state is saved and all 24 behaviours are run from it, the clip's own behaviour included. So every pair
(behaviour before → behaviour after) occurs, from 3 different moments of the gait. Each branch: 10 frames before the switch,
20 after.

**Count (per body, training split):** 24 behaviours × 2 clips = 48 clips; 48 clips × 3 branch points × 24 behaviours =
**3,456 branches**. Validation and held-out: 24 clips × 3 × 24 = 1,728 each.

![How a branch is made: top-down centre-of-mass paths of all 24 commands from one saved state (black dot), hexapod and B1, in the body frame at the switch; black = the 10 shared frames, colour = command family, dashed = own command. Bottom: the camera view 1 s after the switch for four commands. Turning commands also slide the body slightly outward in the first second (lateral label of the same sign)](../results/deck/weekly_1008/branch_diagram.png)

| per body | last week | now |
|---|---|---|
| training clips (3.3 s each) | 48 | 48 main clips + **3,456 branches** |
| training frames | ≈3,200 (2.6 min) | ≈110,000 (**1.5 h**) |
| training pairs (frame t → t + 5) | ≈2,900 | **40,704** (2,688 + 38,016), about 14× |
| actions seen from one state | 1 | 24 |
| held-out | 24 clips (overlapping training) | 24 clips + 1,728 branches, unseen rooms |

Clips: [hexapod branches](../results/deck/weekly_1008/branches_hexapod.mp4), [B1 and hexapod branches](../results/deck/weekly_1008/branches_b1_hexapod.mp4), [shorter-legged hexapod test clips](../results/deck/weekly_1008/c08_test_clips.mp4).

---

## Result 1: training on other actions' futures makes the model read them

Joint pretraining (six-legged hexapod + B1), the same number of steps, without and with branches; held-out branches.

Metric: Pearson r between the motion read by the model and the true motion, across the 24 behaviours run from one state;
per channel forward / lateral / yaw (higher is better, 1 = perfect ranking).

| read | hexapod c10 | hexapod c08 (never trained) | B1 |
|---|---|---|---|
| direct from the action | 0.97 / 0.85 / 0.78 | 0.97 / 0.81 / 0.76 | 0.98 / 0.98 / 0.47 |
| **real** future, no branches | 0.32 / 0.34 / 0.16 | 0.36 / 0.40 / 0.11 | 0.26 / 0.48 / 0.23 |
| **real** future, with branches | **0.90 / 0.68 / 0.86** | **0.91 / 0.69 / 0.84** | **0.86 / 0.73 / 0.64** |
| **predicted** future, no branches | 0.57 / 0.57 / 0.26 | 0.65 / 0.56 / 0.22 | 0.54 / 0.76 / 0.34 |
| **predicted** future, with branches | **0.85 / 0.80 / 0.41** | **0.85 / 0.76 / 0.34** | **0.83 / 0.85 / 0.42** |

- The read of another action's real future goes from 0.1–0.5 to 0.64–0.91, including the never-trained hexapod.
- Predicted yaw stays the weakest channel (0.34–0.42). The action projector places yaw on a direction the forward model does
  not read; fitting the projector through the forward model raises hexapod yaw to 0.61.

---

## Result 2: rollout now comes close to direct

Columns: joint pretraining without / with branches (two seeds); hexapod-only pretraining with B1 adapted on 44 clips.

Metric: normalised score from error E, the mean L2 between achieved and goal Froude over the decision steps:
(E_random − E) / (E_random − E_oracle); 0 = random candidate, 1 = best candidate in hindsight (higher is better). The
decision span is now 21 frames (1 s, the length of the label) instead of last week's 11.

| test | no branches | branches, S0 / S1 | hexapod-only, B1 adapted |
|---|---|---|---|
| c10 direct | +0.94 | +0.92 / +0.92 | +0.93 |
| c10 rollout (robot's current frame) | +0.59 | +0.71 / +0.68 | +0.69 |
| c08 direct (zero-shot) | +0.91 | +0.90 / +0.90 | +0.91 |
| c08 rollout (current frame) | +0.58 | +0.68 / +0.64 | +0.67 |
| B1 direct | +0.97 | +0.94 / +0.94 | +0.97 |
| B1 rollout (current frame) | +0.72 | +0.75 / +0.73 | +0.70 |
| B1 direct, goal read from video | +0.35 | +0.84 / +0.83 | +0.83 |
| B1 rollout, goal read from video | +0.24 | +0.63 / +0.62 | +0.57 |

**Goal from the recorded motion (proprioceptive) vs goal read from the goal video**, B1, same metric:

| pretraining | recorded goal: direct / rollout | goal read from video: direct / rollout |
|---|---|---|
| joint, no branches | +0.97 / +0.72 | +0.35 / +0.24 |
| joint, with branches | +0.94 / +0.75 | **+0.84 / +0.63** |
| joint, with branches, random rooms | +0.94 / +0.71 | +0.58 / +0.41 |

- Same setup, no branches vs with branches: rollout from the current frame c10 +0.59 → +0.71 / +0.68, c08 +0.58 → +0.68 / +0.64; B1 +0.72 → +0.75 / +0.73.
- Reading the goal from video improves most (B1 +0.35 → +0.84).
- A body absent from pretraining (hexapod-only, B1 adapted) is close to joint pretraining.
- Rollout is not expected to beat direct on this test: goals and candidates are steady walking, where one command has one
  outcome.


---

## Is anything in the data inflating these results?

Results 1 and 2 were measured in rooms scaled to each body's camera height (hexapods 8 m, B1 17.65 m), so every body saw a
similar-looking world. Before testing a new robot, we checked whether this, or anything else in the data, does the work instead
of the motion.

| check | result | consequence |
|---|---|---|
| **room size:** held-out data re-rendered with room size random in 8–26.5 m for every body, same physics, same models | hexapod and every command-driven read unchanged; reading B1's motion from real video drops (below) | all data re-rendered with random room size and start position; pretraining repeated |
| **one frame shows the speed?** ridge probe on a single frame | R² up to 0.69 in rooms seen in training, ≤ 0.13 in new rooms | the room, not the image, carried it |
| **do forward / lateral / yaw describe the camera's motion?** | with the fixed camera offset, R² ≥ 0.99 at 1 s; left out: gait sway | labels are complete for planar motion |
| **wall distance as a speed cue?** | within-clip correlation −0.07 to +0.13 | no shortcut found |

Metric: normalised score from E (as in Result 2); Pearson r for the real-future read.

| room size (same models as Result 2) | original rooms | random size |
|---|---|---|
| hexapod selection, direct / rollout | +0.92 / +0.71 | +0.92 / +0.70 |
| B1 selection, direct / rollout | +0.94 / +0.75 | +0.94 / +0.73 |
| B1 goal read from video, direct / rollout | +0.84 / +0.63 | +0.77 / +0.51 |
| B1 real-future read, forward | 0.87 | 0.39 |

Result 2's selection does not depend on the room scaling; its B1 video-reading numbers partly do (B1 had seen one room size).
Each table below names the rooms its model was trained and tested in.

**Random-room setup.** Every clip is re-rendered with the same physics and labels; only the room changes:
- room size drawn from 8–26.5 m (log-uniform) and the clip's start position drawn inside it;
- room colours and floor texture as in the original rooms: one room per clip (a clip and all its branches share it), in both setups.

**Room size and a new robot.** B1 absent from pretraining (hexapod only), adapted on 44 tuned clips, against B1 in joint pretraining.

Metric: normalised score from error E, mean L2 between achieved and goal Froude (0 = random, 1 = best in hindsight).

| | pretraining (bodies, rooms) | B1 data in the model | test rooms | direct | rollout |
|---|---|---|---|---|---|
| 1 | hexapod + B1, original rooms | 40,704 B1 pairs (in pretraining) | original | +0.94 | +0.75 |
| 2 | hexapod only, original rooms | 44 tuned clips (2.4 min), original room | original | +0.97 | +0.70 |
| 3 | hexapod + B1, random rooms | 40,704 B1 pairs (in pretraining) | random | +0.94 | **+0.71** |
| 4 | hexapod only, random rooms | 44 tuned clips (2.4 min), random rooms | random | +0.94 | +0.44 |

(B1 adapted on 44 tuned clips, no babbling; tuned library; direct / rollout from the robot's current frame)

- **1 vs 2:** in the original rooms, adapting on 2.4 min comes within 0.05 of having B1 in pretraining.
- **3 vs 4:** in random rooms, pretraining keeps +0.71; adapting on 2.4 min reaches +0.44.
- Rows 2 and 4 use the same 44 clips with the same room colours; they differ only in room size and start position. With
  44 clips, each at one size, adaptation does not reach the level of B1 in pretraining.
- Same 44 clips, each rendered in 3 random-size rooms (same motion, different size, colours and start position) for
  adaptation: rollout +0.48 (1 room per clip: +0.44). Room variety in the adaptation clips alone does not close the gap to row 3.

![The same moment of one held-out clip in its original room and in 8 / 14.6 / 26.5 m rooms, hexapod and B1](../results/deck/weekly_1008/room_sizes.png)

---

## Does z transfer across bodies?

Same tests as last week, on held-out clips:
- body-ID probe: logistic regression z → body (chance 0.33, lower = more shared);
- cross-body R²: a read-out fitted on the six-legged hexapod's z only, applied to another body without refitting (higher =
  more shared);
- k-NN mixing: share of each point's 10 nearest neighbours from another body, relative to random mixing (1 = fully mixed);
- retrieval: how similar in motion each hexapod transition's nearest neighbour from another body is (1 = identical motion).

| pretraining | body-ID probe (chance 0.33, lower = more shared) | cross-body R² c10 → c08 (fwd / lat / yaw) | cross-body R² c10 → B1 (fwd / lat / yaw) | k-NN mixing | retrieval c10 → c08 | retrieval c10 → B1 |
|---|---|---|---|---|---|---|
| last week: hexapod only, B1 adapted | 0.68 | +0.70 / +0.34 / +0.17 | −0.73 / +0.11 / −0.04 | 0.47 | 0.38 | 0.25 |
| joint, no branches | 0.48 | +0.59 / +0.43 / +0.34 | +0.09 / −0.28 / −0.90 | 0.50 | 0.52 | 0.04 |
| joint, with branches (S0 / S1) | 0.55 / 0.53 | +0.76 / +0.52 / +0.47 · +0.77 / +0.43 / +0.49 | −0.63 / +0.14 / +0.04 · −0.27 / +0.53 / −0.34 | 0.46 / 0.46 | 0.63 / 0.66 | 0.25 / 0.30 |
| hexapod only with branches, B1 adapted (44 tuned clips) | **0.40** | +0.74 / +0.42 / +0.53 | **+0.27 / +0.17 / +0.15** | **0.62** | 0.63 | 0.30 |
| hexapod only, random rooms, B1 adapted from 88 babbling clips | 0.46 | +0.56 / +0.42 / +0.37 | −0.06 / −0.11 / +0.08 | 0.58 | 0.58 | 0.12 |
| joint, random rooms | 0.50 | +0.65 / +0.34 / +0.39 | +0.15 / −1.54 / −0.47 | 0.49 | 0.57 | −0.05 |

- Between the two hexapods, branches make z more shared (retrieval 0.52 → 0.63–0.66; forward R² +0.59 → +0.76).
- With B1, joint pretraining keeps the bodies more apart (body-ID 0.50–0.55). Pretraining on the hexapod alone and adapting
  B1 gives the most shared z: body-ID 0.40, mixing 0.62, and the only positive cross-body R² on all three channels.

![z of the three bodies, joint pretraining with branches: PCA and UMAP, coloured by body (top) and behaviour (bottom)](../results/deck/weekly_1008/latent_with_branches.png)

---

## Physics closed loop

Model: joint pretraining with branches, **original rooms** (the model of Results 1 and 2). Pretraining, goals and the live
camera view all in the original rooms (sized to the body).

Every 2 frames the planner picks one candidate, by direct or by rollout from the robot's live camera view; 33 decisions per run,
6 goals = held-out hexapod clips, one episode per goal, no falls.
- **B1:** walks under its own policy in MuJoCo; candidates = its 24 tuned held-out clips.
- **Six-legged hexapod (c10):** CoppeliaSim, joint commands; candidates = its 48 training clips.
- **Shorter-legged hexapod (c08, never trained, zero-shot):** candidates = its 24 held-out clips.
- Hexapods: the chosen candidate's gait recipe runs on one continuous gait clock, with a 4-frame blend at every switch (as the
  branches were recorded).

Metric: error E, mean L2 between achieved and goal Froude over the decision steps (lower is better).

| goal | B1 direct | B1 rollout | c10 direct | c10 rollout | c08 direct | c08 rollout |
|---|---|---|---|---|---|---|
| turn 0.29 | **0.046** | 0.064 | **0.048** | 0.103 | **0.051** | 0.108 |
| turn 0.56 | **0.035** | 0.090 | **0.039** | 0.165 | **0.028** | 0.168 |
| sideways left 0 | **0.015** | 0.081 | **0.029** | 0.103 | **0.035** | 0.205 |
| sideways right 1 | **0.021** | 0.059 | **0.028** | 0.035 | **0.036** | 0.044 |
| speed 7.1 | **0.039** | 0.048 | **0.042** | 0.049 | — | — |
| speed 8.8 | **0.071** | 0.103 | **0.077** | 0.125 | **0.067** | 0.113 |
| **mean** | **0.038** | 0.074 | **0.044** | 0.097 | **0.043** | 0.128 |

(c08 speed 7.1: not run, the start view did not pass the camera check.)

Achieved motion on the turn 0.56 goal, Froude forward / lateral / yaw:

| | goal | direct | rollout |
|---|---|---|---|
| c10 | 0.15 / −0.07 / 0.06 | 0.15 / −0.06 / 0.07 | 0.00 / −0.07 / 0.01 |
| c08 | 0.15 / −0.07 / 0.06 | 0.15 / −0.07 / 0.07 | 0.00 / −0.05 / 0.01 |

- Direct follows every goal on all three bodies (E 0.038–0.044).
- Rollout follows the sideways-right and speed goals; on the turn and sideways-left goals it matches only the lateral part of
  the goal: forward and turning are missing (B1 keeps forward but turns too little).
- Offline, with the same model, candidates and goals, rollout has the same error on these goals (c10 turn 0.56: 0.176 offline,
  0.165 in physics); the average over all 24 goals (+0.71) hides it.

---

## B1 adapted from random commands: do random candidates still work?

**Question.** So far B1's candidates were 24 clips tuned to the hexapod's behaviours: the library whose motion lies closest to
every goal (lowest error of the best candidate in hindsight), i.e. an upper bound. A new robot has no such clips. If its candidates are random clips that only cover the motion
(Froude) range, does selection still work?

**Setup.** Pretrained on the hexapod only (clips + branches), B1 never seen. B1 adapted (LoRA on ITM + FTM, new projector) from
random velocity commands to its trained walking policy (no task knowledge), 0.6 to 9.7 min, random rooms. Candidates: 24 of
these random clips, picked by clustering B1's own motion (never by the goals). Goals: held-out hexapod clips.

![Measured motion (CoM Froude) of the hexapod's pretraining data (grey), B1's random-command data (orange) and the tuned B1 clips (blue)](../results/deck/weekly_1008/b1_babbling_coverage_scatter.png)

- The random-command data covers the hexapod's motion range and the combinations in between.

Metric: normalised score from error E, mean L2 between achieved and goal Froude (0 = random candidate, 1 = best candidate in
hindsight).

| minutes of random commands (clips) | 0.6 (11) | 1.2 (22) | 2.4 (44) | 4.8 (88) | 9.7 (176) |
|---|---|---|---|---|---|
| random candidates: direct | +0.61 | **+0.72** | +0.56 | +0.50 | +0.47 |
| random candidates: rollout | +0.01 | +0.08 | +0.12 | +0.16 | +0.16 |
| tuned candidates (upper bound): direct | +0.86 | +0.87 | +0.86 | +0.88 | +0.87 |
| tuned candidates (upper bound): rollout | +0.14 | +0.22 | +0.24 | +0.41 | +0.32 |

![Score against minutes of random commands, tuned candidates (left) and random candidates (right); dotted = B1 in pretraining](../results/deck/weekly_1008/b1_babbling_budget.png)

- **Direct works with random candidates** from about 1 min of data (+0.72); the best random candidate is further from the goals
  than the best tuned clip (error 0.076 vs 0.023), so the scale is harder.
- **Rollout fails with random candidates** (+0.01 to +0.16, close to random). Even with tuned candidates it reaches only +0.41;
  tuned clips as adaptation data give the same +0.44 in random rooms (+0.70 in the original rooms), so the room setup limits
  rollout first. Random candidates switch command inside the clip and lie close together in motion, which rollout must separate
  through the predicted future.

---

## Next: a test where the world model is needed, and the plan

**1. Rollout on turning and sideways goals.** On flat open ground the outcome of a command does not depend on the view, so
direct is optimal there; rollout still has to read forward and turning from the imagined future, and on the turn goals it
does not (page above). Fitting the action projector through the forward model does not change it offline (c10 turn 0.56:
0.176 → 0.172). Holding each decision longer (2 → 8 steps, 0.5 → 2 s) lowers rollout's mean error offline
(0.094 → 0.077) and fixes the milder turn (0.123 → 0.077), but not the strongest turn (0.176 → 0.169).

**2. Wall test: predictive vs reactive.** The robot walks toward a wall; a controller that does not use the view (direct) keeps
walking, rollout can see the wall coming in the imagined future and turn early.

| | now | planned |
|---|---|---|
| data | walking in open rooms, walls far | an automatic controller walks toward walls and turns when close (both bodies); several commands from the same state near a wall |
| score | Froude only | Froude + distance to the wall read from the imagined view |
| compared | direct, rollout | direct (no view), reactive rule (turn when close), current view without prediction, rollout |
| metric | Froude error | collision rate, closest distance to the wall, progress |

First check, B1 walking straight at a wall (24 clips, 1–6 m, original rooms): the distance to the wall is readable from one
frozen-encoder frame (R² 0.96, error 0.30 m) and from the forward model's predicted frames up to 2 s ahead (R² 0.90–0.96). On
straight walking the prediction is not better than the current frame (the distance changes by ≈ 0.5 m in 2 s), so the test
needs states near a wall with several commands, as the branches.

**3. Rooms and view.** Rooms stay as in the original setup (sized to the body, one room per clip); the view is randomised
instead: brightness, texture and lighting.

**4. Later:** a robot with no controller (four-legged hexapod, CPG babbling), random candidates.

☐ rollout on the strongest turn: per-candidate read  ☐ wall data: design  ☐ view randomisation: planned  ☐ paper outline (IROS, 8 pages)
