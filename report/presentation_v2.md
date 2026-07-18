# Presentation v2
### Cross-Morphology Locomotion via Latent Action World Models

Structure follows the investigative arc: state the problem, prove it is real, show why existing
answers do not reach it, confront the strongest objection, then walk through what was actually
tried, what failed, and what the failure taught. Results are reported as they came, including the
two that forced a change of method.

---

## Slide 1. Title

Cross-Morphology Locomotion via Latent Action World Models

Can a robot learn *what a movement is*, separately from *which body performed it*?

Candidate: Disthorn Suttawet
University advisor: Mr. Bawornsak Sakulkueakulsuk (IFR, KMUTT)
Lab advisor: Prof. Poramate Manoonpong (Bio-inspired Robotics and Neural Engineering Lab, VISTEC)

---

## Slide 2. How this project arrived here

The direction changed three times. Each change came from a specific finding, not a preference.

| Stage | Direction | What forced the change |
|---|---|---|
| Initial | Learn locomotion priors from real animal video | Real video has no action labels, so the latent action cannot be grounded |
| Pivot 1 | Simulation only, one species, three leg lengths | Advisor review, weeks 4 and 5: simulation supplies action labels for free |
| Pivot 2 | CoppeliaSim, not IsaacSim | The lab's *Medauroidea* model runs on CoppeliaSim; the earlier plan named the wrong simulator |
| Pivot 3 | Validation by linear probe and silhouette, not PCA | PCA answers a different question than the one being asked (Slide 16) |

Speaker note: the point of this slide is that the scope narrowed on purpose. Narrowing removed the
part of the problem that could not be measured.

---

## Slide 3. The problem: a policy is welded to one body

Training a locomotion policy means fitting a network that maps robot state to joint action. That
mapping absorbs the body it was trained on. Change limb proportions, mass distribution, or lose a
leg, and the mapping no longer produces walking.

The consequence is economic. Every new morphology restarts training from zero, at hours to days of
compute per body. The field calls this cross-embodiment generalization, and for legged systems it
remains open.

---

## Slide 4. First, prove the problem exists

An experiment is only meaningful if the bodies genuinely behave differently. If the same command
produced the same motion on every body, there would be nothing to transfer and nothing to study.

Three *Medauroidea extradentata* variants in CoppeliaSim, identical in topology and in their
18-dimensional joint action space, differing only in leg length. All three received a bit-identical
command sequence.

| Morphology | Distance travelled (5 episodes, 200 steps) |
|---|---|
| Long, 1.0x | 4.125 m, standard deviation 0.434 |
| Medium, 0.75x | 3.562 m, standard deviation 0.015 |
| Short, 0.5x | 2.646 m, standard deviation 0.002 |

The slowest long-leg run still exceeded the fastest short-leg run, so the groups do not overlap.
Foot swing clearance separates them as well: short legs stay in a narrow band near 0.13 to 0.16 m
while long legs scatter between 0.05 and 0.38 m.

Identical input, different outcome. The gap is real and it is measurable.

---

## Slide 5. Prior work 1: world models learn dynamics, but one domain at a time

DreamerV3 (Hafner et al., 2023) learns a compact latent state and imagines rollouts inside it, so the
policy trains without touching the environment. One algorithm with one hyperparameter set covers more
than 150 domains.

The limitation for this project: each domain still requires its own world model. Nothing carries
across bodies, and the method assumes explicit action labels throughout.

---

## Slide 6. Prior work 2: latent actions remove the dependence on shared action labels

LAC-WM (Huang et al., ICML 2026) discards explicit action labels as the conditioning signal. An
inverse model infers an abstract action z from consecutive observations, and the world model is
conditioned on z rather than on any robot's native command vector. One latent space then covers
several embodiments, and adding embodiments improves it rather than fragmenting it.

This is the architecture the present work adapts. Two limits matter here. LAC-WM was demonstrated on
manipulation only, and the embodiments it unifies have genuinely different action spaces: 10
dimensions for a Franka end effector, 20 for a bimanual humanoid, 138 for human hand keypoints.

---

## Slide 7. Prior work 3: the two closest results, and what each still requires

QWM (Danesh et al., 2026) reaches zero-shot transfer to unseen quadruped morphologies. It does so by
reading limb lengths, mass, and torque limits out of each robot's CAD file and conditioning the world
model on those numbers. It uses proprioception rather than vision, and it has no latent action: the
action stays the raw 12-dimensional joint target.

Li et al. (RA-L 2021) plan in a learned latent action space for a hexapod and a quadruped. Reading
the paper closely, the two robots are trained separately. Each has its own latent space, its own
policy, its own dynamics model. No experiment trains on one body and evaluates on another, and limb
length is never varied.

Neither one asks whether a latent action *organizes itself* by behaviour when nobody supplies the
morphology.

---

## Slide 8. The objection that decides this project

Raised in advisor review, week 4, and never yet answered:

> If the policy and the robot both need joint commands in the end, why convert to a latent space
> at all, and then convert back?

The objection sharpens against our own setting. LAC-WM needs a latent action because its embodiments
cannot share an action vector. Our three variants share an identical 18-dimensional command space.
On that reading, the motivation for a latent action disappears.

The answer is that the heterogeneity is not in the action space. It is in the dynamics. Slide 4 is
the evidence: the same 18 numbers produce 4.13 m of travel on one body and 2.65 m on another. The
latent action is not being asked to reconcile different action formats. It is being asked to describe
what the body *did* in a way that survives the body changing.

This reframing is testable, and Slide 19 gives the experiment that decides it.

---

## Slide 9. Pipeline

Phase 1, pretraining. The encoder stays frozen; three modules are trained.

```
frame_t ─┐
         ├─▶ [V-JEPA2, frozen] ─▶ e_t, e_{t+1}
frame_t+1┘                          │
                                    ├─▶ [ITM] ─▶ z_t (64-d latent action)
                                    │              │
                          e_t ──────┴──────────────┼─▶ [FTM] ─▶ ê_{t+1}   L_recon
                                                   │
                                                   └─▶ [Motion Decoder] ─▶ â_t   L_motion
                                                          (discarded after pretraining)
```

| Block | Input | Output | Trained |
|---|---|---|---|
| V-JEPA2 encoder | RGB 256x256x3 | e_t, 256 x 1408 | No, frozen |
| Inverse Transition Model | [e_t, e_{t+1}], 512 tokens | z_t, 64 dimensions | Yes |
| Forward Transition Model | [e_t, z_t] | ê_{t+1}, 1408 | Yes |
| Motion Decoder | z_t queries e_t | â_t, 18 joint targets in radians | Yes, then discarded |

L = L_recon + lambda * L_motion, with L_recon computed in embedding space, so no pixel decoder is
needed.

Correction against the previous version of this deck: the latent action is 64-dimensional, not 512.
Table 4 of LAC-WM reports 512 as the internal hidden width of the ITM and FTM. Section 4.2 states the
action embedding dimension separately, and it is 64.

---

## Slide 10. Visual encoder and one non-obvious constraint

V-JEPA2 is self-supervised on roughly one million hours of video, with an objective that predicts
masked content in representation space rather than pixel space. That objective rewards motion-relevant
features, which is why it is a reasonable starting point for gait.

The constraint: the released checkpoint is a 64-frame *video* encoder with 2-frame tubelets. Feeding a
real clip lets every frame attend to every other frame, so e_t would carry information from the future.
Measured directly: the same frame encoded alone against the same frame embedded in a clip agrees at
only 0.52 cosine similarity.

Fix: duplicate each frame into the minimal 2-frame tubelet and encode it alone. The output becomes
independent per frame, and it is bit-exact reproducible. V-JEPA 2-AC uses the encoder the same way.

---

## Slide 11. What we had to build before any of this could run

The lab's CoppeliaSim scene is state-only. It has no camera, and no code anywhere in either
repository captures RGB from it. Every V-JEPA2 experiment before this point ran on recorded footage of
a Unitree B1 quadruped from other renderers, which means the encoder had never once seen the stick
insect.

Four failures surfaced while building the missing capture path, each silent rather than loud:

1. Vision sensors in CoppeliaSim look along +Z. Pointing them along -Z returns a fully black frame
   with minimum depth 1.0, and no error.
2. A new sensor defaults to visibility layer 8. The robot sits on layer 1 and the floor on 32768, so
   the sensor renders nothing at all and still reports success.
3. The default floor is a checkerboard, which aliases under sub-pixel motion and creates pixel change
   where no motion occurred.
4. At 30 degrees elevation with a 60 degree field of view, the horizon sits on the top edge of the
   frame, leaving roughly 15 percent of every image as empty black.

Items 3 and 4 both matter for a specific reason given on the next slide.

---

## Slide 12. Background choice is a measurement decision, not decoration

Two backgrounds were tested and both corrupted the signal, in opposite directions.

A checkerboard produces correlation of -0.16 between real pixel motion and embedding change
(p = 7.6e-24), because aliasing moves pixels where nothing moved. A blank white background produces
-0.20 (p = 4.7e-37), and the cause is the opposite: featureless patches carry no information, so the
transformer reuses those tokens as internal working space and their embeddings fluctuate more than the
moving robot's do.

The working configuration is a matte, lightly textured, non-repeating floor, with elevation 40 degrees
and a 45 degree field of view. Empty black pixels drop to 0.00 percent.

The camera is created by script rather than placed by hand, so its pose relative to the robot is
identical across all three morphologies by construction. Measured offset is [0, 1.532, 1.286] for
every variant, and mean frame brightness lands at 128.3, 129.0, and 129.4. If the camera drifted
between recording sessions, the later clustering result would measure the session rather than the
morphology.

---

## Slide 13. Encoder check: is the behaviour signal present at all?

Before training anything, the frozen features must contain the signal the ITM is supposed to extract.
Otherwise the pipeline has nothing to work with.

Data: three morphologies, five episodes each, 200 steps, 3000 frames total, locked render environment.

Result. A linear probe recovers gait phase from e_t at 85.1 percent against a 12.5 percent chance
baseline, rising to 92.7 percent when whole episodes are held out. A shuffled-label control lands at
12.3 percent, which is chance, so the result is not the probe memorising 1408 dimensions from 1440
samples.

The behaviour signal is present. The gate passes.

---

## Slide 14. The same check, read a second way, says the opposite

Silhouette score on the same embeddings, labelled by gait phase, is -0.0222. Read alone, that value
says the phase signal is absent.

Both numbers are correct, because they answer different questions. A probe measures whether
information is *present* and linearly recoverable. Silhouette measures whether that information is the
*dominant* axis of variation in the space. Phase is present but not dominant, and the noise floor
confirms it: frames at the same phase sit 40.23 apart in the embedding space while frames at different
phases sit 44.93 apart, a ratio of only 1.12.

This matters beyond our own numbers. QWM reports silhouette alone. Applying that convention here would
have produced the conclusion that the encoder carries no phase information, while a probe shows 93
percent. Every latent-structure claim in this project therefore reports both.

---

## Slide 15. What the encoder does encode: the body, unmistakably

Morphology is decodable from e_t at 99.9 percent, with the shuffled control at 34.2 percent. Silhouette
by morphology is +0.0835 against -0.0222 for phase, so the body, not the behaviour, is the main axis of
the representation.

This is the expected outcome and it is not a failure. A 0.5x leg genuinely looks different from a 1.0x
leg, and an encoder blind to that difference would be a worse encoder. Removing morphology is the job
of the ITM and cross-augmentation downstream, not of the frozen encoder.

The number is recorded as the baseline that the learned latent must improve on.

---

## Slide 16. The result that changed the method

Phase decodes at 93 to 97 percent within a single body. Training the probe on two morphologies and
testing on the held-out third drops it to 27 to 39 percent.

The phase code does not transfer between bodies. Each morphology carries its own private encoding of
the same gait.

At this point there were two candidate explanations, and they call for opposite responses:

1. The encoder entangles phase with morphology, which is the problem the ITM exists to solve.
2. The label was wrong, and the measurement was never valid.

---

## Slide 17. Diagnosing the label

The phase label was defined as step index modulo 64, because the joint commands repeat exactly every
64 steps. Three problems with that definition surfaced on inspection.

The period of 64 is not a property of insect walking. It is the length of the segment the lab trimmed
out of the recorded animal data, and the loop is not closed: joint angles jump 14.75 degrees at the
seam. More seriously, identical commands do not place different bodies in the same pose. A short leg
reaches the ground earlier than a long leg given the same joint angle, so step 10 is not the same
moment of the gait on all three bodies.

The label was measuring the clock, not the body.

Replacement: 6-bit foot contact state, meaning which of the six feet are planted, taken from the
simulator's force sensors. This is ground truth about the body's actual configuration, and it is the
same family of measure the host lab uses in its own gait analysis.

---

## Slide 18. Re-measuring with a pose-based label

Same encoder, same frames, same probe. Only the definition of behaviour changed.

| Label | Within one body | Across bodies | Ratio |
|---|---|---|---|
| Step modulo 64 (time) | 92.5% | 38.4% | 0.42 |
| 6-bit foot contact (pose) | 85.1% | 55.2% | 0.65 |
| Number of feet planted | 70.0% | 29.6% | 0.42 |

Cross-morphology transfer rises 17 points from relabelling alone. Part of the original entanglement was
measurement error, and the encoder is better than the first result implied.

It is not the whole story, because 55 percent is not 100 percent. A real morphology-entangled residual
remains, and that residual is what the latent action model has to remove.

---

## Slide 19. Why the ceiling is below 100 percent

Contact patterns were compared directly across bodies at matched timesteps. Full agreement on all six
feet occurs in 36 percent of steps between long and medium, and 16 percent between long and short.
Per-leg agreement between long and short is 75 percent.

Even the ground truth disagrees across bodies under identical commands, because a short leg physically
plants its feet on a different schedule. Replaying one command sequence on every body therefore carries
its own ceiling, and any transfer number has to be read against that ceiling rather than against 100.

This is a limitation of the current reference gait, not of the representation, and it is the strongest
argument for moving to a per-morphology controller in the next phase.

---

## Slide 20. Current status against the plan

| Milestone | State |
|---|---|
| Morphology gap is real | Passed, non-overlapping distances across 5 episodes per body |
| Vision capture exists | Built: scripted camera, controlled floor, aligned recorder |
| Encoder carries behaviour signal | Passed, 85 to 93 percent probe accuracy with shuffle control at chance |
| Behaviour label is valid | Revised from time to foot contact, +17 points of transfer |
| Latent action model trained | Not started |
| Latent is morphology-agnostic | Not started |
| Transfer to unseen body | Not started |

---

## Slide 21. What the next phase has to beat

The targets are no longer qualitative. They are the numbers measured on the frozen encoder.

| Quantity | Frozen encoder e_t | Required of latent z_t |
|---|---|---|
| Cross-morphology behaviour transfer | 55.2% | Higher |
| Morphology decodability | 99.0% | Lower |
| Silhouette by morphology | +0.0835 | Lower |

The decisive comparison is an ablation, not a demonstration: a forward model conditioned on the learned
latent against a forward model conditioned on the raw 18-dimensional joint command through a shared
encoder. If raw commands match the latent, the latent bottleneck adds nothing in this setting and the
objection on Slide 8 stands. Running that comparison early is deliberate.

---

## Slide 22. Limitations stated in advance

The reference gait is one recording of one animal, replayed open loop on all three bodies. It is not a
controller tuned per morphology, and Slide 19 shows the ceiling that imposes.

Only forward walking is covered so far. The biological recordings for this species contain no turning
and no stopping, so those behaviours cannot be grounded in the animal data by any method and would have
to be synthesised.

The study is interpolation, not extrapolation: the held-out 0.75x body lies between the two trained
bodies. Extrapolation beyond the trained range is untested.

Everything is in simulation. The dynamics are chaotic across scene reloads, where a difference at
machine epsilon grows to 1.8 m of divergence over 200 steps, so all reported numbers are means over
repeated episodes rather than single runs.

The planned fallback if the latent still clusters by body is UniSkill, which targets cross-embodiment
skill representation directly. The earlier plan named HiLAM, which on reading solves temporal
abstraction rather than embodiment invariance and would inherit the clustering rather than remove it.

---

## Appendix slides, if asked

A1. Contact-based gait diagram for all three morphologies, with the loop seam marked.
A2. Scaling procedure for the leg-length variants and the numerical verification of reach ratios.
A3. Determinism analysis: bit-exact within a scene load, chaotic across reloads, and why episodes are
    collected by reloading.
A4. The 0.52 cosine similarity measurement that motivates single-frame encoding.
