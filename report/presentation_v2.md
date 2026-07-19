# Presentation v2
### Cross-Morphology Locomotion via Latent Action World Models

Framing note for the speaker. The deck argues that vision is *an additional route*, not that existing
methods are wrong. Existing work solves the problem when the new body can be described in advance.
The question here is what to do when it cannot. Everything after the setup follows the investigation
in the order it happened, including the two results that forced a change of method.

---

## Slide 1. Title

Cross-Morphology Locomotion via Latent Action World Models

If a new robot is close to one we already understand, must we start over?

Candidate: Disthorn Suttawet
University advisor: Mr. Bawornsak Sakulkueakulsuk (IFR, KMUTT)
Lab advisor: Prof. Poramate Manoonpong (Bio-inspired Robotics and Neural Engineering Lab, VISTEC)

---

## Slide 2. Background: locomotion control is fitted to one body

A legged locomotion controller is a mapping from robot state to joint commands. However that mapping
is produced, by hand tuning, by reinforcement learning, or by planning inside a learned world model,
it absorbs the geometry of the body it was produced on. Leg length determines how far a given joint
rotation moves a foot. Mass distribution determines what keeps the trunk upright. A contact schedule
that is stable on one body is not stable on another.

The mapping therefore does not survive a change of body. The failure is graded rather than total,
which is what makes it worth measuring rather than merely asserting. Slide 5 reports that measurement
for the three bodies used in this work: one controller, one bit-identical command sequence, three
clearly separated outcomes.

Nor is the shortfall a simple proportionality that could be undone by rescaling the command. Fitted
across the three bodies, walking speed falls with roughly the 0.65 power of leg length, not the 1.0
that naive geometric scaling would predict. Something in the controller has to be refitted.

---

## Slide 3. Problem: the cost, the existing remedies, and what they assume

The standard response is to retrain per body: hours to days of training for each variant, repeated
every time the morphology changes. For a laboratory that builds variants of one design, that cost
recurs indefinitely. The field calls this cross-embodiment generalization, and for legged robots it
remains open.

Two families of methods reduce that cost rather than paying it in full each time. One conditions a
single world model on morphology parameters read out of the robot's design file. The other holds a
policy backbone fixed across platforms and fits small per-robot heads from that robot's on-board
sensing. Slides 6 to 8 give the specific results; what matters at this point is what they have in
common.

Both assume the new body can be described or instrumented before it can benefit: an accurate design
record in the first case, on-board sensing with known joint conventions in the second. Each can be
supplied only by a party with access to the interior of the body or to its design record.

That assumption leaves a gap. Bodies that can be observed but cannot be opened up or documented fall
outside both routes. Animals cannot be fitted with joint encoders. A robot whose configuration has
drifted through repair, payload or wear is described by a file that no longer matches it. Hardware
acquired from another group often has no published kinematics at all.

**Research question.** Whether locomotion behaviour can be represented in a form recoverable from
external observation alone, and therefore transferable to a body that cannot be described in advance.

Speaker note, if asked how far this extends: the limiting case is an extinct animal, known only from a
skeleton and a set of trackways, where no specification exists and none can be produced. That is Ajan
Blink's example and it marks the point where specification-based methods run out entirely. This thesis
does not study animals or fossils. It studies simulated stick insects, because measuring whether a
representation is body-independent requires ground truth that only simulation supplies.

---

## Slide 4. How this project arrived here

The direction changed three times. Each change came from a specific finding, not a preference.

| Stage | Direction | What forced the change |
|---|---|---|
| Initial | Learn locomotion priors from real animal video | Real video has no action labels, so the latent action cannot be grounded |
| Pivot 1 | Simulation only, one species, three leg lengths | Advisor review, weeks 4 and 5: simulation supplies action labels for free |
| Pivot 2 | CoppeliaSim, not IsaacSim | The lab's *Medauroidea* model runs on CoppeliaSim; the earlier plan named the wrong simulator |
| Pivot 3 | Validation by linear probe and silhouette, not PCA | PCA answers a different question than the one being asked (Slide 17) |

Speaker note: the point of this slide is that the scope narrowed on purpose. Narrowing removed the
part of the problem that could not be measured.

---

## Slide 5. First, prove the problem exists

An experiment is only meaningful if the bodies genuinely behave differently. If the same command
produced the same motion on every body, there would be nothing to transfer and nothing to study.

Three *Medauroidea extradentata* variants in CoppeliaSim, identical in topology and in their
18-dimensional joint action space, differing only in leg length. All three received a bit-identical
command sequence.

Body position is read directly from the simulator each step as the world-frame position of the head,
so these are logged ground truth rather than anything inferred from joint angles. Episodes are 200
steps at the simulator's 20 Hz timestep, which is 10 seconds.

> ⚠️ **FIGURES PENDING REGENERATION.** The table below does not reproduce from any data in the repo.
> See `report/NUMBERS.md` section 1. The separation between bodies is real and holds by a wider margin
> than shown; only the values are unreliable. Do not present this slide until the re-run is done.

| Morphology | Mean speed | Net displacement, 5 episodes | Episode range |
|---|---|---|---|
| Long, 1.0x | 0.413 m/s | 4.125 m, sd 0.434 | 3.593 to 4.479 |
| Medium, 0.75x | 0.356 m/s | 3.562 m, sd 0.015 | tight |
| Short, 0.5x | 0.265 m/s | 2.646 m, sd 0.002 | tight |

The distributions do not overlap: the **worst** long-leg episode, 3.593 m, still exceeded the **best**
short-leg episode, 2.648 m. Worth stating precisely, because the commands are bit-identical across
every run, so there is no fast or slow condition here. The spread within a body comes from chaotic
divergence at scene reload, not from anything commanded.

Two honest qualifications. The long-leg spread is bimodal rather than noisy, landing on either 4.479
or 3.593, so its standard deviation is not a spread around a typical value and the two basins should
be reported as such. And the reported quantity is net straight-line displacement in the xy plane, not
path length, so any curvature in the walk is under-counted.

Identical input, different outcome, and the separation survives repeated episodes.

---

## Slide 6. The tool this work builds on: world models

A world model is a learned simulator. Rather than mapping observations straight to actions, the agent
first learns to predict what happens next, and then trains its policy against that prediction instead
of against the real environment.

DreamerV3 (Hafner et al., 2023) is the reference implementation. It encodes each observation into a
compact latent state, learns the transition between consecutive latent states, and then rolls the
policy forward entirely inside that latent space. The environment is not touched during policy
training. One algorithm with one hyperparameter set covers more than 150 domains.

Why this matters here, and it is the reason the thesis is built on a world model rather than a policy.

Training happens in latent space, so whatever representation the model learns is the thing the policy
actually consumes. If that representation can be made body-independent, everything trained on top of
it inherits that property. A policy learned directly on joint commands has no such handle.

It also decouples data collection from training. Rollouts inside the model cost no simulation time,
which is what makes adaptation on a small number of real episodes plausible at all.

And it separates the problem cleanly. A world model has to answer "given the current situation and
this action, what happens next." That question can be asked without committing to what an action *is*,
which is exactly the opening the next slide exploits.

What DreamerV3 does not give us: a model per domain, explicit native action labels throughout, and
nothing carrying from one body to another.

---

## Slide 7. The opening: actions do not have to be joint commands

If the world model only needs *some* action variable to condition on, that variable need not be the
robot's native command vector.

LAC-WM (Huang et al., ICML 2026) takes this literally. An inverse model watches two consecutive
observations and infers an abstract action z that explains the change between them. The world model is
then conditioned on z rather than on any robot's command format. A separate decoder maps z back to
whichever native command the current body uses.

The consequence is what makes it relevant. Because z is defined by *observed change* rather than by
motor format, one latent space covers several embodiments at once, and adding embodiments improves the
shared model instead of fragmenting it.

Map that onto the problem from Slide 3. The obstacle was that every existing remedy needs the new body
described or instrumented in advance. But an action inferred from observed change needs neither. It
needs only that the change is visible. That is the structural reason this architecture, and not a
morphology-conditioned one, is the starting point here.

The caveat, stated up front: LAC-WM was demonstrated on manipulation, not locomotion. Its embodiments
have genuinely disjoint action spaces, 10 dimensions for a Franka end effector, 20 for a bimanual
humanoid, 138 for human hand keypoints, and that disjointness is what motivates a shared latent there.
The three bodies in this work all share an identical 18-dimensional joint space, so that particular
motivation does not transfer and a different one has to be established. Slide 11 does that.

So the architecture is promising but unproven in this domain. The next slide asks what the locomotion
field has actually done.

---

## Slide 8. Meanwhile, what the locomotion field is actually doing

The latent-action idea is established in manipulation. Legged locomotion has pursued cross-morphology
transfer too, and it has produced stronger results than anything shown here so far. It is worth
seeing what route it took.

QWM (Danesh et al., 2026) transfers to unseen quadruped morphologies with frozen weights. It reads
limb lengths, mass, and torque limits out of each robot's CAD file and conditions the world model on
those numbers. **Input is proprioception.** This is the strongest zero-shot result in the area.
*What it needs:* an accurate machine-readable description of the new body, available in advance.

L3P (Zheng et al., 2025) shares a latent policy backbone across seven quadruped platforms, freezing it
and fine-tuning a small encoder and decoder per robot. **Input is proprioception and foot force.**
*What it needs:* the new robot instrumented, and its joint conventions known, so the per-robot heads
can be fitted.

Li et al. (RA-L 2021) plan in a learned latent action space on a hexapod and a quadruped, with each
robot trained separately. **Input is proprioception.**
*What it needs:* a full training run per body, since nothing is shared between them.

The pattern is the point. Every one of these reads the body from the inside. Vision has transformed
manipulation over the same period, and legged cross-morphology work has largely not taken it up. That
is the space this thesis is trying to occupy, and the next two slides argue why it is worth occupying
rather than merely unoccupied.

---

## Slide 9. What every current route has in common

The honest way to put this is not that these methods need something and this thesis needs nothing.
Every method here needs something. What differs is **where the requirement lands**.

| Method | Needs before it can help a new body | Requirement lives |
|---|---|---|
| DreamerV3 | A full training run for that body | On the body |
| QWM | A CAD or USD file with correct limb lengths and masses | On the body |
| L3P | Proprioception and foot force read out of that body, with its joint conventions known | On the body |
| Li et al. | A separate latent space and dynamics model per body | On the body |
| This work | A camera that can see it move | On the observer |

That distinction is the whole argument, so it is worth being precise about it.

A requirement that lives on the body has to be satisfied by whoever owns, built, or can open up that
body. You cannot fit encoders to a stick insect. You cannot recover the joint conventions of a robot
whose documentation was never written. You certainly cannot instrument an extinct animal.

A requirement that lives on the observer can be satisfied by anyone who can get a camera in front of
the subject. That is a weaker precondition, and weaker preconditions are what make a method apply in
more places.

To be clear about what this thesis still needs: an external camera, and, during pretraining, logged
joint commands from the bodies used to train the encoder and transition models. The claim is not that
observation is free. It is that observation is obtainable in cases where instrumentation is not.

These methods are not competitors here. QWM's zero-shot transfer is stronger than anything proposed
below wherever a correct CAD file exists. The question is what remains available when it does not.

---

## Slide 10. Why vision, when every locomotion result on the last slide used proprioception

This is the choice the whole thesis rests on, so it needs a real answer rather than a preference.

**Proprioception has no shared space across bodies; vision does.** A hexapod reports 18 joint angles, a
quadruped 12, in different orders and conventions. These are not the same vector space, so a model
consuming proprioception cannot share a single input representation across bodies. That is precisely why
L3P must fit a separate encoder and decoder per robot. A camera, by contrast, produces the same format
regardless of what stands in front of it. If the goal is one shared latent space covering many bodies,
vision is the only modality that is natively common. Stated honestly: the three bodies in this study do
share an identical 18-dimensional joint space, so this argument is about the general case the method is
aimed at, not about the specific test being run.

**Our own data shows the proprioceptive signal does not transfer.** Foot force predicts body velocity
well within a single body, R² = +0.926. Fitted on one body and applied to another, the same relationship
collapses, R² between -0.33 and -5.23. Negative R² means worse than predicting the mean. The mapping from
internal sensing to outcome is body-specific, and it inverts rather than merely degrading. This is the
strongest evidence available here, because it is measured on the actual bodies rather than argued.

**Proprioception reports the command; vision reports the consequence.** Joint angles describe what the
body did internally. They do not distinguish walking forward from slipping in place from falling over.
The task is defined in the world, and only an external view observes the world.

**Access.** Slide 9's argument, in one line: proprioception requires reaching inside the body, and for
animals, undocumented hardware, or anything that has drifted from its specification, that is unavailable
in principle.

**Vision is also the harder setting, which makes a positive result mean more.** Morphology decodes from
the raw visual features at 99.9 percent, silhouette +0.0835, against -0.0222 for behaviour. Body shape
dominates what the encoder sees. The joint commands here are bit-identical across all three bodies, so
proprioception carries almost no morphological signature at all. Vision maximises the confound the latent
action is supposed to remove, so demonstrating invariance here is a stronger claim than demonstrating it
where the confound is weak.

**What this is not.** It is not a claim that proprioception is inferior, and this work uses it. Logged
joint commands supply the action labels during pretraining, and foot-force readings supply the contact
labels used for evaluation from Slide 21 onward. The claim is narrower: proprioception cannot serve as
the *observation channel* through which behaviour is compared across bodies.

**Cost, stated plainly.** Pixels are noisier than joint encoders. Rendering style dominates the raw
features, which caused a real problem documented on Slide 15. The sim-to-real gap is wider in pixels than
in joint space. And the encoder is expensive to run. Slides 14 and 15 show what that cost looked like in
practice.

---

## Slide 11. The objection that decides this project

Raised in advisor review, week 4, and never yet answered:

> If the policy and the robot both need joint commands in the end, why convert to a latent space
> at all, and then convert back?

The objection sharpens against our own setting. LAC-WM needs a latent action because its embodiments
cannot share an action vector. Our three variants share an identical 18-dimensional command space.
On that reading, the motivation for a latent action disappears.

The answer is that the heterogeneity is not in the action space. It is in the dynamics. Slide 5 is
the evidence: the same 18 numbers produce 4.13 m of travel on one body and 2.65 m on another. The
latent action is not being asked to reconcile different action formats. It is being asked to describe
what the body *did* in a way that survives the body changing.

This reframing is testable, and Slide 24 gives the experiment that decides it.

---

## Slide 12. Pipeline

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
                                                     (not scored in Phase 1; weights kept for Phase 2)
```

| Block | Input | Output | Trained |
|---|---|---|---|
| V-JEPA2 encoder | RGB 256x256x3 | e_t, 256 x 1408 | No, frozen |
| Inverse Transition Model | [e_t, e_{t+1}], 512 tokens | z_t, 64 dimensions | Yes |
| Forward Transition Model | [e_t, z_t] | ê_{t+1}, 1408 | Yes |
| Motion Decoder | z_t queries e_t | â_t, 18 joint targets in radians | Yes; unused in Phase 1 scoring, weights kept |

L = L_recon + lambda * L_motion, with L_recon computed in embedding space, so no pixel decoder is
needed.

Correction against the previous version of this deck: the latent action is 64-dimensional, not 512.
Table 4 of LAC-WM reports 512 as the internal hidden width of the ITM and FTM. Section 4.2 states the
action embedding dimension separately, and it is 64.

---

## Slide 13. Visual encoder and one non-obvious constraint

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

## Slide 14. What we had to build before any of this could run

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

## Slide 15. Background choice is a measurement decision, not decoration

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

## Slide 16. Encoder check: is the behaviour signal present at all?

Before training anything, the frozen features must contain the signal the ITM is supposed to extract.
Otherwise the pipeline has nothing to work with.

Data: three morphologies, five episodes each, 200 steps, 3000 frames total, locked render environment.

Result. A linear probe recovers gait phase from e_t at 85.1 percent against a 12.5 percent chance
baseline, rising to 92.7 percent when whole episodes are held out. A shuffled-label control lands at
12.3 percent, which is chance, so the result is not the probe memorising 1408 dimensions from 1440
samples.

The behaviour signal is present. The gate passes.

---

## Slide 17. The same check, read a second way, says the opposite

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

## Slide 18. What the encoder does encode: the body, unmistakably

Morphology is decodable from e_t at 99.9 percent, with the shuffled control at 34.2 percent. Silhouette
by morphology is +0.0835 against -0.0222 for phase, so the body, not the behaviour, is the main axis of
the representation.

This is the expected outcome and it is not a failure. A 0.5x leg genuinely looks different from a 1.0x
leg, and an encoder blind to that difference would be a worse encoder. Removing morphology is the job
of the ITM and cross-augmentation downstream, not of the frozen encoder.

The number is recorded as the baseline that the learned latent must improve on.

---

## Slide 19. The result that changed the method

Phase decodes at 93 to 97 percent within a single body. Training the probe on two morphologies and
testing on the held-out third drops it to 27 to 39 percent.

The phase code does not transfer between bodies. Each morphology carries its own private encoding of
the same gait.

At this point there were two candidate explanations, and they call for opposite responses:

1. The encoder entangles phase with morphology, which is the problem the ITM exists to solve.
2. The label was wrong, and the measurement was never valid.

---

## Slide 20. Diagnosing the label

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

## Slide 21. Re-measuring with a pose-based label

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

## Slide 22. Why the ceiling is below 100 percent

Contact patterns were compared directly across bodies at matched timesteps. Full agreement on all six
feet occurs in 36 percent of steps between long and medium, and 16 percent between long and short.
Per-leg agreement between long and short is 75 percent.

Even the ground truth disagrees across bodies under identical commands, because a short leg physically
plants its feet on a different schedule. Replaying one command sequence on every body therefore carries
its own ceiling, and any transfer number has to be read against that ceiling rather than against 100.

This is a limitation of the current reference gait, not of the representation, and it is the strongest
argument for moving to a per-morphology controller in the next phase.

---

## Slide 23. Current status against the plan

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

## Slide 24. What the next phase has to beat

The targets are no longer qualitative. They are the numbers measured on the frozen encoder.

| Quantity | Frozen encoder e_t | Required of latent z_t |
|---|---|---|
| Cross-morphology behaviour transfer | 55.2% | Higher |
| Morphology decodability | 99.0% | Lower |
| Silhouette by morphology | +0.0835 | Lower |

The decisive comparison is an ablation, not a demonstration: a forward model conditioned on the learned
latent against a forward model conditioned on the raw 18-dimensional joint command through a shared
encoder. If raw commands match the latent, the latent bottleneck adds nothing in this setting and the
objection on Slide 11 stands. Running that comparison early is deliberate.

---

## Slide 25. Limitations stated in advance

The reference gait is one recording of one animal, replayed open loop on all three bodies. It is not a
controller tuned per morphology, and Slide 22 shows the ceiling that imposes.

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
