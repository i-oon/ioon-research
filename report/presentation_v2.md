# Presentation — Cross-Morphology Locomotion via Visual Latent Action World Models
### Ideal / corrected deck (25 slides). Proposal, ~25 min, mixed committee (some new to the project).

Conventions used here:
- **Section label** (declarative) + **sub-headline** (a question when we are investigating, a statement when we are reporting a result).
- `[FIG: name]` marks which generated figure goes on the slide.
- `Speaker:` lines are for you, not the slide.
- Numbers are the regenerated / reproducible values (`report/NUMBERS.md`, `exp_*.md`). Report ranges, not false-precision decimals.

Section order (fixed so Literature Review is contiguous and Problem Formulation is not sandwiched inside it):
Background (2–4) → Literature Review (5–9) → Problem Formulation (10) → Research Question (11) → Method (12–16) → Evaluation (17) → Preliminary Results (18–20) → Final Protocol (21) → Decisive Ablation (22) → Outcomes (23) → Scope (24) → Contributions (25).

---

## Slide 1 — Title

**Cross-Morphology Locomotion via Visual Latent Action World Models**

Disthorn Suttawet
6-month Work Integrated Learning internship, Vidyasirimedhi Institute of Science and Technology (VISTEC)
University advisor: Mr. Bawornsak Sakulkueakulsuk (FIBO, KMUTT)
Lab advisor: Prof. Poramate Manoonpong (Bio-inspired Robotics and Neural Engineering Lab, VISTEC)

---

## Slide 2 — Background: A locomotion controller is fitted to one body

A locomotion controller maps the robot's current state to a motor command: **π : sₜ → aₜ**.
For the simulated stick insect, **aₜ ∈ ℝ¹⁸** (6 legs × 3 joints).

- Legged controllers are written in robot-specific variables: joint positions, torques, motor velocities, contact measurements. These work while the body is known and unchanged.
- Even when morphology is not given as an input, it is **baked into the controller through training**. The controller has learned not "how to walk" but "how *this* body walks."
- **When the body changes, the same numerical command can produce a different physical outcome.**

Bodies do not stay fixed: leg dimensions change between prototypes, payload and repair alter the dynamics, and one controller may need to run across related platforms.

The field calls this **cross-morphology locomotion transfer** (or cross-embodiment generalization):
**sₜ →(π) aₜ →(physical body, morphology m) sₜ₊₁**

Speaker: this is the setup slide. One sentence to land: the command is not the behaviour; the body sits between them.

---

## Slide 3 — Background: Does morphology alone change the outcome? (Step −1)

**Question:** does changing leg length alone change locomotion when the motor command is held constant?

**H₀ (null): the locomotion outcomes are equivalent across morphologies.** If H₀ cannot be rejected, cross-morphology transfer is not yet a real problem in this setting, and nothing after this slide is worth building.

Setup: 3 *Medauroidea extradentata* variants in CoppeliaSim, identical topology and identical 18-D joint action space, differing only in leg length. All three receive a **bit-identical** command sequence. Body position is read directly from the simulator (ground truth, not inferred). Two distance measures are reported so the open-loop gait's side-to-side oscillation cannot be misread as a locomotion difference.

![Step -1: morphology gap](fig_step_minus1.png)
*Trajectories at true aspect ratio + per-episode dots with mean±sd.*

| Morphology | Path length (n=5) | Net displacement |
|---|---|---|
| Long, 1.0× | 5.217 ± 0.069 m | 4.404 ± 0.187 m |
| Medium, 0.75× | 4.149 ± 0.019 m | 3.569 ± 0.010 m |
| Short, 0.5× | 3.228 ± 0.011 m | 2.729 ± 0.011 m |

**Result — H₀ is rejected for all three pairs:** Mann-Whitney p = 0.0079, **Cliff's δ = 1.00** (every episode of the longer body exceeds every episode of the shorter one; distributions do not overlap). p = 0.0079 is the smallest value the test can return at n = 5, i.e. the strongest statement this sample size allows.

The gap is also not a rescaling: walking speed falls with roughly the **0.69 power** of leg length, not 1.0.

Speaker: the punchline is the stats line, not the table. "Same command, different outcome, and it does not overlap."

---

## Slide 4 — Background: We need a middle language of locomotion change

The action space is already shared (all three are 18-D), yet the controller does not transfer. So what should carry across bodies?

![Same angle, different reach](fig_ik_intuition.png)
*Use the LEFT panel here: same joint angles (−60°, −40°) put the foot 0.94 apart on long vs short. Same motor command ≠ same physical consequence. (The right panel is used on Slide 20.)*

- **Joint command** is too body-specific: `aₜ = [q₁ᵗᵃʳᵍᵉᵗ, …, q₁₈ᵗᵃʳᵍᵉᵗ]ᵀ`. Different bodies need different commands to realise the same behaviour.
- **A task label** (walk / turn) is too coarse to condition a dynamics model.

We want something **between** the two — an *observable locomotion event*: entering swing, establishing contact, transferring support, moving forward. Four properties we will **require and test** (design requirements, not established facts):

| Requirement | Meaning |
|---|---|
| Behavioural grounding | describes an observable locomotion change |
| Predictive sufficiency | with the current state, helps predict what happens next |
| Morphology robustness | similar behaviours stay comparable across bodies |
| Executability | retains enough information to recover body-specific motor commands |

Speaker: this frames the whole thesis — behaviour-level, not command-level, not task-level.

---

## Slide 5 — Literature Review: Existing methods transfer by supplying information about the new body

| Strategy | Example | Main idea | Body-specific info required | Limitation |
|---|---|---|---|---|
| Per-body retraining | DreamerV3 (Hafner et al., 2023) | learn a new controller per body | new interactions, rewards, optimisation | training cost repeats per body |
| Morphology conditioning | QWM (Danesh et al., 2026) | condition policy/world-model on physical parameters | leg lengths, masses, torque limits, CAD/URDF/USD | assumes morphology is accurately known |
| Online system identification | — | infer hidden dynamics from recent history | proprioception, past commands, observed responses | needs the robot's internal signals + interaction |
| Shared policy + per-robot adapters | L3P (Zheng et al., 2025) | shared backbone, per-robot encoder/decoder | sensor definitions, joint ordering, actuator conventions | a new adapter must still be fitted per platform |

**Common structure:** all obtain body information through at least one of three channels — **Design specification ∨ Internal sensing ∨ New interaction**.

**Shared assumption:** *the learner has access to an internal description of the body.*

---

## Slide 6 — Literature Review: Research gap — the body is observable but not described

Prior strategies work when at least one is available: an accurate geometric/dynamic model, known joint/actuator conventions, proprioceptive/force measurements, permission to collect new interaction data, or time to retrain.

**Research gap:** *Can visual transitions provide a shared, behaviour-level representation across morphology changes — and how does it compare with proprioception?*

Why this gap is real:
- Even for a lab robot, CAD/URDF/motor specs are an approximation. The *documented* body is what we believe it has; the *physical* body is what actually moves. QWM concedes this by routing "unmodeled real-world residuals" through a separate latent.
- Motivating cases: animals, and undocumented / repaired / payload-changed hardware whose real parameters no longer match their design files.
- We may observe how a system walks while having no access to its joint encoders, actuator commands, or exact dynamics.
- Across substantially different bodies, proprioceptive vectors differ in dimension, ordering, units, and semantics. **External vision offers a common observation format.**

*If the body cannot be described directly, perhaps its dynamics can be inferred from how its observable state changes over time.*

---

## Slide 7 — Literature Review: Why test vision when proprioception is available?

This thesis does **not** assume vision is superior. It tests whether vision can recover a transferable behaviour representation, and compares it against proprioception.

Most locomotion systems use proprioception (joint angles, joint velocities, IMU, foot forces, motor states). These are powerful but **body-internal and convention-specific** — proprioception reports the internal command/state.

Vision is different — it reports the **external consequence**: same input format on any body, external, needs no joint conventions or CAD, captures what happened in the world.

![Different internal interfaces, common external format](fig_obs_format.png)
*Body A / Body B: **different internal interfaces → common external format** (plain-text vectors with index meanings, then identical H×W×3).*

But vision is also the **harder** setting, and we say so up front: a camera sees morphology very clearly. In the current pilot, raw V-JEPA2 features decode morphology at **around 100%**.

*Can the latent action model extract behaviour-relevant change from visual features that strongly contain body shape?*

Speaker note (only if asked "why not just remap the indices?"): remapping needs documentation of both bodies — exactly what the access argument on Slide 6 assumes is missing.

---

## Slide 8 — Literature Review: Why a world model?

A world model is a learned simulator. The feature that matters here is that **it must predict consequences**.

It learns how actions produce transitions by approximating the environment's dynamics: **ŝₜ₊₁ = F_θ(sₜ, aₜ)**.
For visual input, images become compact features first: **eₜ = E(oₜ)**, and the transition model predicts **êₜ₊₁ = F_θ(eₜ, aₜ)**, trained by **L_pred = ‖êₜ₊₁ − eₜ₊₁‖²**.

We use a world model not mainly to imagine rollouts, but because **transition prediction is an objective test** of whether an action representation captures meaningful locomotion change.

`[FIG: DreamerV3 paper figure]` — Mastering Diverse Domains through World Models (Hafner et al., 2023). One algorithm, one hyperparameter set, 150+ domains.
*Limitation for us:* a model per domain, explicit action labels throughout, nothing carries across bodies — and it conditions on the robot's native command aₜ, whose physical meaning changes with morphology.

---

## Slide 9 — Literature Review: What should the world model treat as an action?

Standard formulation: **(eₜ, aₜ) → eₜ₊₁**. But across morphologies, **aₜ = same numerical command → eₜ₊₁ − eₜ = different physical change**.

So instead of assuming the native command is the shared action, we ask whether an intermediate variable can be inferred from the observed transition: **(eₜ, eₜ₊₁) → zₜ**.

`[FIG: LAC-WM paper figure]` — **LAC-WM (Huang et al., ICML 2026)** discards explicit action labels as the conditioning signal. An inverse model infers an abstract action z from consecutive observations; the world model is conditioned on z rather than any robot's native command. One latent space then covers several embodiments, and adding embodiments improves it rather than fragmenting it.

*(Citation verified: Huang Huang et al., "Cross-Embodiment Robot Foundation World Models with Latent Actions" (LAC-WM), ICML 2026.)*

- **LAC-WM's setting:** different robots have genuinely different action *formats*.
- **This project's setting:** robots share the same 18-D action format, **but the same 18-D command has a different dynamics meaning per body.** That is the gap we adapt the idea to.

---

## Slide 10 — Problem Formulation: External observation shows the consequences of hidden body dynamics

An observer records a sequence **o₁, o₂, …, o_T**, where oₜ is an external observation of the body at time t.

A single observation reveals body shape, limb configuration, approximate pose, foot locations, surrounding terrain. It does **not** give exact link lengths, masses, actuator limits, or internal sensor readings — but it gives the **visible consequences** of those hidden properties.

A single image says what the system looks like. A **transition** between images says how it behaves: **Oₜ, aₜ → Oₜ₊₁** reveals which limbs move, which feet enter/leave contact, how the body responds to support, whether it progresses or slips, how balance changes.

The learning problem becomes **sₜ₊₁ = f_m(sₜ, aₜ)**: the morphology m may be unknown but it shapes the observable transition.
*This changes the problem from body description to transition prediction.*

---

## Slide 11 — Research Question: Can observable locomotion change become a shared action representation?

**Primary question.** Can a latent action inferred from visual state transitions preserve behaviour-relevant locomotion information while reducing morphology-specific information across different leg lengths?

**Secondary question.** How does the visual latent representation compare with raw visual features, native joint commands, and proprioceptive observations?

| Testable hypothesis | Statement |
|---|---|
| **H1 — Behavioural information** | Foot-contact / support-transition is decodable from zₜ (behaviour is preserved). |
| **H2 — Reduced morphology dependence** | Morphology is **less** recoverable from zₜ than from the raw visual features eₜ. |
| **H3 — Predictive sufficiency** | (eₜ, zₜ) predicts eₜ₊₁ (the latent action explains the visual transition). |
| **H4 — Added value over native commands** | A latent-conditioned dynamics model is compared against one conditioned on the 18-D command. If the native-command model does equally well across morphologies, the latent action may be unnecessary in this controlled setting. |

**\*Success is not "vision wins."** The intended result is a scientific comparison: which information source produces the most transferable behaviour representation, under which assumptions, and at what cost.

---

## Slide 12 — Method: Data source and pre-processing

`[FIG: pipeline diagram, stages 1–2 highlighted, 3–6 greyed]`

- **Simulator (CoppeliaSim 4.10) → camera sensor → RGB frame 256×256×3.**
- Consecutive frames fₜ, fₜ₊₁ → **frozen V-JEPA2 RGB tokenizer** → per-frame embeddings **eₜ, eₜ₊₁ ∈ ℝ²⁵⁶ˣ¹⁴⁰⁸** (256 patch tokens × 1408).
- In parallel, a **joint logger** records the logged joint-position target **aₜ ∈ ℝ¹⁸** that caused the transition.
- Each RGB frame and joint command are recorded in the **same simulation step**, so aₜ corresponds to the observed transition oₜ → oₜ₊₁.

Speaker: this is the "where the data comes from" slide; the trainable part is greyed and expanded on Slide 14.

---

## Slide 13 — Method: Frozen front-end (V-JEPA2)

V-JEPA2 is self-supervised on ~1 million hours of video with an objective that predicts masked content in **representation space**, which rewards motion-relevant features — a reasonable starting point for gait.

We adopt V-JEPA2's frozen RGB tokenizer as the visual encoder, extracting per-frame **eₜ ∈ ℝ²⁵⁶ˣ¹⁴⁰⁸** with **no fine-tuning**. These feed the ITM and FTM, which learn the shared latent action zₜ.

`[FIG: V-JEPA2 encoder figure + patch pipeline]`
frameₜ ∈ ℝ²⁵⁶ˣ²⁵⁶ˣ³ → **patch split** (256×256 ÷ 16 = 16×16 grid → 256 patches) → **positional embedding** → **ViT transformer (frozen)** → **eₜ ∈ ℝ²⁵⁶ˣ¹⁴⁰⁸**. Weights come from 1M hours of video + 1M images.

Whether these frozen features actually contain locomotion information is tested empirically (Slide 18), not assumed.

---

## Slide 14 — Method: How zₜ is learned

`[FIG: full pipeline diagram, stages 3–6 visible]`

- **ITM (Inverse Transition Model)** = infer zₜ from the observed change eₜ → eₜ₊₁.
- **FTM (Forward Transition Model)** = test whether zₜ explains the next visual state.
- **Motion Decoder (MD)** = keep zₜ connected to executable joint commands.

Signals: ITM updates from **both** objectives through zₜ (z takes gradient both ways). Reconstruction loss **L_recon = ‖êₜ₊₁ − eₜ₊₁‖²** is computed in embedding space, so **no pixel decoder is needed**. Motion loss **L_motion = ‖âₜ − aₜ‖²** uses the logged joint targets.

**Keep the Motion Decoder's weights after pretraining** — it is the module that maps a latent action back to a body-specific joint command, so it carries the "executability" property (Slide 4) and is the bridge to any later control use. (It is *not* discarded.)

---

## Slide 15 — Method: Proposed trainable modules

**ITM: (eₜ, eₜ₊₁) → zₜ.** Compresses the observed transition into a latent action explaining what changed.
Input: [eₜ, eₜ₊₁] (512 tokens × 1408) + a query. ×4 causal self-attention (eₜ₊₁ absorbs context of eₜ), ×N cross-attention (query focuses on moving patches). **Output: zₜ ∈ ℝ⁶⁴.**

**FTM: (eₜ, zₜ) → êₜ₊₁.** Tests whether current state + zₜ are sufficient to explain the next visual state.
Input: eₜ (256×1408) + zₜ (64). ×8 transformer blocks: self-attn(eₜ) → self-attn(zₜ) → cross-attn(eₜ→zₜ). Output: êₜ₊₁ (256×1408). **L_recon = ‖êₜ₊₁ − eₜ₊₁‖².**

**Motion Decoder: (eₜ, zₜ) → âₜ.** Requires zₜ to retain information tied to executable, body-specific commands.
Input: zₜ (64) + eₜ (256×1408). cross-attn(zₜ→eₜ) + MLP. Output: âₜ ∈ ℝ¹⁸. **L_motion = ‖âₜ − aₜ‖².**

**L_total = L_recon + λ · L_motion.** ITM receives training signal from both objectives through zₜ.
(λ is not published in LAC-WM; start equal and ablate. z_t = 64 per LAC-WM §4.2 "action embedding dimension of 64" — the 512 in Table 4 is the hidden width.)

---

## Slide 16 — Method: What prevents a shortcut?

Problem: without a safeguard, zₜ can become a **compressed copy of the next state** rather than a representation of the action between states (the ITM smuggles eₜ₊₁ into zₜ).

**Fix — cross-augmentation.** Apply two independently sampled augmentations A and B to the frame pair before encoding. Use the **same** augmentation parameters for fₜ and fₜ₊₁ within a branch (temporal change preserved), but A and B are sampled independently.

`[FIG: cross-augmentation diagram]`
- ITM sees pair **A**: zₜ = ITM(eₜᴬ, eₜ₊₁ᴬ).
- FTM starts from pair **B** and is scored against pair **B**: êₜ₊₁ = FTM(eₜᴮ, zₜ), **L_recon = ‖êₜ₊₁ − eₜ₊₁ᴮ‖²**.

Because the ITM's view of t+1 (aug A) is not what the FTM is scored against (aug B), copying exact pixels no longer helps — zₜ must capture the abstract action. (Both augmentations use the same one frozen encoder, applied per frame; there is one encoder, not several.)

---

## Slide 17 — Evaluation: What counts as success?

| Requirement | Measurement | Success criterion |
|---|---|---|
| Behaviour preservation | foot-contact macro-F1 (balanced) | zₜ transfers better than raw features eₜ |
| Morphology robustness | morphology probe accuracy; silhouette as support | morphology is **less** decodable from zₜ than from eₜ |
| Predictive value | next-embedding error on held-out medium | F(eₜ, zₜ) transfers better / degrades less than F(eₜ, aₜ) |
| Executability | Motion Decoder reconstruction MAE | MD(eₜ, zₜ) outperforms MD(eₜ, 0) |
| Adaptation efficiency | medium-leg learning curve | pretrained model reaches target error with fewer episodes than from scratch |

**Two conditions must hold together:**
**BehaviorTransfer(zₜ) > BehaviorTransfer(eₜ)  AND  MorphologyDecode(zₜ) < MorphologyDecode(eₜ).**
Reducing morphology alone is not enough: a collapsed representation would also make morphology hard to decode. Behaviour must survive at the same time.

---

## Slide 18 — Preliminary Check: Frozen V-JEPA2 features encode morphology (and behaviour, entangled)

![Morphology encoding evidence](fig_morphology_evidence.png)

![Behaviour encoding: sanity check](fig_sanity_check.png)

Shared setup: input = frozen V-JEPA2 eₜ (1408-d/frame, mean-pooled over 256 patches, encoder never trained); model = logistic-regression linear probe + standardisation; data = 3 bodies × 5 episodes × 200 steps = 3000 frames, same bit-identical command sequence; 5-fold CV.

**Morphology encoding** (target = which body, chance 33%):
- (a) supervised probe → **~100%**, holds under episode-grouped CV (not frame memorisation).
- (b) unsupervised PCA → PC1 orders short < medium < long **with no labels** (self-organises).
- (c) UMAP → illustration only, not evidence.
- Reads as: **body identity is a dominant axis of eₜ — the baseline zₜ must reduce.**

**Behaviour encoding** (target = which feet planted, macro-F1, chance 0.125):
- within-body **0.84**, cross-body **0.16**, shuffle **≈ chance** (0.84 is real, not an artefact).
- Reads as: **eₜ encodes behaviour but entangled with body shape** (wider leg-length gap → worse transfer; L↔S worst). The signal zₜ must **keep** while making it body-independent.

**Pilot caveat (one line, keep on the slide):** single session per body (morphology confounded with session), wave gait (not tripod), contact threshold 0.5 N, top-8 patterns = 43% of frames, within-body 0.83 ± 0.18. Numbers are preliminary and will shift after re-collection.

---

## Slide 19 — Preliminary Check: Identical commands, different physical states

![Same command, three bodies, different states](fig_same_command.png)

Type: direct measurement (no model) — reads the simulator state at one timestep.
Controlled: the 18-D joint command qₜ, **bit-identical across all 3 bodies** (max pairwise difference 0.000000).
Measured: per-foot contact force and the rendered RGB frame. Varied: only leg length.

**Same command → different physical state:** at this step the left-middle foot carries 5.7 N (long) and 9.3 N (short) but only 0.3 N (medium) — airborne on the medium body while planted on the other two.

**The command cannot tell the bodies apart** — it is identical, yet the outcome is not. This is why the pilot's shared aₜ makes the latent action vacuous (nothing to retarget, the Motion Decoder has no reason to condition on the body), and why **per-body commands (IK retargeting) are needed before training**.

---

## Slide 20 — Preliminary Check: Better expert data — shared targets, body-specific commands

![IK retargeting: same foot target, different angles](fig_ik_intuition.png)
*Use the RIGHT panel here: same foot target → different per-body angles.*

Shared task-space target → body-specific joint commands.
Foot position depends on angle **and** link length. Fixing the *angles* does not fix the *behaviour*; fixing the *foot target* does — and the per-body angle difference (long (−12.3°, −124.3°) vs short (−53.6°, −41.8°) for the same target) is exactly the body-specific command the Motion Decoder must learn to produce.

Where the shared target comes from:
- **Old CSV (`ds_loopsm`)** = joint angles + contact only → foot trajectory must be derived via forward kinematics; one animal, one cycle.
- **New 66k CSV (`expert_66k_aug3c_fcontact`)** = foot trajectories logged directly → ready-made IK targets, plus the simulator's own binary contact labels.

Speaker note (pre-empt "isn't MD just learning IK?"): the claim is that the model recovers this body-specific retargeting **from observation alone, without being given the kinematics** — not that IK is unknown.

---

## Slide 21 — Final Experimental Protocol: train on two bodies, test on one

**Execution plan.**
Platform: CoppeliaSim. Robot: *Medauroidea extradentata*, 3 morphologies (short / medium / long), 6 legs × 3 joints. Native action: **task-space foot trajectories** (→ IK per body). Data policy: **IK-retargeted expert dataset**. Behaviours: walk (may extend to turn / stop — these come for free from IK as extra foot trajectories; watch turn-vs-drift and a possibly-trivial stop). Train on **short + long**; hold out **medium** as the interpolation transfer test (extrapolation is a future direction).

Camera control: script-created camera identical across morphologies, third-person, ~40° elevation, ~45° FOV, matte lightly-textured floor, fixed relative offset following the body, no empty-background pixels.

**Validation logic (each step gates the next):**
- ✅ Step 0 — Does a morphology gap exist? (Slide 3)
- ✅ Step 1 — Does frozen vision contain locomotion information? (Slide 18)
- ⬜ Step 2 — Does zₜ preserve behaviour while reducing morphology information?
- ⬜ Step 3 — Does zₜ improve held-out transfer over raw commands? (Slide 22)

Recorded per step: RGB, proprioception, command, contact, next RGB.

*(This slide replaces the earlier "Experimental Design / TBD" draft — there is now one protocol, not two.)*

---

## Slide 22 — Decisive Ablation: does the latent action add value beyond raw joint commands?

Compare three transition models under the **same encoder, FTM capacity, dataset, and training budget**:

| Transition model | Conditioning input | What it tests |
|---|---|---|
| Observation only | F(eₜ, 0) | how predictable is motion with no action? |
| Raw action | F(eₜ, aₜ) | standard body-specific command representation |
| Latent action | F(eₜ, zₜ) | proposed transition-based representation |

**Primary comparison:** **L_pred^medium(eₜ, zₜ) < L_pred^medium(eₜ, aₜ)** on the held-out medium body.

The latent representation is useful **only if** it improves held-out transition prediction (or adaptation) relative to raw commands. If the raw-command model matches it across morphologies, the latent action is not justified in this controlled setting — and we would report that. This ablation *is* the thesis, so it is run early, not last.

---

## Slide 23 — Possible Outcomes: what would the results mean?

| Behaviour in zₜ | Morphology in zₜ | Prediction / transfer | Interpretation |
|---|---|---|---|
| Preserved | Reduced | Improved | **Intended result** — a morphology-invariant behaviour representation. |
| Preserved | Still high | Improved | Useful representation, but it did not remove body shape — partial success. |
| Lost | Reduced | Poor | Representation collapse — morphology dropped by destroying the signal. |
| Preserved | Reduced | No improvement | Clean latent space, but no measurable transfer benefit over eₜ. |
| Preserved | Reduced | Worse than aₜ | Raw commands remain the more useful action in this controlled setting. |

**Success requires all three simultaneously: behaviour preserved + morphology reduced + transfer improved.** Any single one alone is not enough — this is what makes the test decisive rather than self-confirming.

---

## Slide 24 — Scope and Limitations

**Included:** simulation-based hexapod locomotion; three leg-length morphologies; fixed third-person RGB camera; forward locomotion and foot-contact behaviour; medium-leg interpolation test; ITM, FTM, Motion Decoder, and frozen V-JEPA2 encoder.

**Not claimed:** real-robot or animal transfer; extrapolation beyond the training morphology range; camera-viewpoint invariance; generalization to manipulation or complex terrain; fully autonomous control through zₜ.

**Honest caveat:** IK creates comparable task-space objectives, but it does not guarantee identical contact dynamics or identical behaviour across morphologies. The pilot numbers on Slides 18–19 are diagnostic (single session per body → morphology confounded with session; wave gait; top-8 contact = 43% coverage) and will be regenerated on the IK dataset.

---

## Slide 25 — Contributions and Milestones

**Expected contributions.** A controlled benchmark for locomotion representation across changing body morphology. A visual latent-action world model combining ITM, FTM, and command reconstruction. An evaluation framework separating behaviour preservation, morphology leakage, predictive sufficiency, executability, and adaptation efficiency. Evidence establishing whether visual latent actions improve held-out morphology transfer over raw joint commands.

**Objectives.** Design and train the latent-action pipeline (ITM/FTM/MD) on simulation video with auto-logged action labels, stick insect × 3 morphologies. Test whether the learned latent action is morphology-agnostic (PCA / probe on zₜ across bodies performing the same behaviour). Test transfer to the unseen medium leg, and **measure whether the resulting world model reduces the data needed for a new morphology.**

| Stage | Completion criterion |
|---|---|
| Dataset redesign | valid IK gait, synchronised observations, distinct per-body commands |
| Representation training | stable ITM–FTM–MD optimisation without collapse |
| Representation evaluation | behaviour retained while morphology/session signal decreases |
| Transfer evaluation | held-out medium results compared with required baselines |
| Thesis conclusion | determine where latent action helps, fails, or needs stronger supervision |

The contribution is **not** an assumption that latent action must work — it is a controlled test of whether transition-based visual representations provide measurable cross-morphology value.
