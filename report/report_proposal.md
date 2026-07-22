# PROPOSAL DRAFT: written sections
### Cross-Morphology Locomotion via Visual Latent Action World Models

> This file contains the written content for the sections your advisor asked you to write now
> (Abstract TH/EN, Ch.1, Ch.2, Ch.3, References). Section numbers match the template so you can
> copy each block straight into Word. Sections marked "ยังไม่ต้องเขียน" are omitted.
>
> Two facts baked into the text (from our decisions):
> - **University advisor**: Mr. Bawornsak Sakulkueakulsuk (KMUTT / IFR).
>   **Lab advisor (internship)**: Prof. Poramate Manoonpong (VISTEC).
> - We are **vision-grounded but NOT vision-only**: the latent action is grounded on auto-logged
>   joint commands via a motion-decoding loss (inherited from LAC-WM). The proposal states this honestly.
> - `[[DECIDE]]` marks a spot where you must choose/confirm before final submission.

---

# 3. บทคัดย่อ (ภาษาไทย)

การพัฒนาหุ่นยนต์เดินด้วยการเรียนรู้แบบเสริมกำลัง (Reinforcement Learning) ในปัจจุบันมีข้อจำกัดสำคัญคือ นโยบายควบคุม
(policy) ที่ฝึกได้จะผูกติดกับสัณฐานวิทยา (morphology) ของหุ่นยนต์ตัวนั้นโดยตรง เมื่อรูปร่างของหุ่นยนต์เปลี่ยนไป เช่น
ความยาวขาเปลี่ยน นโยบายเดิมจะใช้ไม่ได้และต้องฝึกใหม่ทั้งหมด ซึ่งใช้เวลาและทรัพยากรสูง งานวิจัยนี้จึงศึกษาการเรียนรู้
"การกระทำแฝงที่ไม่ขึ้นกับสัณฐานวิทยา" (morphology-agnostic latent action) จากวิดีโอในสภาพแวดล้อมจำลอง โดยประยุกต์
แนวคิดของแบบจำลองโลกเชิงการกระทำแฝง (Latent Action World Model) ซึ่งเดิมใช้ในงานหยิบจับวัตถุ (manipulation)
มาสู่โดเมนการเคลื่อนที่ (locomotion) เป็นครั้งแรก

งานวิจัยใช้แมลงกิ่งไม้ (stick insect, *Medauroidea extradentata*) ในโปรแกรมจำลอง CoppeliaSim จำนวนสามสัณฐาน
ที่มีความยาวขาต่างกัน (1.0, 0.75 และ 0.5 เท่า) โดยใช้ตัวเข้ารหัสภาพ V-JEPA2 แบบตรึงค่า (frozen) ร่วมกับแบบจำลอง
การเปลี่ยนผ่านเชิงผกผัน (Inverse Transition Model) และเชิงไปข้างหน้า (Forward Transition Model) เพื่อสกัดการกระทำ
แฝง z_t วัตถุประสงค์หลักคือเพื่อพิสูจน์ว่า z_t เกาะกลุ่มตาม "พฤติกรรม" (เช่น การเดิน) แทนที่จะเกาะกลุ่มตาม "รูปร่างขา"
และเพื่อแสดงว่าแบบจำลองโลกที่ผ่านการฝึกล่วงหน้าช่วยให้สัณฐานที่ไม่เคยเห็นมาก่อนเรียนรู้ได้เร็วขึ้นโดยใช้ข้อมูลน้อยลง

ความแตกต่างสำคัญจากงานก่อนหน้าคือ แบบจำลองรับรู้สภาพหุ่นยนต์ผ่าน "ภาพ" เท่านั้น และ **ไม่เคยถูกบอกข้อมูล
ความยาวขา** (spec-free / implicit) ต่างจากงานที่ต้องอ่านข้อมูลโครงสร้างจากไฟล์ CAD ทั้งนี้การฝึกใช้ป้ายกำกับการกระทำ
(joint command) ที่บันทึกอัตโนมัติจากตัวจำลองเพื่อยึดโยงความหมายของ z_t โดยไม่ใช้ป้ายกำกับดังกล่าวในขั้นใช้งาน

*(ผลการทดลองจะเพิ่มเติมเมื่อดำเนินการเสร็จ)*

**คำสำคัญ:** การเคลื่อนที่ข้ามสัณฐาน / การกระทำแฝง / แบบจำลองโลก / การเรียนรู้เชิงถ่ายโอน / V-JEPA2

---

# 4. Abstract (English)

Reinforcement-learning controllers for legged locomotion have a fundamental limitation: a trained policy is
tightly coupled to the specific body (morphology) it was trained on. When the body changes, for example when
leg length changes, the policy becomes unusable and must be retrained from scratch, at high cost in time and
compute. This research studies the learning of a **morphology-agnostic latent action** from simulation video,
adapting the Latent Action World Model paradigm, previously demonstrated only for robotic manipulation, to the
domain of **legged locomotion** for the first time.

The study uses a simulated stick insect (*Medauroidea extradentata*) in CoppeliaSim, instantiated as three
morphologies differing only in leg length (1.0×, 0.75×, and 0.5×). A frozen V-JEPA2 visual encoder is combined
with an Inverse Transition Model and a Forward Transition Model to extract a latent action z_t. The primary
objectives are (i) to demonstrate that z_t organizes by *behaviour* (e.g. walking) rather than by *body shape*,
and (ii) to show that a pretrained world model enables an unseen morphology to be learned with substantially
fewer episodes than training from scratch.

The distinction from prior work lies in what the model is allowed to know about the body. The model is **never
told its leg length**, in contrast to methods that read structural parameters out of CAD files and condition on
them explicitly. This is a condition of the experiment rather than an implementation preference: supplying the
morphology would answer a different question, namely whether a world model can compensate for a body it has
already been described, instead of whether that abstraction can be learned at all. Visual observation is also
the more demanding setting, since morphology is decodable from the raw visual features at approximately 100
percent (a preliminary figure whose attribution to leg length versus recording session awaits a multi-session
control), while the joint commands used in the controlled setup are identical across all three bodies. Training
does use joint-command labels, which the simulator logs automatically at no annotation cost, to ground z_t
through a motion-decoding loss; that grounding module and its weights are retained after pretraining, since it is
the only component that converts a latent action back into an executable, body-specific joint command.

*(Experimental results to be added upon completion.)*

**Keywords:** cross-morphology locomotion / latent action / world model / transfer learning / V-JEPA2

---

# Chapter 1  Introduction

## 1.1  Background and Significance

Legged robots are typically controlled by policies learned through reinforcement learning (RL). A persistent
and expensive limitation of this approach is **morphology specificity**: a policy learned for one body does
not generalize to another. If a robot's legs are lengthened or shortened, its mass distribution changes, or a
limb is damaged, the previously trained policy fails and a new one must be learned from scratch, a process
that can take hours to days of training per body. In contrast, biological organisms share locomotion principles
across vastly different body plans, suggesting that a shared, body-independent representation of *movement* is
possible.

One promising direction is the **World Model**, an internal learned model of environment dynamics that predicts
future states from actions and lets an agent "imagine" rollouts instead of always acting in the real
environment (e.g. DreamerV3). A recent advance, the **Latent Action-Conditioned Robot World Model** (LAC-WM,
Huang et al., 2026), learns a *unified latent action space* shared across multiple robot embodiments, and shows
that a learned latent action generalizes across embodiments better than raw, body-specific action commands.
However, LAC-WM was demonstrated only for **manipulation** (robot arms and hands), and, like other
cross-embodiment methods, the embodiments it unifies have *structurally different action spaces*.

This research asks whether the same idea transfers to **legged locomotion**, and specifically to a setting the
prior work does not address: bodies that share an *identical* action space and differ only in **limb length**.
Here the challenge is not to reconcile different action dimensionalities, but to learn a latent action that
captures the **behaviour** (the intent of a movement, such as "step forward") independently of how a given body
realizes that behaviour, because identical joint commands produce *different resulting motion* on a short leg
versus a long leg. We ground this study in a simulated stick insect (*Medauroidea extradentata*), a
well-studied model organism for insect walking, instantiated at three leg lengths.

The significance is twofold. Scientifically, it tests whether a morphology-invariant notion of behaviour can
be learned from visual observation without the model ever being told the body's dimensions. Practically, if
such a representation transfers, it would cut the cost of deploying a new robot morphology by reusing
locomotion skill already acquired instead of retraining from zero.

## 1.2  Objectives

1. To learn a morphology-agnostic latent action z_t from simulation video of a legged robot, by adapting the
   Latent Action World Model pipeline (frozen visual encoder → Inverse Transition Model → Forward Transition
   Model → Motion Decoder) from manipulation to locomotion.
2. To demonstrate quantitatively that z_t organizes by **behaviour** rather than by **morphology**, using
   linear-probe accuracy and cluster-separation (silhouette / variance-decomposition) metrics across three
   leg-length variants of the same body.
3. To demonstrate that a world model pretrained on two morphologies enables a third, unseen morphology to reach
   the same reconstruction quality with **significantly fewer training episodes** than a model trained from
   scratch (i.e. improved sample efficiency of transfer).

## 1.3  Expected Benefits

1. A demonstration that latent-action world models extend beyond manipulation to legged locomotion.
2. Evidence on whether morphology-invariant behaviour structure can be learned **without** privileged
   morphology information (no CAD/spec input), using only visual observation and auto-logged action labels.
3. A reusable evaluation methodology (behaviour-vs-morphology latent probing) and a controlled, single-axis
   leg-length benchmark for future cross-morphology locomotion studies.
4. A foundation for reducing the retraining cost of deploying legged robots of new body proportions.

## 1.4  Scope of the Research

1. The study is conducted **entirely in simulation** (CoppeliaSim v4.10 with the Bullet physics engine). No
   physical-robot or sim-to-real deployment is in scope.
2. The robot is a single species, the stick insect *Medauroidea extradentata*, instantiated as **three
   morphologies** that differ **only in leg length** (1.0× baseline, 0.75×, 0.5×); all other properties
   (topology, degrees of freedom, 18-dimensional joint-position action space) are held identical. Training
   uses the 1.0× and 0.5× variants; the 0.75× variant is held out for the transfer test. This is an
   **interpolation** study (the unseen body lies between the trained ones), not extrapolation.
3. The primary behaviour studied is **forward walking**. Turning and stopping are planned extensions that the
   inverse-kinematics data pipeline (Section 3.5.3) can supply as additional task-space foot trajectories at no
   architectural cost; whether they are included in this phase depends on the biological/reference data available
   for them. `[[DECIDE]]`: confirm with the advisors whether the proposal commits to walk/turn/stop or to
   forward-walking only for this phase.
4. The visual encoder (V-JEPA2) is used **frozen**; training it is out of scope except for a possible
   fine-tuning of its final blocks if a domain gap is found.

**Explicitly not claimed.** To set expectations precisely, the following are outside the claims of this thesis:
real-robot or animal transfer (the study is entirely in simulation); extrapolation beyond the trained
leg-length range (the held-out body is an interpolation, between the trained ones); invariance to camera
viewpoint (a single fixed viewpoint is used throughout); generalization to manipulation or complex terrain; and
fully autonomous closed-loop control through the latent action. A further honest caveat: the inverse-kinematics
data redesign (Section 3.5.3) creates comparable task-space objectives across bodies, but it does not guarantee
identical contact dynamics or identical behaviour across morphologies, and the preliminary numbers of Section
3.7 are diagnostic and will be regenerated on the redesigned dataset.

## 1.5  Research Procedure

The project follows a staged, milestone-gated procedure in which each stage must pass before the next begins
(this component-by-component validation strategy was set by the lab advisor):

1. **Morphology Gap Check:** confirm that the three leg-length variants produce measurably different behaviour
   under identical commands (otherwise the transfer question is vacuous).
2. **Visual Encoder Sanity Check:** confirm that the frozen V-JEPA2 features carry the behaviour-relevant
   signal (gait/contact state) needed downstream, before any training.
3. **World-Model Training:** train the Inverse/Forward Transition Models and Motion Decoder on the two
   training morphologies.
4. **Latent-Space Validation:** verify that the learned latent action z_t clusters by behaviour, not
   morphology.
5. **Transfer Evaluation:** measure the sample-efficiency gain of the pretrained world model on the unseen
   morphology, against a from-scratch baseline and a raw-action baseline.

---

# CHAPTER 2  Literature Review

The review begins with a short, self-contained account of the fundamental concepts the rest of the chapter uses
(Section 2.1), so that the work is understandable without prior background. It then draws on four bodies of
work: world models, latent action models, cross-morphology legged control, and the biomechanics of insect
walking (which motivates the model organism and the evaluation). Each is reviewed in turn, and the review closes
with the specific research gap this work addresses and an argument for its central design choice.

## 2.1  Fundamental Concepts

This section defines, in plain terms, the concepts the rest of the review and the methodology rely on. It is
limited to what this work actually uses; a reader familiar with machine learning can skip to Section 2.2.

**2.1.1  Reinforcement learning and policies.**
Reinforcement learning (RL) is a framework in which an **agent** interacts with an **environment** over time. At
each step the agent observes the current **state** (for a robot, its configuration and surroundings), chooses an
**action** (for a legged robot, the target angles for its joint motors), and the environment responds with a new
state and a scalar **reward** signalling how good the outcome was. The agent's goal is to maximise the total
reward accumulated over time. The rule the agent uses to choose an action from a state is called its **policy**,
written π: state → action. Learning a good policy by trial and error is how most modern legged robots are
controlled.

The central difficulty this thesis addresses follows directly from this definition. A policy is a function
learned for one specific body: it maps *that body's* states to *that body's* motor commands, and in the process
it silently absorbs that body's geometry and dynamics. When the body changes (a leg is lengthened), the same
policy applied to the new body produces the wrong motion, because the state-to-action mapping it learned no
longer matches the physics. This is why a policy is said to be "tied to its morphology," and why a change of body
normally forces retraining from scratch.

**2.1.2  Neural networks, encoders, and embeddings.**
A **neural network** is a function with many adjustable numbers (parameters, or "weights") that is *trained* by
adjusting those numbers so the function's output matches desired targets on many examples. Given enough data,
neural networks can learn very complex mappings, such as from a raw image to a useful description of its content.

An **encoder** is a neural network that compresses a high-dimensional input (such as a 256×256×3 image, which is
almost two hundred thousand numbers) into a much smaller vector of numbers that summarises the input's meaningful
content. That output vector is called an **embedding** (in this work, e_t). Embeddings are useful because
downstream models can operate on the compact, informative embedding instead of on raw pixels. When an encoder's
weights are held fixed and not adjusted during later training, it is called **frozen**; a frozen encoder acts as
a fixed "translation" from images to embeddings.

**2.1.3  Self-supervised learning.**
Neural networks are commonly trained by **supervised learning**, in which every training example carries a
human-provided label (this image is a cat) and the network learns to reproduce the labels. Labels are expensive,
so a second paradigm, **self-supervised learning**, creates the training signal from the data itself with no
human labels: part of the input is hidden and the network is trained to predict it from the rest. Because the
"answer" is just another part of the same data, self-supervised learning can use enormous unlabeled datasets. The
visual encoder used here (V-JEPA 2, Section 2.2.2) is trained this way on about one million hours of unlabeled
video, which is why it can be reused without any task-specific labels.

**2.1.4  The Transformer and the attention mechanism.**
The learned models in this work (the encoder and the transition models) are built on the **Transformer**
architecture, whose core is the **attention mechanism**. Attention is worth understanding because it explains how
these models decide *what part of the input to look at*.

The intuition is as follows. Split the input into small pieces called **tokens** (for an image, each token is a
small square patch of pixels; for a pair of frames, the tokens are all the patches of both frames). On its own, a
single patch of a leg knows nothing about the rest of the scene. Attention lets every token **look at every other
token and take a weighted summary of them**, so that each token's representation becomes aware of its context.
The patch of a foot can incorporate information from the patch of ground beneath it, or from where the same foot
was in the previous frame.

Mechanically, each token emits three vectors: a **query** ("what am I looking for?"), a **key** ("what do I
offer?"), and a **value** ("what content do I carry?"). To update one token, the model compares that token's
query against every other token's key to get a set of relevance scores, converts those scores into weights that
sum to one, and returns the weighted average of all the tokens' values. A token thus pulls in most strongly from
the tokens most relevant to it. When the tokens attend among themselves this is **self-attention**; when one set
of tokens (for example a single learned query summarising "what changed") attends over another set (for example
all the patches of two frames) it is **cross-attention**. **Causal** self-attention simply restricts each token
to attend only to earlier positions, which is a natural way to express "what happened between an earlier and a
later frame." Stacking many such attention layers, interleaved with small per-token networks, is what makes a
Transformer, and it is the mechanism by which this work's Inverse and Forward Transition Models (Section 3.4)
decide which moving parts of the scene define the latent action.

**2.1.5  Evaluation: probing and metrics.**
Because the central claim of this thesis is about *what information a representation contains*, the evaluation
uses tools that measure exactly that.

A **linear probe** is the primary tool. To ask "does embedding e_t contain information about property Y (for
example, which feet are planted)?", one trains a *simple linear classifier* to predict Y from e_t and measures
how well it does. The classifier is deliberately kept simple (linear) so that success reflects the *quality of
the embedding* rather than the cleverness of the classifier: if even a linear model can read Y out of e_t, then Y
is present in an accessible form. Probing therefore measures the **presence** of information.

Several standard measures accompany the probe. **Accuracy** is the fraction of predictions that are correct, but
it is misleading when the classes are imbalanced (a model that always predicts the most common class can score
high while learning nothing). **Macro-F1** corrects for this by scoring each class separately and averaging with
equal weight, so a model that ignores rare classes is penalised; it is the metric used here for the
imbalanced foot-contact labels. Every probe is compared against a **chance baseline** (the score a random guesser
would obtain, for example 1/8 for eight equally likely classes) so that "above chance" is well defined. To ensure
a probe has genuinely learned rather than memorised, results are reported under **cross-validation** (the data is
split into folds; the model is trained on some folds and tested on the held-out fold, repeated so every example
is tested once) and against a **shuffled-label control** (the labels are randomly permuted; a model that still
scores above chance would be exploiting an artefact, so a control that collapses to chance confirms the real
result is genuine). Finally, a probe measures whether information is *present*, but not whether it *dominates* the
representation; for that this work also reports the **silhouette score**, a measure of how cleanly the embeddings
separate into groups, which captures whether a property is the main axis of variation rather than merely
decodable. Section 3.6.1 explains why both are needed.

## 2.2  World Models and Video Representation Learning

**2.2.1  What a world model is, and why it is used here.**
In reinforcement learning an agent must decide which action to take in each state. A *model-free* agent learns this
mapping by trial and error directly in the environment, which is data-hungry and, for a physical robot, slow and
risky. A *world model* takes a different route: it first learns to **predict the consequences of actions** (given
the current state and an action, it predicts the next state) and thereby builds an internal, learned simulator of
the environment's dynamics. Once such a model exists, the agent can train its policy by "imagining" long sequences
of hypothetical actions inside the model, without touching the real environment. This is the sense in which world
models let an agent "learn in imagination."

The reference implementation is DreamerV3 (Hafner et al., 2023). Rather than predicting raw future pixels, which is
expensive and forces the model to waste capacity on visually irrelevant detail, DreamerV3 encodes each observation
into a **compact latent state** and learns the dynamics *between latent states*. A single agent with one fixed
hyperparameter setting was shown to master more than 150 distinct tasks, from control benchmarks to collecting
diamonds in Minecraft from raw pixels, which established learned latent-state dynamics as a robust and general
tool. Two properties of this design matter for the present work. First, because the policy is trained against the
model's predictions, the *representation* the model learns is the object the policy actually consumes; if that
representation can be made body-independent, everything built on top of it inherits that property. Second, and
central to this thesis, a world model must answer the question "given this state and this action, what happens
next?", and that predictive objective is an operational *test* of whether a proposed action representation
captures meaningful change, independent of any downstream control task. This project uses a world model less to
imagine rollouts than to use its transition-prediction objective as such a test.

A limitation of the standard formulation, which motivates Section 2.3, is that DreamerV3 and its relatives learn a
separate model per domain and condition their predictions on the agent's **native action**, the robot's own
motor command. Nothing is shared across different bodies, and the native command is assumed to be the meaningful
conditioning signal. When the same command has a different physical meaning on a different body, that assumption
breaks.

**2.2.2  Video foundation models and the JEPA objective.**
To apply a world model to visual observations, one first needs to turn images into useful features. A *video
foundation model* is a large network pretrained on massive unlabeled video so that its features transfer to many
downstream tasks. This work uses V-JEPA 2 (Assran et al., 2025). Its training objective is the distinguishing
feature: instead of reconstructing masked pixels (as in masked autoencoders), a JEPA (Joint-Embedding Predictive
Architecture) masks part of the input and predicts the **representation** of the masked part rather than its
pixels. Predicting in representation space frees the model from modelling unpredictable low-level detail (exact
textures, lighting noise) and pushes it toward features that capture *how things move and change*. Trained by this
objective on roughly one million hours of internet video, V-JEPA 2 learns motion-relevant features that transfer
to action recognition, prediction, and planning, and reports state-of-the-art results on motion-centric
benchmarks while remaining competitive on appearance-centric ones.

Two aspects of V-JEPA 2 are directly relevant. First, the accompanying V-JEPA 2-AC (action-conditioned) variant
uses the pretrained encoder as a **frozen per-frame image encoder** feeding a lightweight downstream predictor,
without fine-tuning the encoder. This is the precedent for how the encoder is used here, and it is why a large
frozen backbone can be treated as a fixed perceptual front-end (Section 3.3). Second, V-JEPA 2 was trained on
real-world video only, with no rendered or simulated data. Its behaviour under a rendering domain gap is therefore
not characterised by the original work, which is one reason this project verifies empirically (Section 3.7.2) that
the frozen features carry the required locomotion signal on simulated stick-insect footage before relying on them.

## 2.3  Latent Action Models and Latent Action World Models

**2.3.1  Inverse and forward dynamics models: the general building block.**
Two complementary models recur throughout control, world modelling, and representation learning, and both the
latent-action idea and this project's architecture are built from them. A **forward dynamics model** answers "given
the current state and an action, what is the next state?" and maps (s_t, a_t) to ŝ_{t+1}. A world model's
transition predictor is exactly a forward dynamics model, and it is the component an agent rolls forward to imagine
the future (Section 2.2.1). An **inverse dynamics model** answers the opposite question, "given two consecutive
states, what action caused the transition?" and maps (s_t, s_{t+1}) to â_t.

Both are established, general-purpose techniques rather than the invention of any single paper. Inverse dynamics
models have long been used to recover actions from observation alone: the Intrinsic Curiosity Module (Pathak et
al., 2017) pairs an inverse and a forward model so that an agent can explore using its own prediction error, and
Video PreTraining (Baker et al., 2022) trained an inverse dynamics model to *label* large quantities of unlabeled
gameplay video with the actions that must have produced them, so that behaviour could then be learned from video
without action annotations. Forward dynamics models are the core of the world-model line from Ha and Schmidhuber
(2018) through DreamerV3.

The two models are naturally combined. If the "action" recovered by the inverse model is not the robot's native
command but an inferred latent code, and the forward model is asked to predict the next state from the current
state plus that latent, then the pair *jointly learns* a latent action: the inverse model proposes it, and the
forward model's prediction error tests whether it is sufficient to explain the transition. This inverse-forward
pairing is the shared skeleton of the latent-action methods reviewed next (Section 2.3.2) and of the base method
this project adapts (Section 2.3.3). *(This project instantiates the two models on visual embeddings and, for that
reason, refers to them as the Inverse and Forward Transition Model; the naming rationale is given in Section 3.4.)*

**2.3.2  The idea of a latent action.**
The conditioning problem raised at the end of Section 2.2.1 is that the native motor command is a poor shared
signal across bodies. A *latent action* addresses this by not using the native command at all. The idea is to look
at two consecutive observations and infer a compact code (the latent action) that explains the change between
them. Formally, an *inverse* model reads (o_t, o_{t+1}) and outputs a latent action z_t; this is analogous to
inverse dynamics in control, which asks "what action must have produced this transition?" The key property is that
z_t is defined by *observed change*, not by any robot's motor format, so a single latent space can describe
several embodiments at once, and it can be learned from video that has **no action labels** at all.

This places the approach within the broader paradigm of **learning from observation** (also called imitation from
observation): learning behaviour by watching, without access to the demonstrator's action labels, and, in the
strongest form, from a third-person external view of a body that is not the learner's own. The latent-action line
is one way to make observation-only learning tractable, by recovering a usable action interface from the video
itself.

Several lines of work established this. Genie (Bruce et al., 2024) learned controllable, playable environments from
large collections of unlabeled internet video by inferring a discrete latent action between frames, showing that a
usable action interface can emerge without any action supervision. LAPA (Ye et al., 2024) carried latent-action
pretraining toward real robot control, and CLAM (Liang et al., 2025) introduced *continuous* latent actions, which
are the closest match to the continuous joint control needed for locomotion (a discrete code is ill-suited to
smooth, graded limb movement). Zhang et al. (2025) provide a theoretical analysis of what latent action models
actually capture, which is useful when interpreting whether a learned latent encodes behaviour or merely copies
the next observation, a failure mode this project must actively prevent (Section 3.1, cross-augmentation).

Most directly related to the cross-embodiment goal, UniSkill (Kim et al., 2025) learns an
**embodiment-agnostic skill representation** from video (including human video) so that a robot can imitate a
demonstration performed by a different body. Architecturally it is again the same building block of Section 2.3.1:
it uses an **inverse skill dynamics** model (infer a skill code from an observation pair) and a **forward skill
dynamics** model (predict a future observation from the current one plus the skill), confirming that the
inverse/forward pairing is the standard mechanism across this literature. UniSkill matters here for two reasons.
First, it is evidence that a body-independent representation of *skill* can, in fact, be learned from observation,
which supports the feasibility of the present question. Second, and importantly, it differs from this project in
*how* the invariance is obtained: UniSkill is trained on cross-embodiment data and explicitly optimises its
representation to align across bodies, whereas this project asks whether morphology-invariance can **emerge** from
a world-model bottleneck *without* any dedicated cross-embodiment alignment objective and without the morphology
ever being supplied. UniSkill also targets human-to-robot manipulation and imitation, not a controlled
single-axis leg-length study or a dynamics-prediction world model. The contrast, invariance *aligned by design*
versus invariance *tested for emergence*, is what sharpens the contribution of this work. An explicit
alignment objective of this kind is a possible future extension rather than a contingency the primary result
depends on: if morphology-invariance does not emerge, that outcome is itself a valid and informative result about
the limits of the emergent approach in this setting.

**2.3.3  Latent Action World Models (the base method).**
LAC-WM (Huang et al., 2026) combines the latent-action idea with a world model to unify a single latent action
space across *heterogeneous* manipulation embodiments (for example a robot gripper, a bimanual humanoid, and human
hand keypoints, whose action spaces have different dimensions and meanings). Its architecture is the pipeline this
project adapts, and it has four parts. (i) A **frozen visual encoder** (V-JEPA 2) turns each frame into an
embedding e_t. (ii) An **Inverse Transition Model (ITM)** reads the pair (e_t, e_{t+1}) and produces the latent
action z_t, meaning "what happened between these two frames." (iii) A **Forward Transition Model (FTM)** takes (e_t, z_t)
and predicts the next embedding ê_{t+1}; its reconstruction error tests whether z_t, together with the current
state, is sufficient to explain the observed transition. (iv) A **Motion Decoder (MD)** maps z_t back to the
ground-truth motor action, which anchors the latent to real, executable motion so that it cannot collapse into a
trivial or meaningless code. The world model is then conditioned on z_t rather than on any robot's native command,
so one latent space covers several embodiments, and, importantly, adding embodiments *improves* the shared model
rather than fragmenting it.

A subtle but essential detail is how LAC-WM prevents a shortcut. Because z_t is partly supervised to help
reconstruct e_{t+1}, the ITM could cheat by copying the next frame's content directly into z_t instead of learning
the abstract action. LAC-WM prevents this with **cross-augmentation**: two independent augmentations of the frame
pair are encoded, the ITM sees one while the reconstruction target is drawn from the other, so a copied pixel-level
representation no longer matches the target and z_t is forced to capture augmentation-invariant change. This
project inherits the mechanism (Section 3.1).

LAC-WM reports two headline results: its unified latent action transfers across embodiments better than an
explicit-action baseline (EAC-WM, which is forced to use per-embodiment action encoders), and its performance
*scales positively* with embodiment diversity. Its motivating premise, however, is heterogeneity of *action
spaces*: the baseline pathology it improves on exists precisely because different manipulators have genuinely
different action formats. That premise does **not** hold in this project's single-species, leg-length-only setting,
where all three bodies share an identical 18-dimensional action space. The reframing this requires, from
*action-space* heterogeneity to *dynamics* heterogeneity (where the same command produces different motion per
body), is the subject of Section 2.6.

## 2.4  Cross-Morphology and Cross-Embodiment Legged Locomotion

Making one controller work across bodies is an active problem, and the existing strategies differ mainly in
*what information about the new body they consume*. This subsection reviews the strategies most relevant to legged
locomotion; Section 2.7.3 organises them by the single question of where that information must come from.

**2.4.1  Per-body retraining.**
The baseline strategy is simply to train a new controller for each body. DreamerV3 (Section 2.2.1) is the
representative here: it is highly general across *tasks* but learns a fresh model per domain, with nothing carried
across bodies. This is the cost the field is trying to avoid, and it is the reference against which the
sample-efficiency benefit of the present method is measured (Section 3.6.3).

**2.4.2  Explicit morphology conditioning.**
QWM (Danesh et al., 2026) introduces a morphology-conditioned world model that generalises zero-shot to unseen
quadruped morphologies, and it is the strongest cross-morphology transfer result in this area. It achieves this by
reading physical parameters (limb lengths, masses, torque limits) out of each robot's CAD/USD description and
conditioning the world model on those numbers. Its observation channel is proprioception, and it does not learn a
latent *action*: the morphology is supplied, not inferred. QWM is therefore the closest and strongest prior art on
one axis (world-model transfer across quadruped bodies) and the clearest contrast on another (it assumes an
accurate machine-readable description of the new body). Notably, QWM routes what it calls "unmodeled real-world
residuals" through a separate dynamic latent, an implicit acknowledgement that the design file does not fully
describe the physical body, a point this project develops in Section 2.7.3.

**2.4.3  Online system identification.**
A related family infers hidden dynamics parameters online from a short window of recent interaction: the robot
moves, observes its own responses, and estimates what body it is. This removes the need for a design file but
requires access to the robot's internal signals (proprioception, past commands) and enough new interaction to
identify the parameters, so it still assumes an instrumented, controllable body.

**2.4.4  Shared policy with per-robot adapters.**
L3P (Zheng et al., 2025) shares a common policy backbone across several quadruped platforms while fitting a small
encoder and decoder for each robot. This shares *structure* across bodies, but the per-robot adapters must be
fitted from that robot's proprioception and foot force, with its joint ordering and actuator conventions known.
More broadly, graph- and transformer-based controllers such as MetaMorph (Gupta et al., 2022) learn
morphology-agnostic policies over many procedurally varied bodies, and Ai et al. (2025) study embodiment scaling
laws, showing that training across a large distribution of bodies improves generalisation. These establish
cross-morphology control as tractable but approach it through policy transfer conditioned on the body's internal
representation, not through a vision-based latent action inferred from observed transitions.

**2.4.5  Latent-action approaches in locomotion.**
Closest in spirit, Li et al. (2021) plan in a *learned latent action space* for legged locomotion and demonstrate
the method on both a hexapod and a quadruped. Crucially, their framework is re-instantiated and trained
*separately per robot*: no single latent space or dynamics model is shared or transferred across bodies, so it
does not perform cross-morphology transfer in the sense studied here. It shows that latent actions are a natural
fit for legged control, without yet making the latent space itself the object that transfers.

Across all five families, the observation channel is proprioception or a design file, and the body information is
obtained from inside the body or its specification. None studies whether an **external, vision-based latent action
inferred from observed transitions**, with morphology never supplied, can serve as the shared representation, the
gap stated in Section 2.6.

**2.4.6  Synthesis: positioning the methods.**
Table 2.1 places the main reviewed methods on the axes that matter for cross-morphology transfer: what they
*observe*, where they get *information about the new body*, how they represent an *action*, whether a single model
*transfers across bodies*, and the domain in which they were demonstrated. Read down the "morphology information"
column, every prior method obtains it from inside the body or from a design file; read across the bottom row, the
present work is the only entry that observes externally (vision), is never given the morphology, uses an inferred
latent action, and targets legged locomotion. The empty region this exposes (external observation, morphology
never supplied, latent action, locomotion) is precisely the research gap of Section 2.6. It is also worth noting
that latent-action and cross-embodiment-from-video methods (LAC-WM, UniSkill, and the latent-action pretraining
line) are overwhelmingly demonstrated on **manipulation**; legged locomotion, where the same command produces
different motion purely through body geometry, remains largely unaddressed by this family.

*Table 2.1. Positioning of cross-morphology / cross-embodiment methods. "Transfers across bodies" asks whether a
single trained model is reused on a new body without per-body retraining.*

| Method | Observation | Morphology information from | Action representation | Transfers across bodies? | Domain |
|---|---|---|---|---|---|
| DreamerV3 (Hafner 2023) | state / pixels | retrained per body | native command | no (per-body model) | general control |
| QWM (Danesh 2026) | proprioception | CAD / USD design file | native command | yes, zero-shot | quadruped |
| Online system ID | proprioception | inferred from interaction | native command | yes, after adaptation | legged / general |
| L3P (Zheng 2025) | proprioception + force | per-robot fitted adapter | shared policy latent | partial (adapter per body) | quadruped |
| MetaMorph (Gupta 2022) | proprioception + morphology graph | body graph / structure | native command | yes, within trained set | procedural bodies |
| Li et al. (2021) | proprioception | retrained per body | learned latent action | no (separate per body) | hexapod, quadruped |
| LAC-WM (Huang 2026) | vision | inferred from transition | learned latent action | yes | manipulation |
| UniSkill (Kim 2025) | vision | aligned by training objective | skill latent | yes | manipulation / human |
| **This work** | **vision (external)** | **never supplied** | **learned latent action** | **under test** | **legged locomotion** |

## 2.5  Biomechanics of Insect Walking and Prior Lab Work

**2.5.1  Why a stick insect, and how it walks.**
The stick insect (*Carausius morosus*, *Medauroidea extradentata*) is a classical model organism for legged
locomotion, studied for decades by Cruse, Bässler, Büschges, Dürr and colleagues. It is chosen here for three
reasons: its walking is exceptionally well characterised, so behaviour labels can be defined against a solid
reference; the host lab already maintains a validated simulation model of the exact species (Section 2.5.3); and,
as a slow, statically-oriented walker, it exercises the cross-morphology question on a body whose gait is
governed by leg coordination rather than by high-speed dynamics.

Coordination in stick insects arises not from a central clock but from **decentralised local rules** (the "Cruse
rules") acting between neighbouring legs (for example, a leg is discouraged from starting its return stroke while
its neighbour is still returning, and a leg that completes a step encourages the next). A single walking cycle per
leg alternates a *stance* phase (foot planted, propelling the body) with a *swing* phase (foot lifted, returning
forward). The standard quantitative descriptors are the **anterior and posterior extreme positions (AEP/PEP)**,
the boundaries of the stance stroke; the **duty factor**, the fraction of the cycle a foot is planted; and the
**inter-leg phase**, the timing offset between legs. A well-known consequence of the decentralised control is that
straight walking is largely an *emergent* property of the local rules and bilateral symmetry rather than the
output of a dedicated heading controller: in classic experiments even a single remaining leg still produces the
correct stepping pattern.

Insect gaits form a spectrum with walking speed: at high speed a *tripod* gait (two alternating sets of three
legs) is used, while at low and medium speed the coordination is a more continuous *wave* or *tetrapod-like*
pattern in which legs are recruited in a back-to-front sequence and the clean two-group structure of the tripod is
not present. This spectrum matters for the present study for two practical reasons. First, it justifies labelling
behaviour by the directly observable **foot-contact state** (which feet are planted) rather than by an abstract
phase, because contact is a well-defined physical event across the whole gait spectrum. Second, it sets the
correct expectation for the recorded data: the reference gait used here is a slow-walking pattern, and analysis of
both the pilot recordings and the lab's mature expert controller confirms a phase-staggered wave rather than a
clean tripod (the two canonical tripod groups do not co-activate; the "clean tripod" state is essentially absent).
This is the real gait of a slow-walking stick insect, consistent with the biology, and is not treated as a defect
to be corrected.

**2.5.2  Dynamic similarity and why leg length changes the outcome.**
That identical commands produce different locomotion on different-length legs (Section 3.7.1) is expected from
biomechanics, not an artefact. The dynamic-similarity theory of gaits (Alexander and Jayes, 1983) formalises how
gait and speed scale with size: geometrically similar bodies of different scale move similarly only when compared
at matched dimensionless speed (the Froude number), and identical joint kinematics on a longer limb sweep the foot
through a larger arc and change contact timing and ground reaction. The sub-linear scaling of walking speed with
leg length observed in the pilot (an exponent near 0.69 rather than 1.0, Section 3.7.1) is consistent with this
picture: the outcome of a fixed command is a function of the body's geometry, which is exactly the dynamics
heterogeneity the latent action is meant to abstract away.

**2.5.3  Prior work in the host lab.**
Directly relevant, Larsen et al. (2023, Manoonpong group) model the *same species* in CoppeliaSim under a
decentralised CPG (central pattern generator) controller with foot-contact feedback, explicitly handling
heterogeneous leg lengths, and Chuthong et al. (2026) study resilient stick-insect-inspired control. These provide
the biological data, the validated simulation model, and a controller lineage that this project builds upon. The
mature logged rollouts of this controller family are the source of the shared foot trajectories and binary contact
labels used in the dataset redesign (Section 3.5.3). The present project reuses this substrate but pursues a
different research question: a learned, vision-based, morphology-invariant *representation* of locomotion, rather
than a hand-designed controller.

## 2.6  Research Gap

**2.6.1  The gap in the literature.**
Prior cross-morphology world models either (a) require **explicit** morphology specifications from CAD
(QWM), (b) are trained **separately per body** without transfer (Li et al.), or (c) are demonstrated only for
**manipulation** (LAC-WM). None studies whether a **vision-grounded latent action**, with morphology **never
provided to the model**, organizes by *behaviour* rather than *body shape*, and none isolates the effect of a
single morphological axis (leg length) on that latent. This project addresses exactly that gap: an
implicit/spec-free, vision-based latent-action world model for locomotion, evaluated with a controlled
single-axis leg-length sweep and an explicit behaviour-vs-morphology latent probe. We are careful **not** to
claim vision-only learning: the latent is grounded during training on auto-logged joint commands (a by-product
of simulation, not used at inference), following LAC-WM.

The study is conducted in simulation deliberately, not as a limitation of convenience. Answering the research
question requires *ground-truth* control over the one variable under study: three bodies that are provably
identical in every respect except leg length, driven by provably identical commands, with exact per-foot contact
and body pose logged for evaluation. Only simulation provides this level of control and measurement, which is what
makes the behaviour-versus-morphology comparison clean. Real-robot and sim-to-real transfer are explicitly out of
scope (Section 1.4); the contribution is the controlled demonstration that the abstraction can, or cannot, be
learned, which is a prerequisite to any later physical deployment.

**2.6.2  The necessary reframing: from action-space heterogeneity to dynamics heterogeneity.**
LAC-WM's justification for a shared latent (Section 2.3.3) is that its embodiments have genuinely *different action
spaces*, which forces the explicit-action baseline into per-embodiment encoders. This project cannot borrow that
justification, because its three bodies share an *identical* 18-dimensional action space; a reviewer would rightly
ask why a latent action is needed at all when every body accepts the same command format. The answer is a
different, and arguably cleaner, source of heterogeneity. The bodies differ not in the *format* of the action but
in the *dynamics* that action produces: as Section 3.7.1 measures directly, an identical command yields
non-overlapping motion across leg lengths (p = 0.0079, complete separation). The problem is therefore not to
reconcile different action dimensionalities but to learn a representation of the *behaviour* (the intent of a
movement, such as "advance the middle-left leg into stance") that is invariant to how a given body must actuate
its joints to realise it. This dynamics-heterogeneity framing is what makes the single-axis leg-length setting a
valid and, in fact, more controlled instance of the cross-embodiment problem than the action-space-heterogeneity
setting: every confounding difference except the one under study (leg length) is held fixed.

**2.6.3  The decisive test built into the design.**
Because the action space is shared, the value of the latent action is not assumed but tested head-on: a
latent-conditioned dynamics model is compared against one conditioned directly on the raw 18-dimensional command,
under an identical encoder, model capacity, dataset, and training budget (Section 3.6.3). If the raw-command model
transfers equally well across morphologies, the latent bottleneck adds nothing in this controlled setting, and the
thesis reports that honestly. This ablation is treated as the central experiment rather than a supplementary one.

**2.6.4  Why success requires a two-sided criterion.**
Framed abstractly, the goal is a representation that is *invariant* to a nuisance factor (morphology) while
*preserving* a factor of interest (behaviour). This is the standard problem of learning an invariant, or partially
disentangled, representation, and it comes with a standard pitfall that dictates how success must be measured. It
is trivial to make morphology unrecoverable: a representation that discards all information (one that maps every
input to the same constant) leaks nothing about the body, but it also leaks nothing about the behaviour and is
therefore useless. A criterion that only rewards *reduced morphology decodability* would score this degenerate
"collapse" as a perfect result. For this reason the success criterion is deliberately **two-sided**, requiring
both that morphology becomes *less* decodable from z_t than from the raw features **and** that behaviour remains
*more* transferable, measured at the same time (Section 3.6.1). Only a representation that removes the nuisance
factor *while retaining the content* satisfies both, which is what makes the test meaningful rather than
self-confirming. The same reasoning underlies treating the held-out medium body as a genuine test: it is a
held-out *domain*, and reaching it requires that the representation generalise across the morphology axis rather
than merely fit the two training bodies.

## 2.7  Why the Observation Modality Is Part of the Research Question

Using vision instead of proprioception or CAD parameters is not presented here as a technological preference.
The modality determines whether the research question can be asked at all, and it changes what the eventual
answer would mean.

**2.7.1  Explicit morphology answers a different question.** QWM supplies limb lengths, mass, and torque
limits to the world model and shows that generalization follows. That establishes something worth knowing:
*given* an accurate description of the body, a world model can compensate for it. It cannot establish whether
a body-independent notion of behaviour is *learnable*, because the answer has already been provided in the
input. Our question is the second one, and it is only well posed if the model is never told what body it is
looking at. Spec-free observation is therefore a requirement of the experiment rather than a design choice,
and vision is the natural spec-free channel.

**2.7.2  Vision is the adversarial modality, not the convenient one.** A reasonable objection is that vision is
chosen because it is fashionable. Our own preliminary measurements argue the opposite. Morphology is decodable
from the raw visual features at approximately 100 percent, with a morphology silhouette of +0.0835, and the
leading principal component of the embedding orders the three bodies by leg length with no labels supplied, so
body shape is a dominant axis of the visual representation. A necessary caution accompanies this figure: because
each body is recorded in a single session, morphology and recording session are confounded in the pilot data, and
the multi-session control that would attribute the signal specifically to leg length is planned rather than done;
either way, this is exactly the signal the latent action must remove. The confound-maximising nature of vision is
best stated through what the *command* carries rather than through proprioception in general. In this study's
control setup the joint *commands* are bit-identical across the three bodies, so the command itself carries no
morphology signature; the resulting *measured* proprioception (foot forces, realised joint state) does differ
across bodies, but the standard proprioceptive channel a controller conditions on would not, by construction,
separate them in this setup. Vision, by contrast, sees the body shape directly and in full. Vision therefore
maximises the confound that the latent action is supposed to remove, and demonstrating behaviour-level invariance
under maximal visible morphological difference is a stronger result than demonstrating it where the confound is
weak.

**2.7.3  The requirements differ in where they land, not in whether they exist.** It would be inaccurate to say
that specification-based and proprioception-based methods have requirements and the present method does not.
The present method requires an external camera with a view of the body, and it requires logged joint commands
from the bodies used during pretraining. The relevant difference is that these requirements fall on the
observer, whereas the alternatives place their requirements on the observed body itself.

A CAD or USD file describes the design, not the individual robot: wear, payload, manufacturing tolerance, and
damage are absent from it, and these are exactly the conditions under which morphology adaptation is most
needed. QWM implicitly concedes this by routing "unmodeled real-world residuals" through its dynamic latent.
Proprioception, as used by L3P, requires the target body to carry joint and force sensing and requires its
joint ordering and conventions to be known, so the per-robot encoder and decoder can be fitted. Both
preconditions can be satisfied only by a party with access to the interior of the body or to its design
record. A camera can be positioned by anyone who can see the subject.

This matters because the bodies for which morphology adaptation is most valuable are frequently ones where
interior access is unavailable in principle rather than by oversight: animals, which cannot be fitted with
joint encoders; robots whose configuration has diverged from their documentation through repair, payload, or
wear; hardware acquired from another party without published kinematics; and, in the limiting case, extinct
animals reconstructed from skeletal remains and trackways, for which no specification exists and none can be
produced. For these, an external view is not a preferred channel but the only remaining one.

**2.7.4  Only visual data scales beyond bodies one already owns.** Recordings of legged locomotion exist in
very large quantity for animals and for robots built by other groups. Proprioceptive logs and CAD files do not
exist for bodies one did not build and instrument. A latent action grounded in vision is the only version of
this idea that could later ingest observations of embodiments outside the laboratory, which is the long-horizon
reason the cross-embodiment literature cares about the problem.

**2.7.5  Consistency with the biological framing.** Animals do not exchange kinematic specifications. Insects
and vertebrates infer what another body is doing by observing it. If the claim under test is that behaviour is
an abstraction separable from body shape, evidence gathered through the channel that biological imitation
actually uses is more consistent with that claim than evidence gathered from privileged design parameters.

**2.7.6  Costs accepted.** Vision is noisier and more fragile than proprioception. Raw visual features are
dominated by rendering style rather than behaviour, which forces strict control of camera, lighting, and
background (Section 3.2.3), and two plausible background choices were found empirically to corrupt the signal
in opposite ways. Pixels also widen the eventual sim-to-real gap relative to joint-space observation, and the
frozen encoder is computationally heavy. These costs are accepted because Sections 2.7.1 and 2.7.2 make vision
the setting in which the result, if obtained, actually means what the thesis claims it means.

---

# CHAPTER 3  Research Methodology

## 3.1  System Overview

The system comprises four learned/parameterized components on top of a fixed perceptual front-end, trained
end-to-end except for the frozen encoder:

```
 RGB frame o_t ─▶ [ V-JEPA2 encoder (FROZEN) ] ─▶ visual embedding e_t (256×1408)
                                                      │
              e_t, e_{t+1} ─▶ [ Inverse Transition Model (ITM) ] ─▶ latent action z_t (64-d)
                                                      │
                     e_t, z_t ─▶ [ Forward Transition Model (FTM) ] ─▶ ê_{t+1}   (L_recon)
                                                      │
                     x_t, z_t ─▶ [ Motion Decoder (MD) ] ─▶ â_t  vs  a_t (auto-logged)  (L_motion)
```

Input/output specification of each block (units stated, per the advisors' requirement):

| Block | Input | Output | Trained? |
|---|---|---|---|
| V-JEPA2 encoder | RGB frame, 256×256×3 (uint8) | e_t ∈ ℝ^{256×1408} patch tokens | No (frozen) |
| Inverse Transition Model | [e_t, e_{t+1}] (512 tokens) | z_t ∈ ℝ^{64} (latent action) | Yes |
| Forward Transition Model | [e_t, z_t] | ê_{t+1} ∈ ℝ^{1408} | Yes |
| Motion Decoder | (e_t as context, z_t as query) | â_t ∈ ℝ^{18} (joint targets, rad) | Yes (not used in Phase 1 evaluation; weights retained for downstream use) |

Losses: **L = λ_recon · L_recon + λ_motion · L_motion**, where L_recon = ‖ê_{t+1} − e_{t+1}‖² is computed in
embedding space (no pixel decoder required) and L_motion = ‖â_t − a_t‖². **Cross-augmentation** is applied:
two independent augmentations of each frame pair are encoded, one feeding the ITM and the other the FTM target,
which prevents the ITM from smuggling raw future-frame content into z_t as a shortcut.

## 3.2  Simulation Environment and Morphology Variants

**3.2.1  Simulator and model.** CoppeliaSim v4.10 (Bullet 2.78 engine, 20 Hz timestep). The robot is the
*Medauroidea extradentata* model (six legs, three actuated joints each: ThC/CTr/FTi), borrowed from the host
lab, with an 18-dimensional joint-position-target action space.

**3.2.2  Leg-length variants.** Three scenes are generated by scaling the coxa/femur/tibia segments of all six
legs by a uniform factor and repositioning downstream joints accordingly: **long 1.0× (base), medium 0.75×,
short 0.5×**. Scaling is verified numerically (measured foot-reach ratio matches the target factor for all six
legs).

**3.2.3  Observation capture.** A single fixed vision sensor (256×256, side view, ~40° elevation) is added
programmatically so that its pose relative to the robot is *identical across all morphologies by construction*.
The floor uses a matte, mildly-textured, non-repeating surface (a controlled choice: both a stark checkerboard
and a blank surface were found to corrupt the visual features). The camera tracks the robot's planar position
at fixed height and orientation, keeping apparent size constant. `[[DECIDE]]`: single-camera is used; a
multi-camera setup was considered and rejected (see project notes).

## 3.3  Visual Encoder (Frozen V-JEPA2)

The encoder is `facebook/vjepa2-vitg-fpc64-256` (ViT-g/16, ~1B parameters), used frozen. Because this
checkpoint is natively a 64-frame *video* encoder with 2-frame tubelets, feeding a real clip would let each
frame attend to other timesteps and contaminate its embedding. To obtain an independent per-frame embedding,
each frame is **duplicated into the minimal 2-frame tubelet** and encoded alone, yielding 256 patch tokens of
dimension 1408 per frame. This usage has direct precedent in V-JEPA 2-AC.

## 3.4  Latent Action Model

**Naming.** The two learned transition models below are instances of the inverse and forward dynamics models
described in Section 2.3.1: the ITM infers the latent action from a state transition, and the FTM predicts the next
state from the current state and that latent. This project refers to them as **Transition Models** rather than
*dynamics* models deliberately. The term "dynamics" carries a physical connotation (forces, masses, contact
mechanics), whereas these modules never touch physical quantities; they operate entirely on **transitions between
visual embeddings** produced by the frozen encoder. "Transition Model" names precisely what the module computes (a
mapping over embedding transitions) and avoids implying that the model estimates the body's physical dynamics,
which it does not. The underlying technique is identical to the inverse/forward dynamics models of the literature;
only the name is chosen for descriptive accuracy in the visual-embedding setting.

**3.4.1  Inverse Transition Model (ITM).** Attention-based (causal self-attention blocks with a learned query
token) mapping the pair [e_t, e_{t+1}] to the latent action z_t ∈ ℝ^{64}. The causal structure frames z_t as
"what happened between t and t+1." This is the inverse dynamics model of Section 2.3.1, computing z_t from a
transition instead of a native action.

**3.4.2  Forward Transition Model (FTM).** Attention-based, predicting the next visual embedding ê_{t+1} from
[e_t, z_t]; supervised by L_recon in embedding space.

**3.4.3  Motion Decoder (MD).** Cross-attention with z_t as query over the current visual context, producing
â_t; supervised by L_motion against the auto-logged joint command a_t. This is the only place ground-truth
action is used, and the decoder is **not part of the Phase 1 evaluation**, though its weights are retained. Its
role during pretraining is to anchor z_t to real motion so the latent cannot collapse into a trivial identity.
Beyond this thesis it has a second role: it is the only module that maps a latent action back into executable
joint commands, so any downstream policy emitting z_t would require it. This also answers a standing question
about the design, namely why a latent is introduced at all when the robot ultimately needs joint commands: the
policy operates in the latent space because that is the part expected to transfer across bodies, while the
decoder performs the body-specific conversion. The conversion separates what transfers from what does not.

**3.4.4  Dimensions and hyperparameters.** z_t is 64-dimensional (per LAC-WM §4.2). λ weights, learning rate,
and optimizer are to be determined by ablation (the base paper does not report them). Training uses mixed
precision on a single GPU (RTX 2080 Ti, 11 GB), with small batch size and gradient accumulation to fit the
frozen 1B-parameter encoder in the training loop (cross-augmentation prevents offline embedding caching).

## 3.5  Data Collection

**3.5.1  Pilot dataset (reference-gait replay).** The preliminary results of Section 3.7 were obtained on a pilot
dataset in which each morphology is driven by a single fixed reference gait (replayed real-animal joint
trajectories from one *Medauroidea* individual), capturing per timestep the RGB frame, the 18-dimensional joint
command a_t, the body pose, and the per-foot contact force. Reloading the scene between episodes provides genuine
variation, because the contact-rich dynamics are chaotic and each reload yields a different trajectory, while
within an episode the simulation is deterministic. Behaviour is labelled by the **6-bit foot-contact state**
(which of the six feet are planted), a body-pose label preferable to a time-based phase label for the reason
given in Section 3.7.3.

**3.5.2  Why the pilot dataset is not sufficient for the main experiment.** In the pilot, the *same* joint-command
sequence is applied to all three bodies (this is exactly what makes the morphology-gap test of Section 3.7.1
valid: identical input, different outcome). Verification confirms the commands are bit-identical across bodies
(maximum pairwise difference is zero to machine precision). This property, correct for the gap test, is fatal for
training the latent action model. The Motion Decoder is grounded by L_motion = ‖â_t − a_t‖²; if a_t is identical
across bodies, the target carries no morphology information, the decoder has no reason to condition on the body,
and there is nothing to *retarget*. The latent action z_t would then be describable as a body-independent action
simply because the *command* was already body-independent, which is a vacuous form of the property this thesis
claims to demonstrate. The main experiment therefore requires a dataset in which the *behaviour* is shared across
bodies but the *command* differs per body.

**3.5.3  Dataset redesign (inverse-kinematics retargeting).** The redesign defines each behaviour not as a joint
trajectory but as a **task-space foot trajectory** (a Cartesian path for each foot). Given a shared foot
trajectory, inverse kinematics is solved *separately for each morphology* to obtain the joint commands that place
that body's feet on the shared path: a_t^{body} = IK(foot trajectory, body). Because foot position depends on both
joint angle and link length, the same foot target yields different joint angles on a long leg and a short leg, so
the resulting commands differ per body while the intended behaviour is held constant. This is the precise
condition the Motion Decoder must learn to recover: a body-independent intent (the latent action) mapped to
body-specific joint commands. The source of the shared foot trajectories is a mature logged expert dataset from
the host lab (`expert_66k_aug3c_fcontact`, 66,000 timesteps) that records foot trajectories and the simulator's
own binary contact labels directly, so the trajectories need not be re-derived and the contact labels do not
depend on a hand-chosen force threshold. IK is a closed-form solve requiring no additional training, and it is
introduced as an *addition* to the pipeline rather than a change to the model architecture. A single successful
IK pipeline also supplies the multiple behaviours (walk, and prospectively turn and stop) as additional foot
trajectories, which the latent-space validation of Section 3.6.2 requires. `[[LIMITATION]]`: IK produces
comparable task-space objectives but does not guarantee identical contact dynamics across morphologies, and the
anticipated objection that "learning the Motion Decoder is merely learning IK" is answered by the claim under
test, namely that the model recovers this body-specific retargeting from *observation alone*, without being given
the kinematics.

![Same command, different physical state](fig_same_command.png)

*Figure 3.4. A single timestep with a bit-identical joint command across the three bodies. The left-middle foot
carries 5.7 N on the long body and 9.3 N on the short body but only 0.3 N on the medium body, where it is still
airborne. The command cannot distinguish the bodies; the difference exists only in the resulting physical state.*

![Same angle versus same foot target](fig_ik_intuition.png)

*Figure 3.5. Why per-body commands are needed. Left: identical joint angles place the foot in different positions
on a long versus a short leg. Right: inverse kinematics inverts the constraint, so a shared foot target is reached
by different joint angles per body, which is the body-specific command the Motion Decoder must learn to produce.*

## 3.6  Evaluation Protocol

**3.6.1  Metrics.** Two complementary measures are reported for every latent-structure claim, since reporting
only one can mislead. The first is **linear-probe accuracy**: can a simple classifier decode a label
(behaviour or morphology) from the representation? This measures the *presence* of information. The second is
the **silhouette score together with between-class variance**, which measure *dominance*, that is, whether
that information is the main axis of variation. Cross-morphology **transfer** is measured by training a probe
on two morphologies and testing on the held-out one.

**3.6.2  Milestone gates.**
- *Morphology gap:* pass if the three variants travel measurably different distances under identical commands.
- *Encoder sanity:* pass if behaviour (foot-contact state) is decodable from e_t above a noise floor.
- *Latent validation:* pass if z_t lowers morphology decodability and raises cross-morphology behaviour
  transfer relative to the raw encoder baseline.
- *Transfer:* pass if the pretrained world model reaches the same L_recon on the unseen morphology with
  significantly fewer episodes than a from-scratch model.

**3.6.3  Baselines.** (1) A from-scratch world model on the unseen morphology (the sample-efficiency
reference). (2) A **raw-joint-conditioned** forward model with a shared encoder (an "explicit action" analog):
if the latent action does not beat raw joint conditioning, the latent bottleneck adds no value, which makes
this the decisive ablation. (3) Where applicable, an explicit morphology-conditioned model in the spirit of QWM, as
a contrast between explicit and implicit morphology handling.

## 3.7  Preliminary Validation of the Method

The first two milestones of Section 1.5 have been executed on a **pilot dataset** (the reference-gait replay of
Section 3.5), and their outcomes justify the design choices above. These are diagnostic results whose purpose is
to establish that the methodology is sound before the main training begins; they are **not** the thesis's final
measurements. They will be regenerated on the redesigned IK dataset (Section 3.5), and the exact values are
expected to shift. Every figure below is reproducible from the recorded data. Full results belong in Chapter 4.

**3.7.1  The morphology gap is measurable and statistically decisive.** All three variants were driven with a
bit-identical 18-dimensional command sequence, and the body's world-frame position was read directly from the
simulator. Two distance measures are reported: *path length* (the distance actually walked) and *net displacement*
(straight-line progress), because the open-loop gait has no heading correction and its side-to-side body
oscillation must not be mistaken for a locomotion difference.

| Morphology | Path length (n = 5) | Net displacement (n = 5) |
|---|---|---|
| Long, 1.0× | 5.217 ± 0.069 m | 4.404 ± 0.187 m |
| Medium, 0.75× | 4.149 ± 0.019 m | 3.569 ± 0.010 m |
| Short, 0.5× | 3.228 ± 0.011 m | 2.729 ± 0.011 m |

![Morphology gap under identical commands](fig_step_minus1.png)

*Figure 3.1. Left: top-down body trajectories (five episodes per morphology) under an identical command
sequence, aligned to a common start and heading and drawn at true aspect ratio. Right: per-episode path length
and net displacement with mean ± standard deviation; the three morphologies do not overlap.*

A two-sided Mann-Whitney U test rejects the null hypothesis of equal outcomes for every pair of morphologies at
p = 0.0079, with Cliff's δ = 1.00 (complete separation: every episode of a longer-legged body exceeds every
episode of the shorter one; the worst long-leg net displacement, 4.032 m, still exceeds the best short-leg value,
2.746 m). At n = 5 per group, p = 0.0079 is the smallest value the test can return, i.e. the strongest statement
this sample size permits. The relationship is not a simple rescaling: walking speed falls with roughly the 0.69
power of leg length rather than the 1.0 that naive geometric scaling would predict. This confirms that identical
commands yield genuinely different motion, which is the premise of the whole study.

**3.7.2  The frozen encoder carries the required behaviour signal, entangled with body shape.** Using a linear
probe (logistic regression with standardisation) on the mean-pooled 1408-dimensional embedding, foot-contact
state (which feet are planted, top-8 patterns) is decodable **within a body** at a mean macro-F1 of 0.84
(long 0.83, medium 0.95, short 0.78; per-body variance is high, ±0.18 on the long body, because only five
episodes are available), against a shuffled-label control at chance (macro-F1 ≈ 0.12, chance = 0.125). The
Inverse Transition Model therefore has a usable behaviour signal to extract. The same probe **across bodies**
(train on one morphology, test on another) collapses to a mean macro-F1 of 0.16: the behaviour information is
present but **entangled with body shape**, and transfer degrades as the leg-length gap widens. Morphology itself
is separately decodable from the raw embedding at approximately 100 percent (holding under episode-grouped
cross-validation, so it is not frame memorisation), and its leading principal component orders the three bodies
by leg length with no labels supplied. This is the expected and desirable baseline: the encoder plainly sees that
the bodies differ, and removing that difference while keeping the behaviour signal is precisely the job of the
latent action model. **Two caveats define the pilot status of these numbers:** each morphology was recorded in a
single session, so morphology and recording session (lighting, background) are perfectly confounded and cannot be
separated on this data; and the top-8 contact patterns cover only 43 percent of frames because the replayed gait
is a phase-staggered wave rather than a clean tripod, so the within-body figure is measured on the cleaner,
more frequent half of the data and is likely optimistic. The multi-session control needed to attribute the
morphology signal to leg length rather than session is scheduled with the dataset redesign.

![Frozen features encode morphology](fig_morphology_evidence.png)

*Figure 3.2. Morphology is strongly and ordinally encoded in the frozen embedding. (a) A supervised linear
probe reaches ~100 percent and holds under episode-grouped cross-validation. (b) Unsupervised PCA, given no
labels, orders the three bodies by leg length along its leading component. (c) A UMAP projection, shown for
illustration only. Leg length and recording session are confounded in this pilot.*

![Foot-contact decodability within and across bodies](fig_sanity_check.png)

*Figure 3.3. Foot-contact behaviour is decodable within a body (mean macro-F1 0.84) but collapses across bodies
(mean 0.16), and a shuffled-label control sits at the chance floor. The behaviour information is present but
entangled with body shape, the signal the latent action must preserve while making it body-independent.*

**3.7.3  Two methodological findings that shaped the protocol.** First, a representation can *contain* a signal
without that signal *dominating* it, so a single metric can mislead. Behaviour is decodable from the raw
embedding (probe macro-F1 0.84 within a body), yet morphology is the dominant axis of variation (morphology
silhouette +0.0835 against a much weaker behaviour signal, with morphology decodable at ~100 percent). A linear
probe measures presence; the silhouette score and between-class variance measure dominance. Section 3.6.1
therefore requires both, and additionally reports balanced accuracy / macro-F1 because the contact-pattern
classes are strongly imbalanced. Second, an early time-based behaviour label (step index modulo a hand-chosen
gait length) was abandoned: identical commands do not place different bodies in the same pose, so a time index is
not a body-independent behaviour label, and it was found to be an artefact of the chosen trim window rather than a
physical gait state. A pose-based label (foot-contact state, read from force sensors) is used from this point
forward, and Section 3.5 records its residual limitations.

## 3.8  Remaining Work and Next Steps

The two preliminary milestones of Section 3.7 establish that the morphology gap is real and that the frozen encoder
carries a usable, morphology-entangled behaviour signal. This section states, concretely and in order, every step
that remains, so that the scope of the committed work is explicit. Each step names what it produces and the
condition under which it is considered complete.

**Step A: Dataset redesign (inverse-kinematics retargeting).** Build the IK pipeline of Section 3.5.3: take the
shared task-space foot trajectories from the lab's mature expert log (`expert_66k_aug3c_fcontact`), solve `simIK`
separately per morphology to obtain per-body joint commands, and drive each body with its own command while
recording, per timestep, the RGB frame, the joint command a_t, the body pose, and the simulator's binary
foot-contact labels. *Produces:* a dataset in which behaviour is shared across bodies but the command differs per
body. *Gate:* a_t is verified to differ across bodies (variance well above machine precision), removing the pilot's
shared-command limitation of Section 3.5.2.
- *Sub-step A1 (confound control).* Record each morphology across **several sessions** with varied lighting and
  background, so that a probe trained on one session can be tested on another. This is the control needed to
  attribute the morphology signal of Section 3.7.2 to leg length rather than to recording session.
- *Sub-step A2 (behaviours).* Include the walk trajectory; add turn and stop as additional task-space trajectories
  if reference data permits (Section 1.4), since the latent validation of Step D uses more than one behaviour.

**Step B: Regenerate the preliminary analyses on the redesigned dataset.** Re-run the morphology-gap and
encoder-sanity analyses of Section 3.7 on the IK dataset, using the corrected evaluation of Section 3.6.1
(macro-F1 / balanced accuracy for the imbalanced contact classes, episode-grouped cross-validation, and the
multi-session probe from A1). *Produces:* the final baseline values for e_t against which z_t will be compared.
*Gate:* the qualitative conclusions of Section 3.7 hold on the clean dataset; the morphology-versus-session
attribution is resolved.

**Step C: Train the latent action model (ITM + FTM + Motion Decoder).** Train the three modules of Section 3.4
jointly with the combined loss and cross-augmentation, on the two training morphologies (short and long). *Produces:*
the learned latent action z_t and the trained transition and decoding modules. *Gate:* both losses converge and z_t
does not collapse (the Motion Decoder reconstruction stays informative).

**Step D: Latent-space validation (the two-sided criterion).** Probe z_t for behaviour transfer and for morphology
decodability, and cluster z_t by behaviour across bodies. *Gate:* z_t lowers morphology decodability relative to
e_t **and** raises cross-morphology behaviour transfer relative to e_t, measured together (Sections 2.6.4, 3.6.1).

**Step E: The decisive ablation (run early, not last).** Compare a latent-conditioned forward model, F(e_t, z_t),
against a raw-joint-conditioned one, F(e_t, a_t), and an observation-only one, F(e_t, 0), under an identical
encoder, capacity, dataset, and budget (Sections 2.6.3, 3.6.3). *Gate:* whether the latent improves held-out
prediction over the raw command is the central question; this step is scheduled before the transfer experiment so
that a negative answer is discovered early.

**Step F: Transfer to the unseen morphology.** Adapt the pretrained model to the held-out medium body and measure
the number of episodes needed to reach a target reconstruction error, against a from-scratch baseline. *Gate:* the
pretrained model reaches the target with significantly fewer episodes (the sample-efficiency claim of Section
3.6.3), or the result is reported as negative.

**Step G: Analysis, writing, and scoped extensions.** Consolidate results into Chapter 4. Extensions that the
pipeline makes cheap but that are not part of the core claim: an out-of-range (extrapolation) morphology beyond the
0.5× to 1.0× training range, and an explicit cross-embodiment alignment objective in the manner of UniSkill (Section
2.3.2). These are stated as directions, not commitments.

## 3.9  Work Plan

The timeline follows the steps of Section 3.8. The simulation setup and both preliminary milestones (Section 3.7)
are complete; the remaining steps span August to November, with the decisive ablation (Step E) scheduled early so a
negative answer would surface before the transfer experiment.

| Step (§3.8) | Activity | Aug | Sep | Oct | Nov |
|---|---|---|---|---|---|
| (setup) | Simulation setup, morphology variants, vision capture, preliminary milestones (§3.7) | done | | | |
| A | Dataset redesign: IK retargeting, multi-session capture (A1), behaviours (A2) | x | | | |
| B | Regenerate the preliminary analyses on the IK dataset (final e_t baselines) | x | x | | |
| C | Train the latent action model (ITM + FTM + Motion Decoder) | | x | | |
| E | Decisive ablation: latent vs raw-command vs observation-only (run early) | | x | | |
| D | Latent-space validation (two-sided: behaviour up, morphology down) | | x | x | |
| F | Transfer to the held-out medium body; sample-efficiency vs from-scratch | | | x | |
| G | Analysis, thesis writing, scoped extensions | | | x | x |

`[[DECIDE]]` Confirm the calendar with both advisors. The target is to complete the core experiments (through Step
F) by the end of October, leaving November for analysis and writing.

---

# REFERENCES  *(preliminary; refine formatting to the required style)*

[1] D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. "Mastering Diverse Domains through World Models" (DreamerV3).
    arXiv:2301.04104, 2023.
[2] A. Assran, et al. "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning."
    arXiv:2506.09985, 2025.
[3] H. Huang, S. Yenamandra, A. Majumdar, E. Aljalbout, T. Nagarajan, J. Yang, A. Rai, M. Rabbat, L. Fei-Fei,
    J. Wu, T. Wu, F. Meier. "Cross-Embodiment Robot Foundation World Models with Latent Actions" (LAC-WM).
    ICML 2026.
[4] M. H. Danesh, et al. "Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning" (QWM).
    arXiv:2604.08780, 2026.
[5] T. Li, R. Calandra, D. Pathak, Y. Tian, F. Meier, A. Rai. "Planning in Learned Latent Action Spaces for
    Generalizable Legged Locomotion." IEEE RA-L, 2021 (arXiv:2008.11867).
[6] J. Bruce, et al. "Genie: Generative Interactive Environments." ICML 2024 (arXiv:2402.15391).
[7] S. Ye, et al. "Latent Action Pretraining from Videos" (LAPA). ICLR 2025 (arXiv:2410.11758).
[8] A. Liang, et al. "CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations."
    arXiv:2505.04999, 2025.
[9] C. Zhang, et al. "What Do Latent Action Models Actually Learn?" NeurIPS 2025 (arXiv:2506.15691).
[16] D. Ha, J. Schmidhuber. "Recurrent World Models Facilitate Policy Evolution" (World Models). NeurIPS 2018
    (arXiv:1803.10122).
[17] D. Pathak, P. Agrawal, A. A. Efros, T. Darrell. "Curiosity-driven Exploration by Self-supervised Prediction"
    (Intrinsic Curiosity Module). ICML 2017 (arXiv:1705.05363).
[18] B. Baker, et al. "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos." NeurIPS 2022
    (arXiv:2206.11795).
[19] H. Kim, J. Kang, H. Kang, M. Cho, S. J. Kim, Y. Lee. "UniSkill: Imitating Human Videos via Cross-Embodiment
    Skill Representations." CoRL 2025 (arXiv:2505.08787).
[10] A. D. Larsen, T. H. Büscher, T. Chuthong, T. Pairam, H. Bethge, S. N. Gorb, P. Manoonpong.
    "Self-Organized Stick Insect-Like Locomotion under Decentralized Adaptive Neural Control." Advanced Theory
    and Simulations 6, 2300228, 2023.
[11] T. Chuthong, T. H. Büscher, S. N. Gorb, P. Manoonpong. "Insect-Inspired Resilient Machines." Advanced
    Intelligent Systems, 2026.
[12] H. Cruse. "What mechanisms coordinate leg movement in walking arthropods?" Trends in Neurosciences 13, 1990.
[13] R. McN. Alexander, A. S. Jayes. "A dynamic similarity hypothesis for the gaits of quadrupedal mammals."
    Journal of Zoology 201, 1983.
[14] A. Gupta, et al. "MetaMorph: Learning Universal Controllers with Transformers." ICLR 2022.
[15] B. Ai, et al. "Towards Embodiment Scaling Laws in Robot Locomotion." CoRL 2025 (arXiv:2505.05753).

---

# APPENDIX  *(optional candidates)*
- A. Morphology scaling procedure and numerical verification of leg-reach ratios.
- B. Camera/render configuration and the render-environment control (avoiding checkerboard / blank surfaces).
- C. Milestone results obtained so far (morphology-gap check; encoder sanity check). Move these to Ch.4 once
  the proposal becomes the full thesis.
