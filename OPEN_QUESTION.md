# Cross-embodiment plan — status & open questions

Living doc. Supersedes the old "argue vs prove / terrain / leg-length" questions,
which are now settled (see bottom). Updated 2026-08-08.

Stage 1 measurements that constrain everything below are in **[FINDINGS.md](FINDINGS.md)**;
this file carries only what is still undecided.

## The decision (settled)

Prove — not just argue — that **vision beats proprioception**, via **cross-embodiment
transfer across incomparable joint spaces**. Bodies with different joint counts have
**no shared proprioceptive representation** (18-DOF hexapod vs 12-DOF quadruped can't
be fed to one proprioceptive model, nor aligned), while **vision (pixels) is a common
space**. So a vision latent-action world model can be pretrained across them and
transfer to an unseen body; a proprioceptive one cannot. Claim = "vision *can*,
proprioception *can't*."

## What's DONE

- **B1 quadruped (12-DOF)** — `data/b1_framed/` (14 forward clips from two policies).
  Policy walks in native MuJoCo; we **roll out in MuJoCo → replay kinematically in
  CoppeliaSim** (`sim/rollout_b1_mujoco.py` + `sim/render_b1_replay.py`) because
  CoppeliaSim's engines can't run the policy (MuJoCo won't float the base; Newton/Bullet
  sim2sim gap). Spawn transient cropped.
- **6-leg hexapod (18-DOF)** — `data/ik_walk_8body/` (7 bodies x 30 clips, segment scales varied
  independently), IK retargeting via `sim/collect_ik.py`. Edge clipping 0.0% on every body used.
- **Render consistency — now fixed and verified.** Same spawn point, same travel gate, and the
  cameras were also mismatched: the insect scene used a 0.2618 rad field of view and b1_flat.ttt
  0.4189 rad, 60 percent wider, so B1 frames contained a horizon band the insect frames did not.
  `sim/match_b1_camera.py` copies the insect camera's offset-from-robot, field of view, clipping
  and resolution onto the B1 scene. Background difference on median images fell from **5.03/255
  with 12.8% of pixels off by more than 10** to **1.13/255 with 3.3%**, and outside the robot's
  own footprint to **0.52**, against 0.21 between two insect bodies. Edge contact went from
  10-14% of frames to **0.0%** after `--travel 0.63`. Without this, "embodiment hard to decode
  from `z`" measures the camera.
- Both datasets: `frames` (256², shared vision space) + per-body command + proprioception.
- **Behaviour and speed matched across embodiments.** The insect data is forward walking only, so
  B1's turn, strafe and spin clips are excluded: any turning clip would necessarily be B1 and a
  probe would read behaviour as embodiment. Speed is matched too -- the insect bodies span
  0.00567 to 0.01014 m per frame, and B1 uses **two policies** (2.0 Hz and 1.7 Hz gait, genuinely
  different gaits: 12 against 10 steps per leg, duty 0.52 against 0.61) across seven commanded
  speeds covering the same range, so neither speed nor gait identifies the embodiment.
  14 clips, 1,129 transitions.

## The experiment (test plan)

1. Encode frames with frozen **V-JEPA2** → train **ITM/FTM + per-embodiment Motion
   Decoder** (18-D / 12-D heads) across embodiments. Morphology never told — spec-free.
2. **Validate the latent (two-sided)**: behaviour decodes and transfers across bodies;
   body identity should be hard to decode from `z`. On Stage 1 **only the first half holds** —
   behaviour transfer improves (+0.11 to +0.22 macro-F1 over raw `e_t`) while body identity stays
   **~99% decodable from `z`** across runs differing in data and training length. Nothing in the
   objective removes it: both losses condition on `x_t`, which already carries morphology. Transfer
   to an unseen body works regardless (0.18 with `z` vs 1.67 ablated), so **invariance and
   transferability are separable**. Report both halves; the second is a finding, not a gate.
3. **Cross-embodiment transfer / "loss drop"**: pretrain, adapt to held-out body with
   few clips, measure **reconstruction-loss sample efficiency vs from-scratch**.
4. **Vision-vs-proprioception proof**: the same WM on proprioception **can't** form a
   shared model across incomparable bodies; vision can.
5. **Ablation**: latent-action vs raw-command vs obs-only.

## Q0. What Stage 2 can and cannot claim, given Stage 1 (new, 2026-08-09)

Stage 1 found that the decoder identifies the body from a code in `z` and looks up, rather than
inferring morphology from the frame (FINDINGS F18-F22). Two claims were being run together and
have to be separated, because Stage 1 supports one and predicts the other will fail.

**Claim A — vision forms a shared model across incomparable joint spaces, proprioception cannot.**
Survives, and Stage 1 supports it. One model reconstructs five bodies to 0.5 deg with morphology
never supplied. This is a statement about a shared representation existing, not about
generalisation. An 18-DOF hexapod and a 12-DOF quadruped cannot be fed to one proprioceptive
model at all, so the asymmetry does not depend on transfer succeeding.

**Claim B — that model transfers to an unseen embodiment.** Stage 1 predicts failure. Training on
hexapod + B1 and testing on a 4-leg insect is two training points, which is the configuration F5
and F17 show does not work, and a third embodiment cannot be generated the way extra bodies were.

**Step 3's sample-efficiency framing is not claim B.** Pretrain, fine-tune on N clips of the new
embodiment, compare against from-scratch: the shared backbone carries gait phase and visual
processing, so it can start ahead even when zero-shot fails. Untested and not contradicted.

Practical consequence: report claim A as the result, claim B as a measured limit with its
mechanism, and treat sample efficiency as the transfer claim actually being made.

## Q1. Which cross-embodiment framing? (the main open choice)

- **(A) 6-leg → B1** — feasible **now** with data in hand. Pretrain hexapod, test
  transfer to B1 (or vice-versa). Clean incomparable proof (18-D vs 12-D). 1→1 transfer.
- **(B) Compositional: {6-leg + B1} → 4-leg insect** — *better story.* Train on two
  incomparable topologies (proprioception-can't is baked into the training set), then
  test transfer to a **4-leg stick insect**, which shares *appearance* with the hexapod
  and *leg-count* with B1 → tests whether the model **composes** them. Ablation
  (train 6-leg-only / B1-only / both) shows the composition explicitly.

**Lean:** (B) is the headline if we can produce a 4-leg walker (see Q2); (A) is the
guaranteed-feasible fallback and a good first result. Likely do (A) first, then (B).

## Q2. The 4-leg walker — we build our own (no dependency on yuchen)

We do **not** need yuchen's `cutlegs` policy. The world model only needs 4-leg **frames
+ command**, not that policy's internal obs. The 6-leg CSV gait does **not** propel a
4-leg body (tested: 0.000 m — front-leg removal breaks propulsion), so we need *a*
4-leg controller — and we make it ourselves with a config we own:
- **Retrain via PPO** (`train_ppo.py`, reward = forward velocity) — **no expert demos
  needed** (unlike AIRL), so the missing `expert_cutlegs.csv` is irrelevant. We choose
  the obs and which legs. No config-drift.
- Or a hand-tuned / CPG 4-leg gait.

**Design decision:** likely cut the **front leg pair** (leave middle+hind) so the body
reads clearly as a quadruped → strongest "insect + quadruped → 4-leg insect" composition.

## Q3. 6-leg controller: CSV gait vs policy

`hexapod_v1` uses the **CSV gait** (walks properly, ready). Driving it with the AIRL
policy for "consistency" is **parked**: the AIRL policies aren't faithfully runnable
here — their obs *normalization* config is drifted from the trained weights (tried all
3 candidate obs fields; all give a stationary stance). For the vision dataset the CSV
gait is fully valid (V-JEPA2 sees a hexapod walking either way). **Lean: keep CSV.**

## Q4. To confirm / minor

- Metric = **reconstruction-loss sample efficiency (no policy)**.
- Data volume: current clip counts are a start; may scale the command sweeps.
- Writing caveats (Tee): single-step Markov is deliberate; which modules fine-tune on a
  new body; large-model fine-tuning/scaling limitation.

## Q5. Does removing the body code from `z` make the decoder read the frame? (ANSWERED: yes, and it does not help)

No longer a shot in the dark. `FINDINGS.md` F18 and F19 establish the mechanism this targets:

- `z` splits **64.1 percent gait phase, 11.1 percent body**, so it is doing what it was designed
  for, yet a linear probe recovers the body from it at **0.724** against a 0.200 chance level.
- Crossing the decoder's inputs shows it takes the body **from `z`, not from the frame**: body
  A's frame with body B's latent produces body B's commands to within **3.48 deg**, where the two
  bodies differ by 28.63. The preference strengthens with training.
- From the output side, **0.883 of the mixture weight** sits on a single training body, and the
  segment scales the answer implies are (0.98, 0.98, 0.97) against an actual (0.80, 0.90, 0.90).

So the decoder is running a lookup over five body codes while ignoring a frame that carries leg
lengths in full. A lookup has no entry for an unseen body, which is every failure in F4 to F7.

**Intervention:** `--lambda_adv` puts a gradient-reversal classifier on `z` (`wm/models/adversary.py`),
with `adv_warmup_epochs` ramping it in. Two things were tried first and did not work: rescaling
the motion target (F9) and shrinking the decoder head (F4b, 1.4 to 2.1 times worse).

**What decides it,** in order of how directly each bears on the claim:

| measurement | now | success looks like |
|---|---|---|
| `scripts/swap_pathway.py` | answer follows `z` | answer follows the frame |
| `heldout/motion_zero_x` | not yet measured | larger gap than the control |
| post-hoc probe on frozen `z` | 0.724 | **0.200**, not lower |
| held-out RMSE | 3.57 deg | below 3.0 |

Below-chance probe accuracy is a failure mode, not a success: being wrong 99.8 percent of the
time with five classes needs information, so it means the latent is rotating the code faster
than the classifier tracks it. A 5-epoch smoke run reached 0.002 that way.

**Answer, from `m3d_adv01` against `m3d_bracketed` over seven epochs (FINDINGS F21):**

| | control | adversarial |
|---|---|---|
| held-out error | **0.101** | 0.124 (**1.23x worse**) |
| z-gap | 27.2x | 5.9x |
| **x-gap** | 11.1x | **19.1x** |

The decoder did move onto the frame, by 1.7x on the ablation that measures exactly that, and
transfer still got worse. F20 says why: a ridge probe on mean-pooled `e_t` recovers a held-out
body's segment scales to 0.05, so the information is there and linearly available, but the
decoder reaches the frame only through cross-attention with `z` as the query and never asks for
it. The body code in `z` was a symptom.

**This closes the invariance question.** Forcing invariance neither helps nor is required; what
it does is expose that the decoder cannot use the frame. The open question moved to Q6.

## Q6. Can the decoder be given the view that works? (ANSWERED: yes, and it uses it less)

`--md_head pooled` adds the mean over patch tokens as a zero-initialised residual straight onto
the action, which is the exact view a ridge probe uses to recover a held-out body's segment
scales to 0.05 (FINDINGS F20). Against `m3d_bracketed` over eleven epochs:

| | control | pooled |
|---|---|---|
| held-out error | 0.098 | 0.099 (identical) |
| z-gap | 21.1x | 29.6x |
| **x-gap** | **10.9x** | **1.4x** |

Handed the working view, the decoder relied on the frame **7.6x less** and held transfer level by
leaning harder on `z`. Measured directly on a smoke checkpoint, the residual varies more across
frames of one body (2.80 deg) than across bodies (1.87 deg) -- it tracks gait phase, not leg
length -- and is 1.5 to 1.9 deg against the 28.6 deg that separates two training bodies.

**Five interventions, one worked.** Rescaling the target (F9), shrinking the head (F4b),
stripping the body code from `z` (F21) and handing over the pooled view (F22) all failed;
only more training bodies helped (F16, F17). Capacity, access and latent content are all ruled
out.

## Q7. Is the objective the constraint? (open — and F23 locates where)

**FINDINGS F23 narrows this.** `L_recon` is supposed to make `z` an action by making the next
frame unpredictable without it. Measured: removing `z` costs the forward model **3 to 7 percent**
at every horizon from 1 to 10, in both the two-body and five-body runs, while the Motion Decoder
loses 2,000 to 3,700 percent without it. And with `lambda_recon = lambda_motion = 1.0`, recon
sits at 1.6 against motion's 0.01, so **99 percent of the gradient goes to the term that does not
need `z`**.

The latent is shaped by `L_motion` alone, on one percent of the signal, and `L_motion` is
satisfied by a lookup. That is why no decoder-side change worked.

Two experiments that cost one config value and no new data, neither run:

| change | question |
|---|---|
| `--lambda_motion 100` | given a comparable gradient budget, does `L_motion` still settle for a lookup |
| `--lambda_recon 0` | does dropping a term worth 3 to 7 percent help the latent or hurt it |

The cross-body objective below is the third option and is orthogonal to both.


`L_motion` asks for the right joint command on bodies visible during training. A lookup over
five body codes in `z` satisfies that at lower cost than reading geometry off pixels, by every
route we have opened. Nothing in the loss requires the appearance-to-morphology mapping that
transfer needs, so no architectural change should be expected to produce it.

Candidate changes, none tested:

| change | what it would force |
|---|---|
| decode `z` from body A with the frame of body B, supervised by B's command | the latent cannot carry body identity and the frame must supply it |
| predict the body's segment scales as an auxiliary target | morphology becomes something the loss asks for, not a shortcut it tolerates |
| predict the Cartesian foot trajectory, shared across bodies by construction, and convert with the observed geometry | separates the body-independent intent from the body-specific mapping explicitly |

The first is closest to a control: it uses data we already have, since every body walks the same
expert episodes, so frames and commands are aligned across bodies at each timestep.

---

## Settled / obsolete (was Q1–Q4 in the old version)

- **Argue vs prove** → decided: **prove**, via cross-embodiment (above).
- **Leg amputation (nested, weak proof) vs different body** → chose a genuinely
  different body (B1 quadruped). Amputation reused only as the *4-leg test*, not the proof.
- **Terrain experiment** → dropped (open-loop can't traverse it; poor cost/benefit).
- **Leg-length range (0.5/0.75/1.0 vs 0.7/0.85/1.0)** → moot; leg-length variants are
  now just pretraining diversity, not the core axis.
- **AIRL policy reuse** → parked (config drift; only yuchen's exact obs config +
  normalization would unblock; action bounds + 4-leg=LFRF config already recovered).
