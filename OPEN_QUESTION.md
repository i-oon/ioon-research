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

## Q6. Can the decoder be given the view that works? (open, next)

The probe that recovers morphology sees the mean of all 256 patch tokens. The decoder sees those
tokens only through cross-attention with `z` as the query. The smallest change consistent with
F20 is to feed the decoder the mean-pooled embedding directly alongside the attention path, so
the morphology signal is reachable without `z` having to ask for it.

| outcome | reading |
|---|---|
| held-out error drops toward the 0.18 deg linear-mixture ceiling | the access path was the bottleneck |
| unchanged | the decoder can reach it and still will not use it, which points at the objective rather than the architecture |
| training-body error rises | the mean-pooled path is competing with the attention path rather than adding to it |

Cheap: no new data, one architecture flag, one run against `m3d_bracketed`.

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
