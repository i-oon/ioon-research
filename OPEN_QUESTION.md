# Cross-embodiment plan — status & open questions

Living doc. Supersedes the old "argue vs prove / terrain / leg-length" questions,
which are now settled (see bottom). Updated 2026-08-03.

## The decision (settled)

Prove — not just argue — that **vision beats proprioception**, via **cross-embodiment
transfer across incomparable joint spaces**. Bodies with different joint counts have
**no shared proprioceptive representation** (18-DOF hexapod vs 12-DOF quadruped can't
be fed to one proprioceptive model, nor aligned), while **vision (pixels) is a common
space**. So a vision latent-action world model can be pretrained across them and
transfer to an unseen body; a proprioceptive one cannot. Claim = "vision *can*,
proprioception *can't*."

## What's DONE

- **B1 quadruped (12-DOF)** — `data/b1_v1/` (8 clips: fwd 0.2–0.5, turn, spin, strafe).
  Policy walks in native MuJoCo; we **roll out in MuJoCo → replay kinematically in
  CoppeliaSim** (`sim/rollout_b1_mujoco.py` + `sim/render_b1_replay.py`) because
  CoppeliaSim's engines can't run the policy (MuJoCo won't float the base; Newton/Bullet
  sim2sim gap). Spawn transient cropped.
- **6-leg hexapod (18-DOF)** — `data/hexapod_v1/` (24 clips = long/medium/short × 8),
  CSV gait via `sim/collect_step0.py`.
- **Render consistency (critical)** — B1 scene is built FROM an insect scene
  (`sim/build_b1_scene.py`), so both embodiments share the **identical gray floor +
  lighting + elevation-40 side viewpoint**. Only body/behavior differs → no
  render-style confound. Verified side-by-side.
- Both datasets: `frames` (256², shared vision space) + per-body command + proprioception.

## The experiment (test plan)

1. Encode frames with frozen **V-JEPA2** → train **ITM/FTM + per-embodiment Motion
   Decoder** (18-D / 12-D heads) across embodiments. Morphology never told — spec-free.
2. **Validate latent is morphology-invariant** (two-sided): behavior decodes/transfers
   across bodies; body identity is hard to decode from `z`.
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
