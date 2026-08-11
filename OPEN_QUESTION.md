# Cross-embodiment plan — status & open questions

> **Role**: What still has to be decided.
>
> Only genuinely open items. When a measurement settles one it moves to `FINDINGS.md` and leaves a single line in the settled table at the bottom, so no result is written out twice.

Living doc. Supersedes the old "argue vs prove / terrain / leg-length" questions,
which are now settled (see bottom). Updated 2026-08-08.

Stage 1 measurements that constrain everything below are in **[FINDINGS.md](FINDINGS.md)**;
this file carries only what is still undecided.

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

**Stage 2 has now been run once, and the premise did not hold on its own (F38).** One shared
trunk across the hexapod and the B1, per-embodiment heads, no cross-embodiment term -- which is
what the source method specifies. The latent came out **33.0% embodiment identity** against 39.6%
gait phase, with embodiment decodable at 1.000. For comparison, `lambda_cross` holds the *body*
share at 0.8-1.2% within the insect family.

Training did pull the two together: silhouette **+0.671 to +0.140**, cluster separation
**4.01x to 0.77x**. That is a large real compression and it is visible in the projection. **But
the latent still separates into two clean clusters in the UMAP and the probe is still 1.000** --
so weight sharing alone gets most of the way and does not finish. The question below is no longer
a prediction; it has a number.

**The mechanism that worked in Stage 1 does not port to Stage 2 as written.** `lambda_cross`
decodes body A's latent against body B's frame supervised by B's command, and it is well defined
only because every insect body walks the same expert episodes: `dataset.py` pairs on
`clip["episode"]`, so at a given timestep two bodies share the intent exactly and differ only in
geometry. **The hexapod and B1 share no episodes.** B1's clips come from MuJoCo rollouts under
different policies, so `self.partners` would be empty and `L_cross` would never be computed.

Without it, Stage 2 gets `lag1_ctrl`'s behaviour: body identity decodable from `z` at 0.638
against 0.470 with the term on, the decoder leaning on `z` rather than the frame (z-gap 11.3x
against x-gap 3.6x), and 1.65x worse reconstruction. **`z` becoming an embodiment code is exactly
what Stage 2 must not allow**, since that code has no entry for a third embodiment.

Pairing does not actually require shared episodes -- it requires knowing that two frames show the
same intent. Two measurable stands-in, both available:

| pair on | needs | cost |
|---|---|---|
| body velocity alone | simulator positions | phase mismatched, so the target command is wrong |
| velocity + gait phase from foot forces | force sensors, at training time only | most accurate; privileged data |
| **velocity + gait phase estimated from the frame** | **nothing extra** | phase wrong ~18% of the time |

Force sensors would be used to **build the pairs**, never as model input, so vision-only inference
is unaffected -- the same standing as the ground-truth commands already used as targets. And the
third row is available regardless: one frame identifies which feet are swinging at 0.815 against a
chance of 0.5 (F31).

**The real risk is not the sensor, it is that mis-paired frames are wrong labels**, not merely
noisy ones. Stage 1's pairing is exact; part of why `L_cross` works may be that exactness. Across
embodiments no pairing can be exact, and what counts as "the same phase" for a six-leg tripod and
a four-leg trot has no physically correct answer -- it is a design decision that has to be stated
and defended. **This is the largest untested risk in Stage 2 and there is currently no plan for
it.**

Practical consequence: report claim A as the result, claim B as a measured limit with its
mechanism, and treat sample efficiency as the transfer claim actually being made.


---

## Q11. What else differs from the source paper, and does it matter? (open)

Read against the paper, the reimplementation matches on the things that define the method --
frozen V-JEPA2, the ITM/FTM/MD decomposition, cross-augmentation and its stated purpose, and,
after the `action_lag` correction, the action's time index. Four differences remain:

| | paper | ours | worth acting on |
|---|---|---|---|
| **action chunking** | actions grouped into **5-step** sequences, stated to improve world-model learning | one step | **yes, and now quantified** -- F33: widening the gap to five steps nearly doubles the reconstruction target's real signal, and combined with dropping the crop it moves the signal-to-noise ratio from 0.24x to 0.89x |
| latent dimension | 512 | 64 | maybe; ours is 8x tighter |
| module size | ITM 47M, FTM 94M | about 5M each | probably not at this data scale |
| behavioural diversity | 3 datasets, 150k trajectories, 22 object categories, a deliberate left-or-right choice in the task, 80 percent failures | one gait, one speed, forward only | this is the F31 constraint, restated |

**One difference removes a risk rather than adding one.** The paper has **no cross-embodiment
pairing term**: the shared latent space emerges from sharing the ITM, FTM and MD weights across
embodiments. So `lambda_cross` is **our addition**, and Stage 2 can follow the paper without
solving the pairing problem in Q0.

**Their setting probably does not need it, and that is the point.** The shortcut we measured is
"recognise which body this is and recall its commands", and it only pays when knowing the body
tells you the command. In our data each body does exactly one thing, so body identity is nearly
the whole answer. In theirs, one embodiment performs thousands of different manipulations across
22 object categories, with a deliberate left-or-right choice and 80 percent failures, so knowing
it is a Franka arm says almost nothing about what to do next -- the shortcut buys little and the
model has to read the scene regardless. Their Motion Decoder is also auxiliary rather than the
system's output, so a shortcut there costs them less than it costs us.

**We therefore cannot claim their method has this problem, and should not.** We have not run in
their regime. What we can say is scoped: LAC-WM is tested across embodiments that differ radically
with behaviourally rich data; applied to **cross-morphology** -- bodies that differ slightly, one
behaviour -- the auxiliary motion loss admits a shortcut that defeats transfer, and `lambda_cross`
closes it. That is a regime the paper does not test, so this is an extension, not a correction.

**The concrete proposal this points to**, to put to the professor rather than decide alone: rebuild
the main experiment with a five-step gap and photometric jitter only, which is the first setting in
which the forward model's target is mostly signal rather than augmentation noise (F33). Cost: the
Motion Decoder outputs 5 x 18 = 90 dimensions instead of 18, every number becomes incomparable with
the runs recorded so far, and the copying shortcut the augmentation was there to block has to be
re-measured rather than assumed away. That is a rebuild of Stage 1's main comparison, so it should
be decided before Stage 2 starts, not during.

**And the paper's transfer is not zero-shot.** Adapting to the unseen embodiment is a three-stage
LoRA finetune on 7,265 trajectories of the target robot. Q0's claim B, tested as zero-shot, is
stricter than what the method claims. The sample-efficiency framing in Step 3 is the comparable one.


---

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


---

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


---

## Q3. 6-leg controller: CSV gait vs policy

`hexapod_v1` uses the **CSV gait** (walks properly, ready). Driving it with the AIRL
policy for "consistency" is **parked**: the AIRL policies aren't faithfully runnable
here — their obs *normalization* config is drifted from the trained weights (tried all
3 candidate obs fields; all give a stationary stance). For the vision dataset the CSV
gait is fully valid (V-JEPA2 sees a hexapod walking either way). **Lean: keep CSV.**


---

## Q4. To confirm / minor

- Metric = **reconstruction-loss sample efficiency (no policy)**.
- Data volume: current clip counts are a start; may scale the command sweeps.
- Writing caveats (Tee): single-step Markov is deliberate; which modules fine-tune on a
  new body; large-model fine-tuning/scaling limitation.


---

## Settled, and where the evidence lives

Each of these was an open question that a measurement closed. The full argument and the numbers
are in `FINDINGS.md` at the finding named; nothing is repeated here.

| | question | what settled it | finding |
|---|---|---|---|
| **Q5** | Does removing the body code from `z` make the decoder read the frame? | Yes, and it does not help: the decoder used the frame 2x more and transfer got 1.21x worse. | F21 |
| **Q6** | Can the decoder be given the view that works? | Yes, and it uses it 7.6x less. Access was never the constraint. | F22 |
| **Q7** | Is the objective the constraint? | Yes. `lambda_cross` is the only intervention of six that improved transfer. | F24 |
| **Q8** | What is the latent for, once the decoder stops needing it? | Gait, and only gait: 88.7% of its variance, with body down to 1.2%. | F26 |
| **Q9** | Does the corrected target make the latent do its job? | It triples the transition's contribution (11% to 36%) and changes transfer not at all. The constraint is the data, not the target. | F29, F31 |
| **Q10** | Is the forward model worth keeping? | Yes. It rolls the world forward 1.2-1.5x better than a frozen world out to ten steps; we had only ever scored it on a task the method does not assign it. | F32 |

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
