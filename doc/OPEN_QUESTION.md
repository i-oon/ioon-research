# Cross-embodiment plan — status & open questions

> **Role**: What still has to be decided.
>
> Only genuinely open items. When a measurement settles one it moves to `FINDINGS.md` and leaves a single line in the settled table at the bottom, so no result is written out twice.

Living doc. Supersedes the old "argue vs prove / terrain / leg-length" questions,
which are now settled (see bottom). Updated 2026-08-22.

Stage 1 measurements that constrain everything below are in **[FINDINGS.md](FINDINGS.md)**;
this file carries only what is still undecided.

## Q12. Which bodies belong in the dataset at all? (new, 2026-08-11 — blocking)

Measured in **F42**: two of the nine bodies in `data/ik_walk_8body` do not walk — one moves 0.057 m
in an episode, the other **walks backwards** — and two more crab sideways 2 to 6 times more than
any sound body. Every Stage 2 run globs the whole directory, so about a fifth of the hexapod
gradient went to a robot that does not locomote.

The cause is the reach limit: a two-link leg cannot get closer to its shoulder than
`|femur − tibia|`, and the closest commanded target is at 92.5 mm. Ratio alone is harmless —
bodies at 1.04–1.10 with dead zones of 12–26 mm walk normally.

Open, in the order they have to be decided:

1. ~~**Exclude the two non-walking bodies.**~~ **Done** — `EXCLUDED_BODIES` in
   `wm/data/dataset.py`, applied wherever clips are globbed. Stage 2's hexapod pairs drop from
   15,755 to 12,285. The reruns are the remaining work.
2. **Keep the two 94.6 mm veering bodies — decided, keep.** They walk; the gait differs but it is
   locomotion, not a collapse. Not raised in the current deck. Revisit only if there is time to
   retrain, and note that dropping them would leave three bodies all at ratio 0.83, which ties
   femur to tibia perfectly and destroys the very coverage slide 8 is about.
3. **How to get femur/tibia diversity without leaving the sound range.** The coxa is the answer:
   it positions the shoulder without entering `|femur − tibia|`, and it is behaviourally almost
   free — ARI +0.038 with the gait split, and a 40% coxa change leaves contact patterns agreeing
   at 0.984. Rescaling the foot trajectory per body would work geometrically and **must not be
   done**: `lambda_cross` is well defined only because every body walks identical expert episodes,
   so per-body targets turn a shared intent into a wrong label.
4. ~~**Whether ratio > 1 is admissible at all.**~~ **Resolved, already logged**:
   `sim/scene/make_leg_morphology.py`'s docstring states it directly -- "femur longer than tibia
   inverts the animal's own proportion, so a body above 1.0 is a robot morphology, not a stick
   insect." Ratio > 1 is not geometrically broken (bodies at 1.04-1.10 walk normally, per the
   dead-zone measurement in the same docstring), it is just not a stick insect anymore. If the
   thesis frames this as stick insects specifically, the honest range is ratio ≤ 1.

**Resolved 2026-08-12.** Items 1 and 3 are done and `stage2_clean` has been trained and measured
on two seeds — see F43. The data questions are closed; what they uncovered is not:

**Q13. RESOLVED 2026-08-12: drop it.** The cross-embodiment variance decomposition is not
under-sampled, it is built on a phase label too coarse to be one. Stance fraction takes **8
distinct values** across both embodiments, dominated by 0.5, so quantile edges collapse: asking
for 3 bins gives 2, asking for 4 gives 2 (identical numbers), asking for 6 gives 3. The grid was
never 2 x 6 x 6 = 72 cells; it was 24 to 36. The embodiment share therefore reads **32.0% at three
bins and 12.0% at six**, same checkpoint, same data -- a 2.7x swing from a parameter that was
supposed to be cosmetic.

**F38's headline 33.0% came from this measurement.** Replace it everywhere with the probe (0.994 /
0.992 across seeds) and the identity ablation (1.03x / 1.04x against a random control), which
reproduce to three decimals and say something stronger anyway: the identity is fully present and
nothing uses it.

Stage 1's `z_body_share` is not affected -- insect bodies walk identical expert episodes, so its
grid can use the timestep directly instead of inventing a shared phase label.

*Original question below, kept for the reasoning.*

**Q13 (as asked). Is the variance decomposition salvageable, or should it be dropped?**

`two_way` balances its grid to the smallest cell, which holds six latents, so the whole
measurement rests on 72 points. Two seeds of one config give **12.0% and 6.7%** for the embodiment
share. F38's headline 33.0% rested on the same 72 points and is in the deck.

- `--bins 3` doubles the latents per cell. Does that make the seeds agree? One command, decides
  whether the measurement is under-sampled or unusable.
- If it stays unstable, every claim moves to the probe (0.994 / 0.992) and the ablation
  (1.03x / 1.04x), which reproduce to three decimals. That is a stronger claim anyway — presence
  plus non-use, rather than a share of variance.
- The same question applies to `z_body_share`'s Stage 1 numbers, which use the same machinery.

**Numbers stay as they are until the clean retrain.** Deliberate: the deck keeps its current
figures rather than being patched twice. What is known is the *direction*, and it is favourable
everywhere it has been checked — the veering bodies were making our own claims look worse, not
better:

| measured on | with the veering bodies | sound bodies only |
|---|---|---|
| body share in `z`, control | 11.3% | **5.8%** |
| body share in `z`, cross loss | 1.2% | **0.2%** |
| gait phase share, cross loss | 88.6% | **94.7%** |
| held-out error, `tib_cross` | 27.8 deg, R² −3.16 | **11–13 deg, R² −0.42 to −1.07** |

So cleaning the data should *strengthen* Stage 1's claims and only Stage 2's absolute numbers are
at risk. Re-measure after the retrain rather than editing figures now.

**Loose end to settle at the same time**: `z_content.py` reports the control's body share as 8.8%
and `z_body_share.py` reports 11.3% for the same checkpoint, while both agree on 1.2% for the
cross-loss run. Two of our own scripts disagree by 28% on a figure that is in the deck. Trace it
when the numbers are being redone, not before.

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
and defended. ~~**This is the largest untested risk in Stage 2 and there is currently no plan for
it.**~~

**MEASURED 2026-08-14, and the answer is that none of the three rows works on current data (F45).**
The risk above is no longer untested. Per-leg contact was the most promising signal -- no shared
period needed, corner legs correspond anatomically -- and it fails in a specific, informative way:

| label | overlap | hexapod frames pairable | intent, hexapod | intent, b1 |
|---|---|---|---|---|
| `n_feet_down` | 0.572 | 98.9% | 0.913 | **0.998** |
| `diagonal` | 0.711 | 100% | **0.918** | 0.605 |
| `corner_pattern`, 16-way | **0.240** | **33.8%** | 0.630 | 0.524 |

`intent` = matched-pair over random-pair command distance within one body; 1.0 means the label
says nothing. The fine label means something on both robots but pairs only a third of the hexapod;
the coarse labels pair everything and mean nothing on one side. **Coverage and meaning trade
directly**, because coarsening is what destroys the meaning. The B1 spends 84.6% of its time in
the two trot diagonals while the hexapod spreads over all sixteen patterns, nine of them
hexapod-only.

Velocity-alone, the first row of the table above, is worse still and needed no new measurement:
the hexapod walks **one speed, forward only**, so velocity has no variation to pair *on* from the
insect side at all.

**So the plan is no longer "pick a pairing and defend it".** Either (a) accept that Stage 2
follows the paper without a cross term -- which Q11 notes is what the source method actually does,
making `lambda_cross` our addition rather than a missing piece -- or (b) broaden behavioural
coverage first, since the overlap failure is partly the one-gait-one-speed constraint (F31)
rather than a fact about hexapods and quadrupeds. Option (b) is the AMP-dataset question already
open in Q11, and F45 is now a second, independent reason to take it seriously.

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

## Q14. Does behavioural overlap make a channel shareable? (open — three runs deciding it)

The lever list of 2026-08-15 has been worked through and only one item is still standing:
**make the two robots' behaviour distributions overlap**. Shared supervision is blocked by F45 (no
usable frame pairing), architecture was measured to sharpen per-robot codes rather than share them
(F43, F46), three invariance methods moved nothing (F44), and leg-removal bodies read as the body
they were cut from (F47).

**Overlap in speed alone was tested and did not work.** Five matched speeds gave cross-embodiment
readouts of -4.16 and -5.60; more diversity gave the trunk more to partition by (F57, F60).

**Overlap across three behaviours now exists** and the untrained answer is still no: on the frozen
encoder, forward transfers at **+0.36 +/- 0.10** and lateral and yaw sit at zero (F76). But the
frozen encoder is the *before* condition, and forward speed itself reads **0.31 frozen against
0.85-0.92 trained** (F66) -- so a frozen zero does not decide the question (F77).

**What is open**, in the order it gets answered:

1. **Does training on yaw make it transfer?** Three arms running: control, body head forward-only,
   body head forward+yaw. The middle arm is what makes the third attributable to the channel rather
   than to the new dataset.
2. **Does F66's 0.85-0.92 survive the frame-rate fix?** It was measured across the F74 mismatch.
   Arm 2 settles it, and this is the number the deck currently claims.
3. **Is twelve behaviours enough to resolve effects of this size?** Held out by condition, about
   four test behaviours remain and the spreads run +/- 0.2 to 1.3. If the arms disagree weakly the
   answer may be power rather than substance.
4. **Should the yaw length scale be the stance radius rather than hip height?** Physically the
   moment arm of a turn is where the feet meet the ground, and the two scales differ 4.4x in the
   ratio between the robots. It does not change what transfers -- an affine rescale cancels against
   a standardised target -- but it does change how much the channel identifies the robot, 0.637
   against 0.571 (F77). Switching means re-solving the four `--spin` levels first, since the
   collection is matched on the height version.
5. **Is lateral permanently out of the target?** It fails the robot gate at 0.68 even with the
   frame corrected, and half the B1 clips carry a per-policy lateral artefact (F79, F80). Excluded
   for now; the exclusion is a measurement, not a principle.

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

**Update 2026-08-14 (F47): (B) was built, but the body chosen does not test composition.**
The 4-leg is the *base* insect with the middle legs removed, so its geometry is `c10f10t10`'s --
a training body's -- and its commands are that body's corner columns bit-identically. The latent
sits 0.578 from the base body's against a chance of 0.981, so the model reads it as the base body
at that phase and barely registers the missing legs. Leg count is the only novel axis; the
compositional claim needs at least two.

**Which axes a held-out embodiment can be novel on**, and where the current one stands:

| axis | hexapod -> B1 | hexapod -> 4-leg, as built | fix |
|---|---|---|---|
| action dimensionality | 18 vs 12 | 18 vs 12, **novel** | -- |
| leg count | 6 vs 4 | 6 vs 4, **novel** | -- |
| segment geometry | n/a | **in distribution** | remove legs from a *held-out* body |
| gait / dynamics | wave vs trot | unchanged insect wave | needs a 4-leg controller (Q2) |
| appearance | insect vs quadruped | still a stick insect | a second quadruped, new assets |

**Cheapest correct fix, one collection run and no new tooling**: ghost-remove the middle legs from
`c08f09t09`, already withheld from Stage 2 training, so geometry and leg count are both unseen.
Only middle-loss walks (front-loss tips, hind-loss rears, F44), so the variant is forced, and the
body must be rendered before collecting -- a geometry change can break a gait that worked on the
base scene.

**DONE 2026-08-14 (F48).** `data/ik_4leg_c08f09t09_clean10`, 10 clips from a 30-episode sweep.
The margin is **unchanged**: 2.85x against the base body's 2.86x (1.91 +/- 0.08 deg against a
random backbone's 5.45 +/- 0.16). Geometry and leg count are now both novel and the claim holds.
**Rows 1-3 of the table above are satisfied; rows 4 and 5 -- gait and appearance -- are not.**

**Leg-loss itself is not the problem** and should not be abandoned: it produces a genuinely
incomparable 12-D action space against the hexapod's 18-D, which is the asymmetry the thesis rests
on. What was wrong is the *geometry* it was built from, and that is a one-line change to which
scene the collector loads.

**A stronger version, if there is time**: make the held-out body quadruped-like in *behaviour* as
well as topology, so it shares appearance with the insect and gait with the B1 -- which is what
"composition" was supposed to mean. That needs a 4-leg controller rather than the unchanged
six-leg IK gait, which is Q2's open item. Strongest of all would be a second real quadruped, but
that is new assets and a new policy, not a collection run.


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
| **Q17** | Does the 4-leg body test a new embodiment? | No. The latent places it 0.578 from the body it was cut from, against a chance level of 0.981. It tests a new action space; the B1 held out entirely is the real test. | F47 |
| **Q15** | Does anything transfer to a genuinely different robot? | Yes, and all of it travels through `z`: 1.28x on a held-out B1, dropping to 0.98x -- random weights -- when the latent is zeroed. The decoder's use of the frame carries nothing. | F50 |
| **Q16** | Can the forward model be made to work on a new robot? | Not frozen (0.57-0.71x), and coverage does not fix it (5-8%). But **one target clip clears break-even and nine clear every horizon tested**, about 7x fewer clips than from cold. The claim is cheap adaptation, not zero-shot transfer. | F51, F52 |

---

## Settled / obsolete (was Q1–Q4 in the old version)

- **Argue vs prove** → decided: **prove**, via cross-embodiment (above).
- **How to word the proprioception claim** → settled 2026-08-16: **not** "proprioception cannot do
  this". Morphology-agnostic proprioceptive control exists (joints as a token set over the
  kinematic graph), so that sentence is refutable. The defensible form is that those methods must
  be handed the **kinematic tree** and a camera has to be handed nothing. Four places in the deck
  and FINDINGS were corrected; references still need verifying before they are cited.
- **Leg amputation (nested, weak proof) vs different body** → chose a genuinely
  different body (B1 quadruped). Amputation reused only as the *4-leg test*, not the proof.
- **Terrain experiment** → dropped (open-loop can't traverse it; poor cost/benefit).
- **Leg-length range (0.5/0.75/1.0 vs 0.7/0.85/1.0)** → moot; leg-length variants are
  now just pretraining diversity, not the core axis.
- **AIRL policy reuse** → parked (config drift; only yuchen's exact obs config +
  normalization would unblock; action bounds + 4-leg=LFRF config already recovered).
