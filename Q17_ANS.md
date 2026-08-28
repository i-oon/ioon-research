## How to actually answer Q17

**The one-sentence version (defensible, citation-proofed):**

> No published method learns a **shared latent action space that transfers across legged robots with different leg counts from video alone**: methods that avoid a kinematic tree/URDF either stay within one robot (Li et al. 2020) or within one leg-count family (QWM, Aug 2026); methods that span leg counts use explicit kinematic retargeting (X-Morph) or a shared task-space coordinate that only exists for manipulation (LAC-WM). This project's gap is showing a joint-space action target *can* cross that boundary — but only with a body-motion term, and MSE alone was found to silently discard the action channel across morphology families.

That's the sentence to put in the doc. Below is what actually earns it, and what to steal from each source.

---

## What to learn / adapt from each paper

**1. Li et al. 2020 — "Planning in Learned Latent Action Spaces"**
This is the paper that most needs a direct citation because it's the closest surface match (hexapod + quadruped, "latent action," "generalizable"). What to *take* from it, not just defend against:
- Their ablation design is genuinely useful — they compare against a **library of experts (LIB)**, a **model-free high-level policy (SAC)**, and **model-free low+high-level (SAC-SAC)**. That's a clean baseline taxonomy you could borrow directly for your own ablation table: "learned latent + planner" vs "discrete primitive library" vs "model-free."
- Their **IK-with-damage comparison** (disabling two legs, showing their learned latent space adapts while IK degrades) is a strong, cheap-to-communicate way to demonstrate the value of *not* hard-coding kinematics. You could run an analogous stress test — degrade one body's joint limits and see if your body-motion term still holds — as a low-cost extra result.
- What NOT to take: their framing that "generalizes to multiple robots" is doing weaker work than it sounds. Use that gap explicitly — it's your strongest rhetorical contrast, because you can cite their own methods section to show it's two separately-trained z-spaces.

**2. X-Morph (2026) — human-motion-to-robot pipeline, quadruped/hexapod/quadruped+arm**
- Useful as a **data-source idea**, not a competitor to your method: they show human motion capture is a viable prior for *diverse* morphologies including hexapods. If you're ever data-starved for one of the two robots, their retargeting-then-RL-tracking pipeline is a legitimate way to bootstrap more training clips — worth flagging as a fallback in `direction_plan.md`'s fallbacks section rather than only a literature contrast.
- Their explicit statement that direct retargeting is "visually plausible yet physically inconsistent" is a good citation for *why* naive kinematic retargeting is nontrivial even when you do have the URDF — it strengthens the argument that going kinematics-free isn't just convenience, it's sidestepping a real failure mode.

**3. QWM (Aug 2026) — morphology-conditioned world model, quadrupedal family**
- Most useful methodologically: it's a **world model that trains policies "in imagination"** (inside the learned dynamics) rather than requiring real rollouts, and it demonstrates zero-shot transfer to unseen morphologies within a family. If your project ever wants to scale beyond 3 leg lengths / 2 embodiments, their scale-invariant conditioning trick is a candidate recipe for extending your FTM to *more* bodies within one leg-count family cheaply, before tackling the harder cross-family jump.
- Also useful as a **credibility check on your v-JEPA2/FTM choice**: it's independent, contemporaneous evidence that "roll out policies inside a learned world model" is a live, competitive strategy in exactly this space — reduces the risk that a reviewer says the world-model approach itself is unusual.

**4. LAC-WM / CD-LAM / CAPE (contrastive term precedent)**
- These are your strongest "steal the technique, don't claim the technique" sources. CAPE's finding that removing the contrastive term causes the predictor to **ignore the action query entirely** is nearly a word-for-word precedent for your own finding (b) — worth citing directly as "this failure mode is known in other action-conditioned settings; what's new here is that it appears specifically across morphology families and is invisible in the MSE loss curve itself." That's the correct, defensible scope for the InfoNCE contribution.
- CD-LAM's three-part decomposition (embodiment-centric reconstruction / action-centric contrastive / latent calibration) is worth skimming as a possible ablation structure if the advisor wants you to further decompose *why* the contrastive term works, beyond "it works."

---

## Net framing for the doc

Structure the answer as three moves, in order:
1. **State the sentence above.**
2. **Pre-empt the four papers** (Li et al., X-Morph, QWM, LAC-WM) in one compact paragraph each — cite them, state precisely why they don't close the gap, in the advisor's own words style ("X does Y, but Z is missing").
3. **Turn two of them into methodological deposits**, not just literature contrast: Li et al.'s ablation taxonomy → your baseline comparisons; QWM's in-imagination training → a possible extension path if the advisor asks "what's next."

That gives Q17 a one-sentence answer that survives scrutiny, plus a paragraph that shows you didn't just defend the gap — you mined the near-misses for usable ideas, which is usually what makes an advisor comfortable signing off on a novelty claim.