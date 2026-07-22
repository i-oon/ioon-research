# Experiment: Frozen V-JEPA2 Features Encode Morphology / Session Identity

Figure: `report/fig_morphology_evidence.png` (regenerate: `.venv/bin/python scripts/plot_morphology_evidence.py`)

## Experimental question

How strongly does the frozen visual representation `e_t` encode which body produced the frame, and is
that signal an ordinal, morphology-consistent axis or an arbitrary class separation?

> Scope note. The companion question "does `e_t` contain locomotion information?" is a *separate*
> experiment answered by the foot-contact decodability test (`fig_sanity_check.png`). This experiment
> is only about how strongly body identity is encoded. Keeping them apart matters: a representation can
> carry locomotion signal and body signal at the same time, and the thesis needs both measured
> independently.

## Setup

**Bodies.** Three *Medauroidea extradentata* variants in CoppeliaSim, identical in topology and in their
18-dimensional joint action space, differing only in leg length: long (1.0x), medium (0.75x), short (0.5x).

**Data.** `data/step0_v2/` — 5 episodes per body, 200 steps per episode at 20 Hz, one RGB frame per step.
Total 3 x 5 x 200 = 3000 frames. All three bodies are driven by the same bit-identical joint-command
sequence, so nothing in the *command* distinguishes them; any signal in `e_t` comes from what the camera
sees.

**Representation `e_t`.** Each 256x256 RGB frame is encoded by the frozen V-JEPA2 ViT-g/16 encoder
(`facebook/vjepa2-vitg-fpc64-256`), used as a per-frame image encoder via the frame-duplication trick
(each frame is duplicated into a 2-frame tubelet). This yields 256 patch tokens of dimension 1408, which
are mean-pooled over the 256 patches to a single 1408-d whole-frame vector. The encoder is never
fine-tuned. (Encoder wrapper: `scripts/vjepa2_encoder.py`; pooling: `scripts/step0_encode.py`.)

**Label.** Body identity in {long, medium, short}. For the ordinal test, mapped to the leg-length scale
{1.0, 0.75, 0.5}.

## Three tests, in increasing strength

**(a) Supervised probe — is body identity linearly present?**
A logistic-regression classifier is trained to predict body identity from `e_t`, evaluated two ways:
- standard 5-fold cross-validation, and
- `GroupKFold` grouped by episode (each fold holds out whole episodes, so no frame from a test episode
  is ever seen in training). This rules out the classifier memorising specific near-duplicate frames.

**(b) Unsupervised PCA — does the representation self-organise by leg length?**
Principal component analysis is fit on the mean-centred `e_t` with **no labels**. The frames are then
projected onto PC1 and coloured by body *after the fact*. The test is whether PC1 already orders the
three bodies monotonically (short < medium < long). Because PCA never sees the labels, a correct ordering
means body identity is a *dominant axis of variation* in `e_t`, not merely a decodable one.

**(c) UMAP — illustration only.**
A 2-D UMAP projection (`n_neighbors=30`, `min_dist=0.1`) coloured by body, shown for intuition. It is
explicitly *not* evidence: UMAP geometry changes with seed, neighbourhood size, and metric, and inter-
cluster distances are not quantitatively meaningful. The proof is (a) and (b); UMAP is the picture.

## Controls

- **Shuffle / chance baseline.** Chance for 3 balanced classes is 33.3%; the probe is reported against it.
- **Episode-grouped CV** (in (a)) controls for frame-position / temporal-adjacency leakage.
- **Unsupervised ordering** (in (b)) controls for the possibility that supervision alone manufactures the
  separation — PCA has no labels to exploit.

## Result

- (a) Probe: **~100%** morphology accuracy (chance 33.3%), and it stays ~100% under episode-grouped CV.
- (b) PCA: **PC1 orders short < medium < long monotonically**, with no labels used.
- (c) UMAP: three visibly separated clusters (illustration).

**Conclusion:** body appearance is not just decodable from the frozen visual representation, it is a
dominant, ordinally-structured component of it. This is the baseline the latent action `z_t` must
*reduce* (Step 1.5 target: lower morphology decodability in `z_t` while keeping behaviour decodability).

## The confound stated plainly (why the title says "Morphology / Session Identity")

Each of the three bodies was recorded in a **single session**, so morphology and recording session
(lighting, background, camera realisation) are perfectly correlated in this data. Consequently:

- The 100% probe and the PC1 ordering are **consistent with leg length**, but a per-session rendering
  difference would produce the same result. No analysis on this dataset can separate the two.
- The claim this experiment supports is therefore the weaker, honest one: `e_t` strongly encodes
  **body-or-session identity**. That is still exactly what `z_t` must remove, so the Step 1.5 target is
  unaffected by which of the two it is.

**Required follow-up to attribute the signal to leg length specifically:** record each body across
several sessions with varied lighting and background, then test whether a probe trained on session A of
each body still classifies session B. If it does, the signal is morphology; if it collapses, it was
session. This needs new data and is scheduled with the IK re-collection round. (Logged in
`report/NUMBERS.md` 3.3.)

## Status

Pilot diagnostic, not final evaluation. Single session per body; exact probe value drifts 99.4-100%
with classifier settings, so report as "~100%" rather than a false-precision decimal.
