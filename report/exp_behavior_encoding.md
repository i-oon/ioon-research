# Experiment: Behaviour (Foot-Contact) Encoding in Frozen e_t

Figure: `report/fig_sanity_check.png` (regenerate: `.venv/bin/python scripts/finished/plot_sanity_check.py`)
All numbers regenerated 2026-07-22 on `data/step0_v2/embeddings.npz`, contact threshold 0.5 N.

## Experimental question

Does frozen `e_t` contain foot-contact (behaviour) information, is it a real signal, and does it
transfer across bodies?

## Setup (shared with the morphology experiment)

- **Input:** frozen V-JEPA2 `e_t`, 1408-d per frame (ViT-g/16, mean-pooled over 256 patches, encoder
  never trained).
- **Model:** logistic regression (linear probe) with standardisation.
- **Metric:** macro-F1 (each class weighted equally, robust to imbalance), chance = 1/8 = 0.125.

## (1) The eight contact classes — exact definition

Contact = foot force > 0.5 N. Each frame's 6-foot pattern is a 6-bit code; the top-8 most frequent
codes are used as classes. Feet: FL, ML, HL (front/mid/hind left), FR, MR, HR (right).

| code | feet planted | n | % of kept |
|---|---|---|---|
| 100010 | ML + HR | 287 | 22.5 |
| 111100 | HL + FR + MR + HR | 207 | 16.2 |
| 000110 | ML + HL | 190 | 14.9 |
| 010100 | HL + MR | 131 | 10.3 |
| 001110 | ML + HL + FR | 125 | 9.8 |
| 000100 | HL | 114 | 8.9 |
| 100011 | FL + ML + HR | 111 | 8.7 |
| 110100 | HL + MR + HR | 111 | 8.7 |

**Coverage: 1276 / 3000 = 43% of all frames.** The other 38 patterns (57% of frames) are dropped as
too rare to train on. This is a direct consequence of the wave gait producing many scattered patterns;
it is the main reason these numbers are a *pilot diagnostic*, and it means the within-body score is
measured only on the cleaner, more frequent half of the data (likely optimistic).

## (2) Episodes and samples per morphology

5 episodes per body, 200 steps each (1000 frames/body before the top-8 filter). Kept frames:

| body | kept frames | episodes present |
|---|---|---|
| long | 433 | 0,1,2,3,4 |
| medium | 470 | 0,1,2,3,4 |
| short | 373 | 0,1,2,3,4 |

## (3) Within-body — episode-grouped CV, mean ± std across folds

Cross-validation folds hold out **whole episodes** (`GroupKFold` on episode id), so no frame from a test
episode is seen in training — this rules out temporal-adjacency leakage between near-duplicate frames.

| body | macro-F1 | per-fold |
|---|---|---|
| long | **0.830 ± 0.177** | 0.80, 0.86, 0.98, 1.00, 0.51 |
| medium | **0.954 ± 0.060** | 0.98, 0.84, 0.97, 0.99, 0.99 |
| short | **0.778 ± 0.105** | 0.70, 0.76, 0.76, 0.69, 0.98 |

**Honesty note:** long and short have high fold-to-fold variance (one long fold drops to 0.51). With
only 5 episodes, a single episode landing in the test fold moves the score a lot. Report the ± , not the
mean alone. The signal is clearly above chance in every fold, but the exact value is not tight.

## (4) Cross-body — two protocols

**Protocol A — train one body, test another (6 ordered pairs):**

| train → test | macro-F1 |
|---|---|
| medium → long | 0.315 |
| long → medium | 0.249 |
| medium → short | 0.134 |
| short → medium | 0.127 |
| long → short | 0.072 |
| short → long | 0.047 |
| **mean** | **0.157** |

**Protocol B — train two bodies, test the held-out third:**

| held out | macro-F1 | note |
|---|---|---|
| medium | 0.430 | interpolation (between the two training bodies) |
| long | 0.358 | extrapolation to longest |
| short | 0.148 | extrapolation to shortest |

Both protocols show the same collapse from ~0.84 within-body. Protocol B adds a finding: the held-out
**medium** (interpolation) is the easiest to predict (0.43) and the held-out **short** (the most extreme
morphology) is the hardest (0.15). Transfer degrades with morphological distance — consistent with
Protocol A, where the long↔short pairs (largest leg-length gap) are worst (0.05–0.07).

## (6) Empirical shuffled-label baseline — mean ± std over 5 seeds

Labels permuted, same episode-grouped CV, long body:

**shuffle = 0.115 ± 0.020** (chance 1/8 = 0.125).

Shuffling collapses the score to chance, confirming the within-body 0.83 is real signal and not an
artefact of the probe or the class imbalance.

## What this supports

`e_t` encodes foot-contact behaviour (within-body well above chance and above the shuffle baseline), but
the behaviour is **entangled with body shape**: it does not transfer across morphologies, and it degrades
with morphological distance. This is the signal `z_t` must **preserve while making it body-independent**;
raising cross-body macro-F1 above the ~0.16 baseline is the Step 1.5 success criterion.

## Status

Pilot diagnostic, not final evaluation:
- top-8 covers only 43% of frames (wave gait, many scattered patterns)
- single session per body
- 5 episodes per body → high fold variance on within-body
- contact threshold 0.5 N
Numbers will shift after the IK re-collection (aligned per-body behaviours, contact labels covering all
frames, more episodes). Report as preliminary.
