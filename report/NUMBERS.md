# Numbers ledger

Provenance for every quantitative claim that appears in `presentation_v2.md` and `proposal_draft.md`.

**Decision, 2026-07-19:** do not patch individual figures now. Freeze the project direction first, then
regenerate every number in one clean run so the whole set shares one provenance. This file records what
that run has to produce and which current figures must not be presented until it happens.

Status values:
- `OK` — recomputed from data in this repo and it matched.
- `ORPHANED` — cited in the docs, but no data in this repo reproduces it.
- `UNCHECKED` — not yet verified against source data.

---

## 1. ORPHANED — do not present until regenerated

These are cited in `PROGRESS.md:230`, `sim/SOURCES.md:159-161`, `report/presentation_v2.md` Slide 5, and
`report/proposal_draft.md:438`. The `.npz` files that produced them are not in the repo.

| Claim | Documented | Recomputed from repo data | Verdict |
|---|---|---|---|
| long distance | 4.125 ± 0.434 m | 4.404 ± 0.187 m | does not match |
| medium distance | 3.562 ± 0.015 m | 3.569 ± 0.010 m | matches |
| short distance | 2.646 ± 0.002 m | 2.729 ± 0.011 m | does not match |
| long-leg bimodality, "lands on 4.479 or 3.593" | — | neither value occurs; min episode is 4.032 | unsupported |
| swing clearance, short 0.13–0.16 m / long 0.05–0.38 m | — | from the superseded 2-variant single-episode run; no medium value exists | unsupported |

Likely cause: an earlier collection under different settings, plausibly a different `--warmup` in
`collect_step0.py`, which shifts the start position and therefore net displacement. Unconfirmed.

**Conclusion is unaffected.** Worst long episode (4.032 m) still clears best short episode (2.746 m), by a
wider margin than the documented figures gave. Step -1 still passes. Only the figures are wrong.

Already removed from the deck: the swing-clearance numbers.
Still present in the deck and still wrong: the distance table and the bimodality note on Slide 5.

## 2. OK — recomputed 2026-07-19, reproduce with the command shown

| Claim | Value | Source |
|---|---|---|
| per-episode long distances | 4.5046 / 4.5077 / 4.0323 / 4.5165 / 4.4602 | `data/step0_v2/long_ep{0..4}.npz`, key `head` |
| per-episode medium distances | 3.5875 / 3.5572 / 3.5620 / 3.5712 / 3.5678 | `data/step0_v2/medium_ep*.npz` |
| per-episode short distances | 2.7146 / 2.7209 / 2.7358 / 2.7256 / 2.7459 | `data/step0_v2/short_ep*.npz` |
| v1 agreement (3 episodes) | long 4.498 / 4.567 / 4.536 | `data/step0/` — rules out a v1-vs-v2 artefact |
| mean speeds | 0.440 / 0.357 / 0.273 m/s | distance ÷ 10 s (200 steps @ 20 Hz) |
| leg-length scaling exponent | 0.689 | log-log fit over the three bodies |
| Mann-Whitney U, all three pairs | U=25.0, p=0.0079 | `scipy.stats.mannwhitneyu`, two-sided, n=5 vs 5 |
| Cliff's δ, all three pairs | +1.00 | complete separation; p=0.0079 is the floor at n=5 |

### 2.1 Metric: report path length **and** net displacement

The script currently computes only net straight-line displacement of `/head` in the xy plane,
`norm(p_end - p_start)` (`sim/diagnostics/step_minus1_morphology_gap.py:54`). That measure is reduced by curvature,
and every episode curves because the open-loop gait has no heading correction.

Reporting both, plus their ratio, separates locomotion from steering:

| Body | Path length | Net displacement | Straightness (net/path) | Net drift, mean abs (max) |
|---|---|---|---|---|
| long | 5.217 ± 0.069 (cv 1.3%) | 4.404 ± 0.187 (cv 4.2%) | 0.845 ± 0.046 | 4.5° (14.2°) |
| medium | 4.149 ± 0.019 (cv 0.4%) | 3.569 ± 0.010 (cv 0.3%) | 0.860 ± 0.004 | 5.1° (9.6°) |
| short | 3.228 ± 0.011 (cv 0.3%) | 2.729 ± 0.011 (cv 0.4%) | 0.845 ± 0.002 | 3.4° (6.0°) |

**Path length separates by morphology; straightness does not.** Straightness is 0.845 / 0.860 / 0.845,
essentially constant. Since drift is small, that 15% gap between path and net is mostly the gait's own
side-to-side body oscillation, which is set by the command sequence and so is shared across bodies.
That is what lets the locomotion result be stated without steering contaminating it.

Path length is also lower-variance (cv 1.3% vs 4.2% on the long body). Scaling exponent is 0.688 on
path length against 0.689 on net displacement, so the headline conclusion does not depend on the choice.

**The apparent outlier was a metric artefact.** `long_ep2` records the lowest net displacement (4.032 m)
and the *highest* path length (5.341 m) of its group. It veered 14.2° off heading against 0.3–5.6° for
the other four. It is a steering event, not a slow episode, and must not be dropped.

### 2.2 Corrected: how heading drift is measured

An earlier version of this table reported drift of 39.8° / 12.7° / 19.3° and described every episode as
curving. **Those figures were wrong**, produced by comparing the *instantaneous* heading over the first
10 steps against the last 10 steps. Both estimates sit inside the gait's side-to-side wobble, which
inflated the result by roughly 3–5x (`long_ep2` read 74.9° against a true 14.2°).

Drift is now the angle of the end point relative to the initial heading, in `drift_deg()` of
`scripts/finished/plot_step_minus1.py`. The walks are close to straight; plotting the trajectory panel at equal
aspect makes this visible, and a stretched y-axis makes near-straight walks look like violent swerving.

```
python3 -c "
import numpy as np, glob
for b in ['long','medium','short']:
    d=[float(np.linalg.norm(np.load(f)['head'][-1,:2]-np.load(f)['head'][0,:2]))
       for f in sorted(glob.glob(f'data/step0_v2/{b}_ep*.npz'))]
    print(b, np.round(d,4), 'mean', round(np.mean(d),4), 'sd', round(np.std(d),4))"
```

Note the deck's scaling exponent was 0.7–0.8, corrected to 0.65 against the documented figures, and should
become **0.689** once the distance table is regenerated. It has been wrong twice; check it last, after the
distances are settled.

## 3. Step 0 analysis — checked 2026-07-19, mixed result

Both scripts were re-run against the repo data. Note they read different datasets: `step0_analyze.py`
uses `data/step0/` (3 episodes/body, n=1800) and `step0_analyze_v2.py` uses `data/step0_v2/`
(5 episodes/body, n=3000). Commands: `.venv/bin/python scripts/_archive/step0_analyze.py` and `..._v2.py`.

| Claim | Documented | Reproduced | Status |
|---|---|---|---|
| silhouette, morphology | +0.0835 | +0.0835 | `OK` exact |
| morphology probe (v1) | 99.9% | 99.9% ± 0.1 | `OK` exact |
| morphology probe (v2, 5 ep) | — | 99.6% | `OK` |
| silhouette, phase | -0.0222 | -0.0222 | `INVALID` — see 3.0 |
| phase probe | 92.7%, chance 12.5% | 85.1% ± 5.6 | `INVALID` — see 3.0 |
| cross-body transfer, contact label | 55.2% | 41.3% (`contact_8` across) | `ORPHANED` |
| cross-body transfer, time label | 38.4% | not computed | script no longer produces it |
| contact-pattern agreement across bodies | 16–36% | not produced by either script | `ORPHANED` |
| force→velocity, within body | R² = +0.926 | **no code in the repo computes this** | `ORPHANED` |
| force→velocity, across bodies | R² = -0.33 to -5.23 | **no code in the repo computes this** | `ORPHANED` |

`grep -rn "r2_score" --include=*.py .` returns nothing. The force→velocity regression was never committed.
Those two figures are currently the empirical backbone of the "why not proprioception" argument on
Slide 10 of the deck and must not be presented until the analysis is rewritten and re-run.

Full v2 output for reference:

```
contact_8    classes=9  chance=11.1%   within=83.7%   across=41.3%   transfer-ratio=0.49
n_support    classes=5  chance=20.0%   within=72.6%   across=25.5%   transfer-ratio=0.35
morphology probe = 99.6%
```

### 3.0 `INVALID` — reproduces exactly, but the label is an artefact

A number can reproduce perfectly and still mean nothing. The phase label is defined in
`scripts/_archive/step0_encode.py:65` as `phase = (step % 64) // 8`. It is not an expert gait annotation. 64 is
the length of the replayed command loop and 8 is an arbitrary division of it, so `phase_bin` measures
position within a hand-chosen trim window, not a physically meaningful gait state.

It was deliberately removed from the v2 pipeline. `step0_analyze.py` (v1) still runs on `data/step0/`,
which was encoded before the removal, so the field is still present and the script still reports it.
That is the only reason these numbers regenerate.

Consequences:

- `silhouette(phase) = -0.0222` and `phase probe = 85.1%` are measurements of an artificial variable and
  must not be quoted anywhere.
- **`direction_plan.md` sets a Step 1.5 target of "KEEP/RAISE silhouette(phase)".** That target is
  defined against this artefact and has to be restated against `contact_8` before Step 1.5 can be
  evaluated at all.
- v1's Check 1 and Check 3 both key off `phase_bin`, so both are invalid for the same reason. Check 3's
  "NO clear phase signal" verdict is therefore not evidence of anything either.

Reproducibility and validity are separate audits. Section 2 and section 3 checked reproducibility only.

### 3.1 `n_support` throws away which-foot information

`n_support` counts how many feet are planted (0..6) without recording *which* feet. Different foot
configurations that happen to share a count collapse to the same label, so it cannot distinguish poses
that differ in which legs bear load, which is exactly the behavioural distinction the study needs.

The observed distribution confirms it is degenerate: 5 classes with sizes from 3 to 1204, meaning nearly
every frame carries the same value. Drop it from the label set.

`contact_8` keeps the which-foot information (6-bit code over which feet are planted), so it is the
behaviour label currently worth reporting.

**Gait note (corrected 2026-07-21):** an earlier version of this section described the gait as a tripod
with "two alternating stances." That is wrong. Measured on both the pilot (@0.5N) and the mature expert
`expert_66k_aug3c_fcontact.csv` (using the sim's own binary contact), the canonical tripod grouping
A=FL,HL,MR / B=FR,HR,ML shows within-set correlation ~-0.05 (a tripod needs strongly positive: legs in
one tripod must move together) and clean-tripod frames only ~4%. The gait is a phase-staggered / wave-like
pattern, not a tripod. The expert shows the same structure (front-left duty 29% vs pilot 24%), so this is
a property of this Medauroidea model, not an artefact of our replay or leg scaling. Avoid the word
"tripod" anywhere in the write-up.

### 3.2 Two problems the re-run surfaced that are not stale-number problems

**Severe class imbalance makes raw accuracy the wrong metric.** sklearn warned that the least populated
class has 3 members against `n_splits=5`. `contact_8` spans 118–1605 members over 9 classes; `n_support`
spans 3–1204 over 5. Raw accuracy on distributions like these is close to uninformative, because a
majority-class predictor scores well without learning anything. The within/across transfer comparison
should be restated in balanced accuracy or per-class F1. This changes how Step 0's result is phrased, and
the pass/fail thresholds in `direction_plan.md` need restating to match.

**Check 3 prints a negative result.** The v1 noise-floor check reports same-phase distance 40.23 ± 13.42
against different-phase 44.93 ± 14.44, a ratio of 1.12x, and concludes "NO clear phase signal." Check 3 is
recorded as ABANDONED in `direction_plan.md`, but the script still runs it and still prints a null finding.
Either remove it or address why it disagrees with the probe results on the same embeddings.

## 3.3 Morphology signal: proven decodable, NOT proven to be leg length

Figure `report/fig_morphology_evidence.png`, from `scripts/figures/plot_morphology_evidence.py`. Numbers
regenerated 2026-07-21 on `data/step0_v2/embeddings.npz`.

| Claim | Value | Status |
|---|---|---|
| morphology probe, 5-fold CV | 100.0% (chance 33.3%) | `OK` — supersedes the old 99.9% (that was v1, 3 episodes) |
| morphology probe, grouped by episode | 100.0% | `OK` — rules out frame memorisation |
| PC1 orders short < medium < long | monotonic, unsupervised | `OK` — PCA never sees labels |

**The confound that no analysis on this data can remove.** Each of the three morphologies is a single
recording session, so morphology and session (lighting, background) are perfectly correlated. The 100%
probe and the PC1 ordering are consistent with leg length, but a per-session rendering difference would
produce the same result. Do not claim "the encoder sees leg length" without the experiment below.

**Required experiment (needs new data, cannot run now):** record each body across **several sessions**
with varied lighting and background. If morphology stays decodable across sessions — i.e. a probe trained
on session A of each body still classifies session B — the signal is leg length, not session. This is
the multi-environment control and it belongs in the re-run.

Retired framing: an earlier version of panel (b) trained a regressor on long+short and "predicted" the
held-out medium at 0.75. Dropped, because a constant predictor of the training mean also lands at 0.75
(the mean of 1.0 and 0.5). The unsupervised PCA ordering replaces it as the honest strong result.

## 4. What the clean re-run must produce

1. Re-collect Step -1 with the parameters recorded in the script itself, 5 episodes per body, and keep
   the `.npz` files in the repo. The orphaning happened because the raw arrays were not retained.
2. Report **speed** as the headline quantity, with distance secondary. Speed is duration-independent.
3. Decide whether the metric stays net xy displacement of `/head` or becomes path length of the trunk.
   Net displacement under-counts any curvature, and the open-loop gait has no heading correction.
4. Recompute swing clearance **per swing cycle** rather than as whole-episode `max - min`, across all
   three bodies and 5 episodes, or leave it out.
5. Re-test bimodality on the new sample. Do not carry the claim forward on the old numbers.
6. Refit the scaling exponent last.
7. Write and commit the force→velocity regression. It does not exist in the repo, yet two of its outputs
   are cited as evidence.
8. Decide the classification metric **before** recollecting. If balanced accuracy replaces raw accuracy,
   the Step 0 and Step 1.5 thresholds in `direction_plan.md` have to be restated in the same terms.
9. Resolve Check 3: either delete it from `step0_analyze.py` or explain its null result.
10. Re-run both analysis scripts and update section 3.

## 5. Standing rule

Every number in a document should be regenerable by a command in this file. Anything that is not gets
marked `ORPHANED` and does not go in front of an advisor.
