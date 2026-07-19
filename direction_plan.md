# Research Direction — Cross-Morphology Locomotion via Latent Action World Models

> 📋 **See also**: `report/audit_2026-07.md` — full project audit (2026-07-17). Long-form English record of
> every finding behind the corrections in this file, including the doc-drift table, LAC-WM's exact
> hyperparameters, the literature/baseline inventory, and the 12-week phased path.
> Thai summary in `PROGRESS.md` §10. **Timeline: Aug–Nov, target October.**

---

## Core Claim
Learn a **morphology-agnostic latent action z_t** from simulation video
that clusters by **behavior** (walk/turn/stop), not by body shape.
Prove it transfers to an unseen morphology without retraining.

---

## Pipeline (confirmed)

- **Encoder**: V-JEPA2 RGB tokenizer — frozen ✓ (confirmed: ViT-g/16, 1B params)
  - per-frame, no temporal mixing inside encoder
  - output: 16×16 = 256 patch tokens per frame, each ∈ ℝ^1408
  - pre-encode offline → save e_t to disk (keep encoder in loop only if using cross-augmentation)

  **V-JEPA2 Encoder — Input / Flow / Output**

  ```
  INPUT
    frame_t  ∈ ℝ^{256×256×3}     raw RGB from sim camera

  PREPROCESSING (before encoder, your code)
    aug1 = transform(frame_t)     random crop + color jitter + flip  → ℝ^{256×256×3}
    aug2 = transform(frame_t)     different random params             → ℝ^{256×256×3}

  PATCH SPLITTING
    256×256 ÷ 16 = 16×16 grid → 256 patches of 16×16×3 pixels each

  LINEAR PROJECTION
    each patch → Linear → ℝ^{1408}
    + positional embedding (3D-RoPE)

  VIT TRANSFORMER (frozen, 1B params)
    256 tokens attend to each other (self-attention across patches)
    each token becomes spatially context-aware

  OUTPUT
    e_t¹  ∈ ℝ^{256×1408}   from aug1  → to ITM + Motion Decoder
    e_t²  ∈ ℝ^{256×1408}   from aug2  → to FTM
    e_{t+1}¹ ∈ ℝ^{256×1408} from aug1 → to ITM + L_recon target
  ```

  **Load code**
  ```python
  from transformers import AutoModel
  encoder = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
  encoder.eval()
  for p in encoder.parameters():
      p.requires_grad = False   # frozen — no gradient flows through
  ```

  **Why frozen**: V-JEPA2 pretrained on 1M hours of internet video (VM22M, 22M videos).
  Already learned motion-relevant features. Fine-tuning would require massive compute and risk losing generality.
  Step 0 verifies these features are useful before committing to training.

- **ITM**: attention-based ✓ (confirmed from LAC-WM Table 4)
  - 4 attention blocks, 16 heads, learned query token q_t
  - in: [e_t, e_{t+1}] (512 tokens total) → out: **z_t ∈ ℝ^64**
  - **CORRECTION**: z_t was previously written as ℝ^512 here — that was a misread of LAC-WM Table 4.
    Table 4's "Latent Dimension = 512" is the ITM/FTM **internal hidden width**, not the latent action.
    LAC-WM §4.2 states separately: *"Both models employ an action embedding dimension of 64."* → z_t ∈ ℝ^64
  - NOT a simple MLP

- **FTM**: attention-based ✓ (confirmed from LAC-WM Table 4)
  - 8 blocks: self-attn(e_t) + self-attn(z_t) + cross-attn(e_t→z_t), 16 heads
  - in: [e_t, z_t] → out: ê_{t+1} ∈ ℝ^1408
  - NOT a simple MLP

- **Motion Decoder**: cross-attn(z_t, x_t) + MLP ✓ (confirmed from LAC-WM)
  - z_t = query, x_t = visual context (keys/values)
  - out: â_t ; L_motion = ||â_t − a_t||²
  - **not used when measuring Phase 1, but KEEP THE WEIGHTS** (corrected 2026-07-19; previously
    written as "discarded after pretraining", which is right for the representation claim and wrong
    for anything downstream). The MD is the only bridge from a latent action back to executable joint
    commands. A Phase 2 policy would emit `z_t` and need exactly this module to run it on a body:
    `policy → z_t → MD → 18 joint targets → robot`. Discarding it would leave the latent unexecutable.
  - This is also the answer to Ajan Blink's week-4 objection ("if the robot needs joint commands
    anyway, why convert to a latent and back?"): the policy learns in the latent space *because that
    part transfers across bodies*, while the MD does the body-specific decoding. The conversion is
    what separates the transferable part from the non-transferable part.

- **Cross-augmentation**: YES ✓ (following LAC-WM)
  - two independent augmentations A1, A2 applied to the frame pair (O_t, O_{t+1}) before encoding
    → (x_t¹, x_{t+1}¹) and (x_t², x_{t+1}²)
  - ITM uses pair 1: z_t¹ = ITM(x_t¹, x_{t+1}¹) ; FTM uses pair 2: ê_{t+1}² = FTM(x_t², z_t¹)
  - why: since z_t is partially supervised by L_recon, ITM could shortcut by smuggling
    x_{t+1}'s content directly into z_t instead of learning the actual action — cross-augmentation
    breaks this because ITM's augmented view of t+1 doesn't match what FTM is scored against
    (confirmed from LAC-WM paper, Section "Cross-Augmentation Inputs")
  - encoder called per single frame (image mode, not clip mode) — must stay in training loop
    since each epoch needs fresh random augmentations before encoding

- **Total loss**: L = λ_recon · L_recon + λ_motion · L_motion

- **Simulator**: CoppeliaSim v4.10 ✓ (corrected — earlier "IsaacSim 5.0" was wrong, see Decisions table)

- **Data**: stick insect, 3 morphologies, third-person RGB + auto-logged joint commands
  - **Action type**: joint position targets ℝ^18 (6 legs × 3 joints)
  - **Episode length**: 1000 steps (~16s at 60Hz physics)
  - **Episodes**: ~100 per morphology per behavior (walk/turn/stop) → ~200,000 training pairs for short+long
  - **Camera**: **single** fixed camera, side view, ~30° elevated, sees all 6 legs clearly
    — 🔴 **DOES NOT EXIST YET.** The CoppeliaSim scene has no vision sensor and no code captures RGB from it.
    This is blocker #1; everything from Step 0 onward is gated on building it.
    - **Multi-angle (3-camera) capture was considered and REJECTED**: (1) walk/turn/stop are whole-body
      behaviors — single-view suffices, and this view already avoids occlusion; (2) V-JEPA2 has **no fusion
      mechanism** for simultaneous views — we'd have to invent one with no counterpart in LAC-WM; (3) it
      **triples the render-confound surface** (see Critical Confound below) — 3× the chance of invalidating
      Step 1.5; (4) sim gives ground-truth joint angles free, so occlusion can be checked without more cameras.
    - Revisit **only** if Step 0/1 demonstrates specific leg motions are systematically occluded — a real
      failure, not a speculative upgrade. Full reasoning: `report/audit_2026-07.md` §5.1.
  - **Data collection policy**: ✅ **IK retargeting** — define each behavior as a Cartesian foot trajectory,
    solve per morphology with `simIK`. Gives per-body-different `a_t` with no training. See rationale below.
  - **Behaviors**: walk / turn / stop — 🔴 **only walk exists** (`ds_loopsm.csv` = 67 rows, one forward cycle;
    AIRL reward is forward-velocity-only). Turn/stop must be built via IK retargeting.
  - **Morphologies**: ✅ built + numerically verified (`sim/env/*.ttt`, via `sim/make_leg_morphology.py`) —
    **short 0.5× / medium 0.75× / long 1.0× (base)**, exact reach ratios on all 6 legs
  - **Train morphologies**: short + long leg
  - **Held-out morphology**: medium leg (Step 2 transfer test)

- **Why IK retargeting, and why `a_t` MUST differ per morphology** (this is load-bearing, not a convenience):
  - The Motion Decoder is `MD(x_t, z_t) → â_t` — it conditions on visual context `x_t`.
  - If every morphology receives **identical commands**, `a_t` is identical per behavior → `L_motion` trivially
    forces `z_t` morphology-agnostic, and `MD` never needs `x_t`. A reviewer says: *"of course your latent
    action is body-independent — you fed every body the same action."* **Circular result.**
  - If `a_t` **differs per body** (what IK gives you), `MD` must use `x_t` to know which body it's looking at →
    a `z_t` carrying pure behavior is an **earned** result.
  - Rejected alternative — AIRL retraining per morphology: normalization bounds in `normalized_env*.py` are
    hand-measured literals per body (nothing parameterized by leg length, no tooling to recompute, and the
    clip line is commented out so wrong bounds fail *silently*); expert data is one animal / one gait / one
    trial and is invalid for a rescaled body; ~1 day GPU per run. Not on the critical path.

---

## Decisions Still Needed

| Block | Status | Note |
|---|---|---|
| z_t dimension k | **64 (LAC-WM §4.2: "action embedding dimension of 64")** | previously recorded as 512 — that was Table 4's ITM/FTM *hidden width*, not the latent action. Ablate smaller/larger k for locomotion. Bonus: 8× smaller latent eases the 2080 Ti compute problem |
| λ_recon, λ_motion | **undecided — and NOT in the paper** | LAC-WM never reports numeric λ values anywhere (nor LR/optimizer/schedule). Start equal, ablate. |
| Stick insect DoF | ✅ **confirmed + in hand** — from Ajan YuChen's `airl-insect-walking` (Medauroidea extradentata, CoppeliaSim) | migrated to `sim/env/`. **All 3 leg-length variants now built and numerically verified** (0.5× / 0.75× / 1.0×) via `sim/make_leg_morphology.py` |
| Data collection policy | ✅ **RESOLVED: IK retargeting** (supersedes the earlier "two-phase / AIRL-per-morphology" plan, which does not survive contact with the repo) | Behaviors = Cartesian foot trajectories → `simIK` per morphology → per-body-different `a_t`, zero training. **Why the old plan died**: (1) the mature `66k_aug3c` checkpoints every script points at **are not in this copy** — only logs proving they once existed; (2) the checkpoints that *do* exist are **34-dim obs** and **no `normalized_env*.py` in the repo produces 34 dims** (base = 36; the module that did was deleted) — they likely won't load without reverse-engineering; (3) normalization bounds are hand-measured literals per body with no tooling to recompute, and normalization is **not clipped**, so a 0.5× leg breaks them *silently*; (4) expert data is **one animal / one gait cycle / one trial** — invalid for a rescaled body; (5) ~1 day GPU per run; (6) **no leg-length precedent exists** in that repo — every variant is leg *removal* or terrain. |
| 🔴 **Camera / RGB capture** | **MISSING — blocker #1** | The CoppeliaSim scene has **no vision sensor**; no code anywhere captures RGB from it (zero `getVisionSensorImg` calls; `.ttt` binaries contain no vision-sensor objects; the lab repo has none either). All V-JEPA2 work to date ran on pre-recorded **B1 quadruped** `.mp4`s from *other* renderers — i.e. **Step 0 has never touched the stick insect.** Everything from Step 0 onward is gated on building this. |
| 🔴 **Turn / Stop behaviors** | **MISSING** | Nothing produces them: `ds_loopsm.csv` = 67 rows of one forward gait cycle; AIRL reward = `discriminator_logit + vx*100` (forward velocity only). Must be built via IK retargeting. Load-bearing — K-means(K=3) in Step 1.5 and the "≥3 behaviors" answer to the ICLR critique both depend on it. |
| Step 2 evaluation metric | **training time reduction** (confirmed Ajan Go) | pretrained FTM reaches same L_recon with fewer medium leg episodes than scratch |
| Baseline exact setup | undecided | separate pipeline per morphology vs. scratch RL |
| LAC-WM source code | not found yet | rejected ICLR 2026, accepted ICML 2026 — no public code yet |
| Simulator/framework | **CoppeliaSim confirmed** (supersedes earlier "IsaacSim 5.0 ✓" — that was wrong) | The lab's actual Medauroidea model runs in CoppeliaSim v4.10, not IsaacSim. Installed and connection-tested locally (`sim/`); GUI mode stable, headless mode currently segfaults on cleanup (unresolved, see `sim/SOURCES.md`) |

---

## Execution Plan

---

### Step -1 — Morphology Gap Check ✅ **PASS**
**Goal**: confirm short leg and long leg actually behave differently under same command

| Task | Send identical joint command to short leg and long leg |
|---|---|
| Expected | long leg drags / overshoots, short leg walks normally |
| Pass | visually distinct behavior → morphology gap is real → proceed |
| Fail | identical behavior → leg length difference too small → adjust morphology parameters |

> P'Hap: if same command produces same behavior, the whole experiment is pointless

**RESULT — PASS** (`sim/step_minus1_morphology_gap.py`, 10s run, `step_minus1_comparison.png`).
Controller = the open-loop gait replay baked into `main_script.py`/`ds_loopsm.csv` — morphology-independent by
construction, so no trained policy was needed.

| Metric | short (0.5×) | long / base (1.0×) |
|---|---|---|
| Forward distance | 3.49 m | 4.77 m |
| Body height std | 0.0192 m | 0.0165 m |
| Foot swing clearance (6 legs) | **consistent** ~0.13–0.16 m | **erratic** 0.05–0.38 m |

Front-left foot shows it most clearly: long/base has sharp swing peaks to ~0.39 m; short stays low with a
visibly different rhythm. Identical commands → qualitatively different gait character, not just a scaled
version of the same motion.

> ⭐ **Step -1 is not merely a sanity check — it is the evidence base for the entire motivation.**
> See "The Motivation Problem" below. These numbers are what justify needing a latent action at all.

**Caveat (honest)**: the PASS is a human read against the plan's own qualitative criterion ("visually distinct
behavior"). The script computes and prints the numbers but asserts no threshold — it is not reproducible as an
automated gate.

---

### Step 0 — Visual Encoder Sanity Check
**Goal**: confirm V-JEPA2 frozen features are meaningful for legged locomotion before any training

## ✅ RESULT — PASS (2026-07-17, walk-only pilot, real stick insect)

*(Previously blocked: the scene had no camera, so all earlier V-JEPA2 work ran on B1 quadruped footage from
other renderers. Camera + recorder built 2026-07-17; this is the first run on the actual subject, so Ajan Go's
gate — "test Visual Encoder first" — is now genuinely addressed.)*

**Data**: 3 morphologies × 3 episodes × 200 steps = **1800 frames**, locked render environment.
`sim/collect_step0.py` → `scripts/step0_encode.py` → `scripts/step0_analyze.py`.
Gait period verified empirically = **exactly 64 steps** (`a_t` repeats bit-exactly at 64/128/192) → phase labels
are exact, not estimated. **`a_t` is bit-identical across every morphology and episode** → any morphology signal
in `e_t` is *purely visual*.

### ✅ Check 1 (THE GATE) — phase IS decodable → **PASS**
| probe | accuracy | chance |
|---|---|---|
| linear probe (random 5-fold) | **85.1% ± 5.6** | 12.5% |
| linear probe (**grouped** CV — whole episode held out, no frame leakage) | **92.7% ± 1.8** | 12.5% |
| k-NN (k=5) | 75.3% ± 8.9 | 12.5% |
| **shuffled-label control** | **12.3%** ≈ chance | 12.5% |

Shuffle control lands exactly on chance → **signal is real, not overfitting**. (Mandatory check: 1408 dims vs
~1440 training samples.) **The ITM has something to extract.**

### 📊 Check 2 (BASELINE, not a gate) — morphology dominates, exactly as predicted
- **morphology probe = 99.9% ± 0.1** (chance 33.3%); shuffle control 34.2% ≈ chance
- **silhouette(`e_t` | morphology) = +0.0835**, between-class var **22.4%** ← Step 1.5 must LOWER
- ~~silhouette(`e_t` | phase) = **−0.0222**, between-class var 7.6% ← Step 1.5 must RAISE~~
  **⚠️ RETRACTED 2026-07-19.** `phase` is `(step % 64) // 8`, an artefact of the hand-chosen 64-step trim,
  not an expert gait annotation (`scripts/step0_encode.py:65`). The number regenerates exactly but measures
  an artificial variable, so it cannot serve as a Step 1.5 target. **Restate this target against
  `contact_8`** (6-bit which-feet-planted) before Step 1.5 is evaluated. See `report/NUMBERS.md` §3.0.

### 🔑 THE KEY FINDING — the phase code is **morphology-entangled**

> ⚠️ **The numbers in this section use the retracted `phase` artefact label.** The *conclusion* survives,
> because the same within/across collapse was later reproduced with the `contact_8` label
> (83.7% within → 41.3% across, `report/NUMBERS.md` §3). But every figure in the table below must be
> regenerated against `contact_8` before it is quoted. Do not present these values.

| | phase accuracy |
|---|---|
| **within** one morphology | long **97.3%** · medium **96.3%** · short **93.8%** |
| **across** morphologies (train on 2, test held-out) | long **39.0%** · medium **34.8%** · short **27.0%** |

Phase is nearly perfectly readable *inside* a body but **does not transfer between bodies** (93–97% → 27–39%).
**Each morphology has its own phase manifold** — visible in the UMAP: three separated islands, phase varying
*within* each island, no global phase structure.

> **This is exactly the gap ITM + cross-augmentation exist to close — now measured.**

#### 🔧 UPDATE (v2, 2026-07-18) — part of that gap was a WEAK LABEL, not the encoder
The 27–39% above used `phase = step mod 64` — a *time* label. But `a_t` is a trimmed CSV loop (not a natural
period, loop seam jumps 14.75°), and identical commands ≠ identical pose across bodies (a short leg plants its
foot earlier). So we re-collected with **foot-force ground truth** (`data/step0_v2`, 3×5×200 = 3000 frames)
and relabelled by **6-bit foot contact** (which feet are planted — a real body-pose label, the same quantity
Ajan YuChen's own notebooks use). Cross-morphology transfer, per label:

| label | within-body | **across-body** | transfer |
|---|---|---|---|
| `step mod 64` (time, old) | 92.5% | 38.4% | 0.42 |
| **6-bit foot contact (new)** | 85.1% | **55.2%** | **0.65** |

→ **+17 points** just by labelling true pose instead of time. So the encoder is *better* than v1 implied;
part of the "entanglement" was measurement error. **But 55% ≠ 100%**, so a real morphology-entangled residual
remains — that is the genuine gap for Step 1.5.
**Revised Step 1.5 target: raise cross-morphology contact-transfer from 55% upward, morphology probe below 99%.**
Use **6-bit contact as the primary label** henceforth, not step-mod-64. (`scripts/step0_analyze_v2.py`,
`step0_v2_labels.png`.)
🔴 Caveat: contact label still derives from one animal's replay with a non-smooth loop — must be recollected
once a proper per-morphology expert (lab CPG / AIRL) exists. Raw forces are saved, so other labels can be
tried without re-collecting.

### ⚠️ Methodological finding — silhouette alone would have given the WRONG answer
`silhouette(e_t | phase) = −0.0222` says *"no phase signal"*. The probe says **85–93%**. Both are right:
- **silhouette measures DOMINANCE** (is it the main axis of variation?) → phase is not.
- **a probe measures PRESENCE** (is it there and linearly extractable?) → phase is.

Euclidean geometry in 1408-d is swamped by non-phase variance. Check 3 (noise floor) confirms: same-phase pairs
40.23 vs different-phase 44.93 — only **1.12×**, i.e. phase contributes almost nothing to raw distance.
**QWM (App. F-E) reports silhouette only** — we must report **both**, or we would wrongly call a present signal
absent. Applies to Step 1.5.

### ⚠️ Step 0's criteria were REWRITTEN (2026-07-17) — the original ones were logically wrong

**The original spec said**: *Pass = "morphology doesn't dominate raw `e_t`"; Fail = "morphology dominates →
sim domain gap severe → partial fine-tune last 2 V-JEPA2 blocks"*, with pass criterion
*"silhouette by behavior > silhouette by morphology"*.

**Why that was wrong:**
1. **A 0.5× leg genuinely LOOKS different from a 1.0× leg.** An encoder that *failed* to separate them would be
   blind to real visual content — that would be a **worse** encoder, not a better one.
2. **If raw `e_t` were already morphology-agnostic, the ITM would be unnecessary** — the whole pipeline exists
   precisely because it is not. Removing morphology is **ITM + cross-augmentation's job, not the encoder's**.
3. **The criterion would fail by construction.** Leg length is a large, static, global visual difference; gait
   phase is a subtle, local, dynamic one. On raw embeddings morphology wins essentially always.
4. **The fail action didn't follow.** "Morphology dominates" ≠ "sim domain gap" — different phenomena.
   Fine-tuning cannot fix the former, and shouldn't: we *want* the encoder to see the legs.

**Step 0 is about INFORMATION PRESENCE, not INVARIANCE.** Ajan Go's actual gate was that the encoder must
separate robot (High Relation) from background (Low Relation) — not that it be morphology-blind.

| | Question | Pass condition |
|---|---|---|
| **Check 1** | **Does `e_t` carry gait-phase information?** (the signal ITM must extract) | phase is **decodable** from `e_t` — linear probe / k-NN above chance; silhouette by phase-bin > noise floor |
| **Check 2** | **Does `e_t` separate the morphologies?** | **Expected YES — this is NOT a failure.** Record it as the **baseline** (see below). |
| **Check 3** | **Noise floor** — frozen/standing robot control | `e_t` should be near-constant. Quantifies how much `e_t` variation is real signal vs encoder noise. **Free to produce: just don't actuate the joints.** |
| ~~old Check 3~~ | ~~Temporal similarity heatmap~~ | ❌ **ABANDONED — see below** |

### 💡 Check 2 is the "before" of a before/after result — not a gate

```
Step 0    silhouette(e_t | morphology)  = HIGH   <- encoder plainly sees leg length      [baseline]
Step 1.5  silhouette(z_t | morphology)  = LOW    <- ITM abstracted it away               [the result]
          silhouette(z_t | contact_8)   = HIGH   <- ...while keeping behavior
                                                    (was "phase"; retracted, see Check 2)
```
**The delta between these is the contribution.** Same metric QWM uses (App. F-E: silhouette + between/within
class variance decomposition) — they only ever measured the morphology-negative side; measuring **both** sides
is our differentiator. So Step 0 must *quantify* morphology separation, not merely tolerate it.

### The ONE real failure condition
**`e_t` carries no phase information at all** → the ITM has nothing to extract → genuinely fatal.
*Only then* does the response become: partial fine-tune of the last 2 V-JEPA2 blocks, or revisit camera
framing/resolution (recall the 16×16-patch caveat vs thin legs). We have seen this failure mode for real: on the
B1 footage, render style so dominated `e_t` that behavior signal was undetectable.

**Check 3 — ABANDONED (tested and failed on 3 different backgrounds)**

The idea: per-patch cosine similarity between consecutive frames; expect background (static) → high similarity,
legs (moving) → low. Intended to answer Ajan Go's "High Relation vs Low Relation" question.

Result: **no reliable signal on any background tried.** The predicted positive relationship never appeared:

| Background | Video | Result | Cause |
|---|---|---|---|
| Checkerboard | `forward_walk.mp4` | r = **−0.16**, p = 7.6e-24 | aliasing — pixels change without real motion |
| Blank white | `removebg_forward_walk.mp4` | r = **−0.20**, p = 4.7e-37 | **empty patches fluctuate the MOST** — known ViT artifact (blank tokens repurposed as internal compute space) |
| Thin grid | `play-step-0_realtime.mp4` | r = **−0.006**, p = 0.70 | negative bias removed, but noise floor remains — no positive signal |

**Conclusion: the problem is the per-patch method itself, not the background.** Dropped as primary evidence.
Scripts retained for reference: `scripts/temporal_similarity_{heatmap,quantified,correlation}.py`.
The planned negative control (stop-clip → flat heatmap) was **never implemented** — so there is no evidence
ruling out that this line was noise from the start.

→ **Ajan Go's High/Low Relation question is now answered by Check 1/2 instead** (whole-frame `e_t`, which
averages over 256 patches and doesn't inherit this per-patch noise floor).

**Requirements for the re-run** (learned the hard way):
- **Quantitative metrics, not eyeballing.** `scripts/umap_domain_check.py` currently has **zero** — its
  "3 non-overlapping clusters" conclusion is a visual read of a *stochastic* UMAP projection, and UMAP is known
  to manufacture apparent separation. Report **silhouette score + between/within-class variance** (QWM App. F-E
  methodology); use UMAP only as an illustration, never as evidence.
- Implement the **frozen-robot negative control** (noise floor).
- **Report, don't gate, the morphology separation** — it is the baseline for Step 1.5.

**Step 0 pilot scope (walk-only)** — see "Data collection policy" in the Decisions table:
with only the walk gait available, "behavior" collapses to **gait phase**. That is still a real and sufficient
test of Check 1: *does `e_t` encode where in the stride we are, rather than which body it is?* Phase labels come
free — the gait is a deterministic 63-row CSV loop, so phase is recoverable from the step index (verify the
period empirically against the recorded `a_t`, which is identical across morphologies by construction).

| Pass | phase decodable from `e_t` above the frozen-control noise floor → ITM has something to work with → proceed to Step 1 |
|---|---|
| Fail | phase not decodable → partial fine-tune last 2 V-JEPA2 blocks, and/or revisit camera framing/resolution |
| Record (not a gate) | silhouette(`e_t` \| morphology) — the **baseline** that Step 1.5's `z_t` must beat |

> Ajan Go: test Visual Encoder first, do not proceed to Step 1 until this passes — must show the encoder
> separates robot features (High Relation) from background (Low Relation)

---

## 🔴 CRITICAL CONFOUND — Render-Style Dominance (threatens Step 1.5's validity)

**The finding** (`scripts/umap_domain_check.py`, logged in `PROGRESS.md §5`): three videos of the **same
behavior** (walking), rendered by three different setups (white bg / IsaacSim grid / MuJoCo checkerboard),
produced **three completely non-overlapping UMAP clusters** of whole-frame `e_t`.

**What it means**: raw frozen V-JEPA2 `e_t` is currently more sensitive to **rendering style** (background,
lighting, engine) than to **behavior**. This is expected for a pretrained encoder with no cross-augmentation —
and the V-JEPA2 paper offers no help here: **VideoMix22M contains zero simulated data**, and the paper never
studies rendering-domain gap. Our finding is unexplained by, but not contradicted by, the paper.

**Why it is dangerous**: if camera / lighting / background differ *at all* between morphology recording
sessions, then **Step 1.5 measures which session a clip came from, not morphology vs. behavior.** The result
would look clean and be meaningless. This confound is invisible unless controlled for up front.

**Mitigation — mandatory, at data-collection time:**
1. **Lock the render environment**: identical camera pose, lighting, and background across **every** morphology
   and **every** behavior session. Vary *only* the robot's legs and its motion. Nothing else.
2. **Background choice** — avoid both empirically-found failure modes: **no checkerboard** (aliasing → fake
   motion signal) and **no blank/flat surface** (ViT register-token noise — blank patches fluctuate *most*).
   Prefer a matte, mildly-textured, non-repeating surface.
3. **Gate before Step 1.5**: encode N frames from each morphology session and run the domain-UMAP. Clusters
   **must now overlap**. If they still separate by session, the environment is not locked → data is invalid.
4. Cross-augmentation is designed to suppress exactly this kind of nuisance — but note the caveat below: body
   shape is *real content* that survives crop/color-jitter/flip, so augmentation is **not** a substitute for
   controlling the environment.

> Corroborated independently by `deep_research.md`'s own construct-validity critique: *"Latent space analysis
> must show locomotion-relevant structure, not visual artifacts."* Written for the abandoned direction, still
> bites this one.

---

## 🎯 THE MOTIVATION PROBLEM — why do we need a latent action at all?

**Three independent sources converge on the same objection. It is currently unanswered.**

1. **Ajan Blink, Week 4** (recorded in `feedbacks/feedback_ajan_go.md:21-23`): *"if the policy and robot
   ultimately need joint-space commands, why convert to a latent/frame space at all, adding a converter back?"*
   — **raised, never answered, still open.**
2. **LAC-WM's actual premise**: its embodiments have **genuinely disjoint action spaces** — 10D Franka EE /
   20D bimanual humanoid EE / **138D** human-hand keypoints / 25D BFA. Its EAC-WM baseline is *architecturally
   forced* into per-embodiment action encoders **because of that heterogeneity**. **Our 3 variants share an
   identical 18-dim joint space and identical DOF.** That motivation evaporates — and so does the baseline's
   pathology (our natural EAC-WM analog would just use *one shared* encoder).
3. **`deep_research.md` CP-002**: explicit morphology-conditioning may suffice for **interpolation within a
   family**; implicit/latent approaches are motivated for **extrapolation**. We are interpolation-only
   (medium sits between short and long — Ajan Blink made this exact correction).

**A reviewer will ask: "why do you need a latent action if every body takes the same 18-dim command?"**

### The answer — reframe the motivation
Not **action-space heterogeneity** (we have none) but **dynamics heterogeneity**: identical joint commands
produce *materially different motion* depending on leg length. **Step -1 already proved this** — same command,
3.49 m vs 4.77 m, and swing clearance 0.13–0.16 m (consistent) vs 0.05–0.38 m (erratic).

So the latent action's job here is **not** to reconcile differing action dimensionality — it is to unify the
**effective dynamics mapping** from identical joint commands to different resulting motion/appearance across
leg lengths, kept physically grounded by the motion-decoding loss so it can't collapse into a no-op identity.

**Be explicit in the writeup that this is a reframing, not what LAC-WM tested.**

### The decisive experiment (promoted from "optional baseline" to load-bearing)
**Latent-conditioned FDM vs. raw-joint-conditioned FDM (single shared 18-dim encoder).**
This is EAC-WM ported *honestly* to our setting. **If the latent doesn't beat raw joints, the thesis is
hollow.** Run it early — knowing this in month 1 is worth far more than in month 3.

### Also open: Ajan Blink wants real extrapolation
He explicitly corrected short+long→medium as *"just Interpolation."* Currently **no out-of-range morphology
exists**. `sim/make_leg_morphology.py` makes this nearly free — generating a **1.25×** (or 0.35×) 4th variant
answers him with one command. Cheap, high-value.

---

### Step 1 — Train Phase 1 Pipeline
**Goal**: train ITM + FTM + Motion Decoder on short + long leg

| Task | Train on short + long leg data, cross-augmentation on, LoRA off |
|---|---|
| Monitor | L_recon and L_motion both decreasing over training |
| Sanity check mid-training | sample z_t every 10k steps → UMAP should show emerging structure |
| Pass criterion | L_recon converges, L_motion < threshold → proceed to Step 1.5 |
| Fail | loss not converging → check λ weighting, learning rate, data pipeline |

---

### Step 1.5 — Latent Space Validation
**Goal**: prove z_t is morphology-agnostic before testing transfer

| Task | Collect z_t from short + long + medium leg × 3 behaviors |
|---|---|
| Check 1 | UMAP colored by behavior → 3 clusters (walk / turn / stop) visible |
| Check 2 | UMAP colored by morphology → no separation between short / long / medium |
| Check 3 | K-means (K=3) → cluster labels match behavior labels (quantitative) |
| Pass | clusters by behavior, not morphology → proceed to Step 2 |
| Fail | clusters by morphology → fallback to **UniSkill** (see below — *not* HiLAM, which was the wrong fallback) |

> Note: failure mode = z_t encoding body-shape visual features instead of motion.
> In LAC-WM this was viewpoint clustering. In our work it would be morphology clustering.

---

### Step 2 — Transfer to Unseen Morphology
**Goal**: prove pretrained World Model reduces data needed for medium leg

| Task | Fine-tune ITM + FTM on N medium leg episodes using LoRA rank 2 |
|---|---|
| Condition A | pretrained FTM + N episodes |
| Condition B | scratch FTM + N episodes (baseline) |
| Vary N | 5 / 10 / 20 / 50 / 100 episodes |
| Metric | L_recon on held-out medium leg test set |
| Pass | pretrained reaches same L_recon as scratch with significantly fewer episodes |
| Fail | no gap between pretrained and scratch → z_t did not transfer → revisit Step 1.5 |

> Ajan Go: main claim = "World Model ลด training time อย่างชัดเจน"
> This is interpolation (medium leg is between short and long) — not extrapolation

---

### Step 3 (Future) — Extrapolation
**Goal**: test morphology outside training range

> Out of pre-proposal scope. Good future direction if Steps 1-2 succeed.

---

## Phase 2 Design Sketch (out of thesis scope, but it constrains Phase 1)

Phase 2 is the eventual use: a policy that makes a new body walk. It is not part of this thesis, but
several Phase 1 decisions are only correct or incorrect relative to it, so the target is recorded here.

### What the execution loop requires

```
policy --> z_t --> Motion Decoder --> 18 joint targets --> robot
                   ^ the module Phase 1 was going to throw away
```

### What transfers and what does not (settled empirically, not assumed)

| Component | Role | Transfers across bodies? | Evidence |
|---|---|---|---|
| `z_t` + ITM | "what is being done" (behaviour) | **the thesis bets yes** | to be tested in Step 1.5 |
| FTM | dynamics in embedding space | partially, fine-tune | Step 2 |
| **Motion Decoder** | `z_t` to joint commands, body-specific | **no** | needs `a_t`, which the new body may not supply |
| **reward / physics heads** | latent to reward | **no** | force→velocity is R²=+0.926 within a body but −0.33 to −5.23 across bodies (PROGRESS.md 10.14) |

The shape that falls out: **a shared behaviour latent, with body-specific encoder and decoder heads.**
This is close to what L3P arrived at from a different direction (frozen backbone, per-robot heads),
which is mild evidence the decomposition is the right one. Our version differs in having a world model
and a latent action inferred by an inverse model, neither of which L3P has.

### Three candidate Phase 2 architectures

| | Method | Needs a reward model? | Main risk |
|---|---|---|---|
| **A** | Dreamer-style imagination RL: policy trained purely inside FTM | **yes** | reward is not readable from our latent (PROGRESS.md 10.14), and the tracking camera hides world-frame progress |
| **B** | Planning in embedding space, V-JEPA2-AC style: sample candidate `z`, roll out with FTM, pick the one minimising distance to a goal embedding | **no** | locomotion is cyclic, so "goal state" is awkward to define |
| **C** | Use `z_t` or `e_t` as a pretrained feature space for ordinary RL on the real environment | no, reward comes from the environment | least novel, but most robust |

Current preference: **B is the most interesting and sidesteps the reward problem** (and V-JEPA 2-AC
already demonstrated this exact architecture on real hardware). **C is the safe fallback.** A is the
riskiest given what we now know about reward readability.

### Phase 1 decisions that follow from this

1. **Keep the Motion Decoder weights.** Corrected above.
2. **Keep logging `a_t` for the held-out morphology.** Already done. Needed to test whether the MD
   generalises rather than assuming it does.
3. **Keep logging world-frame head position.** Already done. It is the only reward label available,
   since the tracking camera removes world-frame progress from the image.
4. **New experiment worth adding to Step 2** (cheap, data already collected):
   *does the Motion Decoder generalise across bodies?* Train MD on short + long, then ask it to decode
   `z_t` into joint commands for the medium body.
   - If it generalises, the "new body needs only video" claim survives into Phase 2.
   - If it does not, we learn exactly how much action data a new body requires, which is a useful
     number in its own right.

---

## If Step 1.5 Fails — Fallback

### ❌ ~~HiLAM~~ — WRONG FALLBACK, do not use for this failure mode

Original plan was: freeze ITM → dynamic-chunk `z_t` sequences → skill-level `z^h`, hoping `z^h` clusters by
behavior more cleanly than flat `z_t`. **After reading the paper (`doc/2603.05815v1`), this is a mismatch.**

- HiLAM solves **temporal abstraction** ("existing LAMs... focus on short-horizon frame transitions... capture
  low-level motion while overlooking longer-term temporal structure") — **not embodiment invariance**.
- Its chunking operator is a boundary rule over feature dissimilarity between **temporally-adjacent tokens
  within a single video**. There is **no cross-embodiment alignment objective anywhere in it** — no mechanism
  to disentangle nuisance (body) from behavior, no contrastive/alignment term across agents.
- Therefore: if `z_t` already encodes morphology strongly, chunking **pools existing features** and would build
  a *separate* skill hierarchy per body — **inheriting and reifying** the morphology clustering, not fixing it.
- Its experiments are 100% LIBERO tabletop manipulation. **No locomotion. No code released.**

**HiLAM is only the right fallback if the failure mode is "z_t captures only short-horizon kinematics and
misses longer behavioral structure" — a different problem than the one we fear.**

### ✅ UniSkill — the correct fallback

**UniSkill** (Kim et al. 2025, CoRL) — *"Imitating Human Videos via Cross-Embodiment Skill Representations"*.
- Explicitly targets **cross-embodiment skill representation** — exactly the failure mode of Step 1.5.
- Telling detail: **HiLAM itself uses UniSkill's IDM/FDM as its frozen submodules.** The cross-embodiment
  property HiLAM borrows comes from UniSkill; HiLAM's own contribution (hierarchical chunking) is orthogonal.
- → If `z_t` clusters by morphology, go to the paper that solves *that*, not the one built on top of it.

**Secondary option worth considering**: **DiLA** (Zhang et al. 2026, *Disentangled Latent Action World Models*)
— content/structure disentanglement, aimed at keeping body-specific visual features out of the behavior latent.

**Action**: read UniSkill before Step 1.5 runs, so the fallback is ready rather than discovered under pressure.

---

## Baseline
- **What we beat**: training from scratch per morphology (no transfer)
- **Metric**: sample efficiency on medium leg — with vs. without pretrained FTM
- No existing locomotion cross-morphology baseline → comparison is transfer vs. no-transfer
- LAC-WM baseline (EAC-WM) is manipulation-only, cannot port directly

---

## Notes from ICLR Reviews
- **V-JEPA2 pixel decoder**: not included in V-JEPA2 — must be trained separately if needed for pixel-space output.
  L_recon = ||ê_{t+1} − e_{t+1}||² is computed in *embedding* space → no pixel decoder required for training.
- **Training scale**: LAC-WM used 64 H200 GPUs × 4 days for 3 manipulation datasets (confirmed App. A.5).
  Our 3-morphology locomotion setting is far smaller — reasonable to train on a single node.
- **Why ICLR rejected**: weak evaluation (1 task, 1 baseline), not because the core method is wrong.
  Professors will likely raise same concern → plan for ≥2 baselines and ≥3 behaviors.
- **EAC-WM degrades with more embodiments**; LAC-WM improves. This is our key supporting evidence.

## Lab Resources
- Stick insect model: confirmed — Ajan YuChen's `airl-insect-walking` repo, CoppeliaSim model, migrated to `sim/`
- Data collection policy: ask lab if scripted controller already exists for the model
- P'Beam's work may connect to this in future

## Open Questions
- Fine-tune last 2 V-JEPA2 blocks if Step 0 shows sim gap?
  - Note: V-JEPA2 paper only ablates **fully frozen vs. fully unfrozen** — partial/last-N-block fine-tuning is
    *not* tested anywhere in it. Reasonable extrapolation, but no empirical backing to cite.
- k = 64 (LAC-WM §4.2) or ablate for locomotion?
- λ values? ablate from equal weights (paper gives no numbers — see Decisions table)
- ~~Baseline: EAC-WM analog~~ → **PROMOTED out of open questions. This is now the decisive experiment, not an
  optional baseline.** See "The Motivation Problem" section above.
