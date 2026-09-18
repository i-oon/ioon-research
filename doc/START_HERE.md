```
You are resuming a long research project mid-stream. Read files in this order, then follow the working rules below. Do NOT start any experiment until you've read the state and confirmed it back to me.

═══ IF YOU ARE ON com7, READ THIS FIRST — MID-MIGRATION, 2026-09-18 ═══
Work moved here from aria-desktop because that machine had a hardware fault (GPU-touching job hit
`torch.AcceleratorError`, kernel log showed `NVRM: Xid 79: GPU has fallen off the bus` then
`Xid 154: Node Reboot Required` — the physical PC lost power and shut down, not a clean shutdown).
Suspected root cause: the CPU heatsink is seated but not clamped/tightened, so poor thermal contact
under load trips the platform's thermal-shutdown protection, which cuts power abruptly and takes
the GPU down with it as a symptom. aria-desktop must not run GPU or sustained CPU work again until
that's physically fixed — this is not a rule that applies to com7 itself.

**Transfer channel:** com7 has no SSH access from aria-desktop (unlike BIAS). Code moves via git
(push there, pull here). Large binaries not in git (checkpoints under `wm/runs/`, gitignored) move
via Google Drive only — no automated channel for those.

**This session has none of aria-desktop's accumulated Claude memory** (~60 files of learned
conventions and standing rules under `~/.claude/projects/.../memory/` there — local to that
machine, does not transfer with git). The load-bearing ones, restated directly:
- Never `git commit`/`push`/`amend` without asking first, every time, however routine it looks.
- Never run `rsync --delete` (or any mirror-delete sync) against a remote path that could hold a
  machine's own trained checkpoints (`wm/runs/`, `results/wm/`). This destroyed two old (already
  superseded, already documented) checkpoints on BIAS on 2026-09-18 — pushing code to any remote:
  no `--delete`, and `--exclude='runs/'` on any `wm/` sync as a second layer of protection.
- Never quote a number from memory — read the file, label confirmed vs unverified, log
  measurements as they happen.
- Never publish to claude.ai Artifacts for this user unless they explicitly ask for one by name;
  save charts/reports as local files (this project keeps finished chart/deck images in
  `results/deck/`).
- A selection/candidate-scoring score without a swapped-goal (mismatch) control has been withdrawn
  every time it was reported without one.
- The real planner (`wm/policy/planner.py`) picks a horizon-window from any candidate clip at any
  offset to match a goal (`score_offsets`) — it never selects "a clip" or "a condition" as an
  atomic unit. Evaluate goal-tracking with a genuinely time-varying goal and a continuous
  per-timestep distance error, never a whole-clip-mean goal or a condition-level family-accuracy
  rate (exactly the bug found and fixed in the deck's Slide 22 this session, see below).
- `wm.fit_body_head` for B1 needs `--also hexapod=<hex_dir>` by default, not an afterthought —
  skipping it once already silently corrupted a vision-read RL goal.
- Give each shell command its own separate code block when handing commands to the user.
- Always invoke Python as `python3`, never `python`.
- Default communication register: terse, number/conclusion first, caveats one line each.
- Z is not "morphology-agnostic" — call it a shared body-motion coordinate.
- Don't propose more bodies per embodiment as a fix for anything — the claim is cross-family
  transfer (hexapod ↔ B1), not within-family generalization.

**State of play, most recent work first (full detail: `doc/FINDINGS.md`, most recent entries near
the end; `git log`, most recent commit `a4b965c` on `main`, already pushed to `origin/main`):**

1. Expanded the behavior library from 12 to 24 conditions per body (hexapod + B1) — added
   backward speed, opposite-direction turn, two more strafe levels per side, fixing a real
   coverage asymmetry (original 12 was forward-only speed, one-direction-only turn). New:
   `scripts/dataset/collect_beh24.py` (hexapod), extended `scripts/dataset/recollect_b1_more.py`
   (B1). All new conditions calibrated to shared Froude targets across both bodies.
2. Found and fixed a real FOV bug: B1's newer collection path (`recollect_b1_more.py`) never got
   the ego-mode 90° wide-angle override the original data's script already had, so every new B1
   ego clip this session was shot through a 24° lens meant for an unrelated third-person framing
   fix. Fixed in `sim/render/render_b1_replay.py` (ego mode now defaults to 90°) and in
   `recollect_b1_more.py`. Verified: room corners now visible with real parallax. The original 8
   untouched conditions per body were never affected.
3. Hexapod's turn-direction reversal is UNSOLVED and parked (F219 in FINDINGS.md) — negating
   `--spin` does not reliably reverse the CPG gait's turn direction, the 4th documented instance
   of this exact bug class (F66, F106/F108, F174, F219), root cause never found in any of the
   four. This session's hexapod `turn_neg` data was collected during a lucky live sign-flip
   window — fragile, could be wrong if the instability recurs, do not trust as a permanent fix.
   B1's `turn_neg` is solid (direct `wz` command, not a hand-tuned oscillator).
4. Discovered a serious, pre-existing data-leakage bug affecting three deck claims (Slides 22, 23,
   Section 9 of `report/update_slide.md`). An ad-hoc "held-out" set used in several diagnostics
   this session (random 25% split, seed=0) has ZERO overlap with the model's real deterministic
   held-out set (`wm/data/dataset.py`'s `embodiment_split()`). Every number computed against the
   wrong split was measuring train-set/memorized performance. Slide 23 (0.486/0.667/0.736) is the
   worst-affected — genuinely unmeasured. Section 9 (0.264→0.572) has a smaller, confirmed
   2-of-9-clip leak. Slide 22/F210 ("92% across the whole goal set") had only 1 of 12 goal
   conditions genuinely held out.
5. Built and locally verified a clean, stratified train/held-out split (seed=42, 1 clip per
   condition, zero overlap) — `scripts/dataset/make_clean_split.py`, producing
   `data/egocentric/beh12_c10f10t10_ego_flat_{cleantrain,cleanheldout}` and the B1 equivalent
   (committed to git as real files; local symlinks on aria-desktop were disk-space-only).
6. Started a clean retrain on BIAS (`scripts/run/clean_retrain.sh`, 4 stages: fresh hexapod-only
   pretrain with `--val_fraction 0`, adapt to B1, fit projector, fit body_head with hexapod
   rehearsal). Fixed a real code bug along the way: `--val_fraction 0` makes the validation set
   empty, which `wm/data/dataset.py`'s `MultiEmbodimentPairs.__init__` and
   `EmbodimentBatchSampler._batches_per_group()` didn't handle — both now guard the empty case.
   **Retrain status on BIAS is UNCONFIRMED** — the driving SSH session may have died with
   aria-desktop's power loss. Check
   `ssh ioon@10.204.100.152 'ls -la ~/ioon-research/wm/runs/beh12_hinge_cleansplit/'` before
   assuming a restart is needed; re-running is fine either way.
7. Built a reusable diagnostic, `scripts/diagnostics/objective_experiments/
   froude_match_timevarying.py`, fixing the same time-varying-goal issue as item 4 but for the
   2×2 ablation (`final_2x2x2_test.py` reads its goal as one whole-clip mean; this reads it fresh
   every timestep, both physics and vision-read, against real B1 direct and rollout planners).
   Ran once successfully; a second run hit the aria-desktop hardware fault before finishing —
   re-verify end to end once compute is available. Sample outputs already in
   `results/deck/froude_*.png`.

**Uncommitted on aria-desktop as of the migration** (not yet on `origin/main` — get these by
asking the user for a manual copy, or after they approve a commit+push from aria-desktop):
`wm/data/dataset.py` (the empty-val-set fix above — without it, resuming the retrain hits the
same crash), and `scripts/diagnostics/objective_experiments/froude_match_timevarying.py` (new).

**Open decisions for this session:**
1. Resume vs restart the BIAS retrain (check state first).
2. Do NOT fold the 24-condition beh24 data into this clean retrain — keep it narrowly scoped to
   the split fix alone, so any result change is attributable to one variable. A 24-condition
   retrain is legitimate future work, only after hexapod's `turn_neg`/`side_R_lvl3` fragility is
   resolved for real, not bundled in now.
3. Chase a real fix for hexapod's turn-direction bug (4-for-4 unsolved historically), or
   permanently drop `turn_neg` for hexapod and document the asymmetry.
4. How to correct Slides 22/23/Section 9 once a clean checkpoint exists.
5. **New, found by the slide-deck session, 2026-09-18, not yet run (F221 in FINDINGS.md):** the
   exact same whole-clip-mean-goal bug F220 fixed for the 2×2 test also sits in
   `sim/control/teacher_student_insect.py:body_goal()` — the goal function behind the entire
   F135/F136/F164/F165/F166 fine-action-ranking arc ("coarse works, fine doesn't"). Diagnosed by
   reading the code only; whether fixing it changes that arc's conclusion is untested — needs a
   live CoppeliaSim run (unlike F220's diagnostic), so it's blocked on the same no-GPU rule above.
   Read F221 before touching `teacher_label_quality.py` or re-opening the fine-ranking question.

**Update from a parallel slide-deck session, 2026-09-18 (this machine, same day as the migration
above) — item 4 partially actioned, here's the current state of the deck:**
- Note: this session's own slide numbering differs from the one used above (the deck was
  restructured/renumbered independently of this research work). By content, not number: the old
  "Slide 22" (2×2 test) is now **Slide 21**; old "Slide 23" (ITM linear/MLP/ITM ablation) is now
  **Slide 22**; "Section 9" (zero-shot vs. staged adaptation, 0.264→0.572) is now **Section 10**.
- **Slide 21 (2×2) — fixed, not just flagged.** Rewrote it around
  `froude_match_timevarying.py`'s own already-completed output (`results/deck/froude_match_2x2_*.png`,
  timestamped 2026-09-18 00:40–01:46 — these did finish, despite the note above that a second run
  hit the hardware fault before finishing; only a horizon-comparison plot looks partial). Real
  numbers now in the deck: direct=0.038 vs. rollout=0.082 mean error; vision-goal=0.034 vs.
  physics-goal=0.038. Same story as before (direct beats rollout; vision costs nothing), now on the
  corrected continuous-tracking methodology instead of the old whole-clip-mean one.
- **The old "92% across all 12 goal conditions" aggregate claim is withdrawn from the deck** — that's
  the exact leak-affected number (1-of-12 genuinely held out). Not replaced with anything yet;
  the deck says explicitly not to use it until the clean retrain is confirmed.
- **Slide 22 (ITM ablation, 0.486/0.667/0.736) and Section 10 (0.264→0.572) are flagged in-deck**
  as unverified/pending-confirmation respectively, with the reasoning above stated inline, but
  **no corrected numbers substituted** — that still needs a confirmed clean checkpoint to re-run
  against. This is the part of item 4 still open.
- **When the clean retrain is confirmed:** re-run whatever produced Slide 22's linear/MLP/ITM R²
  table and Section 10's zero-shot/staged ρ table against it, then remove the ⚠ flags in
  `report/update_slide.md` and drop the actual numbers in. Ping back here (or leave a note in this
  file) once that's done so the slide session can pick it up.

**Standing conventions for `report/update_slide.md` (the slide deck), established this session —
follow these on any future edit, don't relitigate them from scratch:**

1. **No debugging-journal narrative.** The deck used to read as "tried X, it broke, tried Y" in the
   order things happened — that's what `doc/FINDINGS.md` is for. Every experiment on the deck states
   its hypothesis/problem first, then method, then result — never the chronological discovery order
   for its own sake.
2. **Every experiment is a numbered Checkpoint with four explicit parts**, per the advisor's own
   Week 16 feedback (`feedbacks/feedback_ajan_blink.md`):
   - **Assumption** — what new condition this step tests, given the prior finding
   - **Input → Output**
   - **Answers** — which of proposal.tex's 3 Objectives (§1.2) it supports
   - **Finding / remaining gap** — the result, plus an explicit bridge naming which experiment
     picks up the open thread next
   Format in the deck: a one-line `**Assumption / Input→Output / Answers.**` tag right after each
   `## Slide N —` / `## N. Methodology —` heading, and a `**Finding / remaining gap.**` tag right
   before its closing Thai block. Unresolved/unfinished work gets an explicit **"Preliminary
   study"** or **"Under investigation"** label instead of being presented as closed.
3. **Experiment IDs (1.1, 2.1a, 3.4, etc.) are content identifiers, not slide-delivery order.**
   When the clearer story requires reading experiment B before experiment A, that's fine — but say
   so explicitly at the top of that arc (see the "Reading order note" before Slide 11) so nobody
   mistakes the physical order for a numbering error.
4. **Real numbers in tables, never qualitative pass/fail framing.** "Before this fix / after this
   fix" with actual measured values — not an abstract "criterion (pre-registered) | before | after"
   grading table. State units and what "before/after" specifically means (before *this* fix, not
   some other change) when it could be ambiguous.
5. **Two fixes for the same underlying problem stay two fixes**, presented independently, unless
   they've actually been run together — never implied to be stacked or causally dependent on each
   other without a test proving it (see Experiment 2.3 vs. 2.4 in the deck: same redundancy, two
   independent levers, explicitly noted as never tested together).
6. **Never state a number without a genuinely held-out, non-leaking split behind it** — this
   session found exactly that failure (Slides 21/22, Section 10, above). When a split is
   questionable, flag or withdraw the number in the deck text itself; don't just remember to
   mention it verbally.
7. **Thai (`บทพูด (TH)`) blocks must match the English they sit under.** If a restructuring pass
   doesn't have time to verify/update a Thai block against new content, remove it rather than
   leave a stale one — a wrong translation is worse than a missing one.
8. **One document.** Don't maintain a second "restructured" or "cleaned up" copy of the deck
   alongside `update_slide.md` — merge anything useful in and delete the copy. Two documents that
   can drift apart will drift apart.
9. **Terminology:** the shared Froude-readout module is the **Cross-Body Head** in prose (per the
   advisor's Week 16 rename from "Body Motion Head" — it didn't communicate its role); the code
   identifier `body_head` stays as-is in backticks when referring to the actual variable.
═══════════════════════════════════════════════════════

═══ READ FIRST, IN THIS ORDER ═══
1. This message (the state + rules below)
2. direction_plan.md — the plan and contribution
3. FINDINGS.md — the evidence. NOT append-only: findings are CORRECTED or WITHDRAWN when a later finding refutes them, or replaced when a new finding supersedes an old one. So FINDINGS reflects the CURRENT state of evidence, not a frozen log. Read the latest findings first, and note any withdrawal/correction annotations (e.g. "F117 withdrawn by F126") — a withdrawn finding must NOT be reused as if still valid.
4. PROGRESS.md — narrative history (append-only)
5. OPEN_QUESTION.md — what's unresolved
6. SIM_GUIDE.md — how to run simulators and collect data
7. YOUR MEMORIES — read all stored project memories; they hold locked decisions, corrected claims, and guards that must not be re-broken.

═══ WHAT THE PROJECT IS ═══
Cross-embodiment locomotion from vision: learn a morphology-agnostic latent action / world model from egocentric video so a behaviour from one robot (18-DOF stick insect) can drive another (12-DOF Unitree B1 quadruped) via a shared body-motion coordinate (Froude-scaled forward/lateral/yaw), with NO kinematic model, NO demonstrations of the target, NO URDF. Bodies: insect (CoppeliaSim), B1 (MuJoCo).


═══ LEARN THE REPO STRUCTURE (survey, don't assume) ═══
Before running or creating anything, MAP the repo yourself and report back — I will NOT hand you the structure because it must match what actually exists, not my memory:

1. Run `ls` / tree on the repo root and key dirs. Report the layout: where do source modules live (wm/, sim/, etc.), where are scripts (scripts/, scripts/dataset/, scripts/diagnostics/), where is data (data/, data/egocentric/), where are runs/checkpoints (wm/runs/), where are results/caches (results/wm/cache/).

2. Identify the NAMING CONVENTIONS from existing files — do NOT invent your own:
   - dataset naming (e.g. beh12_c10f10t10_flat, beh12_b1_flat, beh12_c08f09t09_ego_flat — what do the tokens mean: behaviour-count, body-id, ego/allo, flat?)
   - checkpoint naming (e.g. best.pt, md_refit.pt, teacher_ego.pt, projector_ego.pt — which stage/component each is)
   - script naming (e.g. f183_ldad.sh, com7_pretrain_*.sh, step2_*.sh — the f<N>_ prefix ties a script to a finding; com7_ means it runs on the com7 GPU box)
   - cache naming (keyed by path — ego vs allo caches must not collide)

3. Identify KEY ENTRY POINTS by reading, not guessing: the main train script (wm/train.py?), the diagnostic scripts (scripts/diagnostics/*), the sim/control code (sim/control/teacher_student_insect.py?), how a run is launched on com7 vs locally.

4. WHERE THINGS GO (confirm the convention, don't break it):
   - new datasets → where? new checkpoints → wm/runs/<name>/? new diagnostic scripts → scripts/diagnostics/? logs → where?
   - what runs LOCALLY (has GUI / small) vs on COM7 (the compute box, checkpoints live there)?
   - CoppeliaSim (insect) needs a GUI + exactly ONE instance; MuJoCo (B1) doesn't.

5. Report back a short map: "source here, scripts here, data here, checkpoints here, naming = X, entry points = Y, com7-vs-local = Z." I'll correct anything wrong before you touch the repo.

Do NOT create files, rename anything, or launch runs until you've surveyed and I've confirmed your map. Follow existing conventions exactly — a new file in the wrong place or with an off-convention name breaks scripts that reference paths and makes results unauditable.

```