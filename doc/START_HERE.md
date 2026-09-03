```
You are resuming a long research project mid-stream. Read files in this order, then follow the working rules below. Do NOT start any experiment until you've read the state and confirmed it back to me.

═══ READ FIRST, IN THIS ORDER ═══
1. This message (the state + rules below)
2. direction_plan.md — the plan and contribution
3. FINDINGS.md — the evidence. NOT append-only: findings are CORRECTED or WITHDRAWN when a later finding refutes them, or replaced when a new finding supersedes an old one. So FINDINGS reflects the CURRENT state of evidence, not a frozen log. Read the latest findings first, and note any withdrawal/correction annotations (e.g. "F126 withdrawn by F135") — a withdrawn finding must NOT be reused as if still valid.
4. PROGRESS.md — narrative history (append-only)
5. OPEN_QUESTION.md — what's unresolved
6. SIM_GUIDE.md — how to run simulators and collect data
7. YOUR MEMORIES — read all stored project memories; they hold locked decisions, corrected claims, and guards that must not be re-broken.

═══ WHAT THE PROJECT IS ═══
Cross-embodiment locomotion from vision: learn a morphology-agnostic latent action / world model from egocentric video so a behaviour from one robot (18-DOF stick insect) can drive another (12-DOF Unitree B1 quadruped) via a shared body-motion coordinate (Froude-scaled forward/lateral/yaw), with NO kinematic model, NO demonstrations of the target, NO URDF. Bodies: insect (CoppeliaSim), B1 (MuJoCo).

═══ WHERE IT STANDS, PRECISELY (as of the latest findings) ═══
- CORE POSITIVE (proven): a shared body-motion coordinate transfers across bodies from video (F136). Correspondence-free (no pairing/alignment/retargeting — F170-ish, Check A), unlike Demo-JEPA which needs end-effector retargeting + GTCC.
- THE VIEWPOINT FINDING (proven, the contribution): 3rd-person (allocentric) view makes the pose encode the action (single-frame action R² ~0.78), so forward prediction is phase-completion and the world model ignores the action — measured 6 ways (F153-F169). Egocentric view fixes this: GATE C, null/real 1.03→1.16 (first thing in the whole chain to move it), yaw cross-body 0.07→0.64. Egocentric BROKE the pose-redundancy that killed everything.
- WHAT EGOCENTRIC DID NOT FIX (current wall): action-conditioning (GATE C, coarse, 1-step) ≠ action RANKING (fine) ≠ long-horizon ROLLOUT. Teacher ranks recorded behaviours (83%) but fine perturbations at coin-flip (47%, F179). Gradient through imagination fails at K=1 (F182). These are SEPARATE capabilities. The world model uses the action COARSELY but not precisely enough to rank fine (2.5%) differences or roll out far (reliable only ~3-5 steps).
- STUDENT: pooled student is weak on B1 (within-cond 0.081→0.205 with attention); insect usable (~0.3). Clone-only allocentric passes the F142 walk bar (54%) WITHOUT the world model — any "world model helps" claim must beat 54%, not the old 36%.
- METHODS TRIED, ALL HIT THE SAME WALL: candidate scoring/ranking (F145/F179), objective fixes (ActSWM F153, LDAD/Delta-JEPA F183 — LDAD actively BROKE null/real 1.16→0.99, rejected), gradient/Dreamer (F182), harder perturbations (F181 sweep — off-manifold). The wall is the world model, not the method.
- LEADING HYPOTHESIS for the wall (UNCONFIRMED, still debugging): world model long-horizon


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