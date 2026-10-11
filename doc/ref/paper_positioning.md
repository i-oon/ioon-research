# Paper positioning (2026-10-11, from the user's discussion with Codex; for the IROS draft)

Not claimed: first visual world model; first cross-embodiment world model; "shared latent" in the sense of aligned
motion codes; action-free / purely self-supervised (projectors use recorded commands; Froude is measured supervision).

Closest work (PDFs / notes in this folder; check every citation against the PDF before writing it):
| work | what it does | difference from ours |
|---|---|---|
| LAPO, Learning to Act without Actions (ICLR 2024) | latent actions, inverse + forward model from action-free video, then a policy | the transition -> z -> future idea; no legged cross-body locomotion, no visual hazard selection |
| QWM, hardware-agnostic quadrupedal world model (arXiv 2604.08780) | one world model over many quadruped morphologies, morphology spec as input, zero-shot within quadrupeds | quadruped template only, needs a morphology spec, blind flat-ground velocity tracking (no exteroception) |
| WMP, World Model-based Perception (arXiv 2409.16784) | depth + proprioception recurrent world model feeding a locomotion policy | world-model state as policy input; no per-candidate future simulation at run time; one body |
| LAC-WM, latent action robot foundation world models (notes_lac_wm.md) | shared latent action across embodiments, adaptation to new bodies | closest: novelty cannot be "a shared z across bodies" |

Our gap: whether a jointly trained, visually grounded latent-action world model works across legged bodies with
different action spaces and gaits (hexapod c10 / c08 / quadruped B1), and whether imagined visual consequences of
each candidate improve action selection (direct vs current-image-only vs rollout, same observations / library / starts).

Wording rules:
- "shared task interface" / "jointly trained latent-action model", not "shared latent space" (probes: c10 and B1 not
  aligned over all 64 dims, F-series alignment results); show task results first.
- Wall test is B1 only -> "predictive visual selection on B1"; cross-body results are separate evidence; never
  "cross-body wall avoidance". MuJoCo has no walls -> "virtual boundary", "boundary crossing", never "collision".
- B1 walks with a pre-trained walking policy (commands, not joints) -- say so.
- Working title: "Visually Grounded Latent-Action World Models for Cross-Embodiment Legged Locomotion"; move to
  "predictive action selection" in title/abstract only if the wall test passes the prediction check AND the closed loop.

Paper outline: 1 problem (joint commands not shareable; common goal = Froude; camera consequences) 2 method (ITM, FTM,
per-body projector, Froude head) 3 cross-body evidence (c08 zero-shot, B1, physics) 4 prediction evidence (wall: direct /
current image / rollout + ranking of the 24 candidates' risk on held-out states) 5 limits.

## Research question and test plan (2026-10-11, user + Codex)

RQ: *Can a shared visual latent-action model support motion-goal selection across different legged embodiments, and
can its predicted visual consequences improve action selection when the current scene matters?*
"Shared" = shared model parts + shared Froude goal interface; each body keeps its own action projector (no claim that
all 64 z dims align motions across bodies). The wall test is meant for ALL bodies (B1 first, then c10, c08).

| question | test | baselines |
|---|---|---|
| 1 shared model across bodies? | same Froude goals; c10 trained, c08 zero-shot, B1 after adaptation; report error + adaptation data size | separate per-body models; direct selector; before / after adaptation incl. forgetting on c10 |
| 2 predicts outcomes of different commands from one state? | restore the state, run every command (28 on the wall data); predicted vs true future per horizon; clearance ranking; unsafe detection | command-only; copy current image; true-future oracle |
| 3 does prediction help decisions? | closed loop near the virtual boundary, paired starts, same 28-command library; boundary crossings AND forward progress / Froude goal error | vision-blind direct; CURRENT-IMAGE selector without the FTM (head on (e_t, z) -> outcome, same data); rollout; true-future oracle; TTC rule (knows true distance) as reference |
| 4 avoidance across bodies? | same wall task on the hexapods, comparable rules and goals | test-3 baselines on the hexapods |

Test 3 separates the value of SEEING from the value of PREDICTING: rollout > blind direct but not > current-image
selector = "vision helps", not "prediction helps".

Claims by evidence: (a) shared parts + Froude interface select motion goals across c10 / c08 / adapted B1 -- partly
supported now, report closed loop and rollout limits honestly; (b) on B1, the predicted visual future keeps progress
and avoids the virtual boundary better than the current image alone -- needs test 2 + test 3; (c) cross-embodiment
predictive avoidance -- needs test 4. Never merge (a) + B1-only walls into (c).
Literature notes: Egocentric VSM also adapts to a changed body (do not call it single-body); QWM is zero-shot within a
12-DoF quadruped template with a morphology spec, our B1 is ADAPTATION (say so); WMP feeds a world-model state to an RL
policy, no per-candidate imagination at decision time.
