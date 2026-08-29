"""Pick an action by rolling the forward model over candidates and scoring against a goal.

    e_t ──► for each candidate action sequence a:  projector(a) ──► z ──► roll FTM h steps
                                                              ──► score against e_goal
            execute the first command of the winner, then replan

**The inverse model is not used.** That is not an optimisation; it is the reason this file exists
separately from every measurement in `scripts/diagnostics/`, all of which read `z` off two
ground-truth frames and are therefore reconstruction rather than control.

**Candidates are recorded action sequences, not sampled joint angles and not gait parameters.**

  *Not raw joint angles.* Sampling 18 continuous dimensions produces postures that do not walk and
  that the forward model has never seen; the rollout would be extrapolation and its ranking
  meaningless.

  *Not CPG parameters*, though that is the natural continuous generalisation. A central pattern
  generator is hand-authored knowledge about **this** robot -- how many legs, which are paired,
  what phase offset walks -- and needing one per robot is exactly the cost this project claims not
  to pay. Recorded sequences need none of it: on a new robot you already have the few clips that
  slide 15 adapts the forward model on, and those clips are the candidate set.

**Every candidate is indexed by the same `t`.** All clips are generated from the same settle and
the same oscillator start, so a shared index keeps them at a common point of the gait cycle. Two
reasons, and the second is the one that matters:

  A planner that switched from one recorded sequence to another at mismatched phase would emit a
  discontinuous joint command, which is an execution artefact and not a decision.

  **It is the regime the discrimination was measured in.** F90's phase-aligned rows -- 57.8%
  against a 25% chance level on four speeds of one behaviour -- are what this planner's accuracy
  should be read against. The free-phase rows are 15 points higher and describe a planner that
  could reject a candidate for being at the wrong point of its stride, which this one cannot,
  because its candidates all start where it currently is.
"""
import glob
import os

import numpy as np
import torch

from ..config import from_checkpoint
from ..data.embodiment import REGISTRY, load
from ..models.action_projector import ActionProjector, action_dims_from
from ..models.ftm import ForwardTransitionModel


def condition_of(path):
    """The behaviour label stored in the clip, or its filename if the clip predates the field.

    Read here rather than through `wm.data.embodiment.load`, which does not carry it -- and read
    through **one** function, because a caller that recovers the label a second way will disagree
    with the candidate set the moment the two paths diverge.
    """
    with np.load(path, allow_pickle=True) as data:
        if "condition" in data.files:
            return str(data["condition"])
    return os.path.basename(path)


def load_candidates(directory, embodiment, per_condition=1):
    """One entry per behaviour condition: its label and its recorded command sequence."""
    spec = REGISTRY[embodiment]
    by_condition = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        by_condition.setdefault(condition_of(path), []).append(path)
    out = []
    for cond, paths in sorted(by_condition.items()):
        for path in paths[:per_condition]:
            clip = load(path, spec)
            out.append({"condition": cond, "path": path,
                        "actions": clip["actions"].astype(np.float32)})
    return out


class LatentPlanner:
    """Scores candidates by rolling the forward model; holds no simulator and no encoder.

    Kept free of both so it can be exercised on recorded embeddings before anything is wired to
    CoppeliaSim -- the loop has two independent things that can be wrong, and separating them is
    cheaper than debugging them together.
    """

    def __init__(self, ftm, projector, candidates, embodiment, horizon=5, device="cuda"):
        self.ftm, self.proj = ftm, projector
        self.candidates = candidates
        self.embodiment = embodiment
        self.horizon = int(horizon)
        self.device = torch.device(device)
        self.action_lag = 1

    @classmethod
    def from_checkpoint(cls, ckpt_path, candidates_dir, embodiment="hexapod",
                        projector_path="", horizon=5, per_condition=1, device="cuda"):
        device = torch.device(device)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = from_checkpoint(checkpoint["config"])
        ftm = ForwardTransitionModel(cfg).to(device).eval()
        ftm.load_state_dict(checkpoint["ftm"])
        for p in ftm.parameters():
            p.requires_grad_(False)

        projector_path = projector_path or os.path.join(os.path.dirname(ckpt_path), "projector.pt")
        saved = torch.load(projector_path, map_location="cpu", weights_only=False)
        proj = ActionProjector(cfg, action_dims_from(saved)).to(device).eval()
        proj.load_state_dict(saved["projector"])
        for p in proj.parameters():
            p.requires_grad_(False)

        cands = load_candidates(candidates_dir, embodiment, per_condition)
        if not cands:
            raise ValueError(f"no candidate clips in {candidates_dir}")
        planner = cls(ftm, proj, cands, embodiment, horizon, device)
        planner.action_lag = max(1, cfg.action_lag)
        planner.cfg = cfg
        return planner

    def horizon_at(self, t):
        """How many steps can actually be rolled from `t` before a candidate runs out."""
        room = min(len(c["actions"]) - t - self.action_lag for c in self.candidates)
        return max(1, min(self.horizon, room))

    @torch.no_grad()
    def score(self, e_t, e_goal, t):
        """Predicted-versus-goal error for every candidate, lower is better."""
        h = self.horizon_at(t)
        e_t = e_t.to(self.device).float()
        e_goal = e_goal.to(self.device).float()
        if e_t.dim() == 2:
            e_t = e_t.unsqueeze(0)
        out = []
        for cand in self.candidates:
            a = torch.as_tensor(cand["actions"][t + self.action_lag:t + self.action_lag + h],
                                device=self.device)
            z = self.proj(a, self.embodiment)
            e = e_t
            for i in range(len(z)):
                e = self.ftm(e, z[i:i + 1])
            out.append(float(((e[0] - e_goal) ** 2).mean()))
        return np.asarray(out)

    @torch.no_grad()
    def act(self, e_t, e_goal, t):
        """The command to execute now, plus which candidate produced it and every score."""
        scores = self.score(e_t, e_goal, t)
        i = int(np.argmin(scores))
        cand = self.candidates[i]
        # the command at `t`, not at `t + action_lag`: the lag is how the *target* is defined for
        # scoring, and what the robot executes on this step is this step's command
        return cand["actions"][min(t, len(cand["actions"]) - 1)], i, scores
