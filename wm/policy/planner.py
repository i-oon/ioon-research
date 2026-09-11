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

  **It is the regime the discrimination was measured in.** F80's phase-aligned rows -- 57.8%
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
from ..models.itm import InverseTransitionModel
from ..models.motion_decoder import MotionDecoder


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


class DirectFroudePlanner:
    """Scores candidates by `body_head(proj(a))` against a goal Froude vector -- no rollout, no
    FTM, no encoder, no e_t at all.

    **Why this exists.** `LatentPlanner` above rolls the forward model and compares raw embedding
    distance; `score_by_body_motion.py`'s mode C (the same mechanism read through the shared body
    coordinate instead of raw embedding distance) still failed to clear chance at any horizon on a
    correctly-adapted checkpoint (F184) -- consistent with this project's repeated finding that the
    rollout does not earn its place in selection (F126/F127). Mode A -- exactly this mechanism,
    `score(a) = |body_head(proj(a)) - goal|` -- is the one that DID clear chance (F184, 38-46% vs
    28%). This class is that mechanism, shaped as a drop-in replacement for `LatentPlanner` (same
    `from_checkpoint`/`horizon_at`/`act` interface) so a closed-loop driver needs to swap only the
    planner class, not its control loop.

    The goal is a **fixed Froude vector**, not a per-step embedding -- consistent with mode A,
    which read the goal as a recorded number rather than from frames (mode C, reading the goal from
    frames via the ITM, is the version that failed). A real deployment states a goal as "achieve
    this dimensionless speed", which is exactly this input.
    """

    def __init__(self, projector, md, candidates, embodiment, horizon=5, device="cuda",
                free_offset=False):
        self.proj, self.md = projector, md
        self.candidates = candidates
        self.embodiment = embodiment
        self.horizon = int(horizon)
        self.device = torch.device(device)
        self.action_lag = 1
        # **Additive, default-off.** `free_offset=False` (the default) reproduces the exact
        # behaviour this class always had -- every candidate read at the SAME index `t` as the
        # live episode, one score per candidate. This was a deliberate choice (F80/module
        # docstring): free-phase distractors score 15 points HIGHER on the hexapod library, but
        # were rejected because switching to a candidate at a mismatched phase emits a
        # discontinuous joint command -- "an execution artefact, not a decision." `free_offset=True`
        # revisits that specifically for candidates whose FRAMES are never used (only `actions`,
        # via `body_head(proj(a))`) -- there is no visual-continuity reason offset must equal `t`,
        # only the same physical discontinuity risk F80 flagged. Kept as an opt-in flag, not a
        # replacement, so the original mechanism is always one flag away, not gone.
        self.free_offset = bool(free_offset)

    @classmethod
    def from_checkpoint(cls, ckpt_path, candidates_dir, embodiment="b1", projector_path="",
                        horizon=5, per_condition=1, device="cuda", free_offset=False):
        device = torch.device(device)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = from_checkpoint(checkpoint["config"])

        cands = load_candidates(candidates_dir, embodiment, per_condition)
        if not cands:
            raise ValueError(f"no candidate clips in {candidates_dir}")
        action_dim = cands[0]["actions"].shape[1]

        md = MotionDecoder(cfg, {embodiment: action_dim}).to(device).eval()
        md.load_state_dict(checkpoint["md"], strict=False)
        if md.body_head is None:
            raise ValueError("this checkpoint has no body_head (lambda_body was 0)")
        for p in md.body_head.parameters():
            p.requires_grad_(False)

        projector_path = projector_path or ckpt_path
        saved = torch.load(projector_path, map_location="cpu", weights_only=False)
        proj = ActionProjector(cfg, action_dims_from(saved)).to(device).eval()
        proj.load_state_dict(saved["projector"])
        for p in proj.parameters():
            p.requires_grad_(False)

        planner = cls(proj, md, cands, embodiment, horizon, device, free_offset=free_offset)
        planner.action_lag = max(1, cfg.action_lag)
        planner.cfg = cfg
        planner.channels = [int(c) for c in cfg.body_channels]
        mean_s, std_s = checkpoint["body_stats"]
        planner.mean_s = np.asarray(mean_s).ravel()[:len(planner.channels)]
        planner.std_s = np.asarray(std_s).ravel()[:len(planner.channels)]
        return planner

    def standardize(self, goal_froude):
        """Raw (forward, lateral, yaw) -> the standardised units `body_head` was fit to predict."""
        return (np.asarray(goal_froude, dtype=np.float32) - self.mean_s) / self.std_s

    def horizon_at(self, t):
        room = min(len(c["actions"]) - t - self.action_lag for c in self.candidates)
        return max(1, min(self.horizon, room))

    @torch.no_grad()
    def score(self, goal_std, t):
        """Predicted-versus-goal error for every candidate, lower is better. `goal_std` is
        ALREADY standardised (see `standardize`), same units `body_head` outputs.

        `free_offset=False` (default): one score per candidate, its window fixed at `t` -- the
        original behaviour, unchanged. `free_offset=True`: one score per candidate, but the BEST
        of every valid offset within that candidate -- see `score_offsets` for the per-offset
        detail this collapses. `act` uses that detail to know which offset won, `score` alone
        cannot express it (kept this way so `score`'s return shape never changes)."""
        if self.free_offset:
            return np.asarray([np.min(row) for row in self.score_offsets(goal_std)])
        h = self.horizon_at(t)
        goal = torch.as_tensor(goal_std, dtype=torch.float32, device=self.device)
        out = []
        for cand in self.candidates:
            a = torch.as_tensor(cand["actions"][t + self.action_lag:t + self.action_lag + h],
                                device=self.device)
            z = self.proj(a, self.embodiment)
            pred = self.md.body(None, z).mean(0)
            out.append(float(((pred - goal) ** 2).mean()))
        return np.asarray(out)

    @torch.no_grad()
    def score_offsets(self, goal_std):
        """`free_offset=True` only: every candidate scored at EVERY valid start offset tau (not
        just `t`) -- candidates never expose their frames to this planner, only `actions`, so
        there is no visual-continuity reason a candidate's window must start at the live episode's
        own step index. Returns one array of per-offset scores per candidate (ragged: candidates
        need not share a length).

        **Batched per candidate, one forward pass instead of one per offset.** `ActionProjector`
        and `body_head` are plain per-timestep MLPs (no recurrence), so they broadcast over any
        leading batch dim -- stacking every offset's window into one (n_taus, horizon, action_dim)
        tensor and calling both once per candidate is exactly the same computation as the original
        per-offset python loop, just not re-tracing the model n_taus times. Measured necessary, not
        cosmetic: the per-offset loop made even a 40-goal/12-candidate sweep too slow to finish in
        two minutes."""
        goal = torch.as_tensor(goal_std, dtype=torch.float32, device=self.device)
        out = []
        for cand in self.candidates:
            n = len(cand["actions"])
            h = max(1, min(self.horizon, n - self.action_lag))
            n_taus = max(1, n - self.action_lag - h + 1)
            windows = np.stack([cand["actions"][tau + self.action_lag:tau + self.action_lag + h]
                               for tau in range(n_taus)])           # (n_taus, h, action_dim)
            a = torch.as_tensor(windows, device=self.device)
            z = self.proj(a, self.embodiment)                        # (n_taus, h, z_dim)
            pred = self.md.body(None, z).mean(1)                     # (n_taus, body_dim)
            out.append(((pred - goal) ** 2).mean(-1).cpu().numpy())
        return out

    @torch.no_grad()
    def act(self, goal_std, t):
        """The command to execute now, plus which candidate produced it, every score, and the
        candidate-internal index `tau` the command actually came from.

        **Always returns a 4-tuple** `(action, i, scores, tau)` -- `free_offset=False` sets
        `tau == t` (the original behaviour, unchanged), `free_offset=True` sets `tau` to whichever
        offset scored best, independent of `t`. A caller that only reads a candidate's own
        recorded MOTION (not just its action) at decision time -- e.g. a kinematic closed-loop
        driver posing the body from `motion[i][t]` -- MUST index that motion by `tau`, not `t`,
        under free_offset, or it will execute the wrong candidate's timeline while scoring a
        different one. This is not a hypothetical: `close_loop_direct_froude.py`'s main loop did
        exactly this before `tau` was threaded through.

        `free_offset=True` is the mechanism F80 measured as 15 points more accurate and rejected
        for the discontinuous joint command it can produce; call `action_jump`-style diagnostics
        alongside this to see the size of that discontinuity before trusting the accuracy number
        alone."""
        if self.free_offset:
            rows = self.score_offsets(goal_std)
            i = int(np.argmin([np.min(r) for r in rows]))
            tau = int(np.argmin(rows[i]))
            cand = self.candidates[i]
            return (cand["actions"][min(tau, len(cand["actions"]) - 1)], i,
                   np.asarray([np.min(r) for r in rows]), tau)
        scores = self.score(goal_std, t)
        i = int(np.argmin(scores))
        cand = self.candidates[i]
        return cand["actions"][min(t, len(cand["actions"]) - 1)], i, scores, t


class RolloutFroudePlanner:
    """`score_by_body_motion.py`'s mode C, live: the goal is read from the SOURCE robot's frames
    via the ITM (never a recorded number), and each candidate is scored by rolling the FTM forward
    from the CURRENT observed frame and reading the resulting transition, also via the ITM --
    "frames and rollout only... the condition the project's claim actually needs" per that
    script's own docstring. This is what `LatentPlanner` is not: `LatentPlanner` scores raw
    embedding distance to a goal frame (the mechanism F116/F118 found failing under every
    estimator tried); this reads both sides through the shared body-motion coordinate instead.

    Needs `e_t`, the current observation's embedding, every step -- unlike `DirectFroudePlanner`,
    which never looks at the frame at all. That is the whole methodological difference under test.
    """

    def __init__(self, itm, ftm, projector, md, candidates, embodiment, horizon=5, device="cuda"):
        self.itm, self.ftm, self.proj, self.md = itm, ftm, projector, md
        self.candidates = candidates
        self.embodiment = embodiment
        self.horizon = int(horizon)
        self.device = torch.device(device)
        self.action_lag = 1

    @classmethod
    def from_checkpoint(cls, ckpt_path, candidates_dir, embodiment="b1", projector_path="",
                        horizon=5, per_condition=1, device="cuda"):
        device = torch.device(device)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = from_checkpoint(checkpoint["config"])

        cands = load_candidates(candidates_dir, embodiment, per_condition)
        if not cands:
            raise ValueError(f"no candidate clips in {candidates_dir}")
        action_dim = cands[0]["actions"].shape[1]

        itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(checkpoint["itm"])
        ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(checkpoint["ftm"])
        md = MotionDecoder(cfg, {embodiment: action_dim}).to(device).eval()
        md.load_state_dict(checkpoint["md"], strict=False)
        if md.body_head is None:
            raise ValueError("this checkpoint has no body_head (lambda_body was 0)")
        for m in (itm, ftm, md.body_head):
            for p in m.parameters():
                p.requires_grad_(False)

        projector_path = projector_path or ckpt_path
        saved = torch.load(projector_path, map_location="cpu", weights_only=False)
        proj = ActionProjector(cfg, action_dims_from(saved)).to(device).eval()
        proj.load_state_dict(saved["projector"])
        for p in proj.parameters():
            p.requires_grad_(False)

        planner = cls(itm, ftm, proj, md, cands, embodiment, horizon, device)
        planner.action_lag = max(1, cfg.action_lag)
        planner.cfg = cfg
        planner.channels = [int(c) for c in cfg.body_channels]
        mean_s, std_s = checkpoint["body_stats"]
        planner.mean_s = np.asarray(mean_s).ravel()[:len(planner.channels)]
        planner.std_s = np.asarray(std_s).ravel()[:len(planner.channels)]
        return planner

    def standardize(self, goal_froude):
        """Raw (forward, lateral, yaw) -> the standardised units `body_head` was fit to predict.
        Same convention as `DirectFroudePlanner.standardize` -- needed here too now that
        `--goal_source physics` can pair with either candidate-scoring mechanism."""
        return (np.asarray(goal_froude, dtype=np.float32) - self.mean_s) / self.std_s

    def horizon_at(self, t):
        room = min(len(c["actions"]) - t - self.action_lag for c in self.candidates)
        return max(1, min(self.horizon, room))

    @torch.no_grad()
    def goal_from_frames(self, g0, g1):
        """`body_head(ITM(g0, g1))` -- the goal robot's own frames, no recorded number involved.
        `g0`, `g1` are single-frame embeddings (1, tokens, dim), any horizon apart."""
        g0 = g0.to(self.device).float().unsqueeze(0) if g0.dim() == 2 else g0.to(self.device).float()
        g1 = g1.to(self.device).float().unsqueeze(0) if g1.dim() == 2 else g1.to(self.device).float()
        return self.md.body(None, self.itm(g0, g1)).reshape(-1)

    @torch.no_grad()
    def score(self, e_t, goal, t):
        """Roll the FTM on each candidate from the CURRENT observation, read the transition with
        the ITM, compare its body motion to `goal` (already in `body_head`'s output units)."""
        h = self.horizon_at(t)
        e_t = e_t.to(self.device).float()
        if e_t.dim() == 2:
            e_t = e_t.unsqueeze(0)
        goal = goal.to(self.device).float()
        out = []
        for cand in self.candidates:
            a = torch.as_tensor(cand["actions"][t + self.action_lag:t + self.action_lag + h],
                                device=self.device)
            z = self.proj(a, self.embodiment)
            e = e_t
            for i in range(len(z)):
                e = self.ftm(e, z[i:i + 1])
            pred = self.md.body(None, self.itm(e_t, e)).reshape(-1)
            out.append(float(((pred - goal) ** 2).mean()))
        return np.asarray(out)

    @torch.no_grad()
    def act(self, e_t, goal, t):
        """The command to execute now, plus which candidate produced it and every score."""
        scores = self.score(e_t, goal, t)
        i = int(np.argmin(scores))
        cand = self.candidates[i]
        return cand["actions"][min(t, len(cand["actions"]) - 1)], i, scores
