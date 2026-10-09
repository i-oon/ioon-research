"""One continuous CPG gait clock for the hexapod physics loop (FINDINGS F323 fix).

The old loop played `cands[i]["actions"][t]`: switching candidate jumped the legs to another clip's gait phase. Here the
loop keeps ONE oscillator phase (in cycles) and drives the body from the chosen candidate's CPG RECIPE (`plan_*` fields of
every hexapod clip), exactly as the counterfactual branches were collected (`build_branches.hex_plan`): at a switch the
recipe cross-fades over FADE frames, w = clip((t - T) / FADE, 0, 1), cmd = (1 - w) * cmd_old + w * cmd_new on the same
phase, and the clock rate (pace) is blended the same way. No switch = the candidate's own recipe on the running clock.

Phase rule (collect_ik._state_fields / cpg_commands): cycles_total[t + 1] = cycles_total[t] + BASE / EP * pace[t]
(BASE 8.8, EP 66), and the joint commands at frame t are cpg_commands' row for that phase. `commands_at` evaluates
cpg_commands at an arbitrary phase (no scene needed: the commands are pure; the scene is read only for the leg length
it reports). `check_clip` rebuilds a stored clip from its recipe + cpg_cycles_total and must match its actions.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim/collect"))
import collect_ik as CI  # noqa: E402

BASE, EP, FADE = 8.8, 66, 4
KEYS = ("pace", "spin", "strafe", "lead", "a0", "a1", "a2", "ft_phase", "sym")
CENTRE = {"c10f10t10": "data/counterfactual_walks/c10_walks/_work/centre_pose.npz",
          "c08f09t09": "data/counterfactual_walks/c08_walks/_work/centre_pose.npz"}
CPG_KW = dict(mirror_joints=(0, 1, 2), spin_amp=0.25, symmetric=False, legtune=None)


class _NoScene:
    def loadScene(self, f):
        pass


def centre_of(morph):
    return np.load(os.path.join(ROOT, CENTRE[morph]), allow_pickle=True)["actions"].astype(np.float64).mean(0)


def recipe(clip, t=0):
    """The candidate's recipe at its frame t (every v4 clip holds one behaviour: constant along the clip)."""
    return {k: float(np.asarray(clip[f"plan_{k}"])[min(t, len(clip[f"plan_{k}"]) - 1)]) for k in KEYS}


def commands_at(cycles, rec, centre, rec_new=None, w=0.0):
    """Joint commands at oscillator phase `cycles` (total cycles) for recipe `rec`, cross-faded with weight w to
    `rec_new` on the same phase (cpg_commands' xfade, one frame)."""
    leg_length, CI.leg_length = CI.leg_length, (lambda *a, **k: 0.0)
    try:
        # frames=2, cycles=1, pace=[2*c, 0] -> ph[1] = 2*pi*1*(2c)/2 = 2*pi*c  (row 1 is the wanted frame)
        kw = dict(cycles=1.0, pace=np.array([2.0 * cycles, 0.0]), **CPG_KW)
        r = rec
        xf = None
        if rec_new is not None:
            xf = dict(w=np.array([w, w]), sym=np.full(2, rec_new["sym"]),
                      **{k: np.full(2, rec_new[k]) for k in ("spin", "strafe", "lead", "ft_phase", "a0", "a1", "a2")})
        c, _ = CI.cpg_commands(_NoScene(), "", 2, centre, amps=(np.full(2, r["a0"]), np.full(2, r["a1"]), np.full(2, r["a2"])),
                               lead=np.full(2, r["lead"]), strafe=np.full(2, r["strafe"]), spin=np.full(2, r["spin"]),
                               ft_phase=np.full(2, r["ft_phase"]), sym=np.full(2, r["sym"]), xfade=xf, **kw)
    finally:
        CI.leg_length = leg_length
    return np.asarray(c[1], np.float32)


class Clock:
    """Online driver: `step(rec)` -> joint commands for this frame; a new recipe starts a FADE-frame cross-fade."""

    def __init__(self, cycles0, rec0, centre):
        self.c, self.rec, self.centre = float(cycles0), dict(rec0), centre
        self.new, self.since = None, 0
        self.switches = 0

    def choose(self, rec):
        target = self.new if self.new is not None else self.rec
        if rec == target:
            return
        if self.new is not None:          # a switch during a fade: settle the running fade first
            self.rec = self.new
        self.new, self.since = dict(rec), 0
        self.switches += 1

    def step(self):
        if self.new is None:
            cmd, pace = commands_at(self.c, self.rec, self.centre), self.rec["pace"]
        else:
            w = min(self.since / FADE, 1.0)          # w = 0 on the switch frame, as build_branches (frame T)
            cmd = commands_at(self.c, self.rec, self.centre, self.new, w)
            pace = (1.0 - w) * self.rec["pace"] + w * self.new["pace"]
            self.since += 1
            if w >= 1.0:
                self.rec, self.new = self.new, None
        self.c += BASE / EP * pace
        return cmd


def check_clip(path, morph):
    d = np.load(os.path.join(ROOT, path), allow_pickle=True)
    centre = centre_of(morph)
    rec = recipe(d)
    got = np.stack([commands_at(c, rec, centre) for c in d["cpg_cycles_total"]])
    k = Clock(d["cpg_cycles_total"][0], rec, centre)
    on = np.stack([k.step() for _ in range(len(d["actions"]))])
    a = np.asarray(d["actions"], np.float32)
    return float(np.abs(got - a).max()), float(np.abs(on - a).max())


if __name__ == "__main__":
    import glob
    for morph, pat in (("c10f10t10", "data/counterfactual_walks/c10_clips_heldout/*.npz"),
                       ("c08f09t09", "data/counterfactual_walks/c08_clips_heldout/*.npz")):
        errs = [check_clip(os.path.relpath(p, ROOT), morph) for p in sorted(glob.glob(os.path.join(ROOT, pat)))]
        e = np.asarray(errs)
        print(f"{morph}: {len(e)} clips, max |recipe - stored actions| at stored phase {e[:, 0].max():.2e}, "
              f"online clock {e[:, 1].max():.2e}")
