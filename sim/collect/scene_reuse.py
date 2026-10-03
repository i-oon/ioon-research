"""Scene-reuse mode for `collect_ik.drive_and_record`: several plans in one process, bit-identical physics.

Measured 2026-10-02 (FINDINGS F305, scratch `hexstate/`): CoppeliaSim/Bullet does not repeat a run across
`loadScene` (run 0 after a load always differs), but if the scene is loaded ONCE and every run only does
stop/startSimulation on it (no reload, no room rebuild after the first run), runs become bit-identical
after a warm-up of ~8-10 runs; one switch to another "class" of runs can occur around the 8th run, and the
class differs between instances. Hence:

1. compute every plan's joint commands first (`cpg_commands` loads the scene to read the leg length; here
   through a proxy that loads it once -- nothing it reads changes between loads),
2. load the scene once (first run, which also builds the ego camera / floor / room),
3. warm up with a reference command sequence until two consecutive runs agree bit for bit (at least
   `min_runs` runs), then
4. run every plan; `verify(...)` compares a run's recorded state with the reference run on the shared prefix.

Default off everywhere: nothing in `collect_ik` changes unless `drive_and_record(..., reuse=dict)` is used.

API (one SceneReuse per CoppeliaSim instance, one process per instance):

    R = SceneReuse(sim, scene, drive_kw)        # drive_kw = the drive_and_record keyword args of the walk
    cmds = R.commands(plans, frames, centre, cpg_kw)   # before any run
    ref, n, log = R.warm_up(cmds[0])            # full-state reference run
    out = R.run(cmds[k][:K], state_from=K0)     # -> dict(actions, forces, head, body_quat, state...)
    ok = R.same_as(out, ref, frames=range(a, b))

CLI (several plans, one process; the hexapod v4 walk settings):

    .venv/bin/python3 sim/collect/scene_reuse.py --port P --plans a.json b.json --frames N --centre_from c.npz --out dir
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collect_ik as CI  # noqa: E402

STATE_KEYS = ("state_abdomen_pos", "state_abdomen_quat", "state_joint_pos", "state_link_pose", "state_sim_time")
PER_FRAME = ("actions", "forces", "head", "body_quat") + STATE_KEYS + ("com_pos",)


class _LoadOnce:
    """sim proxy for command computation: loadScene of a file already loaded is a no-op."""

    def __init__(self, sim):
        self._sim, self._loaded = sim, None

    def __getattr__(self, k):
        return getattr(self._sim, k)

    def loadScene(self, f):
        if self._loaded != f:
            CI.settle(self._sim)
            self._sim.loadScene(f)
            self._loaded = f


def plan_commands(sim, scene, plan, frames, centre, cpg_kw):
    """cpg_commands for one --plan dict (same mapping as collect_ik.main)."""
    xf = None
    if "xf_w" in plan:
        xf = dict(w=plan["xf_w"], **{k: plan[f"xf_{k}"] for k in ("spin", "strafe", "lead", "ft_phase", "a0", "a1", "a2")})
        if "xf_sym" in plan:
            xf["sym"] = plan["xf_sym"]
    cmds, info = CI.cpg_commands(sim, scene, frames, centre, amps=(plan["a0"], plan["a1"], plan["a2"]),
                                 lead=plan["lead"], strafe=plan["strafe"], spin=plan["spin"], pace=plan["pace"],
                                 ft_phase=plan["ft_phase"], sym=plan.get("sym"), xfade=xf, **cpg_kw)
    return np.asarray(cmds, np.float32), info


def state_hash(out, keys=("state_link_pose", "state_joint_pos", "forces")):
    h = hashlib.sha1()
    for k in keys:
        h.update(np.ascontiguousarray(out[k]).tobytes())
    return h.hexdigest()[:16]


class SceneReuse:
    def __init__(self, sim, scene, drive_kw):
        self.sim, self.scene, self.kw = sim, scene, dict(drive_kw)
        self.reuse = {}
        self.n_runs = 0
        self.ref = None

    def commands(self, plans, frames, centre, cpg_kw):
        if self.reuse.get("built"):
            raise RuntimeError("compute commands before the scene is loaded for the runs")
        px = _LoadOnce(self.sim)
        return [plan_commands(px, self.scene, p, frames, centre, cpg_kw)[0] for p in plans]

    def run(self, cmds, state_from=0):
        st = {}
        f, a, fc, h, o = CI.drive_and_record(self.sim, self.scene, cmds, state_out=st, state_from=state_from,
                                             reuse=self.reuse, **self.kw)
        self.n_runs += 1
        out = dict(actions=a, forces=fc, head=h, body_quat=o, frames=f, **st)
        return out

    @staticmethod
    def same_as(a, b, frames, a_from=0, b_from=0, keys=PER_FRAME):
        """Bit-identical on walk frames `frames`; per-frame fields start at frame 0 except state_* / com_pos
        (row 0 = frame state_from)."""
        fr = np.asarray(list(frames))
        for k in keys:
            sa = a_from if (k.startswith("state_") or k == "com_pos") else 0
            sb = b_from if (k.startswith("state_") or k == "com_pos") else 0
            if not np.array_equal(np.asarray(a[k])[fr - sa], np.asarray(b[k])[fr - sb]):
                return False
        return True

    def warm_up(self, cmds, min_runs=8, max_runs=30, n_same=2, short_runs=0, short_frames=30, log=print):
        """Run `cmds` (full state) until `n_same` consecutive runs are bit-identical, at least min_runs runs.
        `short_runs`: first run the first `short_frames` frames that many times (cheap warm-up; the class
        switches are counted in runs, not frames: measured at runs 1, 8 and 85 after a load for runs of 30, 60
        and 150 frames alike). Returns (reference run, runs used, class hashes per full run)."""
        t0 = time.time()
        for r in range(short_runs):
            self.run(cmds[:short_frames])
        if short_runs:
            log(f"    {short_runs} short warm-up runs ({short_frames} frames) in {time.time() - t0:.0f} s; "
                f"{self.n_runs} runs since the load")
        prev, hashes, streak = None, [], 1
        for r in range(max_runs):
            t0 = time.time()
            out = self.run(cmds)
            hashes.append(state_hash(out))
            same = prev is not None and self.same_as(out, prev, range(len(cmds)))
            streak = streak + 1 if same else 1
            log(f"    warm-up run {r}: {time.time() - t0:.1f} s class {hashes[-1]} same as previous: {same}")
            if streak >= n_same and r + 1 >= min_runs:
                self.ref = out
                return out, self.n_runs, hashes
            prev = out
        raise RuntimeError(f"no two consecutive identical runs in {max_runs}")


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "scripts", "dataset"))
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--plans", nargs="+", required=True, help="collect_ik --plan JSON files, run in one scene")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--centre_from", required=True)
    ap.add_argument("--scene", default="medauroidea_c10f10t10.ttt")
    ap.add_argument("--cycles", type=float, default=8.8, help="cycles per 66 frames")
    ap.add_argument("--min_warm", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    centre = np.load(a.centre_from)["actions"].astype(np.float64).mean(0)
    plans = [{k: np.asarray(v, float) for k, v in json.load(open(p)).items()} for p in a.plans]
    drive_kw = dict(travel=0.0, warmup=20, cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), ego=True, ego_box=40.0,
                    ego_seed=0, cam_fov=90.0, capture_frames=False)
    cpg_kw = dict(cycles=a.cycles * a.frames / CI.EP, mirror_joints=(0, 1, 2), spin_amp=0.25, symmetric=False,
                  legtune=None)
    R = SceneReuse(sim, a.scene, drive_kw)
    cmds = R.commands(plans, a.frames, centre, cpg_kw)
    ref, n, _ = R.warm_up(cmds[0], a.min_warm)
    os.makedirs(a.out, exist_ok=True)
    for p, c in zip(a.plans, cmds):
        out = R.run(c)
        dst = os.path.join(a.out, os.path.basename(p)[:-5] + ".npz")
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **out, det_class=state_hash(ref), det_warmup_runs=n, det_port=a.port)
        os.replace(tmp, dst)
        print(f"{p} -> {dst}")


if __name__ == "__main__":
    main()
