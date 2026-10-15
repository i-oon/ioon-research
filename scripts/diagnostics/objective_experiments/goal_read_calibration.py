"""Can the goal READ FROM VIDEO be made closer to the true goal without training? (2026-10-15, protocol fixed before running)

The physics loops' vision goal (`wm.policy.planner.vision_goal_froude`: ITM + Froude head on frame pairs, centred window
average) reads strong motions too small (F349, regression to the mean F340). Offline test, cached grayscale embeddings,
no GPU needed:
  read       exactly `vision_goal_froude` with each controlled body's checkpoint (B1 planner: ckpt/b1.pt; hexapod: hex.pt)
  calibrate  per channel true ~ a * read + b, fitted on the 48 c10 TRAINING clips only, applied to the 24 held-out goals
  window     21 frames (current) vs 22 frames (2 c10 strides of ~0.55 s, Ajan Blink's whole-stride averaging)
Reported on the 24 held-out goal clips (the physics goals are 6 of them): per-frame L2 to the recorded goal (as graded in
physics), per-channel slope of read vs true, and the 6 physics goals separately.
PASS (fixed before running): mean held-out L2 down >= 20 % for BOTH checkpoints and none of the 6 physics goals worse by
> 0.01 -> then rerun the vision-goal physics with the calibration.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/goal_read_calibration.py   # -> results/check/goal_read_calibration.txt
"""
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
CK = os.path.join(ROOT, "results/eval/round1_branches_s0_gray_imread/ckpt")
HELD = os.path.join(ROOT, "results/wm/cache/test_v4_goals_c10_heldout_gray.pt")
TRAIN = os.path.join(ROOT, "results/wm/cache/fitproj_files_gray")
PHYS = ("40101", "40110", "40163", "40212", "40012", "40030")
OUT = os.path.join(ROOT, "results/check/goal_read_calibration.txt")


def clips():
    held = {p: v for p, v in torch.load(HELD, map_location="cpu", weights_only=False).items() if p != "__stat__"}
    train = {}
    for f in sorted(glob.glob(os.path.join(TRAIN, "*c10_clips_train*"))):
        p = os.path.basename(f)[:-3].replace("__", "/")
        train[p] = torch.load(f, map_location="cpu", weights_only=False)["e"]
    return train, held


def main():
    from wm.policy.planner import RolloutFroudePlanner, vision_goal_froude
    from wm.data.embodiment import REGISTRY, load
    from wm.evaluate import offset_for
    train, held = clips()
    true = {p: np.asarray(load(p, REGISTRY["hexapod"])["body_motion"])[:, :3] for p in list(train) + list(held)}
    L = [f"Goal read from video vs recorded goal; calibration fitted on {len(train)} c10 training clips, tested on "
         f"{len(held)} held-out goals; per-frame L2 (as graded in physics); slope = read vs true per channel (1 = right size)"]
    verdict = []
    for body, ck in (("B1 planner", "b1.pt"), ("hexapod planner", "hex.pt")):
        emb = "b1" if ck == "b1.pt" else "hexapod"
        rp = RolloutFroudePlanner.from_checkpoint(os.path.join(CK, ck), os.path.join(
            ROOT, "data/counterfactual_walks/b1_clips_heldout" if emb == "b1" else "data/counterfactual_walks/c10_clips_train"),
            embodiment=emb, horizon=2, per_condition=999, device="cpu")
        off = offset_for(torch.load(os.path.join(CK, ck), map_location="cpu", weights_only=False), "hexapod")
        for win in (21, 22):
            rd = {p: vision_goal_froude(rp, e.float(), win, off) for p, e in list(train.items()) + list(held.items())}
            X = np.concatenate([rd[p][:len(true[p])] for p in train]); Y = np.concatenate([true[p][:len(rd[p])] for p in train])
            ab = [np.polyfit(X[:, j], Y[:, j], 1) for j in range(3)]               # true = a * read + b, train clips only
            cal = lambda r: np.stack([ab[j][0] * r[:, j] + ab[j][1] for j in range(3)], 1)  # noqa: E731
            res = {}
            for tag, fn in (("raw", lambda r: r), ("calibrated", cal)):
                err, per, sl = [], {}, []
                Rh = np.concatenate([fn(rd[p])[:len(true[p])] for p in held]); Th = np.concatenate([true[p][:len(rd[p])] for p in held])
                sl = [np.polyfit(Th[:, j], Rh[:, j], 1)[0] for j in range(3)]
                for p in held:
                    n = min(len(rd[p]), len(true[p])); e = float(np.linalg.norm(fn(rd[p])[:n] - true[p][:n], axis=1).mean())
                    err.append(e)
                    ep = os.path.basename(p)[10:15]
                    if ep in PHYS:
                        per[ep] = e
                res[tag] = (np.mean(err), per, sl)
            L.append(f"--- {body} ({ck}), window {win}: calibration a (fwd / lat / yaw) "
                     + " / ".join(f"{a[0]:.2f}" for a in ab))
            for tag in ("raw", "calibrated"):
                m, per, sl = res[tag]
                L.append(f"  {tag:10s} mean L2 {m:.4f} | slope " + " / ".join(f"{v:.2f}" for v in sl) + " | physics goals "
                         + " ".join(f"{k}:{per[k]:.3f}" for k in PHYS if k in per))
            drop = 1 - res["calibrated"][0] / res["raw"][0]
            worse = max(res["calibrated"][1][k] - res["raw"][1][k] for k in res["raw"][1])
            L.append(f"  -> mean L2 {100 * drop:.0f} % LOWER; worst physics-goal change {worse:+.3f} (negative = better)")
            if win == 21:
                verdict.append(drop >= 0.20 and worse <= 0.01)
    L.append("PASS (both checkpoints, window 21: >= 20 % lower and no physics goal worse by > 0.01): "
             + ("YES -> rerun the vision-goal physics with the calibration" if all(verdict) else "NO"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
