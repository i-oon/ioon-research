"""Per-channel scale / offset / absolute error of the reads in a counterfactual_readout.py --dump (2026-10-09).

Selection compares a read with the goal in ABSOLUTE Froude (L2), so a read that ranks the 24 commands right (Pearson r, what
counterfactual_readout reports) can still select wrongly if it is shrunk or shifted. Per group (one start state, 24 commands):
  slope   least-squares slope of read on truth across the 24 commands (1 = right scale, < 1 = shrunk toward the group mean)
  offset  mean(read) - mean(truth) over the 24 commands (state-level shift)
  r       Pearson across the 24 commands (as counterfactual_readout)
  MAE     mean |read - truth|
and a selection test inside each group: each branch's truth is the goal in turn, the command whose read is nearest (L2 over 3
channels) is picked; error = L2(truth of the pick, goal); also per channel |truth_pick - goal|. Oracle = 0 (the goal's own
command is in the group).

    .venv/bin/python3 scripts/diagnostics/objective_experiments/readout_channels.py results/check/rollout_channels/c10.npz
"""
import sys

import numpy as np

CH = ("fwd", "lat", "yaw")


def stats(T, R):
    sl, off, r, mae = [], [], [], []
    for t, x in zip(T, R):
        tc, xc = t - t.mean(0), x - x.mean(0)
        sl.append((tc * xc).sum(0) / np.maximum((tc * tc).sum(0), 1e-12))
        off.append(x.mean(0) - t.mean(0))
        r.append((tc * xc).sum(0) / np.maximum(np.sqrt((tc * tc).sum(0) * (xc * xc).sum(0)), 1e-12))
        mae.append(np.abs(x - t).mean(0))
    return [np.mean(np.asarray(v), 0) for v in (sl, off, r, mae)]


def select(T, R):
    err, per = [], []
    for t, x in zip(T, R):
        for g in range(len(t)):
            i = int(np.argmin(np.linalg.norm(x - t[g], axis=1)))
            err.append(np.linalg.norm(t[i] - t[g])); per.append(np.abs(t[i] - t[g]))
    return float(np.mean(err)), np.mean(np.asarray(per), 0)


def main(path):
    d = np.load(path)
    names = sorted({k.rsplit("|", 1)[0] for k in d.files if "|" in k})
    for nm in names:
        T = d[f"{nm}|truth"]
        spread = np.mean([t.std(0) for t in T], 0)
        print(f"\n=== {path}  {nm}: {len(T)} groups x 24 commands; true within-state spread (std) "
              + " / ".join(f"{s:.3f}" for s in spread))
        print(f"  {'read':<8} {'slope fwd/lat/yaw':<20} {'offset':<24} {'r':<18} {'MAE':<22} select E  (per-channel error of the pick)")
        for k in ("direct", "real cf", "FTM cf"):
            if f"{nm}|{k}" not in d.files:
                continue
            R = d[f"{nm}|{k}"]
            sl, off, r, mae = stats(T, R)
            e, per = select(T, R)
            f = lambda v, p=2: " / ".join(f"{x:+.{p}f}" for x in v)  # noqa: E731
            print(f"  {k:<8} {f(sl):<20} {f(off, 3):<24} {f(r):<18} {f(mae, 3):<22} {e:.3f}  ({f(per, 3)})")
        e0 = np.mean([np.linalg.norm(t[:, None] - t[None], axis=2).mean() for t in T])
        print(f"  random pick within the state: E {e0:.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
