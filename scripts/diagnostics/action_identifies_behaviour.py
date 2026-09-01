"""Does the action, on its own, say which behaviour is being performed?

**This is the question underneath F93, F96 and F97, asked without a world model in the way.**
Stage 3 fine-tuned the projector and the forward model together and the forward model responded by
**ignoring the action channel entirely** -- its prediction given `proj(a)` matched its prediction
given the mean latent to three decimals, at every checkpoint, while the training loss halved. A
model does not discard an input that helps. That points past "the projector is hard to fit" to
"there is nothing in the action to fit", and that claim is checkable directly.

A planner selects an action because the action implies a behaviour. So: train a classifier from
action alone to condition, and read the accuracy. Chance is 1/12. **If the hexapod's actions
identify its behaviour and the B1's do not, no planner built on the B1's action space can work**,
whatever the world model does, and the three failures are one fact seen from three sides.

Two window sizes, because a single 20 Hz sample is a snapshot of a 50 Hz control signal and a
behaviour may only be visible over a few of them -- which is also the premise of action chunking.

    .venv/bin/python3 scripts/diagnostics/action_identifies_behaviour.py
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

FAMILY = lambda c: c.rsplit("_", 1)[0] if "_" in c else c


def load_dir(d, key):
    clips = []
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            if key not in z:
                continue
            clips.append({"a": np.asarray(z[key], np.float32), "cond": str(z["condition"]),
                          "name": os.path.basename(p)})
    return clips


def windows(clips, ids, conds, w):
    X, y = [], []
    for i in ids:
        a, k = clips[i]["a"], conds[clips[i]["cond"]]
        for t in range(len(a) - w + 1):
            X.append(a[t:t + w].ravel()); y.append(k)
    return torch.tensor(np.array(X)), torch.tensor(y)


def run(clips, label, w, epochs=300, seed=0):
    conds = {c: i for i, c in enumerate(sorted({c["cond"] for c in clips}))}
    inv = {i: c for c, i in conds.items()}
    # hold out one clip per condition -- a frame-level split leaves near-duplicate windows of the
    # same clip on both sides and reports memorisation as accuracy (F76)
    by_cond = {}
    for i, c in enumerate(clips):
        by_cond.setdefault(c["cond"], []).append(i)
    g = np.random.default_rng(seed)
    val_ids = [g.choice(v) for v in by_cond.values()]
    train_ids = [i for i in range(len(clips)) if i not in val_ids]

    Xtr, ytr = windows(clips, train_ids, conds, w)
    Xva, yva = windows(clips, val_ids, conds, w)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(), nn.Linear(256, len(conds)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        nn.functional.cross_entropy(net(Xtr), ytr).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xva).argmax(1)
        exact = (pred == yva).float().mean().item()
        fam = np.mean([FAMILY(inv[int(p)]) == FAMILY(inv[int(t)]) for p, t in zip(pred, yva)])
        # chance for the family score is not 1/N: families hold unequal numbers of conditions
        fam_chance = np.mean([sum(FAMILY(inv[j]) == FAMILY(inv[int(t)]) for j in range(len(conds)))
                              / len(conds) for t in yva])
    print(f"  {label:<26}{w:>3}{exact:>10.0%}{1 / len(conds):>9.0%}"
          f"{fam:>11.0%}{fam_chance:>9.0%}   {len(Xtr)} train windows")
    return exact, fam, fam_chance


def main():
    sets = [("hexapod, joint targets", "data/allocentric/beh12_c10f10t10_flat", "actions"),
            ("B1, policy actions", "data/allocentric/beh12_b1_flat", "action"),
            ("B1, joint positions", "data/allocentric/beh12_b1_flat", "joint_pos")]
    print(f"  {'source':<26}{'win':>3}{'cond':>10}{'chance':>9}{'family':>11}{'chance':>9}")
    out = {}
    for label, d, key in sets:
        clips = load_dir(os.path.join(ROOT, d), key)
        if not clips:
            print(f"  {label:<26}  -- no clips with '{key}' in {d}")
            continue
        for w in (1, 5):
            out[(label, w)] = run(clips, label, w)

    print("\n`cond` names one of twelve conditions from the action alone; `family` allows the right")
    print("behaviour at the wrong amplitude. Both are scored on clips the classifier never saw, and")
    print("both carry their own chance rate -- the family rate is NOT 1/12, because the families")
    print("hold unequal numbers of conditions.\n")
    print("**`B1, joint positions` is the control.** It asks whether the behaviour is present in the")
    print("B1's proprioception at all. If positions identify the behaviour and policy actions do")
    print("not, the information exists on the robot and the action is simply not where it lives --")
    print("which is what 'the action is a response, not a plan' means, stated as a measurement.")


if __name__ == "__main__":
    main()
