"""Balance the wall data across bodies (DATA_PLAN 13 fairness rule, 2026-10-14).

Target per split and cell (distance band x approach group x side): the B1 cell count of the REFERENCE set (the B1 data as
collected before the balance top-up: train 157 / val 30 / test 57 start states), split evenly between left and right
(odd: the extra one left). Both bodies are brought to exactly these counts by choosing START STATES (a state = one branch
point, all its 28 command files): within a cell, states from episodes not yet used are preferred (most independent data),
then deterministic by episode number. Unused states' branch files are MOVED (not deleted) to
`<DATA>/_physics/_unused_balance/<split>/`; their episode files stay (other states of the episode may be kept). Rendered
copies, if any, move to `_unused_balance/rendered_<split>/`. A cell without enough states is reported as SHORT.

    .venv/bin/python3 scripts/dataset/balance_wall_data.py --body c10 [--apply]   # default: dry run (prints the plan)
The reference counts are stored once in results/check/wall_balance_targets.json (from B1's reference episodes).
"""
import argparse
import collections
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts/dataset", "sim/collect", "sim/control", "sim/render"):
    sys.path.insert(0, os.path.join(ROOT, p))
import check_wall_data as G  # noqa: E402

TARGETS = os.path.join(ROOT, "results/check/wall_balance_targets.json")
B1_REF_MAX_EP = {"train": 8042, "val": 8532, "test": 8833}     # B1 as collected 2026-10-12 (before the balance top-up)


def states(W, split):
    D = os.path.join(W.DATA, "_physics", split)
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "ep*_b*_c00.npz"))):
        y = np.load(f, allow_pickle=True)
        grp, side, cs = G.state_geometry(W, W.F, y, W.PREFIX)
        s = cs if grp == "corner" else side
        base = os.path.basename(f)
        rows.append(dict(ep=int(base[2:7]), b=int(base.split("_b")[1].split("_")[0]), stem=base[:-8],
                         band=W.band_of(float(y["start_clearance"])), grp=grp, side="L" if s > 0 else "R"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=("b1", "c10"))
    ap.add_argument("--apply", action="store_true", help="move the unused states (default: dry run)")
    a = ap.parse_args()
    if a.body == "c10":
        import collect_c10_wall_data as C
        W = C.W
    else:
        import collect_b1_wall_data as W
    if not os.path.exists(TARGETS):
        if a.body != "b1":
            raise SystemExit("run --body b1 first: the targets come from B1's reference episodes")
        tg = {}
        for split in ("train", "val", "test"):
            ref = [r for r in states(W, split) if r["ep"] <= B1_REF_MAX_EP[split]]
            cell = collections.Counter((r["band"], r["grp"]) for r in ref)
            tg[split] = {f"{bd}|{g}|{sd}": (n + 1) // 2 if sd == "L" else n // 2
                         for (bd, g), n in cell.items() for sd in "LR"}
        os.makedirs(os.path.dirname(TARGETS), exist_ok=True)
        json.dump(tg, open(TARGETS, "w"), indent=1)
        print("targets ->", os.path.relpath(TARGETS, ROOT))
    tg = json.load(open(TARGETS))
    total_short = 0
    for split in ("train", "val", "test"):
        rows = states(W, split)
        by = collections.defaultdict(list)
        for r in rows:
            by[f"{r['band']}|{r['grp']}|{r['side']}"].append(r)
        keep, used_ep, short = [], collections.Counter(), {}
        for key in sorted(tg[split], key=lambda k: len(by.get(k, [])) - tg[split][k]):   # scarcest cells first
            need = tg[split][key]; pool = list(by.get(key, []))
            pick = []
            while len(pick) < need and pool:
                pool.sort(key=lambda r: (used_ep[r["ep"]], r["ep"], r["b"]))
                r = pool.pop(0); pick.append(r); used_ep[r["ep"]] += 1
            if len(pick) < need:
                short[key] = need - len(pick)
            keep += pick
        kept = {(r["ep"], r["b"]) for r in keep}
        drop = [r for r in rows if (r["ep"], r["b"]) not in kept]
        extra_cells = sorted({f"{r['band']}|{r['grp']}|{r['side']}" for r in rows} - set(tg[split]))
        print(f"== {a.body} {split}: {len(rows)} states -> keep {len(keep)} (target {sum(tg[split].values())}), "
              f"move {len(drop)}; episodes kept {len(used_ep)}" + (f"; cells not in the target: {extra_cells}" if extra_cells else ""))
        print("   SHORT: " + (", ".join(f"{k} {v}" for k, v in sorted(short.items())) if short else "none"))
        total_short += sum(short.values())
        if a.apply and drop:
            P = os.path.join(W.DATA, "_physics")
            U = os.path.join(P, "_unused_balance", split); UR = os.path.join(P, "_unused_balance", f"rendered_{split}")
            os.makedirs(U, exist_ok=True)
            for r in drop:
                for f in glob.glob(os.path.join(P, split, r["stem"] + "_c*.npz")):
                    shutil.move(f, os.path.join(U, os.path.basename(f)))
                for f in glob.glob(os.path.join(W.DATA, split, r["stem"] + "_c*.npz")):
                    os.makedirs(UR, exist_ok=True); shutil.move(f, os.path.join(UR, os.path.basename(f)))
            print(f"   moved {len(drop)} states to {os.path.relpath(U, ROOT)}")
    print("BALANCE", "EXACT" if total_short == 0 else f"{total_short} STATES SHORT", "(dry run)" if not a.apply else "")


if __name__ == "__main__":
    main()
