"""Does the planner's score separate *how fast*, or only *which behaviour*?

F-speed showed the candidate library is not the limit: on 9 closed-loop runs a behaviour travelling
at the demonstrated rate was in the list every time, and the planner picked it 3 times. So the
fault is in selection. This asks where.

**The mechanism under test.** The score is an L2 distance between a predicted frame embedding and
the goal frame embedding. Two frames showing the same posture at different travel speeds look
almost identical; two frames showing different postures do not. **If that is right, the score is
dominated by which behaviour and has almost nothing left for how fast** -- which is exactly the
symptom: right family, wrong amplitude, on both robots.

Measured two ways from the scores already stored in each run:

    between families   spread of the score across behaviour families
    within a family    spread across the amplitudes of one family

**If the within-family spread is a small fraction of the between-family spread, the score cannot
resolve speed** however good the forward model gets, and no amount of extra candidates helps. The
fix would have to change what is being scored.

    .venv/bin/python3 scripts/diagnostics/planning/does_score_see_speed.py \\
        results/wm/closed_loop/heldout_fewshot/*.npz
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FAMILY = lambda c: c.rsplit("_", 1)[0] if "_" in c else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--demo_dir", default="",
                    help="the robot's clips, to read what each candidate actually achieved")
    args = ap.parse_args()

    print(f"  {'run':<30}{'between':>10}{'within':>9}{'ratio':>8}   most-chosen family")
    ratios = []
    for path in args.runs:
        with np.load(path, allow_pickle=True) as d:
            if "scores" not in d.files:
                print(f"  {os.path.basename(path):<30}  no scores stored")
                continue
            sc = np.asarray(d["scores"], np.float64)
            cands = [str(c) for c in np.asarray(d["candidates"], str)]
            chosen = [str(c) for c in np.asarray(d["chosen"], str)]
        rows = ~np.isnan(sc).any(axis=1)
        sc = sc[rows]
        if len(sc) < 3:
            continue
        fams = np.array([FAMILY(c) for c in cands])

        # **Normalise per step before pooling.** The absolute score drifts with the frame, so a raw
        # spread across steps would measure that drift rather than how the candidates differ.
        z = (sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-9)

        between, within = [], []
        for step in z:
            means = {f: step[fams == f].mean() for f in np.unique(fams)}
            between.append(np.std(list(means.values())))
            w = [step[fams == f].std() for f in np.unique(fams) if (fams == f).sum() > 1]
            within.append(np.mean(w))
        b, w = float(np.mean(between)), float(np.mean(within))
        ratios.append(w / max(b, 1e-9))
        planned = [c for c in chosen if not c.startswith("warm:")]
        top = max(set(FAMILY(c) for c in planned), key=lambda f:
                  sum(FAMILY(c) == f for c in planned)) if planned else "-"
        print(f"  {os.path.basename(path):<30}{b:>10.3f}{w:>9.3f}{w / max(b,1e-9):>8.2f}   {top}")

    # **A wide spread is not the same as a useful one.** If the score varies inside a family but
    # the variation has nothing to do with speed, the picture above looks identical to a score that
    # resolves amplitude. So rank the family's candidates by score, rank them by how well their own
    # recorded speed matches the demonstration, and correlate.
    if args.demo_dir:
        import glob
        from diagnostics.planning.score_closed_loop import channels
        lib = {}
        for p in sorted(glob.glob(os.path.join(args.demo_dir, "*.npz"))):
            with np.load(p, allow_pickle=True) as d:
                head = d["head"] if "head" in d.files else d["base_pos"]
                quat = d["body_quat"] if "body_quat" in d.files else d["base_quat"]
                dt = float(d["dt"]) if "dt" in d.files else 0.05
                emb = str(d["embodiment"]) if "embodiment" in d.files else "hexapod"
                c = channels(head.astype("float64"), quat.astype("float64"), dt, emb)
                lib.setdefault(str(d["condition"]), []).append(np.median(c, 0))
        lib = {k: np.mean(v, 0) for k, v in lib.items()}

        print(f"\n  {'run':<30}{'rank corr within the right family':>36}")
        corrs = []
        for path in args.runs:
            with np.load(path, allow_pickle=True) as d:
                if "scores" not in d.files:
                    continue
                sc = np.asarray(d["scores"], np.float64)
                cands = [str(c) for c in np.asarray(d["candidates"], str)]
                want = str(d["condition"])
            sc = sc[~np.isnan(sc).any(axis=1)]
            fam = FAMILY(want)
            idx = [i for i, c in enumerate(cands) if FAMILY(c) == fam and c in lib]
            if len(idx) < 2 or want not in lib:
                continue
            k = int(np.argmax(np.abs(lib[want])))
            miss = np.array([abs(lib[cands[i]][k] - lib[want][k]) for i in idx])
            per_step = []
            for step in sc:
                a, b = step[idx], miss
                if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                    continue
                ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
                per_step.append(np.corrcoef(ra, rb)[0, 1])
            if per_step:
                corrs.append(float(np.mean(per_step)))
                print(f"  {os.path.basename(path):<30}{corrs[-1]:>36.2f}")
        if corrs:
            m = float(np.mean(corrs))
            print(f"\n  mean rank correlation **{m:+.2f}** -- +1 would mean the score orders the "
                  f"family's\n  amplitudes exactly by how well they match the demonstration, 0 "
                  f"means the ordering\n  carries no speed information at all.\n")
            if abs(m) < 0.2:
                print("**The spread inside a family is noise.** The score moves between amplitudes")
                print("of the same behaviour without tracking which amplitude is right, so it looks")
                print("informative and is not. **Scoring a predicted frame against a goal frame")
                print("does not measure speed**, and the fix has to change the quantity scored --")
                print("not the candidates, not the forward model.")

    if ratios:
        r = float(np.mean(ratios))
        print(f"\n  within-family spread is **{r:.0%}** of the between-family spread, averaged "
              f"over {len(ratios)} runs.\n")
        if r < 0.5:
            print("**The score is a behaviour detector with speed as an afterthought.** Most of its")
            print("range is spent separating families; what is left to separate amplitudes inside a")
            print("family is a fraction of that, so the ranking resolves *what* and not *how fast*.")
            print("More candidates cannot fix this -- the quantity being scored has to change.")
        else:
            print("**The score does carry amplitude information.** Within-family differences are")
            print("comparable to between-family ones, so the failure to hit speed is not the score")
            print("being blind to it -- look at the projector and the goal frame instead.")


if __name__ == "__main__":
    main()
