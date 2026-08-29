"""Run the planner over recorded frames, before any simulator is involved.

The closed loop has two independent things that can be wrong -- the planner's choice and the
execution -- and this exercises the first alone. Frames come from a held-out clip instead of from
CoppeliaSim; everything else is the control-time path, projector and forward model only.

    at each step t of a demonstration clip:
        e_t     the demonstration's own frame          (in the loop this comes from the camera)
        e_goal  the demonstration h steps ahead        (what we are trying to reach)
        the planner scores every behaviour and picks one

**Success is picking the demonstration's own condition.** A planner that cannot do that on the
demonstration's own frames cannot do it on frames its own actions produced, so this is a necessary
condition and a cheap one.

Two numbers, and the second is the one to trust:

  exact      the chosen candidate is the demonstration's condition
  family     the chosen candidate is the right *behaviour* -- speed, turn or sideways -- whatever
             the level. F90 measured level discrimination at 57.8% against a 25% chance, so a
             planner that gets the family right and the level wrong is behaving as measured rather
             than failing.

  .venv/bin/python3 scripts/diagnostics/plan_open_loop.py --ckpt wm/runs/beh12_hexonly/best.pt
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.policy.planner import LatentPlanner, condition_of  # noqa: E402


def family(condition):
    return condition.rsplit("_", 1)[0] if "_" in condition else condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--projector", default="")
    ap.add_argument("--candidates_dir", default="data/beh12_c10f10t10_flat")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--demos", type=int, default=12, help="demonstration clips to run")
    ap.add_argument("--stride", type=int, default=5, help="replan every N frames of the demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = os.path.join(ROOT, args.ckpt)
    planner = LatentPlanner.from_checkpoint(
        ckpt_path, os.path.join(ROOT, args.candidates_dir), args.embodiment,
        os.path.join(ROOT, args.projector) if args.projector else "",
        horizon=args.horizon, device=str(device))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # **Demonstrations must not be the candidates themselves.** `load_candidates` takes the first
    # clip of each condition; taking the demo from the same file would ask the planner to match a
    # sequence against its own frames, which it would do perfectly and mean nothing.
    used = {c["path"] for c in planner.candidates}
    paths = [p for p in sorted(glob.glob(os.path.join(ROOT, args.candidates_dir, "*.npz")))
             if p not in used]
    if not paths:
        raise SystemExit("every clip is in the candidate set; nothing left to demonstrate with")
    rng = np.random.default_rng(args.seed)
    paths = [paths[i] for i in rng.permutation(len(paths))[:args.demos]]

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    offset = offset_for(checkpoint, args.embodiment)
    spec = REGISTRY[args.embodiment]

    print(f"{len(planner.candidates)} candidates, {len(paths)} demonstrations, "
          f"horizon {args.horizon}, replan every {args.stride} frames\n")
    exact, fam, total = 0, 0, 0
    confusion = collections.Counter()
    for path in paths:
        clip = load(path, spec)
        e = encode_clip(encoder, clip["frames"], args.chunk).float()
        if offset is not None:
            e = e - offset.to(e.device)
        want = condition_of(path)
        picks = []
        for t in range(1, len(e) - args.horizon - planner.action_lag - 1, args.stride):
            h = planner.horizon_at(t)
            _, i, _ = planner.act(e[t:t + 1], e[t + h], t)
            got = planner.candidates[i]["condition"]
            picks.append(got)
            exact += got == want
            fam += family(got) == family(want)
            total += 1
            confusion[(want, got)] += 1
        top = collections.Counter(picks).most_common(1)[0]
        print(f"{want:<16} -> {top[0]:<16} {top[1]}/{len(picks)} steps"
              f"{'' if family(top[0]) == family(want) else '   WRONG FAMILY'}")

    print(f"\nexact condition  {exact / max(total, 1):.1%}   "
          f"chance {1 / len(planner.candidates):.1%}")
    fams = len({family(c['condition']) for c in planner.candidates})
    print(f"right behaviour  {fam / max(total, 1):.1%}   chance {1 / fams:.1%}")
    print(f"{total} decisions over {len(paths)} demonstrations")

    worst = [(f"{w} -> {g}", n) for (w, g), n in confusion.most_common()
             if w != g and family(w) != family(g)][:5]
    if worst:
        print("\nmost common cross-behaviour confusions:")
        for label, n in worst:
            print(f"  {label:<40}{n}")


if __name__ == "__main__":
    main()
