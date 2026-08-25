"""Can the forward model tell a good candidate action from a bad one?

Every sensitivity number so far -- `sweep z`, frame dominance, the projector's rollout gap -- is a
**ratio measured on a scale with no natural zero**. None of them answers the question the closed
loop actually asks, which is a ranking:

    given K candidate action sequences, one of them the true one,
    does rolling the forward model put the true one first?

That is what a planner does, and it is pass/fail against a chance level of 1/K. **Only the action
projector and the FDM are used** -- the inverse model needs the next frame and can never run in the
loop, which is the constraint LAC-WM states and answers the same way.

**Two arms, and the difference between them is the diagnosis.**

    projector   z = projector(a)          the deployment path
    latent      z = ITM(e_t, e_t+1) of the candidate's *own* clip   an upper bound

If `latent` ranks well and `projector` does not, the projector is the bottleneck and the source
method's stage 3 -- jointly fine-tuning projector and FDM -- is the fix. **If neither ranks well,
the forward model is the bottleneck** and no amount of projector work helps.

**Distractors come from different behaviour conditions**, never from the same one, or a "wrong"
candidate could be a near-duplicate of the right answer. The true candidate is scored from the same
starting frame as every distractor, so the frame cannot break the tie on its own.

  .venv/bin/python3 scripts/diagnostics/plan_discriminates.py --ckpt wm/runs/beh12_hexonly/best.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.action_projector import ActionProjector  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def condition_of(path):
    with np.load(path, allow_pickle=True) as data:
        if "condition" in data.files:
            return str(data["condition"])
    return os.path.basename(path)


def load_clips(paths, name, encoder, checkpoint, cache, chunk, device):
    """Embeddings on the CPU, actions and condition per clip.

    Embeddings stay off the device for the same reason as in `wm/fit_projector.py`: one clip is
    about 94 MB and the whole set does not fit alongside the model.
    """
    out = []
    offset = offset_for(checkpoint, name)
    for path in paths:
        clip = load(path, REGISTRY[name])
        if path not in cache:
            cache[path] = encode_clip(encoder, clip["frames"], chunk).cpu().half()
        e = cache[path].float()
        if offset is not None:
            e = e - offset
        out.append({"e": e.half(), "a": clip["actions"].astype(np.float32),
                    "cond": condition_of(path), "path": path})
    return out


@torch.no_grad()
def roll(ftm, e0, z_seq):
    """Close the forward model on its own output for len(z_seq) steps."""
    e = e0
    for i in range(len(z_seq)):
        e = ftm(e, z_seq[i:i + 1])
    return e


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--projector", default="",
                    help="defaults to projector.pt beside --ckpt, written by wm.fit_projector")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--data", default="",
                    help="override the held-out clips recorded in the projector checkpoint. Using "
                         "clips the projector was fitted on inflates the projector arm.")
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--trials", type=int, default=200, help="start frames sampled across clips")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = os.path.join(ROOT, args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval()
    ftm.load_state_dict(checkpoint["ftm"])

    proj_path = args.projector or os.path.join(os.path.dirname(ckpt_path), "projector.pt")
    saved = torch.load(proj_path, map_location="cpu", weights_only=False)
    if "val_paths" not in saved:
        raise SystemExit(f"{os.path.relpath(proj_path, ROOT)} predates held-out clip recording. "
                         "Refit with `.venv/bin/python3 -m wm.fit_projector --ckpt <ckpt>`.")
    proj = ActionProjector(cfg, saved["action_dims"]).to(device).eval()
    proj.load_state_dict(saved["projector"])

    name = args.embodiment
    paths = (sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz"))) if args.data
             else saved["val_paths"][name])
    if len(paths) < 2:
        raise SystemExit(f"need at least two clips, got {len(paths)}")

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    cache = {}
    clips = load_clips(paths, name, encoder, checkpoint, cache, args.chunk, device)
    del encoder
    torch.cuda.empty_cache()

    conds = sorted({c["cond"] for c in clips})
    print(f"{name}: {len(clips)} held-out clips over {len(conds)} conditions"
          f"{'  (--data override)' if args.data else '  (projector held-out set)'}")
    if len(conds) < 2:
        raise SystemExit("every held-out clip is the same condition; distractors would be "
                         "near-duplicates of the answer. Widen the split or pass --data.")

    lag = max(1, cfg.action_lag)
    rng = np.random.default_rng(args.seed)
    hmax = max(args.horizons)
    # (arm, horizon) -> [rank of the true candidate, one per trial]
    ranks = {(arm, h): [] for arm in ("projector", "latent") for h in args.horizons}

    for _ in range(args.trials):
        ci = int(rng.integers(len(clips)))
        clip = clips[ci]
        span = min(len(clip["e"]) - hmax - 1, len(clip["a"]) - lag - hmax)
        if span <= 1:
            continue
        t = int(rng.integers(1, span))

        # the true candidate, then distractors drawn from *other* conditions
        picks = [(ci, t)]
        others = [i for i, c in enumerate(clips) if c["cond"] != clip["cond"]]
        if len(others) < args.candidates - 1:
            raise SystemExit(f"only {len(others)} clips of a different condition; "
                             f"--candidates {args.candidates} cannot be filled")
        for j in rng.choice(len(others), size=args.candidates - 1, replace=False):
            other = clips[others[j]]
            ospan = min(len(other["e"]) - hmax - 1, len(other["a"]) - lag - hmax)
            picks.append((others[j], int(rng.integers(1, max(2, ospan)))))

        e0 = clip["e"][t:t + 1].to(device).float()
        for h in args.horizons:
            truth = clip["e"][t + h].to(device).float()
            scores = {"projector": [], "latent": []}
            for (pi, pt) in picks:
                cand = clips[pi]
                a = torch.as_tensor(cand["a"][pt + lag:pt + lag + h], device=device)
                scores["projector"].append(
                    float(((roll(ftm, e0, proj(a, name)) - truth) ** 2).mean()))
                # the candidate's own latent, inferred from its own clip -- the best `z` any
                # projector could possibly produce for that action sequence
                ce = cand["e"][pt:pt + h + 1].to(device).float()
                z = torch.cat([itm(ce[i:i + 1], ce[i + 1:i + 2]) for i in range(h)])
                scores["latent"].append(float(((roll(ftm, e0, z) - truth) ** 2).mean()))
            for arm, sc in scores.items():
                # rank 1 = the true candidate scored best
                ranks[(arm, h)].append(1 + int(np.sum(np.asarray(sc) < sc[0])))

    chance = 1.0 / args.candidates
    print(f"\n{args.candidates} candidates, chance top-1 = {chance:.1%}, "
          f"chance mean rank = {(args.candidates + 1) / 2:.1f}\n")
    print(f"{'arm':<12}{'horizon':>9}{'top-1':>9}{'top-2':>9}{'mean rank':>12}{'trials':>9}")
    for arm in ("latent", "projector"):
        for h in args.horizons:
            r = np.asarray(ranks[(arm, h)])
            if not len(r):
                continue
            print(f"{arm:<12}{h:>9}{np.mean(r == 1):>9.1%}{np.mean(r <= 2):>9.1%}"
                  f"{np.mean(r):>12.2f}{len(r):>9}")

    print("\n`latent` is the ceiling: the best z any projector could emit for that action.")
    print("`projector` is the deployment path. **latent good / projector bad** means the projector")
    print("is the bottleneck and stage 3 is the fix. **both at chance** means the forward model")
    print("cannot rank candidates at all, and no projector work changes that.")


if __name__ == "__main__":
    main()
