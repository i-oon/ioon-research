"""Does insect pretraining transfer locomotion, or only familiarity with the feature space?

F52 showed that ITM and FTM pretrained on stick insects adapt to the B1 from roughly 7x fewer
target clips than the same architecture from random init. `scratch` controls for "any
initialisation at all", but it does not separate two very different readings of that margin:

    the model learned how legged bodies move, and that survives a change of robot   <- our claim
    the model learned the shape of V-JEPA2's feature manifold, nothing about motion <- much weaker

Both predict a pretrained advantage over random weights, so the existing table cannot tell them
apart. This trains two arms that differ in exactly one thing -- what the ITM is given as its
second frame -- and hands both to `finetune_ftm.py` for the identical B1 sweep.

    real        (e_t, e_{t+1}), FTM predicts e_{t+1}.  One timestep.
    shuffled    (e_t, e_s) for a random s in the SAME clip.  Adjacency removed.
    far         (e_t, e_s) restricted to |s - t| >= min_gap.  Long baselines only.

Shuffling *within* a clip is the point. Drawing the partner from another clip would destroy the
body and the scene at the same time as the ordering, and a difference in transfer would then have
three explanations instead of one.

**What `shuffled` does NOT do, and this matters.** It does not remove motion. Its partners average
21.9 frames away and the measured gait cycle is 19 (F53), so a shuffled pair shows the body at two
points of a stride -- a *long-baseline* view of motion, not the absence of one. Measured,
`shuffled` matches `real` at every budget, which therefore licenses only the narrower reading:

    a one-step transition is not the useful window; a stride-scale one carries at least as much

and NOT "motion does not transfer". `far` is the arm that tests the narrow reading directly, by
never showing a short pair at all.

This also settles a tension with slide 11, where a second frame was worth only 1.11x on the
command. That was measured at t+1, where consecutive frames barely differ and the delta sits in
the noise. Two frames half a cycle apart pin phase and direction. The two results agree once the
window is stated.

The two pretraining losses are NOT comparable and should not be read against each other: shuffled
pairs sit further apart in the embedding, so that arm is solving a harder reconstruction. Only the
downstream B1 numbers carry the comparison.

These arms are trained ITM+FTM only, without the decoder or the cross-body term, because those are
not what `finetune_ftm.py` loads. They are therefore weaker than `stage1_m3d_cross` in absolute
terms -- which does not matter, since the comparison is real against shuffled at a matched budget,
not either against the full Stage 1 run.

Run on com7; encoding plus two arms is over an hour of GPU.

  .venv/bin/python3 -u scripts/diagnostics/pretrain_control.py --clips 100 --steps 6000
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def encode_all(paths, chunk, device):
    """Frozen embeddings per clip, held on the CPU in half precision.

    One clip of 66 frames is 66 x 256 x 1408; at float32 a hundred clips is 9.5 GB of host memory
    and at float16 it is 4.8. Batches are cast back to float32 as they go to the device, so the
    models still train in full precision.
    """
    encoder = VJEPA2FrameEncoder(device=device, dtype=torch.float32)
    out = []
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=True) as data:
            frames = data["frames"]
        out.append(encode_clip(encoder, frames, chunk).cpu().half())
        if (i + 1) % 20 == 0:
            print(f"  encoded {i + 1}/{len(paths)} clips", flush=True)
    del encoder
    torch.cuda.empty_cache()
    return out


def train_arm(mode, clips, cfg, steps, lr, batch, seed, device, save, min_gap=10):
    """Train ITM+FTM on `clips`; `mode` decides only which frame partners e_t.

    `save(tag, itm, ftm)` is called at a third and two thirds of the budget as well as at the end.
    The point is not to pick a best checkpoint -- there is no validation here -- but to make the
    question "was this pretrained long enough to mean anything" answerable from the same run. If
    B1 transfer is the same at 2/3 of the budget as at the end, the budget was sufficient; if it is
    still climbing, any comparison between arms is being read off an undertrained pair.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    itm = InverseTransitionModel(cfg).to(device)
    ftm = ForwardTransitionModel(cfg).to(device)
    opt = torch.optim.AdamW(list(itm.parameters()) + list(ftm.parameters()),
                            lr=lr, weight_decay=1e-4)
    milestones = {max(1, steps // 3): "third", max(2, 2 * steps // 3): "twothirds", steps: "best"}
    itm.train(); ftm.train()
    recent = []
    for step in range(steps):
        clip = clips[rng.integers(len(clips))]
        n = len(clip)
        t = rng.integers(0, n - 1, size=min(batch, n - 1))
        # The one line that separates the arms. `real` takes the frame that actually follows;
        # `shuffled` takes any other frame of the same clip, so the pair spans the same manifold
        # with the time ordering removed. Rerolling the collisions keeps `shuffled` from quietly
        # including e_t paired with itself, which is a no-change transition and the one case the
        # two arms would agree on.
        #
        # `far` exists because `shuffled` does not separate what it was built to separate. Its
        # partners average 21.9 frames away against a measured gait cycle of 19 (F53), so it is
        # not a pair without motion -- it is a pair with a *long baseline*, and it mixes short
        # ones in as well. Restricting to |partner - t| >= min_gap asks whether the long baseline
        # alone carries the pretraining benefit, which is the claim that one timestep is the wrong
        # window rather than the claim that motion does not matter.
        if mode == "real":
            partner = t + 1
        elif mode == "shuffled":
            partner = rng.integers(0, n, size=len(t))
            while (collide := partner == t).any():
                partner[collide] = rng.integers(0, n, size=int(collide.sum()))
        else:                                              # `far`: long baselines only
            partner = rng.integers(0, n, size=len(t))
            while (near := np.abs(partner - t) < min_gap).any():
                partner[near] = rng.integers(0, n, size=int(near.sum()))
        e_t = clip[t].to(device).float()
        e_partner = clip[partner].to(device).float()
        z = itm(e_t, e_partner)
        loss = torch.nn.functional.mse_loss(ftm(e_t, z), e_partner)
        opt.zero_grad(); loss.backward(); opt.step()
        recent.append(loss.item())
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"  {mode:<9} step {step + 1}/{steps}  loss {np.mean(recent[-200:]):.5f}",
                  flush=True)
        if (step + 1) in milestones:
            itm.eval(); ftm.eval()
            save(milestones[step + 1], itm, ftm)
            itm.train(); ftm.train()
        del e_t, e_partner, z, loss
    itm.eval(); ftm.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ik_walk_m3d_clean")
    ap.add_argument("--arch_ckpt", default="wm/runs/s1_fwd_m3d_cross/best.pt",
                    help="only its Config is used, so both arms match the run they explain")
    ap.add_argument("--out_dir", default="wm/runs")
    ap.add_argument("--modes", nargs="+", default=["real", "shuffled"],
                    choices=["real", "shuffled", "far"])
    ap.add_argument("--min_gap", type=int, default=10,
                    help="`far` only: smallest |partner - t| allowed, in frames. The gait cycle "
                         "is 19 frames, so 10 is half a cycle.")
    ap.add_argument("--clips", type=int, default=100)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--encode_device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    arch = args.arch_ckpt if os.path.isabs(args.arch_ckpt) else os.path.join(ROOT, args.arch_ckpt)

    checkpoint = torch.load(arch, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])

    # Both arms see the same clips in the same order; the seed decides sampling, not membership.
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise SystemExit(f"no clips in {data_dir}")
    paths = [paths[i] for i in np.random.default_rng(args.seed).permutation(len(paths))]
    paths = paths[:args.clips]
    print(f"{len(paths)} clips from {args.data}, architecture from {args.arch_ckpt}", flush=True)

    gb = sum(66 * 256 * 1408 * 2 for _ in paths) / 1e9
    print(f"caching embeddings on the CPU: about {gb:.1f} GB at float16", flush=True)
    clips = encode_all(paths, args.chunk, args.encode_device)
    print(f"encoded, {sum(len(c) for c in clips)} frames held on the CPU\n", flush=True)

    for mode in args.modes:
        out = os.path.join(ROOT, args.out_dir, f"pretrain_{mode}")
        os.makedirs(out, exist_ok=True)

        def save(tag, itm, ftm, mode=mode, out=out):
            path = os.path.join(out, f"{tag}.pt")
            torch.save({"config": checkpoint["config"], "itm": itm.state_dict(),
                        "ftm": ftm.state_dict(), "epoch": 0,
                        "pretrain_mode": mode, "pretrain_clips": len(paths),
                        "pretrain_steps": args.steps, "pretrain_tag": tag}, path)
            print(f"  -> {path}", flush=True)

        train_arm(mode, clips, cfg, args.steps, args.lr, args.batch, args.seed, device, save,
                  args.min_gap)
        print("", flush=True)
        torch.cuda.empty_cache()

    print("Check first that the real arm learned anything at all -- one budget, two splits:")
    print(f"  .venv/bin/python3 -u scripts/diagnostics/finetune_ftm.py "
          f"--ckpt wm/runs/pretrain_real/best.pt --clips 5 --splits 2")
    print("If its `pretrained` row does not clearly beat `scratch`, the arms are undertrained and")
    print("the comparison below cannot be read. Raise --steps rather than interpreting it.\n")
    print("Then the full sweep against each arm, comparing the `pretrained` rows:")
    for mode in args.modes:
        print(f"  .venv/bin/python3 -u scripts/diagnostics/finetune_ftm.py "
              f"--ckpt wm/runs/pretrain_{mode}/best.pt --clips 1 3 5 7 9 --splits 3")
    print("\nAnd if the two arms differ, confirm the budget was enough by sweeping `twothirds.pt`")
    print("of the real arm: a transfer number still climbing with pretraining budget means the")
    print("comparison was read off an undertrained pair.")


if __name__ == "__main__":
    main()
