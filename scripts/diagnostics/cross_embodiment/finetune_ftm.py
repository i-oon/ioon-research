"""How few clips of a new robot does it take to adapt the forward model to it?

F50 and F51 established that a frozen forward model does not survive a change of robot: an
insect-trained FTM rolled on B1 video scores 0.57-0.71x against holding the frame still, worse
than predicting no motion at all. F51 also showed that is not an architectural limit -- the same
design trained on both robots rolls the B1 at 1.34-1.53x -- and that coverage inside a family
moves the forward model by only 5-8% where it moves the motion decoder by 3.9x.

So the frozen forward model is not the question worth asking. The source method's own transfer is
a three-stage LoRA finetune on 7,265 target trajectories, not zero-shot, and the comparable
question is **sample efficiency**: with N clips of a robot the model has never seen, how good a
forward model can be adapted, and does insect pretraining make that cheaper than starting cold?

    pretrained   ITM and FTM from a Stage 1 checkpoint, fine-tuned on N B1 clips
    scratch      same architecture, random init, trained on the same N clips
    baseline     hold the frame still -- the bar the rollout has to clear to be worth anything

**Two traps this is built around.**

Only the FTM and ITM are adapted; the V-JEPA2 encoder stays frozen, as everywhere else in the
project. Adapting the encoder would change what `e_t` means and make the rollout numbers
incomparable with F32, F51 and slide 12.

And the rollout is scored against **hold-still on the same clips**, not against the pretrained
model's own starting point. A forward model that improves on itself while still losing to a frozen
world has not become useful; 1.0x is the line that matters.

  .venv/bin/python3 scripts/diagnostics/cross_embodiment/finetune_ftm.py --clips 1 3 5 7
"""
import argparse
import copy
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def embeddings_for(encoder, paths, chunk):
    """One frozen embedding sequence per clip, kept on the CPU.

    A clip is 99 x 256 x 1408 floats, about 140 MB; holding all of them on an 11 GB card leaves
    nothing for activations and the first backward pass runs out of memory. Chunks are moved to
    the device as they are used instead.
    """
    out = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            frames = data["frames"]
        out.append(encode_clip(encoder, frames, chunk).cpu())
    return out


def build(ckpt, pretrained, device):
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).to(device)
    ftm = ForwardTransitionModel(cfg).to(device)
    if pretrained:
        itm.load_state_dict(checkpoint["itm"])
        ftm.load_state_dict(checkpoint["ftm"])
    return cfg, itm, ftm


def adapt(itm, ftm, clips, steps, lr, seed, device, batch=8):
    """One-step prediction loss on the target clips, ITM and FTM both trainable.

    One randomly drawn batch of transitions per optimiser step. Only that batch is moved to the
    device, so a clip never has to be resident with its activations -- which was the whole reason
    the embeddings are cached on the CPU.

    **This used to accumulate over every span before stepping**, i.e. full-batch gradient descent,
    so `--steps` counted epochs rather than updates and the cost grew with the clip budget. The
    sweep below then came to 374,400 forward+backward passes and 13 hours on a 2080 Ti for what is
    meant to be a diagnostic. Sampling instead makes the cost per step constant in the budget, so
    the 7-clip cell costs the same as the 1-clip cell and every cell gets the same number of
    updates -- which is also the comparison the table wants, since `pretrained` and `scratch` have
    to be given equal optimisation to be read against each other.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = list(itm.parameters()) + list(ftm.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    spans = [(c, s, min(s + batch, len(c) - 1))
             for c in clips for s in range(0, len(c) - 1, batch)]
    itm.train(); ftm.train()
    last = 0.0
    for step in range(steps):
        e_cpu, s, t = spans[rng.integers(len(spans))]
        opt.zero_grad()
        e_t = e_cpu[s:t].to(device)
        e_next = e_cpu[s + 1:t + 1].to(device)
        z = itm(e_t, e_next)
        loss = torch.nn.functional.mse_loss(ftm(e_t, z), e_next)
        loss.backward()
        opt.step()
        last = loss.item()
        del e_t, e_next, z, loss
    itm.eval(); ftm.eval()
    return last


@torch.no_grad()
def rollout(itm, ftm, clips, horizons, device):
    """Close the forward model on its own output; score against holding the frame still.

    Returns the ratio per horizon and, alongside it, **how far the model predicts the embedding
    moves** as a fraction of how far it really moves. That second number is what separates a model
    that learned the dynamics from one that learned to sit still: an arm predicting no motion
    scores exactly 1.00x by construction, so a ratio near 1.0 is ambiguous until you know whether
    the prediction moved at all.
    """
    scores = {"model": {k: [] for k in horizons}, "hold": {k: [] for k in horizons},
              "moved": {k: [] for k in horizons}, "truth": {k: [] for k in horizons}}
    for e_cpu in clips:
        e = e_cpu.to(device)
        n = len(e)
        z = torch.cat([itm(e[i:i + 1], e[i + 1:i + 2]) for i in range(n - 1)])
        for start in range(1, n - max(horizons) - 1):
            predicted = e[start:start + 1]
            for step in range(1, max(horizons) + 1):
                predicted = ftm(predicted, z[start + step - 1:start + step])
                if step in scores["model"]:
                    truth = e[start + step]
                    scores["model"][step].append(((predicted[0] - truth) ** 2).mean().item())
                    scores["hold"][step].append(((e[start] - truth) ** 2).mean().item())
                    # displacement from the frame the rollout started at, predicted and actual
                    scores["moved"][step].append(((predicted[0] - e[start]) ** 2).mean().item())
                    scores["truth"][step].append(((truth - e[start]) ** 2).mean().item())
        del e
    ratio = {k: float(np.mean(scores["hold"][k])) / float(np.mean(scores["model"][k]))
             for k in horizons}
    moved = {k: float(np.mean(scores["moved"][k])) / float(np.mean(scores["truth"][k]))
             for k in horizons}
    return ratio, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/s1_fwd_m3d_cross/best.pt")
    ap.add_argument("--data", default="data/allocentric/fwd_b1_50hz")
    ap.add_argument("--clips", type=int, nargs="+", default=[1, 3, 5, 7])
    ap.add_argument("--test_clips", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000,
                help="optimiser updates, each on one sampled batch -- not epochs")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--splits", type=int, default=2)
    # `scratch` is random init trained on the target clips: it depends on the config and the
    # splits, never on which checkpoint --ckpt points at. Comparing two pretrained arms therefore
    # recomputes an identical control, which is half the sweep. Measure it once, then pass
    # `--arms pretrained` for the second arm.
    ap.add_argument("--arms", nargs="+", default=["pretrained", "scratch"],
                    choices=["pretrained", "scratch"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--encode_device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    encoder = VJEPA2FrameEncoder(device=args.encode_device, dtype=torch.float32)
    cached = {p: e for p, e in zip(paths, embeddings_for(encoder, paths, args.chunk))}
    del encoder
    torch.cuda.empty_cache()
    print(f"{len(paths)} target clips, encoder frozen, ITM+FTM adapted\n")

    print(f"{'clips':>6} {'model':<12}" + "".join(f"{'h=' + str(h):>9}" for h in args.horizons))
    results = []
    for n in args.clips:
        rows = {a: [] for a in args.arms}
        motion = {a: [] for a in args.arms}
        # `train = order[:n]` and `test = order[-test_clips:]` silently overlap once
        # n + test_clips exceeds the pool, and the ratio then scores partly on clips the model was
        # fitted on. With 14 B1 clips and 4 held out, budget 11 leaked exactly one of the four.
        # Refuse rather than report an inflated number.
        if n + args.test_clips > len(paths):
            raise SystemExit(f"budget {n} + {args.test_clips} test clips exceeds the {len(paths)} "
                             f"available: the split would overlap. Largest clean budget is "
                             f"{len(paths) - args.test_clips}.")
        for split in range(args.splits):
            print(f"  ... {n} clips, split {split + 1}/{args.splits}", flush=True)
            order = np.random.default_rng(args.seed + split).permutation(len(paths))
            train = [cached[paths[i]] for i in order[:n]]
            test = [cached[paths[i]] for i in order[-args.test_clips:]]
            for name in args.arms:
                pretrained = name == "pretrained"
                _, itm, ftm = build(args.ckpt, pretrained, device)
                adapt(itm, ftm, train, args.steps, args.lr, args.seed + split, device)
                ratio, moved = rollout(itm, ftm, test, args.horizons, device)
                rows[name].append(ratio)
                motion[name].append(moved)
                del itm, ftm
                torch.cuda.empty_cache()
        # Report the spread, not only the mean. A 1.05x that a reader cannot separate from the
        # 0.96x beside it is not evidence, and averaging three splits throws away exactly the
        # number that settles it.
        for name in args.arms:
            mean = {h: float(np.mean([r[h] for r in rows[name]])) for h in args.horizons}
            move = {h: float(np.mean([r[h] for r in motion[name]])) for h in args.horizons}
            lo = {h: float(np.min([r[h] for r in rows[name]])) for h in args.horizons}
            hi = {h: float(np.max([r[h] for r in rows[name]])) for h in args.horizons}
            results.append((n, name, mean, lo, hi))
            print(f"{n:>6} {name:<12}"
                  + "".join(f"{mean[h]:>9.2f}x" for h in args.horizons), flush=True)
            print(f"{'':>6} {'  range':<12}"
                  + "".join(f"{lo[h]:>5.2f}-{hi[h]:.2f}" for h in args.horizons), flush=True)
            print(f"{'':>6} {'  moves':<12}"
                  + "".join(f"{move[h]:>9.2f}" for h in args.horizons), flush=True)

    print("\nAbove 1.00x means the adapted forward model beats holding the frame still on the")
    print("target robot. Below it, the rollout is worse than predicting no motion and cannot")
    print("support planning however low its training loss went.")
    print()
    print("`moves` is the predicted displacement from the starting frame as a fraction of the")
    print("actual one. **An arm that predicts no motion scores exactly 1.00x by construction**, so")
    print("a ratio near 1.0 with `moves` near 0 is a model that learned to sit still, not one that")
    print("learned the dynamics -- and it cannot support planning either.")


if __name__ == "__main__":
    main()
