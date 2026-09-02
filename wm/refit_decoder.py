"""Refit the MotionDecoder alone, against the ceiling F173 pre-registered.

    .venv/bin/python3 -m wm.refit_decoder --ckpt wm/runs/beh12_ego/best.pt \\
        --sources hexapod=data/egocentric/beh12_c08f09t09_ego_flat \\
                  b1=data/egocentric/beh12_b1_ego_flat

**This is the step that decides whether teacher-student has a base, and it is not a training
improvement.** Trained on egocentric, the decoder reached 0.076 on train motion and never left 1.53
on validation -- worse than predicting the mean. F173 then put a dual ridge on the head's own input
and read **0.608 on the insect and 0.334 on the B1**, against a trained head at -0.53. **The
information was in front of it and it extracted none of it**, so the curve is a training failure and
the head is refittable. This refits it and checks that claim.

**Pass is reaching the reference, not reaching 1.0.** 0.334 is what a *linear* readout finds in an
egocentric B1 frame plus `z`; a refit landing near 0.3 has recovered what is there. Reading it
against 1.0 turns a success into a near-total failure, which is the standing rule this script prints
in its own verdict so the number cannot be quoted without it.

**It is a lower bound and the word "ceiling" was wrong.** The ridge is linear and this head is not.
Run on the allocentric checkpoint as a control, the refit reaches **0.982 against a 0.938 reference
and 0.910 against 0.789** -- 105% and 115%. So the egocentric refit is expected slightly *above*
0.608 and 0.334, and a row landing far under its reference is the stop.

**Everything but the decoder is frozen.** The ITM supplies `z = ITM(e_t, e_t+1)` under `no_grad`, so
the latent this is scored on is the one GATE C was measured on and not a new one -- refitting the
head and the latent together would make the result incomparable with both F172 and the ceiling.

**The split is the ceiling's split, exactly**: by clip, in behaviour families, the odd-indexed clip
of each family held out. A within-clip split on periodic locomotion scores the neighbouring frame.
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from residual_structure import FAMILY  # noqa: E402
from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.evaluate import upgrade_decoder_state  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

CEILING = {"hexapod": 0.608, "b1": 0.334}


def collect(name, directory, ck, cfg, itm, cache_path, chunk, stride, device):
    """Per transition: the frame, the frozen latent, the command, and which clip it came from."""
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, directory), name, encoder, ck, cache,
                   chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    E, Z, A, clip_id = [], [], [], []
    for ci, c in enumerate(clips):
        e = c["e"].float()
        if len(e) < 4:
            continue
        for t in range(1, len(e) - 2, stride):
            E.append(e[t].half())
            with torch.no_grad():
                Z.append(itm(e[t:t + 1].to(device), e[t + 1:t + 2].to(device))[0].float().cpu())
            A.append(c["a"][t].flatten().float())
            clip_id.append(ci)

    # the ceiling's split, reproduced rather than re-invented
    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id)):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    held = {ci for v in order.values() for ci in v[1::2]}
    te = torch.tensor([c in held for c in clip_id])
    return torch.stack(E), torch.stack(Z), torch.stack(A), te, len(clips)


def r2(pred, true, train_mean):
    """The ceiling's metric: standardised per dimension, residual over variance about the train mean."""
    ss = ((pred - true) ** 2).sum()
    return float(1 - ss / max(float(((true - train_mean) ** 2).sum()), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sources", nargs="+", required=True, metavar="EMBODIMENT=DIR")
    ap.add_argument("--cache_dir", default="results/wm/cache")
    ap.add_argument("--cache_prefix", default="ego")
    ap.add_argument("--cache", nargs="+", default=[], metavar="EMBODIMENT=PATH",
                    help="point one embodiment at an existing cache instead of "
                         "`<cache_dir>/<prefix>_<embodiment>.pt`. **The egocentric hexapod cache is "
                         "`ego_hex.pt`, not `ego_hexapod.pt`** -- GATE D wrote it under the short "
                         "name, and without this the run silently re-encodes 48 clips and writes a "
                         "second multi-gigabyte copy of embeddings that already exist.")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2,
                    help="**the knob the original run did not have enough of.** Train motion 0.076 "
                         "against validation 1.53 is memorisation, so a refit that changes nothing "
                         "else would reproduce it.")
    ap.add_argument("--ceiling", nargs="+", default=[], metavar="EMBODIMENT=R2",
                    help="override the pre-registered ceiling, for the allocentric control arm "
                         "where it is 0.938 / 0.789 rather than 0.608 / 0.334. **Not a knob for "
                         "the egocentric run** -- F173 fixed those two numbers before this script "
                         "existed, which is the whole point of having fixed them.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for spec in args.ceiling:
        name, value = spec.split("=", 1)
        CEILING[name] = float(value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)

    overrides = dict(spec.split("=", 1) for spec in args.cache)
    data = {}
    for spec in args.sources:
        name, directory = spec.split("=", 1)
        cache = os.path.join(ROOT, overrides.get(
            name, os.path.join(args.cache_dir, f"{args.cache_prefix}_{name}.pt")))
        data[name] = collect(name, directory, ck, cfg, itm, cache,
                             args.chunk, args.stride, device)
        E, Z, A, te, n = data[name]
        print(f"{name:<10}{n} clips from {directory}, "
              f"{int((~te).sum())} train / {int(te.sum())} test transitions, split by clip")

    # standardise on the training half only, per embodiment, exactly as the ceiling did
    stats = {}
    for name, (E, Z, A, te, _) in data.items():
        tr = ~te
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
        stats[name] = (mu, sd, A[tr].sub(mu).div(sd).mean(0))

    heads = {name: data[name][2].shape[1] for name in data}
    md = MotionDecoder(cfg, heads).to(device)
    md.load_state_dict(upgrade_decoder_state(ck["md"]), strict=False)
    # **Reinitialised heads, kept backbone.** The backbone is what read the frame at all; the heads
    # are what memorised. Loading then reinitialising is deliberate -- starting the whole decoder
    # from scratch would confound "the head can be refitted" with "the backbone was the problem".
    for head in md.heads.values():
        for m in head.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02); torch.nn.init.zeros_(m.bias)
    opt = torch.optim.AdamW(md.parameters(), lr=args.lr, weight_decay=args.wd)

    print(f"\n  reference, pre-registered in F173 (a linear lower bound, not a ceiling): " +
          "  ".join(f"{k} {v:.3f}" for k, v in CEILING.items()))
    print("  **Pass is reaching the reference, not 1.0. Allocentric control: 105% / 115%.**\n")
    print(f"  {'epoch':>7}" + "".join(f"{n + ' train':>16}{n + ' test':>15}" for n in data))

    best = {n: -1e9 for n in data}
    for epoch in range(1, args.epochs + 1):
        md.train()
        for name, (E, Z, A, te, _) in data.items():
            mu, sd, _ = stats[name]
            idx = torch.nonzero(~te).flatten()
            idx = idx[torch.randperm(len(idx))]
            for i in range(0, len(idx), args.batch):
                sl = idx[i:i + args.batch]
                pred = md(E[sl].to(device).float(), Z[sl].to(device), name)
                target = A[sl].sub(mu).div(sd).to(device)
                loss = torch.nn.functional.mse_loss(pred, target)
                opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 10 and epoch != args.epochs:
            continue
        md.eval()
        line = f"  {epoch:>7}"
        for name, (E, Z, A, te, _) in data.items():
            mu, sd, tr_mean = stats[name]
            scores = []
            for mask in (~te, te):
                idx = torch.nonzero(mask).flatten()
                preds = []
                with torch.no_grad():
                    for i in range(0, len(idx), 128):
                        sl = idx[i:i + 128]
                        preds.append(md(E[sl].to(device).float(), Z[sl].to(device), name).cpu())
                scores.append(r2(torch.cat(preds), A[idx].sub(mu).div(sd), tr_mean))
            best[name] = max(best[name], scores[1])
            line += f"{scores[0]:>16.3f}{scores[1]:>15.3f}"
        print(line)

    print(f"\n  {'embodiment':>12}{'best test R2':>14}{'reference':>12}{'of reference':>15}")
    for name in data:
        c = CEILING.get(name)
        share = f"{best[name] / c:>14.0%}" if c else f"{'--':>14}"
        print(f"  {name:>12}{best[name]:>14.3f}{(c if c else float('nan')):>12.3f}{share}")

    print("\n  **Read every row against its own reference, never against 1.0.** 0.334 is what a")
    print("  LINEAR readout finds in an egocentric B1 frame plus z, so a refit near 0.3 has")
    print("  recovered what is there and is passing. **The reference is a lower bound, not a")
    print("  ceiling**: this head is nonlinear, and the allocentric control reaches 105% and 115%")
    print("  of its own ridge, so landing slightly above is expected rather than suspicious.")
    print("  A row far UNDER its reference is the stop -- the head is still not extracting what the")
    print("  ridge proved is present, teacher-student has no base, and that is a stop rather than a")
    print("  tuning problem to iterate on quietly.")

    if args.out:
        torch.save({"md": md.state_dict(), "config": ck["config"], "best_test_r2": best},
                   os.path.join(ROOT, args.out))
        print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
