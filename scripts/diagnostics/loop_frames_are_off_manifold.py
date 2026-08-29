"""Are the frames the loop drives itself into outside what the model was fitted on?

**The gap this is for.** Scored on recorded clips the planner picks the right behaviour family 62%
of the time (F100). Running as a controller it is in the right family on **38-71%** of its steps
(F102). Nothing measured so far separates two explanations:

  * the frames differ -- the loop visits states no recording contains, and the model is worse there;
  * the frames are fine and something else in the loop is wrong.

Two measurements, both on frames already stored inside each closed-loop result:

    novelty     distance from each frame's embedding to its nearest neighbour among the recorded
                clips, against the same statistic for held-out recorded frames. **A held-out clip
                is the control**: it is also unseen, so whatever distance it shows is the floor.

    prediction  the forward model's one-step error on the loop's own trajectory, driven by the
                action the loop actually executed, against its error on recorded clips.

**If the loop's frames are no further out than a held-out clip's and the prediction error matches,
distribution shift is not the explanation** and the search should move elsewhere. If both rise,
it is, and training a policy on the states it reaches is the fix that follows.

    .venv/bin/python3 scripts/diagnostics/loop_frames_are_off_manifold.py \\
        results/wm/closed_loop/b1_physics3/*.npz --data data/beh12_b1_flat --embodiment b1
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


def pooled(e):
    """Mean over tokens. Nearest-neighbour distance over 256x1408 is dominated by texture; the
    pooled vector is what the planner's ranking effectively varies over."""
    return torch.nn.functional.normalize(e.mean(1), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="wm/runs/beh12_hexonly/stage3_b1_nce_v2.pt")
    ap.add_argument("--projector", default="wm/runs/beh12_hexonly/projector_stage3_nce_v2.pt")
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--holdout", type=int, default=4, help="recorded clips kept out of the bank")
    ap.add_argument("--chunk", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    saved = torch.load(os.path.join(ROOT, args.projector), map_location="cpu", weights_only=False)
    proj = ActionProjector(cfg, saved["action_dims"]).to(device).eval()
    proj.load_state_dict(saved["projector"])
    off = offset_for(ck, args.embodiment)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    bank_paths, held_paths = paths[:-args.holdout], paths[-args.holdout:]

    @torch.no_grad()
    def enc(frames):
        e = encode_clip(encoder, frames, args.chunk).float()
        return (e - off) if off is not None else e

    # **Encode everything before the forward model is asked for anything.** The encoder is 300M
    # parameters and the forward model needs the card too; holding both ran out of memory inside
    # attention. Embeddings come back on the CPU in half precision and are moved in batches.
    bank = torch.cat([pooled(enc(load(p, REGISTRY[args.embodiment])["frames"])).half()
                      for p in bank_paths]).float()
    held_e, held_a = [], []
    for p in held_paths:
        clip = load(p, REGISTRY[args.embodiment])
        held_e.append(enc(clip["frames"]).half())
        held_a.append(torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32))
    run_e, run_names = [], []
    for run in args.runs:
        with np.load(run, allow_pickle=True) as d:
            run_e.append((enc(d["frames"]).half(),
                          [str(c) for c in np.asarray(d["chosen"], str)]))
        run_names.append(os.path.basename(run))
    # **The closure holds it too.** `del encoder` alone leaves `enc` referencing the module, so
    # 300M parameters stay resident and the forward model runs out of card.
    del encoder, enc
    torch.cuda.empty_cache()

    @torch.no_grad()
    def novelty(v):
        d = 1.0 - (v @ bank.T)          # cosine distance, both normalised
        return d.min(1).values

    @torch.no_grad()
    def fdm_err(e, actions):
        errs = []
        for i in range(0, len(e) - 1, 8):
            j = min(i + 8, len(e) - 1)
            eb = e[i:j].to(device)
            z = proj(actions[i:j].to(device), args.embodiment)
            errs.append(((ftm(eb, z) - e[i + 1:j + 1].to(device)) ** 2).flatten(1).mean(1).cpu())
        return torch.cat(errs)

    print(f"  bank: {len(bank_paths)} recorded clips | control: {len(held_paths)} held out\n")
    print(f"  {'source':<34}{'novelty':>10}{'1-step error':>14}")

    nov_h, err_h = [], []
    lag = max(1, cfg.action_lag)
    for e_half, a in zip(held_e, held_a):
        e = e_half.float()
        nov_h.append(novelty(pooled(e)))
        n = min(len(e) - 1, len(a) - lag)
        err_h.append(fdm_err(e[:n + 1], a[lag:lag + n]))
    nov_h, err_h = torch.cat(nov_h), torch.cat(err_h)

    # **The control that separates "hard states" from "impossible pairs".** Same held-out frames,
    # but driven by the action another clip issued at the same index. The state is in distribution
    # and the action is a real command; only the *pairing* never occurred. If this rises to what
    # the loop shows, the loop's frames are not the problem -- the pairs it assembles are.
    err_x = []
    for i, (e_half, _a) in enumerate(zip(held_e, held_a)):
        e = e_half.float()
        other = held_a[(i + 1) % len(held_a)]
        n = min(len(e) - 1, len(other) - lag)
        err_x.append(fdm_err(e[:n + 1], other[lag:lag + n]))
    err_x = torch.cat(err_x)
    print(f"  {'held-out recorded clips':<34}{nov_h.mean():>10.4f}{err_h.mean():>14.4f}")
    # **And the same again with the gait phase broken.** The control above keeps the time index,
    # so the borrowed command still arrives at the right point of the stride. The loop does not:
    # switching clips 36-44 times leaves the robot's actual phase drifted from the index it reads
    # the command at. Shifting by three steps -- about a sixth of a stride -- separates "wrong
    # command" from "wrong command at the wrong moment".
    err_p, shift = [], 3
    for i, (e_half, _a) in enumerate(zip(held_e, held_a)):
        e = e_half.float()
        other = held_a[(i + 1) % len(held_a)]
        n = min(len(e) - 1, len(other) - lag - shift)
        err_p.append(fdm_err(e[:n + 1], other[lag + shift:lag + shift + n]))
    err_p = torch.cat(err_p)

    label = "  same frames, another clip's action"
    print(f"  {label:<34}{nov_h.mean():>10.4f}{err_x.mean():>14.4f}"
          f"   {1.0:>5.2f}x  {err_x.mean() / err_h.mean():>5.2f}x")
    label2 = "  the same, 3 steps out of phase"
    print(f"  {label2:<34}{nov_h.mean():>10.4f}{err_p.mean():>14.4f}"
          f"   {1.0:>5.2f}x  {err_p.mean() / err_h.mean():>5.2f}x")

    lib = {}
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            lib[str(z["condition"])] = np.asarray(z["action"] if "action" in z.files
                                                  else z["actions"], np.float32)

    for name, (e_half, chosen) in zip(run_names, run_e):
        e = e_half.float()
        nov = novelty(pooled(e))
        # **`frames[t]` is the state *after* `chosen[t]`.** The loop renders once before acting
        # and does not store that first frame, so the transition `e[t] -> e[t+1]` was caused by
        # `chosen[t+1]`, not `chosen[t]`. Pairing them off by one made the forward model look
        # three times worse on loop frames than on recorded ones, which was this bug and not a
        # property of the loop.
        acts = []
        for t in range(1, min(len(chosen), len(e))):
            cand = chosen[t].split("warm:")[-1]
            src = lib.get(cand)
            acts.append(src[min(t, len(src) - 1)] if src is not None else np.zeros(12, np.float32))
        err = fdm_err(e[:len(acts) + 1], torch.as_tensor(np.array(acts)))
        print(f"  {name:<34}{nov.mean():>10.4f}{err.mean():>14.4f}"
              f"   {nov.mean() / nov_h.mean():>5.2f}x  {err.mean() / err_h.mean():>5.2f}x")

    print("\n  novelty      cosine distance to the nearest recorded frame, pooled embeddings")
    print("  1-step err   forward model predicting the next frame from the executed action")
    print("  the ratios are against the held-out recorded clips, which are also unseen and are")
    print("  therefore the floor rather than zero")


if __name__ == "__main__":
    main()
