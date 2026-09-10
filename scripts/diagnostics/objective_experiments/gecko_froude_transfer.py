"""Does babble-grounding let a Froude goal transfer to gecko at all -- the direct test, no
candidate library, no controller, no new collection.

  .venv/bin/python3 scripts/diagnostics/objective_experiments/gecko_froude_transfer.py

**Why this replaces the selection/mean-rank framing.** F127's benchmark shape (rank N named
candidates) only makes sense when N candidates exist. Hexapod/B1 have 12; gecko has none -- only
a continuous generic-CPG babble sweep, no hand-built behaviour repertoire. Forcing a library onto
gecko would be infrastructure built to match a metric, not to answer the actual question: does the
world model's shared coordinate carry a goal-direction signal to a body it never saw in pretrain,
grounded by babble alone?

**The mechanism, precisely.** Reuse the EXISTING, already-validated `CleanFroudeHead` from
`ftm_froude_stopgrad_probe.py` (`wm/runs/ftm_froude_stopgrad_head_hexapod.pt`) -- trained ONCE, on
hexapod only, reading `pool(FTM(e_t, z).detach())`, predicting `body_motion[t+1]`. Apply it
ZERO-SHOT (no retraining, frozen weights) to:

  - hexapod, held out         -- reproduces the head's own reported number, sanity check
  - B1, held out              -- zero-shot to a body the HEAD never trained on, but the world
                                  model (ITM/FTM) DID see in pretrain -- isolates "does the head
                                  zero-shot generalise across bodies at all"
  - gecko, held out           -- zero-shot to a body neither the head NOR the world model's
                                  pretrain ever saw; z comes only from `fit_gecko_projector`'s
                                  babble-fitted projector -- the actual claim-3 test

Three-body ladder, not two, because a gecko number in isolation cannot be told apart from "heads
don't zero-shot generalise at all" (visible already at B1) versus "babble-grounding specifically
fails" (visible only at gecko, beyond whatever the B1 zero-shot rung already costs).

**Pre-registered reading:**
  - Gecko rho comparable to the B1 zero-shot rung -> transfer survives losing pretrain exposure
    entirely; babble alone grounds a novel morphology into the shared coordinate. Claim (3) holds.
  - Gecko rho far below the B1 rung (near zero) -> babble grounds *some* signal (the 0.252x
    rollout-gap improvement over mean-z already shows that) but it does not survive the trip
    through the shared head to real Froude on a body absent from pretrain. Bounds the claim.
  - Report forward/lateral/yaw separately: forward is the channel that has fought all session.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

# F181's zero-shot numbers (median rho, z=proj) -- the "before" this script's --teacher_ckpt /
# --proj_ckpt overrides are meant to be compared against once a fine-tune has run.
F199_HEX_BASELINE = 0.454
F199_B1_BASELINE = 0.427
F199_GECKO_ZEROSHOT = 0.113

# **Pre-registered bars for the fine-tune test (this session, both operationalising a qualitative
# call into a checkable number).** Forgetting gate: hex/B1 must not have degraded materially --
# 80% of their zero-shot number, since the fine-tune's whole point is adding gecko without paying
# for it elsewhere. Recovery bar: gecko must close most of the gap to the B1/hex range, not just
# move off its zero-shot floor -- 80% of B1's number (the nearer, not the higher, reference) is the
# line between "recovered toward the in-pretrain range" and "improved but still a different regime".
FORGET_TOL = 0.8
RECOVER_FRAC = 0.8

DIRS = {
    "hexapod": os.path.join(ROOT, "data/egocentric/beh12_c10f10t10_ego_flat"),
    "b1": os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat"),
    "gecko": os.path.join(ROOT, "data/gecko/babble"),
}
CHANNEL_NAMES = ["forward", "lateral", "yaw"]
HELD_OUT_FRAC = 0.2
SEED = 0
POOL_DIM = 1408
HIDDEN = 128
SBATCH = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CleanFroudeHead(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(POOL_DIM), nn.Linear(POOL_DIM, HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, n_channels))

    def forward(self, x):
        return self.net(x)


def gather_body(name, directory, encoder, checkpoint, channels, mean_s, std_s):
    """Same shape as `ftm_froude_stopgrad_probe.py`'s own gather loop: every 4th transition,
    (e_t, e_next, body_motion[t], body_motion[t+1], action[t]), held out BY CLIP."""
    paths = sorted(glob.glob(os.path.join(directory, "*.npz")))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(paths))
    n_held = max(1, int(len(paths) * HELD_OUT_FRAC))
    held_paths = {paths[i] for i in order[:n_held]}

    E_t, E_next, Bm_next, Actions, Held = [], [], [], [], []
    off = offset_for(checkpoint, name)
    for p in paths:
        clip = load(p, REGISTRY[name])
        e = encode_clip(encoder, clip["frames"], 2).float().to(device)
        if off is not None:
            e = e - off.to(device)
        bm = (torch.tensor(np.asarray(clip["body_motion"])[:, channels], dtype=torch.float32,
                           device=device) - mean_s) / std_s
        actions = torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32, device=device)
        n = min(len(e), len(bm), len(actions)) - 1
        for t in range(1, n, 4):
            E_t.append(e[t].cpu()); E_next.append(e[t + 1].cpu())
            Bm_next.append(bm[t + 1]); Actions.append(actions[t]); Held.append(p in held_paths)
        del e
    torch.cuda.empty_cache()
    return (torch.stack(E_t), torch.stack(E_next), torch.stack(Bm_next), torch.stack(Actions),
            torch.tensor(Held), len(paths), n_held)


def score(name, z_name, e_t_cpu, e_next_cpu, bm_next, actions_all, held_mask, itm, ftm, proj, head):
    held_idx = held_mask.nonzero(as_tuple=True)[0]
    with torch.no_grad():
        if z_name == "itm":
            z_list = []
            for s in range(0, len(e_t_cpu), SBATCH):
                sl = slice(s, s + SBATCH)
                z_list.append(itm(e_t_cpu[sl].to(device), e_next_cpu[sl].to(device)))
            z_all = torch.cat(z_list)
        else:
            z_all = proj(actions_all.to(device), name)

        pooled_list = []
        for s in range(0, len(e_t_cpu), SBATCH):
            sl = slice(s, s + SBATCH)
            pred_next = ftm(e_t_cpu[sl].to(device), z_all[sl], "hexapod")
            pooled_list.append(pred_next.mean(1))
        pooled = torch.cat(pooled_list)

        pred = head(pooled[held_idx])
        true_h = bm_next[held_idx]
        rhos = [spearmanr(true_h[:, c].cpu().numpy(), pred[:, c].cpu().numpy())[0]
               for c in range(true_h.shape[1])]
        cos = F.cosine_similarity(pred, true_h, dim=1).median().item()
    return rhos, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", default=os.path.join(
        ROOT, "wm/runs/beh12_body_stopgrad/teacher_stopgrad.pt"))
    ap.add_argument("--proj_ckpt", default=os.path.join(
        ROOT, "wm/runs/beh12_body_stopgrad/projector_stopgrad_gecko.pt"))
    ap.add_argument("--head_ckpt", default=os.path.join(
        ROOT, "wm/runs/ftm_froude_stopgrad_head_hexapod.pt"))
    ap.add_argument("--bodies", nargs="+", default=["hexapod", "b1", "gecko"],
                    help="which of DIRS to score; drop 'gecko' for a checkpoint that never saw "
                         "it (e.g. Option A's hexapod+B1-only pretrain/fine-tune)")
    args = ap.parse_args()

    checkpoint = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(checkpoint["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(checkpoint["ftm"])

    proj_saved = torch.load(args.proj_ckpt, map_location="cpu", weights_only=False)
    proj = ActionProjector(cfg, action_dims_from(proj_saved)).to(device).eval()
    proj.load_state_dict(proj_saved["projector"])

    head_saved = torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
    channels = head_saved["channels"]
    heads = {}
    for z_name, state in head_saved["heads"].items():
        h = CleanFroudeHead(len(channels)).to(device).eval()
        h.load_state_dict(state)
        heads[z_name] = h
    for m in (itm, ftm, proj, *heads.values()):
        for p in m.parameters():
            p.requires_grad_(False)

    mean_s = torch.tensor(np.asarray(checkpoint["body_stats"][0]).ravel()[:len(channels)],
                          dtype=torch.float32, device=device)
    std_s = torch.tensor(np.asarray(checkpoint["body_stats"][1]).ravel()[:len(channels)],
                         dtype=torch.float32, device=device)

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    print(f"\n{'body':<10}{'z src':<8}{'n held':>8}" +
         "".join(f"{c + ' rho':>12}" for c in CHANNEL_NAMES) + f"{'median rho':>13}{'cos(med)':>10}")
    proj_median = {}
    for name in args.bodies:
        e_t, e_next, bm_next, actions_all, held_mask, n_clips, n_held = gather_body(
            name, DIRS[name], encoder, checkpoint, channels, mean_s, std_s)
        print(f"# {name}: {n_clips} clips, {n_held} held out, {int(held_mask.sum())} transitions "
             f"held out of {len(held_mask)}")
        for z_name in ("itm", "proj"):
            rhos, cos = score(name, z_name, e_t, e_next, bm_next, actions_all, held_mask, itm, ftm,
                              proj, heads[z_name])
            med = float(np.median(rhos))
            if z_name == "proj":
                proj_median[name] = med
            print(f"{name:<10}{z_name:<8}{int(held_mask.sum()):>8}" +
                 "".join(f"{r:>12.3f}" for r in rhos) + f"{med:>13.3f}{cos:>10.3f}")
        del e_t, e_next, bm_next, actions_all, held_mask
        torch.cuda.empty_cache()

    if "gecko" not in args.bodies:
        print("\n(gecko not scored -- this checkpoint/run wasn't asked to include it; VERDICT "
             "block skipped, it depends on a gecko number)")
        return

    hex_bar, b1_bar = FORGET_TOL * F199_HEX_BASELINE, FORGET_TOL * F199_B1_BASELINE
    recover_bar = RECOVER_FRAC * F199_B1_BASELINE
    hex_ok = proj_median["hexapod"] >= hex_bar
    b1_ok = proj_median["b1"] >= b1_bar
    gecko_ok = proj_median["gecko"] >= recover_bar

    print("\n" + "=" * 78)
    print("VERDICT (z=proj median rho, against this session's pre-registered bars)")
    print("=" * 78)
    print(f"forgetting gate  hexapod {proj_median['hexapod']:+.3f} "
         f"({'PASS' if hex_ok else 'FAIL'}, bar {hex_bar:.3f} = {FORGET_TOL:.0%} of "
         f"F181 zero-shot {F199_HEX_BASELINE:.3f})")
    print(f"                 b1      {proj_median['b1']:+.3f} "
         f"({'PASS' if b1_ok else 'FAIL'}, bar {b1_bar:.3f} = {FORGET_TOL:.0%} of "
         f"F181 zero-shot {F199_B1_BASELINE:.3f})")
    print(f"recovery bar     gecko   {proj_median['gecko']:+.3f} "
         f"({'PASS' if gecko_ok else 'FAIL'}, bar {recover_bar:.3f} = {RECOVER_FRAC:.0%} of "
         f"B1's {F199_B1_BASELINE:.3f}; F181 zero-shot was {F199_GECKO_ZEROSHOT:.3f})")

    if hex_ok and b1_ok and gecko_ok:
        print("\n-> CLAIM (3) HOLDS: fine-tuning the world model on gecko's babble recovers a real")
        print("   Froude-transfer signal toward the in-pretrain range, without degrading hex/B1.")
    elif not (hex_ok and b1_ok):
        print("\n-> FORGETTING GATE FAILED: the fine-tune cost hex/B1 real performance. Any gecko")
        print("   recovery is confounded with this and should not be reported as a clean win.")
    else:
        print("\n-> RECOVERY BAR NOT MET: hex/B1 held (no forgetting), but gecko did not recover")
        print("   toward the in-pretrain range. Deeper than domain shift alone -- the coordinate")
        print("   may not transfer to a sufficiently novel body even with adaptation. Bounds the")
        print("   claim rather than confirming it.")


if __name__ == "__main__":
    main()
