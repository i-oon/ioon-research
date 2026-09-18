"""Cheap premise-check before building anything: if `z` were REPLACED entirely by ground-truth
Froude (3-D) as the action decoder's conditioning code, how much would action reconstruction
(`L_motion`, the loss tied with `L_recon` for the heaviest weight in this project's whole
objective) suffer?

**Why this, not a full retrain.** The proposal on the table collapses ITM's output from `z`
(64-D) to `froude_t` (3-D) directly -- `ITM(e_t,e_t+1) -> froude_t`, then `FTM(froude_t, e_t) ->
e_t+1`, `MD(froude_t, e_t) -> a_t`. Building that for real means retraining the whole pipeline
end to end, ~9 hours per F180's own note on this recipe's cost. Before paying that, this script
asks the narrower, decisive question first: holding the SAME cross-attention action-decoder
architecture (`wm.models.motion_decoder.MotionDecoder`) and the SAME frozen frame tokens, does a
freshly-fit decoder conditioned on ground-truth Froude (the best case for the proposal -- no
body_head regression error in the way) come close to one freshly-fit on real `z`? If Froude alone
already lets the decoder reconstruct the true action about as well as `z` does, the proposal is
promising. If it collapses, the reason is exactly what F138/F180 already found: Froude is a NET
outcome of a transition, not the transition itself, and many different actions (different gaits at
the same net speed) can share one Froude value -- information the decoder needs and `z` currently
carries, that Froude alone structurally cannot.

**What is and isn't controlled here.** Both decoders are FRESHLY fit (not the checkpoint's own
already-trained action heads) on the identical (x_t, action) pairs, same architecture, same
optimizer/epochs -- the only difference is the conditioning code (real `z` from the frozen ITM vs.
ground-truth `body_motion[t]`, the best-case oracle version of `froude_t`, not `body_head`'s own
noisy prediction of it). Fit separately per body (different action dims, 18 vs 12) on the union
of that body's `_cleantrain` clips, scored on its own `_cleanheldout` clips.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/froude_bottleneck_action_ceiling.py
"""
import dataclasses
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

CKPT = "wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt"
BODIES = {
    "hexapod": ("data/egocentric/beh12_c10f10t10_ego_flat_cleantrain",
                "data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout"),
    "b1": ("data/egocentric/beh12_b1_ego_flat_cleantrain",
           "data/egocentric/beh12_b1_ego_flat_cleanheldout"),
}
EPOCHS = 150  # mini-batched now (see fit_decoder) -- this is ~150 * (n/batch) gradient steps, not 150 total
LR = 3e-4


def collect(body, train_dir, heldout_dir, cfg, itm, channels, encoder, device, offset):
    spec = REGISTRY[body]
    train_paths = sorted(glob.glob(os.path.join(ROOT, train_dir, "*.npz")))
    heldout_paths = sorted(glob.glob(os.path.join(ROOT, heldout_dir, "*.npz")))
    overlap = set(os.path.basename(p) for p in train_paths) & \
             set(os.path.basename(p) for p in heldout_paths)
    if overlap:
        raise SystemExit(f"{body}: train/heldout overlap, refusing to score: {overlap}")

    Xt, Zr, Fr, A, is_val = [], [], [], [], []
    for split_paths, val_flag in ((train_paths, False), (heldout_paths, True)):
        for p in split_paths:
            clip = load(p, spec)
            with np.load(p, allow_pickle=True) as d:
                frames = d["frames"]
            e = encode_clip(encoder, frames, 4).float().to(device)
            if offset is not None:
                e = e - offset.to(device)
            with torch.no_grad():
                z = itm(e[:-1], e[1:])
            motion = np.asarray(clip["body_motion"])[:, channels]
            actions = np.asarray(clip["actions"])
            n = min(len(e) - 1, len(motion) - 1, len(actions) - 1)
            Xt.append(e[:n].cpu())
            Zr.append(z[:n].cpu())
            Fr.append(torch.tensor(motion[:n], dtype=torch.float32))
            A.append(torch.tensor(actions[:n], dtype=torch.float32))
            is_val.extend([val_flag] * n)
            print(f"  encoded {body:8s} {os.path.basename(p)} n={n} {'[test]' if val_flag else '[train]'}")

    return (torch.cat(Xt), torch.cat(Zr), torch.cat(Fr), torch.cat(A),
            np.asarray(is_val, dtype=bool))


def fit_decoder(cfg, action_dim, x_t, code, action, is_val, device, epochs=EPOCHS, lr=LR,
                batch=64):
    """Full cross-attention forward over every transition at once OOM'd a shared 16GB card (the
    token grid is large per transition) -- mini-batched training and chunked inference instead,
    same effective objective."""
    md = MotionDecoder(cfg, {"default": action_dim}).to(device)
    opt = torch.optim.Adam(md.parameters(), lr=lr, weight_decay=1e-4)
    train_idx = np.where(~is_val)[0]
    x_tr, c_tr, a_tr = x_t[train_idx], code[train_idx], action[train_idx]
    n = len(train_idx)
    last_loss = None
    for ep in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            opt.zero_grad()
            pred = md(x_tr[idx].to(device), c_tr[idx].to(device), "default")
            loss = torch.nn.functional.mse_loss(pred, a_tr[idx].to(device))
            loss.backward(); opt.step()
            last_loss = float(loss.item())
    md.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_t), batch):
            sl = slice(start, start + batch)
            preds.append(md(x_t[sl].to(device), code[sl].to(device), "default").cpu())
    return torch.cat(preds), last_loss


def report(name, pred, action, mask, ref_mean):
    err = (pred[mask] - action[mask]).abs().mean(-1)
    ss_res = ((pred[mask] - action[mask]) ** 2).sum().item()
    ss_tot = ((action[mask] - ref_mean) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    print(f"  {name:28s} n={mask.sum():3d}  mean|err|={err.mean():.4f}  R2(vs train mean)={r2:+.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, CKPT), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).eval().to(device)
    itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    channels = [int(c) for c in cfg.body_channels]
    cfg_froude = dataclasses.replace(cfg, z_dim=len(channels))
    encoder = VJEPA2FrameEncoder(device=str(device), dtype=torch.float32)

    for body, (train_dir, heldout_dir) in BODIES.items():
        offset = offset_for(ck, body)
        x_t, z_real, froude_true, action, is_val = collect(
            body, train_dir, heldout_dir, cfg, itm, channels, encoder, device, offset)
        action_dim = action.shape[-1]
        train_mean = action[~is_val].mean(0)

        print(f"\n--- {body}: fitting decoder on real z ({z_real.shape[-1]}-D) ---")
        pred_z, loss_z = fit_decoder(cfg, action_dim, x_t, z_real, action, is_val, device)
        print(f"  final train loss (MSE) {loss_z:.4f}")

        print(f"--- {body}: fitting decoder on ground-truth Froude ({froude_true.shape[-1]}-D) ---")
        pred_fr, loss_fr = fit_decoder(cfg_froude, action_dim, x_t, froude_true, action, is_val, device)
        print(f"  final train loss (MSE) {loss_fr:.4f}")

        print(f"\n=== {body}: action reconstruction, held out ({is_val.sum()} transitions) ===")
        report("decoder(x_t, real z)", pred_z, action, is_val, train_mean)
        report("decoder(x_t, ground-truth Froude)", pred_fr, action, is_val, train_mean)
        print(f"=== {body}: action reconstruction, train ({(~is_val).sum()} transitions) ===")
        report("decoder(x_t, real z)", pred_z, action, ~is_val, train_mean)
        report("decoder(x_t, ground-truth Froude)", pred_fr, action, ~is_val, train_mean)

    print("\nif the Froude-conditioned decoder's held-out R2 is close to the real-z decoder's, the")
    print("Froude-as-bottleneck proposal survives this premise check. If it collapses, Froude alone")
    print("cannot stand in for z as the action decoder's code -- consistent with F138/F180.")


if __name__ == "__main__":
    main()
