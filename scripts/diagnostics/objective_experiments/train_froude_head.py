"""Parameterized version of `ftm_froude_stopgrad_probe.py`: train a CleanFroudeHead against a
GIVEN checkpoint's own frozen FTM, on a GIVEN embodiment's data.

  .venv/bin/python3 scripts/diagnostics/objective_experiments/train_froude_head.py \\
      --teacher_ckpt wm/runs/beh12_hexonly_stopgrad/best.pt \\
      --proj_ckpt wm/runs/beh12_hexonly_stopgrad/projector_a.pt \\
      --embodiment hexapod --data_dir data/egocentric/beh12_c10f10t10_ego_flat \\
      --out wm/runs/beh12_hexonly_stopgrad/ftm_froude_head_hexapod.pt

**Why a head has to be retrained per checkpoint, not reused across them.** The head reads
`pool(FTM(e_t, z))` -- FTM's own output, in whatever statistics THIS checkpoint's FTM produces.
Two different checkpoints have differently-weighted FTMs even at identical architecture, so a head
trained against one checkpoint's FTM output distribution is not a valid probe of another's. This
is the same reuse the original `ftm_froude_stopgrad_probe.py` did for `beh12_body_stopgrad`; this
script exists so the identical methodology can be pointed at `beh12_hexonly_stopgrad` (Option A)
and any future fine-tuned checkpoint without hand-editing constants each time.

Output format matches `ftm_froude_stopgrad_probe.py`'s save (`{"heads": {...}, "channels": [...]}`)
so `gecko_froude_transfer.py --head_ckpt` (or any future ladder script) can load it unchanged.
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
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

CHANNEL_NAMES = ["forward", "lateral", "yaw"]
HELD_OUT_FRAC = 0.2
SEED = 0
POOL_DIM = 1408
HIDDEN = 128
ITERS = 1500
BATCH = 64
LR = 1e-3
BAR = 2 * 0.055
SBATCH = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CleanFroudeHead(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(POOL_DIM), nn.Linear(POOL_DIM, HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, n_channels))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--proj_ckpt", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    checkpoint = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(checkpoint["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(checkpoint["ftm"])
    proj_saved = torch.load(args.proj_ckpt, map_location="cpu", weights_only=False)
    proj = ActionProjector(cfg, action_dims_from(proj_saved)).to(device).eval()
    proj.load_state_dict(proj_saved["projector"])
    for m in (itm, ftm, proj):
        for p in m.parameters():
            p.requires_grad_(False)

    channels = [int(c) for c in cfg.body_channels]
    mean_s = torch.tensor(np.asarray(checkpoint["body_stats"][0]).ravel()[:len(channels)],
                          dtype=torch.float32, device=device)
    std_s = torch.tensor(np.asarray(checkpoint["body_stats"][1]).ravel()[:len(channels)],
                         dtype=torch.float32, device=device)

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(paths))
    n_held = int(len(paths) * HELD_OUT_FRAC)
    held_paths = {paths[i] for i in order[:n_held]}
    print(f"{len(paths)} clips, {n_held} held out\n")

    E_t, E_next, Bm_next, Actions, Held = [], [], [], [], []
    for p in paths:
        clip = load(p, REGISTRY[args.embodiment])
        e = encode_clip(encoder, clip["frames"], 2).float()
        bm = (torch.tensor(np.asarray(clip["body_motion"])[:, channels], dtype=torch.float32,
                           device=device) - mean_s) / std_s
        actions = torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32, device=device)
        n = min(len(e), len(bm), len(actions)) - 1
        for t in range(1, n, 4):
            E_t.append(e[t]); E_next.append(e[t + 1])
            Bm_next.append(bm[t + 1]); Actions.append(actions[t]); Held.append(p in held_paths)

    e_t_cpu = torch.stack(E_t).cpu(); e_next_cpu = torch.stack(E_next).cpu()
    bm_next = torch.stack(Bm_next); actions_all = torch.stack(Actions)
    held_mask = torch.tensor(Held)
    print(f"{len(e_t_cpu)} transitions, {held_mask.sum().item()} held out\n")
    torch.cuda.empty_cache()

    with torch.no_grad():
        z_itm_list = []
        for s in range(0, len(e_t_cpu), SBATCH):
            sl = slice(s, s + SBATCH)
            z_itm_list.append(itm(e_t_cpu[sl].float().to(device), e_next_cpu[sl].float().to(device)))
        z_itm = torch.cat(z_itm_list)
        z_proj = proj(actions_all, args.embodiment)

        pred_next_pooled = {}
        for name, z_all in (("itm", z_itm), ("proj", z_proj)):
            pooled_list = []
            for s in range(0, len(e_t_cpu), SBATCH):
                sl = slice(s, s + SBATCH)
                e_t_b = e_t_cpu[sl].float().to(device)
                pred_next = ftm(e_t_b, z_all[sl], args.embodiment)
                pooled_list.append(pred_next.mean(1))
            pred_next_pooled[name] = torch.cat(pooled_list).detach()

    train_idx = (~held_mask).nonzero(as_tuple=True)[0].to(device)
    held_idx = held_mask.nonzero(as_tuple=True)[0].to(device)

    trained_heads = {}
    for z_name, pooled_pred in pred_next_pooled.items():
        torch.manual_seed(SEED)
        head = CleanFroudeHead(len(channels)).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=LR)
        for it in range(ITERS):
            idx = train_idx[torch.randint(0, len(train_idx), (BATCH,), device=device)]
            pred = head(pooled_pred[idx])
            loss = F.mse_loss(pred, bm_next[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            pred_real = head(pooled_pred[held_idx])
            true_h = bm_next[held_idx]
            mean_input = pooled_pred[train_idx].mean(0, keepdim=True).expand(len(held_idx), -1)
            pred_mean = head(mean_input)
            cos_real = F.cosine_similarity(pred_real, true_h, dim=1)
            cos_mean = F.cosine_similarity(pred_mean, true_h, dim=1)
            gap = cos_real.median().item() - cos_mean.median().item()
            rhos = [spearmanr(true_h[:, c].cpu().numpy(), pred_real[:, c].cpu().numpy())[0]
                   for c in range(len(channels))]

        trained_heads[z_name] = head.state_dict()
        print(f"z={z_name:<6} action-lever gap={gap:+.3f} (bar {BAR:.3f})  "
             f"median rho={float(np.median(rhos)):+.3f}")
        print("  per-channel rho: " + "  ".join(f"{n}={r:+.3f}" for n, r in zip(CHANNEL_NAMES, rhos)))

    torch.save({"heads": trained_heads, "channels": channels}, args.out)
    print(f"\nsaved -> {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
