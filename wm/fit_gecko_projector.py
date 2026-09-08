"""Extend the existing hexapod+b1 projector with a gecko entry, fit only on gecko babble.

  .venv/bin/python3 -m wm.fit_gecko_projector

Reuses `wm/runs/beh12_body_stopgrad/projector_stopgrad.pt` (confirmed this session: fitted
against `best.pt`, action_dims {hexapod: 18, b1: 12}). `ActionProjector.nets` is a
`nn.ModuleDict` with zero shared trunk per embodiment (see `wm/models/action_projector.py`), so
adding a "gecko" key and fitting only that key's weights leaves the loaded hexapod/b1 weights
byte-identical -- this is not a joint retrain.

Same two numbers as `wm/fit_projector.py`, same reason: z MSE is the training objective and the
weaker test; rollout gap (through the frozen FTM) is what a planner actually consumes.
"""
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.fit_projector import gather  # noqa: E402
from wm.models.action_projector import ActionProjector  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

PROJ_CKPT = "wm/runs/beh12_body_stopgrad/projector_stopgrad.pt"
GECKO_DIR = "data/gecko/babble"
CACHE = "results/wm/cache/gecko_embeddings.pt"
EPOCHS = 200
LR = 1e-3
VAL_FRAC = 0.2
OUT = "wm/runs/beh12_body_stopgrad/projector_stopgrad_gecko.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj_saved = torch.load(os.path.join(ROOT, PROJ_CKPT), map_location="cpu", weights_only=False)
    base_ckpt_path = proj_saved["ckpt"]
    checkpoint = torch.load(os.path.join(ROOT, base_ckpt_path), map_location="cpu",
                             weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval()
    ftm.load_state_dict(checkpoint["ftm"])
    for p in list(itm.parameters()) + list(ftm.parameters()):
        p.requires_grad_(False)

    cache_path = os.path.join(ROOT, CACHE)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    lag = max(1, cfg.action_lag)

    e, z, a, c, paths = gather("gecko", os.path.join(ROOT, GECKO_DIR), encoder, itm, checkpoint,
                                cache, chunk=2, lag=lag, device=device)
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()
    print(f"gecko: {len(paths)} clips, {len(a)} transitions gathered")

    old_dims = proj_saved["action_dims"]
    new_dims = {**old_dims, "gecko": a.shape[1]}
    proj = ActionProjector(cfg, new_dims).to(device)
    missing, unexpected = proj.load_state_dict(proj_saved["projector"], strict=False)
    # only the freshly-added gecko keys should be missing; anything else means the reuse
    # checkpoint's hexapod/b1 weights did not load cleanly
    assert all(k.startswith(("nets.gecko", "mean_gecko", "std_gecko")) for k in missing), missing
    assert not unexpected, unexpected
    print(f"loaded {len(old_dims)} existing embodiments untouched: {list(old_dims.keys())}")

    proj.set_stats("gecko", a.mean(0).cpu(), a.std(0).cpu())
    for name in old_dims:
        for p in proj.nets[name].parameters():
            p.requires_grad_(False)

    ids = torch.unique(c)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))
    val_ids = ids[order[:max(1, int(VAL_FRAC * len(ids)))]]
    val_mask = torch.isin(c, val_ids).to(device)
    val_paths = [paths[i] for i in val_ids.tolist()]
    print(f"gecko: {len(ids)} clips, {int(val_mask.sum())} of {len(c)} transitions held out")

    opt = torch.optim.Adam(proj.nets["gecko"].parameters(), lr=LR)
    m = ~val_mask
    for epoch in range(EPOCHS):
        proj.train(); opt.zero_grad()
        loss = torch.nn.functional.mse_loss(proj(a[m], "gecko"), z[m])
        loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1:4d}  train {loss.item():.4f}")

    proj.eval()
    with torch.no_grad():
        zp = proj(a[val_mask], "gecko")
        base = z[m].mean(0, keepdim=True).expand_as(z[val_mask])
        z_mse = torch.nn.functional.mse_loss(zp, z[val_mask]).item()
        z_base = torch.nn.functional.mse_loss(base, z[val_mask]).item()
        idx = torch.nonzero(val_mask.cpu(), as_tuple=True)[0]
        gap_num = gap_den = 0.0
        for i in range(0, len(idx), 32):
            sl = idx[i:i + 32]
            e_b = e[sl].to(device).float()
            truth = ftm(e_b, z[val_mask][i:i + 32])
            gap_num += ((ftm(e_b, zp[i:i + 32]) - truth) ** 2).mean().item() * len(sl)
            gap_den += ((ftm(e_b, base[i:i + 32]) - truth) ** 2).mean().item() * len(sl)
            del e_b, truth
        gap, gap_base = gap_num / len(idx), gap_den / len(idx)

    print(f"\n{'embodiment':<12}{'z MSE':>10}{'vs mean-z':>11}{'rollout gap':>14}{'vs mean-z':>11}")
    print(f"{'gecko':<12}{z_mse:>10.4f}{z_mse / max(z_base, 1e-9):>11.3f}"
          f"{gap:>14.4f}{gap / max(gap_base, 1e-9):>11.3f}")
    print("\nRatio below 1.0 beats predicting the mean z; the rollout column is what a")
    print("planner/selector actually consumes.")

    all_val_paths = {**proj_saved["val_paths"], "gecko": val_paths}
    out = os.path.join(ROOT, OUT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"projector": proj.state_dict(), "ckpt": base_ckpt_path,
                "val_paths": all_val_paths, "action_dims": new_dims}, out)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
