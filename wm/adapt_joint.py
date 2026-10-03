"""Stage 3 of LAC-WM's adaptation: fine-tune the action projector and the forward model together.

    stage 1  `wm.adapt`          ITM + FTM, LoRA rank 2, frame pairs only
    stage 2  `wm.fit_projector`  projector from scratch, FTM frozen, target z = ITM(e_t, e_t+k)
    stage 3  this file           projector + FTM jointly, LoRA rank 2 on the FTM, end-to-end
    (stage 4 `wm.fit_body_head` is this project's own addition; LAC-WM has no body head)

LAC-WM (`doc/ref/notes_lac_wm.md` 5.3-5.4): "jointly fine-tune the projector and FDM end-to-end using
LoRA rank 2", the longest of the three stages (35k of 60k iterations), and the stage their ablation
shows matters most. Stage 2 only asks the projector to imitate the ITM's latent; this stage asks the
FTM, driven by the projector's latent, to predict the REAL next frame -- the quantity rollout-based
selection consumes. The ITM stays frozen: it defines z and reads the goal/rollout.

Loss: MSE(FTM(e_t, proj(a_t..t+k-1)), e_{t+k}), plus optionally `--lambda_z` x MSE(proj(a), ITM(e_t,
e_t+k)) to keep the projector on the ITM's latent, which the direct path (`body_head(proj(a))`) reads.
LAC-WM has no such term (default 0). Stride comes from the checkpoint (`wm/data/strided.py`).

Data: the clips of `--data` except the projector's held-out clips (`projector_b1.pt`'s `val_paths`),
which are used to report held-out prediction error before and after.

    .venv/bin/python3 -m wm.adapt_joint --ckpt wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_s4.pt \\
        --projector wm/runs/beh24_stride5_cleansplit/b1_lora_c3/projector_b1.pt \\
        --data data/egocentric/beh24_b1_ego_flat_cleantrain --embodiment b1 \\
        --out wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_s3.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.data.strided import action_chunks, first_pair_of, pair_latents, stride_of  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.lora import apply_lora, merge_and_unwrap_lora  # noqa: E402


def transitions(paths, spec, cache, encoder_box, itm, k, lag, offset, device):
    """Per clip: (e on CPU half, start indices t, action chunks, ITM latents)."""
    out = []
    for p in paths:
        clip = load(p, spec)
        if p not in cache:
            if encoder_box[0] is None:
                from vjepa2_encoder import VJEPA2FrameEncoder
                encoder_box[0] = VJEPA2FrameEncoder(dtype=torch.float32)
            cache[p] = encode_clip(encoder_box[0], clip["frames"], 2).cpu().half()
        e = cache[p].float()
        if offset is not None:
            e = e - offset.float().reshape(e.shape[1:])
        # transitions start at the clip's `first_pair` (CF branches; 0 otherwise = unchanged), and
        # `e` is stored from there so `gather`'s e[t], e[t + k] index the same transition as a[t]
        s0 = first_pair_of(clip)
        n = len(e) - k - s0
        if n <= 0 or len(clip["actions"]) < s0 + n + lag + k - 1:
            continue
        a = torch.as_tensor(action_chunks(clip["actions"], lag, k, n, start=s0), dtype=torch.float32)
        with torch.no_grad():
            z = pair_latents(itm, e.to(device), k, n, start=s0).cpu()
        out.append((e[s0:].half(), a, z))
    return out


def batches(data, batch, rng, k):
    idx = [(ci, t) for ci, (e, a, z) in enumerate(data) for t in range(len(a))]
    while True:
        sel = [idx[i] for i in rng.choice(len(idx), batch, replace=False)]
        yield sel


def gather(data, sel, k, device):
    e_t = torch.stack([data[ci][0][t] for ci, t in sel]).float().to(device)
    e_n = torch.stack([data[ci][0][t + k] for ci, t in sel]).float().to(device)
    a = torch.stack([data[ci][1][t] for ci, t in sel]).to(device)
    z = torch.stack([data[ci][2][t] for ci, t in sel]).to(device)
    return e_t, e_n, a, z


@torch.no_grad()
def evaluate(ftm, proj, data, k, device, emb):
    """Held-out: prediction error of FTM driven by proj(a) vs by the ITM's own z, and z error."""
    num_p = num_i = num_z = cnt = 0.0
    for ci, (e, a, z) in enumerate(data):
        for s in range(0, len(a), 16):
            sel = [(ci, t) for t in range(s, min(s + 16, len(a)))]
            e_t, e_n, aa, zz = gather(data, sel, k, device)
            zp = proj(aa, emb)
            num_p += F.mse_loss(ftm(e_t, zp), e_n, reduction="sum").item() / e_n[0].numel()
            num_i += F.mse_loss(ftm(e_t, zz), e_n, reduction="sum").item() / e_n[0].numel()
            num_z += F.mse_loss(zp, zz, reduction="sum").item() / zz.shape[1]
            cnt += len(sel)
    return num_p / cnt, num_i / cnt, num_z / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="stage 2/4 output carrying itm/ftm/md/projector")
    ap.add_argument("--projector", required=True, help="stage 2 projector file (for its val_paths)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_rank", type=int, default=2)
    ap.add_argument("--lambda_z", type=float, default=0.0,
                    help="keep proj(a) near ITM z (direct path's input); LAC-WM has no such term")
    ap.add_argument("--cache", default="results/wm/cache/beh12_embeddings.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    k, lag, emb = stride_of(cfg), max(1, cfg.action_lag), args.embodiment
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    ftm = ForwardTransitionModel(cfg).to(device)
    ftm.load_state_dict(ck["ftm"])
    proj = ActionProjector(cfg, action_dims_from(ck)).to(device)
    proj.load_state_dict(ck["projector"])

    val_paths = set(torch.load(os.path.join(ROOT, args.projector), map_location="cpu",
                               weights_only=False)["val_paths"][emb])
    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    train_p = [p for p in paths if p not in val_paths]
    val_p = [p for p in paths if p in val_paths]
    print(f"stride {k}; {len(train_p)} train clips, {len(val_p)} held-out clips (projector's split)")

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    n0, box = len(cache), [None]
    off = offset_for(ck, emb)
    spec = REGISTRY[emb]
    train = transitions(train_p, spec, cache, box, itm, k, lag, off, device)
    val = transitions(val_p, spec, cache, box, itm, k, lag, off, device)
    if len(cache) > n0:
        torch.save(cache, cache_path)
    box[0] = None
    torch.cuda.empty_cache()

    ftm.eval(); proj.eval()
    before = evaluate(ftm, proj, val, k, device, emb)

    n_lora = apply_lora(ftm, rank=args.lora_rank)
    ftm.to(device)
    params = [p for p in ftm.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    print(f"LoRA rank {args.lora_rank} on {n_lora} FTM layers; projector fully trainable; ITM frozen; "
          f"lambda_z {args.lambda_z}")
    rng = np.random.default_rng(args.seed)
    it = batches(train, args.batch, rng, k)
    ftm.train(); proj.train()
    for step in range(args.steps):
        e_t, e_n, a, z = gather(train, next(it), k, device)
        zp = proj(a, emb)
        loss = F.mse_loss(ftm(e_t, zp), e_n)
        if args.lambda_z > 0:
            loss = loss + args.lambda_z * F.mse_loss(zp, z)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1:5d}  loss {loss.item():.4f}", flush=True)
    ftm.eval(); proj.eval()
    merge_and_unwrap_lora(ftm)
    ftm.to(device)
    after = evaluate(ftm, proj, val, k, device, emb)

    print(f"\nheld-out ({len(val_p)} clips)      FTM(e,proj(a)) vs e_next   FTM(e,ITM z) vs e_next   "
          f"proj(a) vs ITM z")
    for tag, (p_, i_, z_) in (("before", before), ("after", after)):
        print(f"  {tag:<8}{p_:>26.4f}{i_:>25.4f}{z_:>19.4f}")

    saved = dict(ck)
    saved["ftm"] = {kk: v.cpu() for kk, v in ftm.state_dict().items()}
    saved["projector"] = {kk: v.cpu() for kk, v in proj.state_dict().items()}
    saved["joint_fit"] = {"data": args.data, "steps": args.steps, "lr": args.lr, "stride": k,
                          "lora_rank": args.lora_rank, "lambda_z": args.lambda_z, "source": args.ckpt,
                          "held_out": sorted(os.path.basename(p) for p in val_p)}
    out = os.path.join(ROOT, args.out)
    torch.save(saved, out)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
