"""Fit `a_t -> z_t` against a trained checkpoint, so the world model can be driven without the future.

  .venv/bin/python3 -m wm.fit_projector --ckpt wm/runs/beh12_body_yaw/last.pt

The ITM is frozen and supplies the target: for every transition, `z = ITM(e_t, e_{t+1})` is what the
projector must reproduce from `a_{t+1}` alone -- the command that *caused* that transition, which is
the `action_lag` convention the collector and `predict_actions.py` already use.

**Two numbers are reported and they are not the same question.**

`z MSE` is how well the projector reproduces the latent. It is the training objective and it is the
weaker test: `z` is 64-D and correlated, so a low error can still put the prediction somewhere the
forward model behaves differently -- the same identifiability problem that made weight-vector
comparison useless in F66, where ridge coefficients on correlated `z` read at chance even for the
best-transferring run.

`rollout gap` is the one to trust. It feeds the projector's `z` to the FDM and compares the predicted
next embedding against what the *true* `z` predicts. That is what planning actually consumes: a
candidate action is only useful if the world model's answer for it is right. A projector can score
well on the first and badly on the second, and only the second would be a real failure.

Both are reported against a **baseline of predicting the mean `z`**, because an MSE against a
64-D correlated target has no interpretable scale on its own.
"""
import argparse
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
from wm.evaluate import encode_clip, offset_for, upgrade_decoder_state  # noqa: E402
from wm.models.action_projector import ActionProjector  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def gather(name, directory, encoder, itm, checkpoint, cache, chunk, lag, device, exclude=()):
    """Per clip: the frozen latent, the action that caused it, and the current embedding.

    **Embeddings come back on the CPU in half precision.** One clip is 65 x 256 x 1408 floats,
    about 94 MB; 48 of them per embodiment is 4.5 GB, and holding both robots' plus the encoder on
    an 11 GB card runs out inside the ITM's attention. Only the final rollout needs them, and it
    reads them in batches.

    Also returns a **clip index per transition**. The split below has to be by clip, and clips are
    not all the same length here -- 57 to 65 transitions -- so no fixed block size recovers the
    boundaries.
    """
    E, Z, A, C, P = [], [], [], [], []
    skipped = 0
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        if any(os.path.basename(path).startswith(p) for p in exclude):
            skipped += 1
            continue
        clip = load(path, REGISTRY[name])
        if path not in cache:
            cache[path] = encode_clip(encoder, clip["frames"], chunk).cpu().half()
        e = cache[path].float().to(device)
        off = offset_for(checkpoint, name)
        if off is not None:
            e = e - off.to(device)
        n = len(e) - 1
        with torch.no_grad():
            z = torch.cat([itm(e[t:min(t + 8, n)], e[t + 1:min(t + 8, n) + 1])
                           for t in range(0, n, 8)])
        actions = torch.as_tensor(clip["actions"], dtype=torch.float32, device=device)
        # the command that caused frames[t] -> frames[t+1]; short clips are dropped rather than
        # padded, since a padded action is a wrong label and F45 measured what wrong labels cost
        if len(actions) < n + lag:
            continue
        E.append(e[:n].cpu().half()); Z.append(z); A.append(actions[lag:lag + n])
        C.append(torch.full((n,), len(C), dtype=torch.long))
        P.append(path)
        del e
        torch.cuda.empty_cache()
    if skipped:
        print(f"{name}: excluded {skipped} clips matching {list(exclude)}")
    if not E:
        raise SystemExit(f"no clips left in {directory} after excluding {list(exclude)}")
    return torch.cat(E), torch.cat(Z), torch.cat(A), torch.cat(C), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hex_dir", default="data/beh12_hex_flat")
    ap.add_argument("--b1_dir", default="data/beh12_b1_v2",
                    help="empty string fits the hexapod alone, for a checkpoint that never saw a "
                         "quadruped")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="clip-name prefixes to leave out, e.g. a body the checkpoint held out. "
                         "**Without this the projector is fitted on the very body the run is "
                         "meant to be blind to**, and the held-out claim quietly becomes a "
                         "few-shot one.")
    ap.add_argument("--cache", default="results/wm/cache/beh12_embeddings.pt")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval()
    ftm.load_state_dict(checkpoint["ftm"])
    for p in list(itm.parameters()) + list(ftm.parameters()):
        p.requires_grad_(False)

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    lag = max(1, cfg.action_lag)

    data = {}
    for name, d in (("hexapod", args.hex_dir), ("b1", args.b1_dir)):
        if not d:
            continue
        data[name] = gather(name, os.path.join(ROOT, d), encoder, itm, checkpoint,
                            cache, args.chunk, lag, device, tuple(args.exclude))
    if not data:
        raise SystemExit("no source directories given")
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    # 300M frozen parameters that nothing below uses; on an 11 GB card that is the difference
    # between the rollout batching fitting and not
    del encoder, cache
    torch.cuda.empty_cache()

    proj = ActionProjector(cfg, {n: v[2].shape[1] for n, v in data.items()}).to(device)
    for name, (_, _, a, _c, _p) in data.items():
        proj.set_stats(name, a.mean(0).cpu(), a.std(0).cpu())

    # **Split by clip, not by frame.** Consecutive frames of one clip are near-duplicates, so a
    # frame-level split leaves the training data in the test set -- the leak that made yaw look
    # like it transferred at +0.31 until it was held out by condition instead (F76).
    # **`// 60` was not a clip boundary.** Clips here run 57 to 65 transitions, so a fixed block
    # size straddles them: every block held out shared a clip with the training set, which is the
    # frame-level leak this comment exists to prevent. Grouped on the index `gather` records.
    splits, val_paths = {}, {}
    for name, (e, z, a, c, paths) in data.items():
        ids = torch.unique(c)
        order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))
        val_ids = ids[order[:max(1, int(args.val_frac * len(ids)))]]
        splits[name] = torch.isin(c, val_ids).to(device)
        # **Recorded, not recomputed.** Anything scoring this projector afterwards has to know which
        # clips it never saw, and re-deriving the split elsewhere would silently drift the moment a
        # clip is added, renamed, or dropped for being too short.
        val_paths[name] = [paths[i] for i in val_ids.tolist()]
        print(f"{name:<10} {len(ids)} clips, {int(splits[name].sum())} of {len(c)} "
              f"transitions held out")

    opt = torch.optim.Adam(proj.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        proj.train(); opt.zero_grad(); loss = 0.0
        for name, (_, z, a, _c, _p) in data.items():
            m = ~splits[name]
            loss = loss + torch.nn.functional.mse_loss(proj(a[m], name), z[m])
        loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1:4d}  train {loss.item():.4f}")

    print(f"\n{'embodiment':<12}{'z MSE':>10}{'vs mean-z':>11}{'rollout gap':>14}{'vs mean-z':>11}")
    proj.eval()
    for name, (e, z, a, _c, _p) in data.items():
        m = splits[name]
        with torch.no_grad():
            zp = proj(a[m], name)
            base = z[~m].mean(0, keepdim=True).expand_as(z[m])
            z_mse = torch.nn.functional.mse_loss(zp, z[m]).item()
            z_base = torch.nn.functional.mse_loss(base, z[m]).item()
            # What planning actually consumes: does the FDM answer the same way for this z?
            # Batched, and the embeddings arrive from the CPU -- the whole held-out set at once is
            # a 512-token attention over ~600 transitions and does not fit alongside the model.
            idx = torch.nonzero(m.cpu(), as_tuple=True)[0]
            gap_num = gap_den = 0.0
            for i in range(0, len(idx), 32):
                sl = idx[i:i + 32]
                e_b = e[sl].to(device).float()
                truth = ftm(e_b, z[m][i:i + 32])
                gap_num += ((ftm(e_b, zp[i:i + 32]) - truth) ** 2).mean().item() * len(sl)
                gap_den += ((ftm(e_b, base[i:i + 32]) - truth) ** 2).mean().item() * len(sl)
                del e_b, truth
            gap, gap_base = gap_num / len(idx), gap_den / len(idx)
        print(f"{name:<12}{z_mse:>10.4f}{z_mse / max(z_base, 1e-9):>11.3f}"
              f"{gap:>14.4f}{gap / max(gap_base, 1e-9):>11.3f}")
    print("\nRatios are against predicting the mean z: below 1.0 is better than knowing nothing,")
    print("and the rollout column is the one that decides whether planning can use this.")

    out = args.out or os.path.join(os.path.dirname(os.path.join(ROOT, args.ckpt)), "projector.pt")
    torch.save({"projector": proj.state_dict(), "ckpt": args.ckpt,
                "val_paths": val_paths, "action_dims": {n: v[2].shape[1] for n, v in data.items()}},
               out)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
