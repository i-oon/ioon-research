"""Which input tells the decoder what body it is looking at: the frame, or the latent?

The decoder reads two things, x_t and z, and both are computed from frames of the same body, so
both can carry its identity. Which one the decoder actually uses decides whether constraining
the frame pathway would change anything.

Every body walks the same expert episodes, so at a given timestep two bodies are at the same
point in the gait cycle and their inputs can be crossed: decode with body A's frame and body
B's latent, then ask which body's ground-truth commands the answer resembles. If the output
follows the frame, the frame is the identity channel; if it follows the latent, the latent is.

Run from the repository root:
  .venv/bin/python3 scripts/swap_pathway.py --ckpt wm/runs/m3d_bracketed/epoch006.pt \\
      --bodies c10f10t10 c10f10t06 --encode_device cpu
"""
import argparse
import os
import sys
from dataclasses import fields

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import Config  # noqa: E402
from wm.data.dataset import load_clip  # noqa: E402
from wm.evaluate import decode, encode_clip, latents, offset_for, upgrade_decoder_state  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bodies", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--episodes", type=int, nargs="+", default=[101, 113, 124])
    ap.add_argument("--data_dir", default="data/allocentric/fwd_hex8body")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--encode_device", default="")
    ap.add_argument("--embodiment", default="hexapod",
                    help="which head to decode through, for cross-embodiment checkpoints. The "
                         "swap is always between two bodies of one embodiment: a hexapod and a "
                         "quadruped share no timestep, so their inputs cannot be crossed.")
    args = ap.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in checkpoint["config"].items()
                    if k in {f.name for f in fields(Config)}})
    cfg.train_morphs = tuple(cfg.train_morphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    itm = InverseTransitionModel(cfg).to(device).eval()
    # A cross-embodiment checkpoint keys its heads and its action statistics by embodiment; a
    # single-morphology one has one unnamed head. Same branch as score_body.py.
    if "action_stats" in checkpoint:
        stats = checkpoint["action_stats"]
        if args.embodiment not in stats:
            raise SystemExit(f"--embodiment {args.embodiment!r} not in {sorted(stats)}")
        md = MotionDecoder(cfg, heads={k: len(v[0]) for k, v in stats.items()}).to(device).eval()
        mean, std = (np.asarray(v, dtype=np.float32) for v in stats[args.embodiment])
        head = args.embodiment
        print(f"cross-embodiment checkpoint {sorted(stats)}, swapping inside '{head}'")
    else:
        md = MotionDecoder(cfg).to(device).eval()
        mean, std = checkpoint["action_mean"], checkpoint["action_std"]
        head = "default"
    itm.load_state_dict(checkpoint["itm"])
    md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))
    # a `center_embeddings` checkpoint saw a per-embodiment mean subtracted; feeding it raw
    # embeddings is a silent distribution shift rather than an error
    offset = offset_for(checkpoint, args.embodiment)
    offset = offset.to(device) if offset is not None else None

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(ROOT, args.data_dir)
    encoder = VJEPA2FrameEncoder(
        device=args.encode_device or str(device),
        dtype=torch.float32 if args.encode_device == "cpu" else torch.float16)

    frame, latent, truth = {}, {}, {}
    for body in args.bodies:
        embeds, zs, acts = [], [], []
        for episode in args.episodes:
            clip = load_clip(os.path.join(data_dir, f"{body}_ep{episode}.npz"))
            emb = encode_clip(encoder, clip["frames"], args.chunk).to(device)
            if offset is not None:
                emb = emb - offset
            embeds.append(emb[:-1])
            zs.append(latents(itm, emb, args.chunk))
            acts.append(np.degrees(clip["actions"][:-1]))
        frame[body] = torch.cat(embeds)
        latent[body] = torch.cat(zs)
        truth[body] = np.concatenate(acts)
        print(f"encoded {body}: {len(truth[body])} transitions", flush=True)
    del encoder
    torch.cuda.empty_cache()

    a, b = args.bodies
    rmse = lambda p, t: float(np.sqrt(((p - t) ** 2).mean()))
    print(f"\nthe two bodies' own commands differ by {rmse(truth[a], truth[b]):.2f} deg")
    print(f"\n{'frame from':<12}{'latent from':<13}{'RMSE vs ' + a:>16}{'RMSE vs ' + b:>16}{'follows':>10}")
    for frame_body in args.bodies:
        for latent_body in args.bodies:
            pred = np.degrees(
                decode(md, frame[frame_body], latent[latent_body], args.chunk, head) * std + mean)
            ra, rb = rmse(pred, truth[a]), rmse(pred, truth[b])
            follows = frame_body if (ra < rb) == (frame_body == a) else latent_body
            if frame_body == latent_body:
                follows = "-"
            print(f"{frame_body:<12}{latent_body:<13}{ra:>16.2f}{rb:>16.2f}{follows:>10}")

    print("\nRows 2 and 3 are the crossed cases. The column with the lower error names the input\n"
          "the decoder is taking the body from.")


if __name__ == "__main__":
    main()
