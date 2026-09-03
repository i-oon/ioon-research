"""Does a latent inferred from an *unseen embodiment* still tell the decoder what to output?

F46 crossed the decoder's two inputs between two bodies of the same embodiment and found it
answers with the latent's body. This asks the same question across embodiments, on the one pair
where it is exactly definable.

The 4-leg insect is the base stick insect with its middle legs ghost-removed, driven by the
*unchanged* six-leg IK gait. So its 12 commands are the base body's corner-leg columns
**bit-identically** -- verified, max difference 0.0000 deg -- and it walks the same expert
episodes. F45 killed cross-embodiment pairing for the B1 because nothing links a hexapod frame to
a quadruped frame; none of that applies here, because intent is shared by construction.

That gives a well-posed swap the B1 cannot support:

    frame from a SHORT hexapod body   +   latent from the 4-LEG video
    -> decode through the hexapod head, compare the corner columns against

        the short body's own commands      = it followed the frame
        the base body's commands           = it followed the latent, and the base body's
                                             commands are the 4-leg's, exactly

Both answers are commands the hexapod head already produces for training bodies, so neither is
out of reach for it -- which is what makes the comparison fair. The two differ by about 21 deg.

**Why this is the version that bears on the thesis.** The claim is that a shared visual latent
carries a skill to an embodiment the model never trained on. If a latent inferred from 4-leg video
still drives the decoder's output, the latent is doing embodiment-independent work. If the decoder
ignores it and answers from the frame it is holding, then whatever carries the 4-leg few-shot
result (F44) is the frozen encoder's features, not the learned latent.

**The trap.** The ITM never saw a 4-legged robot, so this is out of distribution for it by
construction -- that is the question, not a defect. But it does mean a *degenerate* answer is
possible: if the decoder outputs roughly the same thing whatever latent it is given, neither
column wins and the test says nothing. The two same-embodiment rows are printed as controls so
that case is visible rather than being read as a result.

  .venv/bin/python3 scripts/diagnostics/setting/swap_embodiment.py --ckpt wm/runs/stage2_clean/best.pt
"""
import argparse
import os
import sys
from dataclasses import fields

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import Config  # noqa: E402
from wm.data.dataset import load_clip  # noqa: E402
from wm.evaluate import decode, encode_clip, latents, offset_for, upgrade_decoder_state  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

# hexapod legs are FL ML HL FR MR HR at three joints each; the 4-leg keeps FL HL FR HR in that
# order, so these are the columns of an 18-D hexapod command that the 12-D 4-leg command equals.
CORNERS = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in (0, 2, 3, 5)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frame_body", default="c10f06t06",
                    help="hexapod body supplying the frame; must differ in geometry from the "
                         "base body, or there is no contrast to detect")
    ap.add_argument("--base_body", default="c10f10t10",
                    help="the body whose geometry the 4-leg shares, so its commands are the "
                         "4-leg's ground truth on the corner legs")
    ap.add_argument("--episodes", type=int, nargs="+", default=[101, 130, 144])
    ap.add_argument("--hexapod_dir", default="data/allocentric/fwd_hex8body")
    ap.add_argument("--fourleg_dir", default="data/ik_4leg_middleloss_clean9")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--encode_device", default="")
    args = ap.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in checkpoint["config"].items()
                    if k in {f.name for f in fields(Config)}})
    cfg.train_morphs = tuple(cfg.train_morphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "action_stats" not in checkpoint:
        raise SystemExit("this test needs a cross-embodiment checkpoint (one with action_stats)")
    stats = checkpoint["action_stats"]
    itm = InverseTransitionModel(cfg).to(device).eval()
    md = MotionDecoder(cfg, heads={k: len(v[0]) for k, v in stats.items()}).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))
    mean, std = (np.asarray(v, dtype=np.float32) for v in stats["hexapod"])
    offset = offset_for(checkpoint, "hexapod")
    offset = offset.to(device) if offset is not None else None

    encoder = VJEPA2FrameEncoder(
        device=args.encode_device or str(device),
        dtype=torch.float32 if args.encode_device == "cpu" else torch.float16)

    def gather(directory, name_of):
        """Embeddings, latents and corner-column ground truth, concatenated over episodes."""
        embeds, zs, acts = [], [], []
        for episode in args.episodes:
            path = os.path.join(ROOT, directory, f"{name_of}_ep{episode}.npz")
            clip = load_clip(path)
            emb = encode_clip(encoder, clip["frames"], args.chunk).to(device)
            if offset is not None:
                emb = emb - offset
            action = clip["actions"][:-1]
            embeds.append(emb[:-1])
            zs.append(latents(itm, emb, args.chunk))
            acts.append(np.degrees(action if action.shape[1] == 12 else action[:, CORNERS]))
        return torch.cat(embeds), torch.cat(zs), np.concatenate(acts)

    four_x, four_z, four_truth = gather(args.fourleg_dir, "middleloss")
    hex_x, hex_z, hex_truth = gather(args.hexapod_dir, args.frame_body)
    base_x, base_z, base_truth = gather(args.hexapod_dir, args.base_body)
    del encoder
    torch.cuda.empty_cache()

    rmse = lambda p, t: float(np.sqrt(((p - t) ** 2).mean()))
    n = min(len(four_truth), len(hex_truth))
    print(f"\n4-leg against its base body {args.base_body}: "
          f"{rmse(four_truth[:n], base_truth[:n]):.4f} deg  (must be ~0, it is the same command)")
    print(f"the two candidate answers differ by {rmse(hex_truth[:n], base_truth[:n]):.2f} deg")

    def run(frame_name, frame_e, latent_name, latent_z):
        k = min(len(frame_e), len(latent_z), n)
        pred = np.degrees(
            decode(md, frame_e[:k], latent_z[:k], args.chunk, "hexapod") * std + mean)[:, CORNERS]
        to_frame = rmse(pred, hex_truth[:k] if frame_name != "4-leg" else four_truth[:k])
        to_latent = rmse(pred, four_truth[:k] if latent_name == "4-leg" else hex_truth[:k])
        print(f"{frame_name:<12}{latent_name:<12}{to_frame:>16.2f}{to_latent:>16.2f}"
              f"{'' if frame_name == latent_name else ('frame' if to_frame < to_latent else 'latent'):>10}")

    print(f"\n{'frame from':<12}{'latent from':<12}{'RMSE vs frame':>16}{'RMSE vs latent':>16}"
          f"{'follows':>10}")
    run("4-leg", four_x, "4-leg", four_z)
    run(args.frame_body, hex_x, args.frame_body, hex_z)
    run(args.frame_body, hex_x, "4-leg", four_z)
    run("4-leg", four_x, args.frame_body, hex_z)

    print("\nRows 1 and 2 are controls: each input from one source, so both columns describe the\n"
          "same body and the pair says how well the decoder reproduces it at all. Rows 3 and 4\n"
          "are the crossed cases. If the crossed rows follow the latent, a latent inferred from\n"
          "an embodiment the model never trained on still drives the output -- the property the\n"
          "thesis needs. If they follow the frame, the latent is not carrying the behaviour.")


if __name__ == "__main__":
    main()
