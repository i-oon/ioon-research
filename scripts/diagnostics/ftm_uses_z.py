"""Does the forward model read the latent, or is the next frame guessable without it?

**The suspicion.** `L_recon` asks the FTM to predict `e_{t+1}` from `e_t` and `z_t`. For a periodic
gait at constant speed, `e_{t+1}` is largely guessable from `e_t` alone -- so nothing forces the
model to read `z`, and the reconstruction term would be training the latent for nothing. LAC-WM
names this as the reason for its auxiliary motion loss: it "mitigates shortcuts". Our logs are
consistent with it -- `recon` moves 1.5580 to 1.5400 between the control and the body-head arm while
`motion` moves 38%, so the body term reshapes the decoder and leaves the forward model alone.

Stage 1 measured this for the **decoder**: z-gap 20-39x, x-gap 7-15x (`wm/README.md`). Nobody has
measured it for the **forward model**, and the two are independent -- the decoder can depend on `z`
entirely while the FTM ignores it.

**Three views, because no single one is conclusive.** They were briefly three separate scripts;
they answer one question and belong together.

**1. Sensitivity.** Hold the frame at `e_0` and sweep every latent in the clip through it:
`FTM(e_0, z_0), FTM(e_0, z_1), ...`. If `z` carries content the predictions must fan out; if they
collapse to a point, the model is not reading it. The symmetric sweep -- fix `z_0`, vary the frame --
puts the two influences on one scale. **Nothing is pushed out of distribution**, unlike zeroing `z`,
which the model has never seen and which therefore cannot distinguish "ignored" from "unfamiliar".

**2. One-step ablation, in distribution.** Replace `z_t` with a real latent that is wrong in a
specific way:

    wrong behaviour   a latent from a different condition
    wrong phase       a latent from another frame of the same clip

Those two separate **what** the latent encodes. If shuffling within a clip costs nothing but a
latent from another condition does, then `z` carries clip-level behaviour and not the per-step
transition -- which is a different finding from "ignored", and matters more.

**3. Rollout.** Compounding over a horizon, closing the loop on the model's own output. A wrong
latent costs almost nothing in one step, because the frame barely changes; over eight steps it
should send the rollout somewhere else entirely. `hold` (nothing moves) is the floor.

**Split by behaviour group throughout**, because the whole suspicion is that *forward walking
specifically* is too easy. If the latent looks ignored in the speed conditions and used in the turn
conditions, the shortcut is real and behaviour variety is what removes it.

**Why this runs before the closed loop is built.** A planner samples candidate actions, projects
them to latents, rolls the FTM and picks a winner. If the z-sweep spread is near zero, every
candidate returns the same prediction -- the loop has nothing to choose between, however well the
rest of it is built.

  .venv/bin/python3 scripts/diagnostics/ftm_uses_z.py --ckpt wm/runs/beh12_body_fwd/best.pt
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
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def fan(x):
    """Mean distance of each prediction from their centre -- how far the outputs spread."""
    return float(torch.linalg.norm(x - x.mean(0, keepdim=True), dim=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dirs", nargs="+",
                    default=["hexapod=data/beh12_hex_flat", "b1=data/beh12_b1_v2"])
    ap.add_argument("--cache", default="results/wm/cache/beh12_embeddings.pt")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    enc = None
    rng = np.random.default_rng(args.seed)

    for spec in args.dirs:
        name, _, d = spec.partition("=")
        by_beh = {}
        for p in sorted(glob.glob(os.path.join(ROOT, d, "*.npz"))):
            if p not in cache:
                if enc is None:
                    enc = VJEPA2FrameEncoder(dtype=torch.float32)
                with np.load(p, allow_pickle=True) as c:
                    cache[p] = encode_clip(enc, c["frames"], 2).cpu().half()
            e = cache[p].float().to(device)
            off = offset_for(ck, name)
            if off is not None:
                e = e - off.to(device)
            with np.load(p, allow_pickle=True) as c:
                # keyed by condition as well as axis: "a latent from another clip of the same
                # behaviour group" is not a wrong latent -- speed_c5.8 and speed_c7.1 are both
                # "speed", and two clips of the *same* condition are the same behaviour outright
                by_beh.setdefault(str(c["behaviour"]), {}).setdefault(
                    str(c["condition"]), []).append(e)

        print(f"\n{name}")
        print(f"{'':<10}{'--- sensitivity ---':^32}{'-- one step (MSE) --':^34}{'- rollout @%d -' % args.horizon:^24}")
        print(f"{'behaviour':<10}{'sweep z':>10}{'sweep e':>10}{'z/step':>12}"
              f"{'correct':>11}{'wrong beh':>12}{'wrong phase':>11}{'real':>12}{'shuffled':>12}")
        for beh, conds in sorted(by_beh.items()):
            acc = []
            names = sorted(conds)
            flat = [(cn, e) for cn in names for e in conds[cn]]
            for cn, e in flat:
                # a latent from a genuinely different condition; falls back within the axis only
                # if this group has one condition, which none of ours does
                others = [c for c in names if c != cn] or names
                donor = conds[others[rng.integers(len(others))]][0]
                n = len(e) - 1
                with torch.no_grad():
                    z = torch.cat([itm(e[t:t + 1], e[t + 1:t + 2]) for t in range(n)])
                    other = donor
                    m = min(n, len(other) - 1)
                    zo = torch.cat([itm(other[t:t + 1], other[t + 1:t + 2]) for t in range(m)])
                    zo = zo.repeat((n // max(m, 1)) + 1, 1)[:n]
                    perm = torch.as_tensor(rng.permutation(n), device=device)
                    tgt = e[1:n + 1]

                    e0 = e[0:1].expand(n, *e.shape[1:])
                    s_z = fan(ftm(e0, z))
                    s_e = fan(ftm(e[:n], z[0:1].expand(n, -1)))
                    step = float(torch.linalg.norm(tgt - e[:n], dim=-1).mean())

                    ok = float(((ftm(e[:n], z) - tgt) ** 2).mean())
                    wb = float(((ftm(e[:n], zo) - tgt) ** 2).mean())
                    wp = float(((ftm(e[:n], z[perm]) - tgt) ** 2).mean())

                    # Rollout, closing the loop on the model's own output. Averaged over every
                    # start the clip allows rather than only frame 0 -- one rollout per clip gave
                    # four samples per group, which is not enough to compare two curves.
                    def roll(zs):
                        h = min(args.horizon, n)
                        out = []
                        for s0 in range(0, n - h):
                            cur = e[s0:s0 + 1]
                            for k in range(h):
                                cur = ftm(cur, zs[s0 + k:s0 + k + 1])
                            out.append(float(((cur[0] - e[s0 + h]) ** 2).mean()))
                        return float(np.mean(out)) if out else float("nan")
                    r_real, r_shuf = roll(z), roll(z[perm])
                acc.append((s_z, s_e, step, ok, wb, wp, r_real, r_shuf))
            a = np.mean(acc, axis=0)
            print(f"{beh:<10}{a[0]:>10.3f}{a[1]:>10.3f}{a[0] / max(a[2], 1e-9):>12.3f}"
                  f"{a[3]:>11.4f}{a[4]:>12.4f}{a[5]:>11.4f}{a[6]:>12.4f}{a[7]:>12.4f}")

    if enc is not None:
        torch.save(cache, cache_path)
    print("\n`sweep z` near 0, or `wrong beh` equal to `correct`, means the forward model ignores the")
    print("latent and L_recon trains it for nothing. `wrong phase` equal to `correct` while")
    print("`wrong beh` is worse means z carries clip-level behaviour, not the per-step transition.")


if __name__ == "__main__":
    main()
