"""Is the one-step prediction task easy enough to solve without reading the action at all?

    .venv/bin/python3 scripts/diagnostics/action_necessity.py \\
        --ckpt wm/runs/beh12_actswm/best.pt --data data/allocentric/beh12_c08f09t09_flat \\
        --embodiment hexapod

**This decides whether a separation term can ever bite.** F154 found the hinge cost long-horizon
accuracy and returned no sensitivity. Two explanations survive that and they call for opposite work:
`lambda_recon` is out of balance and wants tuning, or **one-step prediction does not require `z`**,
in which case no weighting saves the hinge and the prediction task itself has to be made harder.

The measurement is one step only, and it swaps what drives the forward model while holding the
state fixed:

  real       `ITM(e_t, e_t+1)` -- the action that actually happened
  null       `ITM(e_t, e_t)`   -- F151's null: the model's own name for "nothing happened"
  shuffled   a real latent from a random position in the same clip -- an action the model has seen,
             paired with a state that never preceded it
  mean       the clip's own mean latent
  hold       predicting no motion at all, the floor every ratio is read against

**The number that decides it is `null/real`.** Near 1.0 means the action channel buys nothing at one
step and the objective cannot teach sensitivity. Comfortably above 1.0 means the channel is used and
F154 is a weighting problem.

Reported per behaviour family, because the families differ: a single-speed gait fixes its own phase
from the frame (F119's within-clip trap) while turning does not.
"""
import argparse
import collections
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

FAMILY = lambda cond: "side" if cond.startswith("side") else cond.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--lags", type=int, nargs="+", default=[1],
                    help="**how many frames one predicted step spans.** At lag `k` the action is "
                         "`ITM(e_t, e_t+k)` and the target is `e_t+k`, a single application of the "
                         "forward model -- exactly what training with frameskip `k` would ask of "
                         "it. `null/real` at each lag says whether the action would *matter* at "
                         "that spacing; the ratio against holding still says whether anything is "
                         "predictable there at all. **The forward model was fitted at lag 1, so "
                         "the second column is pessimistic for k > 1 and only the first is a clean "
                         "read on the task.**")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])

    cache_path = os.path.join(ROOT, args.cache or f"results/wm/cache/fid_{args.embodiment}.pt")
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, encoder, ck, cache,
                   args.chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    keys = ("real", "null", "shuf", "mean", "hold")
    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}\n")
    print(f"  {'lag':>4}{'family':>10}{'real':>10}{'null':>10}{'shuf':>10}{'mean':>10}{'hold':>10}"
          f"{'null/real':>11}{'shuf/real':>11}{'real<null':>11}{'real/hold':>11}{'n':>7}")

    summary = {}
    for lag in args.lags:
        tot = collections.defaultdict(lambda: collections.defaultdict(float))
        cnt = collections.Counter()
        wins = collections.Counter()
        with torch.no_grad():
            for c in clips:
                fam = FAMILY(c["cond"])
                e = c["e"].float().to(device)
                if len(e) < lag + 3:
                    continue
                # the action that spans `lag` frames, at every position it exists
                zs = torch.cat([itm(e[t:t + 1], e[t + lag:t + lag + 1])
                                for t in range(len(e) - lag)])
                z_bar = zs.mean(0, keepdim=True)
                order = torch.randperm(len(zs), generator=torch.Generator().manual_seed(0))
                for t in range(1, len(e) - lag - 1, args.stride):
                    truth = e[t + lag]
                    z_null = itm(e[t:t + 1], e[t:t + 1])
                    err = {
                        "real": ftm(e[t:t + 1], zs[t:t + 1]),
                        "null": ftm(e[t:t + 1], z_null),
                        "shuf": ftm(e[t:t + 1], zs[order[t % len(zs)]].unsqueeze(0)),
                        "mean": ftm(e[t:t + 1], z_bar),
                    }
                    err = {k: float(((v[0] - truth) ** 2).mean()) for k, v in err.items()}
                    err["hold"] = float(((e[t] - truth) ** 2).mean())
                    for scope in ("all", fam):
                        for k in keys:
                            tot[scope][k] += err[k]
                        cnt[scope] += 1
                        if err["real"] < err["null"]:
                            wins[scope] += 1
        for scope in ["all"] + sorted(k for k in cnt if k != "all"):
            n = cnt[scope]
            r = {k: tot[scope][k] / n for k in keys}
            print(f"  {lag:>4}{scope:>10}" + "".join(f"{r[k]:>10.4f}" for k in keys)
                  + f"{r['null'] / max(r['real'], 1e-9):>11.3f}"
                  + f"{r['shuf'] / max(r['real'], 1e-9):>11.3f}"
                  + f"{100 * wins[scope] / n:>10.1f}%"
                  + f"{r['real'] / max(r['hold'], 1e-9):>11.3f}{n:>7}")
            if scope == "all":
                summary[lag] = (r['null'] / max(r['real'], 1e-9),
                                r['real'] / max(r['hold'], 1e-9),
                                100 * wins[scope] / n)
        print()

    print(f"  {'lag':>4}{'null/real':>11}{'real/hold':>11}{'real<null':>11}   reading")
    for lag, (nr, rh, w) in sorted(summary.items()):
        if nr >= 1.10 and rh < 1.0:
            note = "**viable** -- the action matters and the state is still predictable"
        elif nr >= 1.10:
            note = "action matters but prediction has failed here"
        elif rh < 1.0:
            note = "predictable, but the action still buys nothing"
        else:
            note = "neither"
        print(f"  {lag:>4}{nr:>11.3f}{rh:>11.3f}{w:>10.1f}%   {note}")


if __name__ == "__main__":
    main()
