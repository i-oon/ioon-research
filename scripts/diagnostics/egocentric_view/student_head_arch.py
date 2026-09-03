"""Attention or convolution over the token grid -- and does either fit the 20 Hz budget?

    .venv/bin/python3 scripts/diagnostics/egocentric_view/student_head_arch.py \\
        --ckpt wm/runs/beh12_ego/best.pt --data data/egocentric/beh12_c08f09t09_ego_flat \\
        --embodiment hexapod --cache results/wm/cache/ego_hex.pt

**The student architecture decision, made by measurement before a student is built.** F175 measured
a pooled readout at **0.263 on the insect and 0.081 on the B1** egocentrically, against a
cross-attention decoder over the same tokens at **0.847 / 0.778**. `pooled(e) = e.mean(-2)` discards
where things are in the frame, and egocentrically where things are IS how the world moves, which is
the command. **No published egocentric locomotion policy pools it away** -- Hu et al. (2207.03386)
use a ResNet with an RNN, Xiao et al. a CNN with an LSTM, and the behaviour-cloning work in
2605.14106 states the four-layer CNN is there for its spatial inductive bias.

**The mechanism does not transfer even though the principle does.** Those policies train the CNN
end to end, so the encoder learns whatever spatial features the policy needs. V-JEPA2 here is
**frozen** and hands over a fixed 16x16 grid of 1408-dim tokens; the student has to read that grid
rather than learn one. So the choice is how to read it:

    pooled          `e.mean(-2)` into the Student MLP -- the current design, and the baseline
    conv            a small convolution over the grid, Hu-style spatial inductive bias, cheap
    attention       a learned query cross-attending over the 256 tokens -- what F175 measured

**Accuracy is only half the decision.** Pooling was adopted to hold 20 Hz. So this also times each
head **and times the frozen encoder**, because if the encoder already eats the budget then the head's
cost is not what the choice should turn on -- and that is a measurement, not an assumption.

Split by clip in families, `within cond` reported alongside, the same rule as
`pooled_student_check.py`, so the numbers land on the same scale as the bar F176 locked.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnostics.objective_experiments.residual_structure import FAMILY  # noqa: E402


class Pooled(nn.Module):
    """The current `Student`: mean over tokens, then two GELU layers."""

    def __init__(self, token_dim, goal_dim, action_dim, hidden=512, grid=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(token_dim + goal_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, action_dim))

    def forward(self, tok, goal):
        return self.net(torch.cat([tok.mean(1), goal], -1))


class Conv(nn.Module):
    """Project the token dimension down, then two strided convolutions over the 16x16 grid.

    **The cheap way to keep spatial structure.** The 1x1 projection is what makes it affordable:
    convolving 1408 channels directly would cost more than the attention it is meant to undercut.
    """

    def __init__(self, token_dim, goal_dim, action_dim, hidden=512, grid=16, width=128):
        super().__init__()
        self.grid = grid
        self.proj = nn.Conv2d(token_dim, width, 1)
        self.body = nn.Sequential(nn.GELU(), nn.Conv2d(width, width, 3, stride=2, padding=1),
                                  nn.GELU(), nn.Conv2d(width, width, 3, stride=2, padding=1),
                                  nn.GELU(), nn.AdaptiveAvgPool2d(2))
        self.net = nn.Sequential(nn.Linear(width * 4 + goal_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, action_dim))

    def forward(self, tok, goal):
        b = tok.shape[0]
        x = tok.transpose(1, 2).reshape(b, -1, self.grid, self.grid)
        x = self.body(self.proj(x)).flatten(1)
        return self.net(torch.cat([x, goal], -1))


class Attention(nn.Module):
    """A learned query, cross-attending over the 256 tokens -- F175's 0.778 path."""

    def __init__(self, token_dim, goal_dim, action_dim, hidden=512, grid=16, heads=8):
        super().__init__()
        self.kv = nn.Linear(token_dim, hidden)
        self.q = nn.Linear(goal_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(nn.Linear(hidden + goal_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, action_dim))

    def forward(self, tok, goal):
        kv = self.kv(tok)
        q = self.q(goal).unsqueeze(1)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.net(torch.cat([self.norm(out.squeeze(1)), goal], -1))


HEADS = {"pooled  (the current Student)": Pooled, "conv  (spatial, cheap)": Conv,
         "attention  (F175's path)": Attention}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--bar", type=float, default=None,
                    help="the within-condition bar this body is read against: 0.294 insect, 0.059 "
                         "B1 (F176). **Not a target to beat by tuning** -- it is what the pooled "
                         "design reaches, and the question is how much of the token grid's 0.847 / "
                         "0.778 a runnable head recovers.")
    ap.add_argument("--time_encoder", action="store_true",
                    help="**time the frozen encoder too.** Pooling was adopted to hold 20 Hz; if "
                         "V-JEPA2 already spends most of the 50 ms then the head's cost is not what "
                         "the choice should turn on, and that has to be measured rather than "
                         "assumed either way.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])

    cache_path = os.path.join(ROOT, args.cache or f"results/wm/cache/fid_{args.embodiment}.pt")
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, encoder, ck, cache,
                   args.chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del cache

    enc_ms = None
    if args.time_encoder:
        frame = np.zeros((1, 256, 256, 3), np.uint8)
        for _ in range(3):
            encoder.encode(list(frame))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            encoder.encode(list(frame))
        torch.cuda.synchronize()
        enc_ms = (time.perf_counter() - t0) * 100.0
    del encoder
    torch.cuda.empty_cache()

    chans = [int(c) for c in getattr(cfg, "body_channels", (0, 1, 2))]
    goal_of = {}
    for c in clips:
        rec = load(os.path.join(ROOT, args.data, c["path"]), REGISTRY[args.embodiment])
        goal_of[c["path"]] = np.asarray(rec["body_motion"])[:, chans].mean(0).astype(np.float32)

    E, G, A, cond_id, clip_id = [], [], [], [], []
    for ci, c in enumerate(clips):
        e = c["e"].float()
        if len(e) < 4:
            continue
        for t in range(1, len(e) - 2, args.stride):
            E.append(e[t].half())
            G.append(torch.from_numpy(goal_of[c["path"]]))
            A.append(c["a"][t].flatten().float())
            cond_id.append(c["cond"]); clip_id.append(ci)
    E = torch.stack(E); G = torch.stack(G); A = torch.stack(A)
    cond_id = np.array(cond_id); clip_id = np.array(clip_id)

    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    held = {ci for v in order.values() for ci in v[1::2]}
    te = torch.tensor([c in held for c in clip_id]); tr = ~te
    for M in (G, A):
        M.sub_(M[tr].mean(0)).div_(M[tr].std(0) + 1e-6)

    An = A.numpy()
    base_of = {c: An[tr.numpy()][cond_id[tr.numpy()] == c].mean(0)
               for c in set(cond_id[tr.numpy()].tolist())}
    centre = torch.from_numpy(np.stack([base_of.get(c, An[tr.numpy()].mean(0))
                                        for c in cond_id[te.numpy()]]))

    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}")
    print(f"{int(tr.sum())} train / {int(te.sum())} test transitions, split by clip, "
          f"tokens {E.shape[1]} x {E.shape[2]}")
    if enc_ms is not None:
        print(f"\n  **the frozen encoder alone: {enc_ms:.1f} ms per frame** "
              f"({enc_ms / 50.0:.0%} of a 50 ms step at 20 Hz)")
    print(f"\n  {'head':>30}{'params':>10}{'R2':>9}{'within':>9}{'ms/frame':>11}{'of 50ms':>9}")

    idx_te = torch.nonzero(te).flatten()
    for name, cls in HEADS.items():
        torch.manual_seed(0)
        net = cls(E.shape[2], G.shape[1], A.shape[1], grid=int(round(E.shape[1] ** 0.5))).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        idx_tr = torch.nonzero(tr).flatten()
        best = (-1e9, -1e9)
        for epoch in range(args.epochs):
            net.train()
            perm = idx_tr[torch.randperm(len(idx_tr))]
            for i in range(0, len(perm), args.batch):
                sl = perm[i:i + args.batch]
                loss = nn.functional.mse_loss(
                    net(E[sl].to(device).float(), G[sl].to(device)), A[sl].to(device))
                opt.zero_grad(); loss.backward(); opt.step()
            if epoch % 10 and epoch != args.epochs - 1:
                continue
            net.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(idx_te), 128):
                    sl = idx_te[i:i + 128]
                    preds.append(net(E[sl].to(device).float(), G[sl].to(device)).cpu())
            pred = torch.cat(preds)
            ss = float(((pred - A[idx_te]) ** 2).sum())
            r2 = 1 - ss / max(float(((A[idx_te] - A[tr].mean(0)) ** 2).sum()), 1e-9)
            wr = 1 - ss / max(float(((A[idx_te] - centre) ** 2).sum()), 1e-9)
            if r2 > best[0]:
                best = (r2, wr)

        one, gone = E[:1].to(device).float(), G[:1].to(device)
        with torch.no_grad():
            for _ in range(5):
                net(one, gone)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                net(one, gone)
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / 50.0
        n_par = sum(p.numel() for p in net.parameters())
        print(f"  {name:>30}{n_par / 1e6:>9.1f}M{best[0]:>9.3f}{best[1]:>9.3f}"
              f"{ms:>11.2f}{ms / 50.0:>8.1%}")

    if args.bar is not None:
        print(f"\n  the pooled design's within-condition bar for this body: {args.bar:.3f}")
    print("\n  **Read the accuracy and the timing together.** Pooling was adopted for the 20 Hz")
    print("  budget, so a spatial head is only a fix if it fits it. If the encoder line above")
    print("  already dominates the 50 ms step, the head's milliseconds are not what the decision")
    print("  should turn on and pooling was never buying what it was adopted for.")


if __name__ == "__main__":
    main()
