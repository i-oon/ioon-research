"""Does ITM's learned, nonlinear transform of a frame pair beat a plain LINEAR function -- or a
generic MLP of the same capacity -- of the same frame pair, for reading Froude? The slice of "does
going through z help" this project had not tested: F138 already showed the pair (not just one
frame) matters (removing the transition costs 28-34%); this asks whether ITM's specific transition
structure matters, or any nonlinear function of the raw pair would do as well.

**Two baselines, both skipping ITM/z entirely**:
  - `Linear(concat(pooled(e_t), pooled(e_next))) -> Froude`, closed-form ridge -- no nonlinearity.
  - `MLP(concat(pooled(e_t), pooled(e_next))) -> Froude`, same depth/width as `body_head`
    (LayerNorm, Linear->128, GELU, Linear->body_dim) so a win for ITM isn't just "any nonlinearity
    beats a linear map" -- the MLP baseline has that same capacity without ITM's specific
    transition-model structure (its two-frame cross-attention, trained jointly with the FTM/proj).

**Compared against**: the existing, already-fit `body_head(ITM(e_t, e_next))` pipeline, run
forward-only (not refit) on the SAME test clips.

**Important asymmetry, stated rather than hidden, and now fixed**: `body_head` is fit on the UNION
of both bodies' data (`wm/fit_body_head.py --also hexapod=...`), one shared set of weights asked to
serve hexapod AND B1 jointly. The first version of this correction fit the linear/MLP baselines on
hexapod alone -- a hexapod-only specialist beating a shared, cross-embodiment head at a hexapod-only
task proves nothing about the ITM/z pathway, just the ordinary cost of sharing. Fixed: linear/MLP
are now fit on the exact same union (`_cleantrain` for both bodies, concatenated) `body_head` was,
and scored per-body on each body's own `_cleanheldout` set (12 hexapod + 12 B1, both stratified,
seed=42, `scripts/dataset/make_clean_split.py`) -- the same training condition for all three
methods, and the same held-out clips this checkpoint's own `body_head` was validated against
(F222).

**Second confound, controlled here too: bottleneck vs transition structure.** `z` (ITM's output) is
a small, fixed-size bottleneck shaped jointly by every loss in the pipeline (reconstruction,
next-embedding prediction, hinge, rollout, body) -- not optimized only for reading Froude. The raw
`concat(pooled(e_t), pooled(e_next))` pair baselines above have strictly MORE information available
(the full uncompressed embedding pair) and are fit for this one task alone, so a baseline win there
conflates "the bottleneck throws away useful information" with "ITM's transition modeling adds
nothing." Fixed by adding a second pair of baselines fit on `z` itself instead of the raw pair --
same capacity, same union-fit recipe, only the input differs. If linear(z)/MLP(z) still lose to
`body_head(ITM(.))`, the bottleneck itself is fine and the earlier pair-baseline win was about
information access, not about ITM's transition structure being unnecessary. If linear(z)/MLP(z)
also beat `body_head`, the transition structure itself -- not just the bottleneck -- is the
weakness, since all three now share the identical, already-bottlenecked input.

    .venv/bin/python3 scripts/diagnostics/cross_embodiment/linear_vs_itm_froude.py
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

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


def pooled(e):
    return e.mean(-2)


def ridge_fit(X, Y, lam=1.0):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    W = np.linalg.solve(A, Xb.T @ Y)
    return W


def ridge_predict(W, X):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return Xb @ W


class PairMLP(nn.Module):
    """Same shape as `MotionDecoder.body_head` when `not body_sees_frame`: LayerNorm, Linear->hidden,
    GELU, Linear->out. Only the input is different (raw pooled pair, not ITM's z)."""

    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def mlp_fit(X, Y, device, epochs=3000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    mean, std = Y.mean(0, keepdims=True), Y.std(0, keepdims=True).clip(min=1e-6)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor((Y - mean) / std, dtype=torch.float32, device=device)
    net = PairMLP(X.shape[-1], Y.shape[-1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xt), Yt)
        loss.backward(); opt.step()
    net.eval()
    return net, mean, std, float(loss.item())


def mlp_predict(net, mean, std, X, device):
    with torch.no_grad():
        pred_std = net(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return pred_std * std + mean


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, CKPT), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).eval().to(device)
    itm.load_state_dict(ck["itm"])
    md = MotionDecoder(cfg, {"b1": 12}).eval().to(device)
    md.load_state_dict(ck["md"], strict=False)
    channels = [int(c) for c in cfg.body_channels]
    mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
    std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]
    encoder = VJEPA2FrameEncoder(device=str(device), dtype=torch.float32)

    X_pair, X_z, Y_true, Z_itm_head, is_val, body_of = [], [], [], [], [], []
    for body, (train_dir, heldout_dir) in BODIES.items():
        spec = REGISTRY[body]
        offset = offset_for(ck, body)
        train_paths = sorted(glob.glob(os.path.join(ROOT, train_dir, "*.npz")))
        heldout_paths = sorted(glob.glob(os.path.join(ROOT, heldout_dir, "*.npz")))
        overlap = set(os.path.basename(p) for p in train_paths) & \
                 set(os.path.basename(p) for p in heldout_paths)
        if overlap:
            raise SystemExit(f"{body}: train/heldout overlap, refusing to score: {overlap}")
        paths = train_paths + heldout_paths
        val_idx = set(range(len(train_paths), len(paths)))

        for i, p in enumerate(paths):
            clip = load(p, spec)
            with np.load(p, allow_pickle=True) as d:
                frames = d["frames"]
            e = encode_clip(encoder, frames, 4).float().to(device)
            if offset is not None:
                e = e - offset.to(device)
            ep = pooled(e)
            pair = torch.cat([ep[:-1], ep[1:]], dim=-1).mean(0)  # concat(e_t, e_next), pooled, avg over clip
            with torch.no_grad():
                z = itm(e[:-1], e[1:])
                fr_pred_std = md.body_head(z).mean(0).cpu().numpy()
            fr_pred = fr_pred_std * std_s + mean_s   # un-standardize: body_head predicts standardized units
            goal = np.asarray(clip["body_motion"])[:, channels].mean(0)

            X_pair.append(pair.cpu().numpy())
            X_z.append(z.mean(0).cpu().numpy())
            Y_true.append(goal)
            Z_itm_head.append(fr_pred)
            is_val.append(i in val_idx)
            body_of.append(body)
            print(f"  encoded {body:8s} {os.path.basename(p)}  {'[test]' if i in val_idx else '[train]'}")

    X_pair = np.stack(X_pair); X_z = np.stack(X_z); Y_true = np.stack(Y_true)
    Z_itm_head = np.stack(Z_itm_head)
    is_val = np.asarray(is_val); body_of = np.asarray(body_of)

    # fit on the UNION of both bodies' train clips -- the same recipe wm.fit_body_head.py uses for
    # body_head itself, so linear/MLP are asked to share across bodies exactly as much as ITM is.
    W = ridge_fit(X_pair[~is_val], Y_true[~is_val], lam=10.0)
    Y_lin_pred = ridge_predict(W, X_pair)

    net, mean_y, std_y, final_loss = mlp_fit(X_pair[~is_val], Y_true[~is_val], device)
    print(f"\n  MLP(pair) baseline: final train loss (standardized MSE, both bodies pooled) {final_loss:.4f}")
    Y_mlp_pred = mlp_predict(net, mean_y, std_y, X_pair, device)

    # second baselines: same fit recipe, but on z (ITM's own output) instead of the raw pair --
    # isolates the bottleneck-vs-transition-structure question (see docstring).
    W_z = ridge_fit(X_z[~is_val], Y_true[~is_val], lam=10.0)
    Y_linz_pred = ridge_predict(W_z, X_z)

    net_z, mean_yz, std_yz, final_loss_z = mlp_fit(X_z[~is_val], Y_true[~is_val], device)
    print(f"  MLP(z) baseline: final train loss (standardized MSE, both bodies pooled) {final_loss_z:.4f}")
    Y_mlpz_pred = mlp_predict(net_z, mean_yz, std_yz, X_z, device)

    def report(name, pred, mask, ref_mean):
        err = np.abs(pred[mask] - Y_true[mask]).sum(-1)
        ss_res = ((pred[mask] - Y_true[mask]) ** 2).sum()
        ss_tot = ((Y_true[mask] - ref_mean) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        print(f"  {name:32s} n={mask.sum():3d}  median|err|={np.median(err):.4f}  "
             f"mean|err|={err.mean():.4f}  R2(vs train mean)={r2:+.3f}")

    def report_all(mask, ref_mean):
        report("linear(pair) -- raw frame pair, no ITM", Y_lin_pred, mask, ref_mean)
        report("MLP(pair) -- raw frame pair, no ITM", Y_mlp_pred, mask, ref_mean)
        report("linear(z) -- ITM's own bottleneck", Y_linz_pred, mask, ref_mean)
        report("MLP(z) -- ITM's own bottleneck", Y_mlpz_pred, mask, ref_mean)
        report("existing body_head(ITM(.))", Z_itm_head, mask, ref_mean)

    for body in BODIES:
        b_mask = body_of == body
        b_train_mean = Y_true[b_mask & ~is_val].mean(0)  # this body's OWN train mean, not the pooled one
        print(f"\n=== {body}: held out ({(b_mask & is_val).sum()} clips) ===")
        report_all(b_mask & is_val, b_train_mean)
        print(f"=== {body}: train split ({(b_mask & ~is_val).sum()} clips), for reference ===")
        report_all(b_mask & ~is_val, b_train_mean)

    pooled_train_mean = Y_true[~is_val].mean(0)
    print(f"\n=== both bodies pooled: held out ({is_val.sum()} clips) ===")
    report_all(is_val, pooled_train_mean)


if __name__ == "__main__":
    main()
