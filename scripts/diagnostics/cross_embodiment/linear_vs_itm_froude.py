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

**Important asymmetry, stated rather than hidden**: the linear baseline is fit fresh on a split
defined here; the existing checkpoint's `body_head`/ITM were fit earlier on their own (different,
unknown-to-this-script) split, which may overlap these test clips. That would flatter the existing
pipeline, not the baseline -- so if the baseline still matches or beats it, that is not an artifact
of this asymmetry; if the baseline loses, the gap could be partly inflated by this.

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
from wm.data.embodiment import HEXAPOD, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

CKPT = "wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1_hex_v2.pt"
DATA_DIR = "data/egocentric/beh12_c10f10t10_ego_flat"
SEED = 0
VAL_FRACTION = 0.25


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
    offset = offset_for(ck, "hexapod")
    encoder = VJEPA2FrameEncoder(device=str(device), dtype=torch.float32)

    paths = sorted(glob.glob(os.path.join(ROOT, DATA_DIR, "*.npz")))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(paths))
    val_n = max(1, int(VAL_FRACTION * len(paths)))
    val_idx = set(order[:val_n].tolist())

    X_pair, Y_true, Z_itm_head, is_val = [], [], [], []
    for i, p in enumerate(paths):
        clip = load(p, HEXAPOD)
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
        mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
        std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]
        fr_pred = fr_pred_std * std_s + mean_s   # un-standardize: body_head predicts standardized units
        goal = np.asarray(clip["body_motion"])[:, channels].mean(0)

        X_pair.append(pair.cpu().numpy())
        Y_true.append(goal)
        Z_itm_head.append(fr_pred)
        is_val.append(i in val_idx)
        print(f"  encoded {os.path.basename(p)}  {'[test]' if i in val_idx else '[train]'}")

    X_pair = np.stack(X_pair); Y_true = np.stack(Y_true); Z_itm_head = np.stack(Z_itm_head)
    is_val = np.asarray(is_val)

    W = ridge_fit(X_pair[~is_val], Y_true[~is_val], lam=10.0)
    Y_lin_pred = ridge_predict(W, X_pair)

    net, mean_y, std_y, final_loss = mlp_fit(X_pair[~is_val], Y_true[~is_val], device)
    print(f"\n  MLP baseline: final train loss (standardized MSE) {final_loss:.4f}")
    Y_mlp_pred = mlp_predict(net, mean_y, std_y, X_pair, device)

    def report(name, pred, mask):
        err = np.abs(pred[mask] - Y_true[mask]).sum(-1)
        ss_res = ((pred[mask] - Y_true[mask]) ** 2).sum()
        ss_tot = ((Y_true[mask] - Y_true[~mask].mean(0)) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        print(f"  {name:32s} n={mask.sum():3d}  median|err|={np.median(err):.4f}  "
             f"mean|err|={err.mean():.4f}  R2(vs train mean)={r2:+.3f}")

    print(f"\n=== held out ({is_val.sum()} clips), train mean baseline is the train split's own mean ===")
    report("linear ridge (no ITM, no z)", Y_lin_pred, is_val)
    report("MLP, same capacity as body_head (no ITM)", Y_mlp_pred, is_val)
    report("existing body_head(ITM(.))", Z_itm_head, is_val)
    print(f"\n=== train split ({(~is_val).sum()} clips), for reference ===")
    report("linear ridge (no ITM, no z)", Y_lin_pred, ~is_val)
    report("MLP, same capacity as body_head (no ITM)", Y_mlp_pred, ~is_val)
    report("existing body_head(ITM(.))", Z_itm_head, ~is_val)


if __name__ == "__main__":
    main()
