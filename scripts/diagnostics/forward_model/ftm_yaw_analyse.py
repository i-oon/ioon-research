"""Analyse results/check/ftm_yaw/<tag>_<name>.npz from ftm_yaw_dump.py (see its docstring)."""
import sys, numpy as np

CH = ("fwd", "lat", "yaw")
FAM = {"speed_vx": "speed", "turn": "turn", "side": "side"}


def corr(x, y):
    return np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan


def fam(n):
    if n.startswith("turn"): return "turn"
    if n.startswith("side"): return "side"
    if "vx-" in n or "bwd" in n: return "bwd"
    return "fwd"


def ridge_cv(X, Y, groups, lam_grid=(1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4), train_X=None):
    """leave-groups-out (4 folds) dual ridge; per-group Pearson r of prediction vs truth. train_X: fit on
    a different feature set (e.g. real deltas) and apply to X."""
    TX = X if train_X is None else train_X
    sc = TX.std() + 1e-12; TX = TX / sc; X = X / sc
    lam_grid = [l * TX.shape[1] / 100 for l in lam_grid]
    ug = np.unique(groups); folds = np.array_split(ug, 4)
    best = None
    for lam in lam_grid:
        pred = np.zeros_like(Y)
        for f in folds:
            te = np.isin(groups, f); tr = ~te
            mu = TX[tr].mean(0); ym = Y[tr].mean(0)
            A = TX[tr] - mu
            alpha = np.linalg.solve(A @ A.T + lam * np.eye(len(A)), Y[tr] - ym)
            pred[te] = (X[te] - mu) @ (A.T @ alpha) + ym
        r = [np.nanmean([corr(pred[groups == g, j], Y[groups == g, j]) for g in ug]) for j in range(3)]
        if best is None or r[2] > best[0][2]:
            best = (r, lam)
    return best


def main(path):
    d = np.load(path)
    T = d["truth"]; names = d["names"]; nG = len(T)
    fams = np.array([[fam(n) for n in row] for row in names])
    print(f"== {path}  groups {nG}  K {d['K']}  P {d['P']}")
    print("-- within-group r across 24 commands (mean over groups) fwd / lat / yaw")
    for k in ("F_real", "F_direct", "F_ftm_proj", "F_ftm_true"):
        r = [[corr(d[k][g][:, j], T[g][:, j]) for j in range(3)] for g in range(nG)]
        print(f"  {k:<11}" + " / ".join(f"{x:.2f}" for x in np.nanmean(r, 0)))
    print("-- read-out amplitude: slope of read on truth (pooled within-group centred), fwd / lat / yaw")
    for k in ("F_real", "F_direct", "F_ftm_proj", "F_ftm_true"):
        Xc = d[k] - d[k].mean(1, keepdims=True); Tc = T - T.mean(1, keepdims=True)
        print(f"  {k:<11}" + " / ".join(f"{(Xc[..., j] * Tc[..., j]).sum() / (Tc[..., j] ** 2).sum():.2f}" for j in range(3)))
    print("-- truth std across commands (mean over groups):", " / ".join(f"{x:.3f}" for x in T.std(1).mean(0)))
    print("-- yaw r within family (8-command turn family, 4 per sign etc.)")
    for fm in ("turn", "side", "fwd", "bwd"):
        row = []
        for k in ("F_real", "F_direct", "F_ftm_proj", "F_ftm_true"):
            row.append(np.nanmean([corr(d[k][g][fams[g] == fm, 2], T[g][fams[g] == fm, 2]) for g in range(nG)]))
        print(f"  {fm:<5} yaw truth std {T[:, :, 2][fams == fm].std():.3f} | real {row[0]:.2f} direct {row[1]:.2f} "
              f"ftm_proj {row[2]:.2f} ftm_true {row[3]:.2f}")
    print("-- spread of the future embedding across 24 commands (full tokens): total var, share explained by fwd/lat/yaw")
    for k in ("real", "proj", "true"):
        sh = d["share_" + k].mean(0)
        print(f"  {k:<5} var {d['var_' + k].mean():.1f}  share " + " / ".join(f"{x:.3f}" for x in sh))
    print("-- predicted vs real centred future: cos, norm ratio (mean per family)")
    for k in ("proj", "true"):
        s = "  " + k + ": "
        for fm in ("fwd", "bwd", "turn", "side"):
            s += f"{fm} cos {d['cos_' + k][fams == fm].mean():.2f} nr {d['norm_' + k][fams == fm].mean():.2f} | "
        print(s)
    # z separation
    print("-- z: ridge probe (4-fold by group) from z, within-group r fwd / lat / yaw")
    g_ = np.repeat(np.arange(nG), T.shape[1])
    for k in ("z_real", "z_proj"):
        r, lam = ridge_cv(d[k].reshape(-1, d[k].shape[-1]), T.reshape(-1, 3), g_)
        print(f"  {k}: " + " / ".join(f"{x:.2f}" for x in r))
    # z distance between commands, per family pair: is turn L vs turn R separated in z_proj as in z_real?
    for k in ("z_real", "z_proj"):
        Z = d[k] - d[k].mean(1, keepdims=True)
        tot = (Z ** 2).sum(-1).mean()
        tl = [i for i, n in enumerate(names[0]) if n.startswith("turn") and not n.endswith("neg")]
        tr = [i for i, n in enumerate(names[0]) if n.startswith("turn") and n.endswith("neg")]
        dlr = ((Z[:, tl].mean(1) - Z[:, tr].mean(1)) ** 2).sum(-1).mean()
        print(f"  {k}: |mean(turn+) - mean(turn-)|^2 / mean |z - group mean|^2 = {dlr / tot:.2f}")
    zr = d["z_real"] - d["z_real"].mean(1, keepdims=True); zp = d["z_proj"] - d["z_proj"].mean(1, keepdims=True)
    cs = (zr * zp).sum(-1) / (np.linalg.norm(zr, axis=-1) * np.linalg.norm(zp, axis=-1) + 1e-9)
    print("  cos(z_proj, z_real) centred, per family: " + " ".join(f"{fm} {cs[fams == fm].mean():.2f}" for fm in ("fwd", "bwd", "turn", "side")))
    # linear probes on pooled deltas
    groups = np.repeat(np.arange(nG), T.shape[1]); Y = T.reshape(-1, 3)
    print("-- linear probe (ridge, 4-fold by group) on 4x4-pooled delta; within-group r fwd / lat / yaw")
    Xr = d["pool_real"].reshape(len(Y), -1).astype(np.float32)
    for k in ("real", "proj", "true"):
        X = d["pool_" + k].reshape(len(Y), -1).astype(np.float32)
        r, lam = ridge_cv(X, Y, groups)
        print(f"  fit+test on {k:<5}: " + " / ".join(f"{x:.2f}" for x in r) + f"  (lam {lam:g})")
        if k != "real":
            r, lam = ridge_cv(X, Y, groups, train_X=Xr)
            print(f"  fit real, test {k:<5}: " + " / ".join(f"{x:.2f}" for x in r))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
