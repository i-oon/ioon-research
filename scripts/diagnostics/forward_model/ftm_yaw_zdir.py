"""z-space view of the yaw loss (reads ftm_yaw_dump.py output): within-group centred z, the share of z_real's
variance along each channel's ridge direction, and how well z_proj reproduces z_real along that direction."""
import sys, numpy as np

def corr(x, y):
    return np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan

for p in sys.argv[1:]:
    d = np.load(p); T = d["truth"]; nG = len(T)
    Zr = d["z_real"] - d["z_real"].mean(1, keepdims=True); Zp = d["z_proj"] - d["z_proj"].mean(1, keepdims=True)
    Tc = T - T.mean(1, keepdims=True)
    zr, zp, t = Zr.reshape(-1, Zr.shape[-1]), Zp.reshape(-1, Zp.shape[-1]), Tc.reshape(-1, 3)
    print(f"== {p}")
    print(f"  within-group var: z_real {(zr ** 2).sum(1).mean():.3f}  z_proj {(zp ** 2).sum(1).mean():.3f}  "
          f"|z_proj - z_real|^2 {((zp - zr) ** 2).sum(1).mean():.3f}")
    for j, c in enumerate(("fwd", "lat", "yaw")):
        w = zr.T @ t[:, j]; w /= np.linalg.norm(w)  # covariance direction
        sr, sp = zr @ w, zp @ w
        print(f"  {c}: z_real var share along its direction {np.var(sr) / (zr ** 2).sum(1).mean():.4f} | "
              f"z_proj share {np.var(sp) / (zp ** 2).sum(1).mean():.4f} | r(real_dir, truth) {corr(sr, t[:, j]):.2f} r(proj_dir, truth) {corr(sp, t[:, j]):.2f} | r(proj_dir, real_dir) {corr(sp, sr):.2f} | "
              f"std ratio proj/real {sp.std() / sr.std():.2f}")
