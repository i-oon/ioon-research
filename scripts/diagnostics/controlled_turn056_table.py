"""Table E for the controlled before/after B1 physics loop (controlled_turn056_loop.sh): mean L2 error at decision steps
(every 2 frames, as the loop) plus per-channel mean signed diff (achieved - goal) and mean achieved/goal, fwd/lat/yaw."""
import glob, os, numpy as np
B = "results/wm/closed_loop_rr/physics/controlled_turn056"
rows = []
for m in ("old", "new", "random"):
    for mech in (("random",) if m == "random" else ("direct", "rollout")):
        Es, D, A = [], [], []
        for f in sorted(glob.glob(f"{B}/{m}/ep*/{mech}_w21/b1_hexapod_ep40110.npz")):
            d = np.load(f); a, g = d["achieved_froude"], d["goal_froude_t"]; k = np.arange(0, len(a), 2)
            Es.append(np.linalg.norm(a[k] - g[k], axis=1).mean()); D.append((a[k] - g[k]).mean(0)); A.append(a[k].mean(0))
        G = g[k].mean(0); Es = np.array(Es); D = np.mean(D, 0); A = np.mean(A, 0)
        rows.append(f"{m:6s} {mech:8s} n={len(Es)} E {Es.mean():.4f} (eps {' '.join(f'{e:.4f}' for e in Es)}; range {np.ptp(Es):.4f}) | "
                    f"diff fwd {D[0]:+.3f} lat {D[1]:+.3f} yaw {D[2]:+.3f} | achieved {A[0]:.3f}/{A[1]:+.3f}/{A[2]:.3f} goal {G[0]:.3f}/{G[1]:+.3f}/{G[2]:.3f}")
txt = "\n".join(rows); print(txt)
open(f"{B}/table_E.txt.tmp", "w").write(txt + "\n"); os.replace(f"{B}/table_E.txt.tmp", f"{B}/table_E.txt")
