"""Render-lock / structure check (Step 1a).

Encodes every frame with frozen V-JEPA2 (mean-pooled e_t), then checks:
  - does e_t carry BODY (morphology)?  -> expected (the raw-e_t signal the latent later removes)
  - THE GATE: within each (body, behavior), can we decode which REPEAT a frame came from?
    Repeats are the same body+behavior re-recorded, so if render is locked they should be
    ~chance (overlap). If repeat is decodable, the recording setup leaks a session/render signal.

  python3 scripts/render_lock_check.py --data data/ik_v2 --out results/render_lock_ik_v2
"""
import argparse, glob, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/render_lock")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    frames, body, epi, rep, beh = [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        fr = d["frames"]
        b = str(d["morph"]) if "morph" in d.files else os.path.basename(f).split("_")[0]
        e = str(int(d["expert_episode"])) if "expert_episode" in d.files else "0"
        r = str(int(d["repeat"])) if "repeat" in d.files else "0"
        bh = str(d["behavior"]) if "behavior" in d.files else "walk"     # pre-behavior clips = walk
        frames.append(fr); body += [b]*len(fr); epi += [e]*len(fr); rep += [r]*len(fr); beh += [bh]*len(fr)
    frames = np.concatenate(frames)
    body, epi, rep, beh = map(np.array, (body, epi, rep, beh))
    be = np.array([f"{b}_{e}_{bh}" for b, e, bh in zip(body, epi, beh)])   # body+behavior group (for the gate)
    print(f"{len(files)} clips, {len(frames)} frames | bodies={sorted(set(body))} "
          f"behaviors={sorted(set(beh))} episodes={sorted(set(epi))} repeats={sorted(set(rep))}")

    from vjepa2_encoder import VJEPA2FrameEncoder
    enc = VJEPA2FrameEncoder()
    embs = []
    for i in range(0, len(frames), args.batch):
        e = enc.encode(list(frames[i:i + args.batch]))
        embs.append(e.float().mean(1).cpu().numpy())
        print(f"  encoded {min(i+args.batch, len(frames))}/{len(frames)}", end="\r")
    E = np.concatenate(embs); print(f"\nembeddings {E.shape}")
    np.savez_compressed(os.path.join(args.out, "emb.npz"), E=E, body=body, epi=epi, rep=rep)

    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    Es = StandardScaler().fit_transform(E)

    def dec(X, y):
        return cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=3).mean()

    print("\n=== structure (raw e_t) ===")
    print(f"silhouette(body)     = {silhouette_score(Es, body):+.3f}   (expected > 0: morphology present)")
    if len(set(beh)) > 1:
        print(f"silhouette(behavior) = {silhouette_score(Es, beh):+.3f}")
    else:
        print("silhouette(behavior) = skipped (only one behavior)")
    print(f"body decode          = {dec(Es, body):.3f}   (chance {1/len(set(body)):.3f})")
    if len(set(beh)) > 1:
        print(f"behavior decode      = {dec(Es, beh):.3f}   (chance {1/len(set(beh)):.3f})  "
              "[raw e_t; the latent should improve behavior-transfer across bodies]")

    print("\n=== RENDER-LOCK GATE: within each (body,behavior), decode the REPEAT ===")
    print("(same body+behavior re-recorded -> should be ~chance if the render is locked)")
    accs, ch = [], 1 / max(2, len(set(rep)))
    for g in sorted(set(be)):
        m = be == g; yr = rep[m]
        if len(set(yr)) > 1:
            a = dec(Es[m], yr); accs.append(a)
            print(f"  {g:14s}: repeat decode {a:.3f}  (chance {1/len(set(yr)):.3f})")
    if accs:
        mean = float(np.mean(accs))
        print(f"\n  MEAN repeat-decode = {mean:.3f}  vs chance {ch:.3f}  ->  "
              f"{'PASS — repeats overlap, render locked' if mean < 1.5*ch else 'FAIL — repeats separate → render/session confound'}")
    else:
        print("  skipped (no repeated clips within a body/behavior group)")

    try:
        import umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Z = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(Es)
        fig, ax = plt.subplots(1, 4, figsize=(23, 5.5))
        for k, (lab, col) in enumerate([("body", body), ("behavior", beh), ("episode", epi), ("repeat", rep)]):
            for v in sorted(set(col)):
                mm = col == v; ax[k].scatter(Z[mm, 0], Z[mm, 1], s=5, alpha=0.5, label=v)
            ax[k].set_title(f"UMAP by {lab}"); ax[k].legend(fontsize=6, markerscale=2); ax[k].axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "umap.png"), dpi=110)
        print(f"\nUMAP -> {os.path.join(args.out, 'umap.png')}")
    except Exception as e:
        print("UMAP skipped:", e)


if __name__ == "__main__":
    main()
