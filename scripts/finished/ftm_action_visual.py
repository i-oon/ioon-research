"""Does the forward model's prediction change with the action? One start state, 24 candidate actions (B1).

Real futures: the rendered counterfactual outcomes in `results/wm/cache/counterfactual_horizon.pt`
(every candidate's own motion applied from the same start state, 5 steps, rendered, V-JEPA2-encoded).
Predictions: FTM(e_s, proj(a)) of the given checkpoint. The FTM predicts embeddings, not pixels, so
the image strip shows, for each action, the library frame whose embedding (mean over tokens) is
nearest to (top) the real future and (bottom) the prediction.

    .venv/bin/python3 scripts/figures/ftm_action_visual.py \\
        --ckpt wm/runs/fmd_beh24_s0/b1_lora_c3/ckpt_lib_s4.pt --out results/deck/weekly
Outputs: ftm_action_froude.png (Froude read from real vs predicted future, per action, vs truth),
         ftm_action_strip.png (6 actions: start frame, nearest frame to real / predicted future).
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts", "sim/control", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from rollout_state_action_anova import Models  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.policy.planner import action_chunk_at, condition_of, load_candidates  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--state", default="", help="t,s key of the start state; default: widest true spread")
    ap.add_argument("--out", default="results/deck/weekly")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = "cuda"
    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "b1", per_condition=999)
    n = len(cands)
    bm = [np.asarray(load(c["path"], REGISTRY["b1"])["body_motion"])[:, :3] for c in cands]
    CF = torch.load(os.path.join(ROOT, "results/wm/cache/counterfactual_horizon.pt"))
    EM = torch.load(os.path.join(ROOT, "results/wm/cache/selection_eval_cands.pt"))
    m = Models(os.path.join(ROOT, args.ckpt), "b1", cands[0]["actions"].shape[1], dev)
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
    k = m.stride
    rd = lambda z: m.md.body(None, z).cpu().numpy() * std + mean
    keys = sorted(CF)
    if args.state:
        key = tuple(int(x) for x in args.state.split(","))
    else:
        key = max(keys, key=lambda ts: np.stack([b[ts[0]:ts[0] + k].mean(0) for b in bm])[:, 0].std())
    t, s = key
    rec = CF[key]
    e_s = rec["e_s"].float().to(dev).expand(n, -1, -1)
    real = rec["traj"][:, k - 1].float().to(dev)
    a = np.stack([action_chunk_at(c["actions"], t + m.action_lag, k) for c in cands])
    z = m.proj(torch.as_tensor(a, device=dev), "b1")
    pred = m.ftm_step(e_s, z)
    truth = np.stack([b[t:t + k].mean(0) for b in bm])
    f_real, f_pred = rd(m.itm(e_s, real)), rd(m.itm(e_s, pred))
    # how much the future differs across actions: mean pairwise distance of mean-pooled embeddings
    spread = lambda x: torch.cdist(x.mean(1), x.mean(1)).mean().item()
    sp_real, sp_pred = spread(real), spread(pred)

    order = np.argsort(truth[:, 0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, j, nm in zip(axes, (0, 2), ("forward", "yaw")):
        x = np.arange(n)
        ax.plot(x, truth[order, j], color="black", lw=2, label="true Froude of the action")
        ax.plot(x, f_real[order, j], color="#5b7fa6", lw=2, marker="o", ms=3, label="read from the real future")
        ax.plot(x, f_pred[order, j], color="#d9822b", lw=2, marker="s", ms=3, label="read from the predicted future")
        ax.set(xlabel="24 candidate actions from one start state (sorted by true forward speed)",
               ylabel=f"{nm} Froude", title=f"{nm}")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Same start state, 24 different actions (B1). Spread of the future across actions: "
                 f"real {sp_real:.1f}, predicted {sp_pred:.1f} (mean pairwise embedding distance)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    fig.savefig(os.path.join(ROOT, args.out, "ftm_action_froude.png"), dpi=150)
    plt.close(fig)

    # nearest library frame (mean-pooled embedding) to the real and the predicted future
    pool_e, pool_ref = [], []
    for ci, c in enumerate(cands):
        e = EM[c["path"]].float()
        pool_e.append(e.mean(1))
        pool_ref += [(ci, tt) for tt in range(len(e))]
    pool_e = torch.cat(pool_e).to(dev)
    near = lambda x: [pool_ref[i] for i in torch.cdist(x.mean(1), pool_e).argmin(1).tolist()]
    nr, npd = near(real), near(pred)
    frames = lambda ci, tt: np.load(cands[ci]["path"], allow_pickle=True)["frames"][tt]
    pick = order[np.linspace(0, n - 1, 6).astype(int)]
    fig, ax = plt.subplots(3, 6, figsize=(12, 6.4))
    for col, i in enumerate(pick):
        ax[0, col].imshow(frames(s, t)); ax[0, col].set_title(condition_of(cands[i]["path"]), fontsize=8)
        ax[1, col].imshow(frames(*nr[i])); ax[2, col].imshow(frames(*npd[i]))
        for r in range(3):
            ax[r, col].set_xticks([]); ax[r, col].set_yticks([])
    ax[0, 0].set_ylabel("start frame\n(same for all)"); ax[1, 0].set_ylabel("real future\n(nearest frame)")
    ax[2, 0].set_ylabel("predicted future\n(nearest frame)")
    fig.suptitle("Same start state, different actions: real vs predicted future (nearest library frame)")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, args.out, "ftm_action_strip.png"), dpi=120)
    distinct = lambda L: len({L[i] for i in range(n)})
    print(f"state t={t} clip={s} | future spread real {sp_real:.2f} pred {sp_pred:.2f} | distinct nearest frames "
          f"over 24 actions: real {distinct(nr)}, pred {distinct(npd)} | r(read,truth) fwd real "
          f"{np.corrcoef(f_real[:, 0], truth[:, 0])[0, 1]:.2f} pred {np.corrcoef(f_pred[:, 0], truth[:, 0])[0, 1]:.2f}")


if __name__ == "__main__":
    main()
