"""Same start state, different actions: is each command's IMAGINED outcome close to its TRUE outcome? (2026-10-14)

Replaces `action_future_strip.py` (Ajan Blink week 20: its green / red was an eye test on pictures; the rule -- the
prediction's nearest real future out of 24, in embedding space -- was nowhere on the figure and counted near-identical
futures as wrong). Here the rule is numeric and printed on the figure:

  one held-out B1 start state (`b1_branches_heldout`: the same physical state, every one of the 24 commands run in
  physics for 1 s); per command:
    TRUE      mean Froude (forward / lateral / yaw) over the 1 s after the switch (rows first_pair .. +19, labels as in
              training: centre of mass, never across the switch)
    IMAGINED  the planner's rollout read: 4 FTM steps (0.25 s each) from the real start embedding, each command chunk
              through the projector, Froude head on every imagined transition, averaged (RolloutFroudePlanner)
    PASS      |IMAGINED - TRUE| <= 1 SD on EVERY channel, SD = the spread of the TRUE values across the 24 commands of
              this state (the prediction is closer to its own outcome than commands typically differ from each other)
  also: within-state Spearman per channel (does the imagination rank the 24 outcomes like physics?).

One model per process (a grayscale and a colour checkpoint cannot share the encoder):
    .venv/bin/python3 scripts/figures/action_outcome_check.py run --ckpt "no branches=results/eval/round1_clips_s0/ckpt/b1.pt"
    .venv/bin/python3 scripts/figures/action_outcome_check.py run --ckpt "current=results/eval/round1_branches_s0_gray_imread/ckpt/b1.pt"
    .venv/bin/python3 scripts/figures/action_outcome_check.py plot      # -> results/deck/weekly_1015/action_outcome_check.png
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data/counterfactual_walks/b1_branches_heldout")
OUTD = os.path.join(ROOT, "results/deck/weekly_1015")
TMP = os.path.join(ROOT, "results/eval/_action_outcome_check")
STEPS = 4                                   # 4 x 0.25 s = the branch's 1 s after the switch (imread_k 4)
PICK = ("speed_vx0.50", "speed_vx-0.40", "turn_w0.075", "turn_w0.075_neg", "side_L_lvl1", "side_R_lvl1")


def state_files():
    """the first held-out branch state (sorted) that has all 24 commands"""
    g = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
        x = np.load(f, allow_pickle=True)
        g[(str(x["cf_source"]), int(x["cf_t"]))].append(f)          # one state = source clip + branch frame
    key = next(k for k in sorted(g) if len(g[k]) == 24)
    return key, sorted(g[key], key=lambda f: int(np.load(f, allow_pickle=True)["cf_command_index"]))


@torch.no_grad()
def run(spec):
    from wm.policy.planner import RolloutFroudePlanner, action_chunk_at
    from wm.data.embodiment import REGISTRY, load
    from wm.evaluate import encode_clip, offset_for, offset_as
    lab, ck = spec.split("=", 1)
    rp = RolloutFroudePlanner.from_checkpoint(os.path.join(ROOT, ck), os.path.join(ROOT, "data/counterfactual_walks/b1_clips_heldout"),
                                              embodiment="b1", horizon=2, per_condition=999, device="cuda")
    from vjepa2_encoder import VJEPA2FrameEncoder
    enc = VJEPA2FrameEncoder(dtype=torch.float32)
    off = offset_for(torch.load(os.path.join(ROOT, ck), map_location="cpu", weights_only=False), "b1")
    key, files = state_files()
    k = rp.stride
    names, true, pred = [], [], []
    for f in files:
        x = np.load(f, allow_pickle=True); c = load(f, REGISTRY["b1"])
        P = int(x["first_pair"]); T = P - 1
        e = encode_clip(enc, np.asarray(x["frames"])[[T]], 1).float()
        if off is not None:
            e = e - offset_as(off.float(), (1, *e.shape[1:]))
        e_t = e.cuda(); cur = e_t; reads = []
        for i in range(STEPS):
            ch = action_chunk_at(np.asarray(c["actions"]), T + rp.action_lag + i * k, k)
            nxt = rp.ftm(cur, rp.proj(torch.as_tensor(ch[None], device="cuda"), "b1"))
            reads.append(rp.md.body(e_t, rp.itm(cur, nxt)).float().cpu().numpy()[0])
            cur = nxt
        pred.append(np.mean(reads, 0) * rp.std_s + rp.mean_s)
        true.append(np.asarray(c["body_motion"])[P:P + STEPS * k, :3].mean(0))
        names.append(str(x["cf_command_name"]))
    os.makedirs(TMP, exist_ok=True)
    json.dump(dict(label=lab, ckpt=ck, state=list(map(str, key)), names=names, true=np.asarray(true).tolist(),
                   pred=np.asarray(pred).tolist(), gray=os.environ.get("VJEPA_GRAY") == "1"),
              open(os.path.join(TMP, f"{lab.replace(' ', '_')}.json"), "w"))
    print(lab, "done; VJEPA_GRAY", os.environ.get("VJEPA_GRAY"))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(TMP, "*.json")))]
    runs.sort(key=lambda r: r["label"] != "no branches")
    key, files = state_files()
    names = runs[0]["names"]
    true = np.asarray(runs[0]["true"]); sd = true.std(0)
    cols = [names.index(p) for p in PICK if p in names]
    x0 = np.load(files[0], allow_pickle=True); T = int(x0["first_pair"]) - 1
    fig, ax = plt.subplots(2 + len(runs), len(cols), figsize=(2.3 * len(cols), 2.2 * (2 + len(runs))))
    ch = ("fwd", "lat", "yaw")
    summary = []
    for r in runs:
        pr = np.asarray(r["pred"]); ok = (np.abs(pr - true) <= sd).all(1)
        rho = [spearmanr(pr[:, j], true[:, j]).correlation for j in range(3)]
        r["ok"], r["rho"] = ok, rho
        summary.append(f"{r['label']}: {ok.sum()}/24 commands pass; Spearman fwd {rho[0]:+.2f} lat {rho[1]:+.2f} yaw {rho[2]:+.2f}")
    for c_, i in enumerate(cols):
        x = np.load(files[i], allow_pickle=True)
        ax[0, c_].imshow(x["frames"][T]); ax[0, c_].set_title(names[i], fontsize=8)
        ax[1, c_].imshow(x["frames"][T + STEPS * 5])
        for rr, r in enumerate(runs, start=2):
            a = ax[rr, c_]; pr = np.asarray(r["pred"])[i]
            xs = np.arange(3)
            a.bar(xs - 0.18, true[i], 0.34, color="0.35", label="true (physics)")
            a.bar(xs + 0.18, pr, 0.34, color="#1f5fd1", label="imagined")
            a.errorbar(xs - 0.18, true[i], yerr=sd, fmt="none", ecolor="0.6", capsize=3, lw=1)
            a.axhline(0, color="0.8", lw=0.8); a.set_xticks(xs); a.set_xticklabels(ch, fontsize=7)
            a.set_ylim(min(-0.2, true.min() - 0.05), max(0.3, true.max() + 0.05)); a.tick_params(labelsize=6)
            bad = [ch[j] for j in range(3) if abs(pr[j] - true[i][j]) > sd[j]]
            a.set_xlabel("PASS" if not bad else "off: " + ", ".join(bad), fontsize=7,
                         color="#2e7d32" if not bad else "#c62828")
            for s in a.spines.values():
                s.set_edgecolor("#2e7d32" if not bad else "#c62828"); s.set_linewidth(2.5)
    for r_ in range(2):
        for c_ in range(len(cols)):
            ax[r_, c_].set_xticks([]); ax[r_, c_].set_yticks([])
    ax[0, 0].set_ylabel("start frame\n(same state)", fontsize=8)
    ax[1, 0].set_ylabel("real view\nafter 1 s", fontsize=8)
    for rr, r in enumerate(runs, start=2):
        ax[rr, 0].set_ylabel(f"{r['label']}\nFroude, 1 s", fontsize=8)
    ax[2, 0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.suptitle("Same start state (B1, held-out), 24 commands run in physics. PASS = imagined Froude within 1 SD of the "
                 "true Froude on every channel (SD = spread of the true values across the 24 commands; grey whiskers)\n"
                 + "   |   ".join(summary), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(OUTD, exist_ok=True)
    out = os.path.join(OUTD, "action_outcome_check.png"); fig.savefig(out, dpi=130)
    print("\n".join(summary)); print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "plot"))
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    run(a.ckpt) if a.cmd == "run" else plot()
