"""Plot gait/contact diagrams for the IK forward-walk dataset.

The goal is to show that the IK-retargeted clips preserve the expert contact
pattern across long/medium/short morphologies. Black bars mean stance/contact.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "sim/env/expert_66k_aug3c_fcontact.csv")
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
MORPHS = ["long", "medium", "short"]
EP = 66
NAME_RE = re.compile(r"^(long|medium|short)_ep(\d+)(?:_r\d+)?\.npz$")


def smooth_contacts(contacts, window=3):
    contacts = np.asarray(contacts, dtype=float)
    if len(contacts) < window:
        return contacts > 0.5
    pad = window // 2
    out = np.empty_like(contacts, dtype=bool)
    kernel = np.ones(window)
    for leg in range(contacts.shape[1]):
        x = np.pad(contacts[:, leg], pad, mode="edge")
        out[:, leg] = np.convolve(x, kernel, mode="valid") >= (window // 2 + 1)
    return out


def stance_onsets(contact_1d):
    x = np.asarray(contact_1d, dtype=bool)
    return np.flatnonzero((~x[:-1]) & x[1:]) + 1


def stride_period(contacts):
    c = smooth_contacts(contacts)
    intervals = []
    for leg in range(c.shape[1]):
        d = np.diff(stance_onsets(c[:, leg]))
        intervals.extend(d[(d >= 4) & (d <= max(4, len(c) // 2))].tolist())
    if not intervals:
        return None
    med = float(np.median(intervals))
    keep = [d for d in intervals if 0.5 * med <= d <= 1.5 * med]
    return max(1, int(round(np.median(keep or intervals))))


def phase_offsets(contacts, period, ref=0):
    contacts = smooth_contacts(contacts).astype(float)
    n = len(contacts)
    if not period:
        return np.full(6, np.nan)
    r = contacts[:, ref] - contacts[:, ref].mean()
    lags = np.arange(-n + 1, n)
    keep = (lags >= 0) & (lags < period)
    out = []
    for i in range(6):
        c = contacts[:, i] - contacts[:, i].mean()
        if c.std() < 1e-6 or r.std() < 1e-6:
            out.append(np.nan)
            continue
        cc = np.correlate(c, r, "full")
        out.append((lags[keep][np.argmax(cc[keep])] % period) / period)
    return np.asarray(out)


def circular_distance(a, b):
    d = np.abs(a - b) % 1.0
    d = np.minimum(d, 1.0 - d)
    return float(np.nanmean(d))


def load_ik_contacts(path, threshold):
    d = np.load(path)
    return smooth_contacts(d["forces"] > threshold)


def load_expert_contacts(df, ep):
    rows = df.iloc[ep * EP:(ep + 1) * EP]
    return smooth_contacts(rows[[f"contact_{leg}" for leg in LEGS]].to_numpy(float))


def plot_contact(ax, contacts, title):
    c = smooth_contacts(contacts)
    x = np.arange(len(c))
    for i in range(6):
        ax.fill_between(x, i + 0.05, i + 0.95, where=c[:, i], step="pre", color="black", linewidth=0)
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_yticklabels(LEGS)
    ax.set_ylim(0, 6)
    ax.set_xlim(0, len(c) - 1)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("frame")


def collect_files(data_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        morph, ep = m.group(1), int(m.group(2))
        out[(morph, ep)] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ik_walk_3sec")
    ap.add_argument("--out", default="results/gait_ik_walk_3sec")
    ap.add_argument("--contact-threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(CSV)
    files = collect_files(args.data)
    episodes = sorted({ep for _, ep in files})
    if not episodes:
        raise SystemExit(f"no IK walk npz files found in {args.data}")

    metrics_rows = []
    for ep in episodes:
        expert = load_expert_contacts(df, ep)
        expert_period = stride_period(expert)
        expert_phase = phase_offsets(expert, expert_period)
        expert_duty = expert.mean(axis=0)

        fig, axes = plt.subplots(4, 1, figsize=(10, 6.6), sharex=True)
        plot_contact(axes[0], expert, f"expert ep{ep}")

        for ax, morph in zip(axes[1:], MORPHS):
            path = files.get((morph, ep))
            if path is None:
                ax.set_axis_off()
                continue
            c = load_ik_contacts(path, args.contact_threshold)
            plot_contact(ax, c, f"IK {morph} ep{ep}")
            period = stride_period(c)
            phase = phase_offsets(c, period)
            metrics_rows.append({
                "ep": ep,
                "morph": morph,
                "period": period if period is not None else np.nan,
                "phase_distance_to_expert": circular_distance(phase, expert_phase),
                "duty_mae_to_expert": float(np.nanmean(np.abs(c.mean(axis=0) - expert_duty))),
                "support_mean": float(c.sum(axis=1).mean()),
            })

        fig.suptitle(f"IK-retargeted walk contact pattern vs expert, episode {ep}", y=0.99)
        fig.tight_layout()
        out_path = os.path.join(args.out, f"gait_diagram_ep{ep}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")

    rows = pd.DataFrame(metrics_rows)
    rows.to_csv(os.path.join(args.out, "metrics.csv"), index=False)

    # Summary contact sheet: one compact panel per episode.
    fig, axes = plt.subplots(len(episodes), 4, figsize=(14, 2.35 * len(episodes)), squeeze=False)
    for r, ep in enumerate(episodes):
        plot_contact(axes[r, 0], load_expert_contacts(df, ep), f"expert ep{ep}")
        for cidx, morph in enumerate(MORPHS, start=1):
            path = files.get((morph, ep))
            if path is None:
                axes[r, cidx].set_axis_off()
            else:
                plot_contact(axes[r, cidx], load_ik_contacts(path, args.contact_threshold), f"{morph}")
    fig.suptitle("IK forward-walk gait diagrams: expert vs retargeted morphologies", y=0.995)
    fig.tight_layout()
    summary = os.path.join(args.out, "gait_diagram_all.png")
    fig.savefig(summary, dpi=150)
    plt.close(fig)
    print(f"saved {summary}")

    print("\nmetrics summary")
    print(rows.groupby("morph")[["phase_distance_to_expert", "duty_mae_to_expert", "support_mean"]].mean().round(3))
    print(f"\nmetrics -> {os.path.join(args.out, 'metrics.csv')}")


if __name__ == "__main__":
    main()
