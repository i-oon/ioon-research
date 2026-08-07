"""Gait diagnostics: is the learned policy's gait actually *like the expert/discriminator*?

Rolls out a checkpoint, then produces (all saved to results/gait/<tag>/):
  1. gait_diagram.png  - stance/swing phase plot, 6 legs, LEARNED (top) vs EXPERT (bottom).
                         This is the "gait phase plot": black = foot down (stance).
  2. gait_cycle.png    - average one-stride template on a shared 0-100% phase axis. This is
                         directly comparable even when sampling rates and stride lengths differ.
  3. g_hist.png        - histogram of discriminator reward g(s') for the learned rollout's
                         states vs the EXPERT's states. If they overlap, the learned gait
                         looks "expert-like" through the discriminator's eyes.
  4. metrics.txt       - numbers to compare: per-leg duty factor (learned vs expert),
                         tripod anti-phase score, mean g (learned vs expert), forward reach,
                         lateral drift, yaw drift, fall step.

Needs its own sim (the training sims are busy), e.g. CoppeliaSim_d on 23063:
  cd /home/aria/CoppeliaSim_d && ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23063
  .venv/bin/python3 scripts/gait_report.py --port 23063 \
      --scene sim/env/medauroidea_stick_insect.ttt \
      --ckpt amp/logs/insect_long/<ts>/model/step100000 --tag long_100k
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "amp"))
from networks.actor import ActorNetworkPolicy          # noqa: E402
from networks.discrim import AIRLDiscrim                # noqa: E402
from common.normalized_env_66k import CoppeliaSimEnv    # noqa: E402

LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
SEG = ["TC", "CF", "FT"]
CSV = os.path.join(ROOT, "sim/env/expert_66k_aug3c_fcontact.csv")
EP = 66


def load_expert(df, ep, env):
    """Return (contacts [T,6], normalized 28-D states [T,28], dt) for one expert episode."""
    rows = df.iloc[ep * EP:(ep + 1) * EP]
    contact = rows[[f"contact_{l}" for l in LEGS]].to_numpy(float)
    raw = np.concatenate([
        rows[["body_z"]].to_numpy(float),
        rows[["body_roll", "body_pitch", "body_yaw"]].to_numpy(float),
        rows[[f"motor_pos_{l}_{s}" for l in LEGS for s in SEG]].to_numpy(float),
        contact,
    ], axis=1)                                          # order matches env OBS_SPEC (28-D)
    dt = float(np.median(np.diff(rows["sim_time"].to_numpy())))
    return contact, env.normalize_observation(raw), dt


def smooth_contacts(contacts, window=3):
    """Remove one-frame contact chatter with a short majority filter."""
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
    """Robust stride period in samples from per-leg stance-onset intervals.

    Summing all six contacts is invalid for alternating/tripod gaits: the number of feet down can
    remain almost constant and its autocorrelation then selects a long envelope/harmonic.  Onset
    intervals retain each leg's actual cycle.  Pool all legs and use a robust median so learned and
    expert trajectories may have different sample rates without biasing the estimate.
    """
    c = smooth_contacts(contacts)
    n = len(c)
    intervals = []
    for leg in range(c.shape[1]):
        d = np.diff(stance_onsets(c[:, leg]))
        intervals.extend(d[(d >= 4) & (d <= n // 2)].tolist())
    if not intervals:
        return None
    med = float(np.median(intervals))
    keep = [d for d in intervals if 0.5 * med <= d <= 1.5 * med]
    return max(1, int(round(np.median(keep or intervals))))


def phase_offsets(contacts, period, ref=0):
    """Per-leg phase lag vs a reference leg, in fraction of a stride [0,1). Rate-invariant."""
    contacts = smooth_contacts(contacts).astype(float)
    n = len(contacts)
    if not period:
        return np.full(6, np.nan)
    r = contacts[:, ref].astype(float) - contacts[:, ref].mean()
    lags = np.arange(-n + 1, n)
    keep = (lags >= 0) & (lags < period)
    out = []
    for i in range(6):
        c = contacts[:, i].astype(float) - contacts[:, i].mean()
        if c.std() < 1e-6 or r.std() < 1e-6:
            out.append(np.nan); continue
        cc = np.correlate(c, r, "full")
        out.append((lags[keep][np.argmax(cc[keep])] % period) / period)
    return np.array(out)


def coord_distance(pa, pb):
    """Mean circular distance between two phase-offset vectors: 0=identical, 0.5=opposite."""
    d = np.abs(pa - pb) % 1.0
    d = np.minimum(d, 1.0 - d)
    return float(np.nanmean(d))


def rollout(env, actor, steps):
    """Deterministic rollout; record contacts, normalized states, head path, vx."""
    state = env.reset()
    head = env.sim.getObject("/head")
    contacts, states, xs, ys, yaws, vxs = [], [], [], [], [], []
    p0 = env.sim.getObjectPosition(head)
    fell = False
    for t in range(steps):
        with torch.no_grad():
            a = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()
        state, reward, terminated, truncated, info = env.step(a)
        contacts.append(env.get_contact())
        states.append(np.asarray(state, float))
        p = env.sim.getObjectPosition(head)
        xs.append(p[0] - p0[0]); ys.append(p[1] - p0[1])
        yaws.append(env.get_bodyorientation()[2]); vxs.append(float(info["vx_avg"]))
        if terminated or truncated:
            fell = terminated
            break
    dt = float(env.sim.getSimulationTimeStep())
    env.stop()
    return (np.array(contacts), np.array(states), np.array(xs), np.array(ys),
            np.array(yaws), np.array(vxs), fell, dt)


def align_to_reference_onset(contacts, ref=0):
    c = smooth_contacts(contacts)
    onsets = stance_onsets(c[:, ref])
    start = int(onsets[0]) if len(onsets) else 0
    return c[start:]


def gait_plot(ax, contacts, period, title, n_cycles):
    """Continuous phase plot on a shared stride-cycle axis, aligned to FL stance onset."""
    contacts = align_to_reference_onset(contacts)
    T = len(contacts)
    x = np.arange(T) / period if period else np.arange(T)
    for i in range(6):
        on = contacts[:, i] > 0.5
        ax.fill_between(x, i + 0.05, i + 0.95, where=on, step="pre", color="k", linewidth=0)
    ax.set_yticks(np.arange(6) + 0.5); ax.set_yticklabels(LEGS)
    ax.set_xlim(0, n_cycles if period else T); ax.set_ylim(0, 6); ax.set_title(title)
    ax.set_xlabel("stride cycles" if period else "step"); ax.invert_yaxis()


def average_cycle(contacts, period, bins=100, ref=0):
    """Average complete reference-leg cycles after resampling each to a common phase grid."""
    c = smooth_contacts(contacts).astype(float)
    if not period:
        period = max(1, len(c))
    onsets = stance_onsets(c[:, ref])
    cycles = []
    target = (np.arange(bins) + 0.5) / bins
    for a, b in zip(onsets[:-1], onsets[1:]):
        length = b - a
        if length < 0.5 * period or length > 1.5 * period:
            continue
        phase = (np.arange(length) + 0.5) / length
        segment = np.empty((bins, c.shape[1]))
        for leg in range(c.shape[1]):
            segment[:, leg] = np.interp(target, phase, c[a:b, leg])
        cycles.append(segment)
    if not cycles:
        start = int(onsets[0]) if len(onsets) else 0
        stop = min(len(c), start + period)
        phase = (np.arange(stop - start) + 0.5) / max(1, stop - start)
        segment = np.empty((bins, c.shape[1]))
        for leg in range(c.shape[1]):
            segment[:, leg] = np.interp(target, phase, c[start:stop, leg])
        cycles.append(segment)
    return np.mean(cycles, axis=0).T, len(cycles)


def duty(contacts):
    return smooth_contacts(contacts).mean(axis=0)       # fraction of a cycle each leg is in stance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23063)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--expert_ep", type=int, default=926, help="expert episode for comparison")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vx_target", type=float, default=0.45)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--track_window", type=int, default=25)
    ap.add_argument("--track_direction_mix", type=float, default=0.5)
    ap.add_argument("--g_ood_coef", type=float, default=2.0)
    args = ap.parse_args()

    import time
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene = os.path.abspath(args.scene)
    ckpt = args.ckpt if args.ckpt.endswith(".pth") else os.path.join(args.ckpt, "actor.pth")
    out = os.path.join(ROOT, "results/gait", args.tag)
    os.makedirs(out, exist_ok=True)

    sim0 = RemoteAPIClient("localhost", port=args.port).require("sim")
    sim0.stopSimulation()
    while sim0.getSimulationState() != 0:
        time.sleep(0.1)
    sim0.loadScene(scene)

    env = CoppeliaSimEnv(port=args.port, OnTimeStep=True, reward_mode="track",
                         vx_target=args.vx_target, track_sigma=args.sigma,
                         track_window=args.track_window,
                         track_direction_mix=args.track_direction_mix)
    actor = ActorNetworkPolicy(state_shape=env.observation_space.shape,
                               action_shape=env.action_space.shape,
                               hidden_units=(64, 64), hidden_activation=torch.nn.Tanh())
    actor.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    actor.eval()

    disc = AIRLDiscrim(state_shape=env.observation_space.shape, gamma=0.995,
                       hidden_units_r=(100, 100), hidden_units_v=(100, 100),
                       hidden_activation_r=torch.nn.ReLU(), hidden_activation_v=torch.nn.ReLU())
    disc.load_state_dict(torch.load(os.path.join(ROOT, "amp/discriminator.pth"), map_location="cpu"))
    disc.eval()

    # learned rollout + expert episode
    lc, lstates, xs, ys, yaws, vxs, fell, ldt = rollout(env, actor, args.steps)
    df = pd.read_csv(CSV)
    # The expert CSV is the canonical LONG morphology used to train discriminator.pth.  Do not
    # normalize it with the current Medium/Short body's scaled body-z bounds: doing so changes the
    # same expert episode's score by morphology (observed 2.83 -> 3.28 -> 4.65), making cross-body
    # g ratios meaningless. Learned states already carry the correct morphology-scaled mapping.
    canonical_expert_env = CoppeliaSimEnv(simulation=False)
    ec, estates, edt = load_expert(df, args.expert_ep, canonical_expert_env)

    def score_states(states):
        s = torch.as_tensor(states, dtype=torch.float32)
        overflow = torch.relu(s.abs() - 1.0)
        support = torch.exp(-args.g_ood_coef * overflow.square().sum(dim=-1))
        with torch.no_grad():
            raw = disc.g(s.clamp(-1.0, 1.0)).squeeze(-1)
        return raw.numpy(), (raw * support).numpy(), support.numpy(), (s.abs() > 1.0).numpy()

    lg_raw, lg, lsup, lood = score_states(lstates)
    eg_raw, eg, esup, eood = score_states(estates)

    # rate-invariant gait descriptors (no gait template assumed)
    lp, ep = stride_period(lc), stride_period(ec)
    lph, eph = phase_offsets(lc, lp), phase_offsets(ec, ep)
    lfreq = (1.0 / (lp * ldt)) if lp else float("nan")   # strides / second (real time)
    efreq = (1.0 / (ep * edt)) if ep else float("nan")
    ld, ed = duty(lc), duty(ec)

    # ---- gait diagram: same number of normalized cycles in both panels ----
    lc_aligned, ec_aligned = align_to_reference_onset(lc), align_to_reference_onset(ec)
    lcycles = len(lc_aligned) / lp if lp else 0
    ecycles = len(ec_aligned) / ep if ep else 0
    common_cycles = max(1.0, min(6.0, lcycles, ecycles)) if (lp and ep) else 1.0
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    gait_plot(ax[0], lc, lp, f"LEARNED  {args.tag}   period={lp} samples = {lfreq:.2f} Hz   ({'FELL' if fell else 'stable'})", common_cycles)
    gait_plot(ax[1], ec, ep, f"EXPERT  ep{args.expert_ep}   period={ep} samples = {efreq:.2f} Hz", common_cycles)
    plt.tight_layout(); plt.savefig(os.path.join(out, "gait_diagram.png"), dpi=120); plt.close()

    # ---- directly comparable average stride on one shared 0-100% phase axis ----
    lprofile, ln = average_cycle(lc, lp)
    eprofile, en = average_cycle(ec, ep)
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for a, profile, title in (
        (ax[0], lprofile, f"LEARNED mean stride ({ln} cycles)"),
        (ax[1], eprofile, f"EXPERT mean stride ({en} cycles)"),
    ):
        a.imshow(profile, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1,
                 extent=(0, 100, 6, 0))
        a.set_yticks(np.arange(6) + 0.5); a.set_yticklabels(LEGS); a.set_title(title)
        a.set_ylabel("leg")
    ax[1].set_xlabel("stride phase (%)   [black = stance probability 1]")
    plt.tight_layout(); plt.savefig(os.path.join(out, "gait_cycle.png"), dpi=120); plt.close()

    # ---- discriminator g (per-frame prior; blind to stepping rate — noted) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(lg.min(), eg.min()), max(lg.max(), eg.max()), 40)
    ax.hist(eg, bins, alpha=0.6, label=f"expert  (mean {eg.mean():.2f})", color="tab:green", density=True)
    ax.hist(lg, bins, alpha=0.6, label=f"learned (mean {lg.mean():.2f})", color="tab:orange", density=True)
    ax.set_xlabel("support-gated discriminator g(s')  [per-frame; does not see rate]"); ax.set_ylabel("density")
    ax.set_title(f"per-frame posture match: {args.tag}"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, "g_hist.png"), dpi=120); plt.close()

    # ---- metrics ----
    def row(name, arr):
        return name + " ".join(f"{v:5.2f}" if not np.isnan(v) else "   NA" for v in arr)
    rate = f"  -> learned steps {lfreq/efreq:.2f}x the expert's rate" if (lp and ep) else "  -> rate NA"
    lines = [
        f"=== gait report: {args.tag}  (vs expert ep{args.expert_ep}; NO gait template assumed) ===",
        f"checkpoint: {ckpt}",
        f"rollout: {len(lc)} steps, {'FELL' if fell else 'ran to limit'}",
        "",
        "-- trajectory --",
        f"forward reach : {xs[-1]:+.2f} m      mean vx : {vxs.mean():+.3f} m/s",
        f"speed target  : {env.vx_target:.3f} m/s (morphology-scaled)",
        f"lateral drift : {ys[-1]:+.2f} m      yaw drift : {np.degrees(yaws[-1]-yaws[0]):+.1f} deg",
        "",
        "-- stepping RATE (compared in real time, since raw step counts differ) --",
        f"sample dt     : learned {ldt:.4f} s   expert {edt:.4f} s",
        f"stride period : learned {lp} samples ({lfreq:.2f} Hz)   expert {ep} samples ({efreq:.2f} Hz)",
        f"plotted cycles: {common_cycles:.2f} in BOTH gait-diagram panels",
        rate,
        "",
        "-- per-frame posture (support-gated discriminator g; blind to rate) --",
        f"g : learned {lg.mean():.2f}+/-{lg.std():.2f}   expert {eg.mean():.2f}+/-{eg.std():.2f}   ratio {lg.mean()/eg.mean():.2f}",
        f"raw clamped g : learned {lg_raw.mean():.2f}   expert {eg_raw.mean():.2f}",
        f"support mean  : learned {lsup.mean():.3f}   expert {esup.mean():.3f}",
        f"OOD state/value: learned {lood.any(1).mean():.1%}/{lood.mean():.1%}   expert {eood.any(1).mean():.1%}/{eood.mean():.1%}",
        "",
        "-- coordination, rate-invariant & expert-referenced --",
        f"phase-offset distance to expert : {coord_distance(lph, eph):.3f}   (0 = same inter-leg timing as expert, 0.5 = opposite)",
        "duty = fraction of a stride in stance ; phase = stride-fraction lag vs FL",
        "          " + " ".join(f"{l:>5}" for l in LEGS),
        row("dutyL   : ", ld),
        row("dutyE   : ", ed),
        row("phaseL  : ", lph),
        row("phaseE  : ", eph),
    ]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out, "metrics.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nsaved -> {out}/  (gait_diagram.png, gait_cycle.png, g_hist.png, metrics.txt)")


if __name__ == "__main__":
    main()
