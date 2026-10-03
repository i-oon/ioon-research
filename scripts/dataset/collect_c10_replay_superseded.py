"""SUPERSEDED (2026-10-02), not a current data script: it writes into data/counterfactual_walks/_superseded/ BY DESIGN.
Shared constants / rules (ORDER, ROLES, schedule, ...) now live in beh24_conditions.py (imported here).
Its output moved to data/counterfactual_walks/_superseded/c10_replay_noise/; the official hexapod set is
made by scripts/dataset/collect_c10_walks_and_branches.py (deterministic scene reuse, F305). Kept for the record.

DATA_PLAN v2 stage 1: hexapod (c10f10t10) main clips -> data/counterfactual_walks/_superseded/c10_replay_noise/hex_main_{train,val,heldout}.

One long physics walk per beh24 condition (`collect_ik.py --record_state --no_frames --frames N`, a
constant `--plan` = the exact beh24 command of `collect_switch_hex.COND`, centre pose fitted from the
canonical speed_c7.1 clip, no expert CSV), cut into 4 non-overlapping 66-frame windows, each rendered
afterwards in its own room with `sim/render/render_hex_replay.py` (recorded link poses, re-centred so the
window's first head position is the room centre).

Condition order / index (DATA_PLAN section 0 item 3): fwd speed 1-4, bwd 1-4, turn left 1-4,
turn right 1-4, side L 1-4, side R 1-4.

Windows: start = W0 + w * (66 + gap), w = 0..3, W0 = 10 frames (one gait cycle after the start), gap per
condition chosen from 8..30 frames to maximise the smallest circular distance between the 4 start gait
phases (so copies start at different points of the cycle as well as at different body poses).

Split rule (fixed): window w of condition index i -> role ROLES[(w + i) % 4], ROLES = (train k=0,
train k=1, val, heldout); rotating with i so no split always gets the earliest/latest part of a walk.
Room seed: train 2*i + k, val 100 + i, heldout 200 + i.

    .venv/bin/python3 scripts/dataset/collect_c10_replay_superseded.py walks
    .venv/bin/python3 scripts/dataset/collect_c10_replay_superseded.py cut
    .venv/bin/python3 scripts/dataset/collect_c10_replay_superseded.py check
    .venv/bin/python3 scripts/dataset/collect_c10_replay_superseded.py video
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "dataset"))
sys.path.insert(0, os.path.join(ROOT, "sim", "render"))
from collect_switch_hex import COND, KEYS, centre_pose, COMMON, CENTRE  # noqa: E402

from beh24_conditions import (EP, W0, GAPS, BASE_CYC, ROLES, LIVE_ROOM, ORDER, FAMILY, seed_of, split_of,  # noqa: E402,F401
                              cycles_per_frame, schedule, tilt_deg)
OUT = "data/counterfactual_walks/_superseded/c10_replay_noise"   # moved there 2026-10-02 (F305); writes there by design
WALKS = os.path.join(OUT, "hex_main_walks")
OLD = "data/egocentric/beh24_c10f10t10_ego_flat"


def walk_path(c):
    return os.path.join(ROOT, WALKS, f"{c}.npz")


def do_walks(args):
    os.makedirs(os.path.join(ROOT, WALKS), exist_ok=True)
    work = os.path.join(ROOT, WALKS, "_work")
    cen = centre_pose(work)
    for i, c in enumerate(ORDER):
        dst = walk_path(c)
        if os.path.exists(dst):
            continue
        g, st, ph, N = schedule(c)
        plan = {k: [float(COND[c][1][k])] * N for k in KEYS}
        wd = os.path.join(work, c); os.makedirs(wd, exist_ok=True)
        pf = os.path.join(wd, "plan.json"); json.dump(plan, open(pf, "w"))
        t0 = time.time()
        cmd = [sys.executable, os.path.join(ROOT, "sim/collect/collect_ik.py"), "--port", str(args.port),
               "--morphs", "c10f10t10=medauroidea_c10f10t10.ttt", "--episodes", str(40000 + 10 * i),
               "--ego_seed", "0", "--plan", pf, "--out", wd, "--centre_from", cen,
               "--record_state", "--no_frames", "--frames", str(N)] + COMMON + ["--ego_box", str(LIVE_ROOM)]
        with open(os.path.join(wd, "collect.log"), "w") as log:
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        src = glob.glob(os.path.join(wd, "c10f10t10_*.npz"))[0]
        rec = dict(np.load(src, allow_pickle=True))
        assert len(rec["actions"]) == N, (c, len(rec["actions"]), N)
        rec.update(condition=c, cond_index=i, family=FAMILY[i], family_level=i % 4, walk_gap=g,
                   window_starts=np.array(st), centre_from=CENTRE)
        np.savez_compressed(dst, **rec)
        print(f"walk {i:2d} {c:<16} N={N} gap={g} phases={np.round(ph, 2)} {time.time() - t0:.0f}s", flush=True)


def walk_health(rec):
    h = rec["head"].astype(float)
    tilt = tilt_deg(rec["state_abdomen_quat"])
    z = rec["state_abdomen_pos"][:, 2]
    return dict(head_z_min=float(h[:, 2].min()), head_z_med=float(np.median(h[:, 2])),
                abd_z_min=float(z.min()), abd_z_med=float(np.median(z)), tilt_max=float(tilt.max()),
                fell=bool(h[:, 2].min() < 0.5 * np.median(h[:, 2]) or tilt.max() > 45))


def old_tags():
    tags = {}
    for p in sorted(glob.glob(os.path.join(ROOT, OLD, "*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            tags[str(d["condition"])] = (str(d["behaviour"]), int(d["level"]))
    return tags


def do_cut(args):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from render_hex_replay import render, ROOM
    sim = RemoteAPIClient("localhost", port=args.port).require("sim")
    tags = old_tags()
    for s in ("train", "val", "heldout"):
        os.makedirs(os.path.join(ROOT, OUT, f"hex_main_{s}"), exist_ok=True)
    per_frame = ("actions", "forces", "head", "body_quat", "step_idx", "state_abdomen_pos", "state_abdomen_quat",
                 "state_joint_pos", "state_link_pose", "state_sim_time", "cpg_phase", "cpg_cycles_total", "com_pos") + \
        tuple(f"plan_{k}" for k in KEYS)
    t0 = time.time()
    for i, c in enumerate(ORDER):
        rec = dict(np.load(walk_path(c), allow_pickle=True))
        if "com_pos" not in rec:                   # walks recorded before 2026-10-01 (F301)
            from wm.data.com import hex_com
            rec["com_pos"] = hex_com(rec["state_link_names"], rec["state_link_pose"])
        for w, st in enumerate(rec["window_starts"]):
            st = int(st)
            role = ROLES[(w + i) % 4]
            split, seed = split_of(role), seed_of(i, role)
            ep = 40000 + 10 * i + w
            dst = os.path.join(ROOT, OUT, f"hex_main_{split}", f"hexapod_ep{ep}.npz")
            if os.path.exists(dst):
                continue
            frames, off, R = render(sim, rec, st, EP, seed, recentre=True, room=ROOM)
            out = {k: rec[k][st:st + EP] for k in per_frame}
            out["head"] = out["head"].astype(np.float64); out["head"][:, :2] += off
            out["head"] = out["head"].astype(np.float32)
            out["state_abdomen_pos"][:, :2] += off
            out["state_link_pose"][:, :, :2] += off
            out["com_pos"][:, :2] += off           # CoM (F301): Froude reference point of the loader
            beh, lvl = tags[c]
            out.update(frames=frames, foot_order=rec["foot_order"], morph=rec["morph"], expert_episode=ep,
                       repeat=w, scale=rec["scale"], behavior="walk", schedule="", gait="cpg",
                       state_joint_names=rec["state_joint_names"], state_link_names=rec["state_link_names"],
                       condition=c, behaviour=beh, level=lvl, family=FAMILY[i], family_level=i % 4,
                       cond_index=i, embodiment="hexapod", room_seed=seed, ego_seed=seed,
                       room_size=float(R["size"]), window_start=st, window_index=w, copy=role,
                       split=split, source_walk=os.path.relpath(walk_path(c), ROOT), offset_xy=off,
                       dt=0.05, centre_from=CENTRE,
                       render="render_hex_replay links mode, re-centred, fov 90, room 8 m")
            np.savez_compressed(dst, **out)
            print(f"{c:<16} w{w} start {st:3d} -> {split:<7} seed {seed:3d} ep{ep}  {time.time() - t0:.0f}s", flush=True)


def do_check(args):
    import wm.data.embodiment as E
    files = {s: sorted(glob.glob(os.path.join(ROOT, OUT, f"hex_main_{s}", "*.npz"))) for s in ("train", "val", "heldout")}
    print("counts", {s: len(v) for s, v in files.items()})
    ok = True
    # (i) loader + finite labels; per-clip mean Froude
    rows = []
    for s, ps in files.items():
        for p in ps:
            clip = E.load(p, E.HEXAPOD)
            bm = clip["body_motion"]
            fin = np.isfinite(bm).all() and np.isfinite(clip["actions"]).all()
            ok &= bool(fin) and clip["frames"].shape == (EP, 256, 256, 3)
            with np.load(p, allow_pickle=True) as d:
                rows.append(dict(path=p, split=s, cond=str(d["condition"]), i=int(d["cond_index"]),
                                 seed=int(d["room_seed"]), copy=str(d["copy"]), start=int(d["window_start"]),
                                 phase=float(d["cpg_phase"][0]), act=d["actions"].astype(float),
                                 head=d["head"].astype(float), off=d["offset_xy"].astype(float),
                                 quat=d["body_quat"].astype(float), mean=bm.mean(0), fin=fin,
                                 abdz=d["state_abdomen_pos"][:, 2].astype(float), tilt=tilt_deg(d["state_abdomen_quat"])))
    print(f"(i) loader ok on {len(rows)} files, all labels finite: {all(r['fin'] for r in rows)}")
    # (ii) labels vs old beh24 clips + signs
    old = {}
    for p in sorted(glob.glob(os.path.join(ROOT, OLD, "*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            cc = str(d["condition"])
        old.setdefault(cc, []).append(E.load(p, E.HEXAPOD)["body_motion"].mean(0))
    print("\n(ii) clip-mean Froude [fwd, lat, yaw]: old beh24 mean +- sd (4 clips) | new 4 windows | dominant-channel check")
    sign_ch = {"fwd": (0, 1), "bwd": (0, -1), "turn_left": (2, 1), "turn_right": (2, -1), "side_L": (1, 1), "side_R": (1, -1)}
    lab_bad = []
    for i, c in enumerate(ORDER):
        o = np.array(old[c]); n = np.array([r["mean"] for r in rows if r["cond"] == c])
        ch, sg = sign_ch[FAMILY[i]]
        signs = bool(np.all(np.sign(n[:, ch]) == sg))
        lo, hi = o[:, ch].min(), o[:, ch].max()
        sd = o[:, ch].std()
        dev = abs(n[:, ch].mean() - o[:, ch].mean())
        within = bool(dev <= max(3 * sd, 0.15 * abs(o[:, ch].mean())))
        if not (signs and within):
            lab_bad.append(c)
        print(f"  {c:<16} old {np.round(o.mean(0), 3)} sd {np.round(o.std(0), 3)} | new ch{ch} "
              f"{np.round(n[:, ch], 3)} (old range {lo:.3f}..{hi:.3f}) | sign {'ok' if signs else 'BAD'} "
              f"mean diff {dev:.3f} {'ok' if within else 'OUT'}")
    print(f"  conditions out of tolerance / sign: {lab_bad or 'none'}")
    # (iii) distinctness
    print("\n(iii) distinctness per condition: min over pairs of max|action diff| (rad), start phases, start xy (walk frame), heading")
    dmin_all = 9.0
    for c in ORDER:
        rs = sorted([r for r in rows if r["cond"] == c], key=lambda r: r["start"])
        dm = min(np.abs(a["act"] - b["act"]).max() for k, a in enumerate(rs) for b in rs[k + 1:])
        dmin_all = min(dmin_all, dm)
        hd = [float(np.degrees(E.heading(r["quat"][:1], "hexapod")[0])) for r in rs]
        xy = [tuple(np.round(r["head"][0, :2] - r["off"], 2)) for r in rs]
        print(f"  {c:<16} dAct {dm:.3f} starts {[r['start'] for r in rs]} phase {[round(r['phase'], 2) for r in rs]} "
              f"xy {xy} heading {[round(h) for h in hd]}")
    print(f"  smallest same-condition action difference {dmin_all:.3f} rad (0 would be a duplicate)")
    ok &= dmin_all > 1e-3
    # (iv) seeds
    exp = {}
    for r in rows:
        role = r["copy"]; exp_seed = seed_of(r["i"], role)
        ok &= r["seed"] == exp_seed and split_of(role) == r["split"]
    sets = {s: sorted(r["seed"] for r in rows if r["split"] == s) for s in files}
    print(f"\n(iv) seeds train {sets['train'][:3]}..{sets['train'][-1]} n={len(set(sets['train']))}, "
          f"val {sets['val'][0]}..{sets['val'][-1]} n={len(set(sets['val']))}, heldout {sets['heldout'][0]}.."
          f"{sets['heldout'][-1]} n={len(set(sets['heldout']))}; train==0..47 {sets['train'] == list(range(48))}, "
          f"val==100..123 {sets['val'] == list(range(100, 124))}, heldout==200..223 {sets['heldout'] == list(range(200, 224))}; "
          f"shared across splits: {set(sets['train']) & set(sets['val']) | set(sets['train']) & set(sets['heldout']) | set(sets['val']) & set(sets['heldout']) or 'none'}")
    ok &= sets["train"] == list(range(48)) and sets["val"] == list(range(100, 124)) and sets["heldout"] == list(range(200, 224))
    # (v) falls + room bounds
    from render_hex_replay import ROOM
    walks = {c: dict(np.load(walk_path(c), allow_pickle=True)) for c in ORDER}
    hl = {c: walk_health(w) for c, w in walks.items()}
    fell = [c for c, h in hl.items() if h["fell"]]
    reach = max(np.abs(r["head"][:, :2]).max() for r in rows)
    print(f"\n(v) falls: {fell or 'none'}; walk head z min {min(h['head_z_min'] for h in hl.values()):.3f} m "
          f"(medians {min(h['head_z_med'] for h in hl.values()):.3f}..{max(h['head_z_med'] for h in hl.values()):.3f}), "
          f"max tilt from start {max(h['tilt_max'] for h in hl.values()):.1f} deg; "
          f"window max |x|,|y| from room centre {reach:.2f} m (walls at {ROOM / 2:.1f} m)")
    ok &= not fell and reach < ROOM / 2 - 0.5
    print("\nALL GATES", "PASS" if ok and not lab_bad else "FAIL / see above")


def do_video(args):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    pick = args.video_conds
    rows_out = []
    for c in pick:
        ps = sorted(p for s in ("train", "val", "heldout") for p in glob.glob(os.path.join(ROOT, OUT, f"hex_main_{s}", "*.npz")))
        rs = []
        for p in ps:
            with np.load(p, allow_pickle=True) as d:
                if str(d["condition"]) == c:
                    rs.append((str(d["copy"]), int(d["room_seed"]), int(d["window_start"]), d["frames"]))
        rs.sort(key=lambda r: ROLES.index(r[0]))
        rows_out.append((c, rs))
    writer = imageio.get_writer(os.path.join(ROOT, "results/check/hex_main_samples.mp4"), fps=10)
    for t in range(EP):
        grid = []
        for c, rs in rows_out:
            tiles = []
            for role, seed, st, fr in rs:
                im = Image.fromarray(fr[t]); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, 255, 14], fill=(0, 0, 0))
                dr.text((3, 2), f"{c} {role} seed {seed} start {st} t{t}", fill=(255, 255, 255))
                tiles.append(np.asarray(im))
            grid.append(np.concatenate(tiles, 1))
        writer.append_data(np.concatenate(grid, 0))
    writer.close()
    print("-> results/check/hex_main_samples.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("plan", "walks", "cut", "check", "video"))
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--video_conds", nargs="+", default=["turn_s0.29", "side_R_lvl1", "speed_c7.1"])
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.step == "plan":
        for i, c in enumerate(ORDER):
            g, st, ph, N = schedule(c)
            print(f"{i:2d} {c:<16} cyc/frame {cycles_per_frame(c):.3f} gap {g} starts {st} phases {np.round(ph, 2)} N {N}")
    else:
        {"walks": do_walks, "cut": do_cut, "check": do_check, "video": do_video}[a.step](a)


if __name__ == "__main__":
    main()
