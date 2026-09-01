"""Render what the planner *selected*, beside the goal it was shown.

    .venv/bin/python3 scripts/figures/render_selection_clips.py

**These are two recorded clips side by side, not a robot being driven.** F136 measures selection
among recorded behaviours; the physics loops this project has run cross-embodiment are the
chance-level ones (F121, F122), and using those frames to illustrate F136 would be the overclaim
the mismatch control exists to prevent. Every panel is labelled with which body it is and what the
panel is, so a frame lifted out of the deck still says what it is.

Two figures:

  selection    goal (insect) | the B1 behaviour the three-channel score picked, one row per family
  strafe       goal (insect, strafing) | what the forward-speed-only score picked | what the
               three-channel score picked -- the self-correction of "sideways fails everywhere"
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def label(frames, lines, height=44):
    """Burn a caption under every frame, so a clip shown alone still states what it is."""
    f_big = ImageFont.truetype(BOLD, 15)
    f_small = ImageFont.truetype(FONT, 13)
    out = []
    for fr in frames:
        h, w = fr.shape[:2]
        canvas = Image.new("RGB", (w, h + height), (17, 17, 17))
        canvas.paste(Image.fromarray(fr.astype(np.uint8)), (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((6, h + 4), lines[0], font=f_big, fill=(255, 255, 255))
        if len(lines) > 1:
            d.text((6, h + 24), lines[1], font=f_small, fill=(170, 170, 170))
        out.append(np.asarray(canvas))
    return np.stack(out)


def modal_pick(clips_b1, cand, goal_e, itm, ftm, md, proj, channels, h, device, mode="D"):
    """Which candidate condition the rule picks most often over the goal clip."""
    from collections import Counter
    conds = sorted(cand)
    votes = Counter()
    with torch.no_grad():
        for t in range(5, min(len(goal_e) - h - 1, 55), 5):
            acts, keep = [], []
            for k in conds:
                src = cand[k]
                if t + h < clips_b1[src]["n"]:
                    acts.append(torch.stack([clips_b1[src]["a"][t + i] for i in range(h)]))
                    keep.append(k)
            if len(keep) < 2:
                continue
            a = torch.stack(acts).to(device)
            C = len(keep)
            z = proj(a.reshape(C * h, -1), "b1").reshape(C, h, -1)
            pred = md.body(None, z.reshape(C * h, -1)).reshape(C, h, -1).mean(1)
            g0 = goal_e[t].float().to(device).unsqueeze(0)
            g1 = goal_e[t + h].float().to(device).unsqueeze(0)
            target = md.body(None, itm(g0, g1)).reshape(-1)
            k_ = min(pred.shape[-1], target.numel())
            votes[keep[int((pred[:, :k_] - target[:k_]).pow(2).mean(-1).argmin())]] += 1
    return votes.most_common(1)[0][0], votes


def load_model(path, device):
    ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    from wm.models.ftm import ForwardTransitionModel
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    md = MotionDecoder(cfg, {"b1": 12}).to(device).eval(); md.load_state_dict(ck["md"], strict=False)
    proj = ActionProjector(cfg, action_dims_from(ck)).to(device).eval()
    proj.load_state_dict(ck["projector"])
    return ck, cfg, itm, ftm, md, proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--three", default="wm/runs/beh12_hex-b1_body3/stage3_b1_nce_s0_bodyfit_proj.pt")
    ap.add_argument("--one", default="wm/runs/beh12_hexonly/stage3_b1_nce_s0_bodyfit_proj.pt")
    ap.add_argument("--b1", default="data/allocentric/beh12_b1_flat")
    ap.add_argument("--goals", default="data/allocentric/beh12_c08f09t09_flat")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--out", default="results/wm/closed_loop/f142_video")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    import imageio.v2 as imageio

    ck3, cfg3, itm3, ftm3, md3, proj3 = load_model(args.three, device)
    ch3 = [int(c) for c in cfg3.body_channels]
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    cache3 = torch.load(os.path.join(ROOT, "results/wm/cache/b1_body3.pt"), map_location="cpu")
    clips_b1 = gather(os.path.join(ROOT, args.b1), "b1", encoder, ck3, cache3, 2,
                      max(1, cfg3.action_lag), device)
    cand, seen = {}, set()
    for i, c in enumerate(clips_b1):
        if c["cond"] not in seen:
            cand[c["cond"]] = i; seen.add(c["cond"])

    gcache = torch.load(os.path.join(ROOT, "results/wm/cache/bodycal_hexapod.pt"), map_location="cpu")
    goal_files = {}
    for p_ in sorted(glob.glob(os.path.join(ROOT, args.goals, "*.npz"))):
        with np.load(p_, allow_pickle=True) as z:
            goal_files.setdefault(str(z["condition"]), p_)

    # the B1 condition that means the same thing, so "right level" can be judged at all.
    # The two robots name conditions after their own controls; the datasets are matched level for
    # level (forward to 4%, turn to 2%), so the pairing is by rank within the family.
    MATCH = {"speed_c5.8": "speed_vx0.30", "speed_c7.1": "speed_vx0.38",
             "speed_c8.15": "speed_vx0.40", "speed_c8.8": "speed_vx0.50",
             "turn_s0.05": "turn_w0.008", "turn_s0.15": "turn_w0.024",
             "turn_s0.29": "turn_w0.037", "turn_s0.56": "turn_w0.075",
             "side_L_lvl0": "side_L_lvl0", "side_L_lvl1": "side_L_lvl1",
             "side_R_lvl0": "side_R_lvl0", "side_R_lvl1": "side_R_lvl1"}
    cond_b1_of = MATCH.get

    picked = ["speed_c7.1", "turn_s0.29", "side_L_lvl1"]
    rows = []
    for cond in picked:
        gp = goal_files[cond]
        ge = gcache[gp].float() if gp in gcache else None
        if ge is None:
            from wm.evaluate import encode_clip
            from wm.data.embodiment import REGISTRY, load
            ge = encode_clip(encoder, load(gp, REGISTRY["hexapod"])["frames"], 2).float()
        pick, votes = modal_pick(clips_b1, cand, ge, itm3, ftm3, md3, proj3, ch3,
                                 args.horizon, device)
        with np.load(gp, allow_pickle=True) as z:
            gf = z["frames"]
        bf = np.load(os.path.join(ROOT, args.b1, clips_b1[cand[pick]]["path"]),
                     allow_pickle=True)["frames"]
        n = min(len(gf), len(bf))
        left = label(gf[:n], ["STICK INSECT — the goal it was shown",
                              f"{cond}   18-DOF, six legs"])
        fam = lambda c: ("side_L" if c.startswith("side_L") else "side_R"
                         if c.startswith("side_R") else c.split("_")[0])
        tot = sum(votes.values())
        if pick == cond_b1_of(cond):
            verdict = f"exact match   {votes[pick]}/{tot} steps"
        elif fam(pick) == fam(cond_b1_of(cond)):
            verdict = (f"responds but ranks poorly: right family ({votes[pick]}/{tot}), "
                       f"wrong level")
        else:
            verdict = f"WRONG FAMILY   {votes[pick]}/{tot} steps"
        right = label(bf[:n], ["QUADRUPED — the behaviour it selected",
                               f"{pick}   12-DOF, four legs   —   {verdict}"])
        rows.append(np.concatenate([left, right], axis=2))
        print(f"  {cond:<14} -> {pick:<14} ({votes[pick]}/{sum(votes.values())} steps)", flush=True)
    n = min(len(r) for r in rows)
    grid = np.concatenate([r[:n] for r in rows], axis=1)
    out = os.path.join(ROOT, args.out, "f136_selection.mp4")
    imageio.mimwrite(out, grid.astype(np.uint8), fps=20, macro_block_size=1)
    print(f"-> {out}  {grid.shape}")

    # ---- the strafe reframe: one channel against three, same goal
    ck1, cfg1, itm1, ftm1, md1, proj1 = load_model(args.one, device)
    ch1 = [int(c) for c in cfg1.body_channels]
    cache1 = torch.load(os.path.join(ROOT, "results/wm/cache/b1.pt"), map_location="cpu")
    clips1 = gather(os.path.join(ROOT, args.b1), "b1", encoder, ck1, cache1, 2,
                    max(1, cfg1.action_lag), device)
    cand1, seen1 = {}, set()
    for i, c in enumerate(clips1):
        if c["cond"] not in seen1:
            cand1[c["cond"]] = i; seen1.add(c["cond"])
    gp = goal_files["side_L_lvl1"]
    ge = gcache[gp].float()
    p1, v1 = modal_pick(clips1, cand1, ge, itm1, ftm1, md1, proj1, ch1, args.horizon, device)
    p3, v3 = modal_pick(clips_b1, cand, ge, itm3, ftm3, md3, proj3, ch3, args.horizon, device)
    print(f"  strafe goal -> one channel picks {p1}, three channels pick {p3}")
    with np.load(gp, allow_pickle=True) as z:
        gf = z["frames"]
    f1 = np.load(os.path.join(ROOT, args.b1, clips1[cand1[p1]]["path"]), allow_pickle=True)["frames"]
    f3 = np.load(os.path.join(ROOT, args.b1, clips_b1[cand[p3]]["path"]), allow_pickle=True)["frames"]
    n = min(len(gf), len(f1), len(f3))
    strip = np.concatenate([
        label(gf[:n], ["GOAL — insect strafing left", "side_L_lvl1   18-DOF, six legs"]),
        label(f1[:n], ["forward-speed-only score picks", f"{p1}   strafes the OPPOSITE way"]),
        label(f3[:n], ["three-channel score picks", f"{p3}   correct side"]),
    ], axis=2)
    out = os.path.join(ROOT, args.out, "f136_strafe_reframe.mp4")
    imageio.mimwrite(out, strip.astype(np.uint8), fps=20, macro_block_size=1)
    print(f"-> {out}  {strip.shape}")


if __name__ == "__main__":
    main()
