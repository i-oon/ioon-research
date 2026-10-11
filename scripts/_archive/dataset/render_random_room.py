"""Random-room (rr) re-render of ALL counterfactual-walk splits: data/counterfactual_walks/rr_{c10,b1}_{clips,branches}_
{train,val,heldout} and rr_c08_{clips,branches}_heldout. Physics unchanged: kinematic replay of each file's stored state
through render_shift_heldout's rs path (visible floor held at the source level, wall skirts, far clip >= 2 S).

Room per room seed (train 0-47, val 100-123, heldout 200-223; shared by c10 / B1 / c08, and every branch uses its
source clip's room):
  size S   log-uniform 8.0 .. 26.47 m (render_shift_heldout.size_range), stratified per split (n seeds = n equal
           log-bins, each once), seed SIZE_SEED + split index;
  offset o the clip start (default room centre: hexapod (0, 0), B1 the source clip's frame-0 base) relative to the
           room centre, heading kept. Drawn uniform (seed OFFSET_SEED, room seed) over the offsets for which every
           camera position of every body's clip AND all its branches stays >= margin_b = MARGIN_K (2.0) x (body's max
           camera height) inside each wall (the floor-visibility rule of the rs minimum size). Per axis; if an axis has no
           valid offset, its midpoint (the shrink toward the centre that violates least) is used and rr_offset_shrunk
           is set.
Fields: all source fields bit-for-bit except frames and room_size (= S); new rr_room_size, rr_room_offset,
rr_room_size_original, rr_size_seed, rr_offset_seed, rr_offset_shrunk, rr_margin, rr_source, rr_render.

    S=scripts/dataset/render_random_room.py
    $PY $S place                     # sizes + offsets -> data/counterfactual_walks/rr_rooms.json
    $PY $S gate_a --ports ...        # zero offset, original room through this path == stored frames
    $PY $S review --ports ...        # results/check/random_room/ sheets + videos
    $PY $S render --ports ... --stage c10_trainval|b1_trainval|heldout
    $PY $S verify [--stage ...]
(instances: render_shift_heldout.py launch / stop)
"""
import argparse
import glob
import json
import os
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
import render_shift_heldout as RS  # noqa: E402

CW = RS.CW
CHK = os.path.join(ROOT, "results/check/random_room")
ROOMS = os.path.join(CW, "rr_rooms.json")
SIZE_SEED = 20261006
OFFSET_SEED = 20261007
# Margin = MARGIN_K x the body's max camera height. 1.05 (the bare floor-visibility rule) left the B1 at 8 m pushed to
# 1.0 m from a wall: the last frames were nearly all wall (review 2026-10-05). 2.0 keeps the wall at >= 2 camera heights,
# i.e. the floor in at least the bottom ~30 % of the image (FOV 90, level camera).
MARGIN_K = 2.0
SPLITS = {"train": list(range(48)), "val": list(range(100, 124)), "heldout": list(range(200, 224))}
BODY_SPLITS = {"c10": ("train", "val", "heldout"), "b1": ("train", "val", "heldout"), "c08": ("heldout",)}
STAGES = {"c10_trainval": [("c10", "train"), ("c10", "val")], "b1_trainval": [("b1", "train"), ("b1", "val")],
          "heldout": [("c10", "heldout"), ("b1", "heldout"), ("c08", "heldout")]}
RR_RENDER = ("random_room: kinematic replay of the stored state (render_shift_heldout rs path: visible floor held at "
             "the source level, wall skirts, far clip >= 2S), room size rr_room_size from one shared range, room "
             "placed so the clip start sits at rr_room_offset from the room centre, heading kept; room seed + camera + "
             "FOV 90 as the source")


def sdir(body, kind, split):
    return os.path.join(CW, f"{body}_{kind}_{split}")


def odir(body, kind, split):
    return os.path.join(CW, f"rr_{body}_{kind}_{split}")


def sizes():
    lo, hi = RS.size_range()
    out = {}
    for si, (split, seeds) in enumerate(SPLITS.items()):
        n = len(seeds)
        rng = np.random.default_rng(SIZE_SEED + si)
        perm, u = rng.permutation(n), rng.random(n)
        s = np.exp(np.log(lo) + (perm + u) / n * (np.log(hi) - np.log(lo)))
        out.update({sd: float(s[k]) for k, sd in enumerate(seeds)})
    return out


def do_place(a):
    """Camera xy relative to the default centre, per (room seed, body), over each clip + its branches."""
    S = sizes()
    pts, zmax = {}, {}
    for body, splits in BODY_SPLITS.items():
        for split in splits:
            cen = {}
            for kind in ("clips", "branches"):
                for p in sorted(glob.glob(os.path.join(sdir(body, kind, split), "*.npz"))):
                    with np.load(p, allow_pickle=True) as f:
                        seed = int(f["room_seed"]); cp = np.asarray(f["cam_pose"], float)
                        if body == "b1":
                            src = str(f["cf_source_path"]) if "cf_source_path" in f.files else os.path.relpath(p, ROOT)
                            if src not in cen:
                                with np.load(os.path.join(ROOT, src)) as g:
                                    cen[src] = np.asarray(g["base_pos"][0, :2], float)
                            c = cen[src]
                        else:
                            c = np.zeros(2)
                    xy = cp[:, :2] - c
                    k = (seed, body)
                    lo_, hi_ = pts.get(k, (np.full(2, np.inf), np.full(2, -np.inf)))
                    pts[k] = (np.minimum(lo_, xy.min(0)), np.maximum(hi_, xy.max(0)))
                    zmax[body] = max(zmax.get(body, 0.0), float(cp[:, 2].max()))
            print(f"{body} {split}: scanned", flush=True)
    margin = {b: MARGIN_K * z for b, z in zmax.items()}
    rooms = {}
    for seed, sz in S.items():
        h = sz / 2
        lo, hi = np.full(2, -np.inf), np.full(2, np.inf)
        bodies = [b for b in BODY_SPLITS if (seed, b) in pts]
        for b in bodies:
            pmin, pmax = pts[(seed, b)]
            lo = np.maximum(lo, -(h - margin[b]) - pmin)
            hi = np.minimum(hi, (h - margin[b]) - pmax)
        rng = np.random.default_rng([OFFSET_SEED, seed])
        u = rng.random(2)
        shrunk = [bool(lo[i] > hi[i]) for i in range(2)]
        o = np.where(lo <= hi, lo + u * (hi - lo), (lo + hi) / 2)
        clear = min(float(min(h - np.abs(pts[(seed, b)][0] + o).max(), h - np.abs(pts[(seed, b)][1] + o).max())
                          - margin[b]) for b in bodies)
        rooms[str(seed)] = dict(size=sz, offset=[float(o[0]), float(o[1])], valid_lo=lo.tolist(), valid_hi=hi.tolist(),
                                shrunk=shrunk, bodies=bodies, min_clearance_beyond_margin=clear)
    json.dump(dict(size_range=list(RS.size_range()), size_seed=SIZE_SEED, offset_seed=OFFSET_SEED, margin=margin,
                   rule="size stratified log-uniform per split; offset uniform over the valid box (all bodies, clip + "
                        "branches, camera >= margin inside every wall), midpoint if an axis has none",
                   rooms=rooms), open(ROOMS + ".tmp", "w"), indent=1)
    os.replace(ROOMS + ".tmp", ROOMS)
    ns = sum(any(r["shrunk"]) for r in rooms.values())
    cl = [r["min_clearance_beyond_margin"] for r in rooms.values()]
    print(f"margin {margin}; {len(rooms)} rooms, {ns} shrunk; clearance beyond margin min {min(cl):.3f} m; "
          f"|offset| max {max(np.abs(r['offset']).max() for r in rooms.values()):.2f} m")


def rooms():
    return json.load(open(ROOMS))


def jobs(stage):
    J = []
    for body, split in STAGES[stage]:
        for kind in ("clips", "branches"):
            for p in sorted(glob.glob(os.path.join(sdir(body, kind, split), "*.npz"))):
                J.append((body, p, os.path.join(odir(body, kind, split), os.path.basename(p))))
    return J


def write_rr(src, dst, d, fr, cp, room, R, margin):
    if not np.array_equal(cp, d["cam_pose"]):
        raise RuntimeError(f"cam_pose changed: {np.abs(cp - d['cam_pose']).max():.2e}")
    if fr.shape != d["frames"].shape:
        raise RuntimeError(f"frames shape {fr.shape} != {d['frames'].shape}")
    out = {k: v for k, v in d.items() if k != "frames"}
    out.update(frames=fr, room_size=np.float64(room["size"]), rr_room_size=np.float64(room["size"]),
               rr_room_offset=np.asarray(room["offset"], np.float64),
               rr_room_size_original=np.float64(d["room_size"]) if "room_size" in d else np.float64(np.nan),
               rr_size_seed=np.int64(SIZE_SEED), rr_offset_seed=np.int64(OFFSET_SEED),
               rr_offset_shrunk=np.asarray(room["shrunk"], bool), rr_margin=np.float64(margin),
               rr_room_height=np.float64(R["height"]), rr_ground_uv=np.float64(R["ground_uv"]),
               rr_source=np.array(os.path.relpath(src, ROOT)), rr_render=np.array(RR_RENDER))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + f".tmp{os.getpid()}_{threading.get_ident()}.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, dst)


def do_render(a):
    RM = rooms()
    J = [j for st in a.stage for j in jobs(st) if not os.path.exists(j[2])]
    print(f"{len(J)} files to render ({a.stage})", flush=True)

    def fn(sim, j):
        body, src, dst = j
        with np.load(src) as f:
            seed = int(f["room_seed"])
        room = RM["rooms"][str(seed)]
        d, fr, cp, R = RS.render_file(sim, body, src, RS.rs_room(room["size"]), room["offset"])
        write_rr(src, dst, d, fr, cp, room, R, RM["margin"][body])
        return f"S {room['size']:.2f} o {room['offset'][0]:+.2f} {room['offset'][1]:+.2f}"
    RS.run_pool(J, a.ports, fn, "rr")


def do_gate_a(a):
    J = []
    for body, splits in BODY_SPLITS.items():
        for split in splits:
            for kind in ("clips", "branches"):
                fs = sorted(glob.glob(os.path.join(sdir(body, kind, split), "*.npz")))
                J += [(body, fs[0], kind, split), (body, fs[len(fs) // 2], kind, split)]
    res = []

    def fn(sim, j):
        body, src, kind, split = j
        out = {}
        for name, ov in (("orig_path_zero_offset", None), ("rs_path_orig_size_zero_offset", RS.src_room(body))):
            d, fr, cp, R = RS.render_file(sim, body, src, ov, (0.0, 0.0))
            diff = np.abs(fr.astype(int) - d["frames"].astype(int))
            out[name] = dict(mae=float(diff.mean()), max=int(diff.max()), frac_px_diff=float((diff.max(-1) > 0).mean()),
                             corr=float(np.corrcoef(fr.ravel().astype(float), d["frames"].ravel().astype(float))[0, 1]),
                             cam=float(np.abs(cp - d["cam_pose"]).max()))
        res.append(dict(body=body, split=split, kind=kind, file=os.path.relpath(src, ROOT), **out))
        o, r = out["orig_path_zero_offset"], out["rs_path_orig_size_zero_offset"]
        return f"orig mae {o['mae']:.4f} max {o['max']} | rs-path mae {r['mae']:.3f} frac {r['frac_px_diff']:.4f}"
    RS.run_pool(J, a.ports, fn, "gate_a")
    os.makedirs(CHK, exist_ok=True)
    json.dump(res, open(os.path.join(CHK, "gate_a.json"), "w"), indent=1)
    for r in res:
        o, s = r["orig_path_zero_offset"], r["rs_path_orig_size_zero_offset"]
        print(f"{r['body']:4s} {r['split']:7s} {r['kind']:8s} {os.path.basename(r['file']):24s} orig: mae {o['mae']:.4f} "
              f"max {o['max']} corr {o['corr']:.6f} cam {o['cam']:.0e} | rs path: mae {s['mae']:.3f} "
              f"px {s['frac_px_diff']:.4f} corr {s['corr']:.4f}")


def do_review(a):
    """Per body: the train clips whose room is smallest / median / largest, with their actual random placement."""
    from PIL import Image, ImageDraw
    import imageio
    RM = rooms()["rooms"]
    J = []
    for body in BODY_SPLITS:
        split = "heldout" if body == "c08" else "train"
        fs = sorted(glob.glob(os.path.join(sdir(body, "clips", split), "*.npz")))
        seeds = {p: int(np.load(p)["room_seed"]) for p in fs}
        order = sorted(fs, key=lambda p: RM[str(seeds[p])]["size"])
        for name, p in (("small", order[0]), ("mid", order[len(order) // 2]), ("large", order[-1])):
            J.append((body, p, name))
    out = {}

    def fn(sim, j):
        body, src, name = j
        room = RM[str(int(np.load(src)["room_seed"]))]
        d, fr, cp, R = RS.render_file(sim, body, src, RS.rs_room(room["size"]), room["offset"])
        out[j] = (fr, d["frames"], room)
        return name
    RS.run_pool(J, a.ports, fn, "review")
    os.makedirs(CHK, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "sim/scene"))
    from ego_camera import ego_view_profile
    rep = []
    for body in BODY_SPLITS:
        rows, vids = [], []
        for j in [j for j in J if j[0] == body]:
            fr, ref, room = out[j]
            for lab, F in (("stored", ref), (f"{j[2]} {room['size']:.1f} m off {room['offset'][0]:+.1f},"
                                             f"{room['offset'][1]:+.1f}", fr)):
                row = Image.new("RGB", (192 * 4 + 190, 192))
                for k, t in enumerate([0, 20, 40, len(F) - 1]):
                    row.paste(Image.fromarray(F[t]).resize((192, 192)), (190 + 192 * k, 0))
                dr = ImageDraw.Draw(row)
                dr.text((4, 60), os.path.basename(j[1]), fill=(255, 255, 255)); dr.text((4, 80), lab, fill=(255, 255, 0))
                rows.append(row)
            vids.append(fr)
            m = np.mean([ego_view_profile(f) for f in ref], 0)
            rep.append(dict(body=body, file=os.path.basename(j[1]), size=room["size"], offset=room["offset"],
                            rowprof_min=float(min(np.corrcoef(ego_view_profile(f), m)[0, 1] for f in fr)),
                            min_frame_std=float(fr.reshape(len(fr), -1).std(1).min())))
        sheet = Image.new("RGB", (rows[0].width, 192 * len(rows)))
        for k, r in enumerate(rows):
            sheet.paste(r, (0, 192 * k))
        sheet.save(os.path.join(CHK, f"sheet_{body}.png"))
        with imageio.get_writer(os.path.join(CHK, f"video_{body}.mp4"), fps=10) as w:
            for t in range(min(len(v) for v in vids)):
                w.append_data(np.concatenate([v[t] for v in vids], 1))
    json.dump(rep, open(os.path.join(CHK, "review.json"), "w"), indent=1)
    for r in rep:
        print(r)


def do_verify(a):
    RM = rooms()
    NEW = {"rr_room_size", "rr_room_offset", "rr_room_size_original", "rr_size_seed", "rr_offset_seed",
           "rr_offset_shrunk", "rr_margin", "rr_room_height", "rr_ground_uv", "rr_source", "rr_render"}
    bad = []
    for st in a.stage:
        n = {}
        for body, src, dst in jobs(st):
            key = os.path.basename(os.path.dirname(dst)); n[key] = n.get(key, 0) + 1
            if not os.path.exists(dst):
                bad.append((dst, "missing")); continue
            s, r = RS.load(src), RS.load(dst)
            ks = set(s) - {"frames", "room_size"}
            if set(r) - NEW - {"frames", "room_size"} != ks:
                bad.append((dst, "keys"))
            for k in ks:
                if s[k].dtype != r[k].dtype or s[k].shape != r[k].shape or s[k].tobytes() != r[k].tobytes():
                    bad.append((dst, f"field {k}"))
            room = RM["rooms"][str(int(s["room_seed"]))]
            if float(r["room_size"]) != room["size"] or list(r["rr_room_offset"]) != room["offset"]:
                bad.append((dst, "room"))
            if r["frames"].shape != s["frames"].shape:
                bad.append((dst, "frames"))
        for key, c in n.items():
            d = os.path.join(CW, key)
            print(f"{key}: {len(glob.glob(os.path.join(d, '*.npz')))} files (expected {c}), "
                  f"tmp {len(glob.glob(os.path.join(d, '*.tmp*')))}", flush=True)
    print(f"{len(bad)} problems"); [print(b) for b in bad[:30]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("place", "gate_a", "review", "render", "verify"))
    ap.add_argument("--ports", type=int, nargs="+", default=[25300])
    ap.add_argument("--stage", nargs="+", default=list(STAGES))
    a = ap.parse_args()
    dict(place=do_place, gate_a=do_gate_a, review=do_review, render=do_render, verify=do_verify)[a.cmd](a)


if __name__ == "__main__":
    main()
