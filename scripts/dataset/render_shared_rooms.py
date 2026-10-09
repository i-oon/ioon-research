"""Shared-room rendering of the TRAINING data (2026-10-09): every training clip and all its branches rendered in the SAME
4 room looks, so no room look belongs to a clip or a behaviour (FINDINGS F324 / F330: z carries the room; each original
training room holds one behaviour). Egocentric VSM's design: a few ground/room textures shared by everything.

  version v (0..3): room appearance seed APPEAR[v] for EVERY clip (wall textures + colours + floor texture,
                    ego_camera.build_texture_box / randomise_ground), room size and start position as the ORIGINAL
                    renders (sized to the body: hexapod 8 m, B1 17.65 m; clip start at the room centre) -> only the look
                    changes; physics, labels and every non-frame field are copied bit-for-bit.
  lighting:         the scene's 4 lights scaled by one gain g ~ log-uniform [0.5, 1.6] and each light on with p 0.75
                    (at least one on), drawn per (version, source clip) -- a clip and its branches share one lighting.
Validation / held-out data keep their original renders (unseen looks at test time).

Output dirs data/counterfactual_walks/sr{v}_{c10,b1}_{clips,branches}_train (same file names); fields: source fields
except frames / cam_pose + sr_version, sr_appear_seed, sr_light_gain, sr_light_on. Writes: tmp + rename; resumable.

    S=scripts/dataset/render_shared_rooms.py
    $PY render_shift_heldout.py launch --ports P..   # own CoppeliaSim instances (never 23000), stop afterwards
    $PY $S gate   --ports P                          # original seed, default lights through this path == stored frames
    $PY $S sheet  --ports P                          # results/check/shared_rooms/sheet.png (show the user first)
    $PY $S render --ports P1 P2 ... [--versions 0 1 2 3] [--bodies c10 b1]
    $PY $S verify
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
import render_shift_heldout as RS  # noqa: E402

CW = os.path.join(ROOT, "data/counterfactual_walks")
CHK = os.path.join(ROOT, "results/check/shared_rooms")
APPEAR = {0: 2000, 1: 2001, 2: 2002, 3: 2003}      # unused by every other data set (0-923, 1000-1147)
BODIES = ("c10", "b1")
KINDS = ("clips", "branches")
LIGHT_SEED = 20261009


def src_dir(body, kind):
    return os.path.join(CW, f"{body}_{kind}_train")


def out_dir(v, body, kind):
    return os.path.join(CW, f"sr{v}_{body}_{kind}_train")


def lighting(v, room_seed):
    rng = np.random.default_rng([LIGHT_SEED, v, int(room_seed)])
    gain = float(np.exp(rng.uniform(np.log(0.5), np.log(1.6))))
    on = rng.random(4) < 0.75
    if not on.any():
        on[rng.integers(4)] = True
    return gain, on


def light_hook(sim, d):
    """RS.SCENE_HOOK: reads the clip's own lighting from d["_light"] (thread-safe: nothing per-clip is global)."""
    if d.get("_light") is None:
        return
    gain, on = d["_light"]
    if True:
        hs = sim.getObjectsInTree(sim.handle_scene, sim.object_light_type)
        for i, h in enumerate(sorted(hs, key=lambda x: sim.getObjectAlias(x))):
            st, amb, dif, spe = sim.getLightParameters(h)
            sim.setLightParameters(h, int(bool(on[i % len(on)])), None,
                                   [min(1.0, x * gain) for x in dif], [min(1.0, x * gain) for x in spe])


def render(sim, body, src, appear_seed, light=None):
    """render_shift_heldout.render_file (original room, no override) with the appearance seed replaced and the lighting
    hook installed; returns (source dict, frames, cam_pose, R)."""
    d = RS.load(src)
    d2 = dict(d, room_seed=np.int64(appear_seed), _light=light)
    RS.SCENE_HOOK = light_hook            # constant for all threads; the per-clip lighting rides in d2
    if body == "b1":
        cp_src = str(d["cf_source_path"]) if "cf_source_path" in d else None
        centre = RS.load(os.path.join(ROOT, cp_src))["base_pos"][0] if cp_src else d["base_pos"][0]
        fr, cp, R = RS.b1_render(sim, d2, np.asarray(centre[:2], float))
    else:
        fr, cp, R = RS.hex_render(sim, d2, RS.HEX_SCENE[body], None)
    return d, fr, cp, R


def do_gate(a):
    """original seed + untouched lights through this path must reproduce the stored frames exactly."""
    J = []
    for body in BODIES:
        J += [(body, sorted(glob.glob(os.path.join(src_dir(body, k), "*.npz")))[i]) for k in KINDS for i in (0, 7)]
    res = []

    def fn(sim, j):
        body, src = j
        d = RS.load(src)
        _, fr, cp, R = render(sim, body, src, int(d["room_seed"]), None)
        diff = np.abs(fr.astype(int) - d["frames"].astype(int))
        res.append(dict(file=os.path.relpath(src, ROOT), mae=float(diff.mean()), max=int(diff.max())))
        return f"mae {diff.mean():.4f} max {diff.max()}"
    RS.run_pool(J, a.ports, fn, "gate")
    os.makedirs(CHK, exist_ok=True)
    json.dump(res, open(os.path.join(CHK, "gate.json"), "w"), indent=1)
    bad = [r for r in res if r["max"] > 0]
    print(f"gate: {len(res) - len(bad)}/{len(res)} exact" + ("" if not bad else f"; NOT exact: {bad}"))


def do_sheet(a):
    from PIL import Image, ImageDraw
    rows = []
    J = []
    for body in BODIES:
        fs = sorted(glob.glob(os.path.join(src_dir(body, "clips"), "*.npz")))
        J += [(body, fs[i]) for i in (0, 30)]
    out = {}

    def fn(sim, j):
        body, src = j
        d = RS.load(src)
        tiles = [d["frames"][20]]
        for v, s in APPEAR.items():
            _, fr, _, _ = render(sim, body, src, s, lighting(v, int(d["room_seed"])))
            tiles.append(fr[20])
        out[j] = tiles
        return "ok"
    RS.run_pool(J, a.ports, fn, "sheet")
    for j in J:
        row = np.hstack([np.asarray(Image.fromarray(t).resize((192, 192))) for t in out[j]])
        rows.append(row)
    im = Image.fromarray(np.vstack(rows))
    dr = ImageDraw.Draw(im)
    for c, lab in enumerate(["original"] + [f"shared room {v}" for v in APPEAR]):
        dr.text((4 + 192 * c, 4), lab, fill=(255, 255, 0))
    os.makedirs(CHK, exist_ok=True)
    im.save(os.path.join(CHK, "sheet.png"))
    g = Image.fromarray(np.vstack(rows)).convert("L")
    g.save(os.path.join(CHK, "sheet_gray.png"))
    print("->", os.path.relpath(CHK, ROOT), "sheet.png sheet_gray.png (rows: c10, c10, B1, B1; frame 20)")


def do_render(a):
    J = []
    for v in a.versions:
        for body in a.bodies:
            for kind in KINDS:
                os.makedirs(out_dir(v, body, kind), exist_ok=True)
                for src in sorted(glob.glob(os.path.join(src_dir(body, kind), "*.npz"))):
                    dst = os.path.join(out_dir(v, body, kind), os.path.basename(src))
                    if not os.path.exists(dst):
                        J.append((v, body, src, dst))
    print(f"{len(J)} files to render")

    def fn(sim, j):
        v, body, src, dst = j
        d0 = RS.load(src)
        light = lighting(v, int(d0["room_seed"]))
        d, fr, cp, R = render(sim, body, src, APPEAR[v], light)
        out = {k: d[k] for k in d if k not in ("frames", "cam_pose")}
        out.update(frames=fr, cam_pose=cp, sr_version=np.int64(v), sr_appear_seed=np.int64(APPEAR[v]),
                   sr_light_gain=np.float64(light[0]), sr_light_on=np.asarray(light[1], bool))
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, dst)
        return f"v{v} {body} {os.path.basename(src)}"
    RS.run_pool(J, a.ports, fn, "render")


def balanced_rooms(body):
    """ONE render per file (data volume unchanged), room look assigned so no look belongs to a behaviour: per behaviour
    the 8 units (2 main clips + 6 branch groups, a group = one source clip x one branch point, its 24 files share the
    look) get looks (u + behaviour) % 4 -> every behaviour in all 4 looks, 2 units each. Returns {src path: version}."""
    units = {}
    for src in sorted(glob.glob(os.path.join(src_dir(body, "clips"), "*.npz"))):
        with np.load(src, allow_pickle=True) as d:
            units.setdefault(int(d["cond_index"]), {}).setdefault(("clip", os.path.basename(src)), []).append(src)
    for src in sorted(glob.glob(os.path.join(src_dir(body, "branches"), "*.npz"))):
        with np.load(src, allow_pickle=True) as d:
            key = ("group", str(d["cf_source"]), int(d["cf_t"]))
            units.setdefault(int(d["cf_source_cond_index"]), {}).setdefault(key, []).append(src)
    out = {}
    for b, us in units.items():
        keys = sorted(us, key=lambda k: (k[0] != "clip", k[1:]))
        assert len(keys) == 8, (body, b, len(keys))
        for u, k in enumerate(keys):
            for src in us[k]:
                out[src] = (u + b) % 4
    return out


def bal_dir(body, kind):
    return os.path.join(CW, f"srbal_{body}_{kind}_train")


def do_render_balanced(a):
    J = []
    for body in a.bodies:
        rooms = balanced_rooms(body)
        cnt = np.bincount(list(rooms.values()), minlength=4)
        print(f"{body}: {len(rooms)} files, per look {cnt.tolist()}")
        for src, v in rooms.items():
            kind = "branches" if "_branches_" in src else "clips"
            os.makedirs(bal_dir(body, kind), exist_ok=True)
            dst = os.path.join(bal_dir(body, kind), os.path.basename(src))
            if not os.path.exists(dst):
                J.append((v, body, src, dst))
    print(f"{len(J)} files to render")

    def fn(sim, j):
        v, body, src, dst = j
        d0 = RS.load(src)
        light = lighting(v, int(d0["room_seed"]) if "cf_source" not in d0 else
                         int(d0["room_seed"]) * 1000 + int(d0["cf_t"]))
        d, fr, cp, R = render(sim, body, src, APPEAR[v], light)
        out = {k: d[k] for k in d if k not in ("frames", "cam_pose")}
        out.update(frames=fr, cam_pose=cp, sr_version=np.int64(v), sr_appear_seed=np.int64(APPEAR[v]),
                   sr_light_gain=np.float64(light[0]), sr_light_on=np.asarray(light[1], bool))
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, dst)
        return f"v{v} {body}"
    RS.run_pool(J, a.ports, fn, "render_balanced")


def do_verify(a):
    problems = 0
    for v in list(APPEAR) + ["bal"]:
        for body in BODIES:
            for kind in KINDS:
                srcs = sorted(glob.glob(os.path.join(src_dir(body, kind), "*.npz")))
                od = bal_dir(body, kind) if v == "bal" else out_dir(v, body, kind)
                outs = sorted(glob.glob(os.path.join(od, "*.npz")))
                tmp = glob.glob(os.path.join(od, "*.tmp.npz"))
                if not os.path.isdir(od):
                    continue
                print(f"sr{v}_{body}_{kind}_train: {len(outs)} / {len(srcs)} files, tmp {len(tmp)}")
                problems += (len(outs) != len(srcs)) + len(tmp)
                for src in srcs[:: max(1, len(srcs) // 12)]:
                    dst = os.path.join(od, os.path.basename(src))
                    if not os.path.exists(dst):
                        continue
                    s, o = RS.load(src), RS.load(dst)
                    for k in s:
                        if k in ("frames", "cam_pose"):
                            continue
                        if not np.array_equal(np.asarray(s[k]), np.asarray(o[k])):
                            print(f"  field {k} differs: {dst}"); problems += 1
                    if o["frames"].shape != s["frames"].shape or o["frames"].std() < 5:
                        print(f"  frames wrong shape / blank: {dst}"); problems += 1
                    if not np.allclose(o["cam_pose"], s["cam_pose"], atol=1e-6):
                        print(f"  cam_pose differs (room moved?): {dst}"); problems += 1
    print(f"{problems} problems")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("gate", "sheet", "render", "render_balanced", "verify"))
    ap.add_argument("--ports", type=int, nargs="+", default=[25720])
    ap.add_argument("--versions", type=int, nargs="+", default=list(APPEAR))
    ap.add_argument("--bodies", nargs="+", default=list(BODIES))
    a = ap.parse_args()
    {"gate": do_gate, "sheet": do_sheet, "render": do_render, "render_balanced": do_render_balanced,
     "verify": do_verify}[a.cmd](a)
