"""F293: re-render B1 v3 frames whose camera orientation followed the buggy `_face_forward` quaternion.

render_b1_replay (and build_b1_cf_branches) posed the base -- and the ego camera parented to it -- from
the stored base_quat, so wherever the buggy quaternion's orientation differs from the corrected one the
frames show the wrong view. This re-renders, in the exact v3 scene (counterfactual_readout_b1.
build_b1v3_scene == render_b1_replay --ego --match_floor --ground_uv_mult 1.0, room seed from the
clip's render_note), every clip / CF branch whose orientation differs anywhere by more than THRESH deg
(angle between the normalised buggy quaternion and the corrected one), from the corrected poses. Old
frames kept as `frames_buggy`. Before anything is written, an UNAFFECTED clip is re-rendered and must
reproduce its stored frames (pixel corr >= 0.9999).

    .venv/bin/python3 scripts/dataset/rerender_b1_f293.py [--only sources|branches] [--dry]
Needs CoppeliaSim on --port.
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "sim/render", "sim/scene", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
THRESH = 0.05
SRC_DIRS = ("data/egocentric_v3/beh24_b1_ego_flat", "data/egocentric_v3/beh12_b1_ego_flat")
CF_DIRS = ("data/egocentric_v3/b1_cf_branches_train", "data/egocentric_v3/b1_cf_branches_val")
SPLIT_SRC = {"b1_cf_branches_train": "beh24_b1_ego_flat_cleantrain", "b1_cf_branches_val": "beh24_b1_ego_flat_cleanval"}


def orient_diff_deg(q_ok, q_bug):
    a = np.asarray(q_ok, float); b = np.asarray(q_bug, float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))


def seed_of(d):
    return int(re.search(r"seed (\d+)", str(d["render_note"])).group(1))


def render_clip(pose, pos, quat, jpos, spawn):
    out = []
    for p, q, j in zip(pos, quat, jpos):
        p = np.asarray(p, float).copy(); p[:2] -= spawn
        out.append(pose(p, q, j))
    return np.asarray(out, np.uint8)


def corr(a, b):
    return float(np.corrcoef(a.ravel().astype(float), b.ravel().astype(float))[0, 1])


def save(path, data):
    tmp = path[:-4] + ".f293tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--only", choices=("sources", "branches"), default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from counterfactual_readout_b1 import build_b1v3_scene
    from wm.data.embodiment import heading
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    scene = lambda d: build_b1v3_scene(sim, args.scene, seed_of(d),  # noqa: E731
                                       float(heading(d["base_quat"][:1].astype(float), "b1")[0]))
    t0 = time.time()
    try:
        # pipeline check on an unaffected clip
        ref = os.path.join(ROOT, "data/egocentric_v3/beh24_b1_ego_flat/b1_ep0.npz")
        d = np.load(ref, allow_pickle=True)
        fr = render_clip(scene(d), d["base_pos"][:8], d["base_quat"][:8], d["joint_pos"][:8], d["base_pos"][0, :2])
        c = min(corr(a, b) for a, b in zip(fr, d["frames"][:8]))
        print(f"pipeline check b1_ep0 (unfixed clip) (8 frames): min pixel corr {c:.5f}", flush=True)
        if c < 0.9999:
            raise SystemExit("re-render does not reproduce the v3 pipeline")

        if args.only in (None, "sources"):
            for sd in SRC_DIRS:
                n_r = 0
                below = []
                for p in sorted(glob.glob(os.path.join(ROOT, sd, "*.npz"))):
                    if os.path.islink(p):
                        continue
                    with np.load(p, allow_pickle=True) as f:
                        data = {k: f[k] for k in f.files}
                    if not str(data.get("pose_fix", "")).startswith("F293"):
                        continue
                    if "frames_buggy" in data:
                        n_r += 1
                        continue
                    dmax = float(orient_diff_deg(data["base_quat"], data["base_quat_buggy"]).max())
                    if dmax <= THRESH:
                        below.append((os.path.basename(p), dmax))
                        continue
                    if args.dry:
                        n_r += 1
                        continue
                    new = render_clip(scene(data), data["base_pos"], data["base_quat"], data["joint_pos"],
                                      data["base_pos"][0, :2])
                    data["frames_buggy"], data["frames"] = data["frames"], new
                    data["render_note"] = np.array(str(data["render_note"]) + "; F293 re-rendered from corrected poses")
                    save(p, data)
                    n_r += 1
                    print(f"  {sd} {os.path.basename(p)} orient diff max {dmax:.2f} deg -> re-rendered "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
                print(f"{sd}: re-rendered {n_r}; fixed but under {THRESH} deg: {len(below)}, max diff "
                      f"{max([b[1] for b in below], default=0):.4f} deg", flush=True)

        if args.only in (None, "branches"):
            for cd in CF_DIRS:
                srcdir = os.path.join(ROOT, "data/egocentric_v3", SPLIT_SRC[os.path.basename(cd)])
                files = sorted(glob.glob(os.path.join(ROOT, cd, "*.npz")))
                by_src = {}
                for p in files:
                    with np.load(p, allow_pickle=True) as f:
                        by_src.setdefault(str(f["cf_source"]), []).append(p)
                n_r = n_skip = 0
                for src, ps in sorted(by_src.items()):
                    s = np.load(os.path.join(srcdir, src), allow_pickle=True)
                    src_rr = "frames_buggy" in s.files
                    pose = None
                    for p in ps:
                        with np.load(p, allow_pickle=True) as f:
                            data = {k: f[k] for k in f.files}
                        if "frames_buggy" in data:
                            n_r += 1
                            continue
                        dmax = float(orient_diff_deg(data["base_quat"], data["base_quat_buggy"]).max())
                        if not src_rr and dmax <= THRESH:
                            n_skip += 1
                            continue
                        if args.dry:
                            n_r += 1
                            continue
                        if pose is None:
                            pose = scene(s)
                        new = render_clip(pose, data["base_pos"], data["base_quat"], data["joint_pos"],
                                          s["base_pos"][0, :2])
                        data["frames_buggy"], data["frames"] = data["frames"], new
                        data["render_note"] = np.array(str(data["render_note"]) + "; F293 re-rendered from corrected poses")
                        save(p, data)
                        n_r += 1
                    print(f"  {cd} source {src}: done ({n_r} re-rendered, {(time.time() - t0) / 60:.1f} min)", flush=True)
                print(f"{cd}: re-rendered {n_r}, unchanged {n_skip}", flush=True)
    finally:
        sim.stopSimulation()
    print(f"RERENDER_F293_DONE {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
