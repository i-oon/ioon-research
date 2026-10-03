"""Train / val / heldout leakage check over every split directory used for training, validation and heldout.

For each split dir, every frame of every clip is hashed (md5 of the raw uint8 bytes). A frame hash that
appears in two different splits is content overlap (the same image in train and val, say). Also checked:
  - the same resolved file (symlink target) in two splits;
  - counterfactual branches: `cf_source` must be a clip of the branch's OWN split's source dir, and the
    source episode must not occur in any other split's source dir;
  - hexapod switching: `expert_episode` and `ego_seed` (room seed) distinct across splits.
Also prints, per dir, files and training pairs counted as wm/data/dataset.py MultiEmbodimentPairs counts
them (range(first_pair, len(frames) - reach), reach = max(rollout_k * frame_stride, action_lag + chunk - 1)
= max(2 * 5, 1 + 5 - 1) = 10).

Exit status 1 on any overlap.
    .venv/bin/python3 scripts/diagnostics/dataset/check_splits.py [--workers 8] [--no_c08]
        (default: the current data, data/counterfactual_walks: c10_clips_* / c10_branches_* (F305), b1_clips_* /
        b1_branches_*, and the c08 test set)
    .venv/bin/python3 scripts/diagnostics/dataset/check_splits.py --legacy   (the older data/egocentric{,_v3} dirs)
The c08f09t09 test set (collect_c08_test_set.py: c08_clips_heldout, c08_branches_heldout; dropped with --no_c08) is included
as split "test" (never trained on): no frame / file may be shared with train / val / heldout, no train / val / heldout
file may carry morph c08f09t09, and every c08 branch's source is a c08 test main clip.
"""
import argparse
import glob
import hashlib
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H, V = "data/egocentric", "data/egocentric_v3"
SPLITS = {   # dir -> (split, body, kind)
    f"{H}/beh24_c10f10t10_ego_flat_cleantrain": ("train", "hexapod", "clip"),
    f"{H}/beh24_c10f10t10_ego_flat_cleanval": ("val", "hexapod", "clip"),
    f"{H}/beh24_c10f10t10_ego_flat_cleanheldout": ("heldout", "hexapod", "clip"),
    f"{H}/beh24_c10f10t10_switch_train": ("train", "hexapod", "switch"),
    f"{H}/beh24_c10f10t10_switch_val": ("val", "hexapod", "switch"),
    f"{H}/beh24_c10f10t10_switch_heldout": ("heldout", "hexapod", "switch"),
    f"{V}/beh24_b1_ego_flat_cleantrain": ("train", "b1", "clip"),
    f"{V}/beh24_b1_ego_flat_cleanval": ("val", "b1", "clip"),
    f"{V}/beh24_b1_ego_flat_cleanheldout": ("heldout", "b1", "clip"),
    f"{V}/b1_cf_branches_train": ("train", "b1", "branch"),
    f"{V}/b1_cf_branches_val": ("val", "b1", "branch"),
    f"{V}/b1_cf_branches_heldout": ("heldout", "b1", "branch"),
    f"{H}/beh12_c08f09t09_ego_flat": ("test", "hexapod_c08f09t09", "clip"),
}
CW = "data/counterfactual_walks"
SPLITS_CW = {   # DATA_PLAN v2 main clips (stage 1 hexapod: scene-reuse c10 clips, F305; stage 2 B1)
    f"{CW}/c10_clips_train": ("train", "hexapod", "main"),
    f"{CW}/c10_clips_val": ("val", "hexapod", "main"),
    f"{CW}/c10_clips_heldout": ("heldout", "hexapod", "main"),
    # stage 3 (2026-10-02): the B1 main clips re-rendered with the corrected ego mount (branch sources) and the
    # counterfactual branches of both bodies
    f"{CW}/b1_clips_train": ("train", "b1", "main"),
    f"{CW}/b1_clips_val": ("val", "b1", "main"),
    f"{CW}/b1_clips_heldout": ("heldout", "b1", "main"),
    f"{CW}/c10_branches_train": ("train", "hexapod", "cw_branch"),
    f"{CW}/c10_branches_val": ("val", "hexapod", "cw_branch"),
    f"{CW}/c10_branches_heldout": ("heldout", "hexapod", "cw_branch"),
    f"{CW}/b1_branches_train": ("train", "b1", "cw_branch"),
    f"{CW}/b1_branches_val": ("val", "b1", "cw_branch"),
    f"{CW}/b1_branches_heldout": ("heldout", "b1", "cw_branch"),
}
SPLITS_C08 = {   # held-out morphology, TEST ONLY (collect_c08_test_set.py); rooms 200-223 like the heldout split
    f"{CW}/c08_clips_heldout": ("test", "hexapod_c08f09t09", "test_main"),
    f"{CW}/c08_branches_heldout": ("test", "hexapod_c08f09t09", "test_branch"),
}
REACH = max(2 * 5, 1 + 5 - 1)


def scan(path):
    with np.load(path, allow_pickle=True) as d:
        fr = d["frames"]
        hs = [hashlib.md5(np.ascontiguousarray(f).tobytes()).hexdigest() for f in fr]
        meta = {k: (d[k].item() if d[k].ndim == 0 else None) for k in
                ("first_pair", "cf_source", "cf_source_episode", "cf_branch_index", "expert_episode", "ego_seed",
                 "room_seed", "cond_index", "copy", "window_start", "source_walk", "morph")
                if k in d.files}
        if "source_walk" in d.files:                       # v4 main clips: actions for the distinctness check
            meta["_act"] = (d["actions"] if "actions" in d.files else d["action"]).astype(np.float64)
    return path, os.path.realpath(path), hs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--legacy", action="store_true", help="check the older data/egocentric{,_v3} dirs instead")
    ap.add_argument("--no_c08", action="store_true", help="leave out the c08f09t09 test set")
    args = ap.parse_args()
    global SPLITS
    if not args.legacy:
        SPLITS = dict(SPLITS_CW)
        if not args.no_c08:
            SPLITS.update(SPLITS_C08)
    jobs = []
    for dr in SPLITS:
        ps = sorted(p for p in glob.glob(os.path.join(ROOT, dr, "*.npz")) if ".tmp" not in p)
        if not ps:
            print(f"MISSING/EMPTY {dr}")
        jobs += [(dr, p) for p in ps]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(scan, [p for _, p in jobs], chunksize=4))
    by_dir = defaultdict(list)
    for (dr, _), r in zip(jobs, res):
        by_dir[dr].append(r)

    bad = 0
    print(f"{'dir':<48}{'split':<9}{'body':<19}{'files':>7}{'pairs':>8}")
    tot = defaultdict(int)
    for dr, rs in by_dir.items():
        split, body, kind = SPLITS[dr]
        pairs = sum(max(0, len(hs) - REACH - int(m.get("first_pair") or 0)) for _, _, hs, m in rs)
        tot[(split, body, kind)] += pairs
        print(f"{dr.split('/')[-1]:<48}{split:<9}{body:<19}{len(rs):>7}{pairs:>8}")

    # frame-content overlap across splits
    where = defaultdict(set)
    example = {}
    for dr, rs in by_dir.items():
        split = SPLITS[dr][0]
        for p, _, hs, _ in rs:
            for i, h in enumerate(hs):
                where[h].add(split)
                example.setdefault((h, split), f"{os.path.relpath(p, ROOT)}[{i}]")
    cross = {h: s for h, s in where.items() if len(s) > 1}
    n_frames = len(where)
    print(f"\nunique frame hashes: {n_frames}; hashes in >1 split: {len(cross)}")
    for h, s in list(cross.items())[:20]:
        print("  OVERLAP", sorted(s), [example[(h, x)] for x in sorted(s)])
    bad += len(cross)

    # same file across splits
    real = defaultdict(set)
    for dr, rs in by_dir.items():
        for _, rp, _, _ in rs:
            real[rp].add(SPLITS[dr][0])
    xf = [r for r, s in real.items() if len(s) > 1]
    print(f"files (resolved) in >1 split: {len(xf)}")
    for r in xf[:10]:
        print("  OVERLAP file", os.path.relpath(r, ROOT), sorted(real[r]))
    bad += len(xf)

    # branch sources belong to their own split
    src_names = {SPLITS[dr][0]: {os.path.basename(p) for p, _, _, _ in rs}
                 for dr, rs in by_dir.items() if SPLITS[dr][1] == "b1" and SPLITS[dr][2] == "clip"}
    for dr, rs in by_dir.items():
        split, body, kind = SPLITS[dr]
        if kind != "branch":
            continue
        own = src_names.get(split, set())
        srcs = {str(m["cf_source"]) for _, _, _, m in rs}
        miss = srcs - own
        other = {s: sp for sp, names in src_names.items() if sp != split for s in srcs & names}
        print(f"{dr.split('/')[-1]}: {len(srcs)} sources, not in own split: {len(miss)}, in another split: {len(other)}")
        bad += len(miss) + len(other)

    # v4 branches: the source is a main clip of the branch's own split (same body), and of no other split
    v4_src = defaultdict(set)                         # (body, split) -> main clip names
    for dr, rs in by_dir.items():
        split, body, kind = SPLITS[dr]
        if kind in ("main", "main_alt"):
            v4_src[(body, split)] |= {os.path.basename(p) for p, _, _, _ in rs}
    for dr, rs in by_dir.items():
        split, body, kind = SPLITS[dr]
        if kind != "cw_branch":
            continue
        srcs = {str(m["cf_source"]) for _, _, _, m in rs}
        miss = srcs - v4_src[(body, split)]
        other = {x for sp in ("train", "val", "heldout") if sp != split for x in srcs & v4_src[(body, sp)]}
        seeds = {int(m["room_seed"]) for _, _, _, m in rs}
        print(f"{dr.split('/')[-1]}: {len(srcs)} sources, not in own split: {len(miss)}, in another split: {len(other)}, "
              f"room seeds {len(seeds)}")
        bad += len(miss) + len(other)

    # c08 test set: test only (no c08 file in a train / val / heldout dir), branch sources = c08 test main clips
    c08 = {dr: rs for dr, rs in by_dir.items() if SPLITS[dr][0] == "test" and SPLITS[dr][2].startswith("test_")}
    if c08:
        leak = [p for dr, rs in by_dir.items() if SPLITS[dr][0] != "test" for p, _, _, m in rs
                if str(m.get("morph") or "") == "c08f09t09"]
        srcs = {os.path.basename(p) for dr, rs in c08.items() if SPLITS[dr][2] == "test_main" for p, _, _, _ in rs}
        brs = [m for dr, rs in c08.items() if SPLITS[dr][2] == "test_branch" for _, _, _, m in rs]
        miss = {str(m["cf_source"]) for m in brs} - srcs
        seeds = sorted({int(m["room_seed"]) for dr, rs in c08.items() for _, _, _, m in rs})
        print(f"c08 test set: {len(srcs)} main clips, {len(brs)} branches; c08 files in train/val/heldout dirs: "
              f"{len(leak)}; branch sources not among the c08 main clips: {len(miss)}; room seeds "
              f"{seeds[0] if seeds else '-'}..{seeds[-1] if seeds else '-'} ({len(seeds)})")
        bad += len(leak) + len(miss) + int(seeds != list(range(200, 224)))

    # hexapod switching: episode ids / room seeds distinct across splits
    for key in ("expert_episode", "ego_seed"):
        seen = defaultdict(set)
        for dr, rs in by_dir.items():
            if SPLITS[dr][2] == "switch":
                for _, _, _, m in rs:
                    seen[int(m[key])].add(SPLITS[dr][0])
        dup = [k for k, s in seen.items() if len(s) > 1]
        print(f"switch {key}: {len(seen)} values, shared across splits: {len(dup)}")
        bad += len(dup)

    # v4 main clips: rooms, windows, distinctness (DATA_PLAN section 6 items 2 and 4)
    mains = {dr: rs for dr, rs in by_dir.items() if SPLITS[dr][2] == "main"}
    if mains:
        seeds = defaultdict(set)                      # (body, split) -> seeds
        key_seed = defaultdict(dict)                  # body -> (split, cond, copy) -> seed
        walks = defaultdict(list)                     # (body, walk) -> [(start, split)]
        conds = defaultdict(list)                     # (body, cond) -> actions
        for dr, rs in mains.items():
            split, body, _ = SPLITS[dr]
            for p, _, _, m in rs:
                seeds[(body, split)].add(int(m["room_seed"]))
                key_seed[body][(split, int(m["cond_index"]), str(m["copy"]))] = int(m["room_seed"])
                walks[(body, str(m["source_walk"]))].append((int(m["window_start"]), split))
                conds[(body, int(m["cond_index"]))].append(m["_act"])
        bodies = sorted({b for b, _ in seeds})
        for b in bodies:
            sp = {s: seeds[(b, s)] for s in ("train", "val", "heldout")}
            x = (sp["train"] & sp["val"]) | (sp["train"] & sp["heldout"]) | (sp["val"] & sp["heldout"])
            print(f"{b}: room seeds per split train {len(sp['train'])} val {len(sp['val'])} heldout "
                  f"{len(sp['heldout'])}; shared across splits {len(x)}")
            bad += len(x)
        if len(bodies) == 2:
            a, b = bodies
            same = key_seed[a] == key_seed[b]
            print(f"room seed per (split, condition, copy) identical for {a} and {b}: {same} "
                  f"({len(key_seed[a])} / {len(key_seed[b])} keys)")
            bad += 0 if same else 1
        ovl = 0
        for (b, w), lst in walks.items():
            st = sorted(lst)
            ovl += sum(1 for (s0, _), (s1, _) in zip(st, st[1:]) if s1 - s0 < 66 and b == "hexapod")
            ovl += sum(1 for (s0, _), (s1, _) in zip(st, st[1:]) if s1 - s0 < 165 and b == "b1")
        print(f"windows of one walk overlapping: {ovl} (window = 66 frames hexapod / 165 policy steps B1)")
        bad += ovl
        dmin = min(min(np.abs(x - y).max() for k, x in enumerate(v) for y in v[k + 1:]) for v in conds.values())
        print(f"distinctness: smallest same-condition max|action difference| {dmin:.4f} (0 = duplicate clip)")
        bad += int(dmin < 1e-3)

    print("\npairs per split / body (counterfactual-style sets):")
    for split in ("train", "val", "heldout"):
        if mains:
            print(f"  {split:<8} main clips hexapod {tot[(split, 'hexapod', 'main')]:>6}   b1 {tot[(split, 'b1', 'main')]:>6}"
                  f"   branches hexapod {tot[(split, 'hexapod', 'cw_branch')]:>6}   b1 {tot[(split, 'b1', 'cw_branch')]:>6}")
            continue
        h, b = tot[(split, "hexapod", "switch")], tot[(split, "b1", "branch")]
        print(f"  {split:<8} hexapod switch {h:>6}   b1 branches {b:>6}   ratio {h / max(b, 1):.4f}")
    if any(SPLITS[dr][0] == "test" for dr in by_dir) and mains:
        print(f"  test     c08 main clips {tot[('test', 'hexapod_c08f09t09', 'test_main')]:>6}   c08 branches "
              f"{tot[('test', 'hexapod_c08f09t09', 'test_branch')]:>6}")
    print("RESULT", "OK no cross-split overlap" if bad == 0 else f"FAIL {bad} overlaps")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
