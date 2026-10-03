"""Froude label correctness: edge handling, segment boundaries, and agreement with the old interior.

    .venv/bin/python3 tests/test_froude_labels.py          (or: .venv/bin/python3 -m pytest tests/test_froude_labels.py)

**Missing data is a FAILURE, not a pass.** Several tests read real clips; if the directory is absent
the test fails with "MISSING DATA: ...". Set FROUDE_TESTS_ALLOW_SKIP=1 to turn those into explicit
skips (reported as SKIP, never as PASS) on a machine that does not hold the data.
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import wm.data.embodiment as E  # noqa: E402

ALLOW_SKIP = os.environ.get("FROUDE_TESTS_ALLOW_SKIP", "") == "1"


class Skipped(Exception):
    pass


def _need(ok, what):
    """Fail (default) or skip (FROUDE_TESTS_ALLOW_SKIP=1) when the data a test reads is missing."""
    if ok:
        return
    msg = f"MISSING DATA: {what} (set FROUDE_TESTS_ALLOW_SKIP=1 to skip instead of failing)"
    if not ALLOW_SKIP:
        raise AssertionError(msg)
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skipped(msg)


class _D(dict):
    """A dict of clip fields that the embodiment readers accept as an npz."""
    files = property(lambda self: list(self.keys()))


def _fields(path):
    with np.load(path, allow_pickle=True) as z:
        return _D({k: z[k] for k in z.files})


def old_smooth(x, window):
    return np.convolve(x, np.ones(window) / window, mode="same")


def test_interior_unchanged():
    rng = np.random.default_rng(0)
    x = rng.normal(size=66)
    w = 20
    new, old = E.smooth(x, w), old_smooth(x, w)
    interior = slice(w // 2, 66 - w // 2)
    assert np.allclose(new[interior], old[interior], atol=1e-12), "interior must equal the plain moving average"


def test_constant_signal_flat_at_edges():
    x = np.full(66, 0.3)
    assert np.allclose(E.smooth(x, 20), 0.3), "a constant signal must read the same at the edges"


def test_segment_isolation():
    rng = np.random.default_rng(1)
    pre_a, post = rng.normal(size=(2, 10)), rng.normal(size=21)
    seg = np.r_[np.zeros(10), np.ones(21)]
    ya = E.smooth(np.r_[pre_a[0], post], 20, seg)
    yb = E.smooth(np.r_[pre_a[1], post], 20, seg)
    assert np.array_equal(ya[10:], yb[10:]), "labels after a switch must not depend on the frames before it"


def test_real_clips_steady_edges():
    """Steady B1 clips: the corrected labels at BOTH clip edges (first and last frame) are close to
    mid-clip (the zero-padded labels read ~half of it at either end)."""
    d = "data/counterfactual_walks/b1_clips_train"
    fs = sorted(glob.glob(os.path.join(ROOT, d, "*.npz")))
    _need(fs, d)
    errs = {"first": [], "last": []}
    for f in fs:
        bm = np.asarray(E.load(f, E.REGISTRY["b1"])["body_motion"])[:, 0]
        mid = np.median(bm[20:46])
        for edge, v in (("first", bm[0]), ("last", bm[-1])):
            errs[edge].append(abs(v - mid) / max(abs(mid), 1e-3))
    for edge, e in errs.items():
        assert np.median(e) < 0.25, f"{edge} frame: median relative edge error {np.median(e):.3f}"


# hexapod dirs in data/counterfactual_walks: default "c10" (the official set, F305); FROUDE_TEST_HEX=c08 for the c08 test set
HEXP = os.environ.get("FROUDE_TEST_HEX", "c10")
CF_DIRS_CW = (("data/counterfactual_walks/b1_branches_train", "b1"), (f"data/counterfactual_walks/{HEXP}_branches_heldout", "hexapod"))
CF_DIR_V3 = ("data/_archive_old_datasets/egocentric_v3/b1_cf_branches_train", "b1")   # fallback only
POSE_KEYS = {"b1": ("base_pos", "base_quat", "com_pos"), "hexapod": ("head", "body_quat", "com_pos")}


def _cf_files(per_dir=6):
    """Real counterfactual-branch files: the v4 dirs present, else the v3 B1 branches."""
    dirs = [(d, n) for d, n in CF_DIRS_CW if glob.glob(os.path.join(ROOT, d, "*.npz"))]
    if not dirs and glob.glob(os.path.join(ROOT, CF_DIR_V3[0], "*.npz")):
        dirs = [CF_DIR_V3]
    out = []
    for d, name in dirs:
        fs = sorted(glob.glob(os.path.join(ROOT, d, "*.npz")))
        out += [(f, name) for f in fs[::max(1, len(fs) // per_dir)][:per_dir]]
    return out


def test_cf_branch_labels_independent_of_prefix():
    """On real branch files: `segment` switches 0 -> 1 exactly once, at `first_pair` (== the branch
    index), and the labels from `first_pair` on do not depend on the prefix at all -- replacing every
    pre-branch pose (position, orientation, CoM) with noise leaves them bit-identical."""
    files = _cf_files()
    _need(files, f"counterfactual branches ({', '.join(d for d, _ in CF_DIRS_CW)} or {CF_DIR_V3[0]})")
    rng = np.random.default_rng(0)
    for f, name in files:
        d = _fields(f)
        tag = os.path.relpath(f, ROOT)
        assert "segment" in d and "first_pair" in d, f"{tag}: no segment / first_pair"
        seg, fp = np.asarray(d["segment"]), int(d["first_pair"])
        switches = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        assert len(switches) == 1 and seg[0] == 0 and seg[-1] == 1, f"{tag}: segment {seg}"
        assert switches[0] == fp, f"{tag}: segment switches at {switches[0]}, first_pair {fp}"
        if "cf_branch_index" in d:
            assert int(d["cf_branch_index"]) == fp, f"{tag}: cf_branch_index != first_pair"
        assert "froude_height" in d, f"{tag}: no froude_height (height would depend on the prefix)"
        emb = E.REGISTRY[name]
        ref = emb.read(d)["body_motion"]
        alt = _D(d)
        for k in POSE_KEYS[name]:
            if k in alt:
                a = np.asarray(alt[k], dtype=np.float64).copy()
                a[:fp] = rng.normal(size=a[:fp].shape)
                if "quat" in k:
                    a[:fp] /= np.linalg.norm(a[:fp], axis=1, keepdims=True)
                alt[k] = a
        new = emb.read(alt)["body_motion"]
        assert np.array_equal(new[fp:], ref[fp:]), \
            f"{tag}: labels after the branch depend on the prefix (max diff {np.abs(new[fp:] - ref[fp:]).max():.2e})"
        assert not np.allclose(new[:fp], ref[:fp]), f"{tag}: perturbation had no effect (test is vacuous)"


def test_v4_hex_clip_with_segment():
    """A v4 hexapod branch clip (with `segment`) loads, gives finite (T, 3) labels, carries
    `first_pair` from the file, and its labels from `first_pair` on equal those of the same clip cut
    at `first_pair` (so nothing before the switch reaches them)."""
    fs = sorted(glob.glob(os.path.join(ROOT, f"data/counterfactual_walks/{HEXP}_branches_heldout", "*.npz")))
    _need(fs, f"data/counterfactual_walks/{HEXP}_branches_heldout")
    f = fs[0]
    clip = E.load(f, E.HEXAPOD)
    bm = np.asarray(clip["body_motion"])
    d = _fields(f)
    fp = int(d["first_pair"])
    assert bm.shape == (len(d["frames"]), 3) and np.isfinite(bm).all()
    assert clip["first_pair"] == fp > 0
    cut = _D({k: (v[fp:] if k in ("head", "body_quat", "com_pos", "segment", "actions", "forces", "frames")
                  else v) for k, v in d.items()})
    assert np.array_equal(E.HEXAPOD.read(cut)["body_motion"], bm[fp:]), "labels after first_pair differ from the cut clip"


CW = os.path.join(ROOT, "data/counterfactual_walks")


def _without(path, drop):
    """The clip's fields minus `drop`, as an object the embodiment readers accept."""
    class D(dict):
        files = property(lambda self: list(self.keys()))
    with np.load(path, allow_pickle=True) as z:
        return D({k: z[k] for k in z.files if k not in drop})


def _hex_clip(cond="turn_s0.56"):
    for s in ("train", "val", "heldout"):
        for p in sorted(glob.glob(os.path.join(CW, f"{HEXP}_clips_{s}", "*.npz"))):
            with np.load(p, allow_pickle=True) as z:
                if str(z["condition"]) == cond and "com_pos" in z.files:
                    return p
    return None


def test_no_com_pos_unchanged():
    """(a) Without `com_pos` the readers give exactly the pre-F301 labels (head / base reference,
    median head / base height; B1 `froude_height` override kept)."""
    p = _hex_clip()
    _need(p is not None, "a v4 hexapod clip with com_pos (data/counterfactual_walks/c10_clips_*)")
    if p is not None:
        d = _without(p, ("com_pos",))
        pos, q = d["head"].astype(np.float64), d["body_quat"]
        h = float(np.median(pos[:, 2]))
        old = np.concatenate([E.body_velocity(pos, q, 0.05, "hexapod"), E.yaw_rate(q, 0.05, "hexapod", h)], 1)
        assert np.array_equal(E._hexapod(d)["body_motion"], old), "hexapod without com_pos changed"
    fs = sorted(glob.glob(os.path.join(CW, "b1_clips_train", "*.npz")))[:1]
    _need(fs, "data/counterfactual_walks/b1_clips_train")
    for f in fs:
        d = _without(f, ("com_pos",))
        pos, q = d["base_pos"].astype(np.float64), d["base_quat"]
        h = float(np.median(pos[:, 2]))
        old = np.concatenate([E.body_velocity(pos, q, float(d["dt"]), "b1", height=h),
                              E.yaw_rate(q, float(d["dt"]), "b1", h)], 1)
        assert np.array_equal(E._b1(d)["body_motion"], old), "B1 without com_pos changed"
        d["froude_height"] = np.float64(0.5)
        old = np.concatenate([E.body_velocity(pos, q, float(d["dt"]), "b1", height=0.5),
                              E.yaw_rate(q, float(d["dt"]), "b1", 0.5)], 1)
        assert np.array_equal(E._b1(d)["body_motion"], old), "B1 froude_height override changed"


def test_com_lever_arm_relation():
    """(b) With `com_pos` the hexapod labels are taken at the CoM: lateral at the CoM = lateral at the
    head - yaw x d / h (d = head's forward offset from the CoM, ~0.246 m; same height h), forward equal."""
    ps = [p for c in ("turn_s0.56", "turn_s0.56_neg", "speed_c5.8_bwd", "side_L_lvl2") for p in [_hex_clip(c)] if p]
    _need(ps, "v4 hexapod clips with com_pos (data/counterfactual_walks/c10_clips_*)")
    for p in ps:
        with np.load(p, allow_pickle=True) as z:
            head, com, q = z["head"].astype(np.float64), z["com_pos"].astype(np.float64), z["body_quat"].astype(np.float64)
        lab = E.load(p, E.HEXAPOD)["body_motion"].astype(np.float64)
        h = float(np.median(com[:, 2]))
        fa = E.forward_axis(q, "hexapod")
        dfwd = float(((head - com)[:, :2] * fa).sum(1).mean())
        assert 0.22 < dfwd < 0.27, f"head-CoM forward offset {dfwd:.3f}"
        lat_head = E.body_velocity(head, q, 0.05, "hexapod", height=h)[:, 1].astype(np.float64)
        pred = lat_head - lab[:, 2] * dfwd / h
        assert abs(pred.mean() - lab[:, 1].mean()) < 0.002, f"{os.path.basename(p)} clip mean {pred.mean():.4f} vs {lab[:, 1].mean():.4f}"
        assert np.abs(pred - lab[:, 1]).max() < 0.02, f"{os.path.basename(p)} per frame max {np.abs(pred - lab[:, 1]).max():.4f}"


def test_com_height_is_median_com_z():
    """(c) Froude height = median CoM z: the loader's labels equal body_velocity / yaw_rate at com_pos
    with h = median(com z), for a hexapod and a B1 clip."""
    cases = [(p, E.HEXAPOD, "hexapod", "body_quat") for p in [_hex_clip("speed_c7.1")] if p]
    cases += [(f, E.B1, "b1", "base_quat") for f in sorted(glob.glob(os.path.join(CW, "b1_clips_train", "*.npz")))[:1]]
    _need(cases, "data/counterfactual_walks/c10_clips_* / b1_clips_train with com_pos")
    checked = 0
    for p, emb, name, qk in cases:
        with np.load(p, allow_pickle=True) as z:
            if "com_pos" not in z.files or "froude_height" in z.files:
                continue
            com, q, dt = z["com_pos"].astype(np.float64), z[qk], float(z["dt"])
        h = float(np.median(com[:, 2]))
        ref = np.concatenate([E.body_velocity(com, q, dt, name, height=h), E.yaw_rate(q, dt, name, h)], 1)
        assert np.array_equal(E.load(p, emb)["body_motion"], ref), f"{name}: labels not at median CoM height"
        d = _without(p, ())
        d["com_pos"] = d["com_pos"].copy(); d["com_pos"][:, 2] += 0.1
        alt = emb.read(d)["body_motion"]
        assert not np.allclose(alt, ref), f"{name}: height does not follow com z"
        checked += 1
    assert checked >= 1, "no clip had com_pos without froude_height: nothing was checked"


if __name__ == "__main__":
    fails = skips = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Skipped as e:
                skips += 1
                print(f"SKIP {name}: {e}")
            except Exception as e:  # noqa: BLE001 -- an error is a failure too, not a crash of the runner
                if type(e).__name__ == "Skipped":      # pytest's skip outcome
                    skips += 1
                    print(f"SKIP {name}: {e}")
                    continue
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{fails} failed, {skips} skipped")
    sys.exit(1 if fails else 0)
