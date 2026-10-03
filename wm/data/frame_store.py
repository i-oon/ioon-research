"""Frames read on demand instead of held in RAM.

The round-1 counterfactual arm B trains on 3,456 branch clips per body x 31 frames x 256x256x3,
21 GB of uint8 per body once decompressed; the eager loader (`np.load(...)["frames"]` per clip,
kept in the clip dict) needed ~45 GB on a 31 GB machine and the OS killed processes.

**Storage choice: an uncompressed per-clip `.npy` cache opened with `mmap_mode="r"`**, built
once from the `.npz` (frames only; labels, actions and metadata still come from the `.npz` and
stay in RAM, they are small). Measured: decompressing one clip's `frames` from the `.npz` costs
~12.5 ms (it is one deflate stream, so reading 3 frames costs the whole clip), i.e. ~100 ms of
CPU per batch of 8 on top of augmentation; a memory-mapped read of 3 frames is ~0.3 ms. The disk
cost is the uncompressed size, ~0.2 MB per frame (~44 GB for round-1 arm B's train + val
directories), against ~360 GB free.

Each access opens the map, copies out the requested frames and drops it, so no file handle or
mapping is shared across DataLoader worker forks and the touched pages are not kept in the
process's RSS (they stay in the reclaimable page cache).

Cache location: `$WM_FRAME_CACHE`, default `<repo>/data/_frame_cache`, mirroring the source
path. A cache file older than its source `.npz` is rebuilt; writes are atomic (tmp + rename).
"""
import hashlib
import os
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE = os.path.join(ROOT, "data", "_frame_cache")


def cache_root():
    return os.environ.get("WM_FRAME_CACHE", DEFAULT_CACHE)


def cache_path(npz_path):
    src = os.path.realpath(npz_path)
    rel = os.path.relpath(src, ROOT)
    if rel.startswith(".."):
        # outside the repo: keep it unique without mirroring an absolute path
        rel = os.path.join("_external", hashlib.sha1(src.encode()).hexdigest()[:16],
                           os.path.basename(src))
    return os.path.join(cache_root(), rel[:-4] if rel.endswith(".npz") else rel) + ".frames.npy"


def npz_frames_header(npz_path, key="frames"):
    """(shape, dtype) of `key` inside an .npz without decompressing it (None if absent)."""
    with zipfile.ZipFile(npz_path) as z:
        name = key + ".npy"
        if name not in z.namelist():
            return None
        with z.open(name) as f:
            version = np.lib.format.read_magic(f)
            read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                    else np.lib.format.read_array_header_2_0)
            shape, _, dtype = read(f)
    return shape, dtype


def ensure_cache(npz_path):
    """Path of the uncompressed frames cache for `npz_path`, built if missing or stale."""
    out = cache_path(npz_path)
    try:
        if os.path.getmtime(out) >= os.path.getmtime(npz_path):
            return out
    except OSError:
        pass
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with np.load(npz_path, allow_pickle=True) as data:
        frames = data["frames"]
    tmp = f"{out}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.lib.format.write_array(f, np.ascontiguousarray(frames), allow_pickle=False)
    os.replace(tmp, out)
    return out


class LazyFrames:
    """Array-like stand-in for a clip's `frames`: `len`, `shape`, `dtype`, indexing.

    Indexing returns a fresh in-memory array (an int gives one frame, a slice a stack), the same
    values and dtype the eager array gave. `np.asarray(lazy)` reads the whole clip. Pickles as
    its path, so DataLoader workers reopen it themselves.
    """

    def __init__(self, npz_path):
        self.source = npz_path
        self.path = ensure_cache(npz_path)
        arr = np.load(self.path, mmap_mode="r")
        self.shape, self.dtype = tuple(arr.shape), arr.dtype
        del arr

    def __len__(self):
        return self.shape[0]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __getitem__(self, idx):
        arr = np.load(self.path, mmap_mode="r")
        out = np.array(arr[idx])
        del arr
        return out

    def __array__(self, dtype=None, copy=None):
        out = self[:]
        return out if dtype is None else out.astype(dtype)

    def __repr__(self):
        return f"LazyFrames({self.source!r}, shape={self.shape})"


def resident_frame_bytes(clips):
    """Bytes of frames held in RAM by a list of clip dicts (lazy frames count 0)."""
    total = 0
    for clip in clips:
        f = clip.get("frames")
        if isinstance(f, np.ndarray):
            total += f.nbytes
    return total


def eager_frame_bytes(paths):
    """What loading `frames` of every path eagerly would hold, read from the .npz headers."""
    total = 0
    for p in paths:
        h = npz_frames_header(p)
        if h is not None:
            total += int(np.prod(h[0])) * np.dtype(h[1]).itemsize
    return total


def check_frame_budget(nbytes, limit_gb, what):
    if limit_gb and limit_gb > 0 and nbytes > limit_gb * 1024 ** 3:
        raise MemoryError(
            f"{what}: frames would hold {nbytes / 1024 ** 3:.1f} GB in RAM, over the "
            f"{limit_gb:g} GB limit (--max_frame_ram_gb). Use lazy frame loading "
            f"(--lazy_frames True, the default) or raise the limit if the machine has the RAM.")
