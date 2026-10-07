"""Frozen-encoder embedding caches, keyed by clip path and checked against the file on disk.

The caches used by evaluation (selection_eval, fit_projector, counterfactual_readout, shared_latent_figure)
map a clip's path to its V-JEPA2 embeddings. Keyed by path alone, a clip regenerated at the same path (a
re-render, a corrected label pass) would silently reuse the old embeddings. Each cache therefore also records
the file's size and modification time under the key "__stat__"; `load_cache` drops every entry whose file
has changed or is gone, so it is re-encoded. A cache written before this check has no "__stat__" and is
discarded whole (it cannot be verified).

    cache = load_cache(path)              # dict path -> tensor, only entries matching the files on disk
    cache[p] = encode(...); note(cache, p)
    save_cache(cache, path)
"""
import os

import torch

STAT = "__stat__"


def _stat(p):
    s = os.stat(p)
    return (s.st_size, s.st_mtime_ns)


def load_cache(path):
    if not os.path.exists(path):
        return {STAT: {}}
    raw = torch.load(path, map_location="cpu")
    stats = raw.get(STAT)
    if stats is None:
        print(f"embedding cache {path}: written without file stamps, discarded (will re-encode)")
        return {STAT: {}}
    out, stale = {STAT: {}}, 0
    for k, v in raw.items():
        if k == STAT:
            continue
        if os.path.exists(k) and stats.get(k) == _stat(k):
            out[k] = v
            out[STAT][k] = stats[k]
        else:
            stale += 1
    if stale:
        print(f"embedding cache {path}: {stale} entr{'y' if stale == 1 else 'ies'} stale or missing on disk, "
              f"dropped (will re-encode)")
    return out


def note(cache, p):
    """Record the stamp of file p after cache[p] was (re)computed."""
    cache.setdefault(STAT, {})[p] = _stat(p)


def save_cache(cache, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(cache, tmp)
    os.replace(tmp, path)


def n_entries(cache):
    return sum(1 for k in cache if k != STAT)


def file_cached(path, cache_dir, compute):
    """One clip's embeddings from a per-file disk cache (2026-10-06): never loads a whole cache dict into RAM.

    The entry is keyed by the real path (symlinked subsets reuse the original's entry) and stamped with the file's size +
    mtime; a mismatch recomputes. `compute()` must return a CPU tensor; it is stored fp16 like every other cache."""
    real = os.path.realpath(path)
    rel = real.replace(os.sep, "__")
    f = os.path.join(cache_dir, rel[-200:] + ".pt")
    stamp = _stat(real)
    if os.path.exists(f):
        d = torch.load(f, map_location="cpu")
        if tuple(d["stamp"]) == stamp:
            return d["e"]
    e = compute().cpu().half()
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save({"e": e, "stamp": stamp}, fh)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, f)
    return e


class DiskRows:
    """A (rows, ...) fp16 array on disk (np.memmap) that indexes like a tensor and returns CPU tensors.
    Used by fit_projector so the embeddings of hundreds of clips never sit in RAM."""

    def __init__(self, path, shape):
        import numpy as np
        self.path = path
        self.mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)

    def __len__(self):
        return self.mm.shape[0]

    @property
    def shape(self):
        return self.mm.shape

    def __getitem__(self, k):
        import numpy as np
        if torch.is_tensor(k):
            k = k.cpu().numpy()
        return torch.from_numpy(np.ascontiguousarray(self.mm[k]))
