"""How much of the latent is "which body is this", on bodies the model trained on and on bodies it did not?

z_content.py measured the split on the five training bodies, because a balanced body-by-phase grid
needs every body present at every timestep. The two held-out bodies walk the same expert episodes
as each other, so the same grid can be built from them alone.

Two rows is not five, so the training bodies are also measured pairwise, all ten pairs, to give a
like-for-like reference at the same group size. Without that control a large body share on the
held-out pair could just be an artefact of having only one contrast to estimate it from.
"""
import os, sys, warnings
from dataclasses import fields
warnings.filterwarnings('ignore')
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from vjepa2_encoder import VJEPA2FrameEncoder
from wm.config import from_checkpoint
from wm.data.dataset import load_clip
from wm.evaluate import encode_clip
from wm.models.itm import InverseTransitionModel

TRAIN = ['c10f10t10','c06f10t10','c10f10t06','c06f10t06','c10f06t06']
HELD  = ['c08f09t09','c06f06t06']
# Two of the five training bodies veer: a 94.6 mm dead zone against a 92.5 mm closest target
# leaves them yawing 0.35-0.38 m off course, where every sound body stays under 0.17 m (F42).
# They stay in the headline row because the before/after comparison is matched on them, but the
# split is also reported on the three sound bodies alone, since a body with a distinct gait adds
# between-body variance and would inflate exactly the term being claimed as small.
VEERING = ['c10f10t06','c06f10t06']
SOUND = [b for b in TRAIN if b not in VEERING]
EPS = [6, 20, 22]

CACHE = f'{ROOT}/results/wm/cache/stage2_embeddings.pt'
cache = torch.load(CACHE, map_location='cpu') if os.path.exists(CACHE) else {}
enc, fresh = None, False
E = {}
for b in TRAIN + HELD:
    clips = []
    for e in EPS:
        path = f'{ROOT}/data/ik_walk_8body/{b}_ep{e}.npz'
        if path not in cache:
            if enc is None:
                enc = VJEPA2FrameEncoder(device='cpu', dtype=torch.float32)
            cache[path] = encode_clip(enc, load_clip(path)['frames'], 2)
            fresh = True
        clips.append(cache[path])
    E[b] = clips
if fresh:
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(cache, CACHE)
del enc
print('encoded', flush=True)

def split(ck_path, bodies):
    ck = torch.load(f'{ROOT}/wm/runs/{ck_path}', map_location='cpu', weights_only=False)
    itm = InverseTransitionModel(from_checkpoint(ck['config'])).eval()
    itm.load_state_dict(ck['itm'])
    Z = []
    with torch.no_grad():
        for b in bodies:
            per_body = []
            for clip in E[b]:
                n = len(clip) - 1
                per_body.append(torch.cat([itm(clip[s:min(s+8,n)], clip[s+1:min(s+8,n)+1])
                                           for s in range(0, n, 8)]).numpy())
            Z.append(np.concatenate(per_body))
    per = min(len(z) for z in Z)
    stack = np.stack([z[:per] for z in Z])                 # bodies x timesteps x 64
    centred = stack - stack.reshape(-1, stack.shape[-1]).mean(0)
    pb = (centred.mean(1) ** 2).sum() * per
    pp = (centred.mean(0) ** 2).sum() * len(bodies)
    pr = ((centred - centred.mean(1)[:, None] - centred.mean(0)[None]) ** 2).sum()
    tot = pb + pp + pr
    return 100*pp/tot, 100*pb/tot, 100*pr/tot

from itertools import combinations
for tag, ck in (('control m3d_bracketed ep6', 'm3d_bracketed/epoch006.pt'),
                ('cross   m3d_cross ep8',     'm3d_cross/epoch008.pt')):
    g, b, r = split(ck, TRAIN)
    print(f'{tag:<28} all 5 training bodies     gait {g:5.1f}%   body {b:5.1f}%   rest {r:5.1f}%')
    g, b, r = split(ck, SOUND)
    print(f'{tag:<28} the 3 SOUND bodies        gait {g:5.1f}%   body {b:5.1f}%   rest {r:5.1f}%')
    g, b, r = split(ck, VEERING)
    print(f'{tag:<28} the 2 VEERING bodies      gait {g:5.1f}%   body {b:5.1f}%   rest {r:5.1f}%')
    pairs = [split(ck, list(p)) for p in combinations(TRAIN, 2)]
    import numpy as _np
    m = _np.array(pairs).mean(0); lo = _np.array(pairs)[:,1].min(); hi = _np.array(pairs)[:,1].max()
    print(f'{tag:<28} pairs of TRAINING bodies  gait {m[0]:5.1f}%   body {m[1]:5.1f}%   rest {m[2]:5.1f}%'
          f'   (body ranges {lo:.1f}-{hi:.1f} over 10 pairs)')
    g, b, r = split(ck, HELD)
    print(f'{tag:<28} the 2 HELD-OUT bodies     gait {g:5.1f}%   body {b:5.1f}%   rest {r:5.1f}%\n')
