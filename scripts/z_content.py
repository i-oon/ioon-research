"""Is the latent still a behaviour representation after lambda_cross, or has it been hollowed out?

Stage 2 rests on z carrying behaviour across bodies. lambda_cross drives the decoder onto the
frame so completely that removing z costs it only 2.2-3.2x, against 21x in the control. Two
read-outs decide whether that means z is empty or merely no longer carrying the body:
foot-contact pattern decodable from z, and how z's variance splits between gait and body.
"""
import os, sys, warnings
from dataclasses import fields
warnings.filterwarnings('ignore')
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from vjepa2_encoder import VJEPA2FrameEncoder
from wm.config import Config
from wm.data.dataset import load_clip, contact_labels
from wm.evaluate import encode_clip, behaviour_labels
from wm.models.itm import InverseTransitionModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

BODIES = ['c10f10t10', 'c06f10t10', 'c10f10t06', 'c06f10t06', 'c10f06t06']
EPS = [6, 20, 22]

enc = VJEPA2FrameEncoder(device='cuda', dtype=torch.float16)
# keep clips separate: concatenating them first would create a transition across each clip
# boundary, which is not a transition at all
E, C = {}, {}
for b in BODIES:
    E[b] = [encode_clip(enc, load_clip(f'{ROOT}/data/ik_walk_8body/{b}_ep{e}.npz')['frames'], 8).float().cpu()
            for e in EPS]
    C[b] = [contact_labels(load_clip(f'{ROOT}/data/ik_walk_8body/{b}_ep{e}.npz')['forces'][:-1]) for e in EPS]
del enc; torch.cuda.empty_cache()
print('encoded', flush=True)

def latents_for(ck_path):
    ck = torch.load(f'{ROOT}/wm/runs/{ck_path}', map_location='cpu', weights_only=False)
    cfg = Config(**{k: v for k, v in ck['config'].items() if k in {f.name for f in fields(Config)}})
    cfg.train_morphs = tuple(cfg.train_morphs)
    itm = InverseTransitionModel(cfg).cuda().eval(); itm.load_state_dict(ck['itm'])
    Z, lab, body = [], [], []
    with torch.no_grad():
        for i, b in enumerate(BODIES):
            for clip_e, clip_c in zip(E[b], C[b]):
                n = min(len(clip_e) - 1, len(clip_c))
                parts = []
                for s in range(0, n, 8):
                    t = min(s + 8, n)
                    parts.append(itm(clip_e[s:t].cuda(), clip_e[s + 1:t + 1].cuda()).cpu())
                Z.append(torch.cat(parts).numpy()); lab.append(clip_c[:n]); body += [i] * n
    del itm; torch.cuda.empty_cache()
    return np.concatenate(Z), np.concatenate(lab), np.array(body)

for tag, ck in (('control m3d_bracketed ep20', 'm3d_bracketed/epoch020.pt'),
                ('cross   m3d_cross ep8',      'm3d_cross/epoch008.pt'),
                ('cross   m3d_cross ep27',     'm3d_cross/epoch027.pt')):
    Z, lab, body = latents_for(ck)
    codes, keep = behaviour_labels(lab)
    n_cls = len(set(codes[keep]))
    beh = cross_val_score(LogisticRegression(max_iter=3000), Z[keep], codes[keep], cv=5).mean()
    bod = cross_val_score(LogisticRegression(max_iter=3000), Z, body, cv=5).mean()
    per = len(Z) // len(BODIES)
    stack = Z[:per * len(BODIES)].reshape(len(BODIES), per, -1)
    centred = stack - stack.reshape(-1, Z.shape[-1]).mean(0)
    pb = (centred.mean(1) ** 2).sum() * per
    pp = (centred.mean(0) ** 2).sum() * len(BODIES)
    pr = ((centred - centred.mean(1)[:, None] - centred.mean(0)[None]) ** 2).sum()
    tot = pb + pp + pr
    print(f'\n--- {tag} ---')
    print(f'  behaviour from z : {beh:.4f}   over {n_cls} contact patterns, majority class {max(np.bincount(np.unique(codes[keep], return_inverse=True)[1]))/keep.sum():.3f}')
    print(f'  body from z      : {bod:.4f}   (chance {1/len(BODIES):.3f})')
    print(f'  variance of z    : gait {100*pp/tot:.1f}%   body {100*pb/tot:.1f}%   rest {100*pr/tot:.1f}%')
