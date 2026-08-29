"""How far apart do two frames have to be before the real change beats the augmentation noise?

F25 measured the reconstruction target as 8.51 augmentation noise against 1.97 of signal, where
signal is the embedding distance between consecutive frames. The signal is small because at 20 Hz
consecutive frames barely differ. Action chunking, which the source paper uses with a stride of
five, attacks that from the other side: it does not reduce the noise, it increases the distance
the forward model is asked to cover.
"""
import os, sys
import numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'scripts'))
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder
from wm.data.dataset import clip_paths, load_clip
from wm.data.augment import apply, sample_params
from wm.evaluate import encode_clip

enc=VJEPA2FrameEncoder(device='cpu', dtype=torch.float32)
paths=clip_paths(os.path.join(ROOT,'data/fwd_hex8body'), ('c10f10t10',))[:4]
rng=np.random.default_rng(0)
clean, noise = [], []
for p in paths:
    c=load_clip(p); f=c['frames']
    clean.append(encode_clip(enc, f, 2))
    h,w=f.shape[1:3]
    a=np.stack([apply(x, sample_params(rng,h,w)) for x in f])
    b=np.stack([apply(x, sample_params(rng,h,w)) for x in f])
    ea=encode_clip(enc, a, 2); eb=encode_clip(enc, b, 2)
    noise.append(((ea-eb)**2).mean(dim=(1,2)).numpy())

nfloor=float(np.concatenate(noise).mean())
print(f'{len(paths)} clips of c10f10t10\n')
print(f'augmentation noise, two views of the same frame: {nfloor:.2f}\n')
print(f'{"frames apart":>13} {"real change":>12} {"signal / noise":>15}')
for k in (1,2,3,4,5,8,10,16):
    d=[]
    for e in clean:
        if len(e)>k: d.append(((e[:-k]-e[k:])**2).mean(dim=(1,2)).numpy())
    m=float(np.concatenate(d).mean())
    print(f'{k:>13} {m:12.2f} {m/nfloor:14.2f}x')
