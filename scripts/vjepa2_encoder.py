"""Frozen V-JEPA2 wrapper that encodes single frames in isolation.

Feeds each frame twice into the minimal 2-frame tubelet so the model
operates as a per-frame image encoder rather than a video-clip encoder,
avoiding cross-timestep attention leakage (see scripts/test_vjepa2_frame_isolation.py).
"""
import torch
from transformers import AutoModel, AutoVideoProcessor

MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


class VJEPA2FrameEncoder:
    def __init__(self, device=None, dtype=torch.float16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID, dtype=dtype).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, frames):
        """frames: list of HxWx3 uint8 arrays (one video frame each).

        Returns: (len(frames), 256, 1408) tensor — one independent,
        context-free embedding per input frame.
        """
        clips = [[f, f] for f in frames]  # duplicate each frame into its own 2-frame tubelet
        inputs = self.processor(clips, return_tensors="pt").to(self.device, self.dtype)
        out = self.model(**inputs)
        return out.last_hidden_state  # (batch, 256, 1408)


if __name__ == "__main__":
    import numpy as np

    encoder = VJEPA2FrameEncoder()
    frames = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(4)]
    e = encoder.encode(frames)
    print("encoded shape:", tuple(e.shape))
    assert e.shape == (4, 256, 1408)
    print("OK")
