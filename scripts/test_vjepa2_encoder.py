"""Step 0 sanity check: download V-JEPA2 and confirm output shape on a synthetic clip.

Usage: python scripts/test_vjepa2_encoder.py
"""
import torch
from transformers import AutoModel, AutoVideoProcessor

MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_ID} on {device} ...")
processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

n_frames = model.config.frames_per_clip  # 64
crop = model.config.crop_size            # 256

print(f"Building synthetic clip: {n_frames} frames x {crop}x{crop}x3 (random noise)")
clip = [torch.randint(0, 256, (crop, crop, 3), dtype=torch.uint8).numpy() for _ in range(n_frames)]

inputs = processor(clip, return_tensors="pt").to(device, torch.float16)
print("Processed input shape:", inputs["pixel_values_videos"].shape)

with torch.no_grad():
    out = model(**inputs)

last_hidden = out.last_hidden_state
print("Encoder output shape:", tuple(last_hidden.shape))

tubelet = model.config.tubelet_size
expected_temporal_tokens = n_frames // tubelet
expected_spatial_tokens = (crop // model.config.patch_size) ** 2
expected_total = expected_temporal_tokens * expected_spatial_tokens
print(f"Expected tokens: {expected_temporal_tokens} (temporal) x {expected_spatial_tokens} (spatial) = {expected_total}")
print(f"Hidden size: {model.config.hidden_size}")

assert last_hidden.shape[1] == expected_total, "token count mismatch"
assert last_hidden.shape[2] == model.config.hidden_size, "hidden size mismatch"
print("\nPASS: output shape matches config-derived expectation.")
