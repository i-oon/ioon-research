# beh12 previews — the three sets, as they now stand

One clip per condition, played at the capture rate. **This folder replaces every earlier preview**;
those showed data with defects that have since been corrected, and keeping them invited comparing
the wrong things.

| prefix | set | role |
|---|---|---|
| `hex_c10f10t10__` | `data/beh12_hex_flat` | the body the world model is pretrained on |
| `hex_c08f09t09__` | `data/beh12_c08f09t09_flat` | a held-out body of the same embodiment |
| `b1__` | `data/beh12_b1_flat` | the quadruped, the only executable candidates |

**`turn_all_three.png` is the one to look at first.** All three sets rotate **anticlockwise on
screen** through their strongest turn. They did not until 2026-08-29: the pretraining insect turned
one way while the held-out insect and the B1 turned the other, which made every cross-embodiment
turning result meaningless (F115, F117). On-screen sense is the right language for this because both
scenes ship an identical camera.

**What was corrected in these clips, and why the old previews are gone.**

| | |
|---|---|
| B1 framing | the robot touched an image edge in 61% of frames and 100% of every sideways clip; now 0%, at a 24-degree camera with the floor enlarged and its surface put back at z=0 (F113) |
| B1 camera pinning | every clip carried its own background; now consistent to 0.26 grey levels against the insect's 0.14 (F113) |
| B1 turn levels | `turn_wz0.00` was the forward clip under another name; replaced by four real levels named for the rate they achieve (F114) |
| turn direction | all three sets flipped onto the pretraining body's sense (F115, F117) |

Rebuild with `scripts/dataset/preview_clips.py`. **Watch these before trusting a number**: five of
this project's defects were caught by watching and none by the tables that were passing at the time.
