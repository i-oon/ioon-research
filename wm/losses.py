"""Training objective: L = lambda_recon * L_recon + lambda_motion * L_motion + lambda_adv * L_adv.

L_adv is a body classifier on z behind a gradient reversal, so adding it *subtracts* body
identity from z rather than adding a term the model can satisfy. Its accuracy is logged
separately: the loss going up is the intended direction, and accuracy falling toward chance is
what says it worked.
"""
import torch.nn.functional as F


def compute_losses(pred_next, target_next, pred_action, target_action, cfg,
                   adv_logits=None, morph_id=None, probe_logits=None):
    recon = F.mse_loss(pred_next, target_next)
    motion = F.mse_loss(pred_action, target_action)
    total = cfg.lambda_recon * recon + cfg.lambda_motion * motion
    parts = {"recon": recon.item(), "motion": motion.item()}

    if adv_logits is not None and morph_id is not None and cfg.lambda_adv > 0:
        adv = F.cross_entropy(adv_logits, morph_id)
        total = total + cfg.lambda_adv * adv
        parts["adv"] = adv.item()
        parts["adv_accuracy"] = (adv_logits.argmax(dim=-1) == morph_id).float().mean().item()

    if probe_logits is not None and morph_id is not None:
        # reads a detached z, so adding its loss trains only the probe and leaves the world
        # model's gradients untouched
        probe = F.cross_entropy(probe_logits, morph_id)
        total = total + probe
        parts["probe_accuracy"] = (probe_logits.argmax(dim=-1) == morph_id).float().mean().item()

    parts["total"] = total.item()
    return total, parts
