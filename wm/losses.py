"""Training objective: L = lambda_recon * L_recon + lambda_motion * L_motion + lambda_adv * L_adv.

L_adv is a body classifier on z behind a gradient reversal, so adding it *subtracts* body
identity from z rather than adding a term the model can satisfy. Its accuracy is logged
separately: the loss going up is the intended direction, and accuracy falling toward chance is
what says it worked.
"""
import torch.nn.functional as F


def compute_losses(pred_next, target_next, pred_action, target_action, cfg,
                   adv_logits=None, morph_id=None, probe_logits=None,
                   cross_action=None, cross_target=None,
                   body_pred=None, body_target=None):
    recon = F.mse_loss(pred_next, target_next)
    motion = F.mse_loss(pred_action, target_action)
    total = cfg.lambda_recon * recon + cfg.lambda_motion * motion
    parts = {"recon": recon.item(), "motion": motion.item()}

    if cross_action is not None and cross_target is not None and cfg.lambda_cross > 0:
        # same latent, another body's frame, that body's command as the target
        cross = F.mse_loss(cross_action, cross_target)
        total = total + cfg.lambda_cross * cross
        parts["cross"] = cross.item()

    if body_pred is not None and body_target is not None and cfg.lambda_body > 0:
        # one head, every embodiment: the only term in this loss that asks the same z to decode
        # the same way on both robots
        body = F.mse_loss(body_pred, body_target)
        total = total + cfg.lambda_body * body
        parts["body"] = body.item()

    if adv_logits is not None and morph_id is not None and cfg.lambda_adv > 0:
        adv = F.cross_entropy(adv_logits, morph_id)
        total = total + cfg.lambda_adv * adv
        parts["adv"] = adv.item()
        parts["adv_accuracy"] = (adv_logits.argmax(dim=-1) == morph_id).float().mean().item()

    # reported before the probe is added: the probe is an instrument, and letting its loss into
    # the number that selects best.pt would pick checkpoints for how badly the probe is doing
    parts["total"] = total.item()

    if probe_logits is not None and morph_id is not None:
        # reads a detached z, so this trains only the probe and leaves the world model untouched
        probe = F.cross_entropy(probe_logits, morph_id)
        total = total + probe
        parts["probe_loss"] = probe.item()
        parts["probe_accuracy"] = (probe_logits.argmax(dim=-1) == morph_id).float().mean().item()

    return total, parts
