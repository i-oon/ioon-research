"""Training objective: L = lambda_recon * L_recon + lambda_motion * L_motion."""
import torch.nn.functional as F


def compute_losses(pred_next, target_next, pred_action, target_action, cfg):
    recon = F.mse_loss(pred_next, target_next)
    motion = F.mse_loss(pred_action, target_action)
    total = cfg.lambda_recon * recon + cfg.lambda_motion * motion
    return total, {"recon": recon.item(), "motion": motion.item(), "total": total.item()}
