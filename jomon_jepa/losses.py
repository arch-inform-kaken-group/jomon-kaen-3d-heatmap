from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def vicreg_loss(z_pred: torch.Tensor, z_target: torch.Tensor, sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.04) -> torch.Tensor:
    """
    JEPA latent prediction loss with Variance-Invariance-Covariance regularization
    to prevent representation collapse on small datasets.
    """
    sim_loss = F.smooth_l1_loss(z_pred, z_target)

    if z_pred.shape[0] < 2:
        return sim_coeff * sim_loss

    z_p = z_pred - z_pred.mean(dim=0)
    std_p = torch.sqrt(z_p.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1.0 - std_p))

    B, D = z_p.shape
    cov_p = (z_p.T @ z_p) / (B - 1)
    off_diag = cov_p.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
    cov_loss = off_diag.pow(2).sum() / D

    return sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss


def gaze_saliency_loss(gaze_logits: torch.Tensor, gaze_target: torch.Tensor) -> torch.Tensor:
    """
    Combines KL-Divergence (distribution matching) + (1 - Pearson CC) + L1
    over the N surface points of each pottery vessel.
    """
    B, N, _ = gaze_logits.shape
    pred_scores = gaze_logits.squeeze(-1)  # [B, N]
    tgt = gaze_target.squeeze(-1)          # [B, N]

    # 1. KL Divergence over normalized spatial probability distributions
    log_p = F.log_softmax(pred_scores, dim=-1)
    q = tgt / (tgt.sum(dim=-1, keepdim=True) + 1e-6)
    kl = F.kl_div(log_p, q, reduction="batchmean")

    # 2. Pearson Correlation Coefficient (CC)
    pred_prob = torch.sigmoid(pred_scores)
    vp = pred_prob - pred_prob.mean(dim=-1, keepdim=True)
    vt = tgt - tgt.mean(dim=-1, keepdim=True)
    cc = (vp * vt).sum(dim=-1) / (
        torch.sqrt((vp ** 2).sum(dim=-1) * (vt ** 2).sum(dim=-1)) + 1e-6
    )
    cc_loss = (1.0 - cc).mean()

    # 3. Point-wise L1
    l1 = F.l1_loss(pred_prob, tgt)

    return kl + cc_loss + 0.5 * l1


def emotion_focal_dice_loss(emo_logits: torch.Tensor, emo_target: torch.Tensor, alpha=0.25, gamma=2.0) -> torch.Tensor:
    """Focal + Soft Dice loss across the 5 sparse emotion channels."""
    bce = F.binary_cross_entropy_with_logits(emo_logits, emo_target, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = emo_target * alpha + (1.0 - emo_target) * (1.0 - alpha)
    focal = (alpha_t * ((1.0 - pt) ** gamma) * bce).mean()

    probs = torch.sigmoid(emo_logits)
    inter = (probs * emo_target).sum(dim=1)
    union = probs.sum(dim=1) + emo_target.sum(dim=1)
    dice = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0)).mean()

    return focal + dice


class MultimodalJEPACriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        total = torch.tensor(0.0, device=batch["coord"].device)
        logs = {}
        jepa_out = outputs["jepa"]

        # 1. JEPA Latent Alignment Losses
        jepa_loss = torch.tensor(0.0, device=total.device)
        if "target_global_spatial" in outputs:
            jepa_loss = jepa_loss + vicreg_loss(
                jepa_out["pred_global_spatial"],
                outputs["target_global_spatial"],
                cov_coeff=self.cfg.w_vicreg,
            )
        if "target_global_concept" in outputs:
            jepa_loss = jepa_loss + vicreg_loss(
                jepa_out["pred_global_concept"],
                outputs["target_global_concept"],
                cov_coeff=self.cfg.w_vicreg,
            )
        total = total + self.cfg.w_jepa * jepa_loss
        logs["jepa_loss"] = jepa_loss

        # 2. Spatial Gaze Loss
        if "gaze_logits" in outputs:
            g_loss = gaze_saliency_loss(outputs["gaze_logits"], batch["gaze"])
            total = total + self.cfg.w_gaze * g_loss
            logs["gaze_loss"] = g_loss

        # 3. 5-Emotion Map Loss
        if "emotion_logits" in outputs:
            e_loss = emotion_focal_dice_loss(outputs["emotion_logits"], batch["emotion"])
            total = total + self.cfg.w_emotion * e_loss
            logs["emotion_loss"] = e_loss

        # 4. Text / Next-Concept Loss
        if "text_loss" in outputs:
            t_loss = outputs["text_loss"]
            total = total + self.cfg.w_text * t_loss
            logs["text_loss"] = t_loss

        logs["total_loss"] = total
        return logs