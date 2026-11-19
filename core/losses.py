import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadHardDiceLoss(nn.Module):

    def __init__(
        self,
        ce_weight=1.0,
        hard_dice_weight=1.0,
        k=50.0,
        threshold=0.5,
        smooth=1e-06,
        class_weight=None,
    ):
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0,1), got {threshold}")
        if ce_weight < 0 or hard_dice_weight < 0:
            raise ValueError("Loss weights must be nonnegative.")
        self.ce_weight = float(ce_weight)
        self.hard_dice_weight = float(hard_dice_weight)
        self.k = float(k)
        self.threshold = float(threshold)
        self.smooth = float(smooth)
        self.class_weight = class_weight

    def forward(self, logits, targets):
        if isinstance(logits, dict):
            if "body_logits" not in logits:
                raise KeyError("Missing body_logits.")
            logits = logits["body_logits"]
        if logits.ndim != 4 or logits.size(1) != 2:
            raise ValueError(f"Expected logits [B,2,H,W], got {tuple(logits.shape)}")
        if targets.ndim == 4:
            if targets.size(1) != 1:
                raise ValueError(f"Expected targets [B,1,H,W], got {tuple(targets.shape)}")
            targets_ce = targets.squeeze(1).long()
        elif targets.ndim == 3:
            targets_ce = targets.long()
        else:
            raise ValueError(f"Expected targets [B,H,W] or [B,1,H,W], got {tuple(targets.shape)}")
        if self.class_weight is not None:
            class_weight = torch.tensor(self.class_weight, dtype=logits.dtype, device=logits.device)
            ce_loss = F.cross_entropy(logits, targets_ce, weight=class_weight)
        else:
            ce_loss = F.cross_entropy(logits, targets_ce)
        crack_prob = F.softmax(logits, dim=1)[:, 1:2, :, :]
        hard_prob = torch.sigmoid(self.k * (crack_prob - self.threshold))
        targets_float = targets_ce.unsqueeze(1).to(dtype=logits.dtype)
        hard_flat = hard_prob.flatten(start_dim=1)
        target_flat = targets_float.flatten(start_dim=1)
        intersection = (hard_flat * target_flat).sum(dim=1)
        denominator = hard_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        hard_dice_loss = (1.0 - dice).mean()
        total_loss = self.ce_weight * ce_loss + self.hard_dice_weight * hard_dice_loss
        with torch.no_grad():
            ambiguous_ratio = ((crack_prob >= 0.4) & (crack_prob <= 0.6)).float().mean()
            crack_prob_mean = crack_prob.mean()
            hard_prob_mean = hard_prob.mean()
        return (
            total_loss,
            {
                "Hard/CE_Loss": ce_loss.item(),
                "Hard/Approx_Dice_Loss": hard_dice_loss.item(),
                "Hard/Ambiguous_0p4_0p6_Ratio": ambiguous_ratio.item(),
                "Hard/Crack_Prob_Mean": crack_prob_mean.item(),
                "Hard/Hardened_Prob_Mean": hard_prob_mean.item(),
                "total_loss": total_loss.item(),
            },
        )


def get_loss_function(loss_name="single_hard_dice_loss", **kwargs):
    if loss_name != "single_hard_dice_loss":
        raise ValueError(f"Unsupported loss: {loss_name}")
    return SingleHeadHardDiceLoss(**kwargs)
