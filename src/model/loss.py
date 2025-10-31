import torch
import torch.nn as nn
import torch.nn.functional as F

# A simplified implementation of https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py
# Focal Loss, a modification of cross-entropy loss designed to 
# address class imbalance by focusing on hard-to-classify examples.
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt)**self.gamma

        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = focal_weight * alpha_t * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# A implementation absed on :https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
# Dice loss is used primarily for image segmentation tasks, 
# especially when dealing with imbalanced datasets where one class 
# (like the background or empty voxels in emotion maps in this case) 
# is significantly larger than others.
class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # Frames the 3D predictions into a 2D task
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        # Calculate at the prediction values, (B, C, DHW)
        intersection = (inputs * targets).sum(dim=2)
        union = inputs.sum(dim=2) + targets.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()

# A implementation based on: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
# Commonly used loss for segmentation tasks
class IoULoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (inputs * targets).sum(dim=2)
        total = inputs.sum(dim=2) + targets.sum(dim=2)
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1.0 - iou.mean()


class CombinedSparseLoss(nn.Module):

    def __init__(
        self,
        use_focal=True,
        use_dice=True,
        use_iou=True,
        focal_weight=1.0,
        dice_weight=1.0,
        iou_weight=1.0,
        regression=False,
    ):
        super().__init__()
        self.use_focal = use_focal
        self.use_dice = use_dice
        self.use_iou = use_iou
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

        if use_focal:
            self.focal_loss = FocalLoss(alpha=0.1, gamma=2.0)

        if use_dice:
            self.dice_loss = DiceLoss(smooth=1.0)

        if use_iou:
            self.iou_loss = IoULoss(smooth=1.0)

        self.regression = regression
        if regression:
            self.criterion = F.mse_loss
        else:
            self.criterion = F.binary_cross_entropy_with_logits

    def forward(self, inputs, targets):
        total_loss = 0.0
        losses = {}

        targets = targets.float()

        if self.use_focal:
            focal = self.focal_loss(inputs, targets)
            total_loss += self.focal_weight * focal
            losses['focal'] = focal

        if self.use_dice:
            dice = self.dice_loss(inputs, targets)
            total_loss += self.dice_weight * dice
            losses['dice'] = dice

        if self.use_iou:
            iou = self.iou_loss(inputs, targets)
            total_loss += self.iou_weight * iou
            losses['iou'] = iou

        if self.regression:
            total_loss += self.criterion(inputs, targets) * 0.2
        else:
            total_loss += self.criterion(inputs, targets)

        losses['combined'] = total_loss

        return total_loss, losses

class NonZeroRegularization(nn.Module):

    def __init__(self, weight=1.0, target_sparsity=0.05):
        super().__init__()
        self.weight = weight
        self.target_sparsity = target_sparsity

    def forward(self, predictions):
        probs = torch.sigmoid(predictions)

        # Get the mean of each prediction in the batch
        mean_activation = probs.view(probs.size(0), -1).mean(dim=1)
        target = torch.full_like(mean_activation, self.target_sparsity)
        loss = F.mse_loss(mean_activation, target)

        return self.weight * loss