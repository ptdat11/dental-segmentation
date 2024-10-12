import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

class MulticlassDiceLoss(nn.Module):
    def __init__(
            self, 
            num_classes: int = 14,
            smooth: float = 1.,
            reduction: Literal['mean', 'sum', 'none'] = 'mean',
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor):
        """
        Computes the Dice loss, a measure of set similarity in
        comparing the versus a target which is frequently used in
        evaluating segmentation algorithms.
        Args:
        input: (N, C, H, W)
        target: (N, H, W)
        """
        input = F.softmax(input, dim=-3)
        target = F.one_hot(target, num_classes=self.num_classes).transpose_(-1, -2).transpose_(-2, -3).float()
        sum_dim = (-1, -2)
        
        intersection = 2 * torch.sum(input*target, dim=sum_dim)
        union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        dice = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss
    

class MulticlassFocalLoss(nn.Module):
    def __init__(
            self,
            num_classes: int = 14,
            alpha: float = 0.25,
            gamma: float = 2.,
            reduction: Literal['mean', 'sum', 'none'] = 'mean',
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-7
    
    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor):
        """
        Computes the Focal loss, which is a modification of the standard
        cross-entropy loss to put more focus on hard examples.
        Args:
        input: (N, C, H, W)
        target: (N, H, W)
        """
        target = F.one_hot(target, num_classes=self.num_classes).transpose_(-1, -2).transpose_(-2, -3).float()
        prob = F.softmax(input, dim=-3)
        log_prob = torch.log(prob + self.eps)
        loss = -self.alpha * (1 - prob) ** self.gamma * log_prob * target
        loss = loss.sum(dim=-3)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
    

class MulticlassCombinationLoss(nn.Module):
    def __init__(
            self, 
            num_classes: int = 14,
            w_dice: float = 1.,
            w_focal: float = 1,
            reduction: Literal['mean', 'sum', 'none'] = 'mean',
            dice_kwargs: dict = {'smooth': 1.},
            focal_kwargs: dict = {'alpha': 0.25, 'gamma': 2},
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.dice = MulticlassDiceLoss(num_classes=num_classes, reduction=reduction, **dice_kwargs)
        self.focal = MulticlassFocalLoss(num_classes=num_classes, reduction=reduction, **focal_kwargs)
    
    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor):
        """
        Computes the combination of Dice loss and Focal loss.
        Args:
        input: (N, C, H, W)
        target: (N, H, W)
        """
        dice_loss = self.w_dice * self.dice(input, target)
        focal_loss = self.w_focal * self.focal(input, target)
        loss = dice_loss + focal_loss
        return loss