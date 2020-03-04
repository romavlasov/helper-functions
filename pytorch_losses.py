import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True, **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce)
        loss = self.alpha * (1 - p) ** self.gamma * ce

        if self.reduce:
            return torch.mean(loss)
        return loss
    
    
class BinaryFocalLoss(nn.Module):
    def __init__(self, logits=True, alpha=1, gamma=2, reduce=True, **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.logits = logits
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')
            
        p = torch.exp(-bce)
        loss = self.alpha * (1 - p) ** self.gamma * bce

        if self.reduce:
            return torch.mean(loss)
        return loss
    
 
class CELoss(nn.Module):
    def __init__(self, reduce=True, **kwargs):
        super(CELoss, self).__init__()
        self.reduce = reduce

    def forward(self, inputs, target):
        target = target.long()
        ce = F.cross_entropy(inputs, target, reduction='none')

        if self.reduce:
            return torch.mean(ce)
        return ce


class BCELoss(nn.Module):
    def __init__(self, logits=True, reduce=True, **kwargs):
        super(BCELoss, self).__init__()
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')

        if self.reduce:
            return torch.mean(bce)
        return bce


def focal(*argv, **kwargs):
    return FocalLoss(*argv, **kwargs)


def binary_focal(*argv, **kwargs):
    return BinaryFocalLoss(*argv, **kwargs)


def ce(*argv, **kwargs):
    return CELoss(*argv, **kwargs)


def binary_ce(*argv, **kwargs):
    return BCELoss(*argv, **kwargs)
