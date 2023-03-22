import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss
from .functional import soft_dice_score, to_tensor

SMOOTH = 1
EPS = 1e-7

class DiceLoss(_Loss):
    """ Computes the DiceLoss for a binary task """
    
    def __init__(self, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-7):
        super(DiceLoss, self).__init__()
        
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps
    
    def forward(self, gt, pr):
        
        gt = to_tensor(gt)
        pr = to_tensor(pr)
        
        if self.from_logits:
            pr = F.logsigmoid(pr).exp()
        
        batch_size = gt.size(0)
        dims = (0, 2)
        
        gt = gt.view(batch_size, 1, -1)
        pr = pr.view(batch_size, 1, -1)
        
        if self.ignore_index:
            mask = gt != self.ignore_index
            gt = gt * mask
            pr = pr * mask
        
        scores = self.compute_scores(gt.type(pr.dtype), pr, smooth=self.smooth, eps=self.eps, dims=dims)
        
        loss = -torch.log(scores.clamp_min(self.eps)) if self.log_loss else (1 - scores)
        
        mask = gt.sum(dims) > 0
        loss *= mask.to(loss.dtype)
        
        return self.aggregate_loss(loss)
    
    def aggregate_loss(self, loss):
        return loss.mean()
    
    def compute_scores(self, gt, pr, smooth=SMOOTH, eps=EPS, dims=None):
        return soft_dice_score(gt, pr, smooth, eps, dims)
