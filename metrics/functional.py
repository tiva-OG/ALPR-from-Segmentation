import torch
import numpy as np

SMOOTH = 1.0

def soft_dice_score(gt, pr, smooth=SMOOTH, eps=1e-7, dim=None):
    """ Calculates dice score for prediction """
    
    assert gt.size() == pr.size()
    
    intersection = torch.sum(gt*pr, dim=dim)
    cardinality = torch.sum(gt+pr, dim=dim)
    
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    
    return dice_score


def to_tensor(x, dtype=None):
    """ Converts `x` to torch.Tensor and type `dtype` """
    
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
    
    x = x.type(dtype) if dtype is not None else x
    
    return x

def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1-tl1, br2-tl2
    assert (wh1 >= 0).all() and (wh2 >= 0).all()
    
    intersection_wh = np.maximum(np.minimum(br1 ,br2)-np.maximum(tl1, tl2), 0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    
    return intersection_area/union_area

def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())

def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    
    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break
        
        if non_overlap:
            SelectedLabels.append(label)
    
    return SelectedLabels

