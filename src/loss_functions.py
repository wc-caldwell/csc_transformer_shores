import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, nclasses, smooth=1.0, ignore_index=255):
        super().__init__()
        self.nclasses = nclasses
        self.smooth = smooth
        self.ignore_index = ignore_index
 
    def forward(self, logits, target):
        # logits: (B,C,H,W) at model resolution; target: (B,H,W)
        probs = F.softmax(logits.float(), dim=1)
 
        valid_mask = (target != self.ignore_index).unsqueeze(1).float()  # (B,1,H,W)
        target_1h = F.one_hot(
            target.clamp(0, self.nclasses - 1), self.nclasses
        ).permute(0, 3, 1, 2).float()
 
        probs = probs * valid_mask
        target_1h = target_1h * valid_mask
 
        dims = (0, 2, 3)
        inter = (probs * target_1h).sum(dims)
        card  = probs.sum(dims) + target_1h.sum(dims)
        dice  = (2 * inter + self.smooth) / (card + self.smooth)
        return 1 - dice.mean()

class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss (Sudre et al., 2017)
    """

    def __init__(self, nclasses, weight_mode='batch', ema_momentum=0.99, class_volumes=None, smooth=1., ignore_index=255):
        super().__init__()
        self.nclasses = nclasses
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight_mode = weight_mode
        self.ema_momentum = ema_momentum

        if weight_mode == 'fixed':
            if class_volumes is None:
                raise ValueError("weight_mode='fixed' requires class_volumes")
            v = torch.as_tensor(class_volumes, dtype=torch.float32)
            self.register_buffer('volumes', v)
        elif weight_mode == 'ema':
            self.register_buffer('volumes', torch.zeros(nclasses))
            self.register_buffer('initialized', torch.zeros(1))

    def _weights(self, ref_volume):
        w=torch.zeros_like(ref_volume)
        present = ref_volume > 0.
        w[present] = 1. / ref_volume[present].pow(2)
        return w

    def forward(self, logits, target):
        # logits: (B,C,H,W); target: (B,H,W)
        probs = F.softmax(logits.float(), dim=1)

        valid_mask = (target != self.ignore_index).unsqueeze(1).float()
        target_1h = F.one_hot(
            target.clamp(0, self.nclasses - 1), self.nclasses
        ).permute(0, 3, 1, 2).float()

        probs = probs * valid_mask
        target_1h = target_1h * valid_mask

        dims = (0, 2, 3)
        inter = (probs * target_1h).sum(dims)          
        card = probs.sum(dims) + target_1h.sum(dims)   
        batch_volume = target_1h.sum(dims).detach()    

        if self.weight_mode == 'batch':
            ref_volume = batch_volume
        elif self.weight_mode == 'fixed':
            ref_volume = self.volumes
        else:  # 'ema'
            if self.training:
                if self.initialized.item() == 0:
                    self.volumes.copy_(batch_volume)
                    self.initialized.fill_(1)
                else:
                    m = self.ema_momentum
                    self.volumes.mul_(m).add_(batch_volume, alpha=1 - m)
            ref_volume = self.volumes

        w = self._weights(ref_volume)           
        w = w / w.sum().clamp(min=1e-8)       

        num = 2.0 * (w * inter).sum()
        den = (w * card).sum()
        return 1.0 - (num + self.smooth) / (den + self.smooth)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples and underrepresented classes.
    """
    def __init__(self, gamma=2.0, weight=None, ignore_index=255, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
 
    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        # Get probabilities
        pt = torch.exp(-ce_loss)
        # Apply focal term
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            valid_mask = (targets != self.ignore_index)
            return focal_loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def _tversky_index_per_class(self, logits, targets):
        """Returns a (C,) tensor — one Tversky index per class, unreduced."""
        num_classes = logits.shape[1]
        probs = F.softmax(logits.float(), dim=1)
        valid_mask = (targets != self.ignore_index).float()
        targets_one_hot = F.one_hot(
            targets.clamp(0, num_classes - 1), num_classes
        ).permute(0, 3, 1, 2).float()

        tversky_per_class = []
        for c in range(num_classes):
            pred_c = probs[:, c] * valid_mask
            target_c = targets_one_hot[:, c] * valid_mask
            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            fn = ((1 - pred_c) * target_c).sum()
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_per_class.append(tversky)
        return torch.stack(tversky_per_class)   

    def forward(self, logits, targets):
        # unchanged public behavior: average TI first, then 1 - mean
        return 1 - self._tversky_index_per_class(logits, targets).mean()

class FocalTverskyLoss(TverskyLoss):
    def __init__(self, alpha=0.3, beta=0.7, gamma=4/3, smooth=1.0, ignore_index=255):
        super().__init__(alpha, beta, smooth, ignore_index)
        self.gamma = gamma

    def forward(self, logits, targets):
        ti_per_class = self._tversky_index_per_class(logits, targets)    
        base = (1 - ti_per_class).clamp(min=1e-3)  
        focal_per_class = torch.pow(base, (1/self.gamma))           # exponent per class
        return focal_per_class.mean()                                      # then average

class AsymmetricUnifiedFocalLoss(nn.Module):
    '''
    referenced this git repo for pytorch implimentation(s): 
    https://github.com/oikosohn/compound-loss-pytorch/blob/main/unified_focal_loss_pytorch.py

    THOUGH, some of the code does not actually match the publication equations. I have tried to 
    match as best as possible here
    '''
    def __init__(self, nclasses, rare_classes = [1,2], lam = 0.5, delta=0.6, gamma=0.5, eps=1e-8, ignore_index=255, smooth=1.):
        super().__init__()
        self.nclasses = nclasses
        self.lam = lam
        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.rare_classes = rare_classes        # this should be an array/list, something like [1,2]
        self.eps = eps

    def _asymmetric_focal_loss(self, probs, targets_1h, valid_mask):
        probs = probs.clamp(min=self.eps, max=1 - self.eps)
        # probs = probs * valid_mask
        # targets_1h = targets_1h * valid_mask

        ce = -targets_1h * torch.log(probs)

        rare_class_mask = torch.zeros(self.nclasses, dtype=torch.bool, device=probs.device)
        rare_class_mask[self.rare_classes] = True
        rare_class_mask = rare_class_mask.view(1, -1, 1, 1)   # (1, C, 1, 1)

        maF_rare = self.delta * ce * rare_class_mask
        maF_common = (1-self.delta) * torch.pow(1-probs, self.gamma) * ce * ~rare_class_mask

        loss = (maF_rare + maF_common) * valid_mask

        return loss.sum() / valid_mask.sum().clamp(min=1)

    def _asymmetric_focal_tversky_loss(self, probs, targets_1h, valid_mask):
        # this retrieves the batch dimension, the height, and width during each instance to some all TP, FP, and FN
        dims = (0, 2, 3)

        probs = probs * valid_mask
        targets_1h = targets_1h * valid_mask
        tp = (probs * targets_1h).sum(dims)
        fp = (probs * (1 - targets_1h)).sum(dims)
        fn = ((1 - probs) * targets_1h).sum(dims)
        tversky = (tp + self.smooth) / (tp + (1 - self.delta) * fp + self.delta * fn + self.smooth)

        # establish mask for the target rare classes, 
        # in our case it should be 1 and 2 for whitewater and sediment, respectively
        rare_class_mask = torch.zeros(self.nclasses, dtype=torch.bool, device=probs.device)
        rare_class_mask[self.rare_classes] = True
        base = (1-tversky).clamp(min=1e-3)
        per_class = torch.where(rare_class_mask, torch.pow(base, 1-self.gamma), base)

        return per_class.mean()

    def forward(self, logits, targets):
        probs = F.softmax(logits.float(), dim=1)
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()

        targets_1h = F.one_hot(
            targets.clamp(0, self.nclasses - 1), self.nclasses
        ).permute(0, 3, 1, 2).float()
 
        l_focal = self._asymmetric_focal_loss(probs, targets_1h, valid_mask)
        l_focal_tversky = self._asymmetric_focal_tversky_loss(probs, targets_1h, valid_mask)
 
        return self.lam * l_focal + (1 - self.lam) * l_focal_tversky

class ComboLoss(nn.Module):
    """
    Combo Loss as defined by Taghanaki et al. 2019 https://doi.org/10.1016/j.compmedimag.2019.04.005.

    Combination of a modified CCE and Dice loss functions for training multi-class segmentation task on
    imbalanced datasets.

    beta - penalizes FP and FN in the CCE loss. >0.5 -> FN more penalized and vice versa
    alpha - determines the contribution of the modified CCE and Dice losses on the final Combo Loss
    """

    def __init__(self, nclasses, beta=0.6, alpha=0.5, smooth=1.0, ignore_index=255, eps=1e-7):
        super().__init__()
        self.nclasses = nclasses
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.beta = beta
        self.alpha = alpha
        self.eps = eps
        self.dice = DiceLoss(nclasses, smooth=smooth, ignore_index=ignore_index)

    def forward(self, logits, targets):
        probs = F.softmax(logits.float(), dim=1)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()  # (B,1,H,W)
        targets_1h = F.one_hot(
            targets.clamp(0, self.nclasses - 1), self.nclasses
        ).permute(0, 3, 1, 2).float()

        targets_1h = targets_1h * valid_mask
        # out = (self.beta * (targets_1h - torch.log(probs)) + (1-self.beta) * ((1-targets_1h)*torch.log(1-probs)))
        out = (self.beta * targets_1h * torch.log(probs) + (1-self.beta) * (1-targets_1h) * torch.log(1-probs))
        out = out * valid_mask

        l_mcce = -out.sum() / (valid_mask.sum().clamp(min=1.0) * self.nclasses)
        l_dice = self.dice(logits, targets)

        return self.alpha * l_mcce + (1-self.alpha) * (l_dice)

def lovasz_grad(gt_sorted):
    """Gradient of the Lovász extension w.r.t. sorted errors (Berman et al., 2018)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszSoftmaxLoss(nn.Module):
    """
    Multi-class Lovász-Softmax loss.
    Directly optimizes a smooth surrogate of mean IoU rather than approximating
    overlap the way Dice does — tends to help on thin/small foreground classes.
    classes: 'all' (average over every class every step) or 'present'
             (only classes actually present in the batch — usually the better choice).
    """
    def __init__(self, nclasses, ignore_index=255, per_image=True, classes='present'):
        super().__init__()
        self.nclasses = nclasses
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.classes = classes

    def _flatten_probas(self, probs, targets):
        B, C, H, W = probs.shape
        probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
        targets = targets.reshape(-1)
        valid = targets != self.ignore_index
        return probs[valid], targets[valid]

    def _lovasz_softmax_flat(self, probs, targets):
        if probs.numel() == 0:
            return probs.sum() * 0.
        C = probs.shape[1]
        losses = []
        class_range = range(C) if self.classes in ('all', 'present') else self.classes
        for c in class_range:
            fg = (targets == c).float()
            if self.classes == 'present' and fg.sum() == 0:
                continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            grad = lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))
        if len(losses) == 0:
            return probs.sum() * 0.
        return torch.stack(losses).mean()

    def forward(self, logits, targets):
        probs = F.softmax(logits.float(), dim=1)
        if self.per_image:
            losses = []
            for prob_i, target_i in zip(probs, targets):
                p, t = self._flatten_probas(prob_i.unsqueeze(0), target_i.unsqueeze(0))
                losses.append(self._lovasz_softmax_flat(p, t))
            return torch.stack(losses).mean()
        p, t = self._flatten_probas(probs, targets)
        return self._lovasz_softmax_flat(p, t)

# ================================================================================================================
# NOTE: This has not been implemented yet, but could eb a added since one focus could be maintaing edges between classes
# especially for the shoreline specific implementations

class SobelEdgeLoss(nn.Module):
    """Penalizes mismatch between predicted and GT edge maps via Sobel gradients."""
    def __init__(self, nclasses):
        super().__init__()
        self.nclasses = nclasses
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        ky = kx.t()
        self.register_buffer('kx', kx.view(1,1,3,3))
        self.register_buffer('ky', ky.view(1,1,3,3))

    def _sobel(self, x):  # x: (B,1,H,W)
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, logits, target):
        probs = F.softmax(logits.float(), dim=1)
        target_1h = F.one_hot(target, self.nclasses).permute(0,3,1,2).float()
        loss = 0.
        for c in range(self.nclasses):
            pred_edge = self._sobel(probs[:, c:c+1])
            gt_edge   = self._sobel(target_1h[:, c:c+1])
            loss += F.l1_loss(pred_edge, gt_edge)
        return loss / self.nclasses
