import torch
import torch.nn as nn
import math
from models.hsic import compute_hsic


def decompose_texts(features, num_objs, obj_idx):
    """ features: (K, B, D)
        num_objs: ()
        obj_idx: (K,)
        return: (num_objs, B, D)
    """
    dtype, device = features.dtype, features.device
    dim_feat, num_samples = features.size(-1), features.size(1)
    feats_obj = torch.zeros((num_objs, num_samples, dim_feat), dtype=dtype).to(device)
    for i in range(num_objs):
        feats_obj[i, :, :] = features[torch.where(obj_idx == i)[0], :, :].mean(0)
    return feats_obj


def upperbound_logits(logits, text_norm, img_norm, labels, scale):
    """ logits: (bs, K), the logits of mean weights
        text_norm: (K, B, D)
        img_norm: (bs, D)
        scale: ()
    """
    n_class, n_prompt, embed_dim = text_norm.size()
    batch_size = logits.size(0)

    # compute variance and covariance
    _text_ctr_feat = text_norm - text_norm.mean(dim=1, keepdim=True)  # centralize
    diag_cov_martix = _text_ctr_feat.permute(2, 0, 1) @ _text_ctr_feat.permute(2, 1, 0)  # (D, K, K)
    diag_cov_martix /= n_prompt + 1
    refined_logits = torch.einsum("bd, dik -> bik", [img_norm**2, diag_cov_martix])  # (bs, K, K)
    sigma_ii = refined_logits[torch.arange(batch_size), labels, labels].unsqueeze(-1)  # (bs, 1)
    sigma_jj = refined_logits[:, torch.arange(n_class), torch.arange(n_class)]         # (bs, K)
    sigma_ij = refined_logits[torch.arange(batch_size), labels, :]  # (bs, K)
    sigma = sigma_ii + sigma_jj - 2 * sigma_ij

    logits_new = logits + 0.5 * (scale**2) * sigma
    return logits_new


def upperbound_group_logits(logits, text_norm, img_norm, labels, scale, num_objs, obj_idx):
    """ logits: (bs, K), the logits of mean weights
        text_norm: (K, B, D)
        img_norm: (bs, D)
        labels: (bs,)
        scale, num_objs: ()
        obj_idx: (K,)
    """
    # group text embeddings by object IDs
    text_grouped = decompose_texts(text_norm, num_objs, obj_idx)  # (K,B,D) --> (Ko,B,D)
    n_class, n_prompt, embed_dim = text_grouped.size()
    batch_size = logits.size(0)
    # group labels of (s,o) pair by object IDs
    labels_grouped = obj_idx[labels]  # (bs,), values range from 0 to Ko-1

    # compute variance and covariance
    _text_ctr_feat = text_grouped - text_grouped.mean(dim=1, keepdim=True)  # centralize
    diag_cov_martix = _text_ctr_feat.permute(2, 0, 1) @ _text_ctr_feat.permute(2, 1, 0)  # (D, Ko, Ko)
    diag_cov_martix /= n_prompt + 1
    refined_logits = torch.einsum("bd, dik -> bik", [img_norm**2, diag_cov_martix])  # (bs, Ko, Ko)
    sigma_ii = refined_logits[torch.arange(batch_size), labels_grouped, labels_grouped].unsqueeze(-1)  # (bs, 1)
    sigma_jj = refined_logits[:, torch.arange(n_class), torch.arange(n_class)]         # (bs, Ko)
    sigma_ij = refined_logits[torch.arange(batch_size), labels_grouped, :]  # (bs, Ko)
    sigma = sigma_ii + sigma_jj - 2 * sigma_ij  # (bs, Ko)

    #  de-group the sigma to (bs, K)
    sigma = sigma[:, obj_idx]  # scatter the sigma values of each object to the corresponding pairs

    logits_new = logits + 0.5 * (scale**2) * sigma
    return logits_new



class PartialLabelSmoothing(nn.CrossEntropyLoss):
    def __init__(self, ls: float = 0, **kwargs):
        super().__init__(**kwargs)
        """
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method
        """
        assert 0 <= ls < 1
        self.ls = ls
        self.comp_pairs = None  # (K, 2), initialized in train.py

    
    def smooth_one_hot(self, true_labels: torch.Tensor, smooth_mask: torch.Tensor):
        """ partially smooth the labels by mask
            true_labels: (B,), values range from 0 to K-1
            smooth_mask: (B, K)
        """
        confidence = 1.0 - self.ls
        total_class = smooth_mask.size(-1)
        with torch.no_grad():
            nums_smothclass = torch.sum(smooth_mask, dim=-1, keepdim=True)  # (B, 1)
            batch_ls = torch.where(nums_smothclass > 1, self.ls, 0.0)  # if only one attribute, no need to smooth
            true_dist = batch_ls / (nums_smothclass - 1 + 1e-8).repeat(1, total_class)  # (B, K)
            true_dist = true_dist * smooth_mask
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
        return true_dist
    
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """ input: (B, K), the logits
            target: (B,), the target labels ranging from 0 to K-1
        """
        batch_size, total_class = input.size()
        
        # compute the smoothing mask
        target_attr = self.comp_pairs[target, 0].unsqueeze(1).repeat(1, total_class)  # (B, K)
        all_attr = self.comp_pairs[:, 0].unsqueeze(0).repeat(batch_size, 1)  # (B, K)
        smooth_mask = torch.where(all_attr == target_attr, 1.0, 0.0)

        # Convert labels to distributions
        smooth_labels = self.smooth_one_hot(target, smooth_mask)
        preds = input.log_softmax(dim=-1)
        return torch.mean(torch.sum(-smooth_labels * preds, dim=-1))


class GenCSPLoss(nn.Module):
    def __init__(self, use_gauss=False, use_attrobj_gauss=False, group_gauss=False, group_cfg=None, disentangle=False, ls=0.0, partial_smooth=False):
        super().__init__()
        self.partial_smooth = partial_smooth
        self.cross_entropy_ao = PartialLabelSmoothing(ls=ls) if partial_smooth else nn.CrossEntropyLoss(label_smoothing=ls)
        self.group_cfg = group_cfg
        if group_cfg is not None:
            self.cross_entropy_group = nn.CrossEntropyLoss()
            self.w_attr = group_cfg.get('w_attr', 1.0)
            self.w_obj = group_cfg.get('w_obj', 1.0)
        
        self.disentangle = disentangle
        if self.disentangle:
            self.w_indep = group_cfg.get('w_indep', [0.0, 0.0])
        
        self.use_gauss = use_gauss
        self.use_attrobj_gauss = use_attrobj_gauss
        self.group_gauss = group_gauss
    

    def geometric_consistency(self, text_norm, img_norm, labels, scale):
        """ text_norm: (K, D)
            img_norm: (B, D)
            labels: (B,)
        """
        batch_size = img_norm.size(0)
        text_embed = text_norm[labels]  # (B, D)

        logits_vv = scale * img_norm @ img_norm.t()  # (B, B)
        logits_tt = scale * text_embed @ text_embed.t()  # (B, B)
        loss_inmodal = (logits_vv - logits_tt).square().mean() / (scale**2) * batch_size

        logits_vt = scale * img_norm @ text_embed.t()
        loss_crossmodal = (logits_vt - logits_vt.t()).square().mean() / (scale**2) * batch_size

        return loss_inmodal, loss_crossmodal

    
    def forward(self, outputs, targets, attr=None, obj=None):
        if 'logits' in outputs:
            logits = outputs['logits']
            if self.use_gauss:
                if self.group_gauss:
                    logits = upperbound_group_logits(logits.clone(), outputs['text_samples_norm'], outputs['img_norm'], targets, outputs['tau_inv'], outputs['num_objs'], outputs['obj_idx'])
                else:
                    # logits of upper bound cross-entropy loss
                    logits = upperbound_logits(logits.clone(), outputs['text_samples_norm'], outputs['img_norm'], targets, outputs['tau_inv'])
            
            # cross-entropy functions
            loss = self.cross_entropy_ao(logits, targets)
            losses = {'ce_loss': loss,
                    'total_loss': loss.clone()}
        else:
            losses = {'total_loss': 0}

        if self.group_cfg is not None and self.w_obj > 0 or self.disentangle:
            logits_obj = outputs['logits_obj']
            if len(logits_obj.size()) == 3:
                num_cls = logits_obj.size(-1)
                target_obj = obj.unsqueeze(-1).repeat(1, logits_obj.size(1)).view(-1)
                logits_obj = logits_obj.view(-1, num_cls)
            else:
                target_obj = obj
            if self.use_attrobj_gauss:
                logits_obj = upperbound_logits(logits_obj.clone(), outputs['obj_samples_norm'], outputs['obj_vis_norm'], target_obj, outputs['tau_inv'])
            loss_obj = self.cross_entropy_group(logits_obj, target_obj)
            # update total loss
            losses['total_loss'] += self.w_obj * loss_obj
            losses.update({
                'obj_loss': loss_obj
            })
        
        if self.group_cfg is not None and self.w_attr > 0 or self.disentangle:
            attr_logits = outputs['logits_attr']
            if self.use_attrobj_gauss:
                attr_logits = upperbound_logits(attr_logits.clone(), outputs['attr_samples_norm'], outputs['attr_vis_norm'], attr, outputs['tau_inv'])
            loss_att = self.cross_entropy_group(attr_logits, attr)
            losses['total_loss'] += self.w_attr * loss_att
            losses.update({
                'attr_loss': loss_att
            })
        
        if self.disentangle:
            independency_loss = 0
            # independency for text embeddings
            if self.w_indep[0] > 0:
                indep_text = compute_hsic(outputs['text_feat_att'][attr, :], outputs['text_feat_obj'][obj, :], sigmaX=1, sigmaY=1, norm=True)
                losses.update({
                    'indep_text_loss': indep_text
                })
                independency_loss += self.w_indep[0] * indep_text
            # independency for img embeddings
            if self.w_indep[1] > 0:
                indep_img = compute_hsic(outputs['vis_feat_att'], outputs['vis_feat_obj'], sigmaX=1, sigmaY=1, norm=True)
                losses.update({
                    'indep_img_loss': indep_img
                })
                independency_loss += self.w_indep[1] * indep_img
            # update total loss
            if independency_loss > 0:
                losses['total_loss'] += independency_loss

        return losses