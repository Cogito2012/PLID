import torch
import torch.nn as nn
import math
from models.hsic import compute_hsic


class CELoss(nn.Module):
    def __init__(self, margin_cfg=None, group_cfg=None):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.margin_cfg = margin_cfg
        if margin_cfg is not None:
            self.margin_loss = MarginLoss(margin=margin_cfg['margin'])
            self.margin_factor = margin_cfg['factor']
        
        self.group_cfg = group_cfg
        if group_cfg is not None:
            self.group_loss = GroupCELoss(w_attr=group_cfg.get('w_attr', 1.0),
                                          w_obj=group_cfg.get('w_obj', 1.0))

    
    def forward(self, outputs, targets, attr=None, obj=None):
        # get the predictions
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # cross-entropy loss
        loss = self.cross_entropy(logits, targets)
        losses = {'total_loss': loss, 'ce_loss': loss}

        if self.margin_cfg is not None:
            margin_loss = self.margin_loss(outputs['text_norm'], outputs['pair_idx']) * self.margin_factor
            losses['total_loss'] += margin_loss
            losses.update({'margin_loss': margin_loss})
        
        if self.group_cfg is not None:
            attr_loss, obj_loss = self.group_loss(outputs, attr, obj)
            if attr_loss is not None:
                losses['total_loss'] += attr_loss
                losses.update({'attr_loss': attr_loss})
            if obj_loss is not None:
                losses['total_loss'] += obj_loss
                losses.update({'obj_loss': obj_loss})

        return losses


class GroupCELoss(nn.Module):
    def __init__(self, w_attr=1.0, w_obj=1.0):
        super().__init__()
        self.w_attr, self.w_obj = w_attr, w_obj
        if w_attr > 0:
            self.celoss_attr = nn.CrossEntropyLoss()
        if w_obj > 0:
            self.celoss_obj = nn.CrossEntropyLoss()
    

    def _groupping(self, text_norm, labels):
        """ text_norm: (K, D)
            labels: (K,)
        """
        # text_groupped = torch.zeros_like(text_norm, dtype=dtype).to(device)
        unique_cls = torch.unique(labels)
        num_cls, feat_dim = len(unique_cls), text_norm.size(-1)
        device, dtype = text_norm.device, text_norm.dtype

        mean_intra = torch.zeros((num_cls, feat_dim), dtype=dtype).to(device)
        # cov_intra = torch.zeros((num_cls, feat_dim, feat_dim), dtype=dtype).to(device)
        # num_samples = torch.zeros((num_cls,), dtype=dtype).to(device)
        for i, cls in enumerate(unique_cls):
            idx = torch.where(labels == cls)[0]
            samples = text_norm[idx]  # (N, D)
            # num_samples[i] = samples.size(0)
            # get statistics
            mean_intra[i] = samples.mean(dim=0, keepdim=True)  # (1, D)
            # cov_intra[i] = torch.cov(samples.t()) # (x-xm)(x-xm)^T,  (D, D)
        
        # weights = (num_samples / text_norm.size(0)).unsqueeze(1)  # (ncls, 1)
        # mean_inter = (weights * mean_intra).mean(dim=0, keepdim=True)  # (1, D)
        # cov_inter = (weights * (mean_intra - mean_inter)).t() @ (mean_intra - mean_inter)  # (D, D)
        return mean_intra

    
    def forward(self, outputs, attr_target, obj_target):
        """ img_norm: (bs, D)
            text_norm: (K, D)
            pair_idx: (K, 2)
        """
        img_norm = outputs['img_norm']
        text_norm = outputs['text_norm']
        pair_idx = outputs['pair_idx']
        tau_inv = outputs['tau_inv']
        device = img_norm.device

        loss_attr, loss_obj = None, None
        # group the attr-obj pairs by object ID or attribute ID
        if self.w_obj > 0:
            mean_obj = self._groupping(text_norm, pair_idx[:, 1])  # (Ko, D)
            logits_obj = tau_inv * img_norm @ mean_obj.t()  # (bs, Ko)
            loss_obj = self.celoss_obj(logits_obj, obj_target) * self.w_obj
        
        if self.w_attr > 0:
            mean_attr = self._groupping(text_norm, pair_idx[:, 0])  # (Ka, D)
            logits_attr = tau_inv * img_norm @ mean_attr.t()  # (bs, Ko)
            loss_attr = self.celoss_attr(logits_attr, attr_target) * self.w_attr
        
        return loss_attr, loss_obj


class MarginLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin_ = margin
        self.inf = 1e9
    
    def forward(self, text_norm, pair_idx):
        """ text_norm: (K, D)
            pair_idx: (K, 2)
        """
        device, dtype = text_norm.device, text_norm.dtype
        text_groupped = torch.zeros_like(text_norm, dtype=dtype).to(device)
        # group the attr-obj pairs by object ID
        unique_objs = torch.unique(pair_idx[:, 1])
        num_objs = []
        p = 0
        for oid in unique_objs:
            idx = torch.where(pair_idx[:, 1] == oid)[0]
            text_groupped[p: p+len(idx), :] = text_norm[idx]
            p += len(idx)
            num_objs.append(len(idx))

        covmat = text_groupped @ text_groupped.permute(1, 0) + torch.eye(pair_idx.size(0), dtype=dtype).to(device) * self.inf # (K, K)

        p, loss = 0, 0
        for n in num_objs:
            score_min = covmat[p: p+n, p: p+n].min()
            score_max = torch.cat([covmat[p: p+n, :p], covmat[p: p+n, p+n:]], dim=1).max()
            loss += max(0, -(score_min - score_max) + self.margin_)  # assume that score_min > score_max + margin
            p += n
        loss /= len(num_objs)

        return loss


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


class ProDALoss(nn.Module):
    def __init__(self, lambda_=0.1, margin_cfg=None, group_gauss=False):
        super().__init__()
        self.lambda_ = lambda_
        self.margin_cfg = margin_cfg
        self.group_gauss = group_gauss
        self.cross_entropy = nn.CrossEntropyLoss()

        if margin_cfg is not None:
            self.margin_loss = MarginLoss(margin=margin_cfg['margin'])
            self.margin_factor = margin_cfg['factor']
    

    def forward(self, outputs, labels, attr=None, obj=None):
        """ outputs: dict()
            labels: (bs,)
        """
        logits = outputs['logits']  # (bs, K)
        img_norm = outputs['img_norm']  # (bs, D)
        text_norm = outputs['text_norm'] # (K, B, D)
        tau_inv = outputs['tau_inv']
        prompt_norm = outputs.get('prompt_norm', None)  # (B, D)

        if self.group_gauss:
            logits_new = upperbound_group_logits(logits, text_norm, img_norm, labels, tau_inv, outputs['num_objs'], outputs['obj_idx'])
        else:
            logits_new = upperbound_logits(logits, text_norm, img_norm, labels, tau_inv)
        loss_ce = self.cross_entropy(logits_new, labels)
        
        losses = {'total_loss': loss_ce.clone(), 'ce_loss': loss_ce}

        if prompt_norm is not None:
            # regularization term: semantic orthogonality loss
            dis = prompt_norm @ prompt_norm.permute(1, 0)  # (B, B)
            loss_so = dis[~torch.eye(dis.size(0), dtype=torch.bool, device=prompt_norm.device)].abs().mean()
            losses['total_loss'] += self.lambda_ * loss_so
            losses.update({'so_loss': self.lambda_ * loss_so})
        
        if self.margin_cfg is not None:
            margin_loss = self.margin_loss(text_norm.mean(dim=1), outputs['pair_idx']) * self.margin_factor
            losses['total_loss'] += margin_loss
            losses.update({'margin_loss': margin_loss})

        return losses


class CVAELoss(nn.Module):
    def __init__(self, recon_method='mse', cond_mode='t2i', logits_fusion=False, loss_recon=1.0, loss_kld=0.1):
        super().__init__()
        self.recon_method = recon_method
        self.cond_mode = cond_mode
        self.logits_fusion = logits_fusion
        self.weight_recon = loss_recon
        self.weight_kld = loss_kld
        # loss functions
        self.cross_entropy = nn.CrossEntropyLoss()
        if self.recon_method == 'mse':
            self.recon_loss = nn.MSELoss(reduction='none')
        # self.momentum = momentum
        # self.buffer_size = buffer_size
        # self.buffer = {k: torch.tensor([], dtype=dtype).to(device) for k in range(num_cls)}
        # self.mvn_params = {k: {'mean': torch.zeros((dim_latent), dtype=dtype).to(device), 
        #                        'logvar': torch.zeros((dim_latent), dtype=dtype).to(device),
        #                        } for k in range(num_cls)}  # mean and variance
        # self.factor = factor
        # self.eps = 1e-4 if dtype == torch.float16 else 1e-8
    

    def running_stat(self, samples, targets):
        unique_cls = torch.unique(targets)
        for cls in unique_cls:
            cls = int(cls)
            subset = samples[targets == cls]
            if subset.size(0) == 0:
                continue  # no samples found
            self.buffer[cls] = torch.concat((self.buffer[cls], subset), dim=0)[-self.buffer_size:]
            if self.buffer[cls].size(0) < self.buffer_size:  # buffer is not full
                continue  # continue collecting samples
            # compute statistics
            mean = self.buffer[cls].mean(dim=0)  # (D)
            mean_sq = (self.buffer[cls] ** 2).mean(dim=0)  # (D)
            var = mean_sq - mean ** 2
            # exponential averaging
            self.mvn_params[cls]['mean'] = self.momentum * self.mvn_params[cls]['mean'] + (1 - self.momentum) * mean
            self.mvn_params[cls]['logvar'] = self.momentum * self.mvn_params[cls]['logvar'] + (1 - self.momentum) * (var + self.eps).log()
    

    def cosine_logits(self, x_recon, x_norm, tau_inv=1.0):
        """ x_recon: (B, K, D) or (B, D)
            x_norm: (B, D) or (K, D)
            targets: (B)
        """
        xr_norm = x_recon / x_recon.norm(dim=-1, keepdim=True)
        if self.cond_mode == 't2i': # inner-product 
            logits = tau_inv * torch.bmm(xr_norm, x_norm.unsqueeze(-1)).squeeze(-1)  # (B,K)
        else:
            logits = tau_inv * xr_norm @ x_norm.t()  # (B,K)
        return logits


    def compute_kl(self, mu_q, logvar_q, mu_p=None, logvar_p=None, keepdim=False):
        if (mu_p is None) or (logvar_p is None):
            dtype, device = mu_q.dtype, mu_q.device
            mu_p = torch.zeros_like(mu_q, dtype=dtype).to(device)
            logvar_p = torch.zeros_like(logvar_q, dtype=dtype).to(device)
        term1 = logvar_p - logvar_q - 1
        term2 = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        kld = 0.5 * torch.sum(term1 + term2, dim=-1, keepdim=keepdim)
        return kld


    def forward(self, outputs, targets, attr=None, obj=None):
        # get the predictions
        logits = outputs['logits']  # (B, K)
        recon = outputs['recon']  # (B, K, D)
        src = outputs['src']  # (B, D) or (K, D)
        z_mean = outputs['z_mean']  # (B, K, d)
        z_logvar = outputs['z_logvar']  # (B, K, d)
        tau_inv = outputs['tau_inv'] if self.logits_fusion else 1.0  # scalar

        batch_size, num_cls = logits.size()[:2]
        inds = torch.arange(batch_size).to(logits.device)
        losses = {'total_loss': 0}

        # reconstruction loss
        if self.recon_method == 'mse':
            recon_target = src if self.cond_mode == 't2i' else src[targets]
            recon_loss = self.recon_loss(recon[inds, targets], recon_target).sum(dim=-1).mean()
        elif self.recon_method == 'cos':
            recon_in = recon if self.cond_mode == 't2i' else recon[inds, targets]
            vae_logits = self.cosine_logits(recon_in, src, tau_inv=tau_inv)
            recon_loss = self.cross_entropy(vae_logits, targets)
        
        losses['total_loss'] += recon_loss * self.weight_recon
        losses.update({'recon_loss': recon_loss})

        # KL divergence loss
        kl_loss = torch.mean(self.compute_kl(z_mean[inds, targets], z_logvar[inds, targets]), dim=0)
        losses['total_loss'] += kl_loss * self.weight_kld
        losses.update({'kl_loss': kl_loss})

        # cross-entropy loss
        if self.logits_fusion:
            logits = (logits + vae_logits) * 0.5
        ce_loss = self.cross_entropy(logits, targets)
        losses['total_loss'] += ce_loss
        losses.update({'ce_loss': ce_loss})

        # get running mean and logvar
        # self.running_stat(samples_img, targets)

        # # compute the KL divergence loss
        # unique_cls = torch.unique(targets)
        # loss_kl, num = 0, 0
        # for cls in unique_cls:
        #     cls = int(cls)
        #     if self.buffer[cls].size(0) == self.buffer_size:
        #         loss_kl += self.compute_kl(mean_text[cls], logvar_text[cls],
        #                                    self.mvn_params[cls]['mean'], self.mvn_params[cls]['logvar'])
        #         num += 1
        # if num > 0:
        #     loss_kl = (loss_kl / num ) * self.factor
        #     losses['total_loss'] += loss_kl
        #     losses.update({'kl_loss': loss_kl})

        return losses


class AdaMarginLoss(nn.Module):
    def __init__(self, m=0.4, h=0.333, s=64, t_alpha=0.01, tune_temp=False, eps=1e-4):
        super(AdaMarginLoss, self).__init__()
        self.m = m 
        self.h = h
        self.s = s if not tune_temp else None
        self.eps = eps

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.tensor([20.]))
        self.register_buffer('batch_std', torch.tensor([1.]))

        self.cross_entropy = nn.CrossEntropyLoss()
    

    def rescale_logits(self, img_norm, text_norm, norms, targets):
        """ img_norm: normalized image embeddings
            text_norm: normalized text embeddings
            norms: the L2-norm of image embeddings
            targets: classification targets
        """
        cosine = torch.mm(img_norm, text_norm.t())
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
        
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # g_angular
        m_arc = torch.zeros(targets.size()[0], cosine.size()[1], dtype=cosine.dtype).to(cosine.device)
        m_arc.scatter_(1, targets.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(targets.size()[0], cosine.size()[1], dtype=cosine.dtype).to(cosine.device)
        m_cos.scatter_(1, targets.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_logits = cosine * self.s
        return scaled_logits
    

    def forward(self, outputs, targets, attr=None, obj=None):
        """ outputs: dict()
            labels: (bs,)
        """
        img_norm = outputs['img_norm']  # (B, D)
        text_norm = outputs['text_norm'] # (K, D)
        norms = outputs['norms']  # (B, 1)
        self.s = outputs['tau_inv']

        # rescaling the logits
        scaled_logits = self.rescale_logits(img_norm, text_norm, norms, targets)
        
        # cross-entropy loss
        loss = self.cross_entropy(scaled_logits, targets)
        losses = {'total_loss': loss, 'ce_loss': loss.clone()}
        
        return losses


class DFSPLoss(nn.Module):
    def __init__(self, group_cfg, w_sp=0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.w_attr = group_cfg.get('w_attr', 0.01)
        self.w_obj = group_cfg.get('w_obj', 0.01)
        self.w_sp = w_sp

    
    def forward(self, outputs, targets, attr=None, obj=None):
        logits = outputs['logits']
        logits_att = outputs['logits_att']
        logits_obj = outputs['logits_obj']
        logits_soft_prompt = outputs['logits_soft_prompt']

        # cross-entropy functions
        loss_logit_df = self.cross_entropy(logits, targets)
        loss_logit_sp = self.cross_entropy(logits_soft_prompt, targets)
        loss_att = self.cross_entropy(logits_att, attr)
        loss_obj = self.cross_entropy(logits_obj, obj)

        loss_total = loss_logit_df + self.w_attr * loss_att + self.w_obj * loss_obj + self.w_sp * loss_logit_sp

        losses = {'total_loss': loss_total, 
                  'ce_loss': loss_logit_df,
                  'sp_loss': loss_logit_sp,
                  'attr_loss': loss_att,
                  'obj_loss': loss_obj}

        return losses



class HyperCSPLoss(nn.Module):
    def __init__(self, reg=0.5):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.reg = reg

    
    def forward(self, outputs, targets, attr=None, obj=None):
        # get the predictions
        logits_e = outputs['logits_e']
        logits_h = outputs['logits_h']

        # cross-entropy loss
        loss_e = self.cross_entropy(logits_e, targets)
        loss_h = self.cross_entropy(logits_h, targets)
        loss = self.reg * loss_e + (1-self.reg) * loss_h

        losses = {'total_loss': loss, 
                  'ce_loss': loss.clone(),
                  'h_loss': loss_h,
                  'e_loss': loss_e}

        return losses


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