import os

import clip
import torch
import torch.nn as nn
from models.csp import CSPInterface
from clip_modules.model_loader import load

import numpy as np
from tqdm import tqdm
import re
import pickle
from torch.distributions.beta import Beta
from models.cross_attn import CRAttnBlock, CrossResidualAttentionModel, QuickGELU


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class GenCSP(CSPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_prompt,
        soft_att_obj,
        class_token_ids,
        textdb,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
        vpt=False,
        is_training=True
    ):
        super().__init__(
            clip_model,
            config,
            offset,
            None,
            class_token_ids,
            device=device,
            enable_pos_emb=enable_pos_emb,
            attr_dropout=attr_dropout,
            vpt=vpt,
            is_training=is_training
        )
        self.dim_embed = clip_model.token_embedding.weight.size(-1)
        self.textdb = textdb
        self.context_length = config.context_length
        self.tune_vit = getattr(config, 'tune_vit', False)
        self.use_gauss = getattr(config, 'use_gauss', False)
        self.use_attrobj_gauss = getattr(config, 'use_attrobj_gauss', False)

        self.num_xattn = getattr(config, 'num_xattn', 1)
        self.use_nig = getattr(config, 'use_nig', False)
        self.rewso = getattr(config, 'rewso', False)

        img_dropout = getattr(config, 'img_dropout', 0.0)
        self.img_dropout_layer = nn.Dropout(img_dropout) if img_dropout > 0 else None

        # freeze the clip backbone
        for p in self.parameters():
            p.requires_grad=False
        
        if getattr(config, 'ft_layernorm', False):
            for name, p in self.named_parameters():
                if 'ln' in name:
                    p.requires_grad = True

        # learnable parameters
        self.soft_embeddings = soft_att_obj
        self.soft_prompt = soft_prompt
        self.sow = [0.1, 0.9] if self.rewso else [1, 1]

        prompt_drop = getattr(config, 'prompt_drop', 0.0)
        self.prompt_drop = nn.Dropout(p=prompt_drop) if prompt_drop > 0 else None
        if self.tune_vit:
            for name, param in self.named_parameters():
                if 'visual.transformer.resblocks.23' in name:  # tune the last ViT block
                    param.requires_grad = True

        dropout = getattr(config, 'attn_dropout', 0.)
        fe_arch = getattr(config, 'fe_arch', 'xattn')

        if self.textdb is not None and self.num_xattn > 0:
            if self.num_xattn > 1:
                self.cross_attention = CrossResidualAttentionModel(*[CRAttnBlock(self.dim_embed, self.dim_embed // 64, drop=dropout, return_kv=True) for _ in range(self.num_xattn)]).to(device)
            else:
                self.cross_attention = CRAttnBlock(self.dim_embed, self.dim_embed // 64, drop=dropout).to(device)

        self.with_group = getattr(config, 'with_group', False)
        self.disentangle = getattr(config, 'disentangle', False)
        self.num_aug = getattr(config, 'num_aug', 0)
        if self.with_group or self.disentangle or self.num_aug > 0:
            self.num_attrs = offset
            self.num_objs = soft_att_obj.size(0) - offset
        
        if self.num_aug > 0:
            num_vfe = getattr(config, 'num_vfe', 1) if self.num_xattn == 0 else self.num_xattn
            if fe_arch == 'mlp':
                self.cross_attention_img = CrossResidualAttentionModel(*[CRAttnBlock(self.dim_embed, self.dim_embed // 64, drop=dropout, return_kv=True, with_attn=False, input_mlp_dim=self.num_aug) for _ in range(num_vfe)]).to(device)
            else:
                self.cross_attention_img = CrossResidualAttentionModel(*[CRAttnBlock(self.dim_embed, self.dim_embed // 64, drop=dropout, return_kv=True) for _ in range(num_vfe)]).to(device)

        if self.disentangle:
            disen_arch = getattr(config, 'disen_arch', 'mlp')
            if disen_arch == 'mlp':
                self.mlp_att = nn.Sequential(nn.Linear(self.dim_embed, self.dim_embed * 4), QuickGELU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(self.dim_embed * 4, self.dim_embed)).to(device)
                self.mlp_obj = nn.Sequential(nn.Linear(self.dim_embed, self.dim_embed * 4), QuickGELU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(self.dim_embed * 4, self.dim_embed)).to(device)
            elif disen_arch == 'identity':
                self.mlp_att, self.mlp_obj = nn.Identity(), nn.Identity()
        
        self.no_comp = getattr(config, 'no_comp', False) 
        self.fusion = getattr(config, 'fusion', False)
        self.fusion_factor = getattr(config, 'fusion_factor', 0)
        if isinstance(self.fusion_factor, str) and self.fusion_factor == 'beta':
            beta_prior = getattr(config, 'beta_prior', [1.0, 9.0])  # expectation to be 0.1
            self.fusion_factor = Beta(beta_prior[0], beta_prior[1])

    
    def construct_token_tensors(self, pair_idx):
        """Function creates the token tensor for further inference.

        Args:
            pair_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        dtype = self.clip_model.dtype
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)  # repeat the token_ids of the template 'a photo of x x'
        token_tensor = self.clip_model.token_embedding(
            class_token_ids
        ).type(dtype)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings = self.attr_dropout(self.soft_embeddings)
        # replace with the soft embeddings of attribute
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[
            attr_idx
        ].type(dtype)
        # replace with the soft embeddings of object
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[
            obj_idx + self.offset
        ].type(dtype)

        if self.soft_prompt is not None:
            # adding the correct learnable context
            token_tensor[
                :, 1 : len(self.soft_prompt) + 1, :
            ] = self.soft_prompt.type(self.clip_model.dtype)

        return token_tensor
    

    def construct_multipath_tokens(self):
        dtype, device = self.clip_model.dtype, self.device
        # attribute-aware template tokens
        attr_token = clip.tokenize('a photo of x object',
                                context_length=self.context_length).to(device)
        attr_token_tensor = self.clip_model.token_embedding(
            attr_token.repeat(self.num_attrs, 1)
        ).type(dtype)  # (115, 8, D)
        # object-aware template tokens
        objs_token = clip.tokenize('a photo of x',
                                context_length=self.context_length).to(device)
        objs_token_tensor = self.clip_model.token_embedding(
            objs_token.repeat(self.num_objs, 1) 
        ).type(dtype)  # (245, 8, D)

        # replace with the soft embeddings of attribute and objects
        attr_token_tensor[:, int(attr_token[0].argmax()) - 2, :] = self.soft_embeddings[:self.offset].type(dtype)  # (115, D)
        objs_token_tensor[:, int(objs_token[0].argmax()) - 1, :] = self.soft_embeddings[self.offset:].type(dtype)  # (245, D)

        # adding the correct learnable context
        if self.soft_prompt_attr is not None:
            prompt_ctx = self.prompt_drop(self.soft_prompt_attr) if self.prompt_drop is not None else self.soft_prompt_attr
            attr_token_tensor[
                :, 1 : len(self.soft_prompt_attr) + 1, :
            ] = prompt_ctx.type(self.dtype)

        if self.soft_prompt_objs is not None:
            prompt_ctx = self.prompt_drop(self.soft_prompt_objs) if self.prompt_drop is not None else self.soft_prompt_objs
            objs_token_tensor[
                :, 1 : len(self.soft_prompt_objs) + 1, :
            ] = prompt_ctx.type(self.dtype)
        
        return attr_token_tensor, objs_token_tensor, attr_token, objs_token
    

    def visual(self, x: torch.Tensor, get_feature=False, mode='mixture'):
        x = self.clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        cls2 = x.size(0)

        x_patches = self.clip_model.visual.transformer(x)
        
        x = x_patches.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 0, :]  # ND

        x = self.clip_model.visual.ln_post(x)
        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj

        if get_feature:
            return x, x_patches
        return x, None
    

    def _norm(self, data, dim=-1):
        return data / data.norm(dim=dim, keepdim=True)


    def compute_logits(self, text_feat, img_feat, tau_inv, norm=False, droplayer=None):
        # compute the normalized features
        text_norm = self._norm(text_feat) if norm else text_feat
        img_norm = self._norm(img_feat) if norm else img_feat
        # image feature dropout
        if droplayer is not None:
            img_norm = droplayer(img_norm)
        # inner-product 
        logits = (
            tau_inv
            * img_norm
            @ text_norm.t()
        )  # inner product with temperature scaling, (B, K)
        return logits
    

    def decompose_logits(self, logits, idx):
        att_idx, obj_idx = idx[:, 0], idx[:, 1]
        logits_att = torch.zeros(logits.shape[0], self.num_attrs).to(self.device)
        logits_obj = torch.zeros(logits.shape[0], self.num_objs).to(self.device)
        for i in range(self.num_attrs):
            logits_att[:, i] = logits[:, torch.where(att_idx==i)[0]].mean(-1)
        for i in range(self.num_objs):
            logits_obj[:, i] = logits[:, torch.where(obj_idx==i)[0]].mean(-1)        
        return logits_att, logits_obj
    

    def decompose_texts(self, features, idx):
        """ features: (K, D)
            idx: (K, 2)
        """
        att_idx, obj_idx = idx[:, 0], idx[:, 1]
        dim_feat, dtype = features.size(-1), features.dtype
        feats_att = torch.zeros(self.num_attrs, dim_feat, dtype=dtype).to(self.device)
        feats_obj = torch.zeros(self.num_objs, dim_feat, dtype=dtype).to(self.device)
        for i in range(self.num_attrs):
            feats_att[i, :] = features[torch.where(att_idx == i)[0], :].mean(0)
        for i in range(self.num_objs):
            feats_obj[i, :] = features[torch.where(obj_idx == i)[0], :].mean(0)
        return feats_att, feats_obj


    def decompose_text_base(self, features, idx):
        """ features: (N, K, D)
            idx: (K, 2)
        """
        att_idx, obj_idx = idx[:, 0], idx[:, 1]
        num_samples, dim_feat, dtype = features.size(0), features.size(-1), features.dtype
        feats_att = torch.zeros(num_samples, self.num_attrs, dim_feat, dtype=dtype).to(self.device)
        feats_obj = torch.zeros(num_samples, self.num_objs, dim_feat, dtype=dtype).to(self.device)
        for i in range(self.num_attrs):
            feats_att[:, i, :] = features[:, torch.where(att_idx == i)[0], :].mean(1)
        for i in range(self.num_objs):
            feats_obj[:, i, :] = features[:, torch.where(obj_idx == i)[0], :].mean(1)
        return feats_att, feats_obj

    
    def forward(self, batch_img, idx):
        """ batch_img: (B, 3, 224, 224)
            idx: (K, 2)
        """
        batch_size, num_cls = batch_img.size(0), idx.size(0)
        outputs = {}

        token_tensors = self.construct_token_tensors(idx)
        text_features, text_ft = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
            get_feature=False,
        )

        scale = self.clip_model.logit_scale.exp()
        text_samples = None
        logits_att, logits_obj = None, None
        text_feat_base = None

        if self.textdb is not None:
            # get pairs
            pairs = [(self.textdb['attrs'][attr_id], 
                    self.textdb['objs'][obj_id]) for attr_id, obj_id in idx]
            # generate texts
            text_feat_base = torch.zeros((self.textdb['num_texts'], num_cls, self.dim_embed)).type(self.dtype).to(self.device)
            for c, pair in enumerate(pairs):
                text_feat_base[:, c] = self.textdb['data'][pair]
            
            if self.num_xattn > 0:
                # cross attention
                text_features = self.cross_attention(text_features.type(torch.float), text_feat_base.type(torch.float))
                text_features = text_features.type(self.dtype)  # (K, D)

            if self.use_gauss:
                text_samples = text_features.unsqueeze(1) + text_feat_base.permute(1, 0, 2).contiguous()  # (K, N, D)
        
        if self.use_nig:
            num_texts = getattr(self.config, 'num_texts', 64)
            text_feat_base = torch.rand((num_texts, num_cls, self.dim_embed)).type(self.dtype).to(self.device)
            if self.use_gauss:
                text_samples = text_features.unsqueeze(1) + text_feat_base.permute(1, 0, 2).contiguous()  # (K, N, D)
        
        if self.num_aug > 0:
            # get visual features of all views
            _, num_views, c, h, w = batch_img.size()
            views_feat, img_ft = self.visual(batch_img.view(-1, c, h, w).type(self.dtype), False)   ## (BN, 768), (257, BN, 1024)
            views_feat = views_feat.view(-1, num_views, views_feat.size(-1))
            # cross attention
            visual_res = self.cross_attention_img(views_feat[:, 0].type(torch.float), 
                                                  views_feat[:, 1:].permute(1, 0, 2).contiguous().type(torch.float))
            feat_visual = views_feat[:, 0] + visual_res.type(self.dtype)
        else:
            feat_visual, img_ft = self.visual(batch_img.type(self.dtype), False)   ## (B, 768), (B, 256, 1024)
        
        # aggregate text features of attr & objs
        if not self.no_comp:
            logits = self.compute_logits(text_features, feat_visual, scale, norm=True, droplayer=self.img_dropout_layer)
            outputs.update({'logits': logits})

        if not self.disentangle and self.with_group:
            logits_att, logits_obj = self.decompose_logits(logits, idx)
            outputs.update({
                'logits_attr': logits_att,
                'logits_obj': logits_obj
            })
        
        if self.disentangle:
            # decompose text and visual features of attributes and objects
            text_feat_att, text_feat_obj = self.decompose_texts(text_features, idx)
        
            vis_feat_att = self.mlp_att(feat_visual.type(torch.float))
            vis_feat_obj = self.mlp_obj(feat_visual.type(torch.float))
            # compute cosine similarity logits
            logits_att = self.compute_logits(text_feat_att, vis_feat_att.type(self.dtype), scale, norm=True)
            logits_obj = self.compute_logits(text_feat_obj, vis_feat_obj.type(self.dtype), scale, norm=True)
            outputs.update({
                'logits_attr': logits_att,
                'logits_obj': logits_obj,
                'text_feat_att': text_feat_att,
                'text_feat_obj': text_feat_obj,
                'vis_feat_att': vis_feat_att,
                'vis_feat_obj': vis_feat_obj
            })
            # logits fusion
            if self.fusion and (not self.no_comp):
                att_idx, obj_idx = idx[:, 0], idx[:, 1]  # (K,)
                logits_comp = self.sow[0] * logits_att[:, att_idx] + self.sow[1] * logits_obj[:, obj_idx]  # (B, K)
                factor = float(self.fusion_factor.sample()) if isinstance(self.fusion_factor, Beta) else self.fusion_factor
                outputs['logits'] = (1-factor) * outputs['logits'] + factor * logits_comp
            
            if self.use_attrobj_gauss and text_feat_base is not None:
                attr_feat_base, obj_feat_base = self.decompose_text_base(text_feat_base, idx)
                attr_samples = text_feat_att.unsqueeze(1) + attr_feat_base.permute(1, 0, 2).contiguous()  # (Ka, N, D)
                obj_samples = text_feat_obj.unsqueeze(1) + obj_feat_base.permute(1, 0, 2).contiguous()  # (Ko, N, D)
                outputs.update({
                    'attr_samples_norm': self._norm(attr_samples),
                    'obj_samples_norm': self._norm(obj_samples),
                    'attr_vis_norm': self._norm(vis_feat_att.type(self.dtype)),
                    'obj_vis_norm': self._norm(vis_feat_obj.type(self.dtype)),
                    'tau_inv': scale
                })

        if self.use_gauss and text_feat_base is not None:
            outputs.update({
                'img_norm': self._norm(feat_visual),
                'tau_inv': scale,
                'text_samples_norm': self._norm(text_samples),
                'num_objs': self.num_objs,
                'obj_idx': idx[:, 1]
            })
        return outputs
    

    def set_pairs_group(self, dataset):
        # get pairs
        pairs = torch.tensor([(dataset.attr2idx[attr], dataset.obj2idx[obj])
                                    for attr, obj in dataset.pairs]).to(self.device)
        self.pairs_group = np.array_split(pairs, len(pairs) // self.config.text_encoder_batch_size)


    def enhance_text_features(self, text_rep, pairs_all):
        # get pairs
        num_pairs = pairs_all.size(0)
        pairs = [(self.textdb['attrs'][attr_id], 
                self.textdb['objs'][obj_id]) for attr_id, obj_id in pairs_all]
        # generate texts
        text_feat_base = torch.zeros((self.textdb['num_texts'], num_pairs, self.dim_embed)).type(self.dtype).to(self.device)
        for c, pair in enumerate(pairs):
            text_feat_base[:, c] = self.textdb['data'][pair]
        
        # cross attention
        text_features = self.cross_attention(text_rep.type(torch.float), text_feat_base.type(torch.float))
        text_features = text_features.type(self.dtype)
        return text_features, text_feat_base


    def enhance_text_features_group(self, text_rep):
        ps = 0
        text_features, text_feat_base_all = [], []
        for pair_i in tqdm(self.pairs_group, ncols=0, desc='compute text features'):
            num_pairs = pair_i.size(0)
            pe = ps + num_pairs
            # get pairs
            pairs = [(self.textdb['attrs'][attr_id], 
                    self.textdb['objs'][obj_id]) for attr_id, obj_id in pair_i]
            # generate texts
            text_feat_base = torch.zeros((self.textdb['num_texts'], num_pairs, self.dim_embed)).type(self.dtype).to(self.device)
            for c, pair in enumerate(pairs):
                text_feat_base[:, c] = self.textdb['data'][pair]
            
            # cross attention
            text_features_i = self.cross_attention(text_rep[ps: pe, :].type(torch.float), text_feat_base.type(torch.float))
            text_features.append(text_features_i.type(self.dtype))
            text_feat_base_all.append(text_feat_base)
            ps += num_pairs
        
        text_features = torch.cat(text_features, dim=0)
        text_feat_base_all = torch.cat(text_feat_base_all, dim=1)

        return text_features, text_feat_base_all


    def predict_logits(self, text_rep, dataloader):
        """ Override the default predict logits
            text_rep: unnormalized text features
        """
        self.eval()
        all_attr_gt, all_obj_gt, all_pair_gt = (
            [],
            [],
            [],
        )
        if self.textdb is not None and self.num_xattn > 0:
            all_pairs = torch.concat(self.pairs_group)
            if all_pairs.size(0) > 30000: 
                # for C-GQA dataset, there will be 2.8 million pairs, which needs group-wise computation to save GPU memory
                text_features, text_feat_base = self.enhance_text_features_group(text_rep)
            else:
                text_features, text_feat_base = self.enhance_text_features(text_rep, all_pairs)
        else:
            text_features = text_rep.clone()
        
        if not self.no_comp:
            # divide the text features into group for classification in large semantic space
            classifier_group, pspes = [], []
            ps = 0
            for pair_i in self.pairs_group:
                pe = ps + pair_i.size(0)
                classifier_group.append(text_features[ps: pe, :])
                pspes.append([ps, pe])
                ps += pair_i.size(0)
        
        if self.disentangle and self.fusion:
            all_pairs = torch.concat(self.pairs_group)
            text_feat_att, text_feat_obj = self.decompose_texts(text_features, all_pairs)

        all_logits = torch.Tensor()
        with torch.no_grad():
            for idx, data in tqdm(
                enumerate(dataloader), total=len(dataloader), desc="Testing", ncols=0
            ):
                batch_img = data[0].to(self.device, non_blocking=True)
                scale = self.clip_model.logit_scale.exp()

                if self.num_aug > 0:
                    # get visual features of all views
                    _, num_views, c, h, w = batch_img.size()
                    views_feat, img_ft = self.visual(batch_img.view(-1, c, h, w).type(self.dtype), False)   ## (B, 768), (257, B, 1024)
                    views_feat = views_feat.view(-1, num_views, views_feat.size(-1))
                    # cross attention
                    visual_res = self.cross_attention_img(views_feat[:, 0].type(torch.float), 
                                                        views_feat[:, 1:].permute(1, 0, 2).contiguous().type(torch.float))
                    feat_visual = views_feat[:, 0] + visual_res.type(self.dtype)
                else:
                    feat_visual, img_ft = self.visual(batch_img.type(self.dtype), False)   ## (B, 768), (257, B, 1024)
                
                if not self.no_comp:
                    # for large combinational semantic space, we compute the logits by groups
                    logits = torch.cat([self.compute_logits(classifier, feat_visual, scale, norm=True) 
                                        for classifier in classifier_group], dim=1)
                
                # logits fusion
                if self.disentangle and self.fusion:
                    # decompose visual features of attributes and objects
                    vis_feat_att = self.mlp_att(feat_visual.type(torch.float))
                    vis_feat_obj = self.mlp_obj(feat_visual.type(torch.float))
                    # compute cosine similarity logits
                    logits_att = self.compute_logits(text_feat_att, vis_feat_att.type(self.dtype), scale, norm=True)
                    logits_obj = self.compute_logits(text_feat_obj, vis_feat_obj.type(self.dtype), scale, norm=True)

                    att_idx, obj_idx = all_pairs[:, 0], all_pairs[:, 1]  # (K,)
                    logits_comp = self.sow[0] * logits_att[:, att_idx] + self.sow[1] * logits_obj[:, obj_idx]  # (B, K), a huge matrix due to large K in open space
                    # logits fusion
                    factor = float(self.fusion_factor.mean) if isinstance(self.fusion_factor, Beta) else self.fusion_factor
                    if not self.no_comp:
                        logits = (1-factor) * logits + factor * logits_comp
                    else:
                        logits = logits_comp

                all_logits = torch.cat([all_logits, logits.cpu()], dim=0)
                attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

                all_attr_gt.append(attr_truth)
                all_obj_gt.append(obj_truth)
                all_pair_gt.append(pair_truth)

        all_attr_gt, all_obj_gt, all_pair_gt = (
            torch.cat(all_attr_gt).to("cpu"),
            torch.cat(all_obj_gt).to("cpu"),
            torch.cat(all_pair_gt).to("cpu"),
        )
        return all_logits, all_attr_gt, all_obj_gt, all_pair_gt


def gencsp_init(
    train_dataset,
    config,
    device,
    is_training,
):

    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )
    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))

    soft_att_obj = torch.zeros(
        (len(attributes) + len(classes), clip_model.token_embedding.weight.size(-1)),
    ).to(device)
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    soft_att_obj = nn.Parameter(soft_att_obj).to(device)

    soft_prompt = None
    if getattr(config, 'soft_context', True):
        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init,
                            context_length=config.context_length).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        soft_prompt = nn.Parameter(ctx_vectors).to(device)
    
    token_ids = clip.tokenize('a photo of x x',
                                context_length=config.context_length).to(device)
    offset = len(attributes)
    
    textdb = None
    if getattr(config, 'text_db', None) is not None:
        feat_cache = os.path.join(train_dataset.root, config.text_db[:-4] + '_feat.pkl')  # t5_sentences200_all_pairs_{closed|open}.pkl
        num_texts = getattr(config, 'num_texts', 64)
        assert os.path.exists(feat_cache), "Text feature file does not exist!\n{}".format(feat_cache)
        print("Loading the pre-computed features of text database...")
        if config.text_db.split("_")[0] == 'mistral7b':
            import pickle5 as pickle
        else:
            import pickle
        if num_texts < 64:
            with open(feat_cache, "rb") as f:
                data = pickle.load(f)
            text_features = dict()
            with torch.no_grad():
                for k, v in data.items():
                    text_features[k] = v[:num_texts].clone()
                    del v  # delete this huge tensor
                    torch.cuda.empty_cache()
        else:
            with open(feat_cache, "rb") as f:
                text_features = pickle.load(f)
        textdb = {'data': text_features, 'attrs': attributes, 'objs': classes, 'num_texts': num_texts}

    return (
        clip_model,
        token_ids,
        soft_prompt,
        soft_att_obj,
        offset,
        textdb
    )



def get_gencsp(train_dataset, config, device, is_training=True):

    (
        clip_model,
        token_ids,
        soft_prompt,
        soft_att_obj,
        offset,
        textdb
    ) = gencsp_init(train_dataset, config, device, is_training)

    model = GenCSP(
        clip_model,
        config,
        offset,
        soft_prompt,
        soft_att_obj,
        token_ids,
        textdb,
        device=device,
        enable_pos_emb=True,
        attr_dropout=config.attr_dropout,
        is_training=is_training
    )

    if not is_training:
        return model, None
    
    optim_name = getattr(config, 'optimizer', 'Adam')
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=model.eps
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    return model, optimizer
