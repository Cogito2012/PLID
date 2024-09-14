import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from models.csp import CSPInterface
from clip_modules.model_loader import load
from models.hyptorch.nn import ToPoincare, FromPoincare, HypLinear
from models.hyptorch.pmath import dist_matrix
import numpy as np
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.realpath(__file__))



class HyperCSP(CSPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_prompt,
        soft_att_obj,
        class_token_ids,
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
            soft_att_obj,
            class_token_ids,
            device,
            enable_pos_emb,
            attr_dropout,
            vpt,
            is_training
        )
        self.dim_in = clip_model.token_embedding.weight.size(-1)

        # freeze the clip backbone
        for p in self.parameters():
            p.requires_grad=False

        # additional learnable parameters
        self.soft_prompt = soft_prompt

        # ======= HypViT ========
        # riemannian: False
        # dim_ball: 128
        # curvature: 0.1
        # clip_r: 2.3
        # tau: 0.2
        # ======= HNN (MiniImageNet) ========
        # riemannian: True
        # dim_ball: 512
        # curvature: 0.01
        # clip_r: None
        self.riemannian = getattr(config, 'riemannian', True)
        self.curvature = getattr(config, 'curvature', 1.0)
        self.clip_r = getattr(config, 'clip_r', None)

        reduce_dim = getattr(config, 'reduce_dim', False)
        if reduce_dim:
            self.dim_ball = getattr(config, 'dim_ball', 64)
            e_text = nn.Linear(self.dim_in, self.dim_ball)    # 768 --> 256
            e_img = nn.Linear(self.dim_in, self.dim_ball)
            h_text = HypLinear(self.dim_ball, self.dim_in, c=self.curvature)  # 256 --> 768
            h_img = HypLinear(self.dim_ball, self.dim_in, c=self.curvature)
        else:
            self.dim_ball = self.dim_in
            e_text, e_img = nn.Identity(), nn.Identity()
            h_text, h_img = nn.Identity(), nn.Identity()
        to_poincare = ToPoincare(c=self.curvature,  # hyperbolic c, "0" enables sphere mode
                       ball_dim=self.dim_ball,
                       riemannian=self.riemannian,
                       clip_r=self.clip_r # feature clipping radius
                       )  # Eucleadian to Hyperbolic
        to_euclidean = FromPoincare(c=self.curvature, ball_dim=self.dim_ball)  # Hyperbolic to Eucleadian
        # define layers
        self.e2h_text = nn.Sequential(e_text, to_poincare).to(self.device)
        self.e2h_img = nn.Sequential(e_img, to_poincare).to(self.device)
        self.h2e_text = nn.Sequential(h_text, to_euclidean).to(self.device)  
        self.h2e_img = nn.Sequential(h_img, to_euclidean).to(self.device)  
        self.wf = getattr(config, 'wf', 0.)
        self.tau_h = torch.tensor(config.tau)
        self.tau_e = 1.0 / self.clip_model.logit_scale.exp()


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
            ] = self.soft_prompt.type(self.dtype)

        return token_tensor
    
    
    def _norm(self, data, dim=-1):
        return data / data.norm(dim=dim, keepdim=True)
    

    def compute_logits(self, text_feat, img_feat, tau_inv, norm=False):
        # compute the normalized features
        text_norm = self._norm(text_feat) if norm else text_feat
        img_norm = self._norm(img_feat) if norm else img_feat
        # inner-product 
        logits = (
            tau_inv
            * img_norm
            @ text_norm.t()
        )  # inner product with temperature scaling, (B, K)
        return logits

    
    def forward(self, batch_img, idx):
        """ batch_img: (B, D)
            idx: (K, 2)
        """
        # In this step, the text features are created from the UPDATED prompt!
        token_tensors = self.construct_token_tensors(idx)  # (K, 8, D)

        text_features, _ = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        ) # (K, D)
        
        text_hypfeat = self.e2h_text(text_features.type(torch.float))
        img_hypfeat = self.e2h_img(batch_img.type(torch.float))
        # compute logits in Hyperbolic space
        logits_h = -dist_matrix(img_hypfeat, text_hypfeat, c=self.curvature) / self.tau_h

        text_h2e = self.h2e_text(text_hypfeat).type(self.dtype)
        text_eucfeat = self.wf * self._norm(text_features) +  (1-self.wf) * self._norm(text_h2e)
        img_h2e = self.h2e_img(img_hypfeat).type(self.dtype)
        img_eucfeat = self.wf * self._norm(batch_img) + (1-self.wf) * self._norm(img_h2e)

        # compute logits in Eucleadian space
        logits_e = self.compute_logits(text_eucfeat, img_eucfeat, 1.0 / self.tau_e, norm=False)

        outputs = {'logits_e': logits_e.type(self.dtype),
                   'logits_h': logits_h.type(self.dtype)}
        
        return outputs
    

    def predict_logits(self, text_rep, dataloader):
        """ Override the default predict logits
            text_rep: text features without normalization
        """
        self.eval()
        all_attr_gt, all_obj_gt, all_pair_gt = (
            [],
            [],
            [],
        )
        all_logits = torch.Tensor()
        with torch.no_grad():
            for idx, data in tqdm(
                enumerate(dataloader), total=len(dataloader), desc="Testing", ncols=0
            ):
                batch_img = data[0].to(self.device)
                batch_img = self.encode_image(batch_img)

                text_hypfeat = self.e2h_text(text_rep.type(torch.float))
                img_hypfeat = self.e2h_img(batch_img.type(torch.float))

                text_h2e = self.h2e_text(text_hypfeat).type(self.dtype)
                text_eucfeat = self.wf * self._norm(text_rep) +  (1-self.wf) * self._norm(text_h2e)
                img_h2e = self.h2e_img(img_hypfeat).type(self.dtype)
                img_eucfeat = self.wf * self._norm(batch_img) + (1-self.wf) * self._norm(img_h2e)

                # compute logits in Eucleadian space
                logits = self.compute_logits(text_eucfeat, img_eucfeat, 1.0 / self.tau_e, norm=False)

                attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
                logits = logits.cpu()
                all_logits = torch.cat([all_logits, logits], dim=0)

                all_attr_gt.append(attr_truth)
                all_obj_gt.append(obj_truth)
                all_pair_gt.append(pair_truth)

        all_attr_gt, all_obj_gt, all_pair_gt = (
            torch.cat(all_attr_gt).to("cpu"),
            torch.cat(all_obj_gt).to("cpu"),
            torch.cat(all_pair_gt).to("cpu"),
        )
        return all_logits, all_attr_gt, all_obj_gt, all_pair_gt


def hypercsp_init(
    train_dataset,
    config,
    device,
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

    soft_prompt = None
    if getattr(config, 'soft_prompt', False):
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
    soft_att_obj = nn.Parameter(soft_att_obj).to(device)
    offset = len(attributes)

    return (
        clip_model,
        token_ids,
        soft_prompt,
        soft_att_obj,
        offset
    )



def get_hypercsp(train_dataset, config, device, is_training=True):

    (
        clip_model,
        token_ids,
        soft_prompt,
        soft_att_obj,
        offset
    ) = hypercsp_init(train_dataset, config, device)

    model = HyperCSP(
        clip_model,
        config,
        offset,
        soft_prompt,
        soft_att_obj,
        token_ids,
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
