import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from collections import OrderedDict
from einops import rearrange
import numpy as np
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.realpath(__file__))



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop", nn.Dropout(0.3)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_x = LayerNorm(d_model)
        self.ln_y = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.attention(self.ln_x(x), self.ln_y(y))
        x = x + self.mlp(self.ln_2(x))
        return x


class FusionTextImageBlock(nn.Module):
    def __init__(self, config,
                       dim_embed: int,
                       num_attrs: int, 
                       num_objs: int, 
                       attn_mask: torch.Tensor = None):
        super().__init__()
        self.fusion = config.fusion
        self.width_img = config.width_img
        self.width_txt = config.width_txt
        self.layers = config.SA_K
        self.context_length = config.context_length
        self.attributes = num_attrs
        self.classes = num_objs
        self.with_cross_attn = getattr(config, 'with_cross_attn', True)

        self.img2txt_transform_layer1 = nn.Linear(self.width_img, self.width_txt)
        self.img2txt_transform_layer2 = nn.Linear(dim_embed + 1, self.context_length * (self.attributes + self.classes))
        self.txt2img_transform_layer1 = nn.Linear(self.width_txt, self.width_img)
        self.txt2img_transform_layer2 = nn.Linear(self.context_length * (self.attributes + self.classes), dim_embed + 1)
        self.dropout = nn.Dropout(0.3)
        if self.with_cross_attn:
            self.crossblock_img = CrossResidualAttentionBlock(self.width_img, self.width_img//64, attn_mask)
            self.crossblock_txt = CrossResidualAttentionBlock(self.width_txt, self.width_txt//64, attn_mask)
        self.resblocks_img = nn.Sequential(*[ResidualAttentionBlock(self.width_img, self.width_img//64, attn_mask) for _ in range(self.layers)])
        self.resblocks_txt = nn.Sequential(*[ResidualAttentionBlock(self.width_txt, self.width_txt//64, attn_mask) for _ in range(self.layers)])
        self.txt_fine_tune = nn.Linear(self.width_txt, self.width_txt)


    def decompose(self, text_feature, idx):
        t, l, c = text_feature.shape
        device = text_feature.device
        att_idx, obj_idx = idx[:, 0], idx[:, 1]
        text_att = torch.zeros(t, self.attributes, c).to(device)
        text_obj = torch.zeros(t, self.classes, c).to(device)
        for i in range(self.attributes):
            text_att[:, i, :] = text_feature[:, torch.where(att_idx==i)[0], :].mean(-2)
        for i in range(self.classes):
            text_obj[:, i, :] = text_feature[:, torch.where(obj_idx==i)[0], :].mean(-2)    
        text_decom_feature = torch.cat([text_att, text_obj], dim=1)
        return text_decom_feature


    def compose(self, text_feature, idx):
        t, l, c = text_feature.shape
        device = text_feature.device
        att_idx, obj_idx = idx[:, 0], idx[:, 1]
        text_com_feature = torch.zeros(t, len(idx), c).to(device)
        text_com_feature = text_feature[:, att_idx, :] * text_feature[:, obj_idx + self.attributes, :]
        text_com_feature = self.txt_fine_tune(text_com_feature)
        return text_com_feature



    def img2txt(self, x: torch.Tensor):
        x = self.img2txt_transform_layer1(x)
        x = x.permute(2,1,0)
        x = self.img2txt_transform_layer2(x)
        x = x.permute(2,1,0).reshape(-1, (self.attributes + self.classes), self.width_txt)
        x = self.dropout(x)
        return x

    def txt2img(self, x:torch.Tensor, idx, b: int):
        x = self.decompose(x, idx)
        x = self.txt2img_transform_layer1(x)
        x = rearrange(x, 't l c -> c (t l)')
        x = self.txt2img_transform_layer2(x)
        x = self.dropout(x)
        x = x.permute(1,0).unsqueeze(1).repeat(1,b,1)
        return x
        

    def forward(self, x_image: torch.Tensor, x_text: torch.Tensor, idx, b: int):
        if self.fusion == "BiFusion":
            x_img = self.crossblock_img(x_image, self.txt2img(x_text, idx, b)) if self.with_cross_attn else x_image
            x_txt = self.img2txt(x_image)
            x_text = self.decompose(x_text, idx)
            x_txt = self.crossblock_txt(x_text.repeat(b, 1, 1), x_txt) if self.with_cross_attn else x_text
            x_txt = self.resblocks_txt(x_txt)
            x_txt = self.compose(x_txt, idx)
            x_txt = x_txt.reshape(b, self.context_length, -1, self.width_txt)
            x_img = self.resblocks_img(x_img)
            return x_img, x_txt
        elif self.fusion == "img2txt":
            x_txt = self.img2txt(x_image)
            x_text = self.decompose(x_text, idx)
            x_txt = self.crossblock_txt(x_text.repeat(b, 1, 1), x_txt) if self.with_cross_attn else x_text
            x_txt = self.resblocks_txt(x_txt)
            x_txt = self.compose(x_txt, idx)
            x_txt = x_txt.reshape(b, self.context_length, -1, self.width_txt)
            x_img = self.resblocks_img(x_image)
            return x_img, x_txt
        elif self.fusion == "txt2img":
            x_img = self.crossblock_img(x_image, self.txt2img(x_text, idx, b)) if self.with_cross_attn else x_image
            x_img = self.resblocks_img(x_img)
            x_txt = self.resblocks_txt(x_text)
            return x_img, x_txt
        elif self.fusion == "OnlySPM":
            return x_image, x_text


class DFSP(CLIPInterface):
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
            class_token_ids,
            device=device,
            enable_pos_emb=enable_pos_emb,
            vpt=vpt,
            is_training=is_training
        )
        self.dim_embed = clip_model.token_embedding.weight.size(-1)

        # freeze the clip backbone
        for p in self.parameters():
            p.requires_grad=False

        # learnable parameters
        self.soft_embeddings = soft_att_obj
        self.soft_prompt = soft_prompt

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)
        self.num_attrs = offset
        self.num_objs = soft_att_obj.size(0) - offset
        self.fusion = FusionTextImageBlock(config, self.dim_embed, 
                        num_attrs=self.num_attrs, num_objs=self.num_objs).to(device)
        self.weight = config.res_w
    

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

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.dtype)

        return token_tensor
    

    def visual(self, x: torch.Tensor, get_feature=False):
        x = self.clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip_model.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_model.visual.ln_post(x[:, 0, :])
        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj

        return x, img_feature
    

    def ft_to_logit(self, img, txt):
        img_feature = img.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip_model.visual.ln_post(img_feature[:, 0, :])
        if self.clip_model.visual.proj is not None:
            img_feature = img_feature @ self.clip_model.visual.proj
        
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            txt_feature = txt.permute(0, 2, 1, 3)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    :, torch.arange(txt_feature.shape[1]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )
        else:
            txt_feature = txt.permute(1, 0, 2)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    torch.arange(txt_feature.shape[0]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )
        return img_feature, txt_tf
    

    def decompose_logits(self, logits, idx):
        att_idx, obj_idx = idx[:, 0], idx[:, 1]
        logits_att = torch.zeros(logits.shape[0], self.num_attrs).to(self.device)
        logits_obj = torch.zeros(logits.shape[0], self.num_objs).to(self.device)
        for i in range(self.num_attrs):
            logits_att[:, i] = logits[:, torch.where(att_idx==i)[0]].mean(-1)
        for i in range(self.num_objs):
            logits_obj[:, i] = logits[:, torch.where(obj_idx==i)[0]].mean(-1)        
        return logits_att, logits_obj

    
    def forward(self, batch_img, idx, is_training=True):
        """ batch_img: (B, 3, 224, 224)
            idx: (K, 2)
        """
        b = batch_img.shape[0]
        l, _ = idx.shape
        batch_img, img_ft = self.visual(batch_img.type(self.dtype))   ## bs * 768
        token_tensors = self.construct_token_tensors(idx)
        text_features, text_ft = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
            get_feature=True,
        )  
        batch_img_soft_prompt = batch_img / batch_img.norm(dim=-1, keepdim=True)
        text_features_soft_prompt = text_features / text_features.norm(dim=-1, keepdim=True)
        img_ft, text_ft = self.fusion(img_ft.type(torch.float), text_ft.type(torch.float), idx, b)
        img_ft, text_ft = self.ft_to_logit(img_ft.type(self.dtype), text_ft.type(self.dtype))
        batch_img = self.weight * batch_img + (1 - self.weight) * img_ft
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            text_features = self.weight * text_features.repeat(b, 1, 1) + (1 - self.weight) * text_ft
        else:
            text_features = self.weight * text_features + (1 - self.weight) * text_ft
        idx_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        scale = self.clip_model.logit_scale.exp()
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            logits = (
                scale
                * normalized_img.unsqueeze(1)
                @ idx_text_features.permute(0,2,1)
            ).squeeze()     ###     48 * 1262
        else:
            logits = (
                scale
                * normalized_img
                @ idx_text_features.t()
            )
        if not is_training:
            return logits

        logits_soft_prompt = (
            scale
            * batch_img_soft_prompt
            @ text_features_soft_prompt.t()
        )     

        logits_att, logits_obj = self.decompose_logits(logits_soft_prompt, idx)

        outputs = {'logits': logits, 
                   'logits_att': logits_att,
                   'logits_obj': logits_obj,
                   'logits_soft_prompt': logits_soft_prompt}
        return outputs
    

    def set_pairs_group(self, dataset):
        # get pairs
        pairs = torch.tensor([(dataset.attr2idx[attr], dataset.obj2idx[obj])
                                    for attr, obj in dataset.pairs]).to(self.device)
        self.pairs_group = np.array_split(pairs, 1)  # DFSP has to access to all pairs (cannot be groupped)
    

    def predict_logits(self, text_rep, dataloader):
        """ Override the default predict logits
            text_rep: None by default
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
                # for large combinational semantic space, we compute the logits by groups
                logits_group = torch.Tensor().to(self.device).type(self.dtype)
                for pairs_i in self.pairs_group:
                    logits = self(batch_img, pairs_i, is_training=False)
                    logits_group = torch.cat([logits_group, logits], dim=1)
                
                all_logits = torch.cat([all_logits, logits_group.cpu()], dim=0)
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


def dfsp_init(
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

    ctx_init = "a photo of "
    n_ctx = len(ctx_init.split())
    prompt = clip.tokenize(ctx_init,
                           context_length=config.context_length).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt)
    ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

    token_ids = clip.tokenize('a photo of x x',
                              context_length=config.context_length).to(device)

    soft_prompt = nn.Parameter(ctx_vectors).to(device)
    soft_att_obj = nn.Parameter(soft_att_obj).to(device)
    offset = len(attributes)

    return (
        clip_model,
        token_ids,
        soft_prompt,
        soft_att_obj,
        offset
    )



def get_dfsp(train_dataset, config, device, is_training=True):

    (
        clip_model,
        token_ids,
        soft_prompt,
        soft_att_obj,
        offset
    ) = dfsp_init(train_dataset, config, device)

    dfsp = DFSP(
        clip_model,
        config,
        offset,
        soft_prompt,
        soft_att_obj,
        token_ids,
        device=device,
        enable_pos_emb=True,
    )

    if not is_training:
        return dfsp, None
    
    optim_name = getattr(config, 'optimizer', 'Adam')
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(
            dfsp.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=dfsp.eps
        )
    else:
        optimizer = torch.optim.SGD(dfsp.parameters(), lr=config.lr, momentum=config.momentum)

    return dfsp, optimizer
