import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class CSPInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings,
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
            soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
            vpt=vpt,
            is_training=is_training
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)

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

        return token_tensor


    def compute_text_representations(self, test_dataset, norm=True, get_feat=False):
        """Function computes the attribute-object representations using
        the text encoder.
        Args:
            model (nn.Module): model
            test_dataset (CompositionDataset): CompositionDataset object
                with phase = 'test'
            config (argparse.ArgumentParser): config/args
            device (str): device type cpu/cuda:0
        Returns:
            torch.Tensor: returns the tensor with the attribute-object
                representations
        """
        obj2idx = test_dataset.obj2idx
        attr2idx = test_dataset.attr2idx
        pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                            for attr, obj in test_dataset.pairs]).to(self.device)

        test_pairs = np.array_split(
            pairs, len(pairs) // self.config.text_encoder_batch_size
        )

        rep = torch.Tensor().to(self.device).type(self.dtype)
        text_ft_all = torch.Tensor().to(self.device).type(self.dtype)
        with torch.no_grad():
            for batch_attr_obj in tqdm(test_pairs, ncols=0, desc="compute attr-obj rep"):
                batch_attr_obj = batch_attr_obj.to(self.device)
                token_tensors = self.construct_token_tensors(batch_attr_obj)
                token_ids = self.token_ids
                
                text_features, text_ft = self.text_encoder(
                    token_ids,
                    token_tensors,
                    enable_pos_emb=self.enable_pos_emb,
                    get_feature=get_feat
                )

                if norm:
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )

                rep = torch.cat([rep, text_features], dim=0)
                if get_feat:
                    text_ft_all = torch.cat([text_ft_all, text_ft], dim=1)  # (8, K, d)

        if get_feat:
            return {'rep': rep, 'text_ft': text_ft_all}
        return rep


class CSPCustomInterface(CSPInterface):
    def __init__(self, clip_model, config, offset, soft_embeddings, class_token_ids, device="cuda:0", enable_pos_emb=True, attr_dropout=0, vpt=False, is_training=True):
        super().__init__(clip_model, config, offset, soft_embeddings, class_token_ids, device, enable_pos_emb, attr_dropout, vpt, is_training)

    def forward(self, batch_img, idx):

        # In this step, the text features are created from the UPDATED prompt!
        token_tensors = self.construct_token_tensors(idx)  # (K, 8, D)

        text_features, _ = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )
        _text_features = text_features  # (K, D)

        idx_text_features = _text_features / _text_features.norm(
            dim=-1, keepdim=True
        )
        img_norms = batch_img.norm(dim=-1, keepdim=True)  # L2-norm
        normalized_img = batch_img / img_norms
        tau_inv = self.clip_model.logit_scale.exp()

        logits = (
            tau_inv
            * normalized_img
            @ idx_text_features.t()
        )  # inner product with temperature scaling, (B, K)

        outputs = {'logits': logits, 
                   'img_norm': normalized_img,
                   'text_norm': idx_text_features,
                   'tau_inv': tau_inv,
                   'pair_idx': idx,
                   'norms': img_norms}
        return outputs




class CSP_CVAE_Interface(CSPInterface):
    def __init__(self, clip_model, config, offset, soft_embeddings, class_token_ids, device="cuda:0", enable_pos_emb=True, attr_dropout=0, vpt=False, is_training=True):
        super().__init__(clip_model, config, offset, soft_embeddings, class_token_ids, device, enable_pos_emb, attr_dropout, vpt, is_training)
        dim_in = self.text_encoder.text_projection.size(-1)
        act_method = getattr(config, 'act', 'relu')
        act_layer = nn.Tanh() if act_method == 'tanh' else nn.ReLU()
        self.bn_mean = getattr(config, 'bn_mean', False)
        self.cond_mode = getattr(config, 'cond_mode', 't2i')  # image reconstruction conditioned on text

        # recognition model (encoder)
        mlp_xy_h, dim_h = self._build_mlp([dim_in*2] + config.dim_latent[:-1], act_layer, dtype=self.dtype)
        self.mlp_xy_h = nn.Sequential(mlp_xy_h)
        self.mlp_h_zmean = nn.Linear(dim_h, config.dim_latent[-1], dtype=self.dtype)
        self.mlp_h_zvar = nn.Linear(dim_h, config.dim_latent[-1], dtype=self.dtype)
        if self.bn_mean:
            self.bn_layer = nn.BatchNorm2d(config.dim_latent[-1], eps=self.eps)
            self.bn_layer.weight.requires_grad = False
            self.bn_layer.weight.fill_(0.5)

        # generation model (decoder)
        dims_latent = config.dim_latent[::-1]
        mlp_recon, dim_x = self._build_mlp([dim_in + dims_latent[0]] + dims_latent[1:] + [dim_in], act_layer, dtype=self.dtype)
        if act_method == 'relu': 
            mlp_recon.pop('act{}'.format(len(dims_latent)-1))
        self.mlp_recon = nn.Sequential(mlp_recon)

        self.to(self.device)


    def _build_mlp(self, layer_dims, act_layer, dtype=torch.float32):
        layers = OrderedDict()
        if len(layer_dims) == 1:
            layers['idty'] = nn.Identity()
        else:
            for n, (dim_in, dim_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
                layers['fc{}'.format(n)] = nn.Linear(dim_in, dim_out, dtype=dtype)
                layers['act{}'.format(n)] = act_layer
        return layers, layer_dims[-1]
    

    def _norm(self, data, dim=-1):
        return data / data.norm(dim=dim, keepdim=True)


    def reparametrization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean
    

    def compute_logits(self, text_feat, img_feat, norm=False):
        # compute the normalized features
        text_norm = self._norm(text_feat) if norm else text_feat
        img_norm = self._norm(img_feat) if norm else img_feat
        tau_inv = self.clip_model.logit_scale.exp()
        # inner-product 
        logits = (
            tau_inv
            * img_norm
            @ text_norm.t()
        )  # inner product with temperature scaling, (B, K)
        return logits, tau_inv


    def forward(self, batch_img, idx):
        """ batch_img: (B, D)
        """
        # In this step, the text features are created from the UPDATED prompt!
        token_tensors = self.construct_token_tensors(idx)  # (K, 8, D)

        text_features, _ = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )  # (K, D)

        img_norm = self._norm(batch_img)  # (B, D)
        text_norm = self._norm(text_features)  # (K, D)
        batch_size, num_cls = img_norm.size(0), text_norm.size(0)

        # recognition model (X,Y --> Z)
        input_xy = torch.concat([img_norm.unsqueeze(1).repeat(1, num_cls, 1),
                                 text_norm.unsqueeze(0).repeat(batch_size, 1, 1)], dim=-1)  # (B, K, D+D)
        h = self.mlp_xy_h(input_xy)
        z_mean = self.mlp_h_zmean(h)
        z_logvar = self.mlp_h_zvar(h)
        if self.bn_mean:
            z_mean_in = z_mean.unsqueeze(1).permute(0, 3, 1, 2).contiguous()
            z_mean_out = self.bn_layer(z_mean_in)
            z_mean = z_mean_out.squeeze(2).permute(0, 2, 1).contiguous()
        
        # reparametrization
        zs = self.reparametrization(z_mean, z_logvar)  # (B, K, d)

        if self.cond_mode == 't2i':
            # generation model (Y,Z --> X)
            input_yz = torch.concat([text_norm.unsqueeze(0).repeat(batch_size, 1, 1), zs], dim=-1)  # (B, K, D+d)
            recon = self.mlp_recon(input_yz)  # (B, K, D)
            src = img_norm.clone()  # (B, D)
        elif self.cond_mode == 'i2t':
            # generation model (X,Z --> Y)
            input_xz = torch.concat([img_norm.unsqueeze(1).repeat(1, num_cls, 1), zs], dim=-1)  # (B, K, D+d)
            recon = self.mlp_recon(input_xz)  # (B, K, D)
            src = text_norm.clone()  # (K, D)

        # compute cosine similarities as logits
        logits, tau_inv = self.compute_logits(text_norm, img_norm, norm=False)

        outputs = {'logits': logits, 
                   'z_mean': z_mean,
                   'z_logvar': z_logvar,
                   'recon': recon,
                   'src': src,
                   'tau_inv': tau_inv}
        return outputs


def csp_init(
    train_dataset,
    config,
    device,
):

    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs  # (M,)
    allobj = train_dataset.objs     # (N,)

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )  # (M+N, 8) where ctx_len=8

    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))  # (M+N, 8, D)

    soft_embedding = torch.zeros(
        (len(attributes) + len(classes), orig_token_embedding.size(-1)),
    ).to(device)   # (M+N, D)
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()  # End-of-Seq (EOS) index
        soft_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    soft_embedding = nn.Parameter(soft_embedding)

    class_token_ids = clip.tokenize(
        [config.prompt_template],
        context_length=config.context_length,
    ).to(device)  # (1, 8)
    offset = len(attributes)

    return (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    )



def get_csp(train_dataset, config, device, is_training=True):

    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    with_margin = getattr(config, 'with_margin', False) and config.loss_margin > 0
    with_group = getattr(config, 'with_group', False)
    vpt = getattr(config, 'vpt', False) 
    with_cvae = getattr(config, 'with_cvae', False)
    scale_logits = getattr(config, 'loss', 'CELoss') == 'AdaMarginLoss'
    InterfaceClass = CSPCustomInterface if with_margin or with_group or vpt or scale_logits else CSPInterface
    if with_cvae: InterfaceClass = CSP_CVAE_Interface
    
    interface = InterfaceClass(
        clip_model,
        config,
        offset,
        soft_embedding,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout,
        vpt=vpt,
        is_training=is_training
    )

    if not is_training:
        return interface, None

    learnable_params = [soft_embedding]
    if vpt: learnable_params += [interface.img_encoder.prompt_embeddings]

    if getattr(config, 'tune_temp', False):
        learnable_params += [interface.clip_model.logit_scale]

    optim_name = getattr(config, 'optimizer', 'Adam')
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(
            learnable_params, lr=config.lr, weight_decay=config.weight_decay, eps=interface.eps
        )
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=config.lr, momentum=config.momentum)

    return interface, optimizer


def get_mix_csp(train_dataset, config, device, is_training=True):

    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    with torch.no_grad():
        subset_soft_embeddings = soft_embedding[train_dataset.indices, :]

    subset_soft_embeddings.requires_grad = True

    # reduce the offset to selected offset
    offset = len(train_dataset.attr_indices)
    interface = CSPInterface(
        clip_model,
        config,
        offset,
        subset_soft_embeddings,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )

    if not is_training:
        return interface, None

    optimizer = torch.optim.Adam(
        [subset_soft_embeddings],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    return interface, optimizer
