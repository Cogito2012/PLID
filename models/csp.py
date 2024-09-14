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

    with_group = getattr(config, 'with_group', False)
    vpt = getattr(config, 'vpt', False) 
    InterfaceClass = CSPCustomInterface if with_group or vpt else CSPInterface
    
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
