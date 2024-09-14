## Remove this later
import argparse
import os

import clip
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
import numpy as np
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def coop(train_dataset, config, device, is_training=True, prompt_template="a photo of x x"):
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

    with torch.no_grad():
        frozen_embedding = torch.zeros(
            (len(attributes) + len(classes), clip_model.token_embedding.weight.size(-1)),
        ).to(device)
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            frozen_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    ctx_init = "a photo of "
    n_ctx = len(ctx_init.split())
    prompt = clip.tokenize(ctx_init,
                           context_length=config.context_length).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt)

    ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

    token_ids = clip.tokenize(prompt_template,
                              context_length=config.context_length).to(device)

    soft_embedding = nn.Parameter(ctx_vectors).to(device)
    offset = len(attributes)

    coop = COOP(
        clip_model,
        config,
        offset,
        soft_embedding,
        frozen_embedding,
        token_ids,
        device=device,
        enable_pos_emb=True,
    )

    if not is_training:
        return coop, None
    
    optim_name = getattr(config, 'optimizer', 'Adam')
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(
            [soft_embedding], lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.SGD([soft_embedding], lr=config.lr, momentum=config.momentum)

    return coop, optimizer


class COOP(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config: argparse.ArgumentParser,
        offset,
        soft_embeddings: torch.nn.Parameter,
        frozen_embeddings: torch.nn.Parameter,
        token_ids: torch.tensor,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
    ):
        super().__init__(
            clip_model,
            config,
            token_ids,
            soft_embeddings=soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )
        self.frozen_embeddings = frozen_embeddings
        self.offset = offset

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)  # (K, T)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)  # (batch_size, T, D)

        eos_idx = int(self.token_ids[0].argmax())
        token_tensor[:, eos_idx - 2, :] = self.frozen_embeddings[
            attr_idx
        ].type(self.clip_model.dtype)
        token_tensor[:, eos_idx - 1, :] = self.frozen_embeddings[
            obj_idx + self.offset
        ].type(self.clip_model.dtype)

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_embeddings) + 1, :
        ] = self.soft_embeddings.type(self.clip_model.dtype)

        return token_tensor


    def compute_text_representations(self, test_dataset, norm=True):
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
        with torch.no_grad():
            for batch_attr_obj in tqdm(test_pairs, ncols=0, desc="compute attr-obj rep"):
                batch_attr_obj = batch_attr_obj.to(self.device)
                token_tensors = self.construct_token_tensors(batch_attr_obj)
                token_ids = self.token_ids
                
                text_features, _ = self.text_encoder(
                    token_ids,
                    token_tensors,
                    enable_pos_emb=self.enable_pos_emb,
                )

                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

                rep = torch.cat([rep, text_features], dim=0)

        return rep