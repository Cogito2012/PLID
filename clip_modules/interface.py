import argparse

import torch
import clip
from clip.model import CLIP

from .text_encoder import CustomTextEncoder
from .img_encoder import CustomImgEncoder
import numpy as np
from tqdm import tqdm


class CLIPInterface(torch.nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        config: argparse.ArgumentParser,
        token_ids: torch.tensor,
        soft_embeddings: torch.nn.Parameter = None,
        dtype: torch.dtype = None,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
        vpt: bool = False,
        is_training: bool = True
    ):
        """CLIP interface for our custom modules.

        Args:
            clip_model (CLIP): the clip model
            config (argparse.ArgumentParser): arguments used for
                training
            token_ids (torch.tensor): the input token ids to the text
                encoder
            soft_embeddings (torch.nn.Parameter, optional): the only
                parameter that we finetune in the experiment.
                Defaults to None.
            dtype (torch.dtype, optional): torch dtype for the
                transformer. This allows the half precision option.
                Defaults to None.
            device (torch.device, optional): the device where the model
                should be loaded. Defaults to "cuda:0".
            enable_pos_emb (bool, optional): if true, adds the learned
                positional embeddings. Defaults to False.
            vpt (bool, optional): if true, use Visual Prompt Tuning (shallow)
        """
        super().__init__()

        self.config = config

        self.clip_model = clip_model

        self.eps = 1e-8
        if dtype is None and device == "cpu":
            self.dtype = torch.float32
        elif dtype is None:
            self.dtype = torch.float16
            self.eps = 1e-4
        else:
            self.dtype = dtype

        self.device = device

        self.enable_pos_emb = enable_pos_emb
        self.vpt = vpt
        if vpt:
            vpt_cfg = {'vp_len': config.vp_len}
            self.img_encoder = CustomImgEncoder(clip_model, self.dtype, vpt_cfg=vpt_cfg)

        self.text_encoder = CustomTextEncoder(clip_model, self.dtype, device)
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        self.clip_model.text_projection.requires_grad = False

        self.token_ids = token_ids
        self.soft_embeddings = soft_embeddings

        self.is_training = is_training


    def encode_image(self, imgs):
        if not self.vpt:
            return self.clip_model.encode_image(imgs)
        else:
            return self.img_encoder.encode_image(imgs)

    def encode_text(self, text, enable_pos_emb=True):
        return self.text_encoder.encode_text(
            text, enable_pos_emb=enable_pos_emb
        )

    def tokenize(self, text):
        return self.text_encoder.tokenize(text)

    def set_soft_embeddings(self, se_path):
        embeds = torch.load(se_path)
        se = embeds['soft_embeddings']
        if se.shape == self.soft_embeddings.shape:
            self.state_dict()['soft_embeddings'].copy_(se)
        else:
            raise RuntimeError(f"Error: Incorrect Soft Embedding Shape {se.shape}, Expecting {self.soft_embeddings.shape}!")
        if 'label_embeddings' in embeds:
            le = embeds['label_embeddings']
            self.state_dict()['label_embeddings'].copy_(le)
        

    def construct_token_tensors(self, idx):
        """The function is used to generate token tokens. These
        token tensors can be None or custom. For custom token_tensors
        the class needs to be inherited and the function should be
        replaced.

        Raises:
            NotImplementedError: raises error if the model contains
            soft embeddings but does not make custom modifications.

        Returns:
            torch.Tensor: returns torch.Tensor or None
        """
        if self.soft_embeddings is None:
            return None
        else:
            # Implement a custom version
            raise NotImplementedError


    def forward(self, batch_img, idx):

        # In this step, the text features are created from the UPDATED prompt!
        token_tensors = self.construct_token_tensors(idx)  # (K, 8, D)

        text_features, _ = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )

        #_text_features = text_features[idx, :]
        _text_features = text_features  # (K, D)

        idx_text_features = _text_features / _text_features.norm(
            dim=-1, keepdim=True
        )
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        logits = (
            self.clip_model.logit_scale.exp()
            * normalized_img
            @ idx_text_features.t()
        )  # inner product with temperature scaling, (B, K)

        return logits



    def compute_text_representations(self, test_dataset, norm=True):
        """Function to get the clip text representations.
        Args:
            model (nn.Module): the clip model
            test_dataset (CompositionDataset): the test/validation dataset
            norm: whether normalize the text representations

        Returns:
            torch.Tensor: returns the tensor with the attribute-object
                representations with clip model.
        """
        pairs = test_dataset.pairs
        pairs = [(attr.replace(".", " ").lower(),
                obj.replace(".", " ").lower())
                for attr, obj in pairs]

        prompts = [f"a photo of {attr} {obj}" for attr, obj in pairs]
        tokenized_prompts = clip.tokenize(
            prompts, context_length=self.config.context_length)
        test_batch_tokens = np.array_split(
            tokenized_prompts,
            len(tokenized_prompts) //
            self.config.text_encoder_batch_size)
        rep = torch.Tensor().to(self.device).type(self.dtype)
        with torch.no_grad():
            for batch_tokens in test_batch_tokens:
                batch_tokens = batch_tokens.to(self.device)
                text_features, _ = self.text_encoder(
                    batch_tokens, enable_pos_emb=True)
                if norm:
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )
                rep = torch.cat((rep, text_features), dim=0)
        return rep

    
    def predict_logits(self, text_rep, dataloader):
        """Function to predict the cosine similarities between the
        images and the attribute-object representations. The function
        also returns the ground truth for attributes, objects, and pair
        of attribute-objects.

        Args:
            model (nn.Module): the model
            text_rep (nn.Tensor): the attribute-object representations.
            dataset (CompositionDataset): the composition dataset (validation/test)

        Returns:
            tuple: the logits, attribute labels, object labels,
                pair attribute-object labels
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
                batch_img_feat = self.encode_image(batch_img)
                normalized_img = batch_img_feat / batch_img_feat.norm(
                    dim=-1, keepdim=True
                )

                logits = (
                    self.clip_model.logit_scale.exp()
                    * normalized_img
                    @ text_rep.t()
                )

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