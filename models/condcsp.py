import clip
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from datasets.composition_dataset import _preprocess_utzappos
from collections import OrderedDict
from tqdm import tqdm
import numpy as np


class CondCSPInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_ctx_embeddings,
        soft_embeddings,
        class_token_ids,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            class_token_ids,
            soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)
        self.vis_dim = self.clip_model.visual.output_dim
        self.embed_dim = self.soft_embeddings.size(-1)
        self.dist_learn = getattr(config, 'dist_learn', False)

        self.soft_ctx_embeddings = soft_ctx_embeddings
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(self.vis_dim, self.vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(self.vis_dim // 16, self.embed_dim))
        ])).to(device=device, dtype=self.dtype)

        # freeze visual encoders (to save memory)
        for name, param in self.named_parameters():
            if param.requires_grad and 'visual' in name:
                param.requires_grad = False
    

    def set_soft_embeddings(self, se_path):
        embeds = torch.load(se_path)
        self.state_dict()['soft_embeddings'].copy_(embeds['soft_embeddings'])
        self.state_dict()['soft_ctx_embeddings'].copy_(embeds['ctx_embeddings'])
        self.meta_net.load_state_dict(embeds['meta_net'])


    def construct_token_tensors(self, pair_idx, context_imgs):
        """Function creates the token tensor for further inference.

        Args:
            pair_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj
            context_imgs (torch.Tensor): Shape [B, nctx, D]
    
        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        dtype = self.clip_model.dtype
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)  # repeat the token_ids of the template 'a photo of a x x'
        token_tensor = self.clip_model.token_embedding(
            class_token_ids
        ).type(dtype)  # (M*N, T, D)
        num_cls = len(pair_idx)
        batch_size = len(context_imgs)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings = self.attr_dropout(self.soft_embeddings)  # (M+N, D)
        # replace with the soft embeddings of attribute
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[
            attr_idx
        ].type(dtype)
        # replace with the soft embeddings of object
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[
            obj_idx + self.offset
        ].type(dtype)

        # conditional context
        token_tensor = token_tensor.unsqueeze(1).expand(-1, batch_size, -1, -1).clone()  # (M*N, B, T, D)
        context = context_imgs + self.soft_ctx_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, nctx, D)
        token_tensor[:, torch.arange(batch_size), 1: len(self.soft_ctx_embeddings)+1, :] = context.unsqueeze(0).repeat(num_cls, 1, 1, 1).to(self.dtype)

        return token_tensor
    

    def get_img_condition(self, batch_img):
        # get image condition
        pi = self.meta_net(batch_img)  # (B, D)
        context_shifted = self.soft_ctx_embeddings.unsqueeze(0) + pi.unsqueeze(1)  # (1, nctx, D) + (B, 1, D)
        return context_shifted
    

    def conditioned_text_embeddings(self, context_shifted, idx):
        # In this step, the text features are created from the UPDATED prompt!
        token_tensors = self.construct_token_tensors(idx, context_shifted)  # (K, B, T, D)

        K, B, T = token_tensors.size()[:3]
        token_tensors = token_tensors.view(-1, T, self.embed_dim)  # (K * B, T, D)
        token_ids = self.token_ids.unsqueeze(1).repeat(K, B, 1).view(-1, T)  # (K * B, T)

        text_features, _ = self.text_encoder(
            token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )
        _text_features = text_features.view(-1, B, self.embed_dim)  # (K, B, D)

        return _text_features


    def forward(self, batch_img, idx):

        # get image condition
        context_shifted = self.get_img_condition(batch_img)

        # get conditioned text embeddings are image classifer
        _text_features = self.conditioned_text_embeddings(context_shifted, idx)

        idx_text_features = _text_features / _text_features.norm(
            dim=-1, keepdim=True
        )
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        tau_inv = self.clip_model.logit_scale.exp()

        if not self.dist_learn:
            logits = tau_inv * torch.bmm(normalized_img.unsqueeze(1), 
                                         idx_text_features.permute(1, 2, 0)).squeeze(1)  # (B,1,D) x (B,D,K) --> (B,K)
            outputs = {'logits': logits}
        else:
            _weight_mean = idx_text_features.mean(dim=1)  # (K, D)
            logits = tau_inv * normalized_img @ _weight_mean.t()  # shared-classifier
            outputs = {'logits': logits, 
                       'img_norm': normalized_img,
                       'text_norm': idx_text_features,
                       'tau_inv': tau_inv}

        return outputs
    

    def set_pairs_group(self, dataset):
        # get pairs
        pairs = torch.tensor([(dataset.attr2idx[attr], dataset.obj2idx[obj])
                                    for attr, obj in dataset.pairs]).to(self.device)
        self.pairs_group = np.array_split(
            pairs, len(pairs) // self.config.text_encoder_batch_size
        )


    def predict_logits(self, text_rep, dataloader):
        """ predict logits where the classifier is conditioned on both image and text
        """
        self.eval()
        all_attr_gt, all_obj_gt, all_pair_gt = [], [], []
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
                tau_inv = self.clip_model.logit_scale.exp()
                # get image condition
                context_shifted = self.get_img_condition(batch_img_feat)  # (B, nctx=4, D)

                # for large combinational semantic space, we compute the logits by groups
                logits_group = torch.Tensor().to(self.device).type(self.dtype)
                for pairs_i in self.pairs_group:
                    # get conditioned text embeddings are image classifer
                    text_rep = self.conditioned_text_embeddings(context_shifted, pairs_i)
                    text_rep = text_rep / text_rep.norm(dim=-1, keepdim=True)  # (K, B, D)
                    
                    if getattr(self, 'dist_learn', False):
                        _weight_mean = text_rep.mean(dim=1)  # (K, D)
                        logits = tau_inv * normalized_img @ _weight_mean.t() 
                    else:
                        logits = tau_inv * torch.bmm(normalized_img.unsqueeze(1), 
                                                    text_rep.permute(1, 2, 0)).squeeze(1)
                    # logits = logits.cpu()
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



def condcsp_init(
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
    if config.dataset == 'ut-zappos':
        # cleaning the classes and the attributes
        classes, attributes = _preprocess_utzappos(train_dataset.objs, train_dataset.attrs)
    else:
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
    soft_embedding = nn.Parameter(soft_embedding)  # (M+N, D)

    class_token_ids = clip.tokenize(
        [config.prompt_template],
        context_length=config.context_length,
    ).to(device)  # (1, 8)
    offset = len(attributes)

    # learnable context
    ctx_init = config.prompt_template[:-3]  # "a photo of a "
    n_ctx = len(ctx_init.split())
    prompt = clip.tokenize(ctx_init, context_length=config.context_length).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt)
    ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
    soft_ctx_embedding = nn.Parameter(ctx_vectors).to(device)  # (nctx, D)

    return (
        clip_model,
        soft_ctx_embedding,
        soft_embedding,
        class_token_ids,
        offset
    )



def get_condcsp(train_dataset, config, device, is_training=True):

    (
        clip_model,
        soft_ctx_embedding,
        soft_embedding,
        class_token_ids,
        offset
    ) = condcsp_init(train_dataset, config, device)

    interface = CondCSPInterface(
        clip_model,
        config,
        offset,
        soft_ctx_embedding,
        soft_embedding,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )

    if not is_training:
        return interface, None
    
    learnable_params = [p for name, p in interface.named_parameters() if name in ['soft_embeddings', 'soft_ctx_embeddings'] or 'meta_net' in name]
    optim_name = getattr(config, 'optimizer', 'Adam')
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(
            learnable_params, lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=config.lr, momentum=config.momentum)

    return interface, optimizer

