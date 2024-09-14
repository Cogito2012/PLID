import argparse
import torch
import torch.nn as nn
import clip
from clip_modules.model_loader import load
from clip_modules.interface import CLIPInterface
from datasets.composition_dataset import _preprocess_utzappos
import numpy as np
from tqdm import tqdm


def get_proda(train_dataset, config, device, is_training=True):

    # load the CLIP backbone
    nctx_full = config.context_length + config.context_label_length
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=nctx_full
    )
    embed_size = clip_model.token_embedding.weight.size(-1)  # D=768

    if config.dataset == 'ut-zappos':
        # cleaning the classes and the attributes
        classes, attributes = _preprocess_utzappos(train_dataset.objs, train_dataset.attrs)
    else:
        classes = [cla.replace(".", " ").lower() for cla in train_dataset.objs]
        attributes = [attr.replace(".", " ").lower() for attr in train_dataset.attrs]

    label_embedding, offset = None, None
    if config.soft_label_embed:
        # get the tokenized representation of M attributes and N object names
        tokenized = torch.cat([
                clip.tokenize(tok, context_length=config.context_label_length)
                for tok in attributes + classes
            ])  # (M+N, 8) where ctx_len=8
        
        # embed each token of the (M+N, 8) values
        orig_token_embedding = clip_model.token_embedding(tokenized.to(device))  # (M+N, 8, D)

        # construct the frozen embeddings from attribute + class names
        label_embedding = torch.zeros((len(attributes) + len(classes), embed_size)).to(device)   # (M+N, D)
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()  # End-of-Seq (EOS) index
            label_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)  # average over the context length (=8)        
        label_embedding = nn.Parameter(label_embedding)
        offset = len(attributes)


    # construct the collection of learnable prompts, size = (B, T, D)
    ctx_vectors = torch.empty(config.collection_size, config.context_length, embed_size, dtype=clip_model.dtype).to(device)
    nn.init.normal_(ctx_vectors, std=0.02)
    soft_embedding = nn.Parameter(ctx_vectors)

    proda = ProDA(
        clip_model,
        config,
        offset,
        soft_embedding,
        label_embedding,
        classes,
        attributes,
        device=device,
        enable_pos_emb=True,
        is_training=is_training,
    )

    if not is_training:
        return proda, None
    
    # get the optimizer with learnable parameters: [soft_embedding, label_embedding, tau]
    learnable_params = [p for name, p in proda.named_parameters() if p.requires_grad]
    optim_name = getattr(config, 'optimizer', 'Adam')
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=config.lr, weight_decay=config.weight_decay, eps=proda.eps)
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(learnable_params, lr=config.lr, weight_decay=config.weight_decay, eps=proda.eps)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=config.lr, momentum=config.momentum)

    return proda, optimizer



class ProDA(CLIPInterface):
    def __init__(self,
        clip_model,
        config: argparse.ArgumentParser,
        offset,
        soft_embeddings: torch.nn.Parameter,
        label_embeddings: torch.nn.Parameter,
        objects: list,
        attributes: list,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
        is_training: bool = True,
    ):
        super().__init__(
            clip_model,
            config,
            None,
            soft_embeddings=soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )
        self.label_embeddings = label_embeddings   # (M+N, D)
        self.offset = offset
        self.is_training = is_training
        self.objects = objects
        self.attributes = attributes

        self.context_length = config.context_length
        self.nctx_full = config.context_length + config.context_label_length
        self.soft_label_embed = config.soft_label_embed
        self.group_gauss = getattr(config, 'group_gauss', False)

        self.embed_dim = self.soft_embeddings.size(-1)
        self.collection_size = self.soft_embeddings.size(0)
        self.prompt_bs = config.prompt_bs

        self.rand_insert = getattr(config, 'rand_insert', False)
        # augmentation positions
        if not self.rand_insert:
            self.aug_pos = [0 for _ in range(self.collection_size // 4)] + \
                    [1 for _ in range(self.collection_size // 4)] + \
                    [2 for _ in range(self.collection_size // 2)]
            self.aug_pos = torch.tensor(self.aug_pos).to(self.device)
        self.iter_idx = 0
        self.select_idx = None

        # freeze visual encoders (to save memory)
        for name, param in self.named_parameters():
            if param.requires_grad and 'visual' in name:
                param.requires_grad = False
    

    def get_composable_embeddings(self, class_pairs, token_ids):
        # get the token length for each attribute and object name
        pair_lens = []
        for (attr, obj) in class_pairs:
            tokens = clip.tokenize((attr, obj))  # [SOS][CLS][EOS]
            len_attr = len(tokens[0, 1:tokens[0].argmax()])  # for most names, this is equal to 1
            len_obj = len(tokens[1, 1:tokens[1].argmax()])
            pair_lens.append([len_attr, len_obj])
        
        # get the token embeddings of prefix and suffix
        # eg, "[SOS]X X ... X [Attr] [Obj].[EOS]"
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(token_ids).type(self.dtype)  # (K, T, D)
        
        composable_embeddings = {'prefix': [], 'attr': [], 'obj': [], 'suffix': []}
        for cls, embed in enumerate(embedding):
            p = 1
            composable_embeddings['prefix'].append(embed[:p, :])  # "[SOS]", (1, D)
            p += self.context_length
            composable_embeddings['attr'].append(embed[p: (p + pair_lens[cls][0]), :])  # "[Attr]", (*, D)
            p += pair_lens[cls][0]
            composable_embeddings['obj'].append(embed[p: (p + pair_lens[cls][1]), :])   # "[Obj]", (*, D)
            p += pair_lens[cls][1]
            composable_embeddings['suffix'].append(embed[p:, :])  # ".[EOS]", (*, D)
        
        return composable_embeddings

    
    def get_prompt_batch(self):
        """ Get a batch of prompts
        """
        if self.iter_idx == 0:
            self.select_idx = torch.randperm(self.collection_size).to(self.device)
        batch_idx = self.select_idx[self.iter_idx * self.prompt_bs: (self.iter_idx + 1) * self.prompt_bs]  # (B,)
        soft_embeddings_batch = self.soft_embeddings[batch_idx]  # (B, T, D)
        pos_batch = self.aug_pos[batch_idx] if not self.rand_insert else None # (B,)
        self.iter_idx += 1
        if self.iter_idx == self.collection_size // self.prompt_bs:
            self.iter_idx = 0  # reset position to 0 after a full enumeration
        return soft_embeddings_batch, pos_batch
    

    def get_prompted_embeddings_slow(self, prompt, loc, n_cls, composable_embeddings, attr_obj_embedding=None):
        embed_result = []
        num_prompt = len(prompt)
        if num_prompt > 0:
            for i in range(n_cls):
                prefix = composable_embeddings['prefix'][i].unsqueeze(0).repeat(num_prompt, 1, 1)  # (N, 1, D)
                # get attrbute and object embeddings
                n_attr = composable_embeddings['attr'][i].size(0)
                n_obj = composable_embeddings['obj'][i].size(0)
                if self.soft_label_embed:
                    # learnable embeddings
                    attr = attr_obj_embedding[i, 0].unsqueeze(0).unsqueeze(0).repeat(num_prompt, n_attr, 1)
                    obj = attr_obj_embedding[i, 1].unsqueeze(0).unsqueeze(0).repeat(num_prompt, n_obj, 1)
                else:  # frozen embeddings
                    attr = composable_embeddings['attr'][i].unsqueeze(0).repeat(num_prompt, 1, 1)  # (N, n_attr, D)
                    obj = composable_embeddings['obj'][i].unsqueeze(0).repeat(num_prompt, 1, 1)  # (N, n_obj, D)
                suffix = composable_embeddings['suffix'][i].unsqueeze(0).repeat(num_prompt, 1, 1)  # (N, n_suffix, D)
                # construct embedding
                prompt_cls = torch.cat([
                    prefix, prompt[:, :loc, :], 
                    attr, obj, 
                    prompt[:, loc:, :], suffix
                ], dim=1)
                embed_result.append(prompt_cls)
            embed_result = torch.stack(embed_result, dim=0)  # (K, N, T, D)
        return embed_result
    

    def get_prompted_embeddings(self, prompt, loc, n_cls, composable_embeddings, attr_obj_embedding=None):
        embed_result = []
        num_prompt = len(prompt)
        if num_prompt > 0:
            embed_result = torch.zeros((n_cls, num_prompt, self.nctx_full, self.embed_dim), dtype=self.dtype).to(self.device, non_blocking=True)
            embed_result[:, :, [0], :] = torch.stack(composable_embeddings['prefix'], dim=0).unsqueeze(1).expand(n_cls, num_prompt, 1, -1) # (K, N, 1, D)
            for i in range(n_cls):
                # get attrbute and object embeddings
                n_attr = composable_embeddings['attr'][i].size(0)
                n_obj = composable_embeddings['obj'][i].size(0)
                if self.soft_label_embed:
                    # learnable embeddings
                    attr = attr_obj_embedding[[i], [0]].expand(num_prompt, n_attr, -1)
                    obj = attr_obj_embedding[[i], [1]].expand(num_prompt, n_obj, -1)
                else:  # frozen embeddings
                    attr = composable_embeddings['attr'][i].expand(num_prompt, -1, -1)  # (N, n_attr, D)
                    obj = composable_embeddings['obj'][i].expand(num_prompt, -1, -1)  # (N, n_obj, D)
                suffix = composable_embeddings['suffix'][i].expand(num_prompt, -1, -1)  # (N, n_suffix, D)
                # construct embedding
                embed_result[i, :, 1:, :] = torch.cat([
                    prompt[:, :loc, :], 
                    attr, obj, 
                    prompt[:, loc:, :], suffix
                ], dim=1)  # This will create a contiguous tensor from noncontiguous ones, size: (N, T-1, D)
        return embed_result
    

    def get_nonclass_features(self):
        nc_prompts = [' '.join(['X'] * self.context_length) + '.']  # 'X X ... X.'
        nc_tokenized_prompts = torch.cat([clip.tokenize(p, context_length=self.nctx_full) for p in nc_prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(nc_tokenized_prompts).type(self.dtype)  # (1, T, D)
            prefix_nc = embedding[:, :1, :]  # (1, 1, D)
            suffix_nc = embedding[:, (1+self.context_length):, :]  # (1, *, D)
        nc_tokens = nc_tokenized_prompts.expand(self.collection_size, -1).contiguous()  # (32, T)

        # insert the learnable soft prompt
        nc_prompt_embed = torch.cat([
            prefix_nc.expand(self.collection_size, -1, -1),  # (32, 1, D)
            self.soft_embeddings,  # (32, n_ctx, D)
            suffix_nc.expand(self.collection_size, -1, -1)  # (32, *, D)
        ], dim=1)  # (32, 1 + n_ctx + n_suffix = T, D)
        # get features
        text_features, _ = self.text_encoder(
            nc_tokens,
            nc_prompt_embed,
            enable_pos_emb=self.enable_pos_emb,
        )  # (32, D)

        return text_features

    
    def construct_token_tensors(self, pair_idx):

        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        n_cls = len(pair_idx)

        # get token_ids
        prompt_prefix = ' '.join(['X'] * self.context_length)
        class_pairs = [(self.attributes[idx_a], self.objects[idx_o]) for (idx_a, idx_o) in pair_idx] 
        prompts = [prompt_prefix + ' %s %s.'%(att, obj) for (att, obj) in class_pairs]  # "[SOS]X X ... X [Attr] [Obj].[EOS]"
        self.token_ids = torch.cat([clip.tokenize(p, context_length=self.nctx_full) for p in prompts]).to(self.device)  # (K, n_ctx+8)

        # get composible embeddings
        composable_embeddings = self.get_composable_embeddings(class_pairs, self.token_ids)

        attr_obj_embedding = None
        if self.soft_label_embed:
            # build the attribute-object embeddings
            attr_obj_embedding = torch.concat([
                self.label_embeddings[attr_idx].unsqueeze(1),
                self.label_embeddings[obj_idx + self.offset].unsqueeze(1)
            ], dim=1).type(self.clip_model.dtype)  # (K, 2, D)
        
        # get a batch of prompts
        soft_embeddings_batch, pos_batch = self.get_prompt_batch()  # (B, n_ctx, D)

        if self.rand_insert:
            # insert class embeddings randomly
            token_tensor = self.get_prompted_embeddings(soft_embeddings_batch, int(self.context_length * torch.rand(1)[0]), 
                                                        n_cls, composable_embeddings, attr_obj_embedding)
        else:
            # construct embeddings by inserting into the front, e.g., "[SOS] X X a photo of a [EOS]"
            prompt_front = soft_embeddings_batch[pos_batch == 0]  # (N1, n_ctx, D)
            embed_front = self.get_prompted_embeddings(prompt_front, 0, 
                                                       n_cls, composable_embeddings, attr_obj_embedding)
            # construct embeddings by inserting into the middle, e.g., "[SOS] a photo X X of a [EOS]"
            prompt_mid = soft_embeddings_batch[pos_batch == 1]  # (N2, n_ctx, D)
            embed_mid = self.get_prompted_embeddings(prompt_mid, int(self.context_length * 0.5), 
                                                     n_cls, composable_embeddings, attr_obj_embedding)
            # construct embeddings by inserting into the end, e.g., "[SOS] a photo of a X X [EOS]"
            prompt_end = soft_embeddings_batch[pos_batch == 2]  # (N3, n_ctx, D)
            embed_end = self.get_prompted_embeddings(prompt_end, self.context_length, 
                                                     n_cls, composable_embeddings, attr_obj_embedding)
            token_tensor = torch.concat([p for p in [embed_end, embed_mid, embed_front] if len(p) > 0], dim=1)  # (K, B, T, D)
        
            return token_tensor
        
        return token_tensor


    def forward_single(self, batch_img, idx):

        token_tensors = self.construct_token_tensors(idx)  # (K, B, T, D)
        if self.is_training:
            # get class-agnostic text features
            nc_features = self.get_nonclass_features()
        K, B, T = token_tensors.size()[:3]

        token_tensors = token_tensors.view(-1, T, self.embed_dim)  # (K * B, T, D)
        token_ids = self.token_ids.unsqueeze(1).repeat(1, B, 1).view(-1, T)  # (K, B, T)

        # use text features to compute classifier weights
        text_features, _ = self.text_encoder(
            token_ids,  # (K*B, T)
            token_tensors,  # (K*B, T, D)
            enable_pos_emb=self.enable_pos_emb,
        )
        _text_features = text_features.view(-1, self.prompt_bs, self.embed_dim)  # (K, B, D)

        # normalize
        normalized_text = _text_features / _text_features.norm(dim=-1, keepdim=True)
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)  # (batch_size, D)
        
        # use the empirical mean as the classifier weight
        _weight_mean = normalized_text.mean(dim=1)  # (K, D)

        # logits
        tau_inv = self.clip_model.logit_scale.exp()
        logits = tau_inv * normalized_img @ _weight_mean.t() # inner product with temperature scaling, (batch_size, K)

        if not self.is_training:
            return logits
        
        # get the prompt features
        normalized_prompt = nc_features / nc_features.norm(dim=-1, keepdim=True)

        outputs = {'logits': logits, 
                   'img_norm': normalized_img,
                   'text_norm': normalized_text,
                   'tau_inv': tau_inv,
                   'prompt_norm': normalized_prompt}
        if self.group_gauss:
            outputs.update({
                'num_objs': len(self.objects),
                'obj_idx': idx[:, 1]
            })
        if getattr(self.config, 'with_margin', False) and self.config.loss_margin > 0:
            outputs.update({'pair_idx': idx})
        return outputs
    

    def forward(self, batch_img, idx):
        # if len(idx) > 1000:
        #     text_bs = getattr(self.config, 'text_encoder_batch_size', 64)
        #     pairs_group = np.array_split(idx, len(idx) // text_bs)
        #     logits, norm_texts = [], []
        #     for i, pair in enumerate(pairs_group):
        #         output = self.forward_single(batch_img, pair)
        #         logits.append(output['logits'])  # (bs, K)
        #         norm_texts.append(output['text_norm'])  # (K, B, D)
        #         if i == 0:
        #             norm_img = output['img_norm']
        #             tau_inv = output['tau_inv']
        #             norm_prompt = output['prompt_norm']
        #             if self.group_gauss:
        #                 num_objs = output['num_objs']
        #                 obj_idx = output['obj_idx']
        #     logits = torch.cat(logits, dim=-1)
        #     norm_texts = torch.cat(norm_texts, dim=0)
        #     outputs = {'logits': logits, 'img_norm': norm_img, 'text_norm': norm_texts, 'tau_inv': tau_inv, 'prompt_norm': norm_prompt}
        #     if self.group_gauss:
        #         outputs.update({'num_objs': num_objs, 'obj_idx': obj_idx})
        #     return outputs
        # else:
        #     return self.forward_single(batch_img, idx)
        return self.forward_single(batch_img, idx)


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

                K, B, T, D = token_tensors.size()
                token_tensors = token_tensors.view(-1, T, D)  # (K * B, T, D)
                token_ids = token_ids.unsqueeze(1).repeat(1, B, 1).view(-1, T)  # (K * B, T)
                
                text_features, _ = self.text_encoder(
                    token_ids,
                    token_tensors,
                    enable_pos_emb=self.enable_pos_emb,
                )

                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )
                text_features = text_features.view(-1, B, D).mean(dim=1)  # (K, B, D) --> (K, D)

                rep = torch.cat([rep, text_features], dim=0)

        return rep