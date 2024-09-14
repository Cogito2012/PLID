
import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul



class CustomImgEncoder(torch.nn.Module):
    def __init__(self, clip_model, dtype=torch.float16, device=torch.device('cuda:0'), vpt_cfg=None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.visual = clip_model.visual
        self.vpt = vpt_cfg

        if self.vpt is not None:
            # freeze the transformer backbone
            for name, param in self.visual.named_parameters():
                param.requires_grad = False  # fix the parameters
            # define tunable visual prompts
            prompt_dim = self.visual.conv1.out_channels
            self.prompt_proj = nn.Identity()  # do not project the prompt
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.vpt['vp_len'], prompt_dim).type(self.dtype).to(self.device))
            # xavier_uniform initialization
            kernel = self.visual.conv1.kernel_size
            padding = self.visual.conv1.padding
            stride = self.visual.conv1.stride
            patch_size = [
                (self.visual.input_resolution - kernel[0] + 2 * padding[0]) // stride[0] + 1,
                (self.visual.input_resolution - kernel[1] + 2 * padding[1]) // stride[1] + 1
            ]
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    
    def encode_image(self, image):
        
        x = self.visual.conv1(image.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)  # (B, 257, D)
        if self.vpt is not None:  # insert learnable prompt 
            x = torch.cat((
                x[:, :1, :],
                self.prompt_proj(self.prompt_embeddings).expand(x.shape[0], -1, -1),
                x[:, 1:, :]
            ), dim=1)  # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
    
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x
    
    
    def forward(self, image):
        return self.encode_image(image)