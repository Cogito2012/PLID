import torch
import torch.nn as nn
from collections import OrderedDict



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AdapterLayer(nn.Module):
    def __init__(self, dim_in, dim_r=64, drop=0.1, dtype=torch.float16):
        super().__init__()
        self.bottle_neck = nn.Sequential(OrderedDict([
            ("down_fc", nn.Linear(dim_in, dim_r, bias=False)),
            ("down_drop", nn.Dropout(drop)),
            ("gelu", QuickGELU()),
            ("up_proj", nn.Linear(dim_r, dim_in, bias=False)),
            ("up_drop", nn.Dropout(drop))
        ])).type(dtype)
    
    def forward(self, x):
        return x + self.bottle_neck(x)


class AdapterBlock(nn.Module):
    def __init__(self, block, dim_in, dim_r=64, drop=0.1, dtype=torch.float16):
        super().__init__()
        # pre-trained components
        self.attention = block.attention
        self.ln_1 = block.ln_1
        self.mlp = block.mlp
        self.ln_2 = block.ln_2
        # trainable adapters
        self.adapter_attn = AdapterLayer(dim_in, dim_r, drop, dtype)
        self.adapter_ffn = AdapterLayer(dim_in, dim_r, drop, dtype)
    
    def forward(self, x):
        x = x + self.adapter_attn(self.attention(self.ln_1(x)))
        x = x + self.adapter_ffn(self.mlp(self.ln_2(x)))
        return x


class ViTAdapter(nn.Module):
    def __init__(self, vit, adaptable_blocks=None, dim_r=64, drop=0.1, dtype=torch.float16):
        super().__init__()
        self.adaptable_blocks = list(range(len(vit.resblocks))) if adaptable_blocks is None else adaptable_blocks
        all_blocks = []
        for i, block in enumerate(vit.resblocks):
            if i in adaptable_blocks:
                all_blocks.append(AdapterBlock(block, vit.width, dim_r, drop, dtype))
            else:
                all_blocks.append(block)
        self.blocks = nn.Sequential(*all_blocks)
    
    def forward(self, x):
        return self.blocks(x)


class PrimitiveAdapter(nn.Module):
    def __init__(self, dim_in, reduction=4, ratio=0.5, dtype=torch.float):
        super().__init__()
        self.ratio = ratio
        self.bottle_neck = nn.Sequential(OrderedDict([
            ("down_fc", nn.Linear(dim_in, dim_in // reduction, bias=False)),
            ("down_gelu", QuickGELU()),
            ("up_proj", nn.Linear(dim_in // reduction, dim_in, bias=False)),
            ("up_gelu", QuickGELU()),
        ])).type(dtype)
    
    def forward(self, x):
        x_adapt = self.ratio * self.bottle_neck(x) + (1 - self.ratio) * x
        return x_adapt