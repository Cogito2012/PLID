import torch
import torch.nn as nn
from collections import OrderedDict



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AdapterLayer(nn.Module):
    def __init__(self, dim_in, redu_factor, mode, has_moa=True, dtype=torch.float16):
        super().__init__()
        self.input_dim = dim_in
        self.down_sample_size = self.input_dim // redu_factor
        self.activation = QuickGELU()
        self.mode = mode
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size).type(dtype)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim).type(dtype)
        self.has_moa = has_moa
    
    def forward(self, x, z=None, up_z=None):
        if self.has_moa and self.mode == 'mixture':
            zc = self.down_sampler(x)
            zc = self.activation(zc)
            zc = torch.stack([zc] + z, dim=-1).mean(dim=-1)
            zc = self.up_sampler(zc)
            xc = torch.stack([zc] + up_z, dim=-1).mean(dim=-1)
            return xc
        else:
            x = self.down_sampler(x)
            zc = self.activation(x)
            x = self.up_sampler(zc)
            return zc, x


class CailaBlock(nn.Module):
    def __init__(self, block, modes, dim_in, fusion_key='pair', redu_factor=4, has_moa=True, dtype=torch.float16):
        super().__init__()
        # pre-trained components
        self.attention = block.attention
        self.ln_1 = block.ln_1
        self.mlp = block.mlp
        self.ln_2 = block.ln_2
        self.has_moa = has_moa
        # trainable adapters
        self.adapter1 = nn.ModuleDict(
            {key: AdapterLayer(dim_in, redu_factor, key, has_moa, dtype) for key in modes}
        )
        self.adapter2 = nn.ModuleDict(
            {key: AdapterLayer(dim_in, redu_factor, key, has_moa, dtype) for key in modes}
        )
        self.fusion_key = fusion_key
    
    def forward(self, x, mode):
        # part1
        residual = x
        x = self.ln_1(x)
        x = self.attention(x)
        if mode == 'mixture':
            z_obj, up_z_obj = self.adapter1['obj'](x)
            z_attr, up_z_attr = self.adapter1['attr'](x)
            x = self.adapter1[self.fusion_key](
                    x,
                    z=[z_obj, z_attr],
                    up_z=[up_z_obj, up_z_attr]
                ) + x
        else:
            x = self.adapter1[mode](x)[-1] + x
        x = residual + x

        # part2
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        if mode == 'mixture':
            z_obj, up_z_obj = self.adapter2['obj'](x)
            z_attr, up_z_attr = self.adapter2['attr'](x)
            x = self.adapter2[self.fusion_key](
                    x,
                    z=[z_obj, z_attr],
                    up_z=[up_z_obj, up_z_attr]
                ) + x
        else:
            x = self.adapter2[mode](x)[-1] + x
        x = residual + x
        
        return x


class VisionCaila(nn.Module):
    def __init__(self, vit, adapter_start=0, fusion_start=12, redu_factor=4, has_moa=True, dtype=torch.float16):
        super().__init__()
        self.fusion_start = fusion_start
        all_blocks = []
        for i, block in enumerate(vit.resblocks):
            if i in range(adapter_start, fusion_start):
                all_blocks.append(CailaBlock(block, ['attr', 'obj'], vit.width, 
                                             fusion_key='mixture', redu_factor=redu_factor, has_moa=has_moa, dtype=dtype))
            elif i in range(fusion_start, len(vit.resblocks)):
                all_blocks.append(CailaBlock(block, ['attr', 'obj', 'mixture'], vit.width,
                                             fusion_key='mixture', redu_factor=redu_factor, has_moa=has_moa, dtype=dtype))
            else:
                all_blocks.append(block)
        # self.blocks = nn.Sequential(*all_blocks)
        self.blocks = nn.ModuleList(all_blocks)
    
    def forward(self, x, mode='pair'):
        """ mode: "mixture", or "obj", or "attr"
        """
        # return self.blocks(x, mode)
        start_layer = 0
        if self.fusion_start > 0 and mode == 'mixture':
            start_layer = self.fusion_start
            # the first part
            xa, xo = x, x
            for blk in self.blocks[:start_layer]:
                xa = blk(xa, 'attr')
                xo = blk(xo, 'obj')
            x = torch.cat((xa, xo), dim=0)
        # the rest part
        for blk in self.blocks[start_layer:]:
            x = blk(x, mode)
        return x



class TextCaila(nn.Module):
    def __init__(self) -> None:
        super().__init__()