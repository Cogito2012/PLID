import torch
import torch.nn as nn
from collections import OrderedDict
import clip


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CRAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop: float = 0., return_kv=False, with_attn=True, input_mlp_dim=64):
        super().__init__()

        self.with_attn = with_attn
        if with_attn:
            self.attn = nn.MultiheadAttention(d_model, n_head, dropout=drop)
            self.attn_mask = attn_mask
        else:
            latent_dim = int(input_mlp_dim // 4)
            self.mlp_layers = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(input_mlp_dim, latent_dim)),
                ("fc_drop", nn.Dropout(drop)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(latent_dim, 1)),
            ]))

        self.ln_x = LayerNorm(d_model)
        self.ln_y = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("fc_drop", nn.Dropout(drop)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("proj_drop", nn.Dropout(drop)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.return_kv = return_kv

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def mlp_mixer(self, x: torch.Tensor, y: torch.Tensor):
        y = y.permute(1, 2, 0).contiguous()  # (B, d, T)
        y = self.mlp_layers(y)  # (B, d, 1)
        y = y.permute(2, 0, 1)  # (1, B, d)
        return y


    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ x: query (T=1, B, d)
            y: key & value (T=64, B, d)
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        if self.with_attn:
            x = x + self.attention(self.ln_x(x), self.ln_y(y))
        else:
            x = x + self.mlp_mixer(self.ln_x(x), self.ln_y(y))

        x = x + self.mlp(self.ln_2(x))
        if x.size(0) == 1:
            x = x.squeeze(0)
        if self.return_kv:
            return x, y
        return x


### modified from original openai clip
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.ln_y = LayerNorm(d_model)  # the only newly added layer
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
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        x = x + self.attention(self.ln_1(x), self.ln_y(y))
        x = x + self.mlp(self.ln_2(x))
        if x.size(0) == 1:
            x = x.squeeze(0)
        return x, y


class CrossResidualAttentionModel(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input[0]


class CLIPTextEncoderBlocks(nn.Module):
    def __init__(self, d_model: int, n_layers: int, arch="ViT-L/14", drop=0.5):
        super().__init__()
        self.need_proj = False

        # load openai pre-trained CLIP
        model, preprocess = clip.load(arch)
        text_width = model.transformer.width

        if text_width != d_model:
            self.proj_pre = nn.Sequential(OrderedDict([
                ("ln", nn.LayerNorm(d_model)),
                ("fc", nn.Linear(d_model, text_width)),
                ("drop", nn.Dropout(drop))
            ]))
            self.proj_post = nn.Sequential(OrderedDict([
                ("ln", nn.LayerNorm(text_width)),
                ("fc", nn.Linear(text_width, d_model)),
                ("drop", nn.Dropout(drop))
            ]))

        attn_mask = None  # for inter-sentence attention, we do not need causal mask

        select_blocks = []
        for i, block in enumerate(model.transformer.resblocks[-n_layers:]):
            xattn_block = ResidualAttentionBlock(text_width, text_width // 64, attn_mask)
            pretrained_dict = block.state_dict()
            pretrained_dict.update({
                'ln_y.weight': pretrained_dict['ln_1.weight'].clone(),
                'ln_y.bias': pretrained_dict['ln_1.bias'].clone(),
            })
            xattn_block.load_state_dict(pretrained_dict, strict=True)
            select_blocks.append(xattn_block)
        self.text_enhance = CrossResidualAttentionModel(*select_blocks)

        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ x: query (Kso, d)
            y: key & value (N=32, Kso, d)
        """
        if self.need_proj:
            residual = x
            x, y = self.proj_pre(x), self.proj_pre(y)
        
        x = self.text_enhance(x, y)

        if self.need_proj:
            x = self.proj_post(x)
            x = x + residual
        
        return x



class CLIPViTEncoderBlocks(nn.Module):
    def __init__(self, d_model: int, n_layers: int, arch="ViT-L/14", drop=0.5):
        super().__init__()
        self.need_proj = False

        # load openai pre-trained CLIP
        model, preprocess = clip.load(arch)

        vision_width = model.visual.conv1.weight.shape[0]
        if vision_width != d_model:
            self.proj_pre = nn.Sequential(OrderedDict([
                ("ln", nn.LayerNorm(d_model)),
                ("fc", nn.Linear(d_model, vision_width)),
                ("drop", nn.Dropout(drop))
            ]))
            self.proj_post = nn.Sequential(OrderedDict([
                ("ln", nn.LayerNorm(vision_width)),
                ("fc", nn.Linear(vision_width, d_model)),
                ("drop", nn.Dropout(drop))
            ]))
            self.need_proj = True

        attn_mask = None  # for inter-patch attention, we do not need causal mask

        select_blocks = []
        for i, block in enumerate(model.visual.transformer.resblocks[-n_layers:]):
            xattn_block = ResidualAttentionBlock(vision_width, vision_width // 64, attn_mask)
            pretrained_dict = block.state_dict()
            pretrained_dict.update({
                'ln_y.weight': pretrained_dict['ln_1.weight'].clone(),
                'ln_y.bias': pretrained_dict['ln_1.bias'].clone(),
            })
            xattn_block.load_state_dict(pretrained_dict, strict=True)
            select_blocks.append(xattn_block)
        self.visual_enhance = CrossResidualAttentionModel(*select_blocks)

        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ x: query (Kso, d)
            y: key & value (N=32, Kso, d)
        """
        if self.need_proj:
            residual = x
            x, y = self.proj_pre(x), self.proj_pre(y)
        
        x = self.visual_enhance(x, y)
        
        if self.need_proj:
            x = self.proj_post(x)
            x = x + residual 

        return x