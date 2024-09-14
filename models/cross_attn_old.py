import torch
import torch.nn as nn
from collections import OrderedDict
from einops import rearrange



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


class CrossResidualAttentionModel(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input[0]


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop", nn.Dropout(0.3)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Text2ImgFusion(nn.Module):
    def __init__(self, clip_model, len_ctx, num_attr, num_obj, token_ids):
        super().__init__()

        self.num_attr = num_attr
        self.num_obj = num_obj
        self.token_ids = token_ids
        width_txt = clip_model.ln_final.weight.shape[0]
        width_img = clip_model.visual.conv1.weight.shape[0]
        dim_embed = clip_model.token_embedding.weight.size(-1)

        self.mlp_c = nn.Linear(width_txt, width_img)
        self.mlp_tl = nn.Linear(len_ctx * (num_attr + num_obj), dim_embed + 1)
        self.dropout = nn.Dropout(0.3)

        self.cross_t2i = CRAttnBlock(width_img, width_img // 64)

        self.selfattn_text = ResidualAttentionBlock(width_txt, width_txt // 64) 
        self.selfattn_img = ResidualAttentionBlock(width_img, width_img // 64)

        self.ln_img = clip_model.visual.ln_post
        self.proj_img = clip_model.visual.proj
        self.ln_text = clip_model.ln_final
        self.proj_text = clip_model.text_projection
        self.token_ids = token_ids
    

    def decompose(self, text_feature, idx):
        t, l, c = text_feature.shape
        device = text_feature.device
        att_idx, obj_idx = idx[:, 0], idx[:, 1]
        text_att = torch.zeros(t, self.num_attr, c).to(device)
        text_obj = torch.zeros(t, self.num_obj, c).to(device)
        for i in range(self.num_attr):
            text_att[:, i, :] = text_feature[:, torch.where(att_idx==i)[0], :].mean(-2)
        for i in range(self.num_obj):
            text_obj[:, i, :] = text_feature[:, torch.where(obj_idx==i)[0], :].mean(-2)    
        text_decom_feature = torch.cat([text_att, text_obj], dim=1)
        return text_decom_feature
    
    
    def forward(self, x_image: torch.Tensor, x_text: torch.Tensor, idx):
        """ x_image: (257, B, D=1024)
            x_text: (8, K, d=768)
        """
        batch_size = x_image.size(1)
        # text2image transform
        x = self.decompose(x_text, idx)  # (8, Ka+Ko, d)
        x = self.mlp_c(x)
        x = rearrange(x, 't l c -> c (t l)')  # (D, 8*(Ko+Ka))
        x = self.mlp_tl(x)  # (D, d+1)
        x = self.dropout(x)
        x = x.permute(1,0).unsqueeze(1).repeat(1, batch_size, 1)  # (d+1, B, D)
        # cross attention
        x = self.cross_t2i(x_image.type(x.dtype), x)
        # self-attention
        x_img = self.selfattn_img(x)  # (257, B, D)
        x_txt = self.selfattn_text(x_text.type(x.dtype))  # (8, K, d)
        # image projection
        x_img = x_img.permute(1, 0, 2)  # LBD -> BLD
        x_img = self.ln_img(x_img[:, 0, :])  # (B, D)
        if self.proj_img is not None:
            x_img = x_img.type(x_image.dtype) @ self.proj_img  # (B, d)
        # text projection
        x_txt = x_txt.permute(1, 0, 2)  # (K, 8, d)
        x_txt = self.ln_text(x_txt)
        x_txt = (
            x_txt[
                torch.arange(x_txt.shape[0]), self.token_ids.argmax(dim=-1)
            ].type(x_text.dtype)  # POS of <EOS>
            @ self.proj_text
        )
        return x_img, x_txt