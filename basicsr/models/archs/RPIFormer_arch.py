import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Optional, Sequence

from basicsr.models.modules.Lighten_Cross_Attention import LCA
from basicsr.models.modules.euler_proc import HeightWidthChannelEulerProcessor as EulerProc


###############################################################
# Helper initialization (same as 001/002 for consistency)
###############################################################


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    import math
    import warnings
    from torch.nn.init import _calculate_fan_in_and_fan_out

    def _no_grad_trunc_normal_(tensor_, mean_, std_, a_, b_):
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        if (mean_ < a_ - 2 * std_) or (mean_ > b_ + 2 * std_):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)
        with torch.no_grad():
            l = norm_cdf((a_ - mean_) / std_)
            u = norm_cdf((b_ - mean_) / std_)
            tensor_.uniform_(2 * l - 1, 2 * u - 1)
            tensor_.erfinv_()
            tensor_.mul_(std_ * math.sqrt(2.))
            tensor_.add_(mean_)
            tensor_.clamp_(min=a_, max=b_)
            return tensor_

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    variance = std * std
    if variance == 0:
        variance = 1.0
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


###############################################################
# Basic blocks
###############################################################


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1, keepdim=True)
        inp = torch.cat([img, mean_c], dim=1)
        x = self.conv1(inp)
        illu_fea = self.depth_conv(x)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IG_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in, illu_fea_trans):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        illu = illu_fea_trans.reshape(b, h * w, c)
        q, k, v, illu = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q, k, v, illu)
        )
        v = v * illu
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(
            v.reshape(b, self.num_heads * self.dim_head, h, w)
        ).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


###############################################################
# IGAB with optional LCA residual
###############################################################


class MixedIGAB(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_blocks=2,
        use_lca=False,
        downsample_lca=False,
        lca_gate_scale=0.1,
        lca_gate_init=0.2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.use_lca = use_lca
        self.downsample_lca = downsample_lca
        self.lca_gate_scale = lca_gate_scale
        if downsample_lca:
            self.pool = nn.AvgPool2d(2, stride=2)
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.pool = self.upsample = None

        for _ in range(num_blocks):
            block = nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ])
            self.blocks.append(block)

        if self.use_lca:
            self.lca_blocks = nn.ModuleList([LCA(dim=dim, num_heads=heads, bias=False)
                                             for _ in range(num_blocks)])
            init = float(lca_gate_init)
            self.lca_gates = nn.ParameterList(
                [nn.Parameter(torch.tensor(init)) for _ in range(num_blocks)]
            )

    def forward(self, x, illu_fea):
        x_last = x.permute(0, 2, 3, 1)
        x_first = x
        illu_last = illu_fea.permute(0, 2, 3, 1)

        for idx, (attn, ff) in enumerate(self.blocks):
            ig_out = attn(x_last, illu_fea_trans=illu_last)

            if self.use_lca:
                if self.downsample_lca and x_first.shape[-1] >= 4 and x_first.shape[-2] >= 4:
                    x_low = self.pool(x_first)
                    illu_low = self.pool(illu_fea)
                    lca_low = self.lca_blocks[idx](x_low, illu_low)
                    lca_out = self.upsample(lca_low)
                else:
                    lca_out = self.lca_blocks[idx](x_first, illu_fea)
                delta = lca_out - x_first
                gate = self.lca_gate_scale * torch.tanh(self.lca_gates[idx])
                ig_out = ig_out + gate * delta.permute(0, 2, 3, 1)

            x_last = x_last + ig_out
            x_last = ff(x_last) + x_last
            x_first = x_last.permute(0, 3, 1, 2)

        return x_first


###############################################################
# Denoiser with Euler primary gates + selective LCA
###############################################################


class Denoiser(nn.Module):
    def __init__(
        self,
        in_dim=3,
        out_dim=3,
        dim=31,
        level=2,
        num_blocks=None,
        lca_encoder_layers: Sequence[int] = (),
        lca_decoder_layers: Sequence[int] = (),
        use_lca_bottleneck=True,
        downsample_lca=False,
        euler_cfg: Optional[Dict] = None,
        lca_gate_init=0.2,
        lca_gate_scale=0.1,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 4, 4]
        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        enc_set = set(lca_encoder_layers)
        dec_set = set(lca_decoder_layers)

        # encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                MixedIGAB(
                    dim=dim_level,
                    num_blocks=num_blocks[i],
                    dim_head=dim,
                    heads=max(dim_level // dim, 1),
                    use_lca=(i in enc_set),
                    downsample_lca=downsample_lca,
                    lca_gate_init=lca_gate_init,
                    lca_gate_scale=lca_gate_scale),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # bottleneck
        self.bottleneck = MixedIGAB(
            dim=dim_level,
            dim_head=dim,
            heads=max(dim_level // dim, 1),
            num_blocks=num_blocks[-1],
            use_lca=use_lca_bottleneck,
            downsample_lca=downsample_lca,
            lca_gate_init=lca_gate_init,
            lca_gate_scale=lca_gate_scale,
        )

        # decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                MixedIGAB(
                    dim=dim_level // 2,
                    num_blocks=num_blocks[self.level - 1],
                    dim_head=dim,
                    heads=max((dim_level // 2) // dim, 1),
                    use_lca=((self.level - 1 - i) in dec_set),
                    downsample_lca=downsample_lca,
                    lca_gate_init=lca_gate_init,
                    lca_gate_scale=lca_gate_scale),
            ]))
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # Euler gates
        default_euler_cfg = dict(
            enable_encoder_gate=True,
            enable_decoder_gate=True,
            encoder_init_alpha=0.06,
            decoder_init_alpha=0.01,
            max_gate_alpha=0.3,
        )
        if euler_cfg:
            default_euler_cfg.update(euler_cfg)
        self.enable_euler_enc = default_euler_cfg.get('enable_encoder_gate', True)
        self.enable_euler_dec = default_euler_cfg.get('enable_decoder_gate', True)
        self.max_euler_alpha = default_euler_cfg.get('max_gate_alpha', 0.3)
        self.euler_enc1 = EulerProc(input_dimension=dim)
        self.euler_dec_last = EulerProc(input_dimension=dim)
        self.alpha_euler_enc1 = nn.Parameter(
            torch.tensor(default_euler_cfg.get('encoder_init_alpha', 0.06)))
        self.alpha_euler_dec_last = nn.Parameter(
            torch.tensor(default_euler_cfg.get('decoder_init_alpha', 0.01)))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        fea = self.embedding(x)
        fea_encoder = []
        illu_list = []

        for idx, (IGAB_block, FeaDown, IlluDown) in enumerate(self.encoder_layers):
            fea = IGAB_block(fea, illu_fea)
            if idx == 0 and self.enable_euler_enc:
                alpha = torch.clamp(self.alpha_euler_enc1, 0.0, self.max_euler_alpha)
                fea = fea + alpha * (self.euler_enc1(fea) - fea)
            fea_encoder.append(fea)
            illu_list.append(illu_fea)
            fea = FeaDown(fea)
            illu_fea = IlluDown(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (FeaUp, Fusion, Block) in enumerate(self.decoder_layers):
            fea = FeaUp(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_list[self.level - 1 - i]
            fea = Block(fea, illu_fea)
            if i == self.level - 1 and self.enable_euler_dec:
                alpha = torch.clamp(self.alpha_euler_dec_last, 0.0, self.max_euler_alpha)
                fea = fea + alpha * (self.euler_dec_last(fea) - fea)

        return self.mapping(fea) + x


###############################################################
# Single stage / Multi-stage wrappers
###############################################################


class RPIFormerSingleStage(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=31,
        level=2,
        num_blocks=None,
        lca_encoder_layers: Sequence[int] = (),
        lca_decoder_layers: Sequence[int] = (),
        use_lca_bottleneck=True,
        downsample_lca=False,
        lca_gate_init=0.2,
        euler_cfg: Optional[Dict] = None,
        lca_gate_scale=0.1,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=[*num_blocks, num_blocks[-1]],
            lca_encoder_layers=lca_encoder_layers,
            lca_decoder_layers=lca_decoder_layers,
            use_lca_bottleneck=use_lca_bottleneck,
            downsample_lca=downsample_lca,
            euler_cfg=euler_cfg,
            lca_gate_init=lca_gate_init,
            lca_gate_scale=lca_gate_scale,
        )

    def forward(self, img):
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        return self.denoiser(input_img, illu_fea)


class RPIFormer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=31,
        stage=3,
        num_blocks=None,
        lca_encoder_layers: Sequence[int] = (),
        lca_decoder_layers: Sequence[int] = (),
        use_lca_bottleneck=True,
        downsample_lca=False,
        lca_gate_init=0.2,
        euler_cfg: Optional[Dict] = None,
        lca_gate_scale=0.1,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        modules = [
            RPIFormerSingleStage(
                in_channels=in_channels,
                out_channels=out_channels,
                n_feat=n_feat,
                level=2,
                num_blocks=num_blocks,
                lca_encoder_layers=lca_encoder_layers,
                lca_decoder_layers=lca_decoder_layers,
                use_lca_bottleneck=use_lca_bottleneck,
                downsample_lca=downsample_lca,
                 euler_cfg=euler_cfg,
                lca_gate_init=lca_gate_init,
                lca_gate_scale=lca_gate_scale,
            )
            for _ in range(stage)
        ]
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)
