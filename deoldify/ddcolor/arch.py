# Derived from DDColor (Apache-2.0) — https://github.com/piddnad/DDColor
# DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders (ICCV 2023)
import torch
import torch.nn as nn

from .unet import Hook, CustomPixelShuffle_ICNR, UnetBlockWide, NormType, custom_conv_layer
from .convnext import ConvNeXt
from .transformer_utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from .position_encoding import PositionEmbeddingSine


class DDColor(nn.Module):
    def __init__(
        self,
        encoder_name='convnext-l',
        decoder_name='MultiScaleColorDecoder',
        num_input_channels=3,
        input_size=(256, 256),
        nf=512,
        num_output_channels=3,
        last_norm='Weight',
        do_normalize=False,
        num_queries=256,
        num_scales=3,
        dec_layers=9,
    ):
        super().__init__()
        self.encoder = ImageEncoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'])
        self.encoder.eval()
        test_input = torch.randn(1, num_input_channels, *input_size)
        with torch.no_grad():
            self.encoder(test_input)

        self.decoder = DualDecoder(
            self.encoder.hooks, nf=nf, last_norm=last_norm,
            num_queries=num_queries, num_scales=num_scales,
            dec_layers=dec_layers, decoder_name=decoder_name
        )
        self.refine_net = nn.Sequential(
            custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False, norm_type=NormType.Spectral)
        )

        self.do_normalize = do_normalize
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.normalize(x)
        self.encoder(x)
        out_feat = self.decoder()
        coarse_input = torch.cat([out_feat, x], dim=1)
        out = self.refine_net(coarse_input)
        if self.do_normalize:
            out = self.denormalize(out)
        return out


class ImageEncoder(nn.Module):
    def __init__(self, encoder_name, hook_names):
        super().__init__()
        if encoder_name == 'convnext-t':
            self.arch = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        elif encoder_name == 'convnext-l':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError(f"Encoder {encoder_name} not supported. Use 'convnext-t' or 'convnext-l'.")
        self.hook_names = hook_names
        self.hooks = [Hook(self.arch._modules[name]) for name in hook_names]

    def forward(self, x):
        return self.arch(x)


class DualDecoder(nn.Module):
    def __init__(self, hooks, nf=512, blur=True, last_norm='Weight',
                 num_queries=256, num_scales=3, dec_layers=9, decoder_name='MultiScaleColorDecoder'):
        super().__init__()
        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)

        self.layers = self._make_layers()
        embed_dim = nf // 2
        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)
        self.color_decoder = MultiScaleColorDecoder(
            in_channels=[512, 512, 256],
            num_queries=num_queries, num_scales=num_scales, dec_layers=dec_layers,
        )

    def _make_layers(self):
        decoder_layers = []
        in_c = self.hooks[-1].feature.shape[1]
        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        for i, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if i == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlockWide(in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral)
            )
            in_c = out_c
        return nn.Sequential(*decoder_layers)

    def forward(self):
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        out3 = self.last_shuf(out2)
        return self.color_decoder([out0, out1, out2], out3)


class MultiScaleColorDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_queries=100, nheads=8,
                 dim_feedforward=2048, dec_layers=9, pre_norm=False,
                 color_embed_dim=256, enforce_input_project=True, num_scales=3):
        super().__init__()
        self.num_layers = dec_layers
        self.num_feature_levels = num_scales

        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(num_scales, hidden_dim)

        self.input_proj = nn.ModuleList()
        for in_ch in in_channels:
            if in_ch != hidden_dim or enforce_input_project:
                proj = nn.Conv2d(in_ch, hidden_dim, kernel_size=1)
                nn.init.kaiming_uniform_(proj.weight, a=1)
                if proj.bias is not None:
                    nn.init.constant_(proj.bias, 0)
                self.input_proj.append(proj)
            else:
                self.input_proj.append(nn.Sequential())

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(dec_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm))
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm))
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm))

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features):
        assert len(x) == self.num_feature_levels
        src, pos = [], []
        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i], None).flatten(2).permute(2, 0, 1))
            src.append((self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]).permute(2, 0, 1))

        _, bs, _ = src[0].shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index], memory_mask=None,
                memory_key_padding_mask=None, pos=pos[level_index], query_pos=query_embed)
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)
            output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output).transpose(0, 1)
        color_embed = self.color_embed(decoder_output)
        return torch.einsum("bqc,bchw->bqhw", color_embed, img_features)
