import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from einops import rearrange

from functools import partial
from typing import Any, Callable, List, Optional
from torchvision.models._api import WeightsEnum
from torchvision.utils import _log_api_usage_once

from torchvision.models.swin_transformer import (
    PatchMergingV2,
    SwinTransformerBlock,
    SwinTransformerBlockV2,
)

class FC_Classifier(torch.nn.Module):
    def __init__(self, in_dim, n_classes):
        super(FC_Classifier, self).__init__()

        self.block = nn.Sequential(
            #nn.LazyLinear(256),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            #nn.LazyLinear(512),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            #nn.LazyLinear(n_classes),
            nn.Linear(512, n_classes),
        )

    def forward(self, X):
        return self.block(X)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out, attn  # for attention matrices


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )
        self.last_attn = [None] * depth

    def forward(self, x):

        for i, (attn, ff) in enumerate(self.layers):
            attn_x = attn(x)
            self.last_attn[i] = attn_x[1]
            x = attn_x[0] + x
            x = ff(x) + x

        return self.norm(x)



class TSViTAdaptedSwinTransformer(nn.Module):
    """
    This class is a direct adaptation of the original code from torchvision
    With a few modification in order to be used with the TSViT
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[List[int]]): List of Window size for each stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[List[int]],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMergingV2,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        # TODO: make parameters, not hard-coded
        layers.append(
            nn.Sequential(
                Rearrange("b (h w) d -> b h w d", h=32, w=32),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                stage.append(
                    block(
                        dim,
                        num_heads[0],  # [i_stage]
                        window_size=window_size[i_stage],
                        # NOTE: Experiments
                        # window_size[i_stage] if i_layer == 0 else [window_size[i_stage][0] // 2, window_size[i_stage][1] // 2],
                        # {
                        #     i_layer == 0: window_size[i_stage],
                        #     i_layer
                        #     == 1: [
                        #         window_size[i_stage][0] // 2,
                        #         window_size[i_stage][1] // 2,
                        #     ],
                        # }.get(
                        #     True,
                        #     [
                        #         window_size[i_stage][0] // 4,
                        #         window_size[i_stage][1] // 4,
                        #     ],
                        # ),
                        # window_size[i_stage] if i_layer % 2 == 0 else [window_size[i_stage][0]//2,window_size[i_stage][1]//2],
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2
                            for w in window_size[i_stage]
                        ],
                        # NOTE: Experiments
                        # [
                        #    0 for w in window_size[i_stage]
                        # ],  # pas de shifted window,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return x


def _TSViT_adapted_swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[List[int]],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> TSViTAdaptedSwinTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = TSViTAdaptedSwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        block=SwinTransformerBlockV2,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model


# For MAE
class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, "B H W (P1 P2 C) -> B (H P1) (W P2) C", P1=2, P2=2)
        x = self.norm(x)
        return x


class TSViT(nn.Module):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    For improved training speed, this implementation uses a (365 x dim) temporal position encodings indexed for
    each day of the year. Use TSViT_lookup for a slower, yet more general implementation of lookup position encodings
    """

    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config["img_res"]
        self.patch_size = model_config["patch_size"]
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = model_config["num_classes"]
        self.num_frames = model_config["max_seq_len"]
        self.dim = model_config["dim"]
        if "temporal_depth" in model_config:
            self.temporal_depth = model_config["temporal_depth"]
        else:
            self.temporal_depth = model_config["depth"]
        if "spatial_depth" in model_config:
            self.spatial_depth = model_config["spatial_depth"]
        else:
            self.spatial_depth = model_config["depth"]
        self.heads = model_config["heads"]
        self.dim_head = model_config["dim_head"]
        self.dropout = model_config["dropout"]
        self.emb_dropout = model_config["emb_dropout"]
        self.pool = model_config["pool"]
        self.scale_dim = model_config["scale_dim"]
        assert self.pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        num_patches = self.num_patches_1d**2
        patch_dim = (
            model_config["num_channels"] - 1
        ) * self.patch_size**2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            ),
            nn.Linear(patch_dim, self.dim),
        )
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(
            self.dim,
            self.temporal_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout,
        )
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(
            self.dim,
            self.spatial_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout,
        )
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.patch_size**2)
        )

    def forward(self, x):
        #x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)

        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(
            B, T, self.dim
        )
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(
            self.temporal_token, "() N d -> b N d", b=B * self.num_patches_1d**2
        )
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x[:, : self.num_classes]
        x = (
            x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_classes, self.num_patches_1d**2, self.dim)
        )
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)

        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(
            B, self.num_classes, self.num_patches_1d**2, self.patch_size**2
        ).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x,)



class MACFLY(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.window_size = model_config["window_size"]
        self.space_transformer = _TSViT_adapted_swin_transformer(
            patch_size=[self.patch_size, self.patch_size],
            embed_dim=self.dim,
            depths=self.spatial_depth,
            num_heads=[self.heads],
            window_size=self.window_size,
            stochastic_depth_prob=0,
            weights=None,
            progress=True,
        )
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(2*self.dim),
            nn.Linear(2*self.dim, self.patch_size**2*2*2)
        )
        self.alpha = model_config["alpha"]
        self.classifier = FC_Classifier(in_dim= 2 * self.dim,  n_classes = 2)

    def forward(self, x):
        #x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)

        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)

        
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        y = x.clone()

        x = self.mlp_head(x.reshape(-1, 2*self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, y)

