import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from src.models.TSViT.module import Attention, PreNorm, FeedForward
from src.models.TSViT.module import (
    FC_Classifier,
    FC_Classifier1,
    FC_Classifier2,
    FC_Classifier3,
)


# Transformer: Slightly changed to deal with attention matrices
def get_attention_hooks_and_maps(transformer):
    attn_maps = []

    def hook_fn(module, input, output):
        # Get the attention matrice in the module
        if hasattr(module, "last_attn") and any(
            x is not None for x in module.last_attn
        ):
            for x in module.last_attn:
                attn_maps.append(x.detach().cpu())

    handle = transformer.register_forward_hook(hook_fn)
    return handle, attn_maps


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
        x = x.permute(0, 1, 4, 2, 3)
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
        y = x.clone()

        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(
            B, self.num_classes, self.num_patches_1d**2, self.patch_size**2
        ).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, y)
        #return (x,)