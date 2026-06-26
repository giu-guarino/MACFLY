from functools import partial
from typing import Any, Callable, List, Optional
import torch
from torch import nn
from torchvision.models._api import WeightsEnum
from torchvision.utils import _log_api_usage_once

from einops.layers.torch import Rearrange
from torchvision.models.swin_transformer import (
    PatchMergingV2,
    SwinTransformerBlock,
    SwinTransformerBlockV2,
)
from einops import rearrange


# --------------------------------------------------------------------------------------
# The following functions are only adatpations, slight modifications of the existiong
# Implementation of the swin tranformer from torchvision
# https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py
# --------------------------------------------------------------------------------------


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
