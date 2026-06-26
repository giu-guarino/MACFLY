import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from src.models.builder import MODELS
from src.models.TSViT.TSViT import Transformer
from src.models.TSViTSW.swin_transformer import _TSViT_adapted_swin_transformer
from src.models.TSViT.module import (
        GradReverse,
        grad_reverse,
        FC_Classifier,
        FC_Classifier1,
        FC_Classifier2,
        FC_Classifier3,
        calc_coeff,
        init_weights,
        RandomLayer,
        AdversarialNetwork,
        FC_Classifier_CDANN,
        Entropy,
        grl_hook,
        CDAN,
        )



# TODO: remove the hard-coded parts and use parameters
@MODELS.register_module()
class TSViTSW(nn.Module):
    """
    TSViT + Swin transformer for the spatial encoder
    """
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
            nn.Linear(2*self.dim, self.patch_size**2*2*2)  # NOTE: hard-coded
        )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
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
        
        x = self.mlp_head(x.reshape(-1, 2*self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x,)



@MODELS.register_module()
class TSViTSW_GRL_spa(nn.Module):
    """
    GRL spatial encoder
    token version
    """
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
        self.classifier = FC_Classifier3(n_classes = 2)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
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
        # x_flatten = y.view(x.shape[0],-1)  # au niveau de la classe et par image
        x_flatten = y.view(-1, self.dim*2)  #  au niveau du token
        x_flatten_adv = grad_reverse(x_flatten, self.alpha)
        domain_classification = self.classifier(x_flatten_adv)
        x = self.mlp_head(x.reshape(-1, 2*self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, domain_classification)


# --------------------------------------------------------------------------------------
#                                   EXPERIMENTS
# --------------------------------------------------------------------------------------

# Class and token vs token is just different tokens on which token the grl is apply



@MODELS.register_module()
class TSViTSW_GRL_temp(nn.Module):
    """
    GRL temporal encoder
    class and image version
    """
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
        self.classifier = FC_Classifier3(n_classes = 2)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
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
        y = x.clone()
        x_flatten = y.view(x.shape[0],-1)
        x_flatten_adv = grad_reverse(x_flatten, self.alpha)
        domain_classification = self.classifier(x_flatten_adv)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, 2*self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, domain_classification)


@MODELS.register_module()
class TSViTSW_GRL_tempspa(nn.Module):
    """
    GRL on the temporal encoder and on the spatial encoder 
    class and image version
    """
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
        self.classifier_temp = FC_Classifier2(n_classes = 2)
        self.classifier_spa = FC_Classifier2(n_classes = 2)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
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

        y = x.clone()
        x_flatten = y.view(x.shape[0],-1)
        x_flatten_adv = grad_reverse(x_flatten, self.alpha)
        domain_classification_temp = self.classifier_temp(x_flatten_adv)
        
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        y = x.clone()
        x_flatten = y.view(x.shape[0],-1)
        x_flatten_adv = grad_reverse(x_flatten, self.alpha)
        domain_classification_spa = self.classifier_spa(x_flatten_adv)
        x = self.mlp_head(x.reshape(-1, 2*self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, domain_classification_temp, domain_classification_spa)



@MODELS.register_module()
class TSViTSW_GRL_spa_dis_2_spa(nn.Module):
    """
    GRL + disentanglement(v2) on the spatial encoder
    v2
    """
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
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2*2*2)
        )
        self.alpha = model_config["alpha"]
        self.classifier = FC_Classifier3(n_classes = 2)

        self.spc_classifier  =FC_Classifier3(n_classes=2)
    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
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
        x = x.view(-1,x.shape[3])
        x, x_spc_dis = x[:,:self.dim], x[:,self.dim:]
        x_inv_dis = x
        spc_dom = self.spc_classifier(x_spc_dis)
        x_adv = x.clone()
        x_adv = grad_reverse(x_adv, self.alpha)
        domain_classification = self.classifier(x_adv)
        x = self.mlp_head(x)
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, domain_classification, x_inv_dis, x_spc_dis, spc_dom)


# disentanglement on the kept tokens
@MODELS.register_module()
class TSViTSW_GRL_spa_dis_2_temp(nn.Module): 
    """
    GRL spatial encoder + disentanglement temporal encoder
    """
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
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim//2))
        self.window_size = model_config["window_size"]
        self.space_transformer = _TSViT_adapted_swin_transformer(
            patch_size=[self.patch_size, self.patch_size],
            embed_dim=self.dim // 2,
            depths=self.spatial_depth,
            num_heads=[self.heads],
            window_size=self.window_size,
            stochastic_depth_prob=0,
            weights=None,
            progress=True,
        )
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2*2*2)
        )
        self.alpha = model_config["alpha"]
        self.classifier = FC_Classifier3(n_classes = 2)
        self.spc_classifier  =FC_Classifier3(n_classes=2)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
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
        x, x_spc_dis = x[:, :,:self.dim//2], x[:, :,self.dim//2:]
        x_inv_dis, x_spc_dis = x.reshape(-1, self.dim//2), x_spc_dis.reshape(-1, self.dim//2)
        spc_dom = self.spc_classifier(x_spc_dis)
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim//2).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim//2)
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        y = x.clone()
        x_flatten = y.view(-1, self.dim)  #  token level
        x_flatten_adv = grad_reverse(x_flatten, self.alpha)
        domain_classification = self.classifier(x_flatten_adv)
        
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, domain_classification, x_inv_dis, x_spc_dis, spc_dom)


@MODELS.register_module()
class TSViTSW_GRL_spa_dis(nn.Module):
    """
    It seems to be the first tokens version
    """
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
        self.classifier = FC_Classifier3(n_classes = 2) 
        self.spc_classifier  =FC_Classifier3(n_classes=2)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
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
        x_inv_dis = x[:, :self.num_classes]
        x_spc_dis = x[:,self.num_classes:]
        x_inv_dis = x_inv_dis.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim)
        x_spc_dis = x_spc_dis.reshape(B, self.num_patches_1d**2, -1, self.dim)
        x_inv_dis = torch.mean(x_inv_dis, dim=2).view(B*self.num_patches_1d**2,-1)
        x_spc_dis = torch.mean(x_spc_dis, dim=2).view(B*self.num_patches_1d**2,-1)
        spc_dom = self.spc_classifier(x_spc_dis)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        y = x.clone()
        # x_flatten = y.view(x.shape[0],-1)  # au niveau de la classe et par image
        x_flatten = y.view(-1, self.dim*2)  #  au niveau du token
        x_flatten_adv = grad_reverse(x_flatten, self.alpha)
        domain_classification = self.classifier(x_flatten_adv)
        x = self.mlp_head(x.reshape(-1, 2*self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, domain_classification, x_inv_dis, x_spc_dis, spc_dom)

@MODELS.register_module()
class TSViTSW_DC_GRL(nn.Module):
    """
    Domain Critic + GRL
    """
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
        self.classifier = FC_Classifier3(n_classes = 2)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
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
        x_flatten = y.view(-1, self.dim*2)
        x_flatten_adv = grad_reverse(x_flatten, self.alpha)
        domain_classification = self.classifier(x_flatten_adv)
        tokens = x.clone()
        tokens = x.view(-1, x.shape[-1])
        x = self.mlp_head(x.reshape(-1, 2*self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, tokens, domain_classification)

@MODELS.register_module()
class TSViTSW_DC(nn.Module):
    """
    DOMAIN CRITIC
    """
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

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
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
        tokens = x.clone()
        tokens = x.view(-1, x.shape[-1])
        x = self.mlp_head(x.reshape(-1, 2*self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return (x, tokens)



# # TODO: check if there's a mistake or it just doesn't work
# # A lot of necessary code was commenti, changed  to look for a potential issue, 
# # To not use in this state the following, MAE and CDANN
# 
# from .swin_transformer import _TSViT_adapted_swin_transformer, PatchExpanding
# from timm.models.vision_transformer import PatchEmbed, Block
# from src.uda.pos_embed import get_2d_sincos_pos_embed
# class TSViTSW_MAE(nn.Module):
#     """
#     Masked autoencoder
#     """
#     def __init__(self, model_config):
#         super().__init__()
#         # print('init')
#         self.image_size = model_config['img_res']
#         self.patch_size = model_config['patch_size']
#         self.num_patches_1d = self.image_size//self.patch_size
#         self.num_classes = model_config['num_classes']
#         self.num_frames = model_config['max_seq_len']
#         self.dim = model_config['dim']
#         if 'temporal_depth' in model_config:
#             self.temporal_depth = model_config['temporal_depth']
#         else:
#             self.temporal_depth = model_config['depth']
#         if 'spatial_depth' in model_config:
#             self.spatial_depth = model_config['spatial_depth']
#         else:
#             self.spatial_depth = model_config['depth']
#         self.heads = model_config['heads']
#         self.dim_head = model_config['dim_head']
#         self.dropout = model_config['dropout']
#         self.emb_dropout = model_config['emb_dropout']
#         self.pool = model_config['pool']
#         self.scale_dim = model_config['scale_dim']
#         assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#         num_patches = self.num_patches_1d ** 2
#         self.to_temporal_embedding_input = nn.Linear(366, self.dim)
#         self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
#         self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
#                                                 self.dim * self.scale_dim, self.dropout)
# 
#         self.window_size = model_config["window_size"]
#         self.space_transformer = _TSViT_adapted_swin_transformer(
#             patch_size=[self.patch_size, self.patch_size],
#             embed_dim=self.dim,
#             depths=self.spatial_depth,
#             num_heads=[self.heads],
#             window_size=self.window_size,
#             stochastic_depth_prob=0,
#             weights=None,
#             progress=True,
#         )
# 
#         self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
#         self.dropout = nn.Dropout(self.emb_dropout)
#         
#         patch_dim = (3) * self.patch_size ** 2  # -1 is set to exclude time feature # 3 car RGB
# 
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
#             nn.Linear(patch_dim, self.dim),)
#         
#         # Added
#         self.mask_token = nn.Parameter(torch.zeros(1,1,  1, self.dim))  # 128 hard-coded 
#         self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches_1d**2, 128), requires_grad=False)
#         self.patch_expanding = PatchExpanding(dim = self.dim * 2, norm_layer = nn.LayerNorm) 
#         self.transformer_decoder = Transformer(self.dim, 4, self.heads, self.dim_head,  self.dim * self.scale_dim, 0.0)
# 
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(self.dim),
#             nn.Linear(self.dim, self.patch_size**2*1*3)
#             )
#         self.norm_pix_loss = False
#         self.decoder_embed = nn.Linear(128,128, bias=True)
#         self.decoder_norm = nn.LayerNorm(128)
#         self.decoder_blocks = nn.ModuleList([
#             Block(128,4, 4, qkv_bias=True, norm_layer=nn.LayerNorm)
#             for i in range(4)])
#         self.blocks = nn.ModuleList([
#             Block(128, 4, 4, qkv_bias=True,  norm_layer=nn.LayerNorm)
#             for i in range(4)])
# 
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1024,1, 128), requires_grad=False)
#         self.norm = nn.LayerNorm(128)
# 
#         self.num_patches = 32*32 
#         self.initialize_weights()
#         
#     def initialize_weights(self):
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5)) # cls_token was to True
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0).unsqueeze(2))
#         decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0).unsqueeze(1))
# 
#         # Right now, commented
#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#         # w = self.patch_embed.proj.weight.data
#         # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
# 
#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         # torch.nn.init.normal_(self.cls_token, std=.02)
#         # torch.nn.init.normal_(self.mask_token, std=.02)
# 
#         self.apply(self._init_weights)
# 
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
# 
# 
# 
#     def forward(self, x):
#         gt = x[:,0,:,:,:3].unsqueeze(1)
#         x = x.permute(0, 1, 4, 2, 3)
#         # pas besoin de temporal embedding a priori
#         # x = torch.cat([x[:,0,:3,:,:].unsqueeze(1),x[:,0,-1,:,:].unsqueeze(1).unsqueeze(2)])
# 
#         x = x[ :,0,:3].unsqueeze(1)  # -> une seule date, 3 bandes
#         B, T, C, H, W = x.shape  # [4, 1, 3, 64, 64]
# 
#         # xt = x[:, :, -1, 0, 0]
#         # x = x[:, :, :-1] # on enlève la 7ème bande, Déjà enlevé plus haut
#         # xt = (xt * 365.0001).to(torch.int64)
#         # xt = F.one_hot(xt, num_classes=366).to(torch.float32)
#         # xt = xt.reshape(-1, 366)
#         # temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
# 
#         x = self.to_patch_embedding(x)  # [4096, 1, 128]
# 
#         x = x.reshape(B, -1, T, self.dim)  # [4, 1024, 1, 128]
# 
#         x += self.pos_embed
#         # x += temporal_pos_embedding.unsqueeze(1)
#         x, mask, ids_restore, ids_keep = self.random_masking(x, 0.75) #  0.288085) #  .75)  # [4, 256, 1, 128] # 0.288085 et 27
# 
#         # Mauvais placement
#         # x = x+self.pos_embed
#         
#         x = x.reshape(-1, T, self.dim) # [1024, 1, 128]
# 
#         # modfifié -> taille réduite
#         # x.shape[0] = B * self.num_patches_1d **2 * 0.75
#         num_visible_tokens = x.shape[0] // B # 256 avec un ration de 0.75 de tokens masqués
#         # A priori, je ne sais pas s'il faut que je les garde
#         # cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b= B * num_visible_tokens)  # [1024, 10,128] # avant b=B * self.num_patches_1d ** 2)
#         # x = torch.cat((cls_temporal_tokens, x), dim=1)
#         # Encoder
#         # for blk in self.blocks:
#         #     x = blk(x)
#         # x = self.norm(x)
#         x = self.temporal_transformer(x) # [1024, 1, 128]
# 
# 
# 
#         # Space transformer enlevé
#         # x = x[:, :self.num_classes]
#         # # print(x.shape )
#         # 
#         # # modifié -> num_visible_tokens 
#         # x = x.reshape(B, num_visible_tokens, self.num_classes, self.dim)
#         # x = x.permute(0, 2, 1, 3)
#         # x = x.reshape(B*self.num_classes, num_visible_tokens, self.dim)
#         # space_pos_embedding = self.space_pos_embedding.repeat(B,1,1)
#         # masked_space_pos_embedding = torch.gather(space_pos_embedding,dim=1,index=ids_keep.unsqueeze(-1).repeat(1,1,self.dim))
#         # masked_space_pos_embedding = torch.repeat_interleave(masked_space_pos_embedding,torch.tensor(self.num_classes).to(x.device), dim=0) 
#         # x += masked_space_pos_embedding#self.space_pos_embedding
#         # x = self.dropout(x)
#         # x = self.space_transformer(x)
# 
# 
#         x = self.decoder_embed(x)  # [1024,1,128]
#         # x = self.patch_expanding(x)
#         # x = x.reshape(B,self.num_classes,-1, x.shape[3])  # normalement, avant
#         x = rearrange(x, "(B N) L d -> B L N d", B=B)  # [4, 1, 256, 128]
#         # utiliser les 10 classes ou seulement une dans le décodeur?
#         # si 10 classes 4D par la suite
#         # x = x[:,0,:,:]
#         # print(f"patch expanding output {x.shape}")
#         # print(x.shape, ids_restore.shape)
# 
#         # ids_restore.shape[1] + 1 - x.shape[1], pourquoi +1?
#         mask_tokens = self.mask_token.repeat(x.shape[0],1,  (ids_restore.shape[1] - x.shape[2]), 1)  # [4, 1, 768, 128]
#         x = torch.cat([x[:,:, :, :], mask_tokens], dim=2)  # [4, 1, 768, 128]
#         x = torch.gather(x, dim=2, index=ids_restore.unsqueeze(1).unsqueeze(-1).repeat(1,1, 1, x.shape[3]))
#         x = x + self.decoder_pos_embed  # [4, 1, 1024, 128] et self.decoder_pos_embed [1, 1, 1024, 128]
#         x = rearrange(x, "b c n d -> (b n) c d")
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         x = self.decoder_norm(x)
#         # x = self.transformer_decoder(x)  # [4096, 1, 128]
#         x = rearrange(x, "(b n) c d -> b c n d", b=B )
#         x = rearrange(x, 'b c n d -> b n (c d)')  # [4, 1024, 128]
#         # x = x.reshape(-1, self.num_classes * x.shape[3])
#         x = self.mlp_head(x)  # [4, 1024, 12]  # .reshape(-1, 2*self.dim))
#         loss = self.forward_loss(gt, x, mask)
#         x = rearrange(x, 'b (h w) (p1 p2 t c) -> b t (h p1) (w p2) c', t=1, c=3, h=32, p1=2, p2=2)
#         # x = x.reshape(B, T,C-1, H,W)#self.num_patches_1d**2 * self.patch_size**2) # .permute(0, 2, 3, 1)
#         # x = x.reshape(B, H, W, self.num_classes)
#         # x = x.permute(0, 3, 1, 2)
#         return (x , loss, mask)
# 
# 
#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size**2 *3)
#         imgs: (N, 3, H, W)
#         """
#         p = self.patch_embed.patch_size[0]
#         h = w = int(x.shape[1]**.5)
#         assert h * w == x.shape[1]
#         
#         x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
#         return imgs 
#    
#     def patchify(self, imgs):
#         """
#         imgs: [B,T,H,W,C] #  C=7, +1 -> temporal embedding
#         x: (B, H*W, p1*p2*T*(C-1))
#         """
#         # p = self.patch_embed.patch_size[0]
#         B,T,H,W,C = imgs.shape
#         x = rearrange(imgs, "B T (h p1) (w p2) C -> B (h w) (p1 p2 T C)", p1=2,p2=2) 
#         
#         # TODO: check if it does the same thing
#         # x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
#         # x = torch.einsum('nchpwq->nhwpqc', x)
#         # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
#         return x
# 
# 
#     def forward_loss(self, imgs, pred, mask):
#         """
#         imgs: [B,T,H,W,C+1]
#         pred: [B, H*W, p1*p2*T*C] 
#         mask: [B, H*W], 0 is keep, 1 is remove,
#         """
#         target = self.patchify(imgs)
#         if self.norm_pix_loss:
#             print("pas encore implémenté")
#             exit()
#             # mean = target.mean(dim=-1, keepdim=True)
#             # var = target.var(dim=-1, keepdim=True)
#             # target = (target - mean) / (var + 1.e-6)**.5
# 
#         loss = (pred - target) ** 2
#         loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
# 
#         loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
#         return loss
# 
#     def forward(self, imgs, mask_ratio=0.75):
#         latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
#         pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
#         loss = self.forward_loss(imgs, pred, mask)
#         return loss, pred, mask
# 
#     def random_masking(self, x, mask_ratio):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [B,N, T, D], sequence
#         """
# 
#         B,N, T, D = x.shape  # batch, length=num_tokens, time indice, dim
#         len_keep = int(N * (1 - mask_ratio))
# 
#         noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
# 
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
# 
#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
# 
#         # crucial to repeat otherwise repeat will select only the first value of the third dim
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1,T, D)) 
# 
#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([B, N], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)
#         return x_masked, mask, ids_restore, ids_keep




# @MODELS.register_module()
# class TSViTSW_GRL_CDANN_spa(nn.Module):
#     """
#     GRL spatial encoder using CDANN
#     """
#     def __init__(self, model_config):
#         super().__init__()
#         self.image_size = model_config['img_res']
#         self.patch_size = model_config['patch_size']
#         self.num_patches_1d = self.image_size//self.patch_size
#         self.num_classes = model_config['num_classes']
#         self.num_frames = model_config['max_seq_len']
#         self.dim = model_config['dim']
#         if 'temporal_depth' in model_config:
#             self.temporal_depth = model_config['temporal_depth']
#         else:
#             self.temporal_depth = model_config['depth']
#         if 'spatial_depth' in model_config:
#             self.spatial_depth = model_config['spatial_depth']
#         else:
#             self.spatial_depth = model_config['depth']
#         self.heads = model_config['heads']
#         self.dim_head = model_config['dim_head']
#         self.dropout = model_config['dropout']
#         self.emb_dropout = model_config['emb_dropout']
#         self.pool = model_config['pool']
#         self.scale_dim = model_config['scale_dim']
#         assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#         num_patches = self.num_patches_1d ** 2
#         patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
#             nn.Linear(patch_dim, self.dim),)
#         self.to_temporal_embedding_input = nn.Linear(366, self.dim)
#         self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
#         self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
#                                                 self.dim * self.scale_dim, self.dropout)
#         self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
#         self.window_size = model_config["window_size"]
#         self.space_transformer = _TSViT_adapted_swin_transformer(
#             patch_size=[self.patch_size, self.patch_size],
#             embed_dim=self.dim,
#             depths=self.spatial_depth,
#             num_heads=[self.heads],
#             window_size=self.window_size,
#             stochastic_depth_prob=0,
#             weights=None,
#             progress=True,
#         )
#         self.dropout = nn.Dropout(self.emb_dropout)
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(2*self.dim),
#             nn.Linear(2*self.dim, self.patch_size**2*2*2)
#         )
#         self.classifier = nn.Linear(self.num_classes, self.num_classes)
#         # self.alpha = model_config["alpha"]
#         # self.classifier = FC_Classifier2(n_classes = 2) # 2 domaines: source and target
#         #self.classifier = nn.Sequential()
#         #for i in range(1):
#         #    self.classifier.add_module("classifier"+str(i), self.mlp_head[i])
#    
#     def forward(self, x):
#         x = x.permute(0, 1, 4, 2, 3)
#         B, T, C, H, W = x.shape
# 
#         xt = x[:, :, -1, 0, 0]
#         x = x[:, :, :-1]
#         xt = (xt * 365.0001).to(torch.int64)
#         xt = F.one_hot(xt, num_classes=366).to(torch.float32)
# 
#         xt = xt.reshape(-1, 366)
#         temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
#         x = self.to_patch_embedding(x)
#         x = x.reshape(B, -1, T, self.dim)
#         x += temporal_pos_embedding.unsqueeze(1)
#         x = x.reshape(-1, T, self.dim)
#         cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
#         x = torch.cat((cls_temporal_tokens, x), dim=1)
#         x = self.temporal_transformer(x)
# 
#         
#         x = x[:, :self.num_classes]
#         x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
#         x += self.space_pos_embedding#[:, :, :(n + 1)]
#         x = self.dropout(x)
#         x = self.space_transformer(x)
# 
#         from einops import rearrange
#         f = x.clone()
#         # f = rearrange(f, '(B C) h w d -> (B h w) (C d)', B=B)
#         # features = features.reshape(-1,2*self.dim)
#         # features = features.reshape(B, 
#         # features = features.reshape(B, self.num_classes, 4,1).permute(0, 2, 3, 1)
#         # -----------------------------------------------------------------------------
#         # features = features.reshape(B, H, W, self.num_classes)
#         # features = features.permute(0, 3, 1, 2)
#         # print(features.shape)
#         # features = self.classifier(features)
#         # features = features.view(x.shape[0],-1)
#         # x_flatten_adv = grad_reverse(x_flatten, self.alpha)
#         # domain_classification = self.classifier(x_flatten_adv)
#         
#         x = self.mlp_head(x.reshape(-1, 2*self.dim))# mod 2 *
#         x = x.reshape(B, self.num_classes, self.num_patches_1d**2 //2//2, self.patch_size**2 *2*2).permute(0, 2, 3, 1)
#         x = x.reshape(B, H, W, self.num_classes)
#         
#         f = rearrange(x.clone(), "b h w c -> (b h w) c")
#         g =  self.classifier(f)
#         x = rearrange(g, "(b h w) c -> b c h w", b=B, h=H)
#         g =  nn.Softmax(dim=1)(g)
#         
#         # collapse
#         # ----------------------------------------------------------
#         # f = rearrange(x.clone(), "b h w c -> (b h w) c")
#         # x =  self.classifier(x)
#         # g =  rearrange(nn.Softmax(dim=3)(x), "b h w c -> (b h w) c")
#         # x = x.permute(0, 3, 1, 2)
#         # ----------------------------------------------------------
# 
#         # g = rearrange(nn.Softmax(dim=1)(x), 'B C (h p1) (w p2) -> (B h w) (C p1 p2)', p1=self.patch_size, p2=self.patch_size)
#         # x equimalent segmask above
#         return (x,(f,g))  # x, (f,g) 
#         #(nn.Softmax(dim=1)(x).view(B, -1), features.view(B, -1))#, temp_attn_mat_list, space_attn_mat_list
# # -------------------------------------------------------------------------------------
# 
