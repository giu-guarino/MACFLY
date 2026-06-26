import torch.nn as nn
from src.utils.registry import Registry

# Registries definitions
MODELS = Registry("model")
BACKBONES = Registry("backbone")
NECKS = Registry("necks")
HEADS = Registry("heads")


class EncoderDecoder(nn.Module):
    def __init__(self, backbone, head, neck=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        if self.neck is not None:
            features = self.neck(features)
        out = self.head(features)
        return out


def build_backbone(cfg):
    # TODO: take into account kwargs
    # Best to do cfg_backbone rather than a global config
    cfg_backbone = cfg["backbone"]
    return BACKBONES.build(cfg_backbone)


def build_head(cfg):
    cfg_head = cfg["decode_head"]
    return HEADS.build(cfg_head)


def build_model(cfg_model):

    architecture = cfg_model.pop("type")
    if architecture == "EncoderDecoder":
        backbone = build_backbone(cfg_model)
        head = build_head(cfg_model)
        # TODO: add a possible neck
        model = EncoderDecoder(backbone, head)
        return model
    else:
        cfg = {"type": architecture, "model_config": cfg_model}
        return MODELS.build(cfg)
