from src.utils.registry import Registry

TRAIN_STEP = Registry("train_step")

def build_train_step(cfg, **kwargs):
    cfg = {"type":cfg["RUNTIME"]["train_step"]}
    return TRAIN_STEP.build(cfg, default_args = kwargs)
    
