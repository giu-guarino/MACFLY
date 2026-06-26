import random
import numpy as np
import torch
from .config_files_utils import read_yaml

# -----------------------------------------------------------------------------
# Taken from https://github.com/lhoyer/DAFormer
# Repository own by Lukas Hoyer
# -----------------------------------------------------------------------------

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
   
__all__ = ["read_yaml", "set_random_seed"]
