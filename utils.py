import random
import torchvision.transforms.functional as TF
import torch

class SyncAugment:
    def __init__(
        self,
        angles=(0, 90, 180, 270),
        p_rotate=0.5,
        p_hflip=0.5,
        p_vflip=0.5,
    ):
        self.angles = angles
        self.p_rotate = p_rotate
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, src_x, tgt_x, src_y=None):
        do_hflip = random.random() < self.p_hflip
        do_vflip = random.random() < self.p_vflip
        do_rotate = random.random() < self.p_rotate
        angle = random.choice(self.angles) if do_rotate else None

        def apply_img(x):
            dtype = x.dtype

            if x.dim() == 4:          # (T,C,H,W) → dataset
                x = x.unsqueeze(0)   # (1,T,C,H,W)
                squeeze_b = True
            else:
                squeeze_b = False

            B, T, C, H, W = x.shape

            x = x.float()
            x = x.view(B * T, C, H, W)

            if do_hflip:
                x = TF.hflip(x)
            if do_vflip:
                x = TF.vflip(x)
            if angle is not None:
                x = TF.rotate(x, angle)

            x = x.view(B, T, C, H, W)

            if squeeze_b:
                x = x.squeeze(0)     # torna a (T,C,H,W)

            return x.to(dtype)

        def apply_label(y):
            if y.dim() == 2:
                y = y.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                squeeze = True
            elif y.dim() == 3:
                y = y.unsqueeze(1)               # (B,1,H,W)
                squeeze = False
            else:
                raise ValueError("Unexpected label shape")

            y = y.float()

            if do_hflip:
                y = TF.hflip(y)
            if do_vflip:
                y = TF.vflip(y)
            if angle is not None:
                y = TF.rotate(
                    y,
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST
                )

            y = y.long()

            if squeeze:
                y = y.squeeze(0).squeeze(0)      # (H,W)
            else:
                y = y.squeeze(1)                 # (B,H,W)

            return y

        src_x = apply_img(src_x)
        tgt_x = apply_img(tgt_x)

        if src_y is not None:
            src_y = apply_label(src_y)

        return src_x, tgt_x, src_y