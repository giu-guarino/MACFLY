import numpy as np
import torch.nn.functional as F


def normalise_image(image, Xmin, Xmax):

    image[...] = np.where(image[...] > Xmax, Xmax, image[...])
    image[...] = np.where(image[...] < Xmin, Xmin, image[...])
    image[...] = (image[...] - Xmin) / (Xmax - Xmin)

    return image


# NOTE: Only for square matrices with the same stride in both directions
# TODO: Improve this function
def conv2D_output_shape(input_shape, kernel_size, stride, padding):
    return (input_shape - kernel_size + 2 * padding) / stride + 1


def input_shape(x):
    if x.ndim == 5:
        B, T, H, W, C = x.shape
        return B, H, W, C, T
    elif x.ndim == 4:
        B, C, H, W = x.shape
        return B, H, W, C


def resize_logits_to_labels(
    source_logits, labels, mode="bilinear", align_corners=False
):
    """
    Resize logits (B, H, W, C) to match the spatial size of labels (B, H, W, ...)
    and return a tensor in (B, H, W, C) format.
    """
    if labels.shape[:3] != source_logits.shape[:3]:
        # Permute from (B, H, W, C) to (B, C, H, W) for interpolation
        source_logits = source_logits.permute(0, 3, 1, 2)
        source_logits = F.interpolate(
            source_logits,
            size=labels.shape[1:3],  # Target height and width
            mode=mode,
            align_corners=align_corners,
        )
        source_logits = source_logits.permute(0, 2, 3, 1)

    return source_logits
