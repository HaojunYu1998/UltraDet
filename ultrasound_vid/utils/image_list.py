import torch
from detectron2.structures import ImageList
from torch.nn import functional as F


def imagelist_from_tensors(
    tensors, size_divisibility: int = 0, pad_value: float = 0.0
) -> "ImageList":
    """
    Args:
        tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
            (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
            to the same shape with `pad_value`.
        size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
            the common height and width is divisible by `size_divisibility`.
            This depends on the model and many models need a divisibility of 32.
        pad_value (float): value to pad

    Returns:
        an `ImageList`.
    """
    assert len(tensors) > 0
    assert isinstance(tensors, (tuple, list))
    for t in tensors:
        assert isinstance(t, torch.Tensor), type(t)
        assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
    buf = []
    import numpy as np

    for img in tensors:
        sp = tuple(img.shape)
        sp = np.array(sp).reshape(1, len(sp))
        buf.append(sp)
    buf = np.concatenate(buf, axis=0).max(axis=0)
    max_size = torch.from_numpy(buf)

    if size_divisibility > 0:
        stride = size_divisibility
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = torch.cat(
            [max_size[:-2], (max_size[-2:] + (stride - 1)) // stride * stride]
        )

    image_sizes = [tuple(im.shape[-2:]) for im in tensors]

    if len(tensors) == 1:
        image_size = image_sizes[0]
        padding_size = [
            0,
            max_size[-1] - image_size[1],
            0,
            max_size[-2] - image_size[0],
        ]
        if all(
            x == 0 for x in padding_size
        ):  # https://github.com/pytorch/pytorch/issues/31734
            batched_imgs = tensors[0].unsqueeze(0)
        else:
            padded = F.pad(tensors[0], padding_size, value=pad_value)
            batched_imgs = padded.unsqueeze_(0)
    else:
        # max_size can be a tensor in tracing mode, therefore use tuple()
        batch_shape = (len(tensors),) + tuple(max_size)
        batched_imgs = tensors[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return ImageList(batched_imgs.contiguous(), image_sizes)
