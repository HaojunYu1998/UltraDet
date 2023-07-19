import logging
import numpy as np
from PIL import Image, ImageEnhance
from typing import Any, List, Optional

from detectron2.data import transforms as T
from detectron2.data.transforms import Augmentation, ResizeTransform, PILColorTransform
from fvcore.transforms.transform import Transform


def check_frames_dtype(frames):
    for img in frames:
        assert isinstance(
            img, np.ndarray
        ), "[TransformGen] Needs an numpy array, but got a {}!".format(type(img))
        assert not isinstance(img.dtype, np.integer) or (
            img.dtype == np.uint8
        ), "[TransformGen] Got image of type {}, use uint8 or floating points instead!".format(
            img.dtype
        )
        assert img.ndim in [2, 3], img.ndim
        assert (
            img.shape == frames[0].shape
        ), "[TransformGen] Got inconsist image shapes from one video!"


def apply_augmentations_to_frames(transform_gens, frames):
    """
    Apply a list of :class:`Augmentation` on the input image, and
    returns the transformed image and a list of transforms.

    Args:
        transform_gens (list): list of :class:`TransformGen` instance to
            be applied.
        frames (list of ndarray): uint8 or floating point images with 1 or 3 channels.

    Returns:
        ndarray: the transformed image
        TransformList: contain the transforms that's used.
    """
    for g in transform_gens:
        assert isinstance(g, Augmentation), g

    check_frames_dtype(frames)

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(frames[0])
        assert isinstance(
            tfm, T.Transform
        ), "Augmentation {} must return an instance of Transform! Got {} instead".format(
            g, tfm
        )
        frames = [tfm.apply_image(f) for f in frames]
        tfms.append(tfm)
    return frames, T.TransformList(tfms)


def build_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)

    augmentation = []
    if is_train:
        scale_range = cfg.INPUT.SCALE_TRAIN
        fixed_area = cfg.INPUT.FIXED_AREA_TRAIN
        augmentation.append(ResizeByScale(scale_range, fixed_area))
        augmentation.append(
            ColorJitter({"contrast": 0.3, "sharpness": 0.3, "color": 0.3})
        )
        augmentation.append(T.RandomFlip())
        if cfg.INPUT.CROP.ENABLED:
            augmentation.append(RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        logger.info("Augmentations used in training: " + str(augmentation))
    else:
        scale_range = cfg.INPUT.SCALE_TEST
        fixed_area = cfg.INPUT.FIXED_AREA_TEST
        augmentation.append(ResizeByScale(scale_range, fixed_area))
        logger.info("Augmentations used in testing: " + str(augmentation))

    return augmentation


class ResizeByScale(Augmentation):
    """Resize image By Scale"""

    def __init__(self, scale_range, fixed_area, interp=Image.BILINEAR):
        """
        Args:
            scale_range: float of tuple of floats
            interp: PIL interpolation method
        """
        if isinstance(scale_range, float):
            scale_range = (scale_range, scale_range)
        scale_range = tuple(scale_range)
        self._init(locals())

    def get_transform(self, img):
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        if self.fixed_area > 0:
            shape = img.shape
            scale *= np.sqrt(self.fixed_area / (shape[0] * shape[1]))
        new_shape = (int(img.shape[0] * scale), int(img.shape[1] * scale))

        return ResizeTransform(
            img.shape[0], img.shape[1], new_shape[0], new_shape[1], self.interp
        )


transform_type_dict = dict(
    brightness=ImageEnhance.Brightness,
    contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness,
    color=ImageEnhance.Color,
)


class ColorJitterFunc(object):
    def __init__(self, transform_dict, rand_num):
        self.transforms = [
            (transform_type_dict[k], transform_dict[k]) for k in transform_dict
        ]
        self.rand_num = rand_num

    def __call__(self, img):
        out = img

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (self.rand_num[i] * 2.0 - 1.0) + 1  # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)

        return out


class ColorJitter(Augmentation):
    """Resize image By Scale"""

    def __init__(self, transform_dict):
        """
        Args:
            transform_dict: methods and alpha you want to use for color jittering
        """
        self.transform_dict = transform_dict

    def get_transform(self, img):
        rand_num = np.random.uniform(0, 1, len(self.transform_dict))
        func = ColorJitterFunc(self.transform_dict, rand_num)
        return PILColorTransform(func)


class RandomCrop(Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size=[0.9, 1.1]):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, min_crop_rate and max_crop_rate
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        h_list = [i for i in range(0, h - croph + 1)] if croph < h else [i for i in range(h - croph, 1)]
        w_list = [i for i in range(0, w - cropw + 1)] if cropw < w else [i for i in range(w - cropw, 1)]
        h0 = int(np.random.choice(h_list, 1))
        w0 = int(np.random.choice(w_list, 1))
        return CropTransform(w0, h0, cropw, croph, w, h)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        crop_size = np.asarray(self.crop_size, dtype=np.float32)
        rate = np.random.uniform(crop_size[0], crop_size[1], 2)
        return int(h * rate[0] + 0.5), int(w * rate[1] + 0.5)


class CropTransform(Transform):
    def __init__(
        self,
        x0: int,
        y0: int,
        w: int,
        h: int,
        orig_w: Optional[int] = None,
        orig_h: Optional[int] = None,
    ):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
            orig_w, orig_h (int): optional, the original width and height
                before cropping. Needed to make this transform invertible.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        if (self.w <= self.orig_w) and (self.h <= self.orig_h):
            assert (self.x0 >= 0) and (self.y0 >= 0)
            if len(img.shape) <= 3:
                return img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]
            else:
                return img[..., self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w, :]
        else:
            # pad_coords in ori_img coordinate system
            pad_x0 = min(self.x0, 0)
            pad_y0 = min(self.y0, 0)
            pad_x1 = max(self.x0 + self.w, self.orig_w)
            pad_y1 = max(self.y0 + self.h, self.orig_h)
            pad_w, pad_h = pad_x1 - pad_x0, pad_y1 - pad_y0
            # tranform ori_coords to pad_img coordinate system
            new_x0 = 0 - pad_x0
            new_y0 = 0 - pad_y0
            new_x1 = self.orig_w - pad_x0
            new_y1 = self.orig_h - pad_y0
            # transform crop_coords to pad_img coordinate system
            crop_x0 = self.x0 - pad_x0
            crop_y0 = self.y0 - pad_y0
            crop_x1 = self.x0 + self.w - pad_x0
            crop_y1 = self.y0 + self.h - pad_y0
            img_shape = list(img.shape)
            # print(new_x0, new_y0, new_x1, new_y1, crop_x0, crop_y0, crop_x1, crop_y1)
            if len(img.shape) <= 3:
                img_shape[0] = pad_h; img_shape[1] = pad_w
                # assert False, f"{tuple(img_shape)}"
                new_img = np.zeros(shape=tuple(img_shape))
                new_img[new_y0: new_y1, new_x0 : new_x1] = img
                return new_img[crop_y0: crop_y1, crop_x0: crop_x1]
            else:
                img_shape[-3] = pad_h; img_shape[-2] = pad_w
                # assert False, f"{tuple(img_shape)}"
                new_img = np.zeros(shape=tuple(img_shape))
                new_img[..., new_y0: new_y1, new_x0 : new_x1, :] = img
                return new_img[..., crop_y0: crop_y1, crop_x0: crop_x1, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords