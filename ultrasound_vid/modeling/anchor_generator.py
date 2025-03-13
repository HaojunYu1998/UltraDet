# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import List

import torch
from torch import nn

from detectron2.layers import ShapeSpec
# from detectron2.modeling.anchor_generator import _create_grid_offsets


def _create_grid_offsets(size: List[int], stride: int, offset: float, device_tensor: torch.device):
    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device_tensor.device
    )
    shifts_y = torch.arange(
        offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device_tensor.device
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


class ShiftGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of shifts.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.num_shifts = cfg.MODEL.SHIFT_GENERATOR.NUM_SHIFTS
        self.strides    = [x.stride for x in input_shape]
        self.offset     = cfg.MODEL.SHIFT_GENERATOR.OFFSET
        # fmt: on

        self.num_features = len(self.strides)

    @property
    def num_cell_shifts(self):
        return [self.num_shifts for _ in self.strides]

    def grid_shifts(self, grid_sizes, device_tensor):
        shifts_over_all = []
        for size, stride in zip(grid_sizes, self.strides):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, device_tensor)
            shifts = torch.stack((shift_x, shift_y), dim=1)

            shifts_over_all.append(shifts.repeat_interleave(self.num_shifts, dim=0))

        return shifts_over_all

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate shifts.

        Returns:
            list[list[Tensor]]: a list of #image elements. Each is a list of #feature level tensors.
                The tensors contains shifts of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        shifts_over_all = self.grid_shifts(grid_sizes, features[0])

        shifts = [copy.deepcopy(shifts_over_all) for _ in range(num_images)]
        return shifts
