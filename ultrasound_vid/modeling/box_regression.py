# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch


class Shift2BoxTransform(object):
    def __init__(self, weights):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dl, dt, dr, db) deltas.
        """
        self.weights = weights

    def get_deltas(self, shifts, boxes):
        """
        Get box regression transformation deltas (dl, dt, dr, db) that can be used
        to transform the `shifts` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, shifts)`` is true.

        Args:
            shifts (Tensor): shifts, e.g., feature map coordinates
            boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(shifts, torch.Tensor), type(shifts)
        assert isinstance(boxes, torch.Tensor), type(boxes)

        deltas = torch.cat(
            (shifts - boxes[..., :2], boxes[..., 2:] - shifts), dim=-1
        ) * shifts.new_tensor(self.weights)
        return deltas

    def apply_deltas(self, deltas, shifts):
        """
        Apply transformation `deltas` (dl, dt, dr, db) to `shifts`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single shift shifts[i].
            shifts (Tensor): shifts to transform, of shape (N, 2)
        """
        assert torch.isfinite(deltas).all().item()
        shifts = shifts.to(deltas.dtype)

        if deltas.numel() == 0:
            return torch.empty_like(deltas)

        deltas = deltas.view(deltas.size()[:-1] + (-1, 4)) / shifts.new_tensor(
            self.weights
        )
        boxes = torch.cat(
            (
                shifts.unsqueeze(-2) - deltas[..., :2],
                shifts.unsqueeze(-2) + deltas[..., 2:],
            ),
            dim=-1,
        ).view(deltas.size()[:-2] + (-1,))
        return boxes
