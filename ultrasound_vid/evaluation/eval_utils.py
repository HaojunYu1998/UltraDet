import torch
from detectron2.structures import Boxes
from torch import Tensor


def check_center_cover(pred_boxes: Boxes, gt_boxes: Boxes) -> Tensor:
    """
    For each bbox in pred_boxes, check if the center of bbox is contained in
    any gt_box.

    The box order must be (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    pred_boxes : Boxes(Nx4)
    gt_boxes : Boxes(Mx4)

    Returns
    -------
    Match matrix of shape (NxM)
    """
    pred_boxes = pred_boxes.tensor
    gt_boxes = gt_boxes.tensor
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_boxes.split(1, dim=1)  # Mx1
    x_ctr = pred_boxes[:, 0::2].mean(dim=1, keepdim=True)
    y_ctr = pred_boxes[:, 1::2].mean(dim=1, keepdim=True)

    x_in_gt = torch.logical_and(  # NxMx1
        gt_xmin <= x_ctr[:, None], x_ctr[:, None] <= gt_xmax
    )
    y_in_gt = torch.logical_and(  # NxMx1
        gt_ymin <= y_ctr[:, None], y_ctr[:, None] <= gt_ymax
    )
    pred_in_gt = torch.logical_and(x_in_gt, y_in_gt).squeeze(-1)
    return pred_in_gt  # NxM
