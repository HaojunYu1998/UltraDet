import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.structures import Boxes, pairwise_iou, Instances
from detectron2.layers import ShapeSpec, cat, batched_nms
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from ultrasound_vid.utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


@PROPOSAL_GENERATOR_REGISTRY.register()
class YOLOXHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.n_anchors = 1
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.in_features = cfg.MODEL.YOLOX.IN_FEATURES
        self.score_threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.nms_threshold = cfg.MODEL.YOLOX.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.YOLOX.NMS_TYPE
        self.pre_nms_topk = cfg.MODEL.YOLOX.PRE_NMS_TOPK
        self.post_nms_topk = cfg.MODEL.YOLOX.NUM_PROPOSALS
        feature_shapes = [input_shape[f] for f in self.in_features]
        self.in_channels = feature_shapes[0].channels
        self.strides = cfg.MODEL.YOLOX.STRIDES

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        self.stems.append(
            BaseConv(in_channels=self.in_channels, out_channels=self.in_channels,
                     ksize=1, stride=1, act="silu")
        )
        self.cls_convs.append(
            nn.Sequential(*[
                BaseConv(
                    in_channels=self.in_channels, 
                    out_channels=self.in_channels,
                    ksize=3, stride=1, act="silu"
                ),
                BaseConv(
                    in_channels=self.in_channels, 
                    out_channels=self.in_channels,
                    ksize=3, stride=1, act="silu"
                ),
        ]))
        self.reg_convs.append(
            nn.Sequential(*[
                BaseConv(
                    in_channels=self.in_channels, 
                    out_channels=self.in_channels, 
                    ksize=3, stride=1, act="silu"
                ),
                BaseConv(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    ksize=3, stride=1, act="silu"
                ),
        ]))
        self.cls_preds.append(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.n_anchors * self.num_classes,
                kernel_size=1, stride=1, padding=0,
            )
        )
        self.reg_preds.append(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=4,
                kernel_size=1, stride=1, padding=0,
            )
        )
        self.obj_preds.append(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.n_anchors * 1,
                kernel_size=1, stride=1, padding=0,
            )
        )
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)]
        self.expanded_strides = [None]
        self.initialize_biases()

    def initialize_biases(self):
        prior_prob = 0.01
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        prior_prob = 0.00001
        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, images, features, gt_instances=None):
        """
        Params:
            images: 
            features: List[Tensor] NCHW
            gt_instances: List[Instances]
        """
        features = [features[f] for f in self.in_features]
        self.dtype = features[0].type()
        self.device = features[0].device
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, feature) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, features)
        ):
            feature = self.stems[k](feature)
            cls_x = feature
            reg_x = feature

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(feature)
                )
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output = reg_output.view(
                    batch_size, self.n_anchors, 4, hsize, wsize
                )
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            proposal_losses = self.get_losses(
                images,
                x_shifts,
                y_shifts,
                expanded_strides,
                gt_instances,
                torch.cat(outputs, 1),
                origin_preds,
            )
            return [], proposal_losses
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 6]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            proposals = self.decode_outputs(outputs)
            return proposals, {}

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(self.dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(self.dtype)
        strides = torch.cat(strides, dim=1).type(self.dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        # [cx, cy, w, h, obj_score, cls_score]
        num_frame = outputs.shape[0]
        h, w = self.hw[0]
        H, W = h * self.strides[0], w * self.strides[0]
        proposals = []
        for i in range(num_frame):
            pred_boxes = box_cxcywh_to_xyxy(outputs[i, :, :4])
            scores = outputs[i, :, 4] * outputs[i, :,5]
            pred_classes = torch.zeros_like(scores)
            # Keep top k top scoring indices only.
            num_topk = min(self.pre_nms_topk, pred_boxes.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            scores, topk_idxs = scores.sort(descending=True)
            scores = scores[:num_topk]
            topk_idxs = topk_idxs[:num_topk]
            boxes_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes
            pred_boxes = pred_boxes[boxes_idxs]
            pred_classes = pred_classes[classes_idxs]

            if self.nms_type is None:
                keep = scores.argsort(descending=True)
            else:
                keep = batched_nms(
                    pred_boxes, scores, pred_classes, self.nms_threshold
                )
            keep = keep[:self.post_nms_topk]
            pred_boxes = pred_boxes[keep]
            scores = scores[keep]
            pred_classes = pred_classes[keep]
            proposal = Instances((H, W))
            proposal.set("pred_boxes", Boxes(pred_boxes))
            proposal.set("pred_classes", pred_classes)
            proposal.set("scores", scores)
            proposal.set("proposal_boxes", proposal.pred_boxes)
            proposal.set("objectness_logits", proposal.scores)
            proposals.append(proposal)
        return proposals

    def get_losses(
        self,
        images,
        x_shifts,
        y_shifts,
        expanded_strides,
        gt_instances,
        outputs,
        origin_preds,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            gt_per_image = gt_instances[batch_idx]
            num_gt = len(gt_per_image)
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = gt_per_image.gt_boxes.tensor
                gt_bboxes_per_image = box_xyxy_to_cxcywh(gt_bboxes_per_image)
                gt_classes = gt_per_image.gt_classes
                bboxes_preds_per_image = bbox_preds[batch_idx]
                
                try:
                    (
                        gt_matched_classes, fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds, num_fg_img,
                    ) = self.get_assignments(
                        batch_idx, num_gt, total_num_anchors,
                        gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides, x_shifts, y_shifts,
                        cls_preds, obj_preds, images,
                    )
                except RuntimeError:
                    print("OOM RuntimeError is raised due to the huge \
                           memory cost during label assignment.")
                    torch.cuda.empty_cache()
                
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                l1_target = self.get_l1_target(
                    outputs.new_zeros((num_fg_img, 4)),
                    gt_bboxes_per_image[matched_gt_inds],
                    expanded_strides[0][fg_mask],
                    x_shifts=x_shifts[0][fg_mask],
                    y_shifts=y_shifts[0][fg_mask],
                )
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.float())
            l1_targets.append(l1_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        l1_targets = torch.cat(l1_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        ############ Proposal Losses ############
        num_fg = max(num_fg, 1)
        # loss_cls
        loss_cls = (self.bcewithlog_loss(
            cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
        )).sum() / num_fg
        # loss_obj
        loss_obj = (self.bcewithlog_loss(
            obj_preds.view(-1, 1), obj_targets
        )).sum() / num_fg
        # loss_iou
        loss_iou = (self.iou_loss(
            bbox_preds.view(-1, 4)[fg_masks], reg_targets
        )).sum() / num_fg
        # loss_l1
        loss_l1 = (self.l1_loss(
            origin_preds.view(-1, 4)[fg_masks], l1_targets
        )).sum() / num_fg

        proposal_losses = {
            "loss_rpn_cls": loss_cls,
            "loss_rpn_obj": loss_obj,
            "loss_rpn_reg": 5.0 * loss_iou,
            "loss_rpn_l1": loss_l1,
        }
        return proposal_losses

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self, batch_idx, num_gt, total_num_anchors,
        gt_bboxes_per_image, gt_classes,
        bboxes_preds_per_image,
        expanded_strides, x_shifts, y_shifts,
        cls_preds, obj_preds, images,
    ):
        """
        Params:
            batch_idx: Int
            num_gt: Int
            total_num_anchors: Int
            gt_classes: Tensor (num_gt,)
            gt_bboxes_per_image: Tensor (num_gt, 4) cxcywh
            bboxes_preds_per_image: Tensor (num_prop, 4) cxcywh
        """
        img_size = images[0].shape[-2:]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides, x_shifts, y_shifts,
            total_num_anchors, num_gt, img_size
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        img_size
    ):
        """
        Params:
            gt_bboxes_per_image: Tensor (num_gt, 4) cxcywh
        """
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], dim=2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5
        # clip center inside image
        gt_bboxes_per_image_clip = gt_bboxes_per_image[:, 0:2].clone()
        gt_bboxes_per_image_clip[:, 0] = torch.clamp(gt_bboxes_per_image_clip[:, 0], min=0, max=img_size[1])
        gt_bboxes_per_image_clip[:, 1] = torch.clamp(gt_bboxes_per_image_clip[:, 1], min=0, max=img_size[0])

        gt_bboxes_per_image_l = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        del gt_bboxes_per_image_clip
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
