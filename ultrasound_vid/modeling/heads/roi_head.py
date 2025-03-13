import torch
from torch import nn
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.modeling.poolers import ROIPooler
from ultrasound_vid.modeling.backbone.resnet import (
    BottleneckBlock,
    BasicBlock,
    make_stage,
)
from .temporal_head import TemporalROIHeads
from ultrasound_vid.modeling.heads.fast_rcnn import FastRCNNOutputLayers
from ultrasound_vid.modeling.layers import ROIRelationLayers


@ROI_HEADS_REGISTRY.register()
class SingleFrameRes5ROIHeads(Res5ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    @classmethod
    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor    = 2 ** 3  # res5 is 8x res2
        num_groups              = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group         = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels     = num_groups * width_per_group * stage_channel_factor
        out_channels            = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        half_channel            = cfg.MODEL.RESNETS.HALF_CHANNEL
        res5_out_channel        = cfg.MODEL.RESNETS.RES5_OUT_CHANNEL
        if half_channel: # deprecated, using res5_out_channel to set RDN channels
            res5_out_channel = 256
        stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on
        # assert False, f"{cfg.MODEL.RESNETS.DEPTH} {res5_out_channel} {out_channels}"
        if cfg.MODEL.RESNETS.DEPTH >= 50:
            # if use ResNet-50
            blocks = make_stage(
                BottleneckBlock,
                3,
                stride_per_block=[2, 1, 1],
                in_channels=out_channels // 2,
                bottleneck_channels=bottleneck_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                norm=norm,
                stride_in_1x1=stride_in_1x1,
            )
        else:
            # if use ResNet-18 and 34
            if res5_out_channel != 512:
                blocks = make_stage(
                    BasicBlock,
                    3,
                    stride_per_block=[2, 1, 1],
                    in_channels=out_channels // 2,
                    out_channels=res5_out_channel,
                    norm=norm,
                    short_cut_per_block=[True, False, False],
                )
                out_channels = res5_out_channel
            else:
                blocks = make_stage(
                    BasicBlock,
                    3,
                    stride_per_block=[2, 1, 1],
                    in_channels=out_channels // 2,
                    out_channels=out_channels,
                    norm=norm,
                )
        return nn.Sequential(*blocks), out_channels

    def reset(self):
        return


@ROI_HEADS_REGISTRY.register()
class Res5RelationROIBoxHeads(TemporalROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.

    Only box head. Mask head not supported.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)


        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution   = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type         = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales       = (1.0 / self.feature_strides[self.in_features[0]],)
        sampling_ratio      = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert not cfg.MODEL.MASK_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.relation, out_channels = self._build_relation_module(cfg, out_channels)
        self.box_predictor = FastRCNNOutputLayers(
            out_channels,
            box2box_transform=self.box2box_transform,
            num_classes=self.num_classes,
            cls_agnostic_bbox_reg=self.cls_agnostic_bbox_reg,
            smooth_l1_beta=self.smooth_l1_beta,
            test_score_thresh=self.test_score_thresh,
            test_nms_thresh=self.test_nms_thresh,
            test_topk_per_image=self.test_detections_per_img,
            organ_specific=cfg.MODEL.ORGAN_SPECIFIC.ENABLE,
        )
        self.d_model = out_channels

    def reset(self):
        return

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor    = 2 ** 3  # res5 is 8x res2
        num_groups              = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group         = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels     = num_groups * width_per_group * stage_channel_factor
        out_channels            = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        half_channel            = cfg.MODEL.RESNETS.HALF_CHANNEL
        res5_out_channel        = cfg.MODEL.RESNETS.RES5_OUT_CHANNEL
        if half_channel: # deprecated, using res5_out_channel to set RDN channels
            res5_out_channel = 256
        stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        if cfg.MODEL.RESNETS.DEPTH >= 50:
            # if use ResNet-50
            blocks = make_stage(
                BottleneckBlock,
                3,
                stride_per_block=[2, 1, 1],
                in_channels=out_channels // 2,
                bottleneck_channels=bottleneck_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                norm=norm,
                stride_in_1x1=stride_in_1x1,
            )
        else:
            # if use ResNet-18 and 34
            if res5_out_channel != 512:
                blocks = make_stage(
                    BasicBlock,
                    3,
                    stride_per_block=[2, 1, 1],
                    in_channels=out_channels // 2,
                    out_channels=res5_out_channel,
                    norm=norm,
                    short_cut_per_block=[True, False, False],
                )
                out_channels = res5_out_channel
            else:
                blocks = make_stage(
                    BasicBlock,
                    3,
                    stride_per_block=[2, 1, 1],
                    in_channels=out_channels // 2,
                    out_channels=out_channels,
                    norm=norm,
                )
        return nn.Sequential(*blocks), out_channels

    def _build_relation_module(self, cfg, in_channels):
        return ROIRelationLayers(cfg, in_channels), in_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, batched_inputs, features, proposals, targets=None):
        if self.training:
            return self.forward_train(batched_inputs, features, proposals, targets)
        else:
            return self.forward_test(batched_inputs, features, proposals, targets)

    def forward_train(self, batched_inputs, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        The input images is replaced by batched_inputs, to get video informations.
        """
        proposals = self.label_and_sample_proposals(proposals, targets)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        num_boxes = sum([len(p) for p in proposal_boxes])
        if num_boxes == 0:
            losses = {
                "loss_cls": torch.tensor(0.0, device=proposal_boxes[0].tensor.device),
                "loss_box_reg": torch.tensor(0.0, device=proposal_boxes[0].tensor.device)
            }
            return [], losses

        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        (
            boxes,
            box_features,
            _,
            is_valid,
            valid_boxes_exceptgt,
        ) = self.reorganize_proposals_by_frame(
            proposals, feature_pooled, None
        )
        relation_mask = valid_boxes_exceptgt.unsqueeze(1).repeat(
            1, is_valid.shape[1], 1
        )
        box_features = self.relation(boxes, box_features, mask=relation_mask)
        box_features = box_features.reshape(-1, self.d_model)[is_valid.flatten()]
        predictions  = self.box_predictor(box_features)

        del feature_pooled, box_features
        losses = self.box_predictor.losses(predictions, proposals)
        del features
        return [], losses

    def forward_test(self, batched_inputs, features, proposals, targets=None):

        assert len(batched_inputs) == 1
        assert len(proposals) == 1
        assert targets is None

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        (
            boxes,
            box_features,
            _,
            is_valid,
            valid_boxes_exceptgt,
        ) = self.reorganize_proposals_by_frame(
            proposals, feature_pooled, None
        )
        relation_mask = valid_boxes_exceptgt.unsqueeze(1).repeat(
            1, is_valid.shape[1], 1
        )
        box_features = self.relation(boxes, box_features, mask=relation_mask)
        box_features = box_features.reshape(-1, self.d_model)[is_valid.flatten()]

        predictions = self.box_predictor(box_features)
        pred_instances, _ = self.box_predictor.inference(predictions, proposals)

        return pred_instances, {}