from collections import Counter, deque, namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.layers import ShapeSpec

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.comm import get_rank
from torch import nn

from ultrasound_vid.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals, )
from ultrasound_vid.modeling.layers import ROIRelationLayers
from ultrasound_vid.modeling.heads.fast_rcnn import FastRCNNOutputLayers
from ultrasound_vid.modeling.backbone.resnet import (
    BottleneckBlock,
    BasicBlock,
    make_stage,
)

frame_cache = namedtuple("frame_cache", ["proposal", "feature"])


class TemporalROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(TemporalROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.interval_pre_test = cfg.MODEL.ROI_BOX_HEAD.INTERVAL_PRE_TEST
        self.interval_after_test = cfg.MODEL.ROI_BOX_HEAD.INTERVAL_AFTER_TEST
        self.causal_relation = cfg.MODEL.ROI_BOX_HEAD.CAUSAL_RELATION
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    @property
    def device(self):
        try:
            rank = get_rank()
            return torch.device(f"cuda:{rank}")
        except:
            return torch.device("cpu")

    def _sample_proposals(
        self,
        matched_idxs: torch.Tensor,
        matched_labels: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances],
            targets: List[Instances]) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes)

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name,
                     trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                            "gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name,
                                                trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)))
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def reorganize_proposals_by_video(
        self,
        batched_inputs: List[Dict],
        proposals: List[Instances],
        box_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for relation modules.
        Reorganize proposals by grouping those from same video, and these proposals will
        be regarded as "related inputs" in relation modules. The meaning of batchsize is
        number of videos in relation module.
        We use video_folder to check whether some frames are from the same video.

        Boxes and features of each image are paded to have the same size, so that it can
        be processed in a batch manner.
        """
        if batched_inputs is not None:
            video_folders = [b["video_folder"] for b in batched_inputs]
            counter = Counter(video_folders)
            assert (len(set(counter.values())) == 1
                    ), "videos with different numbers of frames"
            num_videos = len(counter)
            assert num_videos == 1, "Support only one video per GPU!"
        else:
            num_videos = 1

        # box type: x1, y1, x2, y2
        boxes = [p.proposal_boxes.tensor for p in proposals]
        num_proposals = [len(b) for b in boxes]
        box_features = list(box_features.split(num_proposals, dim=0))
        out_num_proposals = max(num_proposals)
        device = boxes[0].device
        valid_boxes = []
        valid_boxes_exceptgt = []
        causal_masks = []

        for i in range(len(boxes)):
            boxes[i] = F.pad(boxes[i],
                             [0, 0, 0, out_num_proposals - num_proposals[i]])
            box_features[i] = F.pad(
                box_features[i],
                [0, 0, 0, out_num_proposals - num_proposals[i]])
            valid_boxes.append(
                torch.arange(out_num_proposals, device=device) <
                num_proposals[i])
            basemask = torch.arange(out_num_proposals,
                                    device=device) < num_proposals[i]
            if self.training:
                gtmask = proposals[i].notgt_bool
                basemask[
                    0:num_proposals[i]] = basemask[0:num_proposals[i]] & gtmask
            valid_boxes_exceptgt.append(basemask)

            causal_mask = torch.arange(
                out_num_proposals * len(boxes),
                device=device) < (i + 1) * out_num_proposals
            causal_mask = causal_mask.unsqueeze(0).repeat(out_num_proposals, 1)
            causal_masks.append(causal_mask)

        boxes = torch.stack(boxes, dim=0)
        box_features = torch.stack(box_features, dim=0)
        valid_boxes = torch.stack(valid_boxes, dim=0)
        valid_boxes_exceptgt = torch.stack(valid_boxes_exceptgt, dim=0)
        causal_mask = torch.cat(causal_masks, dim=0)

        boxes = boxes.reshape(num_videos, -1, 4)
        box_features = box_features.reshape(num_videos, -1,
                                            box_features.shape[-1])
        valid_boxes = valid_boxes.reshape(num_videos, -1)
        valid_boxes_exceptgt = valid_boxes_exceptgt.reshape(num_videos, -1)
        causal_mask = causal_mask.reshape(num_videos, valid_boxes.shape[1],
                                          valid_boxes.shape[1])

        return boxes, box_features, valid_boxes, valid_boxes_exceptgt, causal_mask

    def reorganize_proposals_for_single_video(
            self, proposals: List[Instances],
            box_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for relation modules.
        This function is designed for test mode, which means all proposals are
        from the same video
        """

        # box type: x1, y1, x2, y2
        boxes = [p.proposal_boxes.tensor for p in proposals]
        num_proposals = [len(b) for b in boxes]
        device = boxes[0].device
        boxes = torch.cat(boxes, dim=0)
        boxes = boxes.reshape(1, -1, 4)
        box_features = box_features.reshape(1, -1, box_features.shape[-1])

        if self.causal_relation and sum(num_proposals) > 0:
            causal_masks = []
            for i in range(len(num_proposals)):
                if num_proposals[i] == 0: continue
                causal_mask = torch.arange(sum(num_proposals),
                                           device=device) < sum(
                                               num_proposals[:i + 1])
                causal_mask = causal_mask.unsqueeze(0).repeat(
                    num_proposals[i], 1)
                causal_masks.append(causal_mask)
            causal_mask = torch.cat(causal_masks, dim=0)
            causal_mask = causal_mask.reshape(1, boxes.shape[1],
                                              boxes.shape[1])
        else:
            causal_mask = None
        return boxes, box_features, causal_mask

    def reorganize_proposals_by_frame(
        self,
        proposals: List[Instances],
        box_features: torch.Tensor,
        multi_features: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        # box type: x1, y1, x2, y2
        boxes = [p.proposal_boxes.tensor for p in proposals]
        num_proposals = [len(b) for b in boxes]
        box_features = list(box_features.split(num_proposals, dim=0))
        out_num_proposals = max(num_proposals)
        device = boxes[0].device
        valid_boxes = []
        valid_boxes_exceptgt = []

        for i in range(len(boxes)):
            boxes[i] = F.pad(boxes[i],
                             [0, 0, 0, out_num_proposals - num_proposals[i]])
            box_features[i] = F.pad(
                box_features[i],
                [0, 0, 0, out_num_proposals - num_proposals[i]])
            valid_boxes.append(
                torch.arange(out_num_proposals, device=device) <
                num_proposals[i])
            basemask = torch.arange(out_num_proposals,
                                    device=device) < num_proposals[i]
            if self.training:
                gtmask = proposals[i].notgt_bool
                basemask[
                    0:num_proposals[i]] = basemask[0:num_proposals[i]] & gtmask
            valid_boxes_exceptgt.append(basemask)

        if multi_features is not None:
            # List of (num_frames, num_prop_per_frame, D)
            multi_features = list(multi_features.split(num_proposals, dim=-2))
            for i in range(len(boxes)):
                multi_features[i] = F.pad(
                    multi_features[i],
                    [0, 0, 0, out_num_proposals - num_proposals[i]])
            # (num_frames, B, out_num_prop, D)
            multi_features = torch.stack(multi_features, dim=1)

        boxes = torch.stack(boxes, dim=0)
        box_features = torch.stack(box_features, dim=0)
        valid_boxes = torch.stack(valid_boxes, dim=0)
        valid_boxes_exceptgt = torch.stack(valid_boxes_exceptgt, dim=0)

        return boxes, box_features, multi_features, valid_boxes, valid_boxes_exceptgt

    def forward(
        self,
        batched_inputs: List[Dict],
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            batched_inputs (List[Dict]):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
            detected instances. Returned during inference only; may be [] during training.

            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5TemporalROIBoxHeads(TemporalROIHeads):
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
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
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
        self.relation, out_channels = self._build_relation_module(
            cfg, out_channels)
        self.box_predictor = FastRCNNOutputLayers(
            out_channels,
            box2box_transform=self.box2box_transform,
            num_classes=self.num_classes,
            cls_agnostic_bbox_reg=self.cls_agnostic_bbox_reg,
            smooth_l1_beta=self.smooth_l1_beta,
            test_score_thresh=self.test_score_thresh,
            test_nms_thresh=self.test_nms_thresh,
            test_topk_per_image=self.test_detections_per_img,
        )
        buffer_length = self.interval_pre_test + self.interval_after_test + 1
        self.history_buffer = deque(maxlen=buffer_length)
        self.buffer_length = buffer_length
        self.d_model = out_channels

    def _build_caches(self):
        self.history_buffer.clear()

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2**3  # res5 is 8x res2
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        half_channel = cfg.MODEL.RESNETS.HALF_CHANNEL
        res5_out_channel = cfg.MODEL.RESNETS.RES5_OUT_CHANNEL
        if half_channel:  # deprecated, using res5_out_channel to set RDN channels
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
            return self.forward_train(batched_inputs, features, proposals,
                                      targets)
        else:
            return self.forward_test(batched_inputs, features, proposals,
                                     targets)
