from collections import Counter, deque, namedtuple
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.logger import log_first_n

from ultrasound_vid.modeling.layers import FlowNetS, FlowContextLayers
from ultrasound_vid.modeling.heads import Res5TemporalROIBoxHeads


frame_cache = namedtuple("frame_cache", ["proposal", "image", "feature", "multi_feature"])


@ROI_HEADS_REGISTRY.register()
class Res5FlowContextHeads(Res5TemporalROIBoxHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.context_feature        = cfg.MODEL.CONTEXT_FEATURE
        self.roi_size               = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.roi_rel_position       = cfg.MODEL.ROI_BOX_HEAD.ROI_REL_POSITION
        self.no_aux_loss            = cfg.MODEL.ROI_BOX_HEAD.NO_AUX_LOSS
        # spatial position encoding for RoI feature
        self.build_relative_roi_pos_embed()
        self.num_context_frames = cfg.MODEL.CONTEXT_FLOW_FRAMES
        self.use_iof_align = cfg.MODEL.CONTEXT_IOF_ALIGN
        self.step = cfg.MODEL.CONTEXT_STEP_LEN
        self.flownet_weights = cfg.MODEL.FLOWNET_WEIGHTS
        self.relation = FlowContextLayers(cfg, self.d_model)
        if self.use_iof_align:
            self.flownet = FlowNetS(cfg, flownet_method="DFF")
            self.init_flownet()
        self.context_image_buffer = deque(maxlen=self.num_context_frames)
        self.context_feature_buffer = deque(maxlen=self.num_context_frames)
        self.frame_idx = 0
    
    def init_flownet(self):
        ckpt = torch.load(self.flownet_weights, map_location=self.device)
        self.flownet.load_state_dict(ckpt["state_dict"], strict=False)
        for n, p in self.flownet.named_parameters():
            if "Convolution5_scale" not in n:
                p.requires_grad_ = False

    def get_grid(self, flow):
        m, n = flow.shape[-2:]
        shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)
        shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)
        grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
        workspace = torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)
        flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)
        return flow_grid

    def resample(self, feats, flow):
        flow_grid = self.get_grid(flow)
        warped_feats = F.grid_sample(feats, flow_grid, mode="bilinear", padding_mode="border")
        return warped_feats

    def warp_context_features(self, curr_images, context_images, context_features):
        """
        Params:
            curr_images: (B, 3, H, W)
            context_images: (T, 3, H, W)
        """
        num_context_frames = len(context_images)
        num_local_frames = len(curr_images)
        img_cur_copies = curr_images[None].repeat(num_context_frames, 1, 1, 1, 1).flatten(0,1)
        img_ctxt_copies = context_images[:, None].repeat(1, num_local_frames, 1, 1, 1).flatten(0,1)
        concat_imgs_pair = torch.cat([img_cur_copies, img_ctxt_copies], dim=1)
        # flow: (T * B, 2, H, W)
        flow, scale_map = self.flownet(concat_imgs_pair)
        warped_context_features = []
        for context_feature in context_features:
            H, W = context_feature.shape[-2:]
            flow_i = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=True)
            scale_map_i = F.interpolate(scale_map, size=(H, W), mode="bilinear", align_corners=True)
            context_feature = context_feature[:, None].repeat(1, num_local_frames, 1, 1, 1).flatten(0,1)
            warped_context_feature = self.resample(context_feature, flow_i)
            feature = warped_context_feature * scale_map_i
            warped_context_features.append(feature)
        return warped_context_features

    def extract_multi_features(
        self, proposal_boxes, local_images, context_images, context_features, roi_position
    ):
        assert len(proposal_boxes) == len(local_images)
        # extract frame proposal freatures
        num_context_frames = len(context_images)
        if self.use_iof_align:
            warped_context_features = self.warp_context_features(
                local_images, context_images, context_features
            ) # [(T * B, D, H, W)]
        else:
            warped_context_features = context_features
        # (T * B, num_prop, 7, 7, d_model)
        multi_features = self._shared_roi_transform(
            warped_context_features, 
            [x for _ in range(num_context_frames) for x in proposal_boxes]
        )
        multi_features = self.with_pos_embed(multi_features, roi_position)
        multi_features = multi_features.mean(dim=[2, 3])
        multi_features = multi_features.reshape(num_context_frames, -1, self.d_model)
        return multi_features

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def extract_roi_pos_embed(self):
        if not self.roi_rel_position:
            return None
        roi_position = self.cpb_mlp(self.relative_pos)
        roi_position = roi_position.reshape(self.roi_size, self.roi_size, self.d_model)
        roi_position = roi_position.permute(2, 0, 1)[None]
        # (1, d_model, 7, 7)
        return roi_position
    
    def build_relative_roi_pos_embed(self):
        if not self.roi_rel_position:
            return
        self.cpb_mlp = nn.Sequential(nn.Linear(2, self.d_model, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.d_model, self.d_model, bias=False))
        relative_h = torch.arange(0, self.roi_size, dtype=torch.float32) - (self.roi_size - 1) * 0.5
        relative_w = torch.arange(0, self.roi_size, dtype=torch.float32) - (self.roi_size - 1) * 0.5
        relative_pos = torch.stack(
            torch.meshgrid([relative_h,
                            relative_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 16, 16, 2
        relative_pos[:, :, :, 0] /= (self.roi_size - 1)
        relative_pos[:, :, :, 1] /= (self.roi_size - 1)
        relative_pos *= 8  # normalize to -8, 8
        relative_pos = torch.sign(relative_pos) * torch.log2(
            torch.abs(relative_pos) + 1.0) / np.log2(8)
        self.register_buffer("relative_pos", relative_pos)

    def forward_train(self, batched_inputs, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        The input images is replaced by batched_inputs, to get video informations.
        """
        # process corner case: no proposals
        num_boxes = sum([len(p) for p in proposals])
        device = proposals[0].proposal_boxes.tensor.device
        if num_boxes == 0:
            losses = {
                "loss_cls": torch.tensor(0.0, device=device),
                "loss_box_reg": torch.tensor(0.0, device=device)
            }
            return [], losses

        images = batched_inputs
        num_frames = len(images); assert num_frames > self.num_context_frames
        context_indices = np.random.choice(
            num_frames, self.num_context_frames, replace=False
        )
        
        proposals = self.label_and_sample_proposals(proposals, targets)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        frame_features = [features[f] for f in self.context_feature]
        del features
        # extract box features
        roi_position = self.extract_roi_pos_embed()
        box_features = self._shared_roi_transform(frame_features, proposal_boxes)
        box_features = self.with_pos_embed(box_features, roi_position)
        feature_pooled = box_features.mean(dim=[2, 3])
        multi_features = self.extract_multi_features(
            proposal_boxes, images.tensor, images.tensor[context_indices], 
            [f[context_indices] for f in frame_features], roi_position
        )
        # padding boxes and features
        (
            boxes,                      # (num_frame, num_prop, 4)
            box_features,               # (num_frame, num_prop, d_model)
            multi_features,             # (num_frame, num_frame, num_prop, d_model)
            is_valid,                   # (num_frame, num_prop)
            valid_boxes_exceptgt,       # (num_frame, num_prop)
        ) = self.reorganize_proposals_by_frame(
            proposals, feature_pooled, multi_features
        )
        # relation module
        box_features = self.relation(
            boxes, box_features, multi_features, valid_boxes_exceptgt
        )
        if self.no_aux_loss:
            box_features = box_features[[-1]]
            is_valid = is_valid[[-1]]
            proposals = [proposals[-1]]
        box_features = box_features.reshape(-1, self.d_model)
        box_features = box_features[is_valid.flatten()]
        predictions  = self.box_predictor(box_features)
        del feature_pooled, box_features

        # final layer loss
        losses = self.box_predictor.losses(predictions, proposals)
        for k, v in losses.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                losses[k] = torch.tensor(0.0, device=device)
        return [], losses

    def forward_test(self, batched_inputs, features, proposals, targets=None):
        assert len(proposals) == 1
        assert targets is None
        # prepare context frames
        images = batched_inputs
        roi_position = self.extract_roi_pos_embed()
        # update context buffer
        if self.frame_idx % self.step == 0:
            context_features = [features[f] for f in self.context_feature]
            self.context_image_buffer.append(images.tensor)
            self.context_feature_buffer.append(context_features)
            # update multi_features in history buffer
            if len(self.history_buffer) > 0:
                history_boxes = [x.proposal.proposal_boxes for x in self.history_buffer]
                history_proposals = [x.proposal for x in self.history_buffer]
                history_images = torch.cat([x.image for x in self.history_buffer], dim=0)
                history_features = [x.feature for x in self.history_buffer]
                history_multi_features = [x.multi_feature for x in self.history_buffer]
                new_multi_features = self.extract_multi_features(
                    history_boxes, history_images, images.tensor,
                    context_features, roi_position
                ) # (T, all_prev_prop, D)
                all_context_frames = history_multi_features[0].shape[0] + 1
                new_multi_features = new_multi_features.split([len(x) for x in history_proposals], dim=1)
                history_multi_features = [
                    torch.cat([old, new], dim=0)[max(0, all_context_frames - self.num_context_frames):]
                    for old, new in zip(history_multi_features, new_multi_features)
                ]
                self.history_buffer.clear()
                for prop, img, feat, multi_feat in zip(
                    history_proposals, history_images, history_features, history_multi_features
                ):
                    self.history_buffer.append(frame_cache(prop, img[None], feat, multi_feat))
        # context frames for current frame
        if self.frame_idx == 0:
            context_images = images.tensor
            context_features = [features[f] for f in self.context_feature]
        else:
            context_images = torch.cat([x for x in self.context_image_buffer], dim=0)
            context_features = [
                torch.cat([x[i] for x in self.context_feature_buffer], dim=0)
                for i in range(len(features))
            ]
        self.frame_idx += 1
        # prepare current frame
        proposal_boxes = [x.proposal_boxes for x in proposals]
        frame_features = [features[f] for f in self.context_feature]
        del features
        # extract box features
        box_features = self._shared_roi_transform(frame_features, proposal_boxes)
        box_features = self.with_pos_embed(box_features, roi_position)
        feature_pooled = box_features.mean(dim=[2, 3])
        multi_features = self.extract_multi_features(
            proposal_boxes, images.tensor, context_images,
            context_features, roi_position
        )
        # history buffer ((N, D), (T, N, D))
        self.history_buffer.append(frame_cache(proposals[-1], images.tensor, feature_pooled, multi_features))
        # prepare local frames
        proposals = [x.proposal for x in self.history_buffer]
        feature_pooled = torch.cat([x.feature for x in self.history_buffer], dim=0)
        multi_features = torch.cat([x.multi_feature for x in self.history_buffer], dim=1)
        # padding boxes and features
        (
            boxes,                      # (num_frame, num_prop, 4)
            box_features,               # (num_frame, num_prop, d_model)
            multi_features,             # (ctxt_frame, num_frame, num_prop, d_model)
            is_valid,                   # (num_frame, num_prop)
            valid_boxes_exceptgt,       # (num_frame, num_prop)
        ) = self.reorganize_proposals_by_frame(
            proposals, feature_pooled, multi_features
        )
        # relation module
        box_features = self.relation(
            boxes, box_features, multi_features, valid_boxes_exceptgt
        )
        box_features = box_features[-1, :len(proposals[-1])]
        proposals = [proposals[-1]]
        # final prediction
        predictions = self.box_predictor(box_features)
        pred_instances, _ = self.box_predictor.inference(predictions, proposals)
        return pred_instances, {}

    def reset(self):
        super().reset()
        self.context_image_buffer.clear()
        self.context_feature_buffer.clear()
        self.frame_idx = 0