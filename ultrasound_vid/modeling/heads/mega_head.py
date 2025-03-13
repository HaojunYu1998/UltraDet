from collections import Counter, deque, namedtuple
import torch

from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes

from ultrasound_vid.modeling.heads.temporal_head import Res5TemporalROIBoxHeads

frame_cache = namedtuple("frame_cache", ["proposal", "feature"])


@ROI_HEADS_REGISTRY.register()
class MEGAHeads(Res5TemporalROIBoxHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.

    Only box head. Mask head not supported.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.local_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES
        self.memory_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.MEMORY_FRAMES
        self.memory_buffer = deque(maxlen=self.memory_frames + self.local_frames)
        self.history_buffer = deque(maxlen=self.local_frames)

    def reset(self):
        self.history_buffer.clear()
        self.memory_buffer.clear()

    def forward(self, features, proposals, targets=None, 
                memory_proposals=None, memory_features=None):
        if self.training:
            return self.forward_train(features, proposals, targets, 
                                      memory_proposals, memory_features)
        else:
            return self.forward_test(features, proposals)

    def proposal_roi_feature(self, proposals, features):
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in self.in_features]
        box_features = self._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        return proposal_boxes, feature_pooled

    def forward_train(self, features, proposals, targets, memory_proposals, memory_features):
        """
        See :class:`ROIHeads.forward`.
        The input images is replaced by batched_inputs, to get video informations.
        """
        num_boxes = sum([len(p) for p in proposals])
        device = proposals[0].proposal_boxes.tensor.device
        if num_boxes == 0:
            losses = {
                "loss_cls": torch.tensor(0.0, device=device),
                "loss_box_reg": torch.tensor(0.0, device=device)
            }
            return [], losses
        # extract_proposal_roi_feature
        proposals = self.label_and_sample_proposals(proposals, targets)
        _, feature_pooled = self.proposal_roi_feature(
            proposals, features
        )
        with torch.no_grad():
            memory_boxes, memory_features = self.proposal_roi_feature(
                memory_proposals, memory_features
            )
        del features
        # reorganize_proposals_by_frame
        (
            boxes, box_features, 
            is_valid, valid_boxes_exceptgt, _
        ) = self.reorganize_proposals_by_video(None, proposals, feature_pooled)
        relation_mask = valid_boxes_exceptgt.unsqueeze(1).repeat(
            1, is_valid.shape[1], 1
        ) # (1, N_loc, N_loc)
        # relation module for memory frames
        with torch.no_grad():
            memory_boxes = Boxes.cat(memory_boxes).tensor.reshape(1, -1, 4)
            memory_features = memory_features.reshape(1, -1, self.d_model)
            memory_mask = torch.ones(1, is_valid.shape[1], memory_features.shape[1])
            relation_mask = torch.cat(
                [relation_mask, memory_mask.to(relation_mask)], dim=-1
            ) # (1, N_loc, N_loc + N_mem)
            memory_features = self.relation(
                memory_boxes, memory_features, return_intermediate=True
            )
            memory_features = torch.cat(memory_features, dim=0)[:-1]
        # relation module
        box_features = self.relation.forward_with_reference(
            boxes, box_features, memory_boxes, memory_features, mask=relation_mask
        )[-1]
        box_features = box_features.reshape(-1, self.d_model)
        box_features = box_features[is_valid.flatten()]
        # calculate losses
        predictions = self.box_predictor(box_features)
        del feature_pooled, box_features
        losses = self.box_predictor.losses(predictions, proposals)
        for k, v in losses.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                losses[k] = torch.tensor(0.0, device=device)
        return [], losses

    def forward_test(self, features, proposals):
        assert len(proposals) == 1
        # process current frame
        proposal_boxes, feature_pooled = self.proposal_roi_feature(proposals, features)
        self.history_buffer.append(frame_cache(proposals[0], feature_pooled))
        # prepare local proposals
        proposals = [x.proposal for x in self.history_buffer]
        box_features = [x.feature for x in self.history_buffer]
        box_features = torch.cat(box_features, dim=0)
        boxes, box_features, _ = self.reorganize_proposals_for_single_video(
            proposals, box_features
        )
        # use memory proposals
        if len(self.memory_buffer) > self.local_frames:
            memory_boxes = [x.proposal.proposal_boxes for x in self.memory_buffer][-self.local_frames:]
            memory_features = [x.feature for x in self.memory_buffer][-self.local_frames:]
            memory_boxes = Boxes.cat(memory_boxes).tensor.reshape(1, -1, 4) # (1, memory_frame, 4)
            memory_features = torch.cat(memory_features, dim=1) # (num_layer, memory_frame, d_model)
            # relation module with memory
            box_features = self.relation.forward_with_reference(
                boxes, box_features, memory_boxes, memory_features
            )
        else:
            # relation module without memory
            box_features = self.relation(
                boxes, box_features, return_intermediate=True
            )
        # update memory with current frame
        box_features = torch.cat(box_features, dim=0) # (num_layer+1, local_frame, d_model)
        box_features = box_features.split([len(p) for p in proposals], dim=1)[-1]
        box_features, memory_features = box_features[-1], box_features[:-1]
        self.memory_buffer.append(frame_cache(proposals[-1], memory_features))
        # prediction
        box_features = box_features.reshape(-1, self.d_model)
        proposals = [proposals[-1]]
        predictions = self.box_predictor(box_features)
        pred_instances, _ = self.box_predictor.inference(predictions, proposals)
        return pred_instances, {}