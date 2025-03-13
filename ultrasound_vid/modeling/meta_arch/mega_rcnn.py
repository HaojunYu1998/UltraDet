import logging
from collections import deque
from itertools import chain
from contextlib import contextmanager

import torch
from detectron2.modeling import build_roi_heads, build_backbone, META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.logger import log_first_n
from detectron2.structures import ImageList
from torch import nn
from ultrasound_vid.utils import imagelist_from_tensors
from .temporal_rcnn import TemporalRCNN


@META_ARCH_REGISTRY.register()
class MEGA(TemporalRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.memory_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.MEMORY_FRAMES

    def forward_train(self, batched_inputs):
        batchsize = len(batched_inputs)
        assert batchsize == 1, f"Support only one video per GPU, but receive {batchsize}"
        batched_inputs = list(chain.from_iterable(batched_inputs))
        assert len(batched_inputs) == self.memory_frames + self.num_frames

        # organ switching
        dataset_name = batched_inputs[0]["dataset"]
        organ = dataset_name.split("@")[0].split("_")[0]
        assert organ == "breast" or organ == "thyroid"
        if "cls" in self.organ_specific:
            self.roi_heads.box_predictor.switch(organ)
        if "rpn_cls" in self.organ_specific:
            self.proposal_generator.head.switch(organ)

        images, *_ = self.preprocess_image(batched_inputs)
        local_images = ImageList(
            images.tensor[self.memory_frames:], images.image_sizes[self.memory_frames:]
        )
        memory_images = ImageList(
            images.tensor[:self.memory_frames].detach(), images.image_sizes[:self.memory_frames]
        )
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_instances = gt_instances[self.memory_frames:]
        else:
            raise AttributeError("Failed to get 'instances' from training image.")

        features = self.backbone(images.tensor)

        local_features, memory_features = {}, {}
        for k, v in features.items():
            local_features[k] = v[self.memory_frames:]
            memory_features[k] = v[:self.memory_frames].detach()
        features = local_features

        proposals, proposal_losses = self.proposal_generator(
            local_images, features, gt_instances, is_train=True
        )
        memory_proposals, _ = self.proposal_generator(
            memory_images, memory_features, [], is_train=False
        )

        _, detector_losses = self.roi_heads(
            features, proposals, gt_instances, memory_proposals, memory_features
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return {k: v * self.organ_loss_factor[organ] for k, v in losses.items()}

    def forward_infer(self, frame_input):
        # for test mode
        assert not self.training
        # organ switching
        dataset_name = frame_input["dataset"]
        organ = dataset_name.split("@")[0].split("_")[0]
        assert organ == "breast" or organ == "thyroid"
        if "cls" in self.organ_specific:
            self.roi_heads.box_predictor.switch(organ)
        if "rpn_cls" in self.organ_specific:
            self.proposal_generator.head.switch(organ)
        # extract features
        images, *_ = self.preprocess_image([frame_input])
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None, is_train=False)
        results, _ = self.roi_heads(features, proposals)
        # process results
        r = results[0]
        return self.postprocess(r, frame_input, images.image_sizes[0])

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        if hasattr(self.backbone, "sample_module"):
            self.backbone.sample_module.reset()
        reset_op = getattr(self.roi_heads, "reset", None)
        if callable(reset_op):
            reset_op()
        else:
            log_first_n(logging.WARN, 'Roi Heads doesnt have function "reset"')