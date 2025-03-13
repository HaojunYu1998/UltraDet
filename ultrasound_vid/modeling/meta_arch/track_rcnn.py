import logging
from collections import deque
from itertools import chain

import torch
from detectron2.modeling import build_roi_heads, build_backbone, META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.logger import log_first_n
from torch import nn
from ultrasound_vid.utils import imagelist_from_tensors
from ultrasound_vid.modeling.layers import CacheTracker
from ultrasound_vid.modeling.meta_arch import TemporalRCNN, TemporalProposalNetwork


@META_ARCH_REGISTRY.register()
class TrackTemporalRCNN(TemporalRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tracker = CacheTracker(cfg)

    def forward_infer(self, frame_input):
        results = super().forward_infer(frame_input)
        results = self.tracker.update(results)
        return results

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        self.tracker.reset()
        super().reset()


@META_ARCH_REGISTRY.register()
class TrackTemporalProposalNetwork(TemporalProposalNetwork):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tracker = CacheTracker(cfg)

    def forward_infer(self, frame_input):
        results = super().forward_infer(frame_input)
        results = self.tracker.update(results)
        return results

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        self.tracker.reset()
        super().reset()