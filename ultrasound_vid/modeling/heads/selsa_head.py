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

from ultrasound_vid.modeling.layers import SELSALayers
from ultrasound_vid.modeling.heads import Res5TemporalROIBoxHeads


@ROI_HEADS_REGISTRY.register()
class Res5SELSAHeads(Res5TemporalROIBoxHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation = SELSALayers(cfg, self.d_model)
        assert len(self.in_features) == 1, \
            f"SELSA currently support only 1-level feature, but receive {len(self.in_features)}!"