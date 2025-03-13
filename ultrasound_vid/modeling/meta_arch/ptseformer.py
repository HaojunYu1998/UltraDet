import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import math
from itertools import chain

from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess
from ultrasound_vid.utils.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from ultrasound_vid.utils.box_ops import box_cxcywh_to_xyxy
from ultrasound_vid.modeling.layers import MLP
from ultrasound_vid.modeling.layers.matcher import build_matcher
from ultrasound_vid.modeling.meta_arch.deformable_detr import DeformableDETR, SetCriterion


@META_ARCH_REGISTRY.register()
class PTSEFormer(DeformableDETR):
    def __init__(self, cfg):
        super().__init__(cfg)