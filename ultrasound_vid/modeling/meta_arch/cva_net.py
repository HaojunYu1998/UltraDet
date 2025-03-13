import math
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from collections import deque

from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess
from ultrasound_vid.utils.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from ultrasound_vid.utils.box_ops import box_cxcywh_to_xyxy
from ultrasound_vid.modeling.layers import MLP
from ultrasound_vid.modeling.layers.matcher import build_matcher
from ultrasound_vid.modeling.meta_arch.deformable_detr import DeformableDETR, SetCriterion


@META_ARCH_REGISTRY.register()
class CVANet(DeformableDETR):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.global_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.MEMORY_FRAMES
        self.local_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES
        assert self.global_frames == self.local_frames
        in_channels = self.backbone.output_shape()[self.in_features[0]].channels
        self.avg_pool = nn.AdaptiveAvgPool3d((in_channels, None, None))
        self.history_buffer = deque(maxlen=self.global_frames + self.local_frames)

    def forward(self, batched_inputs):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if self.training:
            batched_inputs = list(chain.from_iterable(batched_inputs))
            assert len(batched_inputs) == (self.global_frames + self.local_frames), \
                f"{len(batched_inputs)} != {self.global_frames} + {self.local_frames}"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_instances = gt_instances[-self.local_frames:]
            targets = self.prepare_targets(gt_instances)
        else:
            batched_inputs = [batched_inputs]

        images, images_whwh = self.preprocess_image(batched_inputs)
        b = len(images.tensor)
        image_list = [images.tensor[i] for i in range(b)]
        nested_images = nested_tensor_from_tensor_list(image_list)

        features = self.backbone(images.tensor)

        if not self.training:
            self.history_buffer.append(features)
            features = {
                k: torch.cat([f[k] for f in self.history_buffer]) for k in self.in_features
            }

        out = {}
        for name, x in features.items():
            m = nested_images.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # aggregate global features
            num_frames = x.size(0)
            if self.training:
                xis = []
                for i in range(num_frames // 2):
                    x_i = torch.cat([
                        x[None, i, :, :, :], x[None, i + num_frames // 2, :, :, :]
                    ], dim=1)
                    x_i = self.avg_pool(x_i) # (in_channel, H, W)
                    xis.append(x_i)
                x = torch.cat(xis, dim=0) # (local_frames, in_channel, H, W)
            else:
                x_i = torch.cat([
                    x[None, max(0, len(x) - 1 - self.local_frames), :, :, :], x[None, -1, :, :, :]
                ], dim=1)
                x = self.avg_pool(x_i) # (in_channel, H, W)
            out[name] = NestedTensor(x, mask[-x.shape[0]:])

        features = [out[name] for name in self.in_features]
        pos = [self.position_embed(f).to(f.tensors.dtype) for f in features]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = features.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.position_embed(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, query_embeds
        )

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        # (n_layers, batch_size, n_query, 1)
        outputs_class = torch.stack(outputs_classes)
        # (n_layers, batch_size, n_query, 4)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        
        # Training
        if self.training:
            # cxcywh
            loss_dict = self.criterion(out, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        # Inference
        assert len(batched_inputs) == 1
        frame_inputs = batched_inputs[0]
        box_cls = out["pred_logits"]
        box_pred = box_cxcywh_to_xyxy(out["pred_boxes"]) * images_whwh[:, None, :]
        results = self.inference(box_cls, box_pred, images.image_sizes)
        frame_result = results[0]
        height = frame_inputs.get("height", frame_inputs["image"].shape[-2])
        width = frame_inputs.get("width", frame_inputs["image"].shape[-1])
        processed_result = detector_postprocess(frame_result, height, width)

        return processed_result