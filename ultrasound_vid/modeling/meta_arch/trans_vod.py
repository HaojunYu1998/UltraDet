import math
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess
from ultrasound_vid.utils.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from ultrasound_vid.utils.box_ops import box_cxcywh_to_xyxy
from ultrasound_vid.modeling.layers import MLP
from ultrasound_vid.modeling.layers.matcher import build_matcher
from ultrasound_vid.modeling.meta_arch.deformable_detr import DeformableDETR, SetCriterion


@META_ARCH_REGISTRY.register()
class TransVOD(DeformableDETR):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, cfg):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__(cfg)
        # re-build SetCriterion
        matcher             = build_matcher(cfg)
        self.aux_loss       = cfg.MODEL.DeformableDETR.AUX_LOSS
        self.decoder_layers = cfg.MODEL.DeformableDETR.DECODER_LAYERS
        class_weight        = cfg.MODEL.DeformableDETR.CLASS_WEIGHT
        giou_weight         = cfg.MODEL.DeformableDETR.GIOU_WEIGHT
        l1_weight           = cfg.MODEL.DeformableDETR.L1_WEIGHT
        losses = ['labels', 'boxes', 'cardinality']
        weight_dict = {
            'loss_ce': class_weight, 
            'loss_bbox': l1_weight,
            'loss_giou': giou_weight
        }
        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(self.decoder_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
                aux_weight_dict.update({k + f'tmp_{i}': v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = TemporalSetCriterion(
            num_classes=self.num_classes, 
            matcher=matcher, 
            weight_dict=weight_dict, 
            losses=losses,
        )
        # temporal head
        hidden_dim                      = self.transformer.d_model
        self.temp_class_embed           = nn.Linear(hidden_dim, self.num_classes)
        self.temp_bbox_embed            = MLP(hidden_dim, hidden_dim, 4, 3)
        self.transformer.class_embed    = self.class_embed[-1]

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.temp_class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.temp_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.temp_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.temp_bbox_embed.layers[-1].bias.data[2:], -2.0)

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
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
        else:
            batched_inputs = [batched_inputs]

        images, images_whwh = self.preprocess_image(batched_inputs)
        b = len(images.tensor)
        image_list = [images.tensor[i] for i in range(b)]
        nested_images = nested_tensor_from_tensor_list(image_list)

        features = self.backbone(images.tensor)

        out = {}
        for name, x in features.items():
            m = nested_images.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
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

        (
            hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, 
            final_hs, final_references_out
        ) = self.transformer(srcs, masks, pos, query_embeds)
        
        # spatial output, for key and reference frames
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
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {"pred_logits": enc_outputs_class, "pred_boxes": enc_outputs_coord}

        # temporal output, for key frame only
        assert final_hs is not None
        reference = inverse_sigmoid(final_references_out)
        output_class = self.temp_class_embed(final_hs)
        tmp = self.temp_bbox_embed(final_hs)
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        output_coord = tmp.sigmoid()
        out["tmp_outputs"] = [{"pred_logits": output_class, "pred_boxes": output_coord}]

        # Training
        if self.training:
            loss_dict = self.criterion(out, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        # Inference
        assert len(batched_inputs) == 1
        frame_inputs = batched_inputs[0]
        box_cls = out['tmp_outputs'][-1]["pred_logits"][[-1]]
        box_pred = box_cxcywh_to_xyxy(out['tmp_outputs'][-1]["pred_boxes"][[-1]]) * images_whwh[:, None, :]
        results = self.inference(box_cls, box_pred, images.image_sizes)
        frame_result = results[0]
        height = frame_inputs.get("height", frame_inputs["image"].shape[-2])
        width = frame_inputs.get("width", frame_inputs["image"].shape[-1])
        processed_result = detector_postprocess(frame_result, height, width)

        return processed_result

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        self.transformer.reset()

class TemporalSetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__(num_classes, matcher, weight_dict, losses, focal_alpha)


    def forward(self, outputs, targets):
        losses, num_boxes = super().forward(outputs, targets, return_intermediate=True)

        # In case of temporal losses
        if 'tmp_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['tmp_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_tmp_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
