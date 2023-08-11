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


@META_ARCH_REGISTRY.register()
class TemporalRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape())
        self.num_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean",
                             torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std",
                             torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.to(self.device)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, inputs):
        if self.training:
            return self.forward_train(inputs)
        else:
            return self.forward_infer(inputs)

    def forward_train(self, batched_inputs):
        batchsize = len(batched_inputs)
        batched_inputs = list(chain.from_iterable(batched_inputs))
        assert len(batched_inputs) == batchsize * self.num_frames

        images, valid_ratios = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        else:
            raise AttributeError(
                "Failed to get 'instances' from training image.")

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)

        # if self.use_deform_context:
        #     _, detector_losses = self.roi_heads({"valid_ratios": valid_ratios},
        #                                         features, proposals,
        #                                         gt_instances)
        # elif self.use_flow_context:
        #     _, detector_losses = self.roi_heads(images, features, proposals,
        #                                         gt_instances)
        # else:
            # _, detector_losses = self.roi_heads(batched_inputs, features,
            #                                     proposals, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals,
                                                gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def forward_infer(self, frame_input):
        assert not self.training

        images, valid_ratios = self.preprocess_image([frame_input])
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)

        results, _ = self.roi_heads(images, features, proposals)
        if results is None:
            return None
        else:
            assert len(results) == 1
            r = results[0]
            return self.postprocess(r, frame_input, images.image_sizes[0])

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        if getattr(self.backbone, "sample_module", None) is not None:
            self.backbone.sample_module.reset()
        reset_op = getattr(self.roi_heads, "reset", None)
        if callable(reset_op):
            reset_op()
        else:
            log_first_n(logging.WARN, 'Roi Heads doesnt have function "reset"')

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [
            x["image"].to(self.device, non_blocking=True)
            for x in batched_inputs
        ]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = imagelist_from_tensors(images,
                                        self.backbone.size_divisibility)
        H, W = images.tensor.shape[-2:]
        valid_ratios = (torch.tensor(images.image_sizes).to(self.device) /
                        torch.tensor([H, W]).reshape(1, 2).to(self.device))
        return images, valid_ratios

    def postprocess(self, instances, frame_input, image_size):
        """
        Rescale the output instance to the target size.
        """
        height = frame_input.get("height", image_size[0])
        width = frame_input.get("width", image_size[1])
        res = detector_postprocess(instances, height, width)
        return res


@META_ARCH_REGISTRY.register()
class TemporalProposalNetwork(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape())
        self.num_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean",
                             torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std",
                             torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.to(self.device)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, inputs):
        if self.training:
            return self.forward_train(inputs)
        else:
            return self.forward_infer(inputs)

    def forward_train(self, batched_inputs):
        batchsize = len(batched_inputs)
        batched_inputs = list(chain.from_iterable(batched_inputs))
        assert len(batched_inputs) == batchsize * self.num_frames
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        features = self.backbone(images.tensor)
        _, proposal_losses = self.proposal_generator(images, features,
                                                     gt_instances)
        # calculate losses
        losses = {}
        losses.update(proposal_losses)
        return losses

    def forward_infer(self, frame_input):
        # for test mode
        assert not self.training
        images = self.preprocess_image([frame_input])
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        result = proposals[0]
        try:
            result.pred_boxes = result.proposal_boxes
            result.scores = result.objectness_logits
        except:
            pass
        return self.postprocess(result, frame_input, images.image_sizes[0])

    def reset(self):
        return

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [
            x["image"].to(self.device, non_blocking=True)
            for x in batched_inputs
        ]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = imagelist_from_tensors(images,
                                        self.backbone.size_divisibility)
        return images

    def postprocess(self, instances, frame_input, image_size):
        """
        Rescale the output instance to the target size.
        """
        height = frame_input.get("height", image_size[0])
        width = frame_input.get("width", image_size[1])
        return detector_postprocess(instances, height, width)
