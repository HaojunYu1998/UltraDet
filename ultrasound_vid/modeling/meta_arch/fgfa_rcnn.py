import logging
from collections import deque
from itertools import chain

import torch
import torch.nn.functional as F
from detectron2.modeling import build_roi_heads, build_backbone, META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.logger import log_first_n
from detectron2.structures import ImageList
from torch import nn
from ultrasound_vid.utils import imagelist_from_tensors
from ultrasound_vid.modeling.layers import FlowNetS, EmbedNet
from ultrasound_vid.modeling.meta_arch.temporal_rcnn import TemporalRCNN


@META_ARCH_REGISTRY.register()
class FGFA(TemporalRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_flow_context = "flowcontext" in cfg.MODEL.ROI_HEADS.NAME.lower()
        self.num_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES
        self.flownet_weights = cfg.MODEL.FLOWNET_WEIGHTS
        self.flownet = FlowNetS(cfg)
        self.embednet = EmbedNet()
        
        self.images_buffer = deque(maxlen=self.num_frames)
        self.features_buffer = deque(maxlen=self.num_frames)
        self.init_flownet()

    def init_flownet(self):
        ckpt = torch.load(self.flownet_weights, map_location=self.device)
        self.flownet.load_state_dict(ckpt["state_dict"], strict=False)
        for p in self.flownet.parameters():
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

    def compute_norm(self, embed):
        return torch.norm(embed, dim=1, keepdim=True) + 1e-10

    def compute_weight(self, embed_ref, embed_cur):
        embed_ref_norm = self.compute_norm(embed_ref)
        embed_cur_norm = self.compute_norm(embed_cur)

        embed_ref_normalized = embed_ref / embed_ref_norm
        embed_cur_normalized = embed_cur / embed_cur_norm

        weight = torch.sum(embed_ref_normalized * embed_cur_normalized, dim=1, keepdim=True)
        return weight

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, inputs):
        if self.training:
            return self.forward_train(inputs)
        else:
            return self.forward_infer(inputs)

    def single_frame_aggregate(self, idx, cur_image, ref_image, cur_feature, ref_feature):
        H, W = cur_feature.shape[-2:]
        num_ref = len(ref_image)
        img_cur_copies = cur_image.repeat(num_ref, 1, 1, 1)
        concat_imgs_pair = torch.cat([img_cur_copies, ref_image], dim=1)
        flow = self.flownet(concat_imgs_pair)
        flow = torch.zeros_like(ref_feature[:, :2])
        warped_ref_feature = self.resample(ref_feature, flow)
        h, w = warped_ref_feature.shape[-2:]
        cur_feature_ = F.interpolate(cur_feature, size=(h, w), mode="bilinear", align_corners=True)

        # calculate embedding and weights
        concat_feature = torch.cat([cur_feature_, warped_ref_feature], dim=0)
        concat_embed_feature = self.embednet(concat_feature)
        cur_embed, ref_embed = torch.split(concat_embed_feature, (1, num_ref), dim=0)

        unnormalized_weights = self.compute_weight(ref_embed, cur_embed)
        weights = F.softmax(unnormalized_weights, dim=0)
        feature = torch.sum(weights * warped_ref_feature, dim=0, keepdim=True)
        feature = F.interpolate(feature, size=(H, W), mode="bilinear", align_corners=True)
        return feature

    def forward_train(self, batched_inputs):
        batchsize = len(batched_inputs)
        batched_inputs = list(chain.from_iterable(batched_inputs))
        assert len(batched_inputs) == batchsize * self.num_frames

        images, images_flow = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            raise AttributeError("Failed to get 'instances' from training image.")

        features = self.backbone(images.tensor)

        # Flow-Guided Feature Aggregation
        image_tensor = images_flow.tensor
        fgfa_features = {}
        for k, feat in features.items():
            num_frame = len(feat)
            feat_list = []
            for idx in range(num_frame):
                feat_i = self.single_frame_aggregate(
                    idx,
                    cur_image=image_tensor[[idx]],
                    ref_image=image_tensor,
                    cur_feature=feat[[idx]],
                    ref_feature=feat,
                )
                feat_list.append(feat_i)
            feat = torch.cat(feat_list, dim=0)
            fgfa_features[k] = feat

        proposals, proposal_losses = self.proposal_generator(
            images, fgfa_features, gt_instances
        )

        if self.use_flow_context:
            _, detector_losses = self.roi_heads(
                images, fgfa_features, proposals, gt_instances
            )
        else:
            _, detector_losses = self.roi_heads(
                batched_inputs, fgfa_features, proposals, gt_instances
            )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def forward_infer(self, frame_input):
        # for test mode
        assert not self.training

        images, images_flow = self.preprocess_image([frame_input])
        features = self.backbone(images.tensor)

        self.images_buffer.append(images_flow.tensor)
        self.features_buffer.append(features)

        image_tensor = torch.cat([image for image in self.images_buffer], dim=0)
        all_features = {
            k: torch.cat([feat[k] for feat in self.features_buffer], dim=0)
            for k in features.keys()
        }

        # Flow-Guided Feature Aggregation
        fgfa_features = {}
        for k, feat in all_features.items():
            idx = -1
            feature = self.single_frame_aggregate(
                idx,
                cur_image=image_tensor[[idx]],
                ref_image=image_tensor,
                cur_feature=feat[[idx]],
                ref_feature=feat,
            )
            fgfa_features[k] = feature

        proposals, _ = self.proposal_generator(images, fgfa_features, None)
        results, _ = self.roi_heads(images, fgfa_features, proposals)
        assert results and len(results) == 1
        r = results[0]
        return self.postprocess(r, frame_input, images.image_sizes[0])

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        self.images_buffer.clear()
        self.features_buffer.clear()
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
        images = [x["image"].to(self.device, non_blocking=True) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images_flow = [x / 255.0 for x in images]
        images = imagelist_from_tensors(images, self.backbone.size_divisibility)
        images_flow = imagelist_from_tensors(images_flow, self.backbone.size_divisibility)
        return images, images_flow

    def postprocess(self, instances, frame_input, image_size):
        """
        Rescale the output instance to the target size.
        """
        height = frame_input.get("height", image_size[0])
        width = frame_input.get("width", image_size[1])
        return detector_postprocess(instances, height, width)
