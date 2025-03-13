# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from detectron2.utils.comm import get_rank
from ultrasound_vid.utils.misc import inverse_sigmoid
from ultrasound_vid.modeling.ops.modules import MSDeformAttn
from ultrasound_vid.modeling.heads.deformable_transformer import (
    DeformableTransformer,
    _get_clones, 
    _get_activation_fn,
)

frame_cache = namedtuple("frame_cache", ["image", "tgt", "query_embed", "memory", "lvl_embed", "valid_ratio", "mask"])


class ContextDeformableTransformer(DeformableTransformer):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, pos_temperature=10000,
                 temporal_attn=False, buffer_length=16,
                 context_feature=["res4"], context_frames=2, context_step=10, 
                 flownet=None, flownet_weights="pretrained_models/flownet.ckpt",):
        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
                 activation, return_intermediate_dec, num_feature_levels, dec_n_points,  enc_n_points,
                 two_stage, two_stage_num_proposals, pos_temperature, temporal_attn, buffer_length)

        self.num_context_frames = context_frames
        self.context_step = context_step
        self.flownet_weights = flownet_weights
        self.flownet = flownet
        self.init_flownet()

        decoder_layer = ContextDeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, 
            nhead, dec_n_points, temporal_attn
        )
        self.decoder = ContextDeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.context_image_buffer = deque(maxlen=self.num_context_frames)
        self.context_feature_buffer = deque(maxlen=self.num_context_frames)
        self.frame_idx = 0

    @property
    def device(self):
        try:
            rank = get_rank()
            return torch.device(f"cuda:{rank}")
        except:
            return torch.device("cpu")
    
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

    def single_frame_warp_context_features(self, curr_image, context_images, context_features):
        num_context_frames = len(context_images)
        img_cur_copies = curr_image.repeat(num_context_frames, 1, 1, 1)
        concat_imgs_pair = torch.cat([img_cur_copies, context_images], dim=1)
        flow, scale_map = self.flownet(concat_imgs_pair)
        warped_features_list = []
        for context_feature in context_features:
            H, W = context_feature.shape[-2:]
            flow_i = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=True)
            scale_map_i = F.interpolate(scale_map, size=(H, W), mode="bilinear", align_corners=True)
            warped_context_feature = self.resample(context_feature, flow_i)
            feature = warped_context_feature * scale_map_i
            feature = feature.flatten(2).permute(0, 2, 1)
            warped_features_list.append(feature)
        # (b, sum_l h*w, c)
        warped_features = torch.cat(warped_features_list, dim=1)
        return warped_features

    def extract_warp_context_features(self, local_images, context_images, context_features):
        warped_context_features_list = []
        for img in local_images:
            warped_context_features = self.single_frame_warp_context_features(
                img[None], context_images, context_features
            )
            warped_context_features_list.append(warped_context_features)
        # (t, b, sum_l h*w, c)
        warped_context_features = torch.stack(warped_context_features_list, dim=1)
        return warped_context_features

    def forward(self, srcs, masks, pos_embeds, query_embed=None, return_intermediate=False, encoder_only=False, images=None):
        """
        Params:
            srcs: List[Tensor(b,c,h,w)] backbone feature maps
            masks: List[Tensor(b,h,w)] feature maps valid masks
            pos_embeds: List[Tensor(b,c,h,w)] feature map pos_embeds
            query_embed: Tensor(nq, c*2) query embeddings
        """
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # (b, h*w, c)
            src = src.flatten(2).transpose(1, 2)
            # (b, h*w)
            mask = mask.flatten(1)
            # (b, h*w, c)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flatten, dim=1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], dim=1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            # (bs, nq, c)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # (bs, nq, 2)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        if not self.training:
            images, tgt, query_embed, memory, lvl_pos_embed_flatten, valid_ratios, mask_flatten = self.concat_temporal(
                images, tgt, query_embed, memory, lvl_pos_embed_flatten, valid_ratios, mask_flatten
            )

        # wrap context features
        if self.training:
            num_frames = len(images)
            context_indices = np.random.choice(
                num_frames, self.num_context_frames, replace=False
            )
            context_images = images[context_indices]
            context_features = memory[context_indices].split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
            context_features = [
                f.reshape(self.num_context_frames, H_, W_, self.d_model).permute(0, 3, 1, 2)
                for f, (H_, W_) in zip(context_features, spatial_shapes)
            ]
            warped_context_memory = self.extract_warp_context_features(
                local_images=images,
                context_images=context_images,
                context_features=context_features,
            )
        else:
            # update context buffer
            if self.frame_idx % self.context_step == 0:
                self.context_image_buffer.append(images[[-1]])
                self.context_feature_buffer.append(memory[[-1]].split([H_ * W_ for H_, W_ in spatial_shapes], dim=1))
            # context frames for current frame
            if self.frame_idx == 0:
                context_images = images[[-1]]
                context_features = memory[[-1]].split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
                context_features = [
                    f.reshape(1, H_, W_, self.d_model).permute(0, 3, 1, 2)
                    for f, (H_, W_) in zip(context_features, spatial_shapes)
                ]
            else:
                context_images = torch.cat([x for x in self.context_image_buffer], dim=0)
                context_features = [
                    torch.cat([
                        x[i].reshape(-1, H_, W_, self.d_model).permute(0, 3, 1, 2) 
                        for x in self.context_feature_buffer
                    ], dim=0) for i, (H_, W_) in enumerate(spatial_shapes)
                ]
            warped_context_memory = self.extract_warp_context_features(
                local_images=images,
                context_images=context_images,
                context_features=context_features,
            )
            self.frame_idx += 1

        # decoder
        hs, inter_references = self.decoder(
            tgt=tgt,                                    # (bs, nq, c)
            reference_points=reference_points,          # (bs, nq, 2)
            src=memory,                                 # (bs, sum_l h * w, c)
            src_t=warped_context_memory,                # (t, 1 or bs, sum_l h * w, c)
            src_spatial_shapes=spatial_shapes,          # (lvl, 2)
            src_level_start_index=level_start_index,    # (lvl, )
            src_valid_ratios=valid_ratios,              # (bs, lvl, 2)
            query_pos=query_embed,                      # (bs, nq, c)
            src_padding_mask=mask_flatten,              # (bs, sum_l h * w)
        )

        inter_references_out = inter_references

        if not self.training:
            hs = hs[:, [-1]]
            init_reference_out = init_reference_out[[-1]]
            inter_references_out = inter_references_out[:, [-1]]

        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None

    @torch.no_grad()
    def concat_temporal(self, image, tgt, query_embed, memory, lvl_pos_embed, valid_ratio, mask):
        """
        Params:
            tgt: (bs, nq, d_model)
            memory: (bs, sum_i: h_i*w_i, d_model)
            lvl_pos_embed: (bs, sum_i: h_i*w_i, d_model)
        """
        bs, nq, c = tgt.shape
        assert bs == 1, f"Only support bs=1, but receive bs={bs}"
        assert len(self.level_embed) == 1, f"Only support lvl=1, but receive lvl={len(self.level_embed)}"
        self.history_buffer.append(
            frame_cache(image, tgt, query_embed, memory, lvl_pos_embed, valid_ratio, mask)
        )
        image           = torch.cat([x.image for x in self.history_buffer], dim=0)
        tgt             = torch.cat([x.tgt for x in self.history_buffer], dim=0)
        query_embed     = torch.cat([x.query_embed for x in self.history_buffer], dim=0)
        memory          = torch.cat([x.memory for x in self.history_buffer], dim=0)
        lvl_pos_embed   = torch.cat([x.lvl_embed for x in self.history_buffer], dim=0)
        valid_ratio     = torch.cat([x.valid_ratio for x in self.history_buffer], dim=0)
        mask            = torch.cat([x.mask for x in self.history_buffer], dim=0)
        return image, tgt, query_embed, memory, lvl_pos_embed, valid_ratio, mask

    def reset(self):
        if self.temporal_attn:
            self.history_buffer.clear()
        self.context_image_buffer.clear()
        self.context_feature_buffer.clear()
        self.frame_idx = 0


class ContextDeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, temporal_attn=False):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # temporal attention
        self.temporal_attn = temporal_attn
        if temporal_attn:
            self.temp_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

        # context attention
        self.context_attn = FlowMSDeformAttn(d_model, n_levels, n_heads)
        self.aggregate_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout6 = nn.Dropout(dropout)
        self.norm5 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, 
                src, src_t, src_spatial_shapes, level_start_index, src_padding_mask=None):
        """
        Params:
            tgt: (bs, nq, c)
            query_pos: (bs, nq, c)
            reference_points: (bs, nq, 2)
            src: (bs, sum_l h * w, c)
            src_t: (t, bs, sum_l h * w, c)
            src_spatial_shapes: (lvl, 2)
            level_start_index: (lvl, )
            src_padding_mask: (bs, sum_l h * w)
        """
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # temporal attention
        if self.temporal_attn:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.temp_attn(q, k, tgt)[0]
            tgt = tgt + self.dropout5(tgt2)
            tgt = self.norm4(tgt)

        # context attention
        T = src_t.shape[0]; B, N, D = tgt.shape            
        multi_tgt = self.context_attn(self.with_pos_embed(tgt, query_pos), reference_points, 
                                        src_t, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt2 = self.aggregate_attn(tgt.reshape(1, B * N, D),
                                    multi_tgt.reshape(T, B * N, D),
                                    multi_tgt.reshape(T, B * N, D))[0].reshape(B, N, D)
        tgt = tgt + self.dropout6(tgt2)
        tgt = self.norm5(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class ContextDeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_t, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                # (bs, nq, 1, 2)
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, 
                           src, src_t, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                # (delta_qx, delta_qy, delta_qw, delta_qh)
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
