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
from collections import deque, namedtuple
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from ultrasound_vid.utils.misc import inverse_sigmoid
from ultrasound_vid.modeling.ops.modules import MSDeformAttn
from ultrasound_vid.modeling.heads.deformable_transformer import (
    DeformableTransformer,
    DeformableTransformerDecoderLayer,
    DeformableTransformerDecoder,
    _get_clones, 
    _get_activation_fn
)

frame_cache = namedtuple("frame_cache", ["memory", "valid_ratio", "mask"])


class PTSEHead(DeformableTransformer):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=1, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, 
                 use_temporal_encoder=False, num_queries=16, buffer_length=12):
        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
                 activation, return_intermediate_dec, num_feature_levels, dec_n_points,  enc_n_points,
                 two_stage, two_stage_num_proposals)
        self.use_temporal_encoder = use_temporal_encoder
        self.num_queries = num_queries
        self.class_embed = None
        self.num_classes = 1
        self.num_frames = buffer_length
        assert not self.two_stage
                                                          
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.pre_decoder = DeformableTransformerDecoder(decoder_layer, num_layers=2, return_intermediate=True)

        self.s_decoder1 = SimpleDecoderV2(num_layers=2)
        self.s_decoder2 = SimpleDecoderV2(num_layers=2)
        self.corr = OursDecoderV2(num_layers=2)
        self.our_decoder = OursDecoderV2(num_layers=2)
        self.reference_points1 = nn.Linear(d_model, 2)
        self._reset_ptseformer_parameters()

    def _reset_ptseformer_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points1.weight.data, gain=1.0)
            constant_(self.reference_points1.bias.data, 0.)
        normal_(self.level_embed)

    @torch.no_grad()
    def concat_temporal(self, memory, valid_ratio, mask):
        """
        Params:
            tgt: (bs, nq, d_model)
            memory: (bs, sum_i: h_i*w_i, d_model)
            lvl_pos_embed: (bs, sum_i: h_i*w_i, d_model)
        """
        self.history_buffer.append(
            frame_cache(memory, valid_ratio, mask)
        )
        memory          = torch.cat([x.memory for x in self.history_buffer], dim=0)
        valid_ratio     = torch.cat([x.valid_ratio for x in self.history_buffer], dim=0)
        mask            = torch.cat([x.mask for x in self.history_buffer], dim=0)
        return memory, valid_ratio, mask

    def forward(self, srcs, masks, pos_embeds, query_embed=None, images=None):
        # -------------------------------1st stage------------------------------------
        ( 
            query_embeds,               # (nq, 2 * d_model)
            mem_stg1_cat,               # (bs, sum_i: h_i*w_i, d_model)
            spatial_shapes,             # (lvl, 2)
            level_start_index,          # (lvl, )
            valid_ratios,               # (bs, lvl, 2)
            mask_stg1_cat,              # (bs, sum_i: h_i*w_i)
        ) = super().forward(srcs, masks, pos_embeds, query_embed, encoder_only=True)

        if not self.training:
            mem_stg1_cat, valid_ratios, mask_stg1_cat = self.concat_temporal(mem_stg1_cat, valid_ratios, mask_stg1_cat)

        dec_utils = spatial_shapes, level_start_index, valid_ratios, mask_stg1_cat[[0]]
        bs = mem_stg1_cat.shape[0]
        # [[(1, h_i * w_i, d_model) for level] for frames]
        mem_cur_stg1_list = [mem_stg1_cat[[b]].split([H_ * W_ for H_, W_ in spatial_shapes], dim=1) for b in range(bs)] 
        mem_ref_stg1_list = [mem_stg1_cat[[b]].split([H_ * W_ for H_, W_ in spatial_shapes], dim=1) for b in range(bs)]

        # [(bs, h_i * w_i, d_model) for level]
        mem_cur_stg1_cat = [torch.cat(item, dim=0) for item in list(zip(*mem_cur_stg1_list))]
        # [(bs, h_i * w_i, d_model) for level]
        mem_ref_stg1_cat = [torch.cat(item, dim=0) for item in list(zip(*mem_ref_stg1_list))]
        
        # -------------------------------2nd stage------------------------------------
        # Spatial Transition Awareness Module (STAM)
        # [(bs, h_i * w_i, d_model) for level]
        mem_ref_stg2_list = []
        for mem_ref, mem_cur in zip(mem_ref_stg1_list, mem_cur_stg1_list):
            mem_ref_stg2_level_list = []
            for mem_ref_level, mem_cur_level in zip(mem_ref, mem_cur):
                mem_ref_stg2_list = []
                mem_ref_stg2 = self.corr(
                    tgt=mem_ref_level, memory=mem_cur_level
                ).squeeze(0)
                mem_ref_stg2_level_list.append(mem_ref_stg2)
            mem_ref_stg2_list.append(mem_ref_stg2_level_list)

        # [(bs, h_i * w_i, d_model) for level]
        mem_ref_stg2_cat = [torch.cat(item, dim=0) for item in list(zip(*mem_ref_stg2_list))]

        # Temporal Feature Aggregation Module (TFAM)
        # [(bs, h_i * w_i, d_model) for level]
        mem_cur_stg2 = []
        for mem_ref_cat_level, mem_cur_level in zip(mem_ref_stg1_cat, mem_cur_stg1_cat):
            mem_cur_stg2_level_list = []
            for mem_cur_level_per_frame in mem_cur_level:
                mem_cur_stg2_level = self.s_decoder1(
                    tgt=mem_cur_level_per_frame[None], memory=mem_ref_cat_level
                ).squeeze(0)
                mem_cur_stg2_level_list.append(mem_cur_stg2_level)
            mem_cur_stg2_level = torch.cat(mem_cur_stg2_level_list, dim=0)
            mem_cur_stg2.append(mem_cur_stg2_level)

        # -------------------------------3rd stage------------------------------------
        # Progressive Aggregation: Correlation
        # [(bs, h_i * w_i, d_model) for level]
        mem_cur_stg3 = []
        for mem_cur, mem_ref_cat in zip(mem_cur_stg2, mem_ref_stg2_cat):
            mem_final_level_list = []
            for mem_cur_per_frame in mem_cur:
                mem_final_level = self.s_decoder2(
                    tgt=mem_cur_per_frame[None], memory=mem_ref_cat
                ).squeeze(0)
                mem_final_level_list.append(mem_final_level)
            mem_final_level = torch.cat(mem_final_level_list, dim=0)
            mem_cur_stg3.append(mem_final_level)

        # -------------------------------4th stage------------------------------------
        # Progressive Aggregation: Gated Correlation
        # [(bs, h_i * w_i, d_model) for level]
        mem_cur_stg4 = []
        for mem_cur3, mem_cur1 in zip(mem_cur_stg3, mem_cur_stg1_cat):
            mem_level = self.our_decoder(tgt=mem_cur3, memory=mem_cur1).squeeze(0)
            mem_cur_stg4.append(mem_level)
        # (bs, cat_level_hw, d_model)
        mem_cur_stg4_cat = torch.cat(mem_cur_stg4, dim=1)

        # -------------------------------5th stage------------------------------------

        bs, _, _ = mem_cur_stg4_cat.shape
        query_embed = query_embeds[None].expand(bs, -1, -1)

        # -------------------------------final stage------------------------------------
        dec_utils = spatial_shapes, level_start_index, valid_ratios, mask_stg1_cat
        assert mask_stg1_cat.shape[:2] == mem_cur_stg4_cat.shape[:2], f"{mask_stg1_cat} != {mem_cur_stg4_cat}"
        hs, reference_points, inter_references = self.d_dec(mem_cur_stg4_cat, query_embed, dec_utils) #query_mix, dec_utils)

        if not self.training:
            hs, reference_points, inter_references = hs[:, [-1]], reference_points[[-1]], inter_references[:, [-1]]

        return hs, reference_points, inter_references, None, None

    def d_dec(self, memory, query_embeds, dec_utils):
        spatial_shapes, level_start_index, valid_ratios, mask_flatten = dec_utils
        tgt, reference_points, query_embed = self.prepare_dec(memory, query_embeds)
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        return hs, reference_points, inter_references

    def pre_dec(self, memory, query_embeds, dec_utils):
        spatial_shapes, level_start_index, valid_ratios, mask_flatten = dec_utils
        tgt, reference_points, query_embed = self.prepare_dec1(memory, query_embeds)
        hs, inter_references = self.pre_decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        return hs, reference_points, inter_references

    def prepare_dec(self, memory, query_embed):
        bs, _, c = memory.shape

        if len(query_embed.shape) == 3:
            query_embed, tgt = torch.split(query_embed, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()

        return tgt, reference_points, query_embed

    def prepare_dec1(self, memory, query_embed):
        bs, _, c = memory.shape

        if len(query_embed.shape) == 3:
            query_embed, tgt = torch.split(query_embed, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points1(query_embed).sigmoid()

        return tgt, reference_points, query_embed


class LSTMLikeSENet(nn.Module):
    def __init__(self, channel, r=16):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.w_z = nn.Linear(channel*2, channel)

    def forward(self, q, v):
        b, n, d = q.shape
        qv = torch.cat((q,v), dim=2)
        z = self.sigmoid(self.w_z(qv))
        h = (1 - z) * v + z * q
        o = h
        return o


class OursDecoderLayerV2(nn.Module):
    """
    modified from Transformerdecoderlayer
    delete selfattention & 2 linear functions
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.se = LSTMLikeSENet(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = self.se(tgt, memory)


        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.se(tgt, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class OursDecoderV2(nn.Module):
    """
    modified from TransformerDecoder
    """
    def __init__(self, num_layers, decoder_layer=None, norm=None, return_intermediate=False):
        super().__init__()
        if decoder_layer is None:
            decoder_layer = OursDecoderLayerV2(d_model=256, nhead=8)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class SimpleDecoderLayerV2(nn.Module):
    """
    modified from Transformerdecoderlayer
    delete selfattention & 2 linear functions
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class SimpleDecoderV2(nn.Module):
    """
    modified from TransformerDecoder
    """
    def __init__(self, num_layers, decoder_layer=None, norm=None, return_intermediate=False):
        super().__init__()
        if decoder_layer is None:
            decoder_layer = SimpleDecoderLayerV2(d_model=256, nhead=8)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        # print('output', output.shape)
        return output.unsqueeze(0)