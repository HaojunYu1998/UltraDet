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
    DeformableTransformerEncoder,
    DeformableTransformerDecoder,
    _get_clones, 
    _get_activation_fn,
)

frame_cache = namedtuple("frame_cache", ["hs", "reference", "query_embed", "memory", "lvl_embed", "valid_ratio", "mask"])


class TransVODHead(DeformableTransformer):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=1, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, 
                 use_temporal_encoder=False, num_query_encoder_layers=3, 
                 num_temporal_decoder_layers=1, num_queries=16, buffer_length=12):
        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
                 activation, return_intermediate_dec, num_feature_levels, dec_n_points,  enc_n_points,
                 two_stage, two_stage_num_proposals)
        self.use_temporal_encoder = use_temporal_encoder
        self.num_queries = num_queries
        self.class_embed = None
        self.num_classes = 1
        self.num_frames = buffer_length
                                                          
        temporal_query_layer = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
        self.temporal_query_encoder = TemporalQueryEncoder(temporal_query_layer, num_query_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.temporal_decoder = DeformableTransformerDecoder(decoder_layer, num_temporal_decoder_layers, 
                                                             return_intermediate=False)

        self._reset_temporal_parameters()

    def _reset_temporal_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        ########## Spatial Transfomer ##########
        ( 
            hs,                         # (lyr, bs, nq, d_model)
            init_reference_out,         # (bs, nq, 2)
            inter_references_out,       # (lyr, bs, nq, 4)
            enc_outputs_class,          # (...)
            enc_outputs_coord_unact,    # (...)
            query_embed,                # (bs, nq, d_model)
            memory,                     # (bs, sum_i: h_i*w_i, d_model)
            lvl_pos_embed_flatten,      # (bs, sum_i: h_i*w_i, d_model)
            spatial_shapes,             # (lvl, 2)
            level_start_index,          # (lvl, )
            valid_ratios,               # (bs, lvl, 2)
            mask_flatten,               # (bs, sum_i: h_i*w_i)
                                        # bs=1 for inference, meaning one frame at a time
        ) = super().forward(srcs, masks, pos_embeds, query_embed, return_intermediate=True)

        # Prepare For Inference
        if not self.training:
            hs, inter_references_out, query_embed, memory, lvl_pos_embed_flatten, valid_ratios, mask_flatten = \
            self.concat_temporal(hs, inter_references_out, query_embed, memory, lvl_pos_embed_flatten, 
                                     valid_ratios, mask_flatten)
        
        # Temporal Query Encoder
        hs_enc = self.temporal_query_encoder(hs[-1], query_embed)

        # Temporal Deformable Transformer Decoder
        final_hs, final_references_out = self.temporal_decoder(
            hs_enc, inter_references_out[-1], memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten
        )

        return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, final_hs, final_references_out

    @torch.no_grad()
    def concat_temporal(self, hs, inter_references, query_embed, memory, lvl_pos_embed, valid_ratio, mask):
        """
        Params:
            hs: (lyr, bs, nq, d_model)
            inter_references: (lyr, bs, nq, 4)
            memory: (bs, sum_i: h_i*w_i, d_model)
            lvl_pos_embed: (bs, sum_i: h_i*w_i, d_model)
        """
        lyr, bs, nq, c = hs.shape
        assert bs == 1, f"Only support bs=1, but receive bs={bs}"
        assert len(self.level_embed) == 1, f"Only support lvl=1, but receive lvl={len(self.level_embed)}"
        self.history_buffer.append(
            frame_cache(hs, inter_references, query_embed, memory, lvl_pos_embed, valid_ratio, mask)
        )
        hs = [x.hs for x in self.history_buffer]
        inter_references = [x.reference for x in self.history_buffer]
        query_embed = [x.query_embed for x in self.history_buffer]
        memory = [x.memory for x in self.history_buffer]
        lvl_pos_embed = [x.lvl_embed for x in self.history_buffer]
        valid_ratio = [x.valid_ratio for x in self.history_buffer]
        mask = [x.mask for x in self.history_buffer]
        # (lyr, bs, nq, d_model)
        hs = torch.cat(hs, dim=1)
        # (lyr, bs, nq, 4)
        inter_references = torch.cat(inter_references, dim=1)
        # (bs, nq, 4)
        query_embed = torch.cat(query_embed, dim=0)
        # (bs, sum_i: h_i*w_i, d_model)
        memory = torch.cat(memory, dim=0)
        # (bs, sum_i: h_i*w_i, d_model)
        lvl_pos_embed = torch.cat(lvl_pos_embed, dim=0)
        # (bs, lvl, 2)
        valid_ratio = torch.cat(valid_ratio, dim=0)
        # (bs, sum_i: h_i*w_i)
        mask = torch.cat(mask, dim=0)
        return hs, inter_references, query_embed, memory, lvl_pos_embed, valid_ratio, mask

    def reset(self):
        self.history_buffer.clear()


class TemporalDeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, 
                 activation='relu', num_ref_frames=3, n_heads=8, n_points=4):
        super().__init__()

        # cross attention 
        self.cross_attn = MSDeformAttn(d_model, num_ref_frames, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

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

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, frame_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
    
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, frame_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt
    

class TemporalDeformableTransformerEncoder(DeformableTransformerEncoder):
    def __init__(self, encoder_layer, num_layers):
        super().__init__(encoder_layer, num_layers)

    def forward(self, tgt, pos, valid_ratios, src, spatial_shapes, frame_start_index, src_padding_mask=None):
        output = tgt
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, src, spatial_shapes, frame_start_index, src_padding_mask)

        return output


class TemporalQueryEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super().__init__()

        # self attention 
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # cross attention 
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

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
    
    def forward(self, query, query_pos=None):
        tgt = query

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # temporal attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.cross_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TemporalQueryEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, tgt, tgt_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, tgt_pos)
        return output