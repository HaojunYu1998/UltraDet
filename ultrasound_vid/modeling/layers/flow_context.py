import torch
import torch.nn as nn
from .relation import ROIRelationLayers


class TemporalContextAggregation(nn.Module):
    
    def __init__(self, cfg, d_model, head_num, dropout=0.1):
        super().__init__()
        self.use_temp_attn = cfg.MODEL.CONTEXT_TEMP_AGGREGATION
        # temporal attn
        if self.use_temp_attn:
            self.temporal_attn = nn.MultiheadAttention(d_model, head_num, dropout=dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)
        else:
            self.temporal_linear = nn.Linear(2 * d_model, d_model)

    @staticmethod
    def with_pos_tmp_embed(tensor, pos, tmp):
        tensor = tensor if pos is None else tensor + pos
        tensor = tensor if tmp is None else tensor + tmp
        return tensor

    def forward(self, tgt, multi_tgt):
        """
        Params:
            tgt: (B, N, D)
            multi_tgt: (T, B, N, D)
        Returns:
            box_feat: (B, N, D)
        """
        T, B, N, D = multi_tgt.shape
        if self.use_temp_attn:
            tgt2 = self.temporal_attn(
                self.with_pos_tmp_embed(tgt, None, None).reshape(1, B * N, D),
                self.with_pos_tmp_embed(multi_tgt, None, None).reshape(T, B * N, D),
                multi_tgt.reshape(T, B * N, D))[0].reshape(B, N, D)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        else:
            multi_tgt = multi_tgt.mean(dim=0)
            tgt = torch.cat([tgt, multi_tgt], dim=-1)
            tgt = self.temporal_linear(tgt)
        return tgt


class FlowContextLayers(ROIRelationLayers):
    """Update frame boxes every layer, and support multi-level features
    """

    def __init__(self, cfg, in_features):
        super().__init__(cfg, in_features)
        flow_layer = TemporalContextAggregation(cfg, in_features, self.head_num)
        self.flow_layers = _get_clones(flow_layer, self.layer_num)
        self.use_one_layer = cfg.MODEL.ROI_BOX_HEAD.USE_ONE_LAYER
        self.layer_index = cfg.MODEL.ROI_BOX_HEAD.NTCA_LAYER_INDEX

    def boxes_position(self, boxes):
        boxes = boxes.reshape(1, -1, 4)
        relative_position = self.extract_relative_position(boxes, boxes)
        position_embedding = self.embedding_relative_position(relative_position)
        return position_embedding

    def forward(self, boxes, features, multi_features, mask_exceptgt):
        """
        Params:
            boxes: (B, N, 4), input image size
            features: (B, N, D), box features
            multi_features: (T, B, N, D)
        Returns:
            tgt: (B, N, D)
        """
        B, N, D = features.shape
        mask_relation = mask_exceptgt.reshape(1, 1, B * N).expand(-1, B * N, -1)
        if self.use_attn:
            box_position = boxes.reshape(1, -1, 4)
        else:
            box_position = self.boxes_position(boxes)

        tgt = features
        multi_tgt = multi_features
        for lid, (flow_layer, layer) in enumerate(zip(self.flow_layers, self.layers)):
            # context attention
            if self.use_one_layer:
                if lid == self.layer_index:
                    tgt = flow_layer(tgt, multi_tgt)
            else:
                tgt = flow_layer(tgt, multi_tgt)
            # proposal relation
            tgt = tgt.reshape(1, B * N, D)
            tgt = tgt + layer(tgt, tgt, tgt, box_position, mask_relation)
            tgt = tgt.reshape(B, N, D)
        return tgt


def _get_clones(module, N):
    import copy
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])