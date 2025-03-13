import math
import torch
import torch.nn as nn


class AttentionSELSA(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(self.d_model, self.d_model)
        self.linear_q = nn.Linear(self.d_model, self.d_model)
        self.linear_k = nn.Linear(self.d_model, self.d_model)
        self.linear_v = nn.Linear(self.d_model, self.d_model)
        self.linear_out = nn.Conv2d(self.d_model, self.d_model, 1)
        self.init_weights()

    def init_weights(self):
        for m in [self.linear, self.linear_q, self.linear_k, self.linear_v]:
            nn.init.normal(m.weight, 0.0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, feat, nongt_mask=None):
        """
        Params:
            feat: (N_all, D)
            nongt_mask: (N_all)
        """
        roi_feat = feat
        num_prop = len(roi_feat)
        nongt_roi_feat = feat if nongt_mask is None else feat[nongt_mask]
        num_nongt_prop = len(nongt_roi_feat)

        roi_feat2 = self.linear(feat)
        q_data = self.linear_q(roi_feat).reshape(num_prop, 1, self.d_model).permute(1, 0, 2)
        k_data = self.linear_k(nongt_roi_feat).reshape(num_nongt_prop, 1, self.d_model).permute(1, 2, 0)
        v_data = self.linear_v(nongt_roi_feat)
        aff = torch.bmm(q_data, k_data) # (1, all_prop, non_gt_prop)
        aff_scale = (1.0 / math.sqrt(float(self.d_model))) * aff
        aff_softmax = aff_scale.softmax(dim=2).view(num_prop, num_nongt_prop)
        # similarity = copy.deepcopy(aff_softmax)
        output = torch.mm(aff_softmax, v_data).reshape(num_prop, self.d_model, 1, 1)
        output = self.linear_out(output).reshape(num_prop, self.d_model)
        output = roi_feat2 + output
        return output

class SELSALayers(nn.Module):
    
    def __init__(self, cfg, in_features):
        super().__init__()
        self.head_num = cfg.MODEL.ROI_BOX_HEAD.RELATION_HEAD_NUMS
        self.layer_num = cfg.MODEL.ROI_BOX_HEAD.SELSA_LAYER_NUMS
        self.d_model = in_features
        selsa_layer = AttentionSELSA(in_features)
        self.selsa_layers = _get_clones(selsa_layer, self.layer_num)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, boxes, features, mask=None):
        nongt_mask = None if mask is None else mask[0, 0, :]
        x = features[0]
        for layer in self.selsa_layers:
            x = layer(x, nongt_mask)
        return x[None]


def _get_clones(module, N):
    import copy
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])