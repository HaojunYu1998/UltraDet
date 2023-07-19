import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionWithGeo(nn.Module):
    def __init__(self, in_features, head_num, bias=True, activation=F.relu, return_atten=False):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttentionWithGeo, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(
                "`in_features`({}) should be divisible by `head_num`({})".format(
                    in_features, head_num
                )
            )
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.return_atten = return_atten
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        self.geo_trans = nn.Linear(in_features, head_num, bias=bias)

    def forward(self, q, k, v, pos_embedding, mask=None):
        # we add pos_embedding for modeling object localization relationships
        # pos_embedding: [batchsize, channel, N_q, N_k]

        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        # to [batchsize * head_num, seq_len, sub_dim]
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        # geometric score: [batchsize * head_num, N_q, N_k]
        pos_embedding = pos_embedding.permute(0, 2, 3, 1)
        geo_score = self.geo_trans(pos_embedding)
        geo_score = geo_score.permute(0, 3, 1, 2)
        geo_score = self._reshape_to_atten(geo_score)
        geo_score = F.relu(geo_score)

        # appeareance score: [batchsize * head_num, N_q, N_k]
        dk = q.size()[-1]
        app_scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(dk)

        # combined score
        scores = (geo_score + 1e-6).log() + app_scores

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)
            mask = self._reshape_to_atten(mask)
            scores = scores.masked_fill(mask == 0, -1e9)
        atten = F.softmax(scores, dim=-1)
        y = atten.matmul(v)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        if self.return_atten:
            atten = self._reshape_from_atten(atten)
            return y, atten
        return y

    def _reshape_to_atten(self, x):
        batch_size, head_num, N_q, N_k = x.size()
        assert head_num == self.head_num
        return x.reshape(batch_size * head_num, N_q, N_k)

    def _reshape_from_atten(self, x):
        batch_size, N_q, N_k = x.size()
        batch_size //= self.head_num
        return (
            x.reshape(batch_size, self.head_num, N_q, N_k)
            .permute(0, 2, 3, 1)
        )

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return (
            x.reshape(batch_size, seq_len, self.head_num, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.head_num, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return (
            x.reshape(batch_size, self.head_num, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def extra_repr(self):
        return "in_features={}, head_num={}, bias={}, activation={}".format(
            self.in_features,
            self.head_num,
            self.bias,
            self.activation,
        )


class ROIRelationLayers(nn.Module):
    """Multi-Head Attention module for"""

    def __init__(self, cfg, in_features):
        super().__init__()
        head_num = cfg.MODEL.ROI_BOX_HEAD.RELATION_HEAD_NUMS
        layer_num = cfg.MODEL.ROI_BOX_HEAD.RELATION_LAYER_NUMS
        self.in_features = in_features
        self.layers = nn.ModuleList(
            [MultiHeadAttentionWithGeo(in_features, head_num) for _ in range(layer_num)]
        )
        feat_range = torch.arange(0, self.in_features / 8)
        div_mat = (
            torch.full_like(feat_range, 1000.0).pow(8.0 / self.in_features * feat_range)
            / 100.0
        )
        self.div_mat = nn.Parameter(div_mat, requires_grad=False)
        self.head_num = head_num
        self.layer_num = layer_num

    def forward(self, boxes, features, mask=None, return_intermediate=False):
        relative_position = self.extract_relative_position(boxes, boxes)
        position_embedding = self.embedding_relative_position(relative_position)

        x = features
        outputs = [x]
        for layer in self.layers:
            x = x + layer(x, x, x, position_embedding, mask)
            outputs.append(x)
        if return_intermediate:
            return outputs
        return x

    def forward_with_reference(self, boxes, features, ref_boxes, ref_features, mask=None):
        """
        Params:
            boxes: (1, N_loc, 4)
            features: (1, N_loc, D)
            ref_boxes: (1, N_mem, 4)
            ref_features: (Lyr, N_mem, D)
        """
        all_boxes = torch.cat([boxes, ref_boxes], dim=1)
        relative_position = self.extract_relative_position(boxes, all_boxes)
        position_embedding = self.embedding_relative_position(relative_position)
        tgt = features
        outputs = [tgt]
        for lid, layer in enumerate(self.layers):
            ref_tgt = ref_features[[lid]]
            tgt = tgt + layer(
                tgt,
                torch.cat([tgt, ref_tgt], dim=1),
                torch.cat([tgt, ref_tgt], dim=1),
                position_embedding, mask
            )
            outputs.append(tgt)
        return outputs

    @staticmethod
    def extract_relative_position(boxes_q, boxes_k, eps=1e-4):
        # batched position extraction

        cx_q = (boxes_q[..., 0] + boxes_q[..., 2]) / 2
        cy_q = (boxes_q[..., 1] + boxes_q[..., 3]) / 2
        w_q = (boxes_q[..., 2] - boxes_q[..., 0]).clamp(min=eps)
        h_q = (boxes_q[..., 3] - boxes_q[..., 1]).clamp(min=eps)

        cx_k = (boxes_k[..., 0] + boxes_k[..., 2]) / 2
        cy_k = (boxes_k[..., 1] + boxes_k[..., 3]) / 2
        w_k = (boxes_k[..., 2] - boxes_k[..., 0]).clamp(min=eps)
        h_k = (boxes_k[..., 3] - boxes_k[..., 1]).clamp(min=eps)

        rel_x = torch.log(
            (cx_k[:, None, :] - cx_q[:, :, None]).abs() / w_k[:, None, :] + eps
        )
        rel_y = torch.log(
            (cy_k[:, None, :] - cy_q[:, :, None]).abs() / h_k[:, None, :] + eps
        )
        rel_w = torch.log(w_q[:, :, None] / w_k[:, None, :] + eps)
        rel_h = torch.log(h_q[:, :, None] / h_k[:, None, :] + eps)

        rel_position = torch.stack([rel_x, rel_y, rel_w, rel_h], dim=-1)
        # [batchsize, N_q, N_k, 4]
        return rel_position

    def embedding_relative_position(self, relative_position):
        position_mat = relative_position.unsqueeze(-1).expand(
            -1, -1, -1, -1, self.div_mat.shape[-1]
        )
        position_mat = torch.div(position_mat, self.div_mat)
        sin_mat = position_mat.sin()
        cos_mat = position_mat.cos()
        embedding = torch.cat([sin_mat, cos_mat], dim=-1)
        embedding = embedding.reshape(*embedding.shape[:-2], self.in_features)
        # (B, D, Nq, Nk)
        embedding = embedding.permute(0, 3, 1, 2)
        return embedding


def _get_clones(module, N):
    import copy
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])