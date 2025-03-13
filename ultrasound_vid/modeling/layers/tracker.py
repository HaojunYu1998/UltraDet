import torch
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
from collections import OrderedDict, deque
from detectron2.structures import Instances, Boxes, pairwise_iou
from detectron2.layers import batched_nms
from scipy.optimize import linear_sum_assignment

from ultrasound_vid.utils.track_utils import KalmanFilter


NEW = 0
TRACKED = 1
LOST = 2
REMOVED = 3


class Tracklet(object):
    """
    A single tracklet.
    """
    def __init__(self, result, track_id, buffer_size=1, box_field_name="pred_boxes", score_field_name="scores"):
        """
        Params:
            result: the first prediction of this tracklet
        """
        assert len(result) == 1
        self.track_id = track_id
        self.state = NEW
        self.history = deque(maxlen=buffer_size)
        self.tracklet_len = 0
        self.box_field_name = box_field_name
        self.score_field_name = score_field_name
        self.lost_frames = 0
        self.score = float(result.get(score_field_name))
        self.kalman_filter = KalmanFilter(result.get(box_field_name))
        self.history.append(result)
        # self.features = result.pred_features

    # @profile
    def update(self, result):
        """
        Append the prediction for this tracklet in this frame
        """
        assert self.state != REMOVED
        self.state = TRACKED
        self.kalman_filter.correct(result.get(self.box_field_name))
        self.history.append(result)
        self.tracklet_len += 1
        self.lost_frames = 0

    def mark_lost(self):
        self.state = LOST
        self.lost_frames += 1

    def mark_removed(self):
        self.state = REMOVED
    
    @property
    def is_activated(self):
        return self.state in [NEW, TRACKED]

    @property
    def last_bbox(self):
        return self.history[-1].get(self.box_field_name)
    
    @property
    def last_score(self):
        return self.history[-1].get(self.score_field_name)

    def __repr__(self):
        return f'id: {self.track_id} len: {self.tracklet_len}'


class CacheTracker(object):
    """
    Tracker for post-processing detection results
    """
    def __init__(self, cfg):
        # List[STrack]
        self.all_tracks = {} # {id: tracklet}
        self.frame_id = 0
        self.sparse_nms_thresh = cfg.MODEL.TRACKER.SPARSE_NMS_THRESH
        self.new_score_thresh = cfg.MODEL.TRACKER.NEW_SCORE_THRESH
        self.valid_score_thresh = cfg.MODEL.TRACKER.VALID_SCORE_THRESH
        self.kalman_iou_thresh = cfg.MODEL.TRACKER.KALMAN_IOU_THRESH
        self.duplicate_iou_thresh = cfg.MODEL.TRACKER.DUPLICATE_IOU_THRESH
        self.track_buffer_size = cfg.MODEL.TRACKER.TRACK_BUFFER_SIZE
        self.next_id = 0

    def reset(self):
        self.all_tracks = {}
        self.next_id = 0
        self.frame_id = 0

    @staticmethod
    def modified_pairwise_iou(boxes1, boxes2, eps=1e-5):
        iou_mat = pairwise_iou(boxes1, boxes2)
        centers1 = boxes1.get_centers()
        centers2 = boxes2.get_centers()
        boxes1 = boxes1.tensor
        boxes2 = boxes2.tensor
        N1, N2 = len(boxes1), len(boxes2)
        boxes1 = boxes1[:, None].repeat(1, N2, 1)
        boxes2 = boxes2[None, :].repeat(N1, 1, 1)
        w1 = boxes1[..., 2] - boxes1[..., 0]
        h1 = boxes1[..., 3] - boxes1[..., 1]
        w2 = boxes2[..., 2] - boxes2[..., 0]
        h2 = boxes2[..., 3] - boxes2[..., 1]
        # dist_pho: (N1, N2)
        dist_pho = torch.pow(centers1[:, None] - centers2[None, :], 2).sum(dim=-1)
        # dist_c: (N1, N2)
        dist_c = torch.pow(
            torch.minimum(boxes1[..., :2], boxes2[..., :2]) - \
            torch.maximum(boxes1[..., 2:], boxes2[..., 2:]), 2
        ).sum(dim=-1)
        dist_mat = dist_pho / dist_c
        # v: (N1, N2)
        v = 4 / (math.pi ** 2) * torch.pow(torch.arctan(w2/(h2+eps)) - torch.arctan(w1/(h1+eps)), 2)
        alpha = v / (1-iou_mat + v + eps)
        # print("*"*10, iou_mat, dist_mat, alpha, v, "*"*10)
        return iou_mat - dist_mat - alpha * v

    def update(self, result):
        """
        Params:
            result: Instances
                * proposal_boxes or pred_boxes: Boxes
                * objectness_logits or scores: Tensor
        Return:
            result with pred_ids
        """
        box_field_name = "pred_boxes" if result.has("pred_boxes") else "proposal_boxes"
        score_field_name = "scores" if result.has("scores") else "objectness_logits"
        device = result.get(box_field_name).tensor.device
        # Filter valid results by scores.
        valid_mask = result.get(score_field_name) > self.valid_score_thresh
        valid_result = result[valid_mask]

        # Filter sparse boxes for first stage matching.
        boxes = valid_result.get(box_field_name).tensor
        scores = valid_result.get(score_field_name)
        classes = torch.zeros(len(valid_result)).long()
        sparse_inds = batched_nms(boxes, scores, classes, self.sparse_nms_thresh)
        sparse_result = valid_result[sparse_inds]

        # Filter all activated tracklets.
        kalman_pred_boxes = []
        activated_tracklet_ids = []
        for track_id, tracklet in self.all_tracks.items():
            if tracklet.is_activated:
                kalman_box = tracklet.kalman_filter.predict().to(device)
                kalman_pred_boxes.append(kalman_box)
                activated_tracklet_ids.append(track_id)

        # Matching.
        if len(activated_tracklet_ids) and len(sparse_result):
            # (n_pred, n_track)
            iou_mat = self.modified_pairwise_iou(sparse_result.get(box_field_name), Boxes.cat(kalman_pred_boxes)).cpu()
            pred_inds, track_inds = linear_sum_assignment(iou_mat, maximize=True)
            # Only the matched pairs with IoU larger than 0.75 are associated.
            valid_mask = (iou_mat[pred_inds, track_inds] > self.kalman_iou_thresh).flatten()
            pred_inds = torch.LongTensor(pred_inds)[valid_mask].tolist()
            track_inds = torch.LongTensor(track_inds)[valid_mask].tolist()
        else:
            pred_inds, track_inds = [], []

        sparse_matched_result = sparse_result[pred_inds]
        matched_ids = torch.LongTensor(activated_tracklet_ids)[track_inds].to(device)
        sparse_matched_result.set("pred_ids", matched_ids)

        unmatched_inds = [i for i in range(len(sparse_result)) if i not in pred_inds]
        sparse_unmatched_result = sparse_result[unmatched_inds]

        # Add result to matched tracklets.
        assert len(matched_ids) == len(sparse_matched_result)
        for i, track_id in enumerate(matched_ids):
            self.all_tracks[int(track_id)].update(sparse_matched_result[i])

        # Mark lost and removed.
        unmatched_track_ids = deepcopy(self.all_tracks.keys() - set(matched_ids.flatten().tolist()))
        for track_id in unmatched_track_ids:
            self.all_tracks[track_id].mark_lost()
            if tracklet.lost_frames > self.track_buffer_size:
                self.all_tracks.pop(track_id)

        # Start new tracklets.
        new_mask = sparse_unmatched_result.get(score_field_name) > self.new_score_thresh
        new_tracklet_result = sparse_unmatched_result[new_mask]
        new_ids = []
        for i in range(len(new_tracklet_result)):
            tracklet = Tracklet(
                new_tracklet_result[i], self.next_id, 
                box_field_name=box_field_name, 
                score_field_name=score_field_name
            )
            self.next_id += 1
            self.all_tracks[tracklet.track_id] = tracklet
            new_ids.append(tracklet.track_id)
        new_tracklet_result.set("pred_ids", torch.LongTensor(new_ids).to(device))

        result = Instances.cat([new_tracklet_result, sparse_matched_result])
        self.frame_id += 1
        return result