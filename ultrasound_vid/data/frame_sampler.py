import logging
import pickle
import platform
import os
import copy
import json
import random
import numpy as np
from path import Path
from glob import glob


class FrameSampler:
    """
    Sample frames from a video segment for training.
    This sampler samples some nearby frames according to hard mining result.
    """

    def __init__(self, cfg):
        self.interval = cfg.INPUT.TRAIN_FRAME_SAMPLER.INTERVAL
        self.num_output_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES

    def __call__(self, video_anno_dict):
        num_frames = video_anno_dict["num_frames"]
        left_interval = self.interval // 2
        right_interval = self.interval - left_interval

        if num_frames > self.interval:
            sample_frame_idx_base = np.arange(
                left_interval, num_frames - right_interval
            ).tolist()  # frame index base
        else:
            sample_frame_idx_base = np.arange(0, num_frames).tolist()
        sample_frame_idx_base = np.array(sample_frame_idx_base)

        # sample center frame by weights
        weights = np.array([1.0 for _ in sample_frame_idx_base])
        weights = weights / weights.sum()
        center_frame_idx = np.random.choice(sample_frame_idx_base, p=weights)

        # sample frames from interval
        start_frame_idx = center_frame_idx - self.interval // 2
        start_frame_idx = max(start_frame_idx, 0)
        end_frame_idx = min(
            num_frames, start_frame_idx + self.interval
        )
        selected_indices = np.arange(start_frame_idx, end_frame_idx)
        replace_choise = self.num_output_frames > len(selected_indices)
        sample_indices = np.random.choice(
            selected_indices, size=self.num_output_frames, replace=replace_choise
        )
        sample_indices = sorted(sample_indices)
        frame_annos = video_anno_dict["frame_anno"]
        sample_frames = [frame_annos[i] for i in sample_indices]
        return sample_frames


class MemoryFrameSampler:
    """
    Sample frames from a video segment for training.
    This sampler samples some nearby frames according to hard mining result.
    """

    def __init__(self, cfg):
        self.interval = cfg.INPUT.TRAIN_FRAME_SAMPLER.INTERVAL
        self.memory_interval = cfg.INPUT.TRAIN_FRAME_SAMPLER.MEMORY_INTERVAL
        self.num_output_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES
        self.num_memory_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.MEMORY_FRAMES

    def __call__(self, video_anno_dict):
        num_frames = video_anno_dict["num_frames"]
        left_interval = self.interval // 2
        right_interval = self.interval - left_interval

        if num_frames > self.interval:
            sample_frame_idx_base = np.arange(
                left_interval, num_frames - right_interval
            ).tolist()  # frame index base
        else:
            sample_frame_idx_base = np.arange(0, num_frames).tolist()
        sample_frame_idx_base = np.array(sample_frame_idx_base)

        weights = np.array([1.0 for _ in sample_frame_idx_base])
        weights = weights / weights.sum()

        # sample local frames
        center_frame_idx = np.random.choice(sample_frame_idx_base, p=weights)
        start_frame_idx = center_frame_idx - self.interval // 2
        start_frame_idx = max(start_frame_idx, 0)
        end_frame_idx = min(
            num_frames, start_frame_idx + self.interval
        )
        selected_indices = np.arange(start_frame_idx, end_frame_idx)
        replace_choise = self.num_output_frames > len(selected_indices)
        sample_indices = np.random.choice(
            selected_indices, size=self.num_output_frames, replace=replace_choise
        )
        sample_indices = sorted(sample_indices)

        # sample memory frames
        start_memory_idx = start_frame_idx - self.memory_interval
        start_memory_idx = max(start_memory_idx, 0)
        end_memory_idx = max(start_frame_idx, start_memory_idx+1)
        selected_memory_indices = np.arange(start_memory_idx, end_memory_idx)
        replace_choise = self.num_memory_frames > len(selected_memory_indices)
        sample_memory_indices = np.random.choice(
            selected_memory_indices, size=self.num_memory_frames, replace=replace_choise
        )
        sample_memory_indices = sorted(sample_memory_indices)
        frame_annos = video_anno_dict["frame_anno"]
        sample_frames = [copy.deepcopy(frame_annos[i]) for i in sample_indices]
        sample_memory_frames = [copy.deepcopy(frame_annos[i]) for i in sample_memory_indices]
        return sample_memory_frames + sample_frames
