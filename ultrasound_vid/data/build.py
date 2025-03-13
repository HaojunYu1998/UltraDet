import itertools
import logging
import os
import json
import copy
import pickle
import torch
import torch.utils.data
import numpy as np
from detectron2.data import samplers
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.utils.comm import get_world_size, get_rank, is_main_process

from .dataset_mapper import UltrasoundTrainingMapper, UltrasoundTestMapper
from .samplers import InferenceSampler, IsolateTrainingSampler
from .utils import (
    hash_idx,
    trivial_batch_collator,
    single_batch_collator,
    worker_init_reset_seed,
)


def get_patient_id(video_name_in_db: str) -> str:
    import re
    from path import Path
    ptn = re.compile(
        r"(.*?)[-_]\d?-?[lrLR]?$",
    )
    patient_id = Path(video_name_in_db).basename().splitext()[0]
    match_ret = ptn.findall(patient_id)
    if match_ret:
        patient_id = match_ret[0]
    return str(patient_id)


def get_video_detection_dataset_dicts(dataset_names, cfg, is_train=True):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (list[str]): a list of dataset names
        cfg: config instance
        is_train:
    """

    assert len(dataset_names)
    dataset_dicts = []
    num_folds = cfg.DATASETS.NUM_FOLDS
    test_folds = cfg.DATASETS.TEST_FOLDS
    for dataset_name in dataset_names:
        cur_dataset_dicts = DatasetCatalog.get(dataset_name)
        assert len(cur_dataset_dicts), "Dataset '{}' is empty!".format(dataset_name)
        cur_dataset_dicts.sort(key=lambda d: d["relpath"])
        if is_train:
            cur_dataset_dicts = [
                d
                for d in cur_dataset_dicts
                if hash_idx(get_patient_id(d["video_key"].split("@")[0]), num_folds)
                not in test_folds
            ]
        else:
            cur_dataset_dicts = [
                d
                for d in cur_dataset_dicts
                if hash_idx(get_patient_id(d["video_key"].split("@")[0]), num_folds)
                in test_folds
            ]
        dataset_dicts.append(cur_dataset_dicts)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    if len(dataset_dicts) == 0:
        return dataset_dicts

    print_dataset_statistics(dataset_dicts)

    return dataset_dicts


class balanced_multi_data_loader:
    """
    An iterable object that behaviors like 'torch.utils.data.DataLoader'.
    It contains several dataloaders of different datasets and returns datapoints from each dataset with the same probability.
    """

    def __init__(self, dataloaders):
        self.num_dataloaders = len(dataloaders)
        self.dataloaders = dataloaders
        self.dataloader_iters = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        idx = torch.randint(self.num_dataloaders, (1,)).item()
        return next(self.dataloader_iters[idx])


def build_video_detection_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:
      * Map each metadata dict into another format to be consumed by the model.
      * Batch them by simply putting dicts into a list.
    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        a torch DataLoader object
    """
    split = cfg.DATASETS.SPLIT
    suffix = cfg.DATASETS.SUFFIX
    if is_main_process():
        print(f"Using {split} split.")
        if len(suffix) > 0: 
            print(f"Using {suffix} data.")
    
    num_workers = get_world_size()
    segs_per_batch = cfg.SOLVER.SEGS_PER_BATCH
    assert (
        segs_per_batch % num_workers == 0
    ), "SOLVER.SEGS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        segs_per_batch, num_workers
    )
    assert (
        segs_per_batch >= num_workers
    ), "SOLVER.SEGS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        segs_per_batch, num_workers
    )
    segs_per_worker = segs_per_batch // num_workers

    dataset_dicts = []
    dataset_dicts.append(
        get_video_detection_dataset_dicts(
            [f"breast{suffix}_{split}"],
            cfg,
            is_train=True,
        )
    )
    datasets = [
        DatasetFromList(dataset_dict, copy=False) for dataset_dict in dataset_dicts
    ]

    if mapper is None:
        mapper = UltrasoundTrainingMapper(cfg, is_train=True)
    datasets = [MapDataset(dataset, mapper) for dataset in datasets]

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    frame_sampler_name = cfg.DATASETS.FRAMESAMPLER
    logger = logging.getLogger(__name__)
    logger.info(
        "Using training sampler {} with frame sampler {}".format(
            sampler_name, frame_sampler_name
        )
    )
    data_loaders = []
    for dataset in datasets:
        if sampler_name == "TrainingSampler":
            sampler = samplers.TrainingSampler(len(dataset))
        elif sampler_name == "IsolateTrainingSampler":
            sampler = IsolateTrainingSampler(len(dataset))
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, segs_per_worker, drop_last=True
        )
        collate_fn = trivial_batch_collator
        # drop_last so the batch always have the same size

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_reset_seed,
        )
        data_loaders.append(data_loader)

    return balanced_multi_data_loader(data_loaders)


def build_video_detection_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_video_detection_dataset_dicts(dataset_name, cfg, is_train=False)
    if len(dataset_dicts) == 0:
        return None

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = UltrasoundTestMapper(cfg)
    dataset = MapDataset(dataset, mapper)

    video_frame_nums = [anno["num_frames"] for anno in dataset_dicts]
    sampler = InferenceSampler(video_frame_nums)
    # Always use 1 videos per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    # since data loader returns a new dataloader for a video, it is no
    # need for using multiprocessing here.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_sampler=batch_sampler,
        collate_fn=single_batch_collator,
    )
    return data_loader


def print_dataset_statistics(dataset_dicts):
    """
    Print dataset statistics.
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
    """
    logger = logging.getLogger(__name__)
    video_num = len(dataset_dicts)
    total_frame_length = 0
    total_object_number = 0
    total_unique_object_number = 0

    for video_dict in dataset_dicts:
        frame_annos = video_dict["frame_anno"]
        total_frame_length += len(frame_annos)
        for frame_id in frame_annos:
            frame_dict = frame_annos[frame_id]
            total_object_number += len(frame_dict["annotations"])
        if "box_tracks" in video_dict:
            total_unique_object_number += len(video_dict["box_tracks"])

    logger.info(f">> >> >> Print dataset statistics. start.")
    logger.info(f"video_num : {video_num}")
    logger.info(f"total_frame_length : {total_frame_length}")
    if video_num == 0:
        return
    logger.info(f"average_frame_length : {total_frame_length / video_num}")
    logger.info(f"total_time_length : {total_frame_length / 30}")
    logger.info(f"average_time_length : {total_frame_length / (video_num * 30)}")
    logger.info(f"total_object_number : {total_object_number}")
    logger.info(f"total_unique_object_number : {total_unique_object_number}")
    logger.info(
        f"average_unique_object_number(per frame) : {total_unique_object_number / total_frame_length}"
    )
    logger.info(f"<< << << Print dataset statistics. done.")
