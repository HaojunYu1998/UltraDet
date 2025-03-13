from detectron2.data import DatasetCatalog, MetadataCatalog
import re
import json
from path import Path
import pandas as pd
from collections import Counter
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
from collections import defaultdict
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
import json
import logging
import os
import pickle
import pandas as pd
from glob import glob
from path import Path

from detectron2.structures import BoxMode
from detectron2.utils.comm import is_main_process, get_rank, get_world_size, synchronize
from ultrasound_vid.data.utils import hash_idx


"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""
PROJ_ROOT = Path(__file__).abspath().parent.parent.parent
THING_CLASSES = ["lesion"]

logger = logging.getLogger(__name__)


def map_frame_annos_to_d2type(anno):
    video_folder = anno["video_folder"]
    video_id = anno["video_id"]
    assert not video_folder.endswith("/")
    relpath = anno["relpath"]
    frame_annos_ori = anno["frame_anno"]
    frame_annos_new = dict()
    for key, (frame_id, fanno) in enumerate(frame_annos_ori.items()):
        assert fanno["img"]["video_id"] == fanno["ann"][0]["video_id"]
        assert fanno["img"]["video_id"] == video_id
        assert fanno["img"]["id"] == fanno["ann"][0]["image_id"]
        assert frame_id == fanno["img"]["frame_id"]
        fanno_new = dict()
        fanno_new["dataset"] = anno["dataset"]
        frame_name = fanno["img"]["file_name"].split("/")[-1]
        fanno_new["file_name"] = video_folder / frame_name
        fanno_new["height"] = fanno["img"]["height"]
        fanno_new["width"] = fanno["img"]["width"]
        fanno_new["frame_idx"] = fanno["img"]["frame_id"]
        fanno_new["video_folder"] = video_folder
        fanno_new["relpath"] = relpath
        fanno_new["annotations"] = []
        for box_info in fanno["ann"]:
            x1, y1, w, h = box_info["bbox"]
            box_new = dict(
                bbox=[
                    x1, y1, x1+w, y1+h
                ],
                bbox_mode=BoxMode.XYXY_ABS,
                track_id=box_info["instance_id"],
                category_id=0, # box_info["category_id"]-1,
            )
            fanno_new["annotations"].append(box_new)
        frame_annos_new[frame_id] = fanno_new

    return frame_annos_new


def get_anno_by_video(coco_anno):
    imgs, vids = {}, {}
    coco_anno_by_video = defaultdict(dict)
    imgToAnns = defaultdict(list)
    if 'annotations' in coco_anno:
        for ann in coco_anno['annotations']:
            imgToAnns[ann['image_id']].append(ann)

    if 'images' in coco_anno:
        for img in coco_anno['images']:
            imgs[img['id']] = img

    if 'videos' in coco_anno:
        for vid in coco_anno['videos']:
            vids[vid['id']] = vid

    for img_id, img in imgs.items():
        ann = imgToAnns[img_id]
        video_id = img["video_id"]
        video_key = vids[video_id]["name"].split("/")[-1] + ".mp4@md5"
        frame_id = img["frame_id"]
        coco_anno_by_video[video_key][frame_id] = {
            "img": img,
            "ann": imgToAnns[img_id],
        }
    return coco_anno_by_video

def load_ultrasound_annotations(
    coco_anno, image_root, dataset_name
):
    logger.info(f">> >> >> Getting annotations start.")
    logger.info(f"dataset name: {dataset_name}")

    dataset_dicts = []
    # world_size = get_world_size()
    # rank = get_rank()
    coco_anno_by_video = get_anno_by_video(coco_anno)
    
    for vid_anno in coco_anno["videos"]:
        video_folder = Path(image_root) / vid_anno["name"]
        relpath = os.path.relpath(video_folder, image_root)
        video_key = video_folder.split("/")[-1] + ".mp4@md5"
        anno = dict()
        anno["dataset"] = dataset_name
        
        anno["video_id"] = vid_anno["id"]
        anno["relpath"] = relpath
        anno["video_key"] = video_key
        anno["video_folder"] = video_folder
        anno["frame_anno"] = coco_anno_by_video[video_key]
        anno["num_frames"] = len(anno["frame_anno"])
        anno["frame_anno"] = map_frame_annos_to_d2type(anno)
        anno["height"] = list(anno["frame_anno"].values())[0]["height"]
        anno["width"] = list(anno["frame_anno"].values())[0]["width"]
        assert anno["height"] == anno["width"]
        # anno.pop("frame_anno")
        dataset_dicts.append(anno)

    return dataset_dicts


def register_dataset(
    # jpg_root, pkl_root, anno_temp_path, us_processed_data, organ
    json_file, image_root, dataset_name
):
    coco_anno = json.load(open(json_file, "r"))
    num_videos = len(coco_anno["videos"])
    DatasetCatalog.register(
        dataset_name,
        lambda dataset_name=dataset_name: load_ultrasound_annotations(
            coco_anno, image_root, dataset_name
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=THING_CLASSES, num_videos=num_videos
    )


def build_bus_dataset(suffix, split):
    BUS_IMAGE_ROOT = (PROJ_ROOT / "datasets" / f"bus_data{suffix}" / "rawframes").realpath()
    BUS_JSON_FILE = (PROJ_ROOT / "datasets" / f"bus_data{suffix}" / f"{split}.json").realpath()
    # BUS_PROCESSED_DATA = BUS_JPG_ROOT.parent
    # BUS_ANNO_TEMP_PATH = PROJ_ROOT / "annotations" / f"bus{suffix}_annos_{split}"
    assert BUS_JSON_FILE.exists()
    # assert BUS_PKL_ROOT.exists()
    # assert BUS_PROCESSED_DATA.exists()
    # BUS_ANNO_TEMP_PATH.makedirs_p()
    register_dataset(
        BUS_JSON_FILE,
        BUS_IMAGE_ROOT,
        # BUS_PKL_ROOT,
        # BUS_ANNO_TEMP_PATH,
        # BUS_PROCESSED_DATA,
        f"breast{suffix}_{split}",
    )

# we don't filter out any videos
build_bus_dataset("_cva", "trainval")
build_bus_dataset("_cva", "test")