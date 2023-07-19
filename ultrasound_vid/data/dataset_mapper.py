import copy

import numpy as np
import torch
import torch.utils.data
from detectron2.data import detection_utils as d2utils
from detectron2.data import transforms as d2trans
from detectron2.data.common import DatasetFromList, MapDataset
from .utils import single_batch_collator, worker_init_reset_seed
from .frame_sampler import FrameSampler
from .video_utils import apply_augmentations_to_frames
from .video_utils import build_augmentation

__all__ = ["UltrasoundTrainingMapper", "UltrasoundTestMapper"]

FrameSamplerDict = {"FrameSampler": FrameSampler}


class UltrasoundTrainingMapper:
    """
    A callable which takes a dataset dict in Ultrasound Dataset format,
    and map it into a format used by the model.
    """

    def __init__(self, cfg, frame_sampler=None, is_train=True):
        self.tfm_gens = build_augmentation(cfg, is_train)
        if frame_sampler is not None:
            self.frame_sampler = FrameSamplerDict[frame_sampler](cfg)
        else:
            self.frame_sampler = FrameSamplerDict[cfg.DATASETS.FRAMESAMPLER](
                cfg)
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below

        # sample nearby frames for training
        frame_dicts = self.frame_sampler(dataset_dict)
        frame_images = []

        for i in range(len(frame_dicts)):
            image = d2utils.read_image(frame_dicts[i]["file_name"],
                                       format=self.img_format)
            d2utils.check_image_size(frame_dicts[i], image)
            frame_images.append(image)

        has_annotation = all(["annotations" in f for f in frame_dicts])
        if not has_annotation:
            frames, transforms = apply_augmentations_to_frames([],
                                                               frame_images)
        else:
            frames, transforms = apply_augmentations_to_frames(
                self.tfm_gens, frame_images)

        image_shape = frames[0].shape[:2]
        assert len(set([f.shape for f in frames
                        ])) == 1, "Got inconsistent frame shapes"

        for i in range(len(frame_dicts)):
            frame_dicts[i]["image"] = torch.as_tensor(
                np.ascontiguousarray(frames[i].transpose(2, 0,
                                                         1).astype("float32")))
            # Can use uint8 if it turns out to be slow some day

        # This is for validation, not for online prediction.
        if not self.is_train:
            for i in range(len(frame_dicts)):
                frame_dicts.pop("annotations", None)
            return frame_dicts

        has_annotation = all(["annotations" in f for f in frame_dicts])
        if self.is_train:
            assert has_annotation

        # Apply transform to annotations
        if has_annotation:
            for i in range(len(frame_dicts)):
                annos = [
                    d2utils.transform_instance_annotations(
                        obj, transforms, image_shape)
                    for obj in frame_dicts[i].pop("annotations")
                ]
                instances = d2utils.annotations_to_instances(
                    annos, image_shape)
                # Create a tight bounding box from masks, useful when image is cropped
                frame_dicts[i]["instances"] = d2utils.filter_empty_instances(
                    instances)

        return frame_dicts


class SingleFrameMapper:
    """
    A callable which takes a dataset dict of a frame, and map it into
    a format used by the model.
    """

    def __init__(self, cfg):
        self.tfm_gens = build_augmentation(cfg, is_train=False)
        self.img_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        image = d2utils.read_image(dataset_dict["file_name"],
                                   format=self.img_format)
        d2utils.check_image_size(dataset_dict, image)

        image, transforms = d2trans.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]
        image_tensor = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1).astype("float32")))
        dataset_dict["image"] = image_tensor
        # dataset_dict.pop("annotations", None)

        annos = [
            d2utils.transform_instance_annotations(obj, transforms,
                                                   image_shape)
            for obj in dataset_dict.pop("annotations")
        ]
        instances = d2utils.annotations_to_instances(annos, image_shape)
        # Create a tight bounding box from masks, useful when image is cropped
        dataset_dict["instances"] = d2utils.filter_empty_instances(instances)
        return dataset_dict


class UltrasoundTestMapper:
    """
    A callable which takes a dataset dict in Ultrasound Dataset format,
    and return a video reader, which is implement by Dataloader.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.mapper = SingleFrameMapper(cfg)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        video_info = copy.deepcopy(dataset_dict)
        frame_dicts = video_info["frame_anno"]
        frame_dicts = list(frame_dicts.values())
        dataset = DatasetFromList(frame_dicts)
        dataset = MapDataset(dataset, self.mapper)

        single_video_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=single_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

        return video_info, single_video_loader
