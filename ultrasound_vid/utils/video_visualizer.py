import logging
import os
import pickle
from copy import deepcopy
import numpy as np
from multiprocessing import Pool
from path import Path
from functools import partial
from PIL import Image

import torch

from detectron2.data import detection_utils as d2utils
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.structures.boxes import pairwise_iou, Boxes
from detectron2.utils.visualizer import Visualizer


class VideoDatasetVisualizer:
    def __init__(self, dataset_name, save_folder, show_anno=False, serialize=True, anno_only=False):
        """
        Visualizer for a dataset
        """
        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._save_folder = os.path.join(save_folder, dataset_name)
        os.makedirs(self._save_folder, exist_ok=True)
        self._metadata = MetadataCatalog.get(dataset_name)
        self._class_names = self._metadata.thing_classes
        dataset_dicts = DatasetCatalog.get(dataset_name)
        dataset_dicts = {d["relpath"]: d for d in dataset_dicts}

        self._dataset_dicts = dataset_dicts

        self.serialize = serialize
        self._show_anno = show_anno
        self._anno_only = anno_only

    def prepare_data(self, predictions_list, dataset_dicts, show_range):
        num_methods = len(predictions_list)

        self._predictions = dict()
        for i, key in enumerate(predictions_list[0].keys()):
            if show_range[0] <= i < show_range[1]:
                self._predictions[key] = []
                for i_method in range(num_methods):
                    self._predictions[key].append(predictions_list[i_method][key])
        self._dataset_dicts = dict()
        for key in self._predictions.keys():
            self._dataset_dicts[key] = dataset_dicts[key]

    def prepare_anno_only_data(self, dataset_dicts, show_range):
        self._dataset_dicts = dict()
        for i, key in enumerate(dataset_dicts.keys()):
            if show_range[0] <= i < show_range[1]:
                self._dataset_dicts[key] = dataset_dicts[key]

    # def draw_ignore_and_static(
    #     self, frame_visualizers, raw_anno, index, static_frame_id, H, W
    # ):
    #     if raw_anno["ignore"]:
    #         for frame_visualizer in frame_visualizers:
    #             frame_visualizer.draw_text("IGNORE", (W // 2, H // 2), font_size=24)

    #     if index in static_frame_id:
    #         for frame_visualizer in frame_visualizers:
    #             frame_visualizer.draw_text(
    #                 "STATIC", (W // 2, H // 2 + 24), font_size=24
    #             )

    def draw_annotation_and_predictions(
        self, frame_visualizers, raw_anno, pred, score_thresh, tp_fp_mode
    ):
        # draw annotations if needed
        annotations = [
            BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
            for x in raw_anno["annotations"]
        ]
        if self._show_anno:
            labels = [x["category_id"] for x in raw_anno["annotations"]]
            labels = [self._class_names[i] for i in labels]
            labels = ["anno: " + lb for lb in labels]
            colors = ["green"] * len(labels)
            for frame_visualizer in frame_visualizers:
                frame_visualizer.overlay_instances(
                    labels=labels, boxes=annotations, assigned_colors=colors
                )

        # draw predictions
        for i, frame_visualizer in enumerate(frame_visualizers):
            pred_i = pred[i]
            if pred_i is None:
                continue
            try:
                pred_i = pred_i[pred_i.scores > score_thresh]
                boxes = pred_i.pred_boxes
                scores = pred_i.scores.numpy()
            except:
                pred_i = pred_i[pred_i.objectness_logits > 0.0]
                boxes = pred_i.proposal_boxes
                scores = pred_i.objectness_logits.numpy()
            classes = pred_i.pred_classes.numpy()
            if tp_fp_mode:
                if len(annotations):
                    annos = Boxes(torch.Tensor(annotations).to(boxes.tensor))
                    iou_mat = pairwise_iou(boxes, annos)
                    tp_mask = (iou_mat > 0.5).any(dim=1).tolist()
                    assert len(tp_mask) == len(boxes)
                    colors = ["blue" if tp else "red" for tp in tp_mask]
                else:
                    colors = ["red"] * len(boxes)
            else:
                colors = ["blue"] * len(boxes)
            labels = [
                self._class_names[c] + f": {s:.3f}" for s, c in zip(scores, classes)
            ]
            
            
            frame_visualizer.overlay_instances(
                labels=labels, boxes=boxes, assigned_colors=colors
            )

    def visualize_anno(self, show_which, workers=0, show_range=None, **kwargs):
        assert show_which in ["video", "track"], f"{show_which} showing not valid."
        dataset_dicts = deepcopy(self._dataset_dicts)
        if show_range is None:
            show_range = [0, 40]
        self.prepare_anno_only_data(dataset_dicts, show_range)
        assert workers >= 0 and isinstance(workers, int)
        logger = self._logger
        logger.info(
            f'>> >> >> Visualizing predicted videos of dataset "{self._dataset_name}" start ... '
        )
        logger.info(
            f">> >> >> Visualizing videos ranging from {show_range[0] + 1} to {min(len(dataset_dicts), show_range[1])}... "
        )
        logger.info(f"{len(self._dataset_dicts)} videos gathered.")
        visualize_func = partial(self._visualize_video, **kwargs)
        if workers == 0:
            for i, relpath in enumerate(self._dataset_dicts.keys()):
                visualize_func(relpath)
        else:
            pool = Pool(workers)
            pool.map(visualize_func, list(self._dataset_dicts.keys()))
            pool.close()
            pool.join()

        logger.info(
            f'<< << << Visualizing predicted videos of dataset "{self._dataset_name}" end. '
        )
        self._dataset_dicts = dataset_dicts
        return

    def visualize_all(
        self, show_which, predictions_list, workers=0, show_range=None, **kwargs
    ):
        """
        :param predictions: dict(key: relpath, value: list of Instances)
        :param workers: number of workers
        :param show_range: the range of videos the visualizer will process
        :return: None
        """
        assert show_which in ["video", "track"], f"{show_which} showing not valid."

        dataset_dicts = deepcopy(self._dataset_dicts)
        if show_range is None:
            show_range = [0, 40]
        self.prepare_data(predictions_list, dataset_dicts, show_range)

        assert workers >= 0 and isinstance(workers, int)
        logger = self._logger
        logger.info(
            f'>> >> >> Visualizing predicted videos of dataset "{self._dataset_name}" start ... '
        )
        logger.info(
            f">> >> >> Visualizing videos ranging from {show_range[0] + 1} to {min(len(predictions_list[0]), show_range[1])}... "
        )
        logger.info(f"{len(self._predictions)} videos gathered.")

        visualize_func = partial(self._visualize_video, **kwargs)
        if workers == 0:
            for i, relpath in enumerate(self._predictions.keys()):
                visualize_func(relpath)
        else:
            pool = Pool(workers)
            pool.map(visualize_func, list(self._predictions.keys()))
            pool.close()
            pool.join()

        logger.info(
            f'<< << << Visualizing predicted videos of dataset "{self._dataset_name}" end. '
        )

        self._dataset_dicts = dataset_dicts

    def _visualize_video(
        self,
        relpath,
        score_thresh=0.5,
        skip_exist=False,
        tp_fp_mode=False,
        ignore_and_static=True,
        save_pdf=False,
        save_jpg=False,
        crf=26,
        custom_ffmpeg="~/.bin/ffmpeg-git-20220910-amd64-static/ffmpeg",
    ):
        if skip_exist:
            filename = relpath.replace("/", "_") + ".mp4"
            output_filename = os.path.join(self._save_folder, filename)
            if os.path.exists(output_filename):
                self._logger.info(f"{relpath} already existed. Skip.")
                return
        self._logger.info(f"{relpath} started.")

        frame_annos = self._dataset_dicts[relpath]["frame_anno"]

        # static_frame_id = self._dataset_dicts[relpath]["video_info"]["static_frames"]

        width = self._dataset_dicts[relpath]["width"]
        height = self._dataset_dicts[relpath]["height"]

        # ignore delayed frames
        # We assume that delay is no more than 30 frames
        frame_annos = list(frame_annos.values())

        num_methods = 1
        if not self._anno_only:
            frame_preds = self._predictions[relpath]
            if self.serialize:
                frame_preds = [pickle.loads(frame_pred) for frame_pred in frame_preds]
            num_methods = len(frame_preds)
            assert 0 <= (len(frame_annos) - len(frame_preds[0])) <= 30

        from vidgear.gears import WriteGear

        filename = relpath.replace("/", "_") + ".mp4"
        output_params = {
            "-vcodec": "libx264",
            "-crf": crf,
            "-input_framerate": 30,
            "-output_dimensions": (width, num_methods * height),
            "-preset": "ultrafast",
        }
        os.environ["NUMEXPR_MAX_THREADS"] = "16"
        video_writer = WriteGear(
            output_filename=os.path.join(self._save_folder, filename),
            compression_mode=True,
            custom_ffmpeg=os.path.expanduser(custom_ffmpeg),
            logging=False,
            **output_params,
        )

        for i in range(len(frame_annos)):
            raw_anno = frame_annos[i]
            if not self._anno_only:
                pred = [frame_preds[i_method][i] for i_method in range(num_methods)]
                if not all(pred):
                    pred = [None for _ in range(num_methods)]
            else:
                assert num_methods == 1
                pred = [None for _ in range(num_methods)]
            frame = d2utils.read_image(raw_anno["file_name"], format="RGB")
            frame_visualizers = [
                Visualizer(deepcopy(frame), self._metadata) for _ in range(num_methods)
            ]

            H, W, *_ = frame.shape

            # if ignore_and_static:
            #     self.draw_ignore_and_static(
            #         frame_visualizers, raw_anno, i, static_frame_id, H, W
            #     )

            self.draw_annotation_and_predictions(
                frame_visualizers, raw_anno, pred, score_thresh, tp_fp_mode=tp_fp_mode
            )

            frame_outs = [
                frame_visualizer.output.get_image()
                for frame_visualizer in frame_visualizers
            ]
            frame_out = np.concatenate(frame_outs)
            if save_pdf:
                pdf_folder = os.path.join(self._save_folder, filename.replace(".mp4", ""))
                os.makedirs(pdf_folder, exist_ok=True)
                pdf_file = os.path.join(pdf_folder, f"frame_{i}.pdf")
                Image.fromarray(frame_out).save(pdf_file)
            if save_jpg:
                jpg_folder = os.path.join(self._save_folder, filename.replace(".mp4", ""))
                os.makedirs(jpg_folder, exist_ok=True)
                jpg_file = os.path.join(jpg_folder, f"frame_{i}.jpg")
                Image.fromarray(frame_out).save(jpg_file)
            video_writer.write(frame_out, rgb_mode=True)

        video_writer.close()
        self._logger.info(f"{relpath} finished.")