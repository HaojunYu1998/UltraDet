import logging
import os
from collections import OrderedDict, deque
from itertools import chain

import numpy as np
import torch
from detectron2.data import detection_utils as d2utils
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Instances, Boxes
from detectron2.structures import pairwise_iou
from detectron2.utils import comm


from ultrasound_vid.evaluation.mean_ap import eval_map
from ultrasound_vid.evaluation.average_recall import average_recall
from ultrasound_vid.evaluation.eval_utils import check_center_cover
from tempfile import NamedTemporaryFile
import pickle
import json

FP_ID = 0


class UltrasoundVideoDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name, rpn_only=False):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        if isinstance(dataset_name, list):
            self._dataset_name = dataset_name[0]
            self._dataset_list = dataset_name
            self.meta = MetadataCatalog.get(dataset_name[0])
        else:
            self._dataset_name = dataset_name
            self._dataset_list = None
            self.meta = MetadataCatalog.get(dataset_name)
        if self._dataset_list is not None:
            buf = []
            for dataset_name in self._dataset_list:
                buf.append(DatasetCatalog.get(dataset_name))
            dataset_dicts = chain.from_iterable(buf)
        else:
            dataset_dicts = DatasetCatalog.get(self._dataset_name)
        dataset_dicts = {d["relpath"]: d for d in dataset_dicts}
        self.dataset_dicts = dataset_dicts
        self._class_names = self.meta.thing_classes
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._prediction_files = list()
        self._rpn_only = rpn_only
        self.predictions = None

    def process(self, video_info, video_outputs, dump_dir):
        if self._rpn_only:
            dump_dir = os.path.join(dump_dir, "rpn_predictions", self._dataset_name)
        else:
            dump_dir = os.path.join(dump_dir, "predictions", self._dataset_name)
        os.makedirs(dump_dir, exist_ok=True)
        relpath = video_info["relpath"]
        video_outputs = [
            o.to(self._cpu_device) if isinstance(o, Instances) else o
            for o in video_outputs
        ]
        save_file = os.path.join(dump_dir, relpath.replace("/", "_")+".pth")
        torch.save({relpath: video_outputs}, save_file)
        self._prediction_files.append(save_file)

    def load_predictions(self, dump_dir):
        if self._rpn_only:
            dump_dir = os.path.join(dump_dir, "rpn_predictions", self._dataset_name)
        else:
            dump_dir = os.path.join(dump_dir, "predictions", self._dataset_name)
        if not comm.is_main_process():
            return
        from glob import glob
        prediction_files = glob(os.path.join(dump_dir, "*.pth"))
        print(f"loaded {len(prediction_files)} pred videos")
        self.predictions = {}
        for f in prediction_files:
            self.predictions.update(torch.load(f, map_location="cpu"))

    def evaluate(self, dump_dir=None, fixed_thresh=None):
        if not comm.is_main_process():
            return
        if self.predictions is None:
            assert dump_dir is not None
            self.load_predictions(dump_dir=dump_dir)
        dataset_dicts = self.dataset_dicts
        num_videos = len(dataset_dicts)
        dataset_dicts_temp_file = NamedTemporaryFile().name
        with open(dataset_dicts_temp_file, "wb") as fp:
            pickle.dump(dataset_dicts, fp)
        (
            mAP, AP50, AP75,
            eval_results,
            prob_thresh,
        ) = calculate_mmdet_ap(
            self.predictions,
            self._dataset_name,
            self.meta,
            dataset_dicts_temp_file,
            self._logger.info,
            fixed_recall=[0.7, 0.8, 0.9],
        )

        if dump_dir is not None and len(eval_results) > 0:
            eval_results_json = {}
            for k, v in eval_results[0].items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, np.float32):
                    v = float(v)
                eval_results_json[k] = v
            json.dump(
                eval_results_json,
                open(os.path.join(dump_dir, f"eval_results_{self._dataset_name}.json"), "w")
            )

        prob_thresh = prob_thresh if fixed_thresh is None else [fixed_thresh]
        video_fp_list = []
        for thresh in prob_thresh:
            video_fp = calculate_video_level_fp(
                self.predictions,
                self._dataset_name,
                self.meta,
                dataset_dicts_temp_file,
                iou_thresh=0.5,
                prob_thresh=[thresh],
                _print=self._logger.info,
            )
            video_fp_list.append(np.mean(list(video_fp.values())))
        ar = calculate_mmdet_ar(
            self.predictions,
            self._dataset_name,
            self.meta,
            dataset_dicts_temp_file,
            self._logger.info,
        )
        ret = OrderedDict()
        ret["Recall"] = {"R@16": ar}
        prec07 = [np.interp(0.7, x["recall"], x["precision"]) for x in eval_results]
        prec08 = [np.interp(0.8, x["recall"], x["precision"]) for x in eval_results]
        prec09 = [np.interp(0.9, x["recall"], x["precision"]) for x in eval_results]
        ret["Precision"] = {"P@R0.7": prec07[0], "P@R0.8": prec08[0], "P@R0.9": prec09[0]}
        ret["Average Precision"] = {"mAP": mAP, "AP50": AP50, "AP75": AP75}
        if fixed_thresh is None:
            FP07, FP08, FP09 = video_fp_list[0], video_fp_list[1], video_fp_list[2]
            ret["FP stat"] = {"FP@R0.7": FP07, "FP@R0.8": FP08, "FP@R0.9": FP09}
        else:
            ret["FP stat"] = {"FP": video_fp_list[0]}
        
        if dump_dir is not None:
            json.dump(
                ret,
                open(os.path.join(dump_dir, f"metrics_{self._dataset_name}.json"), "w")
            )
        
        if self._dataset_list is not None:
            ret["Stats"] = {"num_videos": num_videos}
        try:
            os.remove(dataset_dicts_temp_file)
            self._logger.info(">>> Dataset dicts temp file removed!")
        except:
            self._logger.info("=== Dataset dicts temp file removing failed!!! ===")
        return ret


def calculate_mmdet_ar(
    predictions, 
    dataset_name, 
    meta, 
    dataset_dicts_temp_file, 
    _print=print,
):
    with open(dataset_dicts_temp_file, "rb") as fp:
        dataset_dicts = pickle.load(fp)
    class_names = meta.thing_classes
    assert len(class_names) == 1, "we support one class only"
    _print(f"Evaluating mmdet-style AP on '{dataset_name}' dataset")
    preds, annos = [], []
    dump_anno = {}
    for relpath in predictions.keys():  # iterate video
        frame_preds = predictions[relpath]
        if relpath not in dataset_dicts:
            _print(f">>> {relpath} not in dataset")
            continue
        dump_anno[relpath] = []
        # with open(dataset_dicts[relpath]["frame_annos_path"], "rb") as fp:
        #     frame_annos = pickle.load(fp)
        frame_annos = dataset_dicts[relpath]["frame_anno"]
        assert len(frame_preds) == len(frame_annos)
        frame_annos = list(frame_annos.values())
        assert 0 <= (len(frame_annos) - len(frame_preds)) <= 30
        for pred, raw_anno in zip(frame_preds, frame_annos):  # iterate frame
            anno_image_shape = (raw_anno["height"], raw_anno["width"])
            anno = d2utils.annotations_to_instances(
                raw_anno["annotations"], anno_image_shape
            )
            if pred is None:
                continue
            try:
                pred_boxes = pred.get("proposal_boxes")
            except:
                pred_boxes = pred.get("pred_boxes")
            anno_boxes = anno.get("gt_boxes")
            iou_mat = pairwise_iou(pred_boxes, anno_boxes)
            match_mat = iou_mat > 0.5
            anno.set("matched", match_mat.any(0))
            raw_anno["anno"] = anno
            raw_anno["preds"] = pred
            dump_anno[relpath].append(raw_anno)
            mmdet_pred = []
            for class_id, _ in enumerate(class_names):
                try:
                    pred_boxes = pred.get("proposal_boxes").tensor.numpy()
                    pred_scores = (
                        pred.get("objectness_logits").float().sigmoid().numpy()
                    )
                except:
                    pred_boxes = pred.get("pred_boxes").tensor.numpy()
                    pred_scores = pred.get("pred_classes").float().sigmoid().numpy()
                if len(pred_boxes) == 0:
                    pred_boxes = np.zeros((0, 5))
                else:
                    pred_boxes = np.hstack((pred_boxes, pred_scores.reshape(-1, 1)))
                mmdet_pred.append(pred_boxes)
            anno_boxes = anno.get("gt_boxes").tensor.numpy()
            mmdet_anno = {
                "bboxes": anno_boxes,
                "labels": anno.get("gt_classes").numpy() + 1,  # note: we should add 1
                "labels_ignore": np.array([]),  # fake ignore to prevent bugs
                "bboxes_ignore": np.zeros((0, 4)),
            }
            preds.append(mmdet_pred)
            annos.append(mmdet_anno)
    ar = average_recall(preds, annos)
    return ar


def calculate_mmdet_ap(
    predictions,
    dataset_name,
    meta,
    dataset_dicts_temp_file,
    _print=print,
    fixed_recall=[0.7],
):
    with open(dataset_dicts_temp_file, "rb") as fp:
        dataset_dicts = pickle.load(fp)
    class_names = meta.thing_classes
    _print(f"Evaluating mmdet-style AP on '{dataset_name}' dataset")
    _print(f"The fixed_recall is {fixed_recall}")
    preds, annos = [], []
    for relpath in predictions.keys():  # iterate video
        frame_preds = predictions[relpath]
        if relpath not in dataset_dicts:
            _print(f">>> {relpath} not in dataset")
            continue
        # with open(dataset_dicts[relpath]["frame_annos_path"], "rb") as fp:
        #     frame_annos = pickle.load(fp)
        frame_annos = dataset_dicts[relpath]["frame_anno"]

        frame_preds = [p for p in frame_preds if p is not None]
        frame_annos = list(frame_annos.values())
        assert 0 <= (len(frame_annos) - len(frame_preds)) <= 30

        for pred, raw_anno in zip(frame_preds, frame_annos):  # iterate frame
            anno_image_shape = (raw_anno["height"], raw_anno["width"])
            anno = d2utils.annotations_to_instances(
                raw_anno["annotations"], anno_image_shape
            )
            mmdet_pred = []
            for class_id, _ in enumerate(class_names):
                temp_pred = pred[pred.get("pred_classes") == class_id]
                pred_boxes = temp_pred.get("pred_boxes").tensor.numpy()
                pred_scores = temp_pred.get("scores").numpy()
                if len(pred_boxes) == 0:
                    pred_boxes = np.zeros((0, 5))
                else:
                    pred_boxes = np.hstack((pred_boxes, pred_scores.reshape(-1, 1)))
                mmdet_pred.append(pred_boxes)
            anno_boxes = anno.get("gt_boxes").tensor.numpy()
            mmdet_anno = {
                "bboxes": anno_boxes,
                "labels": anno.get("gt_classes").numpy() + 1,  # note: we should add 1
                "labels_ignore": np.array([]),  # fake ignore to prevent bugs
                "bboxes_ignore": np.zeros((0, 4)),
            }
            preds.append(mmdet_pred)
            annos.append(mmdet_anno)
    AP50, eval_results, prob_thresh = eval_map(preds, annos, iou_thr=0.5, fixed_recall=fixed_recall)
    mAP = AP50
    for thr in range(55, 100, 5):
        iou_thr = thr / 100.0
        AP, _, _ = eval_map(preds, annos, iou_thr=iou_thr, fixed_recall=[0.7])
        mAP += AP
        if thr == 75: AP75 = AP
    mAP = mAP / len([i for i in range(55, 100, 5)])
    return (
        mAP, AP50, AP75,
        eval_results,
        prob_thresh[0],
    )


def calculate_video_level_fp(
    predictions,
    dataset_name,
    meta,
    dataset_dicts_temp_file,
    iou_thresh=0.5,
    prob_thresh=None,
    history_length=5,
    _print=print,
    tp_ratio=0.9,
):
    class BoxNode:
        def __init__(self, box, prev=None, istp=False, visited=False, fp_id=-1):
            self.box = box
            self.prev = prev
            self.istp = istp
            self.visited = visited
            self.fp_id = fp_id

    def fill_fp_id(node):
        global FP_ID
        temp = node
        buf = []
        while True:
            buf.append(temp)
            if temp.visited:
                final_fp_id = temp.fp_id
                break
            if temp.prev is None:
                final_fp_id = FP_ID
                FP_ID += 1
                break
            temp = temp.prev
        for item in buf:
            item.visited = True
            item.fp_id = final_fp_id

    # per video
    def mark_tp_preds(preds, annos):
        marked_box_nodes = []
        for _pred, _anno in zip(preds, annos):
            match_mat = check_center_cover(
                _pred.get("pred_boxes"), _anno.get("gt_boxes")
            )
            buf = []
            for k, istp in enumerate(match_mat.any(axis=1)):
                node = BoxNode(_pred.get("pred_boxes")[k], istp=istp.item())
                buf.append(node)
            marked_box_nodes.append(buf)
        return marked_box_nodes

    def merge_nodes(box_nodes):
        hist = deque(maxlen=history_length)
        # forward
        for nodes in box_nodes:
            if len(hist) > 0:
                flatten_hist_nodes = [n for hist_n in hist for n in hist_n]
                hist_boxes = [n.box for n in flatten_hist_nodes]
                pred_boxes = [n.box for n in nodes]
                if len(hist_boxes) > 0 and len(pred_boxes) > 0:
                    hist_boxes = Boxes.cat(hist_boxes)
                    pred_boxes = Boxes.cat(pred_boxes)
                    ious = pairwise_iou(pred_boxes, hist_boxes)
                    match_mat = ious > iou_thresh
                    istp = match_mat.any(dim=1)
                    for k in range(len(istp)):
                        if istp[k]:
                            prev_idx = ious[k].argmax()
                            nodes[k].prev = flatten_hist_nodes[prev_idx]
            hist.append(nodes)
        # backward
        for nodes in box_nodes[::-1]:
            for n in nodes:
                fill_fp_id(n)

    def select_fp_preds(box_nodes):
        global FP_ID
        fps = []
        for _id in range(FP_ID):
            buf = [n for nodes in box_nodes for n in nodes if n.fp_id == _id]
            seq_len = len(buf)
            tp_len = sum([n.istp for n in buf])
            if tp_len / seq_len > tp_ratio:
                continue
            fps.append(buf)
        return fps

    with open(dataset_dicts_temp_file, "rb") as fp:
        dataset_dicts = pickle.load(fp)
    class_names = meta.thing_classes
    single_class_flag = False
    if len(class_names) == 1:
        single_class_flag = True
    if prob_thresh is None:
        prob_thresh = [0.5] * len(class_names)
    _print(f"Calculating FP rate on '{dataset_name}' dataset")
    _print(f"Prob threshold is {prob_thresh}")
    video_fp_rate = {}
    for class_id, _ in enumerate(class_names):  # iterate class
        fp_cnt = 0
        frame_cnt = 0
        for relpath in predictions.keys():  # iterate videos
            frame_preds = predictions[relpath]
            if relpath not in dataset_dicts:
                _print(f">>> {relpath} not in dataset")
                continue
            # with open(dataset_dicts[relpath]["frame_annos_path"], "rb") as fp:
            #     frame_annos = pickle.load(fp)
            frame_annos = dataset_dicts[relpath]["frame_anno"]
            frame_preds = [p for p in frame_preds if p is not None]
            frame_annos = list(frame_annos.values())
            assert 0 <= (len(frame_annos) - len(frame_preds)) <= 30
            preds, annos = [], []

            for pred, raw_anno in zip(frame_preds, frame_annos):
                anno_image_shape = (raw_anno["height"], raw_anno["width"])
                anno = d2utils.annotations_to_instances(
                    raw_anno["annotations"], anno_image_shape
                )
                if not single_class_flag:
                    pred = pred[pred.get("pred_classes") == class_id]
                    anno = anno[anno.get("gt_classes") == class_id]

                # ignore low probability predictions
                pred = pred[pred.get("scores") > prob_thresh[class_id]]
                preds.append(pred)
                annos.append(anno)
            marked_box_nodes = mark_tp_preds(preds, annos)
            merge_nodes(marked_box_nodes)

            fps = select_fp_preds(marked_box_nodes)

            fp_cnt += len(fps)
            frame_cnt += min(len(frame_preds), len(frame_annos))
            global FP_ID
            FP_ID = 0
        video_fp_rate[class_id] = fp_cnt / frame_cnt * 1800
    return video_fp_rate