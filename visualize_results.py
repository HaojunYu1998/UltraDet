"""
This file shows the results of one or many methods.
"""
import pickle

import torch
from detectron2.utils.logger import setup_logger
from fire import Fire
from path import Path
from ultrasound_vid.utils import VideoDatasetVisualizer

setup_logger(name="ultrasound_vid")
setup_logger(name="detectron2")


def visualize_annotation(save_dir, dataset_name):
    """可视化模型inference结果

    Parameters
    ----------
    save_dir : str
        可视化视频位置
    """

    # hyperparameters for multiprocessing.
    serialize = True  # Always set to true if multi-processing is applied.
    base_range = 40  # How many videos to be passed in a single multiprocessing step.
    workers = 0  # The number of workers to process the videos.
    vis_method = "video"

    visualizer = VideoDatasetVisualizer(
        dataset_name=dataset_name,
        save_folder=save_dir,
        show_anno=True,
        serialize=serialize,
        anno_only=True,
    )

    for i in range(int(len(visualizer._dataset_dicts) / base_range) + 1):
        show_range = [i * base_range, (i + 1) * base_range]
        assert vis_method == "video", "You can only visualize videos here!"
        visualizer.visualize_anno(
            vis_method, workers=workers, show_range=show_range
        )

    print("EOF")


def visualize_results(pred_dir, save_dir, dataset_name, score_thresh):
    """可视化模型inference结果

    Parameters
    ----------
    pred_dir : str
        模型inference的输出结果
    save_dir : str
        可视化视频位置
    """

    # hyperparameters for multiprocessing.
    serialize = True  # Always set to true if multi-processing is applied.
    base_range = 40  # How many videos to be passed in a single multiprocessing step.
    workers = 4  # The number of workers to process the videos.
    vis_method = "video"

    predictions = {}
    for pth_file in Path(pred_dir).files("*.pth"):
        predictions.update(torch.load(pth_file, map_location="cpu"))

    if serialize:
        print(f"Serializing objects...")
        for key in predictions.keys():
            predictions[key] = pickle.dumps(predictions[key], protocol=-1)
        print(f"Serializing done!")

    visualizer = VideoDatasetVisualizer(
        dataset_name=dataset_name,
        save_folder=save_dir,
        show_anno=True,
        serialize=serialize,
    )

    for i in range(int(len(predictions) / base_range) + 1):
        show_range = [i * base_range, (i + 1) * base_range]
        assert vis_method == "video", "You can only visualize videos here!"
        visualizer.visualize_all(
            vis_method, [predictions], workers=workers, show_range=show_range, score_thresh=score_thresh
        )

    print("EOF")


def visualize_tp_fp(pred_dir, save_dir, dataset_name, score_thresh):
    """可视化模型inference结果

    Parameters
    ----------
    pred_dir : str
        模型inference的输出结果
    save_dir : str
        可视化视频位置
    """

    # hyperparameters for multiprocessing.
    serialize = True  # Always set to true if multi-processing is applied.
    base_range = 40  # How many videos to be passed in a single multiprocessing step.
    workers = 4  # The number of workers to process the videos.
    vis_method = "video"

    predictions = {}
    for pth_file in Path(pred_dir).files("*.pth"):
        predictions.update(torch.load(pth_file, map_location="cpu"))

    if serialize:
        print(f"Serializing objects...")
        for key in predictions.keys():
            predictions[key] = pickle.dumps(predictions[key], protocol=-1)
        print(f"Serializing done!")

    visualizer = VideoDatasetVisualizer(
        dataset_name=dataset_name,
        save_folder=save_dir,
        show_anno=True,
        serialize=serialize,
    )

    for i in range(int(len(predictions) / base_range) + 1):
        show_range = [i * base_range, (i + 1) * base_range]
        assert vis_method == "video", "You can only visualize videos here!"
        visualizer.visualize_all(
            vis_method, [predictions], workers=workers, show_range=show_range, 
            score_thresh=score_thresh, tp_fp_mode=True, ignore_and_static=False,
            save_pdf=False, save_jpg=False
        )

    print("EOF")


if __name__ == "__main__":
    import os
    
    visualize_annotation(
        save_dir="visualizations/CVA_BUS_Anno",
        dataset_name="breast_cva_test"    
    )