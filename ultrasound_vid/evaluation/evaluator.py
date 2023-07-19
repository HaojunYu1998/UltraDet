import datetime
import logging
import time
import os
from torch.nn.parallel import DistributedDataParallel
from contextlib import contextmanager
import torch

from detectron2.utils.comm import is_main_process
from detectron2.utils import comm
from detectron2.evaluation.testing import print_csv_format
from detectron2.utils.logger import log_every_n_seconds
from .video_evaluation import UltrasoundVideoDetectionEvaluator
from ultrasound_vid.utils.comm import synchronize
from copy import deepcopy


def inference_on_video_dataset(model,
                               data_loader,
                               dataset_name,
                               save_folder=None,
                               skip_exists=False):
    """
    Run model on the data_loader and evaluate the metrics with evaluator
    on video dataset.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model. For video, it
            is assumed to be a video reader.
        dataset_name:
        save_folder:

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = (torch.distributed.get_world_size()
                   if torch.distributed.is_initialized() else 1)
    logger = logging.getLogger(__name__)
    logger.info("Start inference on dataset {}.".format(dataset_name))
    logger.info("{} videos are allocated to this device.".format(
        len(data_loader)))

    total_videos = len(
        data_loader)  # inference data loader must have a fixed length
    evaluator = UltrasoundVideoDetectionEvaluator(dataset_name)

    start_time = time.perf_counter()
    total_compute_time = 0.0 + 1e-6  # To avoid 0 video in one GPU
    total_frames = 0.0

    with video_inference_context(model), torch.no_grad():
        for i_video, (video_annotation,
                      video_reader) in enumerate(data_loader):
            dump_dir = os.path.join(save_folder, "predictions", dataset_name)
            relpath = video_annotation["relpath"]
            save_file = os.path.join(dump_dir,
                                     relpath.replace("/", "_") + ".pth")
            if skip_exists and os.path.exists(save_file):
                print(f"Skip {relpath}!")
                continue
            if isinstance(model, DistributedDataParallel):
                model.module.reset()
            else:
                model.reset()
            total_frames_i = len(video_reader)
            total_frames += total_frames_i
            compute_time_i = 0.0
            video_outputs = []
            for i_frame, frame in enumerate(video_reader):
                start_compute_time = time.perf_counter()
                output = model(frame)
                video_outputs.append(output)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                compute_time_i += time.perf_counter() - start_compute_time
                frames_per_second = (i_frame + 1) / compute_time_i
                eta = datetime.timedelta(
                    seconds=int(1 / frames_per_second *
                                (total_frames_i - i_frame - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Video {}/{}. Inference done {:5d}/{:5d} frames. {:2.2f} frames/s. ETA={}"
                    .format(
                        i_video + 1,
                        total_videos,
                        i_frame + 1,
                        total_frames_i,
                        frames_per_second,
                        str(eta),
                    ),
                    n=20,
                )

            total_compute_time += compute_time_i
            evaluator.process(video_annotation, video_outputs, save_folder)
            logger.info("Video {}/{} finished. ({:.2f} frames / s)".format(
                i_video + 1, total_videos, total_frames_i / compute_time_i))

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({:.2f} frames / s per device, on {} devices)"
        .format(total_time_str, total_frames / total_time, num_devices))
    total_compute_time_str = str(
        datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.2f} frames / s per device, on {} devices)"
        .format(total_compute_time_str, total_frames / total_compute_time,
                num_devices))
    synchronize()
    evaluator.load_predictions(dump_dir=save_folder)
    results = evaluator.evaluate(dump_dir=save_folder)
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def video_inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
