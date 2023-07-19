"""
Lesion detection for ultrasound video Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import logging
import time
import torch

import numpy as np
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

from collections import OrderedDict
from typing import Any, Dict, List, Set
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import DatasetEvaluator, print_csv_format
from detectron2.utils.logger import setup_logger
from detectron2.utils.comm import is_main_process
from detectron2.solver.build import maybe_add_gradient_clipping

from ultrasound_vid.config import (
    add_ultrasound_config,
)
from ultrasound_vid.data import (
    build_video_detection_train_loader,
    build_video_detection_test_loader,
)
from ultrasound_vid.evaluation import inference_on_video_dataset
from ultrasound_vid.utils.misc import backup_code


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Set "find_unused_parameters=True" to prevent empty gradient bug.
        Set "refresh period" to refresh dataloader periodicly when datasets are
        modified during training.
        """
        super().__init__(cfg)
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                self.model.module,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True,
                check_reduction=False,
            )
            self.model = model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluator()

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_video_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_video_detection_train_loader(cfg)

    @classmethod
    def build_optimizer(cls, cfg, model):
        optimizer_type = cfg.SOLVER.get("OPTIMIZER", "SGD")
        if is_main_process():
            print(f"Using optimizer {optimizer_type}")
        if optimizer_type == "SGD":
            optimizer = super().build_optimizer(cfg, model)
            return optimizer

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        mul_name, mul_value = cfg.SOLVER.LR_MULTIPLIER_NAME, cfg.SOLVER.LR_MULTIPLIER_VALUE
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            for k, v in zip(mul_name, mul_value):
                if k in key:
                    lr = lr * v
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if optimizer_type.upper() == "ADAMW":
            optimizer = torch.optim.AdamW(
                params,
                cfg.SOLVER.BASE_LR,
                betas=cfg.SOLVER.ADAM_BETA,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        split = cfg.DATASETS.SPLIT
        suffix = cfg.DATASETS.SUFFIX
        logger = logging.getLogger("ultrasound_vid")
        output_dir = cfg.OUTPUT_DIR
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        skip_exists = cfg.TEST.SKIP_EXISTS
        results = OrderedDict()
        dataset_name = f"breast{suffix}_{split}"
        data_loader = cls.build_test_loader(cfg, [dataset_name])
        results_i = inference_on_video_dataset(
            model,
            data_loader,
            dataset_name,
            save_folder=output_dir,
            skip_exists=skip_exists,
        )
        results[dataset_name] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info(
                "Evaluation results for {} in csv format:".format(dataset_name)
            )
            print_csv_format(results_i)
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ultrasound_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.AUTO_DIR:
        cfg.OUTPUT_DIR = os.path.join(
            "outputs", os.path.splitext(os.path.basename(args.config_file))[0]
        )

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name="ultrasound_vid",
        abbrev_name="vid",
    )
    return cfg


def main(args):
    cfg = setup(args)
    output_dir = cfg.OUTPUT_DIR
    if is_main_process():
        hash_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_code(
            os.path.abspath(os.path.curdir),
            os.path.join(output_dir, "code_" + hash_tag),
        )
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
