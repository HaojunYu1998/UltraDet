from multiprocessing import Pool

import numpy as np

from ultrasound_vid.evaluation.mean_ap import get_cls_results, tpfp_default


def average_recall(det_results, annotations, maxDets=None, nproc=4, iou_thr=0.5):
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    num_classes = len(det_results[0])  # positive class num
    pool = Pool(nproc)
    AR = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i, maxDets
        )
        # choose proper function according to datasets to compute tp and fp
        tpfp_func = tpfp_default
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_func,
            zip(
                cls_dets,
                cls_gts,
                cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [None for _ in range(num_imgs)],
            ),
        )
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(1, dtype=int)
        for j, bbox in enumerate(cls_gts):
            num_gts[0] += bbox.shape[0]
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        assert recalls.shape[0] == 1, "Multiple recall rows detected!"
        recalls = recalls[0, :]
        rc = recalls[-1]
        AR.append(rc)

    pool.close()
    return np.mean(AR)
