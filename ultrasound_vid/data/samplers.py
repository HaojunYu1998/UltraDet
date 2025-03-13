import itertools
import math
from typing import Optional, List
import torch
from torch.utils.data.sampler import Sampler

from detectron2.utils import comm
import heapq
import numpy as np


def greedy_partition(lst, num_partitions):
    """
    Partition the `lst` into `num_partitions` partitions evenly.
    return:
        part_idxs: List[int], and len(part_idxs) == len(lst),
            part_idxs[i] = partition index assigned to lst[i]
    """
    buf = []
    for i in range(num_partitions):
        heapq.heappush(buf, (0, i))
    idxs = np.argsort(lst)
    part_idxs = [0 for _ in range(len(lst))]
    for idx in idxs[::-1]:
        item = lst[idx]
        value, part_idx = heapq.heappop(buf)
        part_idxs[idx] = part_idx
        new_item = (value + item, part_idx)
        heapq.heappush(buf, new_item)
    return part_idxs


class InferenceSampler(Sampler):
    def __init__(self, video_frame_nums: List):
        """
        Args:
            video_frame_nums (List): num of frames of a list of videos
        """
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        part_idxs = greedy_partition(video_frame_nums, self._world_size)
        self._local_indices = [
            i for i, part_idx in enumerate(part_idxs) if part_idx == self._rank
        ]

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class IsolateTrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        yield from self._infinite_indices()

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                indices = torch.randperm(self._size, generator=g)
                indices = [i for i in indices if i % self._world_size == self._rank]
                yield from indices
            else:
                yield from torch.arange(self._size)
