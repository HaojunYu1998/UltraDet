import hashlib

import numpy as np
from detectron2.utils.env import seed_all_rng
from path import Path


def hash_idx(rel_path, mod):
    """
    Compute the hash index of given path, here we use the relative path to compute.
    """
    idx = int(hashlib.sha256(rel_path.encode("utf-8")).hexdigest(), 16) % mod
    return idx


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def single_batch_collator(batch):
    """
    A batch collator that assumes all inputs are with length == 1.
    And this element is returned
    """
    assert len(batch) == 1
    return batch[0]


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2**31) + worker_id)
